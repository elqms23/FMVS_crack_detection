import os
from typing import Optional, List, Tuple

import cv2
import numpy as np

from config import Config
from ROI.unet import UNet
from ROI.autoROI import AutoROI
from time_utils import hms_from_seconds, safe_filename_from_hms
from CrackDetect.heuristic import detect_crack_boxes_on_crop
from utils import ensure_dir, unique_path

def run(cfg: Config) -> None:
    ensure_dir(cfg.outdir)

    # video open
    cap = cv2.VideoCapture(cfg.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg.video}")
    video_name = os.path.splitext(os.path.basename(cfg.video))[0]# 영상 이름
    outdir = os.path.join(cfg.outdir, video_name)
    ensure_dir(outdir)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0    

    # ---- AutoROI 초기화 ---- 
    model = UNet(in_channels=3, out_channels=1, base=32)
    ar = AutoROI(
        model=model,
        model_path=cfg.roi_model_path,
        device="cuda",
        target_w=896,
        thresh=0.5,
        min_area=2000,
        shrink=0.9,      # 필요시 0.85~0.95로 조정
        ema_alpha=0.2,
    )

    # 검사 range 설정
    # # Seek to start
    # if cfg.ranges:
    #     start_frame = int(cfg.ranges[0][0] * fps)
    # else:
    #     start_frame = int(cfg.start_sec * fps)
        
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # ---- start seek (ranges면 첫 range 시작으로 한번만 이동) ----
    if cfg.ranges is not None:
        first_s, _ = cfg.ranges[0]
        # 프레임 seek이 더 빠를 때가 많음 (Windows/MP4에서 MSEC seek이 느릴 수 있음)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(first_s * fps))
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cfg.start_sec * fps))

    # show/headless 정리
    if cfg.headless:
        cfg.show = False
    
    last_saved_time: Optional[float] = None
    range_i = 0

    # 영상 처리 루프
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        t_sec = cur_frame / fps

        # ---- ranges 처리 ----
        # if cfg.ranges is not None:
        #     s, e = cfg.ranges[range_i]

        #     # 아직 구간 시작 전이면 시작 프레임으로 점프
        #     if t_sec < s:
        #         cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
        #         continue

        #     # 구간 끝났으면 다음 구간으로
        #     if t_sec > e:
        #         range_i += 1
        #         if range_i >= len(cfg.ranges):
        #             print("모든 검사 구간 완료")
        #             break
        #         next_s, _ = cfg.ranges[range_i]
        #         cap.set(cv2.CAP_PROP_POS_FRAMES, int(next_s * fps))
        #         continue
        # else:
        #     # end_sec 지정했으면 종료
        #     if cfg.end_sec is not None and t_sec > cfg.end_sec:
        #         break
        if cfg.ranges is not None:
            s, e = cfg.ranges[range_i]

            # 구간 시작 전이면: 점프하지 말고 그냥 프레임을 빨리 넘기기
            if t_sec < s:
                # decode 부담을 줄이려면 read() 대신 grab() 추천
                cap.grab()
                continue

            # 구간 끝났으면: 다음 구간으로 "한 번만" 점프
            if t_sec > e:
                range_i += 1
                if range_i >= len(cfg.ranges):
                    print("모든 검사 구간 완료")
                    break

                next_s, _ = cfg.ranges[range_i]
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(next_s * fps))
                continue
        else:
            if cfg.end_sec is not None and t_sec > cfg.end_sec:
                break

        # ---- skip ---- # fps가 너무 높을 때 검사 속도 조절용
        if cfg.skip > 1 and (cur_frame % cfg.skip != 0):
            if cfg.show:
                # 원하면 show에서 프레임만 표시 가능(가볍게)
                # cv2.imshow("Inspection", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            continue

        # ---- cooldown ---- # 너무 자주 저장되는 것 방지
        if last_saved_time is not None and (t_sec - last_saved_time) < cfg.cooldown:
            if cfg.show:
                # cv2.imshow("Inspection", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            continue
        
        # ROI 잡기###############################
        # ---- AutoROI: crop + debug ----
        crop, info = ar.process_frame(frame, out_w=320, out_h=320, debug_draw=cfg.show)

        roi_raw = info.get("crop_raw")  # 새로 추가된 원본 비율 ROI
        if roi_raw is None:
            roi_raw = crop

        # ROI를 못 잡으면 넘어감
        if crop is None:
            if cfg.show:
                # debug_vis가 있으면 small 기준 시각화
                vis = info.get("debug_vis")
                if vis is None:
                    vis = info["small"]
                hh, mm, ss = hms_from_seconds(t_sec)
                cv2.putText(vis, f"ROI not found @ {hh}:{mm:02d}:{ss:02d}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow("Inspection", vis)
                # cv2.imshow("mask", info["mask"])
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            continue

        # ---- crack detection on crop ----
        boxes = detect_crack_boxes_on_crop(roi_raw, cfg)
        # 크랙 발견시 저장 및 로그 출력
        if len(boxes) > 0:
            h, m, s = hms_from_seconds(t_sec)
            time_str = safe_filename_from_hms(h, m, s)

            base = f"{video_name}_{time_str}"
            # outpath = unique_path(cfg.outdir, base, ".png")
            outpath = unique_path(outdir, time_str, ".png")
            # save_img = crop.copy()
            save_img = roi_raw.copy()
            for (x, y, w, h2) in boxes[:20]:
                cv2.rectangle(save_img, (x, y), (x + w, y + h2), (0, 0, 255), 2)
            cv2.imwrite(outpath, save_img)

            total_min = int(t_sec // 60)
            sec_in_min = int(t_sec % 60)
            print(f"[CRACK] {h}:{m:02d}:{s:02d} (총 {total_min}분 {sec_in_min}초) -> saved: {outpath}")

            last_saved_time = t_sec

        # ---- show ---- # 실시간 영상 표시(디버깅용)
        if cfg.show:
            vis = info.get("debug_vis")
            if vis is None:
                vis = info["small"]

            # crop 박스도 같이 보여주고 싶으면
            crop_vis = crop.copy()
            for (x, y, w, h2) in boxes[:20]:
                cv2.rectangle(crop_vis, (x, y), (x + w, y + h2), (0, 0, 255), 2)

            # 시간 표시
            hh, mm, ss = hms_from_seconds(t_sec)
            cv2.putText(vis, f"t={hh}:{mm:02d}:{ss:02d}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("Inspection", vis) # 전체 화면
            cv2.imshow("mask", info["mask"])    #마스크 화면
            cv2.imshow("ROI crop", crop_vis)    # 크롭 화면(roi)

            # 중간 종료 키(q)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    if cfg.show:
        cv2.destroyAllWindows()