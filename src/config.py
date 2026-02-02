from dataclasses import dataclass
import argparse
from time_utils import parse_hms_to_seconds, parse_ranges
import os

# Config / Utils
@dataclass
class Config:
    video: str
    outdir: str
    skip: int
    cooldown: float
    show: bool
    headless: bool
    start_sec: float
    end_sec: float | None
    #range
    ranges: list | None

    min_w: int
    min_area: int
    min_aspect: float
    max_aspect: float
    dilate_iter: int
    canny1: int
    canny2: int

    roi_model_path: str = "roi_seg_model_state.pt"

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Crack inspection with ROI capture + cooldown logging")

    p.add_argument("--video", default="./yajong.mp4", help="input video path")
    p.add_argument("--outdir", default="./crack", help="output directory for ROI captures")

    p.add_argument("--skip", type=int, default=2, help="inspect once every N frames (default: 2)")
    p.add_argument("--cooldown", type=float, default=1.0, help="seconds to suppress re-detect/save (default: 1.0)")

    p.add_argument("--show", action="store_true", help="show realtime window after ROI setup")
    p.add_argument("--headless", action="store_true", help="run without showing video after ROI setup")

    p.add_argument("--start", default="0:0:0", help='start time "H:M:S" (or "M:S", "S")')
    p.add_argument("--end", default=None, help='end time "H:M:S" (or "M:S", "S")')
    # range
    p.add_argument(
    "--ranges",
    default=None,
    help='time ranges: "HH:MM:SS-HH:MM:SS,HH:MM:SS-HH:MM:SS,..."'
)

    # Detection tuning (기존 휴리스틱 느낌 유지)
    p.add_argument("--min-w", type=int, default=40, help="min width for crack bbox")
    p.add_argument("--min-area", type=int, default=50, help="min area for crack bbox")
    p.add_argument("--min-aspect", type=float, default=1.2, help="min aspect ratio (w/h)")
    p.add_argument("--max-aspect", type=float, default=4.0, help="max aspect ratio (w/h)")
    p.add_argument("--dilate-iter", type=int, default=1, help="dilate iterations after Canny")
    p.add_argument("--canny1", type=int, default=50, help="Canny threshold1")
    p.add_argument("--canny2", type=int, default=150, help="Canny threshold2")
    p.add_argument(
    "--roi-model",
    default="./src/roi_seg_model_state.pt",
    help="path to ROI segmentation model (.pt)"
)

    return p

def make_config(args: argparse.Namespace) -> Config:
    end_sec = None
    ranges = None

    if args.ranges:
        ranges = parse_ranges(args.ranges)
        start_sec = min(r[0] for r in ranges)  # 가장 빠른 시작점
        end_sec   = max(e for s, e in ranges)
    else:
        start_sec = parse_hms_to_seconds(args.start)
        if args.end is not None:
            end_sec = parse_hms_to_seconds(args.end)

    return Config(
        video=args.video,
        outdir=args.outdir,
        skip=max(1, args.skip),
        cooldown=max(0.0, args.cooldown),
        show=bool(args.show),
        headless=bool(args.headless),
        start_sec=start_sec,
        end_sec=end_sec,
        ranges=ranges,

        min_w=args.min_w,
        min_area=args.min_area,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        dilate_iter=max(0, args.dilate_iter),
        canny1=args.canny1,
        canny2=args.canny2,
        roi_model_path=os.path.abspath(args.roi_model),
    )