from typing import List, Tuple

import cv2
import numpy as np

from config import Config

# -----------------------------
# Crack Detection (heuristic) -> YOLO or segmentation으로 대체
# -----------------------------

### gray scale + blur(노이즈) + blackhat(밝은 배경 위 검은선(크랙)) + thresh(크랙후보 이진화) + contours(박스처리)
def detect_crack_boxes_on_crop(
    frame_bgr: np.ndarray,
    cfg: Config,
) -> List[Tuple[int, int, int, int]]:
    """
    Returns list of bounding boxes (x, y, w, h) in crop coordinates.
    Simple edge/contour heuristic inside ROI.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel_bh)

    _, thresh = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect = w / float(h + 1e-6)
        # filtering conditions
        if w > cfg.min_w and area > cfg.min_area and cfg.min_aspect < aspect < cfg.max_aspect:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
    return boxes