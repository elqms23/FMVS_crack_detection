import cv2
import numpy as np
import torch

# 
class RectEMA:
    """EMA smoothing for rotated rect: ((cx,cy),(w,h),angle)"""
    def __init__(self, alpha=0.2):
        self.alpha = float(alpha)
        self.state = None  # np.array([cx,cy,w,h,angle], float32)

    def update(self, rect):
        (cx, cy), (w, h), angle = rect
        x = np.array([cx, cy, w, h, angle], dtype=np.float32)

        if self.state is None:
            self.state = x
        else:
            self.state = (1.0 - self.alpha) * self.state + self.alpha * x

        cx, cy, w, h, angle = self.state.tolist()
        return ((cx, cy), (w, h), angle)
    
# Auto ROI

class AutoROI:
    def __init__(
        self,
        model,
        model_path: str,
        device: str = "cuda",
        target_w: int = 896,
        thresh: float = 0.5,
        min_area: int = 2000,
        shrink: float = 0.90,
        ema_alpha: float = 0.2,
        use_morph: bool = True,
        morph_ksize: int = 7,
        close_iters: int = 2,
        open_iters: int = 1,
    ):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.target_w = int(target_w)
        self.thresh = float(thresh)
        self.min_area = int(min_area)
        self.shrink = float(shrink)
        self.use_morph = bool(use_morph)

        self.morph_ksize = int(morph_ksize)
        self.close_iters = int(close_iters)
        self.open_iters = int(open_iters)

        self.ema = RectEMA(alpha=ema_alpha)

        # load model
        self.model = model.to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @staticmethod
    def _shrink_rect(rect, shrink):
        (cx, cy), (w, h), angle = rect
        return ((cx, cy), (w * shrink, h * shrink), angle)

    @staticmethod
    def _draw_rect(img_bgr, rect, color=(0, 255, 0), thickness=2):
        box = cv2.boxPoints(rect).astype(np.int32)
        out = img_bgr.copy()
        cv2.drawContours(out, [box], 0, color, thickness)
        return out

    # process
    def preprocess_bgr(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = self.target_w / w
        nh = int(h * scale)

        small = cv2.resize(frame_bgr, (self.target_w, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
        return x, scale, small

    @torch.no_grad()
    def infer_mask(self, frame_bgr):
        x, scale, small = self.preprocess_bgr(frame_bgr)
        x = x.to(self.device)

        logits = self.model(x)              # (1,1,H,W) for UNet
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        mask = (prob > self.thresh).astype(np.uint8) * 255
        return mask, prob, scale, small
    
    # mask  to roi rect
    def roi_from_mask(self, mask_u8):
        if mask_u8 is None:
            return None

        mm = mask_u8
        if self.use_morph:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_ksize, self.morph_ksize))
            mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, k, iterations=self.open_iters)
            mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, k, iterations=self.close_iters)

        cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < self.min_area:
            return None

        rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
        return rect
    
    # crop rect
    def crop_rotated_rect(self, img_bgr, rect, out_w=320, out_h=320, resize=True):
        (cx, cy), (w, h), angle = rect

        # minAreaRect flip fix
        if w < h:
            w, h = h, w
            angle += 90.0

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(
            img_bgr, M,
            (img_bgr.shape[1], img_bgr.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        x0 = int(cx - w / 2); y0 = int(cy - h / 2)
        x1 = int(cx + w / 2); y1 = int(cy + h / 2)

        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(rotated.shape[1], x1)
        y1 = min(rotated.shape[0], y1)

        crop = rotated[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        if resize:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return crop
    
    # main process function
    def process_frame(self, frame_bgr, out_w=320, out_h=320, debug_draw=False):
        
        mask, prob, scale, small = self.infer_mask(frame_bgr)
        rect = self.roi_from_mask(mask)

        crop_raw = None
        crop_sq  = None
        crop = None
        rect_s = None
        debug_vis = None

        if rect is not None:
            rect = self.ema.update(rect)
            rect_s = self._shrink_rect(rect, self.shrink)
            crop_raw = self.crop_rotated_rect(small, rect_s, resize=False) # 원본비율 (heuristic용)
            crop_sq  = self.crop_rotated_rect(small, rect_s, out_w=out_w, out_h=out_h, resize=True) # square (for DL detection)

            if debug_draw:
                # green = original, red = shrunk
                debug_vis = self._draw_rect(small, rect, color=(0, 255, 0), thickness=2)
                debug_vis = self._draw_rect(debug_vis, rect_s, color=(0, 0, 255), thickness=2)

        info = {
            "mask": mask,
            "prob": prob,
            "scale": scale,
            "small": small,
            "rect": rect,
            "rect_shrunk": rect_s,
            "crop_raw": crop_raw,
        }
        if debug_draw:
            info["debug_vis"] = debug_vis

        return crop_sq, info