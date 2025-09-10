"""
main.py
----------
Core functions for loading a YOLO model, running detection, processing results,
drawing bounding boxes, and saving images.
"""
from typing import List, Dict, Optional, Tuple, Any
import os
import time
import numpy as np
import cv2
from PIL import Image
try:
    # ultralytics provides YOLOv5/YOLOv8 style API for v8+
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(
        "Failed to import ultralytics. Install with `pip install ultralytics`."
    ) from e

DEFAULT_MODEL = "yolov8n.pt"  # small, fast, good default


def load_model(model_path: str = DEFAULT_MODEL) -> Any:
    """Load a YOLO model (Ultralytics).

    Args:
        model_path: Path or model name (e.g., 'yolov8n.pt')

    Returns:
        A loaded YOLO model object.
    """
    model = YOLO(model_path)
    return model


def detect_image(
    model: Any,
    image_bgr: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    classes: Optional[List[int]] = None,
    max_det: int = 300,
    agnostic_nms: bool = False,
    device: Optional[str] = None,
) -> Any:
    """Run object detection on a single BGR image (OpenCV format).

    Returns:
        Ultralytics results object.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to detect_image()")
    # Ultralytics YOLO expects RGB by default; it will handle conversion internally,
    # but passing BGR is fine as it converts inside. We'll keep BGR and draw with OpenCV later.
    results = model.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        classes=classes,
        max_det=max_det,
        agnostic_nms=agnostic_nms,
        device=device,
        verbose=False,
    )
    return results


def process_results(results: Any) -> List[Dict[str, Any]]:
    """Convert Ultralytics results to a list of dicts (one per detection).

    Each dict contains: x1,y1,x2,y2,conf,cls,cls_name.
    """
    detections = []
    if not results:
        return detections

    # Ultralytics returns a list-like of Results; we assume one image per call
    res = results[0]
    boxes = getattr(res, "boxes", None)
    names = getattr(res, "names", None) or {}

    if boxes is None or len(boxes) == 0:
        return detections

    # xyxy for corners, .cls for class id, .conf for confidence
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
        det = {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "conf": float(c),
            "cls": int(k),
            "cls_name": names.get(int(k), str(int(k))),
        }
        detections.append(det)

    return detections


def draw_boxes(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    box_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image (BGR)."""
    img = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        label = f"{det['cls_name']} {det['conf']*100:.1f}%"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
        # background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    return img


def save_image(image_bgr: np.ndarray, save_dir: str, prefix: str = "det") -> str:
    """Save image to disk under save_dir with a timestamped filename."""
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(save_dir, f"{prefix}_{ts}.jpg")
    ok = cv2.imwrite(path, image_bgr)
    if not ok:
        raise IOError(f"Failed to save image to {path}")
    return path


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR ndarray."""
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def cv2_to_pil(bgr_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR ndarray to PIL Image."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
