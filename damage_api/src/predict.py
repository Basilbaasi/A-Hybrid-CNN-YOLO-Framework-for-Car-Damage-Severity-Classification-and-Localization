"""Prediction utilities for the car damage detection project."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
import cv2
import numpy as np

try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None  # type: ignore

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore

__all__ = ["predict_cnn", "predict_yolo"]

_CONFIG: Optional[Dict[str, Any]] = None
_CNN_MODEL = None
_YOLO_MODEL = None

def _load_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "params.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def _get_config() -> Dict[str, Any]:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_config()
    return _CONFIG

def _load_cnn_model() -> Any:
    global _CNN_MODEL
    if _CNN_MODEL is not None:
        return _CNN_MODEL
    if load_model is None:
        raise RuntimeError("TensorFlow is required for CNN prediction but is not installed")
    cfg = _get_config()
    model_path = Path(__file__).resolve().parents[1] / cfg["model_paths"]["cnn_model"]
    if not model_path.exists():
        raise FileNotFoundError(f"CNN model file not found at {model_path}")
    _CNN_MODEL = load_model(str(model_path))
    return _CNN_MODEL

def _load_yolo_model() -> Any:
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is required for YOLO prediction but is not installed")
    cfg = _get_config()
    model_path = Path(__file__).resolve().parents[1] / cfg["model_paths"]["yolo_model"]
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model file not found at {model_path}")
    _YOLO_MODEL = YOLO(str(model_path))
    return _YOLO_MODEL

def _preprocess_for_cnn(img: np.ndarray, input_size: Tuple[int, int, int]) -> np.ndarray:
    h, w, _ = input_size
    resized = cv2.resize(img, (w, h))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

def predict_cnn(image_path: str) -> Dict[str, Any]:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image or unsupported format: {image_path}")
    cfg = _get_config()
    model = _load_cnn_model()
    input_size = tuple(cfg["cnn"]["input_size"])
    preprocessed = _preprocess_for_cnn(img, input_size)
    preds = model.predict(preprocessed, verbose=0)[0]
    classes = cfg["classes"]["names"]
    idx = int(np.argmax(preds))
    return {"class": classes[idx], "confidence": float(preds[idx])}

def predict_yolo(image_path: str) -> Dict[str, Any]:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image or unsupported format: {image_path}")
    cfg = _get_config()
    model = _load_yolo_model()
    conf_thres = float(cfg["yolo"]["confidence_threshold"])
    results = model.predict(image_path, conf=conf_thres)
    detections: List[Dict[str, Any]] = []
    classes = cfg["classes"]["names"]
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist() if hasattr(box.xyxy[0], "cpu") else box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0]) if hasattr(box, "cls") else 0
            cls_name = classes[cls_idx] if cls_idx < len(classes) else str(cls_idx)
            detections.append({"box": [round(x, 2) for x in xyxy],
                               "class": cls_name,
                               "confidence": conf})
    if detections:
        top_det = max(detections, key=lambda d: d["confidence"])
        return {"class": top_det["class"],
                "confidence": top_det["confidence"],
                "boxes": detections}
    else:
        return {"class": None, "confidence": 0.0, "boxes": []}