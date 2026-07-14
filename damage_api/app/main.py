"""FastAPI application for car-damage inference.

Run from the repository root with:
    uvicorn damage_api.app.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

LOGGER = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "params.yaml"
STATIC_DIR = APP_DIR / "static"
IMAGE_DIR = STATIC_DIR / "images"


def load_config() -> dict[str, Any]:
    """Read the application configuration once during application setup."""
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


class ModelService:
    """Owns the in-memory CNN and YOLO models for the lifetime of the app."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.cnn_model: Any = None
        self.yolo_model: Any = None
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load both models once. Keep the API available if artifacts are missing."""
        cnn_path = PROJECT_ROOT / self.config["model_paths"]["cnn_model"]
        yolo_path = PROJECT_ROOT / self.config["model_paths"]["yolo_model"]
        missing = [str(path) for path in (cnn_path, yolo_path) if not path.exists()]
        if missing:
            self.error = f"Model file(s) not found: {', '.join(missing)}"
            LOGGER.warning(self.error)
            return
        try:
            # Keep Ultralytics' settings out of the user profile. This avoids
            # Windows permission failures in restricted or service accounts.
            ultralytics_config_dir = PROJECT_ROOT / ".ultralytics"
            ultralytics_config_dir.mkdir(exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_config_dir))
            from tensorflow.keras.models import load_model
            from ultralytics import YOLO

            self.cnn_model = load_model(str(cnn_path))
            self.yolo_model = YOLO(str(yolo_path))
            self.error = None
            LOGGER.info("CNN and YOLO models loaded successfully.")
        except Exception as exc:  # Keep /health available for deployment diagnostics.
            self.cnn_model = self.yolo_model = None
            self.error = f"Unable to load models: {exc}"
            LOGGER.exception("Model loading failed")

    @property
    def ready(self) -> bool:
        return self.cnn_model is not None and self.yolo_model is not None

    def predict_cnn(self, image: np.ndarray) -> tuple[str, float]:
        if self.cnn_model is None:
            raise RuntimeError(self.error or "CNN model is unavailable")
        resized = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
        with self._lock:
            prediction = self.cnn_model.predict(np.expand_dims(resized, axis=0), verbose=0)[0]
        class_index = int(np.argmax(prediction))
        return self.config["classes"]["names"][class_index], float(prediction[class_index])

    def predict_yolo(self, image: np.ndarray) -> list[dict[str, Any]]:
        if self.yolo_model is None:
            raise RuntimeError(self.error or "YOLO model is unavailable")
        with self._lock:
            results = self.yolo_model.predict(
                image, conf=float(self.config["yolo"]["confidence_threshold"]), verbose=False
            )
        boxes_out: list[dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                boxes_out.append({"box": [x1, y1, x2, y2], "confidence": float(box.conf[0])})
        return boxes_out


CONFIG = load_config()
models = ModelService(CONFIG)
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    models.load()
    yield


app = FastAPI(
    title="Car Damage Detection API",
    version="1.0.0",
    description="Hybrid YOLO localization and CNN severity classification service.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


async def read_image(file: Optional[UploadFile]) -> tuple[Optional[np.ndarray], Optional[JSONResponse]]:
    """Validate and decode an uploaded image without writing user input to disk."""
    if file is None or not file.filename:
        return None, JSONResponse({"error": "No file uploaded"}, status_code=400)
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None, JSONResponse({"error": "Invalid image file"}, status_code=400)
    return image, None


def model_unavailable_response() -> JSONResponse:
    return JSONResponse({"error": "Prediction models are unavailable", "detail": models.error}, status_code=503)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/predict/cnn")
async def predict_cnn_api(file: Optional[UploadFile] = File(default=None)) -> JSONResponse:
    image, error = await read_image(file)
    if error:
        return error
    if not models.ready:
        return model_unavailable_response()
    severity, confidence = models.predict_cnn(image)
    return JSONResponse({"class": severity, "confidence": confidence})


@app.post("/predict/yolo")
async def predict_yolo_api(file: Optional[UploadFile] = File(default=None)) -> JSONResponse:
    image, error = await read_image(file)
    if error:
        return error
    if not models.ready:
        return model_unavailable_response()
    return JSONResponse({"boxes": models.predict_yolo(image)})


@app.post("/predict", response_class=HTMLResponse, include_in_schema=False)
async def predict(
    request: Request, file: Optional[UploadFile] = File(default=None)
):
    """Keep the original form endpoint and render the same frontend template."""
    image, error = await read_image(file)
    if error:
        return error
    if not models.ready:
        return model_unavailable_response()
    detections: list[dict[str, Any]] = []
    for detection in models.predict_yolo(image):
        x1, y1, x2, y2 = (int(value) for value in detection["box"])
        crop = image[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
        if crop.size == 0:
            continue
        severity, confidence = models.predict_cnn(crop)
        detections.append({"severity": severity, "confidence": confidence})
        color = {"minor": (0, 255, 0), "moderate": (0, 165, 255)}.get(severity, (0, 0, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(image, f"{detection['confidence']:.2f}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if not detections:
        detections.append({"severity": "unknown", "confidence": 0.0})
    result_name = f"result-{uuid.uuid4().hex}.jpg"
    if not cv2.imwrite(str(IMAGE_DIR / result_name), image):
        return JSONResponse({"error": "Unable to save the result image"}, status_code=500)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result_image": f"/static/images/{result_name}", "detections": detections},
    )


@app.get("/health")
def health() -> dict[str, Any]:
    """Return readiness information without requiring a successful model load."""
    return {
        "status": "ok" if models.ready else "degraded",
        "models": {"cnn": models.cnn_model is not None, "yolo": models.yolo_model is not None},
        "detail": models.error,
    }
