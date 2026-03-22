from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from pathlib import Path
import yaml
from ultralytics import YOLO

# --------------------------------------------------
# Project paths
# --------------------------------------------------
CURRENT_DIR = Path(__file__).resolve()
APP_DIR = CURRENT_DIR.parent
PROJECT_ROOT = CURRENT_DIR.parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "params.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

app = Flask(__name__, template_folder="templates", static_folder="static")

# --------------------------------------------------
# Model paths from config
# --------------------------------------------------
CNN_MODEL_PATH = PROJECT_ROOT / CONFIG["model_paths"]["cnn_model"]
YOLO_MODEL_PATH = PROJECT_ROOT / CONFIG["model_paths"]["yolo_model"]

# --------------------------------------------------
# Load models once at startup
# --------------------------------------------------
cnn_model = load_model(str(CNN_MODEL_PATH))
yolo_model = YOLO(str(YOLO_MODEL_PATH))

# --------------------------------------------------
# Upload / result image folder
# --------------------------------------------------
UPLOAD_FOLDER = APP_DIR / "static" / "images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classes = CONFIG["classes"]["names"]


# --------------------------------------------------
# CNN severity prediction for cropped region
# --------------------------------------------------
def cnn_predict(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = cnn_model.predict(img, verbose=0)[0]
    class_index = np.argmax(pred)

    severity = classes[class_index]
    confidence = float(pred[class_index])   # CNN confidence (0 to 1)

    return severity, confidence


# --------------------------------------------------
# Home page
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict/cnn", methods=["POST"])
def predict_cnn_api():
    if "file" not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files["file"]

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        image_path = tmp.name

    img = cv2.imread(image_path)

    if img is None:
        return {"error": "Invalid image"}, 400

    severity, confidence = cnn_predict(img)

    os.remove(image_path)

    return {
        "class": severity,
        "confidence": confidence
    }

@app.route("/predict/yolo", methods=["POST"])
def predict_yolo_api():
    if "file" not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files["file"]

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        image_path = tmp.name

    img = cv2.imread(image_path)

    if img is None:
        return {"error": "Invalid image"}, 400

    results = yolo_model.predict(img, conf=float(CONFIG["yolo"]["confidence_threshold"]))

    boxes_out = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])

            boxes_out.append({
                "box": [x1, y1, x2, y2],
                "confidence": conf
            })

    os.remove(image_path)

    return {
        "boxes": boxes_out
    }


# --------------------------------------------------
# Predict route
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]

    if file.filename == "":
        return "No selected file", 400

    input_path = UPLOAD_FOLDER / file.filename
    file.save(str(input_path))

    img = cv2.imread(str(input_path))

    if img is None:
        return "Invalid image file", 400

    results = yolo_model.predict(img, conf=float(CONFIG["yolo"]["confidence_threshold"]))

    detections = []

    for r in results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            yolo_conf = float(box.conf[0])   # YOLO confidence for location

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            severity, cnn_confidence = cnn_predict(crop)

            # Below panel should show CNN prediction and CNN confidence
            detections.append({
                "severity": severity,
                "confidence": cnn_confidence
            })

            # Bounding box color should follow CNN severity
            if severity == "minor":
                color = (0, 255, 0)
            elif severity == "moderate":
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # Text on image should show YOLO confidence
            label = f"{yolo_conf:.2f}"
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    result_filename = "result.jpg"
    output_path = UPLOAD_FOLDER / result_filename
    cv2.imwrite(str(output_path), img)

    if len(detections) == 0:
        detections.append({
            "severity": "unknown",
            "confidence": 0.0
        })

    result_image_url = url_for("static", filename=f"images/{result_filename}")

    return render_template(
        "index.html",
        result_image=result_image_url,
        detections=detections
    )


# --------------------------------------------------
# Health route
# --------------------------------------------------
@app.route("/health")
def health():
    return {"status": "ok", "models": ["cnn", "yolo"]}


if __name__ == "__main__":
    host = CONFIG.get("flask", {}).get("host", "0.0.0.0")
    port = CONFIG.get("flask", {}).get("port", 5000)
    app.run(host=host, port=port, debug=False)