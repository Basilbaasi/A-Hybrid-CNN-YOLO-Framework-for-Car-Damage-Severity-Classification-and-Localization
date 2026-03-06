from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__)

# Load models
cnn_model = load_model("car.h5")
yolo_model = YOLO("runs/detect/car_damage_detector2/weights/best.pt")

UPLOAD_FOLDER = "static/images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classes = ["minor", "moderate", "severe"]


# CNN severity prediction
def cnn_predict(img):

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = cnn_model.predict(img)
    class_index = np.argmax(pred)

    return classes[class_index]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)

    # Run YOLO detection
    results = yolo_model.predict(img, conf=0.10)

    detections = []

    for r in results:

        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            severity = cnn_predict(crop)

            detections.append({
                "severity": severity,
                "confidence": round(conf, 2)
            })

            # Color mapping
            if severity == "minor":
                color = (0, 255, 0)
            elif severity == "moderate":
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    output_path = os.path.join(UPLOAD_FOLDER, "result.jpg")
    cv2.imwrite(output_path, img)

    # If no damage detected
    if len(detections) == 0:
        detections.append({
            "severity": "No Damage Detected",
            "confidence": 0
        })

    return render_template(
        "index.html",
        result_image=output_path,
        detections=detections
    )


if __name__ == "__main__":
    app.run(debug=True)