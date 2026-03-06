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
classes = ["minor", "moderate", "severe"]

# CNN prediction function
def cnn_predict(img):

    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = cnn_model.predict(img)
    return classes[np.argmax(pred)]


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

    severity_result = "No Damage Detected"

    for r in results:

        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = img[y1:y2, x1:x2]

            severity = cnn_predict(crop)

            severity_result = severity

            # Color mapping
            if severity == "minor":
                color = (0,255,0)       # GREEN
            elif severity == "moderate":
                color = (0,165,255)     # ORANGE
            else:
                color = (0,0,255)       # RED

            # Draw box
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 3)

            # Label text
            label = severity.upper()

            cv2.putText(
                img,
                label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3
            )

    output_path = os.path.join(UPLOAD_FOLDER, "result.jpg")
    cv2.imwrite(output_path, img)

    return render_template(
        "index.html",
        result_image=output_path,
        severity=severity_result
    )


if __name__ == "__main__":
    app.run(debug=True)