# Car Damage Detection AI
## Hybrid CNN + YOLO Pipeline for Damage Localization and Severity Classification

This project is an end-to-end computer vision system that detects damaged regions in vehicle images and classifies damage severity using a hybrid deep learning pipeline.

It combines:

- **YOLOv8** for damage localization
- **CNN** for severity classification
- **Flask** for an interactive web UI and inference service

The main goal of this project is not only building a working model, but also learning **MLOps through a real project** by repeatedly training, validating, integrating, testing, deploying, modifying, and redeploying the system.

---

## Project Goal

This repository is being developed as a practical MLOps learning project. The focus is to learn how to:

- refactor notebook-based ML code into a modular project
- separate training, validation, inference, and serving
- manage configuration through YAML
- build reproducible environments
- test model APIs
- deploy and later redeploy improved versions
- evolve the project like a real production ML system

---

## What the System Does

Given a car image:

1. **YOLOv8** detects the damaged region
2. The detected region is cropped
3. **CNN** classifies the severity as:
   - Minor
   - Moderate
   - Severe
4. The Flask UI displays:
   - detected damage region
   - bounding box
   - severity prediction
   - confidence scores

---

## Current Architecture

```text
Input Car Image
      ↓
YOLOv8 Damage Detection
      ↓
Crop Detected Region
      ↓
CNN Severity Classification
      ↓
Flask Web UI / API Output
Refactored Project Structure
car_damage/
│
├── app/
│   ├── flask_app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── images/
│
├── src/
│   ├── predict.py
│   ├── train_cnn.py
│   ├── validate_cnn.py
│   ├── train_yolo.py
│   └── validate_yolo.py
│
├── configs/
│   └── params.yaml
│
├── models/
│   ├── car.h5
│   └── best.pt
│
├── data/
│   ├── training/
│   └── validation/
│
├── car_damage_yolo/
│   ├── images/
│   ├── labels/
│   └── data.yaml
│
├── tests/
│   └── test_predict.py
│
├── environment.yml
├── requirements.txt
└── README.md
Why This Refactor Matters

The older version of this repository was centered around a root-level Flask app, notebook-driven experimentation, and direct model usage from a single workflow. The GitHub version still reflects that older structure.

This refactor improves the project by separating:

configuration
training
validation
prediction
serving
testing

This is important for MLOps because it makes the system easier to:

debug
retrain
validate
deploy
version
extend later with Docker, CI/CD, and monitoring
Models Used
1. YOLOv8

Used for detecting the damaged region in the vehicle image.

2. CNN

Used for classifying the severity of the detected damage into:

Minor
Moderate
Severe
Environment Setup
Option 1: Conda

From the project folder:

conda env create -f environment.yml -n cnn_yolo
conda activate cnn_yolo
Option 2: pip + venv
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r car_damage/requirements.txt
How to Run

Important: run module-based commands from the directory above car_damage.

Train CNN
python -m car_damage.src.train_cnn
Validate CNN
python -m car_damage.src.validate_cnn
Train YOLO
python -m car_damage.src.train_yolo
Validate YOLO
python -m car_damage.src.validate_yolo
Run Flask App
python car_damage/app/flask_app.py

Then open:

http://127.0.0.1:5000/
API Endpoints
Health Check
GET /health
CNN-only Prediction
POST /predict/cnn
YOLO-only Prediction
POST /predict/yolo
Full Pipeline Prediction
POST /predict
Testing

Run basic tests:

python -m unittest discover -s car_damage/tests

These tests verify:

CNN prediction output structure
YOLO prediction output structure
Flask health endpoint
Current Development Focus

This project is being developed iteratively with an MLOps mindset:

deploy first with the current working models
improve models step by step
validate each update
integrate and redeploy
track the evolution of the system as a real-world ML project

Upcoming MLOps steps include:

dependency cleanup
Dockerization
deployment
CI/CD
model versioning
monitoring and observability
Notes on UI Logic

In the current integrated UI flow:

bounding box color follows the CNN severity prediction
score displayed on the image shows YOLO confidence
severity panel shows CNN prediction
severity confidence in the panel shows CNN confidence

This keeps localization and classification outputs separated properly.

Repository History Note

The current GitHub repository still shows the older monolithic project layout, root-level app structure, and older README content. The next commit should update the repository to reflect the modular structure and MLOps-focused workflow now implemented locally.

Author

Basil

AI / Data Science student building real-world ML and MLOps projects.

If you found this project useful

Please consider starring the repository.