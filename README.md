# Damage API
### Hybrid CNN–YOLO Pipeline for Vehicle Damage Detection and Severity Classification

## Overview

**Damage API** is an end-to-end computer vision project that detects damaged regions in vehicle images and classifies the severity of the detected damage.

The system combines:

- **YOLOv8** for damage localization
- **CNN** for damage severity classification
- **Flask** for web-based inference and API serving

This project is also designed as a practical **MLOps learning project**, with modular code for training, validation, inference, testing, and later deployment.

---

## Problem Statement

Given an image of a damaged vehicle, the system should:

1. Detect the damaged region
2. Crop the detected region
3. Classify the severity of the damage as:
   - Minor
   - Moderate
   - Severe

---

## System Workflow

```text
Input Vehicle Image
        │
        ▼
YOLOv8 Damage Detection
        │
        ▼
Damage Region Bounding Box
        │
        ▼
Crop Detected Region
        │
        ▼
CNN Severity Classification
        │
        ▼
Final Output
  - Damage Location
  - YOLO Confidence
  - Severity Class
  - CNN Confidence


---

## Project Structure

```text

repo-root/
│
├── damage_api/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── flask_app.py
│   │   ├── templates/
│   │   │   └── index.html
│   │   └── static/
│   │       └── images/
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   ├── train_cnn.py
│   │   ├── validate_cnn.py
│   │   ├── train_yolo.py
│   │   └── validate_yolo.py
│   │
│   ├── configs/
│   │   └── params.yaml
│   │
│   └── models/
│       └── README.txt
│
├── data/
│   ├── training/
│   │   ├── 01-minor/
│   │   │   └── README.txt
│   │   ├── 02-moderate/
│   │   │   └── README.txt
│   │   └── 03-severe/
│   │       └── README.txt
│   │
│   └── validation/
│       ├── 01-minor/
│       │   └── README.txt
│       ├── 02-moderate/
│       │   └── README.txt
│       └── 03-severe/
│           └── README.txt
│
├── car_damage_yolo/
│   ├── images/
│   │   ├── train/
│   │   │   └── README.txt
│   │   └── val/
│   │       └── README.txt
│   │
│   └── labels/
│       ├── train/
│       │   └── README.txt
│       └── val/
│           └── README.txt
│
├── tests/
│   ├── __init__.py
│   └── test_predict.py
│
├── notebooks/
├── environment.yml
├── requirements.txt
├── .gitignore
└── README.md


## Features

- Damage localization using YOLOv8
- Severity classification using CNN
- Flask-based UI for image upload and inference
- API endpoints for prediction
- Separate training and validation scripts
- Config-driven setup using YAML
- Unit test support
- MLOps-friendly modular structure


## Tech Stack

- Python
- TensorFlow / Keras
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Flask
- YAML configuration


## Installation

# Option 1: Conda

Create the environment:
conda env create -f environment.yml
conda activate cnn_yolo

# Option 2: pip

pip install -r requirements.txt

## Important Run Location

Run all commands from the repository root
(the folder that contains damage_api/, data/, car_damage_yolo/, etc.).

Example:

cd <repo-folder-name>

Do not run commands from inside damage_api/.


## Dataset Setup

# 1. CNN Dataset

Place CNN training and validation images in this structure:

data/
├── training/
│   ├── 01-minor/
│   ├── 02-moderate/
│   └── 03-severe/
│
└── validation/
    ├── 01-minor/
    ├── 02-moderate/
    └── 03-severe/
Each class folder should contain the corresponding images.
2. YOLO Dataset
Place YOLO images and label files in this structure:
Plain text
car_damage_yolo/
├── images/
│   ├── train/
│   └── val/
│
└── labels/
    ├── train/
    └── val/
Each image must have a matching label file with the same base filename.
Example:
Plain text
images/train/sample1.jpg
labels/train/sample1.txt
Configuration
All important settings are controlled through:
Plain text
damage_api/configs/params.yaml
This file includes settings for:
CNN input size
batch size
epochs
learning rate
class names
model paths
YOLO confidence threshold
Flask host and port
dataset paths
Important
If your dataset paths differ on another system, update the relevant paths inside:
Plain text
damage_api/configs/params.yaml
Also make sure your YOLO dataset config file points correctly to the dataset root:
Plain text
car_damage_yolo/data.yaml
Update that file if needed when moving the project to another machine.
Training
Train CNN
Bash
python -m damage_api.src.train_cnn
This will:
read configuration from params.yaml
load images from data/training and data/validation
train the CNN model
save the trained model to the configured model path
Train YOLO
Bash
python -m damage_api.src.train_yolo
This will:
read YOLO settings from params.yaml
read dataset structure from car_damage_yolo/data.yaml
train the YOLO detector
save outputs under runs/
Validation
Validate CNN
Bash
python -m damage_api.src.validate_cnn
This prints:
total validation samples
overall accuracy
per-class accuracy
confusion matrix
Validate YOLO
Bash
python -m damage_api.src.validate_yolo
This prints YOLO detection metrics such as:
Precision
Recall
mAP@50
mAP@50-95
Running the Web App
Start the Flask app:
Bash
python damage_api/app/flask_app.py
Then open:
Plain text
http://127.0.0.1:5000/
API Endpoints
Health Check
Http
GET /health
Response:
JSON
{
  "status": "ok",
  "models": ["cnn", "yolo"]
}
Combined Prediction
Http
POST /predict
Used by the web UI and can also be tested manually.
Input:
form-data
key: file
value: image file
CNN-only Prediction
Http
POST /predict/cnn
Returns CNN severity prediction.
YOLO-only Prediction
Http
POST /predict/yolo
Returns YOLO detection output.
UI Behavior
The current UI logic is:
Bounding box color is based on the CNN severity prediction
Score shown on the image is the YOLO confidence
Severity shown below the image is the CNN prediction
Confidence shown below the image is the CNN confidence
This keeps localization confidence and classification confidence separated clearly.
Testing
Run unit tests with:
Bash
python -m unittest discover -s tests
These tests verify:
prediction function output structure
Flask app health endpoint
core inference flow
Models
Model files are not included in the repository by default.
Place trained models in the appropriate configured location before inference.
Typical examples:
CNN model: .h5
YOLO model: .pt
Update paths in:
Plain text
damage_api/configs/params.yaml
if your model filenames or locations are different.
Reproducibility
To reproduce the pipeline on another machine:
Clone the repository
Create environment using environment.yml or requirements.txt
Add datasets to data/ and car_damage_yolo/
Update paths in params.yaml and car_damage_yolo/data.yaml if needed
Train or validate models
Run the Flask app
Current MLOps Direction
This project is being developed as a practical MLOps learning workflow.
The focus is not only on model performance, but also on:
modular project design
training/validation separation
configuration management
testing
deployment readiness
improvement and redeployment cycles
Future improvements may include:
Dockerization
CI/CD
model versioning
cloud deployment
monitoring and logging
Author
Basil Baasi
Notes
This repository is intended to be both:
a working computer vision application
a real-world MLOps learning project