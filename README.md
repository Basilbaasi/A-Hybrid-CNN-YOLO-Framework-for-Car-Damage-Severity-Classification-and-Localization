# Damage API
### Hybrid CNN–YOLO Pipeline for Vehicle Damage Detection and Severity Classification

## Overview

**Damage API** is an end-to-end computer vision project that detects damaged regions in vehicle images and classifies the severity of the detected damage.

The system combines:
- **YOLOv8** for damage localization
- **CNN** for damage severity classification
- **Flask** for web-based inference and API serving

This project is designed as a practical **MLOps learning project**, with modular code for training, validation, inference, and testing.

---

## Author

**Basil C K**

AI & Data Science Graduate | Python Developer | Machine Learning Engineer

- **Portfolio:** https://basilbaasi.github.io/
- **Email:** basilck618@gmail.com
- **LinkedIn:** linkedin.com/in/basilck
- **GitHub:** github.com/basilbaasi
- **Location:** Thrissur, Kerala, India

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


Input Vehicle Image
|
▼
YOLOv8 Damage Detection
|
▼
Damage Region Bounding Box
|
▼
Crop Detected Region
|
▼
CNN Severity Classification
|
▼
Final Output
- Damage Location
- YOLO Confidence
- Severity Class
- CNN Confidence


---

## Project Structure

repo-root/
|
├── damage_api/
│   ├── init.py
│   ├── app/
│   │   ├── init.py
│   │   ├── flask_app.py
│   │   ├── templates/
│   │   │   └── index.html
│   │   └── static/
│   │       └── images/
│   │
│   ├── src/
│   │   ├── init.py
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
|
├── data/
│   ├── training/
│   │   ├── 01-minor/
│   │   ├── 02-moderate/
│   │   └── 03-severe/
│   │
│   └── validation/
│       ├── 01-minor/
│       ├── 02-moderate/
│       └── 03-severe/
|
├── car_damage_yolo/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   │
│   └── labels/
│       ├── train/
│       └── val/
|
├── tests/
│   ├── init.py
│   └── test_predict.py
|
├── notebooks/
├── environment.yml
├── requirements.txt
├── .gitignore
└── README.md


---

## Features

- Damage localization using YOLOv8
- Severity classification using CNN
- Flask-based UI for image upload and inference
- REST API endpoints for prediction
- Separate training and validation scripts
- Config-driven setup using YAML
- Unit test support
- Modular project structure

---

## Tech Stack

- Python
- TensorFlow / Keras
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Flask
- YAML configuration

---

## Installation

### Option 1: Conda (Recommended)

Create the environment:

conda env create -f environment.yml
conda activate cnn_yolo


### Option 2: pip


---

## Important Run Location

Run all commands from the repository root (the folder that contains damage_api/, data/, car_damage_yolo/, etc.).

Example:

cd <repo-folder-name>


Do not run commands from inside damage_api/.

---

## Dataset Setup

### 1. CNN Dataset

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

### 2. YOLO Dataset

Place YOLO images and label files in this structure:

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

images/train/sample1.jpg
labels/train/sample1.txt


---

## Configuration

All important settings are controlled through:

damage_api/configs/params.yaml


This file includes settings for:
- CNN input size
- Batch size
- Epochs
- Learning rate
- Class names
- Model paths
- YOLO confidence threshold
- Flask host and port
- Dataset paths

**Important:** If your dataset paths differ on another system, update the relevant paths inside:

damage_api/configs/params.yaml


Also make sure your YOLO dataset config file points correctly to the dataset root:

car_damage_yolo/data.yaml


Update that file if needed when moving the project to another machine.

---

## Training

### Train CNN

python -m damage_api.src.train_cnn


This will:
- Read configuration from params.yaml
- Load images from data/training and data/validation
- Train the CNN model
- Save the trained model to the configured model path

### Train YOLO

python -m damage_api.src.train_yolo


This will:
- Read YOLO settings from params.yaml
- Read dataset structure from car_damage_yolo/data.yaml
- Train the YOLO detector
- Save outputs under runs/

---

## Validation

### Validate CNN

python -m damage_api.src.validate_cnn


This prints:
- Total validation samples
- Overall accuracy
- Per-class accuracy
- Confusion matrix

### Validate YOLO

python -m damage_api.src.validate_yolo


This prints YOLO detection metrics such as:
- Precision
- Recall
- mAP@50
- mAP@50-95

---

## Running the Web App

Start the Flask app:

python damage_api/app/flask_app.py


Then open: http://127.0.0.1:5000/

---

## API Endpoints

### Health Check

GET /health


Response:

```json
{
  "status": "ok",
  "models": ["cnn", "yolo"]
}

Combined Prediction

POST /predict

Used by the web UI and can also be tested manually.
Input:

form-data
key: file
value: image file

CNN-only Prediction

POST /predict/cnn

Returns CNN severity prediction.

YOLO-only Prediction

POST /predict/yolo

Returns YOLO detection output.


UI Behavior

The current UI logic is:

- Bounding box color is based on the CNN severity prediction
- Score shown on the image is the YOLO confidence
- Severity shown below the image is the CNN prediction
- Confidence shown below the image is the CNN confidence
- This keeps localization confidence and classification confidence separated clearly.


Testing

Run unit tests with:

python -m unittest discover -s tests

- These tests verify:
- Prediction function output structure
- Flask app health endpoint
- Core inference flow

Models


Model files are not included in the repository by default. Place trained models in the appropriate configured location before inference.

Typical examples:

- CNN model: .h5
- YOLO model: .pt

Update paths in:

damage_api/configs/params.yaml

if your model filenames or locations are different.

Reproducibility


To reproduce the pipeline on another machine:

- Clone the repository
- Create environment using environment.yml or requirements.txt
- Add datasets to data/ and car_damage_yolo/
- Update paths in params.yaml and car_damage_yolo/data.yaml if needed
- Train or validate models
- Run the Flask app


Learning Goals

This project is being developed as a hands-on platform to learn:

- Modular project design
- Training and validation workflows
- Configuration management
- Testing practices
- MLOps fundamentals

Future learning areas include Docker, CI/CD, model deployment, and scalable ML systems.


Connect With Me

I am actively seeking opportunities in MLOps, Machine Learning Engineering, and Backend Development roles.

- Portfolio: https://basilbaasi.github.io/
- Email: basilck618@gmail.com
- LinkedIn: linkedin.com/in/basilck
- GitHub: github.com/basilbaasi
-
Feel free to reach out for collaborations, feedback, or opportunities!


License

This project is open source and available for educational and research purposes.