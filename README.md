# Damage API
### Hybrid CNN–YOLO Pipeline for Vehicle Damage Detection and Severity Classification

---

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

```
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
```

---

## Project Structure

```
repo-root/
|
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
│   ├── __init__.py
│   └── test_predict.py
|
├── notebooks/
├── environment.yml
├── requirements.txt
├── .gitignore
└── README.md
```

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

```bash
conda env create -f environment.yml
conda activate cnn_yolo
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

---

## Important Run Location

Run all commands from the repository root.

```bash
cd <repo-folder-name>
```

Do not run commands from inside `damage_api/`.

---

## Dataset Setup

### CNN Dataset

```
data/
├── training/
│   ├── 01-minor/
│   ├── 02-moderate/
│   └── 03-severe/
└── validation/
    ├── 01-minor/
    ├── 02-moderate/
    └── 03-severe/
```

### YOLO Dataset

```
car_damage_yolo/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Example:

```
images/train/sample1.jpg
labels/train/sample1.txt
```

---

## Configuration

```
damage_api/configs/params.yaml
```

Also update:

```
car_damage_yolo/data.yaml
```

---

## Training

```bash
python -m damage_api.src.train_cnn
python -m damage_api.src.train_yolo
```

---

## Validation

```bash
python -m damage_api.src.validate_cnn
python -m damage_api.src.validate_yolo
```

---

## Running the Web App

```bash
python damage_api/app/flask_app.py
```

Open: http://127.0.0.1:5000/

---

## API Endpoints

### Health Check

```
GET /health
```

```json
{
  "status": "ok",
  "models": ["cnn", "yolo"]
}
```

### Combined Prediction

```
POST /predict
```

### CNN-only

```
POST /predict/cnn
```

### YOLO-only

```
POST /predict/yolo
```

---

## Testing

```bash
python -m unittest discover -s tests
```

---

## Models

- CNN: `.h5`
- YOLO: `.pt`

Update paths in:

```
damage_api/configs/params.yaml
```

---

## Reproducibility

1. Clone the repository  
2. Create environment  
3. Add datasets  
4. Update paths  
5. Train/validate  
6. Run app  

---

## Learning Goals

- MLOps fundamentals  
- Testing  
- Modular design  

---

## Connect With Me

- Portfolio: https://basilbaasi.github.io/
- Email: basilck618@gmail.com
- LinkedIn: linkedin.com/in/basilck
- GitHub: github.com/basilbaasi

---

## License

Open source for educational use.
