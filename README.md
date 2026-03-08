# Car Damage Detection AI
## A Hybrid CNN–YOLO Framework for Car Damage Severity Classification and Localization

An AI-powered system that detects **vehicle damage locations** and classifies the **severity of the damage** using a hybrid deep learning architecture combining **YOLOv8 object detection** and **CNN classification**.

The system includes a **Flask-based web interface** that allows users to upload vehicle images and automatically receive damage detection and severity analysis.

---

# Project Overview

Vehicle damage inspection is a critical step in insurance claim processing and vehicle evaluation. Manual inspection can be time-consuming and subjective.

This project proposes an automated deep learning pipeline that can:

- Detect damaged regions in car images
- Classify the severity of the damage
- Provide an interactive web interface for testing

The system integrates:

- **YOLOv8** → damage localization  
- **CNN** → severity classification  
- **Flask** → web interface

---

# System Architecture

```
Input Car Image
        │
        ▼
YOLOv8 Object Detection
(Damage Localization)
        │
        ▼
Bounding Box Extraction
(Crop Damage Region)
        │
        ▼
CNN Classifier
(Severity Prediction)
        │
        ▼
Flask Web Application
(Display Results)
```

The hybrid architecture enables both **damage detection and severity classification** in a single workflow.

---

# Damage Severity Classes

The model classifies vehicle damage into three categories:

| Class | Description |
|------|-------------|
| Minor | Small scratches or light dents |
| Moderate | Noticeable dents or panel damage |
| Severe | Major structural damage |

---

# Technologies Used

| Component | Technology |
|--------|--------|
| Programming Language | Python |
| Web Framework | Flask |
| Object Detection | YOLOv8 (Ultralytics) |
| Classification | CNN (TensorFlow / Keras) |
| Deep Learning Frameworks | TensorFlow, PyTorch |
| Image Processing | OpenCV |
| Frontend | HTML, CSS |
| Visualization | Matplotlib |

---

# Dataset Structure

The dataset is organized into training and validation sets for CNN training.

```
data/
   training/
      01-minor
      02-moderate
      03-severe

   validation/
      01-minor
      02-moderate
      03-severe
```

For YOLO training, the dataset follows the YOLO format:

```
car_damage_yolo/
   images/
      train/
      val/

   labels/
      train/
      val/
```

---

# CNN Severity Classification Model

The CNN model predicts the **severity level of the detected damage region**.

### Input Size

```
224 × 224
```

### Classes

```
Minor
Moderate
Severe
```

### Training Configuration

| Parameter | Value |
|------|------|
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Epochs | 30 |
| Batch Size | 32 |

### Training Results

| Metric | Value |
|------|------|
| Final Training Accuracy | **87.4%** |
| Final Validation Accuracy | **54.4%** |

The difference between training and validation accuracy indicates **mild overfitting**, likely caused by:

- limited dataset size  
- visual similarity between damage categories  
- variations in lighting and camera angle

---

# YOLOv8 Damage Detection Model

YOLOv8 is used to detect damaged regions in the vehicle image.

### Training Configuration

| Parameter | Value |
|------|------|
| Model | YOLOv8n |
| Image Size | 640 |
| Epochs | ~30 |
| Batch Size | 8 |

### Evaluation Metrics

### Evaluation Metrics

| Metric | Score |
|------|------|
| Precision | **0.542** |
| Recall | **0.708** |
| mAP@0.5 | **0.659** |
| mAP@0.5:0.95 | **0.647** |


These results show the model can effectively detect damaged areas in vehicle images.

---

# Web Application Features

The Flask web interface allows users to interact with the AI system.

Features include:

- Image upload interface
- Automatic damage detection
- Multiple damage detection support
- Bounding box visualization
- Severity classification
- Confidence score display
- Result summary panel

---

# Repository Structure

```
car-damage-detection-ai
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models
│   └── car.h5
│
├── runs
│   └── detect
│       └── car_damage_detector2
│           └── weights
│               └── best.pt
│
├── templates
│   └── index.html
│
├── static
│   └── images
│
├── notebooks
│   └── training_pipeline.ipynb
```

---

# Installation and Setup

### Clone the repository

```bash
git clone https://github.com/yourusername/car-damage-detection-ai.git
cd car-damage-detection-ai
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

---

# Access the Web Interface

Open a browser and navigate to:

```
http://127.0.0.1:5000
```

Upload a car image and the system will detect and classify the damage automatically.

---

# Example Workflow

1. Upload vehicle image  
2. YOLO detects damaged regions  
3. Bounding boxes highlight damage locations  
4. Each region is classified by the CNN model  
5. The web interface displays severity and confidence scores

---

# Limitations

- Limited dataset size
- Similar visual features between classes
- Lighting variations may affect detection

These can be improved with larger datasets and better data augmentation.

---

# Future Improvements

Potential improvements include:

- Larger training dataset
- Transfer learning using EfficientNet or ResNet
- Real-time video damage detection
- Docker containerization
- Cloud deployment
- Full MLOps pipeline integration

---

# Author

**Basil C K**  
B.Tech Artificial Intelligence and Data Science

---

# License

This project is developed for **academic and research purposes**.