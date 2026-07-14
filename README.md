# A Hybrid CNN-YOLO Framework for Car Damage Severity Classification and Localization

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-blue?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](file:///c:/Users/basil/OneDrive%20-%20jfmofficial/Desktop/work/car-damage-detection-ai/A-Hybrid-CNN-YOLO-Framework-for-Car-Damage-Severity-Classification-and-Localization/LICENSE)

An end-to-end computer vision framework designed for localized car damage detection and severity classification. The system uses a hybrid approach: **Ultralytics YOLOv8** identifies bounding boxes of damaged areas, and a custom **TensorFlow/Keras Convolutional Neural Network (CNN)** classifies the severity of each localized region as **minor**, **moderate**, or **severe**. The combined pipeline is served via a **FastAPI** web server offering a browser interface and API endpoints.

---

## Table of Contents
1. [Workflow Pipeline Diagram](#workflow-pipeline-diagram)
2. [Key Features](#key-features)
3. [Project Directory Layout](#project-directory-layout)
4. [Environment Setup & Installation](#environment-setup--installation)
    - [Option A: Anaconda Prompt Setup (Recommended)](#option-a-anaconda-prompt-setup-recommended)
    - [Option B: Normal Command Prompt / venv Setup](#option-b-normal-command-prompt--venv-setup)
5. [Trained Models Deployment](#trained-models-deployment)
6. [Running the FastAPI Server](#running-the-fastapi-server)
7. [API Routes & Interactive Docs](#api-routes--interactive-docs)
8. [Training & Validation Pipeline](#training--validation-pipeline)
9. [Running Unit Tests](#running-unit-tests)
10. [License](#license)

---

## Workflow Pipeline Diagram

The flowchart below demonstrates how the hybrid architecture coordinates YOLOv8 localization and TensorFlow CNN classification to deliver annotated bounding boxes and severity labels:

```mermaid
graph TD
    A[User / Client Application] -->|Uploads Image| B(FastAPI Server /predict)
    B -->|Check Models Readiness| C{Models Loaded?}
    C -->|No| D[Return 503 degraded status]
    C -->|Yes| E[Pass full image to YOLOv8]
    
    E -->|Predict Bounding Boxes| F[YOLOv8 Damage Localization]
    F -->|Output Bounding Boxes list| G{Damage boxes found?}
    
    G -->|No| H[Mark Severity as Unknown]
    G -->|Yes| I[Iterate through Bounding Boxes]
    
    I -->|Crop Damage Area| J[Extract Sub-image Crop]
    J -->|Resize to 224x224 & Normalize| K[Preprocess Crop for CNN]
    K -->|Pass to TensorFlow CNN| L[CNN Severity Classifier]
    L -->|Output Probability Distribution| M[Classify: minor / moderate / severe]
    
    M -->|Color Coded Annotations| N[Draw rectangle on image <br> Green: minor | Orange: moderate | Red: severe]
    N -->|Include Confidence Score| O[Annotate image with labels]
    
    O -->|All boxes processed| P[Save Result image as result-UUID.jpg]
    H --> P
    
    P -->|HTML Template Response| Q[Render UI index.html with Annotations]
    P -->|JSON API response| R[Return predictions list & coordinates]
```

---

## Key Features
- **Hybrid Machine Learning Pipeline**: Combines object detection (YOLOv8) with deep learning classification (CNN) to achieve high-precision severity analysis.
- **Color-Coded Visual Feedback**: Annotated output highlights regions of concern using intuitive color codes:
  - <kbd>🟢 Green</kbd> for Minor Damage
  - <kbd>🟠 Orange</kbd> for Moderate Damage
  - <kbd>🔴 Red</kbd> for Severe Damage
- **Flexible REST API**: Dedicated endpoints for pipeline runs, CNN-only classification, or YOLOv8-only localization.
- **Self-Healing / Resilient Server**: FastAPI loads models asynchronously at startup. If binaries are absent, the server runs in a `degraded` state, allowing access to diagnostics `/health` rather than crashing.

---

## Project Directory Layout

Below is the directory structure of the repository:

```text
A-Hybrid-CNN-YOLO-Framework-for-Car-Damage-Severity-Classification-and-Localization/
├── .venv/                         # Standard Python Virtual Environment directory (ignored by git)
├── damage_api/                    # Application and MLOps scripts folder
│   ├── app/                       # FastAPI application core
│   │   ├── static/                # Static assets
│   │   │   └── images/            # Holds sample uploads & UUID-labeled results
│   │   ├── templates/             # Jinja2 HTML templates
│   │   │   └── index.html         # Web UI frontend
│   │   ├── __init__.py
│   │   └── main.py                # FastAPI entry point, endpoint handlers, and ModelService
│   ├── configs/
│   │   └── params.yaml            # Model configuration, thresholds, image dimensions, and dataset paths
│   ├── models/                    # Place trained car.h5 and best.pt weight binaries here (ignored by git)
│   ├── src/                       # Custom python scripts for training, validation, and predicting
│   │   ├── __init__.py
│   │   ├── predict.py             # Prediction utilities for CNN and YOLOv8
│   │   ├── train_cnn.py           # TensorFlow/Keras CNN model architecture & training sequence
│   │   ├── train_yolo.py          # YOLOv8 object detection training loop configuration
│   │   ├── validate_cnn.py        # Validation performance reporter for the CNN model
│   │   └── validate_yolo.py       # Metrics evaluator for YOLOv8
│   └── __init__.py
├── tests/                         # Unit tests
│   ├── __init__.py
│   └── test_predict.py            # API wiring and health checks mock verification
├── environment.yml                # Anaconda environment configuration
├── requirements.txt               # Standard pip package requirements list
├── index.html                     # Root client test interface
├── LICENSE                        # MIT License text
└── README.md                      # Professional documentation file
```

---

## Environment Setup & Installation

All setup and execution commands should be run from the repository root directory.

### Option A: Anaconda Prompt Setup (Recommended)
If you manage dependencies with conda, you can instantiate the environment using the provided [environment.yml](file:///c:/Users/basil/OneDrive%20-%20jfmofficial/Desktop/work/car-damage-detection-ai/A-Hybrid-CNN-YOLO-Framework-for-Car-Damage-Severity-Classification-and-Localization/environment.yml).

1. Open your **Anaconda Prompt** terminal.
2. Navigate to the directory containing this project:
   ```cmd
   cd "C:\Users\basil\OneDrive - jfmofficial\Desktop\work\car-damage-detection-ai\A-Hybrid-CNN-YOLO-Framework-for-Car-Damage-Severity-Classification-and-Localization"
   ```
3. Create the conda environment by executing:
   ```cmd
   conda env create -f environment.yml
   ```
4. Activate the newly created environment:
   ```cmd
   conda activate cnn_yolo
   ```

### Option B: Normal Command Prompt / venv Setup
If you prefer standard python virtual environments with `pip`, set up the virtual environment using [requirements.txt](file:///c:/Users/basil/OneDrive%20-%20jfmofficial/Desktop/work/car-damage-detection-ai/A-Hybrid-CNN-YOLO-Framework-for-Car-Damage-Severity-Classification-and-Localization/requirements.txt).

1. Open your terminal (e.g. Command Prompt, PowerShell, or Git Bash).
2. Navigate to the repository root directory.
3. Create a python virtual environment:
   ```powershell
   python -m venv .venv
   ```
4. Activate the virtual environment:
   - **Windows PowerShell**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows Command Prompt (cmd)**:
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
   - **Linux / macOS Bash / Git Bash**:
     ```bash
     source .venv/bin/activate
     ```
5. Upgrade pip and install dependencies:
   ```bash
   pip install --upgrade pip
   ```
   ```bash
   pip install -r requirements.txt
   ```

---

## Trained Models Deployment

The pre-trained model weights are omitted from git control due to file size. Before launching the FastAPI service, copy your trained binary artifacts to the [models](file:///c:/Users/basil/OneDrive%20-%20jfmofficial/Desktop/work/car-damage-detection-ai/A-Hybrid-CNN-YOLO-Framework-for-Car-Damage-Severity-Classification-and-Localization/damage_api/models) folder:

- **TensorFlow CNN**: Copy `car.h5` to `damage_api/models/car.h5`
- **YOLOv8 Detection**: Copy `best.pt` to `damage_api/models/best.pt`

> [!NOTE]
> If your weight models are named differently or located in another folder, update the `model_paths.cnn_model` and `model_paths.yolo_model` configurations in the parameters file: [params.yaml](file:///c:/Users/basil/OneDrive%20-%20jfmofficial/Desktop/work/car-damage-detection-ai/A-Hybrid-CNN-YOLO-Framework-for-Car-Damage-Severity-Classification-and-Localization/damage_api/configs/params.yaml).

---

## Running the FastAPI Server

Start the application from the repository root using the command below:

```powershell
uvicorn damage_api.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Parameter Breakdown
- `--host 0.0.0.0`: Binds the server to all network interfaces, allowing external computers on the local network to query the endpoints.
- `--port 8000`: Runs the application on port `8000`.
- `--reload`: Enables hot-reloading. The server automatically restarts if any python files or template components are modified. **Omit this flag in production environments** to avoid file-watcher resource consumption:
  ```powershell
  uvicorn damage_api.app.main:app --host 0.0.0.0 --port 8000
  ```

---

## API Routes & Interactive Docs

Once the server is running, the following endpoints are available:

| HTTP Method | Route Path | Content-Type | Returns | Description / Use Case |
| :---: | :--- | :--- | :--- | :--- |
| **GET** | `/` | `text/html` | HTML Page | Renders the browser UI client for uploading and processing images interactively. |
| **POST** | `/predict` | `multipart/form-data` | HTML Page | The browser-form target. Detects, crops, rates, draws annotations, and renders results. |
| **POST** | `/predict/cnn` | `multipart/form-data` | `application/json` | CNN-only classification. Evaluates the severity class of the provided image file. |
| **POST** | `/predict/yolo` | `multipart/form-data` | `application/json` | YOLOv8-only localization. Returns an array of bounding boxes coordinates and confidence. |
| **GET** | `/health` | `application/json` | `application/json` | Returns API status (`ok` or `degraded`) and details on loaded models. |

### API Documentation URLs
Interactive swagger pages can be accessed locally via:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc UI**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### CLI Call Examples

**1. Query Model Severity (CNN-Only JSON):**
```powershell
curl.exe -X POST http://127.0.0.1:8000/predict/cnn -F "file=@C:\path\to\car_damage.jpg"
```
*Sample JSON Response:*
```json
{
  "class": "moderate",
  "confidence": 0.8427
}
```

**2. Query Localization (YOLOv8-Only JSON):**
```powershell
curl.exe -X POST http://127.0.0.1:8000/predict/yolo -F "file=@C:\path\to\car_damage.jpg"
```
*Sample JSON Response:*
```json
{
  "boxes": [
    {
      "box": [112.5, 45.2, 342.1, 280.9],
      "confidence": 0.8924
    }
  ]
}
```

---

## Training & Validation Pipeline

To re-train or validate your custom models:
1. Restore the dataset directories listed under the `data` properties in [params.yaml](file:///c:/Users/basil/OneDrive%20-%20jfmofficial/Desktop/work/car-damage-detection-ai/A-Hybrid-CNN-YOLO-Framework-for-Car-Damage-Severity-Classification-and-Localization/damage_api/configs/params.yaml) (e.g. `data/training`, `car_damage_yolo/images/train`, etc.).
2. Run the MLOps scripts from the repository root:

*   **Train custom CNN**:
    ```powershell
    python -m damage_api.src.train_cnn
    ```
*   **Train custom YOLOv8**:
    ```powershell
    python -m damage_api.src.train_yolo
    ```
*   **Validate custom CNN**:
    ```powershell
    python -m damage_api.src.validate_cnn
    ```
*   **Validate custom YOLOv8**:
    ```powershell
    python -m damage_api.src.validate_yolo
    ```

---

## Running Unit Tests

To run the test cases mock verification framework, execute the following command:

```powershell
python -m unittest discover -s tests
```

---

## License

This project is licensed under the terms of the MIT License. For details, refer to the [LICENSE](file:///c:/Users/basil/OneDrive%20-%20jfmofficial/Desktop/work/car-damage-detection-ai/A-Hybrid-CNN-YOLO-Framework-for-Car-Damage-Severity-Classification-and-Localization/LICENSE) file at the root of the workspace.
