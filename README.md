# A Hybrid CNN-YOLO Framework for Car Damage Severity Classification and Localization

Repository: [Basilbaasi/A-Hybrid-CNN-YOLO-Framework-for-Car-Damage-Severity-Classification-and-Localization](https://github.com/Basilbaasi/A-Hybrid-CNN-YOLO-Framework-for-Car-Damage-Severity-Classification-and-Localization)

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-blue?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An end-to-end computer vision framework for localized car damage detection and severity classification. The system combines **Ultralytics YOLOv8** for damage localization with a custom **TensorFlow/Keras CNN** for severity classification into **minor**, **moderate**, or **severe** categories. The pipeline is served through a **FastAPI** web interface and REST API.

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

## Workflow Overview

The hybrid pipeline works as follows:

1. A user uploads an image through the FastAPI web interface or API.
2. YOLOv8 detects damaged regions and returns bounding boxes.
3. Each detected region is cropped and resized for CNN inference.
4. The CNN predicts the severity as minor, moderate, or severe.
5. The final output includes annotated bounding boxes, confidence scores, and severity labels.

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

The repository layout is shown below:

```text
repo-root/
├── .venv/                         # Local Python virtual environment (ignored by git)
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
If you manage dependencies with conda, you can instantiate the environment using the provided [environment.yml](environment.yml).

1. Open your **Anaconda Prompt** terminal.
2. Navigate to the repository root.
3. Create the conda environment by executing:
   ```cmd
   conda env create -f environment.yml
   ```
4. Activate the newly created environment:
   ```cmd
   conda activate cnn_yolo
   ```

### Option B: Normal Command Prompt / venv Setup
If you prefer standard Python virtual environments with `pip`, set up the environment using [requirements.txt](requirements.txt).

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

The pre-trained model weights are omitted from git control due to file size. Before launching the FastAPI service, copy your trained binary artifacts to the [damage_api/models](damage_api/models) folder:

- **TensorFlow CNN**: Copy `car.h5` to `damage_api/models/car.h5`
- **YOLOv8 Detection**: Copy `best.pt` to `damage_api/models/best.pt`

> [!NOTE]
> If your weight models are named differently or located elsewhere, update the `model_paths.cnn_model` and `model_paths.yolo_model` settings in [damage_api/configs/params.yaml](damage_api/configs/params.yaml).

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
1. Restore the dataset directories referenced in [damage_api/configs/params.yaml](damage_api/configs/params.yaml) (for example `data/training` or `car_damage_yolo/images/train`).
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

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
