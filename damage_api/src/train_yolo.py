"""Train the YOLO model for car damage detection."""

from __future__ import annotations

from pathlib import Path
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore


def load_config() -> dict:
    """Load configuration from params.yaml."""
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "params.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    project_root = Path(__file__).resolve().parents[1]

    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is not installed. Install it first.")

    yolo_cfg = cfg["yolo"]

    model_name = yolo_cfg.get("model_name", "yolov8n.pt")
    data_yaml = yolo_cfg.get("data_yaml", "car_damage_yolo/data.yaml")
    epochs = yolo_cfg.get("epochs", 20)
    imgsz = yolo_cfg.get("imgsz", 640)
    batch_size = yolo_cfg.get("batch_size", 16)
    experiment_name = yolo_cfg.get("experiment_name", "car_damage_detector")

    # Convert relative dataset path to absolute path
    data_yaml_path = project_root / data_yaml
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found at: {data_yaml_path}")

    # Load YOLO base model
    model = YOLO(model_name)

    # Train
    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name=experiment_name,
    )

    print(f"Training completed. Check runs/detect/{experiment_name}/weights for best.pt")


if __name__ == "__main__":
    main()