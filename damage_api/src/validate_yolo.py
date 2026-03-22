from __future__ import annotations

from pathlib import Path
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "params.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    project_root = Path(__file__).resolve().parents[1]

    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is not installed.")

    yolo_model_path = project_root / cfg["model_paths"]["yolo_model"]
    data_yaml_path = project_root / cfg["yolo"]["data_yaml"]

    if not yolo_model_path.exists():
        raise FileNotFoundError(f"YOLO model not found at: {yolo_model_path}")

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found at: {data_yaml_path}")

    model = YOLO(str(yolo_model_path))

    metrics = model.val(data=str(data_yaml_path))

    print("\nYOLO Validation Completed")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"Precision: {metrics.box.mp}")
    print(f"Recall: {metrics.box.mr}")


if __name__ == "__main__":
    main()