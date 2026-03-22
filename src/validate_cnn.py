"""Evaluate the CNN model on the validation dataset."""
from __future__ import annotations
import os
from pathlib import Path
from collections import defaultdict
import yaml
from .predict import predict_cnn

def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "params.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def evaluate() -> None:
    cfg = load_config()
    project_root = Path(__file__).resolve().parents[1]
    data_cfg = cfg.get("data", {})
    val_path = project_root / (data_cfg.get("cnn_validation_path")
                               or data_cfg.get("validation_path")
                               or "data/validation")
    class_names = cfg["classes"]["names"]
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    total = 0
    correct = 0
    per_class_counts = defaultdict(int)
    per_class_correct = defaultdict(int)
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    dir_map = {}
    for class_name in class_names:
        mapped_dir = None
        for item in val_path.iterdir():
            if item.is_dir():
                suffix = item.name.lower().split('-', 1)[-1]
                if suffix == class_name.lower():
                    mapped_dir = item
                    break
        if mapped_dir is None:
            mapped_dir = val_path / class_name
        dir_map[class_name] = mapped_dir

    for class_name in class_names:
        true_idx = class_to_idx[class_name]
        class_dir = dir_map.get(class_name)
        if not class_dir or not class_dir.is_dir():
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            image_path = class_dir / fname
            try:
                result = predict_cnn(str(image_path))
            except Exception as e:
                print(f"Warning: failed to predict {image_path}: {e}")
                continue
            pred_class = result.get("class")
            if pred_class is None:
                continue
            pred_idx = class_to_idx.get(pred_class, -1)
            if pred_idx >= 0:
                confusion[true_idx][pred_idx] += 1
            total += 1
            per_class_counts[class_name] += 1
            if pred_class == class_name:
                correct += 1
                per_class_correct[class_name] += 1

    overall_acc = correct / total if total > 0 else 0.0
    print(f"Total samples: {total}")
    print(f"Overall accuracy: {overall_acc:.4f}\n")
    print("Per‑class accuracy:")
    for class_name in class_names:
        count = per_class_counts[class_name]
        corr = per_class_correct[class_name]
        acc = corr / count if count > 0 else 0.0
        print(f"  {class_name}: {acc:.4f} ({corr}/{count})")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    header = "\t" + "\t".join(class_names)
    print(header)
    for i, row in enumerate(confusion):
        row_str = class_names[i] + "\t" + "\t".join(str(n) for n in row)
        print(row_str)

if __name__ == "__main__":
    evaluate()