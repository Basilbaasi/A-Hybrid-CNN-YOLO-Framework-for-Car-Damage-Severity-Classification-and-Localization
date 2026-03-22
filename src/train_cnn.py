"""Train the CNN model for car damage severity classification."""
from __future__ import annotations
from pathlib import Path
import yaml
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D,
                                     BatchNormalization, Dropout,
                                     Flatten, Dense)
from tensorflow.keras.optimizers import Adam

def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "params.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def build_model(input_shape: tuple[int, int, int], num_classes: int,
                dropout: float) -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    return model

def main() -> None:
    cfg = load_config()
    project_root = Path(__file__).resolve().parents[1]
    data_cfg = cfg.get("data", {})
    train_dir = project_root / (data_cfg.get("cnn_training_path")
                                or data_cfg.get("training_path")
                                or "data/training")
    val_dir = project_root / (data_cfg.get("cnn_validation_path")
                              or data_cfg.get("validation_path")
                              or "data/validation")

    input_size = tuple(cfg["cnn"]["input_size"])
    batch_size = cfg["cnn"]["batch_size"]
    epochs = cfg["cnn"]["epochs"]
    dropout = cfg["cnn"]["dropout"]
    lr = cfg["cnn"]["learning_rate"]
    num_classes = cfg["classes"]["num_classes"]

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    target_size = input_size[:2]
    train_ds = train_gen.flow_from_directory(str(train_dir),
                                             target_size=target_size,
                                             batch_size=batch_size)
    val_ds = val_gen.flow_from_directory(str(val_dir),
                                         target_size=target_size,
                                         batch_size=batch_size)

    model = build_model(input_shape=input_size, num_classes=num_classes,
                        dropout=dropout)
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(learning_rate=lr),
                  metrics=["accuracy"])

    model.fit(train_ds, epochs=epochs, validation_data=val_ds)

    model_path = project_root / cfg["model_paths"]["cnn_model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()