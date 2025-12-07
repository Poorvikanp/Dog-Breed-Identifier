import argparse
import json
from pathlib import Path

import tensorflow as tf


def build_datasets(prepared_dir: Path, img_size=(224, 224), batch_size=32):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        prepared_dir / "train",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        prepared_dir / "val",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model(num_classes: int, img_size=(224, 224)):
    base = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=img_size + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def fine_tune(model: tf.keras.Model, base: tf.keras.Model, train_ds, val_ds, fine_tune_at=100):
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=5)
    return history


def save_artifacts(models_dir: Path, model: tf.keras.Model, class_names, metrics: dict):
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / "breed_classifier.keras")
    with open(models_dir / "class_names.txt", "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(f"{c}\n")
    with open(models_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train dog breed classifier for a dataset key")
    parser.add_argument("--dataset", choices=["dataset1", "dataset2"], required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--models_root", default="models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    prepared = Path(args.data_root) / args.dataset / "prepared"
    train_ds, val_ds, class_names = build_datasets(prepared, (args.img_size, args.img_size))

    model, base = build_model(num_classes=len(class_names), img_size=(args.img_size, args.img_size))
    _ = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    _ = fine_tune(model, base, train_ds, val_ds)

    eval_res = model.evaluate(val_ds, verbose=0)
    metrics = {
        "val_loss": float(eval_res[0]),
        "val_accuracy": float(eval_res[1]),
        "epochs_stage1": int(args.epochs),
        "epochs_stage2": 5,
        "num_classes": len(class_names),
    }

    models_dir = Path(args.models_root) / args.dataset
    save_artifacts(models_dir, model, class_names, metrics)

    print("Saved:")
    print(models_dir / "breed_classifier.keras")
    print(models_dir / "class_names.txt")
    print(models_dir / "metrics.json")


if __name__ == "__main__":
    main()
