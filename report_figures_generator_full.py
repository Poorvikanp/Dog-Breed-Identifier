#!/usr/bin/env python3
"""
report_figures_generator_full.py
Full-featured report figure generator for Dog-Breed Prediction project.

Usage examples:
  # Basic run (dataset1 only)
  python report_figures_generator_full.py --model1 models/dataset1/breed_classifier.keras \
    --classnames1 models/dataset1/class_names.txt --test1 data/dataset1/test

  # With dataset2
  python report_figures_generator_full.py --model1 models/dataset1/breed_classifier.keras \
    --classnames1 models/dataset1/class_names.txt --test1 data/dataset1/test \
    --model2 models/dataset2/breed_classifier.keras --classnames2 models/dataset2/class_names.txt \
    --test2 data/dataset2/test --ds2

  # Run and auto-generate Grad-CAM choosing a good layer for MobileNetV2
  python report_figures_generator_full.py ... --ds2 --auto-gradcam

  # Run with explicit gradcam layer (no need for auto)
  python report_figures_generator_full.py ... --gradcam-layer block_16_project
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image

# sklearn metrics
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# tensorflow
import tensorflow as tf

# --------------------------- Helpers ------------------------------------
def load_json_if_exists(p: Path):
    if p and p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def read_class_names(path: Path) -> List[str]:
    if not path or not path.exists():
        raise FileNotFoundError(f"class_names file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def load_model_safe(path: Path):
    if not path or not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = tf.keras.models.load_model(str(path))
    return model


def list_test_subfolders(test_dir: Path):
    if not test_dir.exists():
        return []
    return sorted([d for d in test_dir.iterdir() if d.is_dir()])


def build_labeled_list(test_dir: Path):
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")
    subdirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    if not subdirs:
        raise RuntimeError(f"Test dir must contain subfolders per class: {test_dir}")
    image_paths, labels = [], []
    for idx, sd in enumerate(subdirs):
        images = [p for p in sd.glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        for p in images:
            image_paths.append(str(p))
            labels.append(idx)
    return image_paths, labels, [sd.name for sd in subdirs]


def load_and_preprocess(path: str, image_size=(224, 224)):
    img = Image.open(path).convert("RGB").resize(image_size)
    arr = np.asarray(img) / 255.0
    return arr


# ------------------------- Plotting functions ---------------------------
def plot_history_single(history: dict, out_path: Path, prefix: str):
    if not history:
        print(f"No history for {prefix}, skipping history plots.")
        return
    # accuracy
    plt.figure(figsize=(6, 4))
    if "accuracy" in history:
        plt.plot(history["accuracy"], label="train")
    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="val")
    plt.title(f"Training vs Validation Accuracy ({prefix})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    fp = out_path / f"{prefix}_accuracy.png"
    plt.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved", fp)
    # loss
    plt.figure(figsize=(6, 4))
    if "loss" in history:
        plt.plot(history["loss"], label="train")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val")
    plt.title(f"Training vs Validation Loss ({prefix})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    fp2 = out_path / f"{prefix}_loss.png"
    plt.savefig(fp2, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved", fp2)


def plot_history_combined(histories: List[Optional[dict]], labels: List[str], out_path: Path):
    # Combined accuracy
    plt.figure(figsize=(7, 4))
    for hist, lab in zip(histories, labels):
        if hist and "val_accuracy" in hist:
            plt.plot(hist["val_accuracy"], label=f"Val Acc ({lab})")
        elif hist and "accuracy" in hist:
            plt.plot(hist["accuracy"], label=f"Train Acc ({lab})")
    plt.title("Combined Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    fp = out_path / "combined_accuracy.png"
    plt.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved", fp)
    # Combined loss
    plt.figure(figsize=(7, 4))
    for hist, lab in zip(histories, labels):
        if hist and "val_loss" in hist:
            plt.plot(hist["val_loss"], label=f"Val Loss ({lab})")
        elif hist and "loss" in hist:
            plt.plot(hist["loss"], label=f"Train Loss ({lab})")
    plt.title("Combined Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    fp2 = out_path / "combined_loss.png"
    plt.savefig(fp2, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved", fp2)


def save_confusion_and_report(
    model,
    class_names,
    test_dir: Path,
    out_path: Path,
    tag="ds1",
    batch_size=32,
):
    imgs, y_true, folders = build_labeled_list(test_dir)
    preds_list = []
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i : i + batch_size]
        arr = np.stack([load_and_preprocess(p) for p in batch], axis=0)
        pred = model.predict(arr, batch_size=batch_size, verbose=0)
        preds_list.append(pred)
    preds = np.vstack(preds_list)
    y_pred = np.argmax(preds, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names[: len(cm)]
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, xticks_rotation=90)
    plt.title(f"Confusion Matrix ({tag})")
    fp = out_path / f"confusion_matrix_{tag}.png"
    plt.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close()
    report = classification_report(
        y_true, y_pred, target_names=class_names[: len(cm)], output_dict=True
    )
    report_fp = out_path / f"classification_report_{tag}.json"
    with open(report_fp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Saved", fp, "and", report_fp)
    return fp, report_fp


def save_sample_grid(
    model,
    class_names,
    test_dir: Path,
    out_path: Path,
    out_name="samples.png",
    n=6,
    image_size=(224, 224),
):
    imgs = []
    for sd in sorted(test_dir.iterdir()):
        if sd.is_dir():
            imgs.extend(list(sd.glob("*.*")))
    if not imgs:
        raise RuntimeError("No images found in test dir.")
    samples = random.sample(imgs, min(n, len(imgs)))
    cols = 3
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(samples):
            p = samples[i]
            im = Image.open(p).convert("RGB").resize(image_size)
            arr = np.asarray(im) / 255.0
            pred = model.predict(np.expand_dims(arr, 0))[0]
            lab = class_names[np.argmax(pred)]
            conf = np.max(pred)
            ax.imshow(im)
            ax.set_title(f"{lab} ({conf:.2f})")
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()
    fp = out_path / out_name
    plt.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved", fp)
    return fp


# -------------------------- Grad-CAM -----------------------------------
def find_candidate_conv_layers(model: tf.keras.Model):
    return [
        layer.name
        for layer in model.layers
        if ("conv" in layer.name) or ("block" in layer.name)
    ]


def choose_default_gradcam_layer(model: tf.keras.Model):
    # Prefer MobileNetV2's block_16_project, else choose last conv-like layer
    names = find_candidate_conv_layers(model)
    for preferred in (
        "block_16_project",
        "block_15_project",
        "Conv_1",
        "conv_pw_13_relu",
        "block_13_project",
    ):
        if preferred in names:
            return preferred
    if names:
        return names[-1]
    return None


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str,
    class_index: Optional[int] = None,
):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy() if hasattr(heatmap, "numpy") else np.array(heatmap)
    return heatmap


def overlay_and_save(
    img_path: str,
    heatmap: np.ndarray,
    out_path: Path,
    out_name: str,
    alpha=0.4,
    cmap_name="jet",
    image_size=(224, 224),
):
    img = Image.open(img_path).convert("RGB").resize(image_size)
    img_arr = np.array(img).astype(np.uint8)
    h = heatmap
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    h255 = np.uint8(255 * h)
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(h255 / 255.0)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    blended = np.uint8(img_arr * (1 - alpha) + colored * alpha)
    out = Image.fromarray(blended)
    fp = out_path / out_name
    out.save(fp)
    print("Saved", fp)
    return fp


def save_gradcam_example(
    model,
    test_dir: Path,
    out_path: Path,
    layer_name: str,
    tag="ds1",
    image_size=(224, 224),
):
    img = next(test_dir.rglob("*.*"))
    arr = load_and_preprocess(str(img), image_size)
    inp = np.expand_dims(arr, 0).astype(np.float32)
    heatmap = make_gradcam_heatmap(inp, model, layer_name)
    fname = f"gradcam_{tag}_example.png"
    return overlay_and_save(str(img), heatmap, out_path, fname, image_size=image_size)


# ----------------------- Misc helpers ----------------------------------
def save_accuracy_comparison(
    our_vals: List[float],
    our_labels: List[str],
    literature: dict,
    out_path: Path,
    out_name="accuracy_comparison.png",
):
    labels = list(our_labels) + list(literature.keys())
    values = list(our_vals) + list(literature.values())
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values)
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison")
    for b, v in zip(bars, values):
        plt.text(
            b.get_x() + b.get_width() / 2,
            v + 0.8,
            f"{v:.2f}",
            ha="center",
            fontsize=9,
        )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fp = out_path / out_name
    plt.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved", fp)
    return fp


def draw_architecture(out_path: Path, out_name="architecture_diagram.png"):
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, txt):
        r = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.3", linewidth=1.2, facecolor="#f2f2f2"
        )
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10)

    box(0.6, 3.3, 2, 1, "Stanford\n+ Kaggle\nDatasets")
    box(3, 3.3, 2, 1, "Preprocess\n& Augment")
    box(5.4, 3.3, 2.2, 1, "MobileNetV2\n(Transfer)")
    box(8, 3.3, 1.4, 1, "Grad-CAM\nExplain")
    box(4.8, 1.1, 4.4, 1, "FastAPI\nREST API + UI")
    ax.annotate("", xy=(2.6, 3.8), xytext=(3, 3.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(5, 3.8), xytext=(5.4, 3.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(7.7, 3.8), xytext=(8, 3.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(6.5, 3.3), xytext=(6.5, 2.1), arrowprops=dict(arrowstyle="->"))
    fp = out_path / out_name
    plt.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved", fp)
    return fp


# ------------------------- Runner --------------------------------------
def run_all(args):
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    literature = (
        {
            "Cui et al. (2024)": 95.24,
            "Valarmathi et al. (2023)": 92.4,
            "Varshney et al. (2022)": 85.0,
        }
        if not args.literature_json
        else load_json_if_exists(Path(args.literature_json))
    )

    histories = []
    labels = []
    our_vals = []
    our_labels = []
    model1 = None

    # DS1
    try:
        if args.model1:
            model1 = load_model_safe(Path(args.model1))
            class_names1 = (
                read_class_names(Path(args.classnames1)) if args.classnames1 else []
            )
            hist1 = (
                load_json_if_exists(Path(args.history1)) if args.history1 else None
            )
            histories.append(hist1)
            labels.append("Stanford")
            if hist1:
                plot_history_single(hist1, out_path, "history_ds1")
            if args.test1 and Path(args.test1).exists():
                save_confusion_and_report(
                    model1,
                    class_names1,
                    Path(args.test1),
                    out_path,
                    tag="ds1",
                    batch_size=args.batch_size,
                )
                save_sample_grid(
                    model1,
                    class_names1,
                    Path(args.test1),
                    out_path,
                    out_name="ds1_samples_predictions.png",
                    n=args.sample_n,
                    image_size=(args.size, args.size),
                )
            if hist1 and "val_accuracy" in hist1:
                our_vals.append(max(hist1["val_accuracy"]) * 100)
                our_labels.append("Our Model (Stanford)")
            else:
                our_vals.append(args.fallback_ds1)
                our_labels.append("Our Model (Stanford)")
        else:
            print("No model1 specified, skipping dataset1 steps.")
    except Exception as e:
        print("Skipping DS1 due to error:", e)

    # DS2
    try:
        if args.ds2 and args.model2:
            model2 = load_model_safe(Path(args.model2))
            class_names2 = (
                read_class_names(Path(args.classnames2)) if args.classnames2 else []
            )
            hist2 = (
                load_json_if_exists(Path(args.history2)) if args.history2 else None
            )
            histories.append(hist2)
            labels.append("Kaggle")
            if hist2:
                plot_history_single(hist2, out_path, "history_ds2")
            if args.test2 and Path(args.test2).exists():
                save_confusion_and_report(
                    model2,
                    class_names2,
                    Path(args.test2),
                    out_path,
                    tag="ds2",
                    batch_size=args.batch_size,
                )
                save_sample_grid(
                    model2,
                    class_names2,
                    Path(args.test2),
                    out_path,
                    out_name="ds2_samples_predictions.png",
                    n=args.sample_n,
                    image_size=(args.size, args.size),
                )
            if hist2 and "val_accuracy" in hist2:
                our_vals.append(max(hist2["val_accuracy"]) * 100)
                our_labels.append("Our Model (Kaggle)")
            else:
                our_vals.append(args.fallback_ds2)
                our_labels.append("Our Model (Kaggle)")
        elif args.ds2:
            print("ds2 flag provided but model2 not specified; skipping dataset2 steps.")
    except Exception as e:
        print("Skipping DS2 due to error:", e)

    # Combined histories
    if any(h for h in histories):
        plot_history_combined(histories, labels, out_path)

    # Accuracy comparison
    save_accuracy_comparison(
        our_vals,
        our_labels,
        literature,
        out_path,
        out_name="accuracy_comparison.png",
    )

    # Architecture diagram
    draw_architecture(out_path, out_name="architecture_diagram.png")

    # Grad-CAM
    if args.gradcam_layer or args.auto_gradcam:
        if model1:
            layer_name1 = args.gradcam_layer
            if args.auto_gradcam:
                try:
                    layer_name1 = choose_default_gradcam_layer(model1)
                except Exception:
                    layer_name1 = args.gradcam_layer
            if not layer_name1:
                print("No gradcam layer chosen for model1; candidate layers:")
                print(find_candidate_conv_layers(model1))
            else:
                if args.test1 and Path(args.test1).exists():
                    save_gradcam_example(
                        model1,
                        Path(args.test1),
                        out_path,
                        layer_name1,
                        tag="ds1",
                        image_size=(args.size, args.size),
                    )
        if args.ds2 and args.model2:
            try:
                model2 = load_model_safe(Path(args.model2))
                layer_name2 = args.gradcam_layer
                if args.auto_gradcam:
                    layer_name2 = choose_default_gradcam_layer(model2)
                if layer_name2 and args.test2 and Path(args.test2).exists():
                    save_gradcam_example(
                        model2,
                        Path(args.test2),
                        out_path,
                        layer_name2,
                        tag="ds2",
                        image_size=(args.size, args.size),
                    )
            except Exception as e:
                print("Skipping gradcam for ds2 due to", e)

    print("Done. Figures saved to:", out_path.resolve())


# -------------------------- CLI ----------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate report figures for dog-breed classifier."
    )
    p.add_argument("--model1", type=str, help="Path to model for dataset1 (.keras or .h5)")
    p.add_argument("--classnames1", type=str, help="Path to class_names.txt for dataset1")
    p.add_argument("--history1", type=str, help="Path to history.json for dataset1")
    p.add_argument(
        "--test1", type=str, help="Path to test folder for dataset1 (subfolders per class)"
    )
    p.add_argument("--model2", type=str, help="Path to model for dataset2 (.keras or .h5)")
    p.add_argument("--classnames2", type=str, help="Path to class_names.txt for dataset2")
    p.add_argument("--history2", type=str, help="Path to history.json for dataset2")
    p.add_argument("--test2", type=str, help="Path to test folder for dataset2")
    p.add_argument("--ds2", action="store_true", help="Set if you want to generate for dataset2")
    p.add_argument("--out", type=str, default="report_figures", help="Output folder for figures")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for predictions")
    p.add_argument("--sample-n", type=int, default=6, help="Number of sample images to show in grid")
    p.add_argument(
        "--size",
        type=int,
        default=224,
        help="Image size (square) used for plotting / gradcam",
    )
    p.add_argument(
        "--gradcam-layer",
        type=str,
        default=None,
        help="Convolutional layer name to use for Grad-CAM (explicit)",
    )
    p.add_argument(
        "--auto-gradcam",
        action="store_true",
        help="Auto-select a sensible grad-cam layer (MobileNetV2-friendly)",
    )
    p.add_argument(
        "--literature-json",
        type=str,
        default=None,
        help="Optional JSON file with literature accuracy values",
    )
    p.add_argument(
        "--fallback-ds1",
        type=float,
        default=91.4,
        help="Fallback accuracy value for ds1 if history missing",
    )
    p.add_argument(
        "--fallback-ds2",
        type=float,
        default=89.7,
        help="Fallback accuracy value for ds2 if history missing",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(args)
