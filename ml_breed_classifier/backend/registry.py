from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image


import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model


class ModelHandle:
    def __init__(self, model, class_names: Optional[list], is_custom: bool, input_size: Tuple[int, int]):
        self.model = model
        self.class_names = class_names
        self.is_custom = is_custom
        self.input_size = input_size


_REGISTRY: Dict[str, ModelHandle] = {}
_DEFAULT_INPUT_SIZE = (224, 224)


def _load_class_names(dir_path: Path) -> Optional[list]:
    class_file = dir_path / 'class_names.txt'
    if class_file.exists():
        with open(class_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    return None


def _load_custom_model(dir_path: Path) -> Optional[tf.keras.Model]:
    keras_path = dir_path / 'breed_classifier.keras'
    h5_path = dir_path / 'breed_classifier.h5'
    if keras_path.exists():
        return load_model(keras_path)
    if h5_path.exists():
        return load_model(h5_path)
    return None


def get_model(dataset_key: str) -> ModelHandle:
    if dataset_key in _REGISTRY:
        return _REGISTRY[dataset_key]

    base_dir = Path('models') / dataset_key
    base_dir.mkdir(parents=True, exist_ok=True)

    custom = _load_custom_model(base_dir)
    if custom is not None:
        class_names = _load_class_names(base_dir)
        handle = ModelHandle(custom, class_names, True, _DEFAULT_INPUT_SIZE)
        _REGISTRY[dataset_key] = handle
        return handle

    # Fallback to ImageNet
    imagenet_model = MobileNetV2(weights="imagenet", include_top=True)
    handle = ModelHandle(imagenet_model, None, False, _DEFAULT_INPUT_SIZE)
    _REGISTRY[dataset_key] = handle
    return handle


def preprocess_image(image: Image.Image, input_size: Tuple[int, int], is_custom: bool) -> np.ndarray:
    resized = image.resize(input_size)
    array = tf.keras.preprocessing.image.img_to_array(resized)
    array = np.expand_dims(array, axis=0)
    # For custom models, training graph already applied preprocess_input inside the model,
    # so we should pass raw [0,255] inputs. Only apply preprocess_input for non-custom (ImageNet) model.
    if not is_custom:
        array = preprocess_input(array)
    return array


def predict(image: Image.Image, dataset_key: str) -> Dict[str, object]:
    handle = get_model(dataset_key)
    x = preprocess_image(image, handle.input_size, handle.is_custom)
    preds = handle.model.predict(x, verbose=0)

    if handle.is_custom and handle.class_names is not None:
        idx = int(np.argmax(preds[0]))
        conf = float(preds[0][idx])
        label = handle.class_names[idx].replace('_', ' ')
        return {"prediction": label, "confidence": round(conf * 100.0, 2)}

    decoded = decode_predictions(preds, top=1)[0][0]
    _class_id, label, score = decoded
    return {"prediction": label.replace('_', ' '), "confidence": round(float(score) * 100.0, 2)}


def load_metrics(dataset_key: str) -> Optional[dict]:
    metrics_path = Path('models') / dataset_key / 'metrics.json'
    if metrics_path.exists():
        import json
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None
