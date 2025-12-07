import tensorflow as tf
from pathlib import Path

model_path = Path("models/dataset1/breed_classifier.keras")
print("Loading model:", model_path)
model = tf.keras.models.load_model(str(model_path))
print("\n=== MODEL SUMMARY ===")
model.summary()   # shows full architecture

print("\n=== LAYER LIST ===")
for i, layer in enumerate(model.layers):
    print(i, layer.name, "-", layer.__class__.__name__, "| output_shape:", getattr(layer, "output_shape", None))

print("\nModel input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
