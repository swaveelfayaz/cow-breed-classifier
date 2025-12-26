import tensorflow as tf
from pathlib import Path

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

keras_model_path = models_dir / "cow_breed_classifier_finetuned.h5"
tflite_model_path = models_dir / "cow_breed_classifier_finetuned.tflite"

print("Loading Keras model:", keras_model_path)
model = tf.keras.models.load_model(keras_model_path)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("✅ Saved TFLite model to:", tflite_model_path)
