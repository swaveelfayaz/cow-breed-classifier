import numpy as np
import tensorflow as tf
from pathlib import Path
import cv2

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "ayrshire",
    "brown_swiss",
    "holstein_friesian",
    "jersey",
    "red_dane",
]

_model = None

def load_model():
    global _model
    if _model is None:
        model_path = Path("models/cow_breed_classifier_finetuned.h5")
        _model = tf.keras.models.load_model(model_path)
    return _model

def preprocess_image_bgr(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, IMG_SIZE)
    image_array = image_resized.astype("float32")
    image_array = (image_array / 127.5) - 1.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_breed_from_bgr(image_bgr):
    model = load_model()
    inp = preprocess_image_bgr(image_bgr)
    probs = model.predict(inp)[0]
    top_idx = int(np.argmax(probs))
    breed = CLASS_NAMES[top_idx]
    confidence = float(probs[top_idx])
    return {
        "success": True,
        "breed": breed,
        "confidence": confidence,
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
    }
