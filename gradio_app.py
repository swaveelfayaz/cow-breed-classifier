import gradio as gr
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf

# Your existing predict.py logic here (copy from src/predict.py)
# ... (load model, preprocess, predict_breed_from_bgr)

def predict_image(img):
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    result = predict_breed_from_bgr(img_bgr)
    return f"**{result['breed'].title()}** (confidence: {result['confidence']:.2%})"

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🐄 Cow Breed Classifier",
    description="Upload cow photo → Get breed prediction!"
)

iface.launch()
