import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
import numpy as np
import cv2

from src.predict import predict_breed_from_bgr

st.title("AI-Powered Cow Breed Classification (5 breeds)")

option = st.radio("Choose input method:", ["Upload image", "Use camera"])

image_bgr = None

if option == "Upload image":
    uploaded = st.file_uploader("Upload a cow image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif option == "Use camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image_bgr is not None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Input image", use_column_width=True)

    if st.button("Predict breed"):
        result = predict_breed_from_bgr(image_bgr)
        if result["success"]:
            st.success(f"Breed: {result['breed']} (confidence: {result['confidence']:.2f})")
            st.write("Probabilities:", result["probabilities"])
        else:
            st.error(result.get("error", "Prediction failed"))
