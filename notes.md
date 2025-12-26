# AI-Powered Cow Breed Detection System – Notes

## Problem statement
Build an end-to-end deep learning system that:
- Takes an image of a cow (from file, camera, or mobile app).
- Uses YOLOv8 to detect the cow and crop the cow region.
- Uses a MobileNetV2-based classifier (transfer learning) to classify the crop into 10 breeds:
  Gir, Sahiwal, Kankrej, Red Sindhi, Ongole, Tharparkar, Jersey,
  Holstein Friesian, Brown Swiss, Ayrshire.
- Outputs `breed_name + confidence` via:
  - REST API (FastAPI),
  - Web UI (Streamlit),
  - Mobile app (Flutter + TFLite).

## Goals
- Good accuracy on all 10 breeds with a small, fast model.
- Clean, modular code structure.
- Reproducible training (notebooks) and easy deployment (API + web + mobile).


## System architecture

Pipeline:
1. User provides an image (web upload, camera, or mobile).
2. Backend runs YOLOv8 (single class: "cow") to detect bounding box.
3. Crop the largest cow region.
4. Preprocess crop (resize to 224×224, MobileNetV2 preprocess).
5. MobileNetV2 classifier (10 classes) → probabilities.
6. Return:
   - top-1 breed and confidence,
   - optional Grad-CAM heatmap (web).

Components:
- Data:
  - Images per breed (raw + processed folders).
  - YOLOv8 detection dataset (images + bounding boxes).
- Models:
  - YOLOv8s detector (`yolov8_cow_best.pt`).
  - MobileNetV2 classifier (`cow_breed_mobilenetv2.h5`, `.tflite`).
- Backend:
  - FastAPI `/predict` endpoint for image upload and JSON response.
- Web:
  - Streamlit app for UI (upload + camera + results + heatmap).
- Mobile:
  - Flutter app using the TFLite model (or calling the API).
