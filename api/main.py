from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

from src.predict import predict_breed_from_bgr

app = FastAPI(title="Cow Breed API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return {"success": False, "error": "Invalid image"}

    return predict_breed_from_bgr(image_bgr)
