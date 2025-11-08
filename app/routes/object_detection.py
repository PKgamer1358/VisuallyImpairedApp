from fastapi import APIRouter, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

router = APIRouter(prefix="/detect", tags=["Object Detection"])

# Load model once at startup
model = YOLO("yolov8n.pt")  # or your custom model path

@router.post("/")
async def detect_objects(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run detection
    results = model(image)

    # Extract detection info
    detections = []
    for r in results[0].boxes:
        detections.append({
            "class": model.names[int(r.cls)],
            "confidence": float(r.conf),
            "bbox": r.xyxy[0].tolist()
        })

    return {"detections": detections}