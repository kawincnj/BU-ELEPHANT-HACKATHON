from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import os
import sys

# Import repo root so we can import main.py
BASE_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(REPO_ROOT)

import main  # loads your models/refs/transform/device

app = FastAPI(title="Elephant Identification (main.py-backed)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(getattr(main, "device", "unknown")),
        "yolo_loaded": True,
        "tni_loaded": True,
        "num_refs": len(getattr(main, "reference_embeddings", {})),
    }

@app.post("/run-main")
async def run_main(
    file: UploadFile = File(...),
    conf_thres: float = 0.7,
    id_sim_thres: float = 0.98
):
    """Accept an image, convert to OpenCV BGR, call main.predict_on_frame, return detections."""
    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(415, "Upload an image (jpeg/png).")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")

    # Bytes -> RGB -> BGR
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    rgb = np.array(img)
    frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    try:
        dets = main.predict_on_frame(
            frame_bgr,
            conf_thres=conf_thres,
            id_sim_thres=id_sim_thres
        )
    except Exception as e:
        raise HTTPException(400, f"Inference failed: {e}")

    return JSONResponse({"detections": dets})
