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
# This assumes 'server.py' is in a subfolder (like 'app') and 'main.py' is in the parent folder
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(REPO_ROOT)

# -----------------------------------------------------------------
# This import runs ONCE on startup, loading all models into memory.
# This is the correct, efficient way to do it.
import main 
# -----------------------------------------------------------------

app = FastAPI(title="Elephant Identification (Optimized)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Be more specific in production!
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """Check if the server and models are loaded."""
    return {
        "status": "ok",
        "device": str(getattr(main, "device", "unknown")),
        "yolo_provider": getattr(main, "YOLO_MODEL_PATH", "unknown"),
        "tni_provider": getattr(main, "ort_session", None).get_providers()[0],
        "num_refs": len(getattr(main, "ref_names", [])),
    }

@app.post("/run-main")
async def run_main(
    file: UploadFile = File(...),
    conf_thres: float = 0.7,
    id_sim_thres: float = 0.98
):
    """Accept an image, run full inference, and return detections."""
    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(415, "Upload an image (jpeg/png).")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")

    # Bytes -> PIL (RGB)
    try:
        img_pil = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    # PIL (RGB) -> NumPy (RGB) -> NumPy (BGR for OpenCV/YOLO)
    frame_rgb = np.array(img_pil)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    try:
        # -----------------------------------------------------------------
        # Call the optimized function from main.py
        detections = main.predict_on_frame(
            frame_bgr,
            conf_thres=conf_thres,
            id_sim_thres=id_sim_thres
        )
        # -----------------------------------------------------------------
    except Exception as e:
        # Catch potential runtime errors during inference
        print(f"Inference failed: {e}")
        raise HTTPException(500, f"Inference failed: {e}")

    return JSONResponse({"detections": detections})