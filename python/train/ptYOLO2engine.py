# File: export_yolo.py
# (Run this file ONCE)

from ultralytics import YOLO
import os

print("Starting YOLO model export...")

# Define paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
YOLO_PT_PATH = os.path.join(MODEL_DIR, r"D:\CODING\BU-ELEPHANT-HACKATHON\models\yolov8n.pt")
YOLO_ENGINE_PATH = os.path.join(MODEL_DIR, "yolov8n.engine")

# Load the slow .pt model
try:
    model = YOLO(YOLO_PT_PATH)
    print(f"Loaded '{YOLO_PT_PATH}' successfully.")
except Exception as e:
    print(f"ERROR: Could not load '{YOLO_PT_PATH}'. Make sure the file exists.")
    print(f"Details: {e}")
    exit()

# Export to TensorRT (.engine) for maximum speed
try:
    print("Exporting to TensorRT (.engine)... This may take a few minutes.")
    model.export(format='engine') 
    print(f"✅ Success! Optimized model saved as '{YOLO_ENGINE_PATH}'.")
except Exception as e:
    print(f"❌ ERROR: TensorRT export failed.")
    print("This usually means your NVIDIA drivers, CUDA, or TensorRT libraries are not set up correctly.")
    print(f"Details: {e}")