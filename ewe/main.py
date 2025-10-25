import cv2
import torch
import onnxruntime as ort 
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from ultralytics import YOLO
import warnings
import time 
import os

warnings.filterwarnings('ignore')

# -------------------------------
# 1. SETUP: Paths and Device
# -------------------------------
# Get the directory of this file (e.g., D:\CODING\BU-ELEPHANT-HACKATHON\python)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 🌟 FIX 1: Correct MODEL_DIR to point up one level to the project root's 'models' folder
MODEL_DIR = os.path.join(ROOT_DIR, "..", "models")

# 🌟 FIX 2: Use os.path.join correctly with *only* the filenames
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.onnx") 
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "tni_elephant_model.onnx") 
REFERENCE_EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "reference_embeddings_best.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{__name__}] PyTorch Device set to: {device}")

# -------------------------------
# 2. Load YOLO Model (Optimized)
# -------------------------------
# This will now correctly check if 'D:\CODING\BU-ELEPHANT-HACKATHON\models\yolov8n.onnx' exists
if not os.path.exists(YOLO_MODEL_PATH):
    print(f"Warning: '{YOLO_MODEL_PATH}' not found. Falling back to slow '.pt' file.")
    YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")
    
yolo_model = YOLO(YOLO_MODEL_PATH)
print(f"[{__name__}] YOLO Model loaded from: {YOLO_MODEL_PATH}")

# -------------------------------
# 3. Load TNI Model (ONNX)
# -------------------------------
try:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    used_provider = ort_session.get_providers()[0]
    ONNX_INPUT_NAME = ort_session.get_inputs()[0].name
    ONNX_OUTPUT_NAME = ort_session.get_outputs()[0].name
    print(f"[{__name__}] ONNX TNI Model loaded. Using provider: {used_provider}")
except Exception as e:
    print(f"❌ FATAL: Could not load ONNX model at {ONNX_MODEL_PATH}: {e}")
    exit()

# -------------------------------
# 4. Load & Prepare Reference Embeddings (Vectorized)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    reference_data = torch.load(REFERENCE_EMBEDDINGS_PATH, map_location=device)
    # Re-using the logic from our previous script to load the data correctly
    if isinstance(reference_data, dict) and 'embeddings' in reference_data:
        # Structure: {'embeddings': tensor, 'labels': list}
        ref_emb_tensor = reference_data['embeddings'].to(device)
        ref_names = reference_data['labels']
    else:
        # Fallback for old structure: {name: tensor, ...}
        ref_names = list(reference_data.keys())
        ref_emb_tensor = torch.stack([reference_data[name].squeeze() for name in ref_names]).to(device)
    
    # Pre-normalize the gallery for the fastest search
    ref_emb_tensor = F.normalize(ref_emb_tensor, p=2, dim=1) 
    print(f"[{__name__}] Reference gallery loaded. {len(ref_names)} classes ready on {ref_emb_tensor.device}.")
except Exception as e:
    print(f"❌ FATAL: Could not load Reference Embeddings at {REFERENCE_EMBEDDINGS_PATH}: {e}")
    exit()

# -------------------------------
# 5. Optimized Identification Function
# -------------------------------
def identify_elephant_optimized(cropped_frame_bgr):
    """
    Identify elephant from a cropped BGR frame using ONNX and vectorized similarity.
    Returns (identity, similarity).
    """
    # 1. Preprocessing
    img_rgb = cv2.cvtColor(cropped_frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0) 
    input_data = img_tensor.cpu().numpy().astype(np.float32)
    
    # 2. ONNX Inference
    ort_inputs = {ONNX_INPUT_NAME: input_data}
    ort_outputs = ort_session.run([ONNX_OUTPUT_NAME], ort_inputs)
    query_emb = torch.from_numpy(ort_outputs[0]).to(device) # (1, D)

    # 3. Batched similarity search (GPU/Vectorized)
    query_emb_norm = F.normalize(query_emb, p=2, dim=1) # Normalize the query
    
    # This is (1, D) @ (D, N) -> (1, N)
    sims = torch.matmul(query_emb_norm, ref_emb_tensor.T).squeeze() # -> (N,)
    
    # Get max similarity and its index
    max_sim, idx_tensor = sims.max(dim=0)
    best_match_index = idx_tensor.item() 
    
    identity = ref_names[best_match_index]
    similarity = max_sim.item()
    
    return identity, similarity

# ============================================================
# API Function (This is what your server.py calls)
# ============================================================
def predict_on_frame(frame_bgr,
                     conf_thres: float = 0.7,
                     id_sim_thres: float = 0.98):
    """
    Run YOLO detection + TNI identification on a single BGR frame.
    Returns a list of detection dictionaries.
    """
    assert isinstance(frame_bgr, np.ndarray), "Input must be a NumPy array"

    # --- AI TASK 1: YOLO Detection ---
    # This now runs the .onnx file on the GPU, which is FAST.
    yolo_device_arg = 0 if device.type == 'cuda' else 'cpu'
    results = yolo_model(frame_bgr, verbose=False, device=yolo_device_arg)[0]
    detections = []

    for box in results.boxes:
        conf = float(box.conf)
        if conf < conf_thres:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        
        # Filter for 'elephant' (class 20 in standard COCO 'yolov8n.pt')
        if cls_id != 20: 
            continue
            
        yolo_lbl = yolo_model.names[cls_id]

        # Crop the detected elephant
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            continue

        # --- AI TASK 2: TNI Identification ---
        identity, similarity = identify_elephant_optimized(crop)
        
        # Final decision logic
        final_name = identity if (similarity >= id_sim_thres) else "Unknown"
        overlay_text = f"ID: {final_name} (Conf: {conf*100:.0f}%, Sim: {similarity:.2f})"

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "yolo_label": yolo_lbl,
            "confidence": conf,
            "identity": final_name,
            "similarity": similarity,
            "overlay_text": overlay_text
        })
 
    return detections

# ============================================================
# Desktop video loop — (for testing)
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print(f"Running {__name__} directly as a desktop script...")
    print("This will process a test video.")
    print("To run the server, execute 'uvicorn server:app --reload'")
    print("="*50 + "\n")

    VIDEO_PATH = r"D:\CODING\BU-ELEPHANT-HACKATHON\Videos\รัศมี.MP4"
    if not os.path.exists(VIDEO_PATH):
        print(f"Warning: Test video not found at {VIDEO_PATH}. Exiting test.")
        exit()
        
    cap = cv2.VideoCapture(VIDEO_PATH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Call the same function our server uses
        detections = predict_on_frame(
            frame, 
            conf_thres=0.7, 
            id_sim_thres=0.98
        )
        
        # Draw results
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            text = det['overlay_text']
            
            color = (0, 255, 0) if det['identity'] != "Unknown" else (0, 165, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("Server Logic Test (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Video processing completed")