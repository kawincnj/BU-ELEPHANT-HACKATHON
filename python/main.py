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
# Set device (Used only for loading reference embeddings and YOLO)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Device set to: {device}")

# -------------------------------
# 🌟 CONFIGURATION 🌟
# (Configuration remains the same)
# -------------------------------
INPUT_IMAGE_PATH = r"C:\Users\HP\Downloads\img_ Rat_sa_mee_back_4.png"
YOLO_MODEL_PATH = r"models\yolov8n.pt"
ONNX_MODEL_PATH = r"models\tni_elephant_model.onnx" 
REFERENCE_EMBEDDINGS_PATH = r"D:\CODING\BU-ELEPHANT-HACKATHON\models\reference_embeddings_best.pt"
CONFIDENCE_THRESHOLD = 0.7
IDENTIFICATION_THRESHOLD = 0.98

# -------------------------------
# 1. Load YOLO Model
# -------------------------------
yolo_model = YOLO(YOLO_MODEL_PATH)

# -------------------------------
# 2. Initialize ONNX Runtime Session
# -------------------------------
ort_session = None
ONNX_INPUT_NAME = None
ONNX_OUTPUT_NAME = None

try:
    # Use CUDAExecutionProvider for reliable GPU acceleration
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    
    # Check which provider is actually used
    used_provider = ort_session.get_providers()[0]
    if 'CUDAExecutionProvider' in used_provider:
        print(f"✅ ONNX Runtime using: CUDAExecutionProvider (Fast fallback)")
    else:
        print(f"⚠️ ONNX Runtime using: {used_provider} (CPU fallback)")
        
    ONNX_INPUT_NAME = ort_session.get_inputs()[0].name
    ONNX_OUTPUT_NAME = ort_session.get_outputs()[0].name
    print(f"✅ ONNX Model loaded successfully.")

except Exception as e:
    print(f"❌ Error loading ONNX model: {e}")
    # Fallback to pure CPU if all GPU/TRT methods fail
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    ONNX_INPUT_NAME = ort_session.get_inputs()[0].name
    ONNX_OUTPUT_NAME = ort_session.get_outputs()[0].name

# -------------------------------
# 3. Image transform for identification
# (Transforms remain the same)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.456], std=[0.229, 0.224, 0.225])
])

# -------------------------------
# 4. Load reference embeddings
# (Loading and pre-stacking logic remains the same)
# -------------------------------
reference_embeddings = None
try:
    reference_embeddings = torch.load(REFERENCE_EMBEDDINGS_PATH, map_location=device) 
    print(f"✅ Reference embeddings loaded. Total classes: {len(reference_embeddings)}")
except Exception as e:
    print(f"❌ Error loading reference embeddings: {e}")
    exit()

ref_names = list(reference_embeddings.keys())
ref_emb_tensor = torch.stack([reference_embeddings[name].squeeze() for name in ref_names]).to(device) 
ref_emb_tensor = F.normalize(ref_emb_tensor, p=2, dim=1) 
print(f"✅ Reference gallery ready on device: {ref_emb_tensor.device}")

# -------------------------------
# Function to identify elephant using ONNX Runtime (Vectorized)
# (This function is unchanged, it's already optimized)
# -------------------------------
def identify_elephant_onnx(cropped_frame, ref_emb_tensor, ref_names, ort_session, input_name, output_name, device):
    """Identify elephant from a cropped frame (numpy array) using ONNX with batched similarity."""
    
    # 1. Preprocessing
    img_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0) 
    input_data = img_tensor.cpu().numpy().astype(np.float32)
    
    # 2. ONNX Inference
    ort_inputs = {input_name: input_data}
    ort_outputs = ort_session.run([output_name], ort_inputs)
    query_emb = torch.from_numpy(ort_outputs[0]).to(device) # (1, D)

    # 3. Batched similarity search (GPU/Vectorized)
    query_emb = F.normalize(query_emb, p=2, dim=1) # Normalize the query
    
    # We transpose ref_emb_tensor from (N, D) to (D, N)
    sims = torch.matmul(query_emb, ref_emb_tensor.T).squeeze() # -> (N,)
    
    # Get max similarity and its index
    max_sim, idx_tensor = sims.max(dim=0)
    best_match_index = idx_tensor.item() 
    
    identity = ref_names[best_match_index]
    
    return identity, max_sim.item()

# -------------------------------
# Main Single Image Processing (TIMERS MOVED)
# -------------------------------
def process_single_image():
    
    # 🌟 NON-AI TASK 1: Load image from disk
    frame = cv2.imread(INPUT_IMAGE_PATH)
    if frame is None:
        print(f"❌ Error: Could not load image at {INPUT_IMAGE_PATH}")
        return
    
    # This will store the combined time of all AI calls
    total_ai_time = 0.0

    # 🌟 AI TASK 1: YOLO Detection 🌟
    # -------------------------------
    if device.type == 'cuda':
        torch.cuda.synchronize() # Wait for GPU to be ready
    start_time_yolo = time.perf_counter()
    
    results = yolo_model(frame)[0] 
    
    if device.type == 'cuda':
        torch.cuda.synchronize() # Wait for GPU to finish
    end_time_yolo = time.perf_counter()
    total_ai_time += (end_time_yolo - start_time_yolo)
    # -------------------------------

    processed_frame = frame.copy()
        
    for box in results.boxes:
        conf = float(box.conf)
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_elephant = frame[y1:y2, x1:x2]
        if cropped_elephant.shape[0] < 50 or cropped_elephant.shape[1] < 50:
            continue
        
        # 🌟 AI TASK 2: TNI Identification 🌟
        # -------------------------------
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time_tni = time.perf_counter()
        
        identity, similarity = identify_elephant_onnx(
            cropped_elephant,
            ref_emb_tensor,
            ref_names,
            ort_session,
            ONNX_INPUT_NAME,
            ONNX_OUTPUT_NAME,
            device
        )
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time_tni = time.perf_counter()
        total_ai_time += (end_time_tni - start_time_tni)
        # -------------------------------
        
        # 🌟 NON-AI TASK 2: Drawing results
        color = (0, 255, 0) if similarity > IDENTIFICATION_THRESHOLD else (0, 165, 255)
        text = f"{identity} ({similarity:.2f})" if similarity > IDENTIFICATION_THRESHOLD else f"Unknown Elephant ({conf:.2f})"
        
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(processed_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(processed_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    
    # 🌟 NON-AI TASK 3: Display image
    total_time_ms = total_ai_time * 1000
    
    cv2.imshow("Elephant Detection & Identification (ONNX/TRT)", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("✅ Image Processing Completed")
    print(f"Input Image: {os.path.basename(INPUT_IMAGE_PATH)}")
    # 🌟 UPDATED PRINT STATEMENT 🌟
    print(f"Time Usage: {total_time_ms:.2f} ms (Total AI Prediction Time Only)")
    print("="*70)

if __name__ == "__main__":
    print("--- RUN 1: WARM-UP ---")
    process_single_image()
    
    print("\n--- RUN 2: REAL SPEED ---")
    process_single_image()