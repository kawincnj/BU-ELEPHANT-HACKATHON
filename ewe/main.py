import os
import warnings
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms
import time
import numpy as np
import onnxruntime as ort

warnings.filterwarnings('ignore')

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Paths (relative to this file)
# -------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
ONNX_TNI_PATH = os.path.join(MODEL_DIR, "tni_model.onnx")
YOLO_WEIGHTS_PATH = os.path.join(MODEL_DIR, "yolo11n.pt")
REF_EMB_PATH = os.path.join(MODEL_DIR, "reference_embeddings.pt")

# -------------------------------
# Load TNI Model (ONNX)
# -------------------------------
print("📦 Loading TNI model (ONNX)...")

tni_session = ort.InferenceSession(
    ONNX_TNI_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
tni_input_name = tni_session.get_inputs()[0].name
tni_output_name = tni_session.get_outputs()[0].name
print(f"✅ TNI model loaded: input='{tni_input_name}', output='{tni_output_name}'")

# -------------------------------
# Load YOLO Model
# -------------------------------
yolo_model = YOLO(YOLO_WEIGHTS_PATH)

def get_elephant_class_id(model):
    names = getattr(model, "names", None)
    if isinstance(names, dict):  # Ultralytics commonly exposes a dict {id: name}
        for i, n in names.items():
            if str(n).lower() == "elephant":
                return int(i)
    # Fallback to known COCO id
    return 20


# -------------------------------
# Image transform for identification
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Load reference embeddings
# -------------------------------
try:
    reference_embeddings = torch.load(REF_EMB_PATH, map_location=device, weights_only=True)
except TypeError:
    reference_embeddings = torch.load(REF_EMB_PATH, map_location=device)


# -------------------------------
# Function to identify elephant from cropped image (BGR ndarray)
# -------------------------------
def identify_elephant(cropped_frame, reference_embeddings, tni_session, tni_input_name, tni_output_name, device):
    """Identify elephant from a cropped frame (numpy BGR array). Returns (identity, similarity)."""
    img_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # ONNX inference
    onnx_input = img_tensor.cpu().numpy()
    emb = tni_session.run([tni_output_name], {tni_input_name: onnx_input})[0]
    emb = torch.tensor(emb, device=device)

    max_sim = -1.0
    identity = None
    for name, ref_emb in reference_embeddings.items():
        ref_emb = ref_emb.to(device)
        if ref_emb.dim() == 1:
            ref_emb = ref_emb.unsqueeze(0)
        sim = F.cosine_similarity(emb, ref_emb, dim=1)
        s = float(sim.item())
        if s > max_sim:
            max_sim = s
            identity = name

    return identity, max_sim


# -------------------------------
# Helper for single-frame inference (used by web API)
# -------------------------------
def _identify_from_ndarray_bgr(crop_bgr, refs, tni_session, tni_input_name, tni_output_name, device, transform):
    """Internal helper: identify from a BGR crop using the ONNX TNI pipeline."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None, -1.0
    crop_rgb = crop_bgr[:, :, ::-1]
    pil_img = Image.fromarray(crop_rgb)
    x = transform(pil_img).unsqueeze(0).to(device)

    # ONNX inference
    onnx_input = x.cpu().numpy()
    emb = tni_session.run([tni_output_name], {tni_input_name: onnx_input})[0]
    emb = torch.tensor(emb, device=device)

    best_name, best_sim = None, -1.0
    for name, ref_emb in refs.items():
        ref = ref_emb.to(device)
        if ref.dim() == 1:
            ref = ref.unsqueeze(0)
        sim = F.cosine_similarity(emb, ref, dim=1).item()
        if sim > best_sim:
            best_sim, best_name = sim, name
    return best_name, best_sim


# ============================================================
# Main inference function for a single frame
# ============================================================
def predict_on_frame(frame_bgr,
                     conf_thres: float = 0.90,
                     id_sim_thres: float = 0.98):
    start = time.perf_counter()
    eid = get_elephant_class_id(yolo_model)

    # Run YOLO with class + confidence constraints
    results = yolo_model.predict(
        frame_bgr,
        conf=conf_thres,
        classes=[eid],         # ← only elephants
        verbose=False
    )[0]

    outs = []
    for box in results.boxes:
        conf = float(box.conf.item() if hasattr(box.conf, "item") else box.conf)
        if conf < conf_thres:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            continue

        name, sim = _identify_from_ndarray_bgr(crop, reference_embeddings, tni_session,
                                               tni_input_name, tni_output_name, device, transform)
        final_name = name if (sim is not None and sim >= id_sim_thres) else "Unknown"
        text = f"Elephant: {final_name} ({conf*100:.1f}%) | Sim: {sim:.3f}"

        outs.append({
            "bbox": [x1, y1, x2, y2],
            "yolo_label": "elephant",
            "confidence": conf,
            "identity": final_name,
            "similarity": float(sim if sim is not None else 0.0),
            "overlay_text": text
        })

    elapsed = time.perf_counter() - start
    return {
        "detections": outs,
        "process_time_sec": round(elapsed, 4),
        "process_time_ms": round(elapsed * 1000, 2),
    }


# ============================================================
# Desktop video loop
# ============================================================
if __name__ == "__main__":
    VIDEO_PATH = r"D:\CODING\BU-ELEPHANT-HACKATHON\Videos\รัศมี.MP4"

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    CONFIDENCE_THRESHOLD = 0.95
    IDENTIFICATION_THRESHOLD = 0.98

    print("🎥 Starting video processing...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ End of video or cannot read frame")
            break

        results = yolo_model(frame)[0]

        for box in results.boxes:
            conf = float(box.conf)
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id] if hasattr(yolo_model, "names") else str(cls_id)

            cropped_elephant = frame[y1:y2, x1:x2]
            if cropped_elephant.shape[0] < 50 or cropped_elephant.shape[1] < 50:
                continue

            identity, similarity = identify_elephant(
                cropped_elephant, reference_embeddings, tni_session,
                tni_input_name, tni_output_name, device
            )

            if similarity <= IDENTIFICATION_THRESHOLD:
                continue

            color = (0, 255, 0)
            text = f"{identity} ({similarity:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("Elephant Detection & Identification (ONNX)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Video processing completed (ONNX mode)")
