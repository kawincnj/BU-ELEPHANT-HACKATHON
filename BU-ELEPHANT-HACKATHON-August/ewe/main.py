import os
import warnings
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms
 
warnings.filterwarnings('ignore')
 
# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# -------------------------------
# TNI Model Definition
# -------------------------------
class TNIModel(torch.nn.Module):
    def __init__(self, embedding_dim=64):
        super(TNIModel, self).__init__()
        # NOTE: torch.hub will fetch weights if not cached. Requires internet on first run.
        resnet = torch.hub.load('pytorch/vision:v0.15.2', 'resnet50', weights='IMAGENET1K_V1')
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = torch.nn.Linear(resnet.fc.in_features, embedding_dim)
 
    def forward_once(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x
 
# -------------------------------
# Paths (relative to this file)
# -------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, r"D:\Code\BU-ELEPHENT-HACKATHON\BU-ELEPHANT-HACKATHON-August\ewe\model")
TNI_WEIGHTS_PATH = os.path.join(MODEL_DIR, r"D:\Code\BU-ELEPHENT-HACKATHON\BU-ELEPHANT-HACKATHON-August\ewe\model\modeltni_elephant_model_best.pth")
YOLO_WEIGHTS_PATH = os.path.join(MODEL_DIR, r"D:\Code\BU-ELEPHENT-HACKATHON\BU-ELEPHANT-HACKATHON-August\ewe\model\yolov8n.pt")
REF_EMB_PATH = os.path.join(MODEL_DIR, r"D:\Code\BU-ELEPHENT-HACKATHON\BU-ELEPHANT-HACKATHON-August\ewe\model\reference_embeddings_best.pt")
 
# -------------------------------
# Load TNI Model
# -------------------------------
tni_model = TNIModel(embedding_dim=64).to(device)
# Some PyTorch versions support weights_only=True; fallback if not available
try:
    state = torch.load(TNI_WEIGHTS_PATH, map_location=device, weights_only=True)
except TypeError:
    state = torch.load(TNI_WEIGHTS_PATH, map_location=device)
tni_model.load_state_dict(state)
tni_model.eval()
 
# -------------------------------
# Load YOLO Model
# -------------------------------
yolo_model = YOLO(YOLO_WEIGHTS_PATH)
 
# -------------------------------
# Image transform for identification
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
def identify_elephant(cropped_frame, reference_embeddings, model, device):
    """Identify elephant from a cropped frame (numpy BGR array). Returns (identity, similarity)."""
    img_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
 
    with torch.no_grad():
        emb = model.forward_once(img_tensor)
 
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
 
# ============================================================
# Web helper (single-image inference) — used by FastAPI server
# ============================================================
def _identify_from_ndarray_bgr(crop_bgr, model, refs, device, transform):
    """Internal helper: identify from a BGR crop using the same TNI pipeline."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None, -1.0
    crop_rgb = crop_bgr[:, :, ::-1]
    pil_img = Image.fromarray(crop_rgb)
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.forward_once(x)
    best_name, best_sim = None, -1.0
    for name, ref_emb in refs.items():
        ref = ref_emb.to(device)
        if ref.dim() == 1:
            ref = ref.unsqueeze(0)
        sim = F.cosine_similarity(emb, ref, dim=1).item()
        if sim > best_sim:
            best_sim, best_name = sim, name
    return best_name, best_sim
 
def predict_on_frame(frame_bgr,
                     conf_thres: float = 0.7,
                     id_sim_thres: float = 0.98):
    """
    Run YOLO detection + TNI identification on a single BGR frame.
    Returns a list of dicts matching your OpenCV overlay semantics:
      {
        "bbox": [x1, y1, x2, y2],
        "yolo_label": "<class>",
        "confidence": 0.87,
        "identity": "Plai Khammoon" | "Unknown",
        "similarity": 0.99,
        "overlay_text": "Elephant: Plai Khammoon (87.0%) | Sim: 0.99"
      }
    """
    import numpy as np
    assert isinstance(frame_bgr, np.ndarray) and frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3, \
        "predict_on_frame expects an OpenCV BGR image (HxWx3)."
 
    results = yolo_model(frame_bgr)[0]
    outs = []
 
    for box in results.boxes:
        conf = float(box.conf)
        if conf < conf_thres:
            continue
 
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        if cls_id != 20: continue
        yolo_lbl = yolo_model.names[cls_id] if hasattr(yolo_model, "names") else str(cls_id)
 
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            continue
 
        name, sim = _identify_from_ndarray_bgr(
            crop, tni_model, reference_embeddings, device, transform
        )
        final_name = name if (sim is not None and sim >= id_sim_thres) else "Unknown"
        overlay_text = f"Elephant: {final_name} ({conf*100:.1f}%) | Sim: {sim:.2f}"
 
        outs.append({
            "bbox": [x1, y1, x2, y2],
            "yolo_label": yolo_lbl,
            "confidence": conf,
            "identity": final_name,
            "similarity": float(sim if sim is not None else 0.0),
            "overlay_text": overlay_text
        })
 
    return outs
 
# ============================================================
# Desktop video loop — only runs when executing this file directly
# ============================================================
if __name__ == "__main__":
    # -------------------------------
    # Main Video Processing Loop
    # -------------------------------
    # Change VIDEO_PATH to your file if needed:
    VIDEO_PATH = r"D:\CODING\BU-ELEPHANT-HACKATHON\Videos\รัศมี.MP4"  # or set to another path
 
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
    CONFIDENCE_THRESHOLD = 0.7
    IDENTIFICATION_THRESHOLD = 0.98  # Minimum similarity for identification
 
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
                cropped_elephant, reference_embeddings, tni_model, device
            )
 
            if similarity > IDENTIFICATION_THRESHOLD:
                color = (0, 255, 0)  # Green
                text = f"{identity} ({similarity:.2f})"
            else:
                color = (0, 165, 255)  # Orange
                text = f"Unknown Elephant ({conf:.2f})"
 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
 
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
 
        cv2.imshow("Elephant Detection & Identification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Video processing completed")