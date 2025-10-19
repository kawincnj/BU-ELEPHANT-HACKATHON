import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')

# ------------------------------- python.exe -m pip install --upgrade pip
# Set device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# TNI Model Definition
# -------------------------------
class TNIModel(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super(TNIModel, self).__init__()
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
# Load TNI Model
# -------------------------------
tni_model = TNIModel(embedding_dim=128).to(device)
tni_model.load_state_dict(torch.load(r"model\modeltni_elephant_model.pth", 
                                      map_location=device, 
                                      weights_only=True))
tni_model.eval()

# -------------------------------
# Load YOLO Model
# -------------------------------
yolo_model = YOLO(r"model\elephant_obj_detect.pt")

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
reference_embeddings = torch.load(r"model\reference_embeddings.pt", 
                                   map_location=device, 
                                   weights_only=True)

# -------------------------------
# Function to identify elephant from cropped image
# -------------------------------
def identify_elephant(cropped_frame, reference_embeddings, model, device):
    """Identify elephant from a cropped frame (numpy array)"""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Transform and prepare for model
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb = model.forward_once(img_tensor)
    
    max_sim = -1
    identity = None
    for name, ref_emb in reference_embeddings.items():
        ref_emb = ref_emb.to(device)
        # Ensure ref_emb has the same shape as emb
        if ref_emb.dim() == 1:
            ref_emb = ref_emb.unsqueeze(0)
        
        # Compute cosine similarity
        sim = F.cosine_similarity(emb, ref_emb, dim=1)
        if sim.item() > max_sim:
            max_sim = sim.item()
            identity = name
    
    return identity, max_sim

# -------------------------------
# Main Video Processing Loop
# -------------------------------
cap = cv2.VideoCapture(r"D:\CODING\BU-ELEPHANT-HACKATHON\Videos\รัศมี.MP4")
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

    # Run YOLO detection
    results = yolo_model(frame)[0]

    for box in results.boxes:
        conf = float(box.conf)

        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]
        
        # Crop the detected elephant
        cropped_elephant = frame[y1:y2, x1:x2]
        
        # Skip if crop is too small
        if cropped_elephant.shape[0] < 50 or cropped_elephant.shape[1] < 50:
            continue
        
        # Identify the elephant
        identity, similarity = identify_elephant(cropped_elephant, reference_embeddings, 
                                                 tni_model, device)
        
        # Choose color based on identification confidence
        if similarity > IDENTIFICATION_THRESHOLD:
            color = (0, 255, 0)  # Green for confident identification
            text = f"{identity} ({similarity:.2f})"
        else:
            color = (0, 165, 255)  # Orange for uncertain
            text = f"Unknown Elephant ({conf:.2f})"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Display the frame
    cv2.imshow("Elephant Detection & Identification", frame)

    # Quit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Video processing completed")