import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# -------------------------------
# Set device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load model (same TNI model class as training)
# -------------------------------
class TNIModel(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super(TNIModel, self).__init__()
        resnet = torch.hub.load('pytorch/vision:v0.15.2', 'resnet50', pretrained=True)
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = torch.nn.Linear(resnet.fc.in_features, embedding_dim)

    def forward_once(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# Initialize model
model = TNIModel(embedding_dim=128).to(device)
model.load_state_dict(torch.load(r"model\modeltni_elephant_model.pth", map_location=device))
model.eval()

# -------------------------------
# Image transform (same as training)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -------------------------------
# Function to identify a new image
# -------------------------------
def identify_elephant(img_path, reference_embeddings, model, device):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb = model.forward_once(img)
    
    max_sim = -1
    identity = None
    for name, ref_emb in reference_embeddings.items():
        ref_emb = ref_emb.to(device)
        # compute cosine similarity along embedding dimension
        sim = F.cosine_similarity(emb, ref_emb.unsqueeze(0), dim=1)
        if sim.item() > max_sim:
            max_sim = sim.item()
            identity = name
    
    return identity, max_sim


# -------------------------------
# Load reference embeddings
# -------------------------------
reference_embeddings = torch.load(r"model\reference_embeddings.pt", map_location=device)

# -------------------------------
# Example usage
# -------------------------------
new_image_path = r"Elephant_Image\IKO\IKO_0015.jpg"
name, confidence = identify_elephant(new_image_path, reference_embeddings, model, device)
print(f"Identified as: {name} with similarity: {confidence:.4f}")
