import torch
from PIL import Image
import os
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# -------------------------------
# Define the TNIModel (same as training)
# -------------------------------
class TNIModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(TNIModel, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # remove final FC
        self.embedding = nn.Linear(resnet.fc.in_features, embedding_dim)

    def forward_once(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, anchor, positive, negative):
        anchor_out = self.forward_once(anchor)
        pos_out = self.forward_once(positive)
        neg_out = self.forward_once(negative)
        return anchor_out, pos_out, neg_out

# -------------------------------
# Set device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Instantiate and load the model
# -------------------------------
model = TNIModel(embedding_dim=128).to(device)
model.load_state_dict(torch.load(r"model\modeltni_elephant_model.pth"))
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
# Create reference embeddings
# -------------------------------
reference_embeddings = {}
data_dir = r"D:\CODING\BU-ELEPHANT-HACKATHON\Elephant_Image"

for elephant_name in os.listdir(data_dir):
    imgs = os.listdir(os.path.join(data_dir, elephant_name))
    embeddings = []
    for img_name in imgs:
        img_path = os.path.join(data_dir, elephant_name, img_name)
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.forward_once(img)
            embeddings.append(emb)
    reference_embeddings[elephant_name] = torch.mean(torch.stack(embeddings), dim=0)

print("Reference embeddings created for all elephants.")

torch.save(reference_embeddings, r"D:\CODING\BU-ELEPHANT-HACKATHON\reference_embeddings.pt")
print("Reference embeddings saved!")
