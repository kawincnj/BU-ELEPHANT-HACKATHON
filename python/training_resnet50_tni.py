# -------------------------------
# Elephant Identification with TNI
# -------------------------------

import os
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Step 1: Image Transforms
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize all images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

# -------------------------------
# Step 2: Custom Triplet Dataset
# -------------------------------
class TripletElephantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)  # list of elephant folders
        self.images_by_class = {
            c: [os.path.join(root_dir, c, img) for img in os.listdir(os.path.join(root_dir, c))]
            for c in self.classes
        }

    def __len__(self):
        return sum(len(imgs) for imgs in self.images_by_class.values())

    def __getitem__(self, idx):
        # Randomly pick anchor class
        anchor_class = random.choice(self.classes)
        negative_class = random.choice([c for c in self.classes if c != anchor_class])

        # Pick two different images from anchor_class
        anchor_img, positive_img = random.sample(self.images_by_class[anchor_class], 2)
        negative_img = random.choice(self.images_by_class[negative_class])

        # Load images
        anchor = Image.open(anchor_img).convert("RGB")
        positive = Image.open(positive_img).convert("RGB")
        negative = Image.open(negative_img).convert("RGB")

        # Apply transform
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

# -------------------------------
# Step 3: Triplet Network Model
# -------------------------------
class TNIModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(TNIModel, self).__init__()
        # Pretrained ResNet-50 backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # remove final FC

        # Embedding layer
        self.embedding = nn.Linear(resnet.fc.in_features, embedding_dim)

    def forward_once(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)  # normalize embeddings
        return x

    def forward(self, anchor, positive, negative):
        anchor_out = self.forward_once(anchor)
        pos_out = self.forward_once(positive)
        neg_out = self.forward_once(negative)
        return anchor_out, pos_out, neg_out

# -------------------------------
# Step 4: Prepare Dataset and Dataloader
# -------------------------------
dataset = TripletElephantDataset(
    root_dir="D:/CODING/BU-ELEPHANT-HACKATHON/Elephant_Image",
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# -------------------------------
# Step 5: Initialize Model, Loss, Optimizer
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TNIModel(embedding_dim=128).to(device)
criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -------------------------------
# Step 6: Training Loop
# -------------------------------
epochs = 40

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for anchor, positive, negative in dataloader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()
        a_out, p_out, n_out = model(anchor, positive, negative)
        loss = criterion(a_out, p_out, n_out)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss/len(dataloader):.4f}")

# -------------------------------
# Step 7: Save the Trained Model
# -------------------------------
torch.save(model.state_dict(), r"D:\CODING\BU-ELEPHANT-HACKATHON\modeltni_elephant_model.pth")
print("Model saved as tni_elephant_model.pth")
