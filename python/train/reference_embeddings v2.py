# -------------------------------
# Elephant Identification with TNI
# (Optimized Code)
# -------------------------------
 
import os
import random
from PIL import Image
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
 
# -------------------------------
# Step 1: Image Transforms (with Data Augmentation for Regularization)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
   
    # 🌟 Data Augmentation to prevent Overfitting 🌟
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # 🌟
   
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
 
# -------------------------------
# Step 2: Custom Triplet Dataset
# -------------------------------
class TripletElephantDataset(Dataset):
    """
    Creates random triplets (Anchor, Positive, Negative) on-the-fly.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
       
        # กรองเพื่อให้แน่ใจว่าเป็นไดเรกทอรีและมีไฟล์ภาพที่เพียงพอ
        self.classes = [c for c in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, c))]
       
        self.images_by_class = {}
        for c in self.classes:
            img_paths = [os.path.join(root_dir, c, img) for img in os.listdir(os.path.join(root_dir, c))]
            # ต้องมีอย่างน้อย 2 ภาพต่อคลาสเพื่อสร้าง (Anchor, Positive)
            if len(img_paths) >= 2:
                self.images_by_class[c] = img_paths
           
        # อัปเดต classes ให้มีเฉพาะคลาสที่มีภาพเพียงพอ
        self.classes = list(self.images_by_class.keys())
 
        if len(self.classes) < 2:
            raise ValueError("Dataset requires at least 2 classes, and each class must have at least 2 images.")
 
    def __len__(self):
        # ความยาวของ Dataset ถูกกำหนดโดยจำนวนตัวอย่างที่เป็นไปได้
        # แต่เนื่องจากเราสุ่ม Triplet เราจะใช้ขนาดรวมทั้งหมดของภาพ
        return sum(len(imgs) for imgs in self.images_by_class.values())
 
    def __getitem__(self, idx):
        # Randomly pick anchor class
        anchor_class = random.choice(self.classes)
       
        # Randomly pick negative class (must be different)
        negative_class = random.choice([c for c in self.classes if c != anchor_class])
 
        # Pick two different images from anchor_class
        anchor_img, positive_img = random.sample(self.images_by_class[anchor_class], 2)
       
        # Pick one image from negative_class
        negative_img = random.choice(self.images_by_class[negative_class])
 
        # Load images
        try:
            anchor = Image.open(anchor_img).convert("RGB")
            positive = Image.open(positive_img).convert("RGB")
            negative = Image.open(negative_img).convert("RGB")
        except Exception as e:
            # ควรจัดการข้อผิดพลาดในการโหลดไฟล์
            print(f"Error loading image: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1)) # สุ่มตัวอย่างใหม่
 
        # Apply transform
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
 
        return anchor, positive, negative
 
# -------------------------------
# Step 3: Triplet Network Model (with Dropout for Regularization)
# -------------------------------
class TNIModel(nn.Module):
    def __init__(self, embedding_dim=64, dropout_rate=0.3):
        super(TNIModel, self).__init__()
        # Pretrained ResNet-50 backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove final FC layer (Average Pooling is the last layer in backbone)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
 
        # 🌟 Dropout Layer 🌟
        self.dropout = nn.Dropout(p=dropout_rate)
 
        # Embedding layer
        self.embedding = nn.Linear(resnet.fc.in_features, embedding_dim)
 
    def forward_once(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
       
        # 🌟 Apply Dropout 🌟
        x = self.dropout(x)
 
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)  # normalize embeddings
        return x
 
    def forward(self, anchor, positive, negative):
        anchor_out = self.forward_once(anchor)
        pos_out = self.forward_once(positive)
        neg_out = self.forward_once(negative)
        return anchor_out, pos_out, neg_out
 
 
# -------------------------------
# Step 4: Prepare Dataset and Dataloader (Train/Test Split)
# -------------------------------
full_dataset = TripletElephantDataset(
    root_dir=r"D:\BU-ELEPHANT-HACKATHON-main\dataset", # ใช้ r"" เพื่อแก้ SyntaxWarning
    transform=transform
)
 
# กำหนดสัดส่วนการแบ่งข้อมูล (เช่น 80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
 
# แบ่ง Dataset
train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
 
# สร้าง DataLoader สำหรับ Training และ Testing
dataloader_train = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4, # เพิ่มประสิทธิภาพการโหลดข้อมูล
    pin_memory=True
)
 
dataloader_test = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
 
# -------------------------------
# Step 5: Evaluation Function
# -------------------------------
def evaluate_triplet_accuracy(model, dataloader, device, margin):
    """Calculates the percentage of triplets that satisfy the margin condition."""
    model.eval()
    correct_triplets = 0
    total_triplets = 0
   
    with torch.no_grad():
        for anchor, positive, negative in dataloader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
 
            a_out, p_out, n_out = model(anchor, positive, negative)
 
            # L2 distance (p=2)
            dist_ap = F.pairwise_distance(a_out, p_out, p=2)
            dist_an = F.pairwise_distance(a_out, n_out, p=2)
           
            # Triplet satisfied condition: d(a, p) + margin < d(a, n)
            is_satisfied = (dist_an > dist_ap + margin).sum().item()
           
            correct_triplets += is_satisfied
            total_triplets += anchor.size(0)
   
    accuracy = correct_triplets / total_triplets if total_triplets > 0 else 0
    return accuracy
 
 
# -------------------------------
# Main Execution Block (Fixes RuntimeError on Windows)
# -------------------------------
 
best_loss = 1
if __name__ == '__main__':
   
    # -------------------------------
    # Step 6: Initialize Model, Loss, Optimizer
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
   
    model = TNIModel(embedding_dim=64, dropout_rate=0.3).to(device)
    criterion = nn.TripletMarginLoss(margin=1.0)
   
    # 🌟 Optimizer with Weight Decay (L2 Regularization) 🌟
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
 
    # -------------------------------
    # Step 7: Training Loop
    # -------------------------------
    epochs = 40
    margin = 1.0
 
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct_train_triplets = 0
        total_train_triplets = 0
       
        for anchor, positive, negative in dataloader_train:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
 
            optimizer.zero_grad()
            a_out, p_out, n_out = model(anchor, positive, negative)
            loss = criterion(a_out, p_out, n_out)
            loss.backward()
            optimizer.step()
 
            epoch_loss += loss.item()
 
            # Calculate Training Triplet Accuracy
            dist_ap = F.pairwise_distance(a_out, p_out, p=2)
            dist_an = F.pairwise_distance(a_out, n_out, p=2)
            is_satisfied = (dist_an > dist_ap + margin).sum().item()
            correct_train_triplets += is_satisfied
            total_train_triplets += anchor.size(0)
           
        train_accuracy = correct_train_triplets / total_train_triplets if total_train_triplets > 0 else 0
        avg_loss = epoch_loss/len(dataloader_train)
       
        # Run evaluation on test set after each epoch
        test_accuracy = evaluate_triplet_accuracy(model, dataloader_test, device, margin=1.0)
 
        if avg_loss < best_loss:
            torch.save(model.state_dict(), r"modeltni_elephant_model_best.pth")
            print("SAVE BEST!")
            best_loss = avg_loss
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")
       
    # -------------------------------
    # Step 8: Save the Trained Model
    # -------------------------------
    final_test_accuracy = evaluate_triplet_accuracy(model, dataloader_test, device, margin=1.0)
    print("\n" + "="*70)
    print(f"Training Complete. Final Test Triplet Accuracy: {final_test_accuracy:.4f}")
    print("="*70)
 
    # ใช้ r"" เพื่อแก้ SyntaxWarning
    torch.save(model.state_dict(), r"D:\CODING\BU-ELEPHANT-HACKATHON\modeltni_elephant_model_latest.pth")
    print("Model saved as modeltni_elephant_model299xd.pth")