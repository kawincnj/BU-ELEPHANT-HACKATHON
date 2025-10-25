import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import os
from collections import OrderedDict

# -------------------------------
# Step 1: Define the TNIModel class (Must be identical to the training script)
# -------------------------------
class TNIModel(nn.Module):
    def __init__(self, embedding_dim=64, dropout_rate=0.3):
        super(TNIModel, self).__init__()
        # Pretrained ResNet-50 backbone
        # NOTE: Using 'default' weights for compatibility if specific version is missing
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Dropout Layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Embedding layer
        self.embedding = nn.Linear(resnet.fc.in_features, embedding_dim)

    def forward_once(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)

        # Dropout is usually disabled during inference, but we keep it here
        # for tracing. The 'eval()' mode (set later) will disable it automatically.
        x = self.dropout(x)

        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)  # normalize embeddings
        return x

    def forward(self, x):
        # NOTE: For ONNX export, we only trace the 'forward_once' path, 
        # as the Triplet Loss logic is only for training and not part of the deployed model.
        # The deployed model will take one image and output its embedding.
        return self.forward_once(x)


# -------------------------------
# Step 2: Load Model and Weights
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 64
DROPOUT_RATE = 0.3
MODEL_PATH = r"models\modeltni_elephant_model_best.pth" # หรือใช้ 'modeltni_elephant_model_latest.pth'

# Initialize model
model = TNIModel(embedding_dim=EMBEDDING_DIM, dropout_rate=DROPOUT_RATE).to(DEVICE)

# Load state dict
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Optional: Fix keys if 'module.' prefix exists from DataParallel training
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.'
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    print(f"✅ Successfully loaded weights from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {MODEL_PATH}. Please ensure the training completed and the file exists.")
    exit()
except Exception as e:
    print(f"❌ Error loading state dict: {e}")
    exit()

# Set model to evaluation mode (crucial for disabling dropout/batchnorm updates)
model.eval()


# -------------------------------
# Step 3: Export to ONNX
# -------------------------------

# Define dummy input (Batch size 1, 3 color channels, 224x224 resolution)
# This size must match the transforms used during training.
DUMMY_INPUT = torch.randn(1, 3, 224, 224, device=DEVICE)
ONNX_MODEL_PATH = "tni_elephant_model.onnx"
OPSET_VERSION = 17 # Use a recent, stable opset

# The list of input/output names is important for deployment in ONNX Runtime
input_names = ["input_image"]
output_names = ["embedding_vector"]

# Export the model
try:
    torch.onnx.export(
        model,
        DUMMY_INPUT,
        ONNX_MODEL_PATH,
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={ # Allows for variable batch size during inference
            'input_image': {0: 'batch_size'},
            'embedding_vector': {0: 'batch_size'}
        }
    )
    print("\n" + "="*70)
    print(f"🎉 Conversion to ONNX successful! Model saved as: {ONNX_MODEL_PATH}")
    print(f"Model Input: {input_names[0]} (Shape: [batch_size, 3, 224, 224])")
    print(f"Model Output: {output_names[0]} (Shape: [batch_size, {EMBEDDING_DIM}])")
    print("="*70)

except Exception as e:
    print(f"❌ ONNX Export Error: {e}")