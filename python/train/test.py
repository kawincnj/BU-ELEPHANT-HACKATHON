# Simple_Loader.py
import torch

try:
    # Set the path to the potentially corrupted file
    file_path = r"models\modeltni_elephant_model_best.pth" 
    
    # Try to load the file
    loaded_data = torch.load(file_path, map_location='cpu')
    
    print(f"Successfully loaded data. Type: {type(loaded_data)}")
    if isinstance(loaded_data, dict):
        print(f"Keys in the state dict: {loaded_data.keys()}")
        
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    print("The file is corrupted or not a valid PyTorch/Python object.")