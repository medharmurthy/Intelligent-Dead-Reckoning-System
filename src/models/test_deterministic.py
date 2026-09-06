import os
import sys
import json
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.tiny_tcn import TinyTCN

def run_test():
    # We will just synthesize a known input
    # Shape: 20 frames, 9 channels. We'll use a simple deterministic sequence
    np.random.seed(42)
    raw_window = np.random.randn(20, 9).astype(np.float32)
    
    # Preprocess
    norm_path = "data/processed/normalization.json"
    with open(norm_path, "r") as f:
        norm_params = json.load(f)
        
    mean = np.array(norm_params["mean"], dtype=np.float32)
    std = np.array(norm_params["std"], dtype=np.float32)
    std[std == 0] = 1e-6
    
    norm_window = (raw_window - mean) / std
    
    # Transpose to [1, 9, 20]
    tensor_in = torch.tensor(np.transpose(norm_window, (1, 0)), dtype=torch.float32).unsqueeze(0)
    
    model = TinyTCN(in_channels=9, num_motion_classes=5)
    model.load_state_dict(torch.load("models/tiny_tcn_best.pth", weights_only=True))
    model.eval()
    
    with torch.no_grad():
        pred_speed, log_var, motion_logits = model(tensor_in)
        
    out = {
        "raw_window": raw_window.tolist(),
        "expected_tensor": tensor_in.numpy().flatten().tolist(),
        "expected_speed": float(pred_speed[0].item()),
        "expected_log_var": float(log_var[0].item()),
        "expected_motion": motion_logits[0].numpy().tolist()
    }
    
    with open("frontend/public/data/test_data.json", "w") as f:
        json.dump(out, f, indent=2)
        
    print("Generated deterministic test data at frontend/public/data/test_data.json")

if __name__ == "__main__":
    run_test()
