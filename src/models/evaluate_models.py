import os
import sys
import numpy as np
import torch

# Ensure sys.path includes src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.tiny_tcn import TinyTCN

def evaluate_and_plot():
    print("--- Evaluating Tiny TCN Model on Held-Out Test Set ---")
    
    test_path = "data/processed/test_windows.npz"
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test set not found at {test_path}. Run create_windows.py first!")
        
    test_data = np.load(test_path)
    X_test_raw, y_test = test_data["X"], test_data["y"]
    
    # Format for PyTorch Conv1D: (N, C=9, T=20)
    X_test = np.transpose(X_test_raw, (0, 2, 1))
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Load Tiny TCN
    tcn_model = TinyTCN(in_channels=9, num_motion_classes=5)
    weights_path = "models/tiny_tcn_best.pth"
    if os.path.exists(weights_path):
        tcn_model.load_state_dict(torch.load(weights_path, weights_only=True))
        print(f"Loaded model weights from {weights_path}")
    else:
        print(f"Warning: {weights_path} not found. Running initialized model.")
    tcn_model.eval()
    
    with torch.no_grad():
        pred_speed, log_var, motion_logits = tcn_model(X_test_tensor)
        pred_speed_mps = pred_speed.numpy()
        pred_speed_kmh = pred_speed_mps * 3.6
        sigma_v = np.sqrt(np.exp(log_var.numpy())) * 3.6  # Uncertainty standard deviation in km/h
        
    gt_speed_kmh = y_test * 3.6
    
    # Compute performance metrics
    mae_kmh = np.mean(np.abs(pred_speed_kmh - gt_speed_kmh))
    rmse_kmh = np.sqrt(np.mean((pred_speed_kmh - gt_speed_kmh)**2))
    
    print(f"\n[Test Set Benchmark Results - Tiny TCN]")
    print(f"Total Test Windows:        {len(y_test)}")
    print(f"Mean Absolute Error (MAE): {mae_kmh:.2f} km/h ({mae_kmh / 3.6:.2f} m/s)")
    print(f"Root Mean Sq Error (RMSE): {rmse_kmh:.2f} km/h ({rmse_kmh / 3.6:.2f} m/s)")

if __name__ == "__main__":
    evaluate_and_plot()
