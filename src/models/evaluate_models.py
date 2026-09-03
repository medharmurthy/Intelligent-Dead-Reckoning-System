import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tiny_tcn import TinyTCN

def evaluate_and_plot():
    print("--- Evaluating Models and Generating Benchmark Plot ---")
    
    val_data = np.load("data/processed/val_windows.npz")
    X_val_raw, y_val = val_data["X"], val_data["y"]
    
    # Format for PyTorch Conv1D: (N, C=9, T=20)
    X_val = np.transpose(X_val_raw, (0, 2, 1))
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    
    # Load Tiny TCN
    tcn_model = TinyTCN(in_channels=9)
    tcn_model.load_state_dict(torch.load("models/tiny_tcn.pth"))
    tcn_model.eval()
    
    with torch.no_grad():
        pred_speed, log_var = tcn_model(X_val_tensor)
        pred_speed_mps = pred_speed.numpy()
        pred_speed_kmh = pred_speed_mps * 3.6
        sigma_v = np.sqrt(np.exp(log_var.numpy())) * 3.6  # Uncertainty standard deviation in km/h
        
    gt_speed_kmh = y_val * 3.6
    time_steps = np.arange(len(gt_speed_kmh)) * 0.5  # 0.5s step size
    
    # Compute performance metrics
    mae_kmh = np.mean(np.abs(pred_speed_kmh - gt_speed_kmh))
    rmse_kmh = np.sqrt(np.mean((pred_speed_kmh - gt_speed_kmh)**2))
    
    print(f"\n[Validation Benchmark Results - Tiny TCN]")
    print(f"Mean Absolute Error (MAE): {mae_kmh:.2f} km/h ({mae_kmh / 3.6:.2f} m/s)")
    print(f"Root Mean Sq Error (RMSE): {rmse_kmh:.2f} km/h ({rmse_kmh / 3.6:.2f} m/s)")
    
    # Plot first 300 validation steps (~2.5 minutes)
    n_plot = min(300, len(gt_speed_kmh))
    plt.figure(figsize=(12, 5))
    plt.plot(time_steps[:n_plot], gt_speed_kmh[:n_plot], label="Vehicle Ground-Truth Speed (CAN-bus)", color="black", linewidth=2.0)
    plt.plot(time_steps[:n_plot], pred_speed_kmh[:n_plot], label="AI Predicted Speed (Tiny TCN from Phone IMU)", color="crimson", linestyle="--", linewidth=1.8)
    
    plt.fill_between(
        time_steps[:n_plot],
        pred_speed_kmh[:n_plot] - sigma_v[:n_plot],
        pred_speed_kmh[:n_plot] + sigma_v[:n_plot],
        color="crimson", alpha=0.2, label="AI Uncertainty (±1σ)"
    )
    
    plt.title(f"Intelligent Dead Reckoning: Smartphone IMU Speed Prediction\nValidation MAE: {mae_kmh:.2f} km/h", fontsize=13, fontweight="bold")
    plt.xlabel("Time (seconds)", fontsize=11)
    plt.ylabel("Vehicle Speed (km/h)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper right")
    
    os.makedirs("data/processed", exist_ok=True)
    plot_path = "data/processed/speed_prediction_benchmark.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Saved benchmark plot to {plot_path}")

if __name__ == "__main__":
    evaluate_and_plot()
