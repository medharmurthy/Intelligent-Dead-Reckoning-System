import os
import sys
import torch
import torch.onnx

# Ensure src root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.tiny_tcn import TinyTCN

def export_to_onnx(weights_path="models/tiny_tcn_best.pth", output_onnx_path="models/idr_tcn.onnx"):
    """
    Exports PyTorch Tiny TCN model to ONNX format (Sections 41-42 of plan.pdf)
    Inputs:
      - x: shape (batch_size, 9, 20) -> 9 sensor channels over 20 time steps (2 seconds @ 10Hz)
    Outputs:
      - pred_speed: forward speed (m/s)
      - log_var: log variance log(sigma_v^2) for EKF measurement covariance
      - motion_logits: 5 motion class probabilities
    """
    print("--- Exporting Tiny TCN Model to ONNX ---")
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    
    model = TinyTCN(in_channels=9, num_motion_classes=5)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: Weights path {weights_path} not found. Exporting initialized model.")
        
    model.eval()
    
    # Dummy input: 1 sample, 9 channels, 20 time frames
    dummy_input = torch.randn(1, 9, 20, dtype=torch.float32)
    
    dynamic_axes = {
        "imu_window": {0: "batch_size"},
        "speed": {0: "batch_size"},
        "log_var": {0: "batch_size"},
        "motion_logits": {0: "batch_size"}
    }
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["imu_window"],
            output_names=["speed", "log_var", "motion_logits"],
            dynamic_axes=dynamic_axes,
            dynamo=False
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["imu_window"],
            output_names=["speed", "log_var", "motion_logits"],
            dynamic_axes=dynamic_axes
        )
    
    file_size_kb = os.path.getsize(output_onnx_path) / 1024
    print(f"Successfully exported ONNX model to: {output_onnx_path}")
    print(f"Model File Size: {file_size_kb:.2f} KB (< 1-2 MB target requirement as per Section 41)")

if __name__ == "__main__":
    export_to_onnx()
