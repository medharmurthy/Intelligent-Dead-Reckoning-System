import os
import sys
import numpy as np
import json

# Add src directory to path to import parse_trip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.preprocessing.parse_trip import load_trip_s1

def extract_window_features(phone_data, vehicle_data, window_size=20, step_size=5):
    """
    Slices continuous IMU sequence into 2-second rolling windows (20 frames at 10Hz).
    
    Feature matrix X per window (20 frames x 9 channels):
      Ch 0: ax
      Ch 1: ay
      Ch 2: az
      Ch 3: gx
      Ch 4: gy
      Ch 5: gz
      Ch 6: accel_magnitude |a|
      Ch 7: gyro_magnitude |w|
      Ch 8: jerk = d|a|/dt
      
    Target y per window:
      v_f: Ground truth forward velocity (m/s) at the window endpoint.
    """
    n_frames = min(len(phone_data), len(vehicle_data))
    
    # Pre-extract arrays for speed
    ax = np.array([p['ax'] for p in phone_data[:n_frames]], dtype=np.float32)
    ay = np.array([p['ay'] for p in phone_data[:n_frames]], dtype=np.float32)
    az = np.array([p['az'] for p in phone_data[:n_frames]], dtype=np.float32)
    
    gx = np.array([p['gx'] for p in phone_data[:n_frames]], dtype=np.float32)
    gy = np.array([p['gy'] for p in phone_data[:n_frames]], dtype=np.float32)
    gz = np.array([p['gz'] for p in phone_data[:n_frames]], dtype=np.float32)
    
    accel_mag = np.array([p['accel_mag'] for p in phone_data[:n_frames]], dtype=np.float32)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2).astype(np.float32)
    
    # Calculate jerk = numerical derivative of accel magnitude
    jerk = np.zeros_like(accel_mag)
    jerk[1:] = (accel_mag[1:] - accel_mag[:-1]) / 0.1  # dt = 0.1s at 10Hz
    
    # Stack 9 channels into array of shape (n_frames, 9)
    raw_channels = np.column_stack([ax, ay, az, gx, gy, gz, accel_mag, gyro_mag, jerk])
    
    # Target speeds in m/s
    target_speeds = np.array([v['speed_mps'] for v in vehicle_data[:n_frames]], dtype=np.float32)
    
    # Sliding windows
    X_windows = []
    y_targets = []
    
    for start in range(0, n_frames - window_size + 1, step_size):
        end = start + window_size
        window_x = raw_channels[start:end, :]  # Shape: (20, 9)
        target_y = target_speeds[end - 1]      # Speed at current moment
        
        X_windows.append(window_x)
        y_targets.append(target_y)
        
    X_windows = np.array(X_windows, dtype=np.float32)  # Shape: (N, 20, 9)
    y_targets = np.array(y_targets, dtype=np.float32)  # Shape: (N,)
    
    return X_windows, y_targets

def build_processed_dataset(output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load raw parsed data
    phone_data, vehicle_data = load_trip_s1()
    
    # 2. Extract 2-second windows (20 frames @ 10Hz, step size 5 = 50% overlap)
    print("\nExtracting rolling 2-second IMU windows...")
    X, y = extract_window_features(phone_data, vehicle_data, window_size=20, step_size=5)
    print(f"Generated {len(X)} total windows. Input Shape: {X.shape}, Target Shape: {y.shape}")
    
    # 3. Train / Validation Split (Sequential trip split: first 80% train, last 20% val)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    # 4. Compute Channel Normalization Statistics (Mean & Std from TRAIN set only)
    mean = np.mean(X_train, axis=(0, 1))  # 9 values
    std = np.std(X_train, axis=(0, 1))    # 9 values
    std[std == 0] = 1e-6                  # Avoid division by zero
    
    # Normalize features: (X - mean) / std
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    
    # 5. Save Normalization Parameters for model deployment/inference
    norm_params = {
        "channel_names": ["ax", "ay", "az", "gx", "gy", "gz", "accel_mag", "gyro_mag", "jerk"],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "window_size": 20,
        "sampling_rate_hz": 10
    }
    norm_path = os.path.join(output_dir, "normalization.json")
    with open(norm_path, "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"Saved normalization parameters to {norm_path}")
    
    # 6. Save Processed Datasets
    train_path = os.path.join(output_dir, "train_windows.npz")
    val_path = os.path.join(output_dir, "val_windows.npz")
    
    np.savez_compressed(train_path, X=X_train_norm, y=y_train)
    np.savez_compressed(val_path, X=X_val_norm, y=y_val)
    
    print(f"Saved Train set ({len(X_train)} windows) to {train_path}")
    print(f"Saved Val set   ({len(X_val)} windows) to {val_path}")
    print("\nDataset preparation completed successfully!")

if __name__ == "__main__":
    build_processed_dataset()
