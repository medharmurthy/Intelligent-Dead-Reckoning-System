import os
import sys
import numpy as np
import json

# Add src directory to path to import parse_trip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.preprocessing.parse_trip import load_all_featured_trips, load_trip_s1

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
    if n_frames < window_size:
        return np.empty((0, window_size, 9), dtype=np.float32), np.empty((0,), dtype=np.float32)
        
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
        
    X_windows = np.array(X_windows, dtype=np.float32)
    y_targets = np.array(y_targets, dtype=np.float32)
    
    return X_windows, y_targets

def build_processed_dataset(output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load multi-driver trips
    all_trips = load_all_featured_trips()
    if not all_trips:
        print("Warning: Multi-trip loading empty. Falling back to S1 sample.")
        all_trips = [load_trip_s1()]
        
    all_X, all_y = [], []
    for p_data, v_data in all_trips:
        X_t, y_t = extract_window_features(p_data, v_data, window_size=20, step_size=5)
        if len(X_t) > 0:
            all_X.append(X_t)
            all_y.append(y_t)
            
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    print(f"\nExtracted Multi-Driver Dataset: {len(X)} total 2-second IMU windows across {len(all_trips)} trips.")
    print(f"Combined Feature Matrix Shape: {X.shape}, Target Vector Shape: {y.shape}")
    
    # 2. Sequential Train / Validation / Test Split (70% Train, 15% Val, 15% Test)
    train_idx = int(len(X) * 0.70)
    val_idx = int(len(X) * 0.85)
    
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]
    
    # 3. Compute Normalization Statistics (Mean & Std from TRAIN set ONLY as per Section 38 & 39)
    mean = np.mean(X_train, axis=(0, 1))  # 9 values
    std = np.std(X_train, axis=(0, 1))    # 9 values
    std[std == 0] = 1e-6                  # Avoid division by zero
    
    # Normalize features: (X - mean) / std
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std
    
    # 4. Save Normalization Parameters
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
    
    # 5. Save Processed Multi-Trip Datasets
    train_path = os.path.join(output_dir, "train_windows.npz")
    val_path = os.path.join(output_dir, "val_windows.npz")
    test_path = os.path.join(output_dir, "test_windows.npz")
    
    np.savez_compressed(train_path, X=X_train_norm, y=y_train)
    np.savez_compressed(val_path, X=X_val_norm, y=y_val)
    np.savez_compressed(test_path, X=X_test_norm, y=y_test)
    
    print(f"Saved Train set ({len(X_train)} windows) to {train_path}")
    print(f"Saved Val set   ({len(X_val)} windows) to {val_path}")
    print(f"Saved Test set  ({len(X_test)} windows) to {test_path}")
    print("\nMulti-Driver Dataset preparation completed successfully!")

if __name__ == "__main__":
    build_processed_dataset()
