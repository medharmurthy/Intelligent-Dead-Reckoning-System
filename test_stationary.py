import onnxruntime as ort
import numpy as np
import json

with open('data/processed/normalization.json', 'r') as f:
    norm = json.load(f)
mean = np.array(norm['mean'], dtype=np.float32)
std = np.array(norm['std'], dtype=np.float32)

sess = ort.InferenceSession('frontend/public/models/idr_tcn.onnx')

def predict_speed(ax, ay, az, gx, gy, gz):
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    jerk = 0.0
    
    frame = np.array([ax, ay, az, gx, gy, gz, accel_mag, gyro_mag, jerk], dtype=np.float32)
    window = np.tile(frame, (20, 1))
    window_norm = (window - mean) / std
    
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    
    if input_shape[1] == 9:
        window_input = window_norm.T
    else:
        window_input = window_norm
        
    window_input = np.expand_dims(window_input, axis=0)
    out = sess.run(None, {input_name: window_input})
    speed_mps = np.array(out[0]).flatten()[0]
    return speed_mps * 3.6

print("Testing Stationary Orientations:")
print(f"1. Perfectly Flat (Z=9.8): {predict_speed(0, 0, 9.8, 0, 0, 0):.2f} km/h")
print(f"2. Tilted Forward 45 deg: {predict_speed(0, 6.93, 6.93, 0, 0, 0):.2f} km/h")
print(f"3. Held Upright (Y=9.8): {predict_speed(0, 9.8, 0, 0, 0, 0):.2f} km/h")
print(f"4. Held Sideways (X=9.8): {predict_speed(9.8, 0, 0, 0, 0, 0):.2f} km/h")
print(f"5. Exact Training Dataset Means: {predict_speed(mean[0], mean[1], mean[2], mean[3], mean[4], mean[5]):.2f} km/h")