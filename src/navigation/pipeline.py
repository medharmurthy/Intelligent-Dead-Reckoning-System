import os
import sys
import json
import math
import collections
import numpy as np
import torch

# Ensure sys.path includes src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.tiny_tcn import TinyTCN
from src.navigation.ins_ekf import EKF2D
from src.navigation.gnss_quality import GNSSQualityMonitor, NavigationStateMode
from src.navigation.map_matching import PrototypeMapMatcher

class CoreNavigationPipeline:
    """
    Unified Core Navigation Pipeline (Sections 55-56 of plan.pdf)
    
    Data flow:
    SensorFrame -> Buffer (20 frames @ 10Hz) -> Model Inference -> EKF Propagation & Update
    -> GNSS Quality & State Machine -> Map Matcher -> NavigationState JSON Output Schema
    """
    def __init__(self, model_weights_path="models/tiny_tcn_best.pth", norm_json_path="data/processed/normalization.json"):
        self.ekf = EKF2D()
        self.gnss_monitor = GNSSQualityMonitor()
        self.map_matcher = PrototypeMapMatcher()
        
        # Load Normalization Parameters
        if os.path.exists(norm_json_path):
            with open(norm_json_path, "r") as f:
                self.norm_params = json.load(f)
            self.mean = np.array(self.norm_params["mean"], dtype=np.float32)
            self.std = np.array(self.norm_params["std"], dtype=np.float32)
        else:
            print(f"Warning: Normalization JSON {norm_json_path} not found. Using default stats.")
            self.mean = np.zeros(9, dtype=np.float32)
            self.std = np.ones(9, dtype=np.float32)
            
        # Load Tiny TCN Model
        self.model = TinyTCN(in_channels=9, num_motion_classes=5)
        if os.path.exists(model_weights_path):
            self.model.load_state_dict(torch.load(model_weights_path, weights_only=True))
            print(f"CorePipeline: Loaded model weights from {model_weights_path}")
        else:
            print(f"CorePipeline Warning: {model_weights_path} not found. Running initialized model.")
        self.model.eval()
        
        # Rolling IMU Window Buffer (20 frames = 2 seconds at 10Hz)
        self.window_size = 20
        self.buffer = collections.deque(maxlen=self.window_size)
        self.last_timestamp = None
        self.last_ai_confidence = 0.8
        
        # Base origin (Lat/Lon reference for local tangent plane ENU coordinates)
        self.ref_lat = 26.1445
        self.ref_lon = 91.7362

    def _enu_to_latlon(self, px, py):
        """Converts local ENU (m) position to Lat/Lon coordinates."""
        lat = self.ref_lat + (py / 111320.0)
        lon = self.ref_lon + (px / (111320.0 * math.cos(math.radians(self.ref_lat))))
        return lat, lon

    def process_frame(self, frame):
        """
        Processes a single SensorFrame input.
        
        frame schema:
        {
          'timestamp': float,
          'ax': float, 'ay': float, 'az': float,
          'gx': float, 'gy': float, 'gz': float,
          'gnss': dict or None -> {'lat': float, 'lon': float, 'speed': float, 'accuracy': float}
        }
        """
        ts = frame.get('timestamp', 0.0)
        ax = frame.get('ax', 0.0)
        ay = frame.get('ay', 0.0)
        az = frame.get('az', 9.81)
        gx = frame.get('gx', 0.0)
        gy = frame.get('gy', 0.0)
        gz = frame.get('gz', 0.0)
        
        # Calculate derived channels
        accel_mag = math.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag = math.sqrt(gx**2 + gy**2 + gz**2)
        
        if len(self.buffer) > 0:
            last_a_mag = self.buffer[-1][6]
            jerk = (accel_mag - last_a_mag) / 0.1
        else:
            jerk = 0.0
            
        raw_ch = np.array([ax, ay, az, gx, gy, gz, accel_mag, gyro_mag, jerk], dtype=np.float32)
        self.buffer.append(raw_ch)
        
        # 1. EKF INS Prediction Step
        if self.last_timestamp is not None:
            dt = ts - self.last_timestamp
            if dt > 0 and dt < 1.0:
                self.ekf.predict(dt, ax, ay, gz)
        self.last_timestamp = ts
        
        # 2. AI Model Inference (when buffer has 20 frames)
        if len(self.buffer) == self.window_size:
            raw_window = np.array(self.buffer, dtype=np.float32) # Shape: (20, 9)
            norm_window = (raw_window - self.mean) / self.std   # Normalized
            
            # Reshape to (1, C=9, T=20) for Conv1D
            tensor_in = torch.tensor(np.transpose(norm_window, (1, 0)), dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                pred_speed, log_var, motion_logits = self.model(tensor_in)
                
            v_ai = float(pred_speed[0].item())
            var_ai = float(torch.exp(log_var[0]).item())
            motion_prob = float(torch.softmax(motion_logits[0], dim=0).max().item())
            
            # AI Confidence metric
            self.last_ai_confidence = max(0.1, min(1.0, motion_prob / (1.0 + var_ai)))
            
            # Apply AI Velocity Update step into EKF
            self.ekf.update_ai_velocity(v_ai, variance_ai=var_ai)
            
        # 3. Non-Holonomic Constraint (NHC) Update
        self.ekf.update_nhc(R_nhc=0.01)
        
        # 4. GNSS Measurement Update & State Machine Evaluation
        gnss_sample = frame.get('gnss', None)
        nis = None
        
        if gnss_sample is not None and 'lat' in gnss_sample:
            gnss_px = (gnss_sample['lon'] - self.ref_lon) * (111320.0 * math.cos(math.radians(self.ref_lat)))
            gnss_py = (gnss_sample['lat'] - self.ref_lat) * 111320.0
            speed = gnss_sample.get('speed', 0.0)
            gnss_vx = speed * math.sin(math.radians(gnss_sample.get('heading', 0.0)))
            gnss_vy = speed * math.cos(math.radians(gnss_sample.get('heading', 0.0)))
            
            nis, _ = self.ekf.update_gnss(gnss_px, gnss_py, gnss_vx, gnss_vy, pos_std=gnss_sample.get('accuracy', 5.0))
            
        mode, gnss_quality = self.gnss_monitor.update(ts, gnss_sample=gnss_sample, nis=nis)
        
        # 5. Map Matching
        cur_px = self.ekf.x[0, 0]
        cur_py = self.ekf.x[1, 0]
        cur_hdg = self.ekf.heading_deg
        
        matched_px, matched_py, matched_hdg, map_conf = self.map_matcher.match_position(cur_px, cur_py, cur_hdg)
        lat, lon = self._enu_to_latlon(matched_px, matched_py)
        
        # 6. Standardized Section 56 Output Schema
        nav_state = {
            "timestamp": int(ts * 1000),
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "speed": round(self.ekf.speed * 3.6, 2), # km/h
            "heading": round(cur_hdg, 1),           # degrees
            "mode": mode,
            "gnssQuality": round(gnss_quality, 2),
            "aiConfidence": round(self.last_ai_confidence, 2),
            "mapConfidence": round(map_conf, 2)
        }
        
        return nav_state
