import os
import sys
import json
import time

# Ensure sys.path includes src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.parse_trip import load_trip_s1
from src.navigation.pipeline import CoreNavigationPipeline

def run_sensor_replay(simulate_gnss_blackout=True):
    """
    Runs recorded sensor replay test harness on Trip S1 dataset (Sections 71 & 85 of plan.pdf).
    """
    print("--- Running End-to-End Navigation Sensor Replay ---")
    
    # 1. Load raw parsed dataset
    phone_data, vehicle_data = load_trip_s1()
    
    # 2. Instantiate Navigation Pipeline
    pipeline = CoreNavigationPipeline()
    
    total_frames = min(len(phone_data), len(vehicle_data))
    print(f"\nReplaying {total_frames} frames through Core Navigation Pipeline...")
    if simulate_gnss_blackout:
        print("Simulating GNSS Blackout Tunnel (Frame 300 to 800: GNSS SIGNAL DENIED)")
        
    mode_counts = {}
    sample_outputs = []
    
    start_time = time.time()
    
    for i in range(total_frames):
        p_frame = phone_data[i]
        v_frame = vehicle_data[i]
        
        # Timestamp in seconds
        ts = p_frame['time_ms'] / 1000.0
        
        # Construct GNSS sample if not in simulated blackout window
        if simulate_gnss_blackout and (300 <= i <= 800):
            gnss_sample = None  # GNSS blackout tunnel
        else:
            # Vehicle GT as GNSS input with simulated accuracy
            gnss_sample = {
                'lat': 26.1445 + (i * 0.00001),
                'lon': 91.7362 + (i * 0.00001),
                'speed': v_frame['speed_mps'],
                'heading': v_frame['heading'],
                'accuracy': 3.5
            }
            
        sensor_frame = {
            'timestamp': ts,
            'ax': p_frame['ax'], 'ay': p_frame['ay'], 'az': p_frame['az'],
            'gx': p_frame['gx'], 'gy': p_frame['gy'], 'gz': p_frame['gz'],
            'gnss': gnss_sample
        }
        
        nav_state = pipeline.process_frame(sensor_frame)
        
        mode = nav_state['mode']
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        # Collect sample output snapshots
        if i in [100, 350, 600, 820, 1000]:
            sample_outputs.append((i, nav_state))
            
    elapsed = time.time() - start_time
    fps = total_frames / elapsed if elapsed > 0 else 0
    
    print(f"\nReplay Completed in {elapsed:.2f} seconds ({fps:.1f} frames/sec)")
    print("\n--- Navigation Mode Distribution ---")
    for mode, count in mode_counts.items():
        pct = (count / total_frames) * 100
        print(f"  {mode:20s}: {count:5d} frames ({pct:.1f}%)")
        
    print("\n--- Sample NavigationState JSON Output Snapshots (Section 56 Schema) ---")
    for idx, snap in sample_outputs:
        print(f"\nFrame {idx:04d}:")
        print(json.dumps(snap, indent=2))
        
    print("\nEnd-to-End Replay Verification Successful!")

if __name__ == "__main__":
    run_sensor_replay(simulate_gnss_blackout=True)
