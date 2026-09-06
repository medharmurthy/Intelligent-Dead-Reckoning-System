import os
import sys
import json
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.navigation.pipeline import CoreNavigationPipeline

def create_frame(ts, ax, ay, az, gx, gy, gz, gnss=None):
    return {
        "timestamp": ts,
        "ax": ax, "ay": ay, "az": az,
        "gx": gx, "gy": gy, "gz": gz,
        "gnss": gnss
    }

def run_scenarios():
    # We will mock the AI output by overriding the pipeline's model inference
    # to return deterministic values, because we only want to test the Navigation Engine's logic,
    # not the actual neural network weights (which we already proved are identical).
    
    class MockPipeline(CoreNavigationPipeline):
        def __init__(self):
            super().__init__()
            self.mock_ai_speed = 10.0
            self.mock_ai_var = 0.1
            self.mock_motion_prob = 0.9

        def process_frame(self, frame):
            # Override inference block
            original_len = len(self.buffer)
            # Run normal frame processing
            nav_state = super().process_frame(frame)
            
            # If buffer reached 20, the superclass ran inference. We will OVERWRITE its effects
            # by doing an AI update with our mock values, but wait, the superclass ALREADY did an update.
            # It's better to just manually feed it or monkeypatch the model.
            return nav_state
            
    # Better yet, let's monkeypatch the model's forward pass!
    import torch
    def mock_forward(x):
        # speed = 10.0, log_var = ln(0.1) = -2.302, logits = [0,0,10,0,0] (Cruising)
        speed = torch.tensor([[10.0]])
        log_var = torch.tensor([[-2.302585]])
        logits = torch.tensor([[0.0, 0.0, 10.0, 0.0, 0.0]])
        return speed, log_var, logits
    
    pipeline = CoreNavigationPipeline()
    pipeline.model.forward = mock_forward
    
    results = {}
    
    # Base IMU
    ax, ay, az = 0.0, 0.0, 9.81
    gx, gy, gz = 0.0, 0.0, 0.0
    
    # Test A: GNSS Available (GNSS_AIDED)
    # Feed 25 frames (2.5 seconds) with GNSS
    states_a = []
    for i in range(25):
        ts = i * 0.1
        gnss = {"lat": 26.1445 + i*0.0001, "lon": 91.7362, "speed": 10.0, "heading": 0.0, "accuracy": 2.0}
        states_a.append(pipeline.process_frame(create_frame(ts, ax, ay, az, gx, gy, gz, gnss)))
    results["TestA"] = states_a[-1]

    # Test B: GNSS Outage (DEAD_RECKONING)
    # Continue from A, but drop GNSS for 30 frames (3 seconds)
    states_b = []
    for i in range(25, 55):
        ts = i * 0.1
        states_b.append(pipeline.process_frame(create_frame(ts, ax, ay, az, gx, gy, gz, None)))
    results["TestB"] = states_b[-1]
    
    # Test C: GNSS Reacquisition
    # Feed noisy GNSS then good GNSS
    states_c = []
    for i in range(55, 60): # 5 noisy frames -> REACQUISITION
        ts = i * 0.1
        gnss = {"lat": 26.1445 + i*0.0001, "lon": 91.7362, "speed": 10.0, "heading": 0.0, "accuracy": 10.0}
        states_c.append(pipeline.process_frame(create_frame(ts, ax, ay, az, gx, gy, gz, gnss)))
    for i in range(60, 65): # 5 good frames -> GNSS_AIDED
        ts = i * 0.1
        gnss = {"lat": 26.1445 + i*0.0001, "lon": 91.7362, "speed": 10.0, "heading": 0.0, "accuracy": 2.0}
        states_c.append(pipeline.process_frame(create_frame(ts, ax, ay, az, gx, gy, gz, gnss)))
    results["TestC"] = states_c[-1]

    # Test D: Constant Forward Motion (Check Speed)
    # The AI model outputs 10.0 m/s. We expect speed to converge near 10.0.
    results["TestD"] = states_c[-1] # We can just assert speed in the JS test

    # Test E: Turning Motion
    # Apply yaw rate (gz = 0.1 rad/s) for 10 frames
    states_e = []
    for i in range(65, 75):
        ts = i * 0.1
        states_e.append(pipeline.process_frame(create_frame(ts, ax, ay, az, gx, gy, gz=0.1, gnss=None)))
    results["TestE"] = states_e[-1]
    
    os.makedirs("frontend/public/data", exist_ok=True)
    with open("frontend/public/data/test_nav.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_scenarios()
