import os
import sys
import json
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.navigation.ins_ekf import EKF2D

def run_test():
    ekf = EKF2D()
    out_steps = []
    
    # Initial state
    out_steps.append({"type": "init", "x": ekf.x.flatten().tolist()})
    
    # Step 1: Predict (constant dt=0.1, some ax/ay/gz)
    dt = 0.1
    ekf.predict(dt, ax_raw=0.5, ay_raw=0.1, gz_raw=0.02)
    out_steps.append({"type": "predict", "x": ekf.x.flatten().tolist(), "P_diag": [ekf.P[i,i] for i in range(8)]})
    
    # Step 2: Predict again
    ekf.predict(dt, ax_raw=0.5, ay_raw=0.1, gz_raw=0.02)
    out_steps.append({"type": "predict", "x": ekf.x.flatten().tolist(), "P_diag": [ekf.P[i,i] for i in range(8)]})
    
    # Step 3: AI Update
    r, S = ekf.update_ai_velocity(v_forward_ai=1.0, variance_ai=0.1)
    out_steps.append({"type": "ai_update", "x": ekf.x.flatten().tolist(), "P_diag": [ekf.P[i,i] for i in range(8)], "r": r, "S": S})
    
    # Step 4: NHC Update
    ekf.update_nhc(R_nhc=0.01)
    out_steps.append({"type": "nhc_update", "x": ekf.x.flatten().tolist()})
    
    # Step 5: GNSS Update
    nis, r_gnss = ekf.update_gnss(px_gnss=0.1, py_gnss=0.1, vx_gnss=1.0, vy_gnss=0.1, pos_std=2.0, vel_std=0.5)
    out_steps.append({"type": "gnss_update", "x": ekf.x.flatten().tolist(), "nis": nis})

    os.makedirs("frontend/public/data", exist_ok=True)
    with open("frontend/public/data/test_ekf.json", "w") as f:
        json.dump(out_steps, f, indent=2)
        
    print("Generated EKF deterministic test data.")

if __name__ == "__main__":
    run_test()
