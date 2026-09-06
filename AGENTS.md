# AGENTS.md — SIH PS26168: Intelligent Dead Reckoning (IDR) Project Guidelines

Welcome! This document defines the project scope, repository architecture, coding standards, and AI agent instructions for **SIH Project PS26168: Intelligent Dead Reckoning System with GNSS Fusion**.

---

## 🎯 Project Identity & Overview

* **Problem Statement**: PS26168 — Continuous smartphone-based vehicle navigation during GNSS outages (tunnels, underpasses, urban canyons).
* **Core Technology Stack**:
  * **Language**: Python 3.12+
  * **ML Framework**: PyTorch, Scikit-Learn, NumPy
  * **Visualization**: Matplotlib
  * **Deployment Target**: ONNX Runtime (Mobile Edge / Chrome PWA)

---

## 📁 Repository Directory Structure

```text
sih/
├── AGENTS.md                     # Agent rules and project guidelines
├── .gitignore                    # Excludes raw data and binary files from Git
├── data/
│   ├── raw/
│   │   └── IO-VNBD/              # Raw downloaded benchmark dataset files (untracked)
│   ├── intermediate/             # Cleaned & calibrated intermediate frames
│   └── processed/
│       ├── normalization.json     # Feature mean (μ) & std (σ) for live model scaling
│       ├── train_windows.npz      # Windowed training tensors (N x 20 x 9)
│       ├── val_windows.npz        # Windowed validation tensors (N x 20 x 9)
│       └── speed_prediction_benchmark.png  # Benchmark evaluation chart
├── models/
│   ├── mlp_baseline.pth          # PyTorch MLP baseline checkpoint
│   └── tiny_tcn.pth              # Multi-Task Tiny TCN model checkpoint (Speed + Uncertainty)
└── src/
    ├── preprocessing/
    │   ├── download_trip.py      # Script to download Git LFS dataset binaries
    │   ├── parse_trip.py         # CSV parser and dataset cleaner
    │   └── create_windows.py     # 2-second rolling window tensor generator
    └── models/
        ├── sklearn_baseline.py   # Ridge & Random Forest baseline evaluation
        ├── mlp_baseline.py       # PyTorch MLP training script
        ├── tiny_tcn.py           # Multi-Task Tiny TCN architecture & trainer
        └── evaluate_models.py    # Validation metric & benchmark plot generator
```

---

## 📐 Data Specs & Feature Definitions

* **Sampling Rate**: 10 Hz (0.1s time step $\Delta t$).
* **Window Size**: 2-second rolling windows (**20 frames**).
* **Input Feature Channels ($C=9$)**:
  1. `ax`: Acceleration X ($m/s^2$)
  2. `ay`: Acceleration Y ($m/s^2$)
  3. `az`: Acceleration Z ($m/s^2$)
  4. `gx`: Gyroscope Pitch rate ($rad/s$)
  5. `gy`: Gyroscope Roll rate ($rad/s$)
  6. `gz`: Gyroscope Yaw rate ($rad/s$)
  7. `accel_mag`: Total Acceleration Magnitude $|a| = \sqrt{a_x^2 + a_y^2 + a_z^2}$
  8. `gyro_mag`: Total Gyroscope Magnitude $|\omega| = \sqrt{g_x^2 + g_y^2 + g_z^2}$
  9. `jerk`: Numerical derivative of acceleration magnitude $j = \frac{d|a|}{dt}$
* **Target Label ($y$)**: Ground-truth forward speed $v_f$ ($m/s$) from vehicle CAN-bus reference.

---

## ⚙️ Coding Standards & Rules for AI Assistants

1. **Do NOT Modify Raw Data**: Never alter original files inside `data/raw/`. Always create processed outputs in `data/processed/`.
2. **Preserve Normalization Integrity**: All model training and live inference MUST consume feature scaling parameters from `data/processed/normalization.json`.
3. **Keep Heavy Binaries Untracked**: Large dataset CSVs, `.npz` binary tensors, and heavy checkpoint weights must be excluded via `.gitignore`.
4. **Maintain Modular Software Interfaces**:
   - **`SensorFrame`**: Raw smartphone IMU data stream.
   - **`AIOutput`**: Predicted velocity $v_f$ + variance $\sigma_v^2$.
   - **`NavigationState`**: Filtered position $(x, y)$, heading $\psi$, navigation mode (`GNSS_AIDED` / `DEAD_RECKONING`).

---

## 🚀 Execution Commands Reference

* **Fetch Raw Dataset**: `python src/preprocessing/download_trip.py`
* **Generate Tensors**: `python src/preprocessing/create_windows.py`
* **Train Tiny TCN**: `python src/models/tiny_tcn.py`
* **Evaluate & Plot**: `python src/models/evaluate_models.py`
