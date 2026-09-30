# Intelligent Dead Reckoning (IDR) System with GNSS Fusion

An inertial navigation engine and PWA web app built for Smart India Hackathon (SIH) Problem Statement PS26168.

When GNSS signals drop in tunnels, urban canyons, or underpasses, standard navigation apps lose tracking. This project uses smartphone IMU sensors (accelerometers and gyroscopes) combined with a Temporal Convolutional Network (TCN) speed model, a 6-state Extended Kalman Filter (EKF), and offline OpenStreetMap (OSM) map matching to estimate vehicle position during outages.

---

## Overview of System Components

- **Speed Model (Tiny-TCN)**: Estimates forward vehicle speed, model variance, and stationary probability from 4-second IMU windows.
- **Stand Calibration**: Aligns phone body-frame IMU data to the vehicle frame.
- **6-State INS EKF**: Tracks position (East, North), heading, speed, gyro bias, and gyro scale factor.
- **Map Matching**: Uses a Hidden Markov Model (HMM) to snap positions onto OpenStreetMap road segments and enforce non-holonomic constraints (NHC).
- **Navigation Modes**: Manages transitions between `GNSS_AIDED`, `DEGRADED`, `DEAD_RECKONING`, and `REACQUISITION`.
- **Browser PWA**: A vanilla JavaScript app supporting live navigation, replay mode, stand calibration, and tunnel simulation, running inference via ONNX Runtime Web.

---

## Performance Benchmark

Evaluated on the IO-VNBD benchmark dataset (test trip `S1`) over simulated GNSS outages (see [docs/RESULTS.md](docs/RESULTS.md)):

| Navigation Engine (1 Hz GNSS) | 60 s Outage Drift | 120 s Outage Drift | Pass Rate (<10% Target) |
|---|---|---|---|
| Baseline Engine | 97.0% median | 81.0% median | 0% |
| New Engine (No Map Matching) | 17.0% median | 14.0% median | 35% |
| New Engine (Full IDR System) | **9.2% [5.6–12.5% CI]** | **9.1% [6.8–13.6% CI]** | **52% pass (32 outages)** |
| Perfect Speed Oracle | 1.0% median | 0.3% median | 100% |

- **Execution speed**: 0.89 ms per 10 Hz frame (~113x faster than real-time on mobile).
- **Code parity**: Python navigation logic (`src/navigation/`) matches the browser JavaScript runtime (`src/web/nav/js/`).

---

## Repository Structure

```text
PS26168/
├── README.md                                 # Project documentation
├── AGENTS.md                                 # Rules, data contracts, and guidelines
├── ProblemStatement.md                       # SIH PS26168 problem statement text
├── requirements.txt                          # Python dependencies
├── calibration.json                          # Saved mount calibration parameters
├── docs/                                     # Documentation
│   ├── SUMMARY.md                            # Documentation index
│   ├── OPTIMIZATION_PLAN.md                  # Design rationale and math details
│   ├── RESULTS.md                            # Complete benchmark results
│   ├── TESTING.md                            # Test guidelines and commands
│   ├── CHANGES.md                            # Changelog
│   ├── NEXT_STEPS.md                         # Planned improvements
│   └── NEW_MODEL_WITH_OLD_PIPELINE.md        # Integration notes
├── data/                                     # Datasets (gitignored)
│   ├── raw/IO-VNBD/                          # Benchmark dataset
│   ├── intermediate/                         # Aligned trip files
│   ├── maps/                                 # Offline GeoJSON OSM maps
│   └── processed/                            # Windowed sequences and normalization config
├── models/                                   # Model checkpoints and ONNX files
│   ├── tiny_tcn_best.pth                     # PyTorch model
│   ├── idr_tcn.onnx                          # Exported ONNX model
│   └── model_config.json                     # Runtime normalization parameters
├── results/                                  # Benchmark metrics and plots
├── src/                                      # Source code
│   ├── preprocessing/                        # Data parsing, feature extraction, windowing
│   ├── models/                               # TCN model, training, ONNX export
│   ├── navigation/                           # EKF, GNSS state machine, map matcher, pipeline
│   ├── evaluation/                           # Benchmark scripts
│   └── web/                                  # Web server, calibrator, PWA app
└── tests/                                    # Pytest suite, JS parity tests, E2E tests
```

---

## Quick Start

### 1. Requirements and Setup

Requires Python 3.10+ and Node.js.

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Data Preparation

Download IO-VNBD benchmark trips, align sensor timestamps, fetch OSM maps, and build training windows:

```bash
python src/preprocessing/download_trip.py
python src/preprocessing/parse_trip.py
python src/preprocessing/fetch_osm.py
python src/preprocessing/create_windows.py
```

### 3. Model Training & Export

Train the Tiny-TCN model and export to ONNX:

```bash
python src/models/tiny_tcn.py --epochs 40
python src/models/evaluate_models.py
python src/models/export_onnx.py
```

---

## Web Application

The web interface includes live navigation, replay mode, stand calibration, and tunnel simulation.

1. **Set up vendor scripts and export replay data:**
   ```bash
   python src/web/setup_vendor.py
   python src/web/export_replay.py --trip S1
   ```

2. **Start the local server:**
   ```bash
   python src/web/server.py
   ```

3. **Open in browser:**
   - **Navigator**: `https://localhost:8443/nav/`
   - **Stand Calibrator**: `https://localhost:8443/calibration/`

To test GNSS outages in replay mode, select trip `S1`, start playback, and click **"Simulate Tunnel"**.

To download offline map data for live navigation in a specific area:
```bash
python src/web/export_live_map.py --lat <LATITUDE> --lon <LONGITUDE> --radius-km 5
```

---

## Testing

### Python Unit Tests
```bash
python -m pytest tests
```

### Python / JavaScript Parity Tests
```bash
python tests/js/make_fixture.py
node tests/js/parity.test.mjs
python tests/js/make_gnss_fixture.py
node tests/js/gnss_monitor.test.mjs
```

### End-to-End Browser Tests
```bash
npm install --prefix tests/e2e
node tests/e2e/ui_e2e.mjs
```

### Benchmark Suite
```bash
python src/evaluation/benchmark.py --trips S1 Y1
```

---
