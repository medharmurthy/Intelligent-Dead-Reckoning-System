# IDR Frontend MVP - Setup & Development Guide

This directory contains the Intelligent Dead Reckoning (IDR) smartphone application, built as a Progressive Web Application (PWA).

## Features
- **Raw Sensor Acquisition:** Accesses 100Hz+ DeviceMotion and Geolocation APIs.
- **10Hz Resampling:** Deterministically down-samples sensors into 20-frame buffers.
- **ONNX Runtime Web:** Executes the PyTorch `Tiny TCN` model locally via WebAssembly (`wasm`) to predict vehicle speed and variance from raw IMU data.
- **TypeScript EKF:** Runs an 8-state Extended Kalman Filter and GNSS State Machine to fuse the AI Velocity with actual GPS drops.
- **Mapping & Replay:** Renders live trajectories with `react-leaflet`, color-codes navigation modes (GNSS, DEGRADED, DR, REACQ), and supports deterministic test-driven replay modes for demoing GNSS loss.

## Installation

```bash
cd frontend
npm install
```

## Running Development Server

```bash
npm run dev
```

## Building for Production (PWA)

```bash
npm run build
```
The output will be inside the `dist/` folder, which can be served using any static HTTP server. Because of the `vite-plugin-pwa` integration, the app will generate a service worker and manifest to be fully installable on Android/iOS and cache the `idr_tcn.onnx` file for offline use.

## Testing

```bash
npm run test
```
Runs deterministic Math integration tests verifying the `ins_ekf.ts` pipeline strictly matches the Python modeling baseline out to 4 decimal places.

## Troubleshooting

- **Permissions:** iOS requires explicit user interaction to trigger `DeviceMotionEvent.requestPermission()`. If permissions are blocked, the app cannot function.
- **ONNX WebAssembly:** If the inference fails to load, ensure `vite-plugin-static-copy` successfully copies `ort-*.wasm` files from `node_modules/onnxruntime-web/dist/` to your output directory.
