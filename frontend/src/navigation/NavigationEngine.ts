import { EKF2D } from './ins_ekf';
import { GNSSQualityMonitor } from './gnss_quality';
import type { NavigationStateModeType } from './gnss_quality';
import type { SensorFrame } from '../types/sensor';
import type { AIOutput } from '../sensors/ModelInferenceService';

export interface NavigationState {
  timestamp: number;
  latitude: number;
  longitude: number;
  speed: number;
  heading: number;
  mode: NavigationStateModeType;
  gnssQuality: number;
  aiConfidence: number;
  mapConfidence: number;
}

export class NavigationEngine {
  private ekf: EKF2D;
  private gnssMonitor: GNSSQualityMonitor;
  private lastTimestamp: number | null = null;
  private lastAiConfidence: number = 0.8;

  private refLat = 0;
  private refLon = 0;
  private R_EARTH = 111320.0;
  private isInitialized = false;
  private gnssHeadingInitialized = false;

  constructor() {
    this.ekf = new EKF2D();
    this.gnssMonitor = new GNSSQualityMonitor();
  }

  private enuToLatLon(px: number, py: number): { lat: number, lon: number } {
    const lat = this.refLat + (py / this.R_EARTH);
    const lon = this.refLon + (px / (this.R_EARTH * Math.cos((this.refLat * Math.PI) / 180.0)));
    return { lat, lon };
  }

  public process(frame: SensorFrame, aiOutput: AIOutput | null, isStationary: boolean = false): NavigationState | null {
    if (!this.isInitialized) {
      if (frame.gnss && frame.gnss.latitude !== null && frame.gnss.longitude !== null) {
        this.refLat = frame.gnss.latitude;
        this.refLon = frame.gnss.longitude;
        this.isInitialized = true;
      } else {
        return null; // Wait for first GNSS to set origin
      }
    }

    const ts = frame.timestamp / 1000.0; // convert to seconds
    const ax = frame.ax;
    const ay = frame.ay;
    const gz = frame.gz;

    // 1. EKF INS Prediction Step
    if (this.lastTimestamp !== null) {
      const dt = ts - this.lastTimestamp;
      if (dt > 0 && dt < 1.0) {
        this.ekf.predict(dt, ax, ay, gz);
      }
    }
    this.lastTimestamp = ts;

    // 2. AI Model Inference Update & Stationary Constraints
    if (isStationary) {
      this.lastAiConfidence = 1.0;
      this.ekf.updateAIVelocity(0.0, 0.001); // High-confidence Zero Velocity Update
    } else if (aiOutput) {
      const v_ai = aiOutput.speed_mps;
      const var_ai = aiOutput.variance;
      const motion_prob = aiOutput.motionConfidence;

      this.lastAiConfidence = Math.max(0.1, Math.min(1.0, motion_prob / (1.0 + var_ai)));
      this.ekf.updateAIVelocity(v_ai, var_ai);
    }

    // 3. NHC Update
    this.ekf.updateNHC(0.01);

    // 4. GNSS Measurement Update
    let nis: number | null = null;
    if (frame.gnss && frame.gnss.latitude !== null && frame.gnss.longitude !== null) {
      const gnss_px = (frame.gnss.longitude - this.refLon) * (this.R_EARTH * Math.cos((this.refLat * Math.PI) / 180.0));
      const gnss_py = (frame.gnss.latitude - this.refLat) * this.R_EARTH;
      const speed = frame.gnss.speed || 0.0;
      const heading = frame.gnss.heading || 0.0;
      
      // Initialize EKF state properly if moving fast enough and we haven't locked heading yet
      if (speed > 1.0 && !this.gnssHeadingInitialized) {
        // Convert Nav Heading (CW from North) to Math Angle (CCW from East)
        let mathAngleDeg = (90 - heading + 360) % 360;
        let headingRad = (mathAngleDeg * Math.PI) / 180.0;
        this.ekf.x.set(4, 0, headingRad);
        this.gnssHeadingInitialized = true;
      }

      const gnss_vx = speed * Math.sin((heading * Math.PI) / 180.0);
      const gnss_vy = speed * Math.cos((heading * Math.PI) / 180.0);

      const result = this.ekf.updateGNSS(gnss_px, gnss_py, gnss_vx, gnss_vy, frame.gnss.accuracy || 5.0, 0.5);
      nis = result.nis;
    }

    const { mode, gnssQuality } = this.gnssMonitor.update(ts, frame.gnss, nis);

    // 5. Map Matching (Skipped in this phase, use raw EKF output)
    const cur_px = this.ekf.x.get(0, 0);
    const cur_py = this.ekf.x.get(1, 0);
    const cur_hdg = this.ekf.headingDeg;
    const map_conf = 0.0;

    const { lat, lon } = this.enuToLatLon(cur_px, cur_py);

    return {
      timestamp: frame.timestamp,
      latitude: parseFloat(lat.toFixed(6)),
      longitude: parseFloat(lon.toFixed(6)),
      speed: parseFloat((this.ekf.speed * 3.6).toFixed(2)),
      heading: parseFloat(cur_hdg.toFixed(1)),
      mode: mode,
      gnssQuality: parseFloat(gnssQuality.toFixed(2)),
      aiConfidence: parseFloat(this.lastAiConfidence.toFixed(2)),
      mapConfidence: parseFloat(map_conf.toFixed(2))
    };
  }

  // Used for reference-equivalence tests
  public setInitialState(x: number, y: number, headingRad: number) {
    this.ekf = new EKF2D(x, y, headingRad);
  }
}
