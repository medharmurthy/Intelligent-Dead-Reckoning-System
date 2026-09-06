import type { SensorFrame } from '../types/sensor';

export interface FeatureFrame {
  timestamp: number;
  channels: [number, number, number, number, number, number, number, number, number];
}

export class FeatureExtractor {
  private lastAccelMag: number | null = null;
  private lastTimestamp: number | null = null;

  public extract(frame: SensorFrame): FeatureFrame {
    const { timestamp, ax, ay, az, gx, gy, gz } = frame;

    const accel_mag = Math.sqrt(ax * ax + ay * ay + az * az);
    const gyro_mag = Math.sqrt(gx * gx + gy * gy + gz * gz);

    let jerk = 0.0;
    if (this.lastAccelMag !== null && this.lastTimestamp !== null) {
      // In training, dt is assumed exactly 0.1s. 
      // The Python code: jerk = (accel_mag - last_a_mag) / 0.1
      // We enforce the exact same formula to perfectly match training distribution.
      jerk = (accel_mag - this.lastAccelMag) / 0.1;
    }

    this.lastAccelMag = accel_mag;
    this.lastTimestamp = timestamp;

    return {
      timestamp,
      channels: [ax, ay, az, gx, gy, gz, accel_mag, gyro_mag, jerk],
    };
  }

  public reset() {
    this.lastAccelMag = null;
    this.lastTimestamp = null;
  }
}
