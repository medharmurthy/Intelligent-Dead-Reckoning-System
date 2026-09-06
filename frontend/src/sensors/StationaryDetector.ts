import type { SensorFrame } from '../types/sensor';

export type StationaryStatus = 'MOVING' | 'POSSIBLY_STATIONARY' | 'STATIONARY';

export interface StationaryState {
  isStationary: boolean;
  status: StationaryStatus;
  confidence: number;
  durationMs: number;
}

export class StationaryDetector {
  private state: StationaryState = {
    isStationary: false,
    status: 'MOVING',
    confidence: 0.0,
    durationMs: 0
  };

  private magBuffer: number[] = [];
  private gyroBuffer: number[] = [];
  private readonly WINDOW_SIZE = 15; // 1.5 seconds at 10Hz
  
  // Thresholds based on typical automotive stationary engine vibration vs hand/vehicle motion
  private readonly ACCEL_VAR_THRESH = 0.05; 
  private readonly GYRO_MAG_THRESH = 0.08;
  
  private stationaryStartTime: number | null = null;

  public getState(): StationaryState {
    return this.state;
  }

  public processFrame(frame: SensorFrame): StationaryState {
    if (frame.ax === null || frame.ay === null || frame.az === null || 
        frame.gx === null || frame.gy === null || frame.gz === null) {
      return this.state;
    }

    const accelMag = Math.sqrt(frame.ax**2 + frame.ay**2 + frame.az**2);
    const gyroMag = Math.sqrt(frame.gx**2 + frame.gy**2 + frame.gz**2);

    this.magBuffer.push(accelMag);
    this.gyroBuffer.push(gyroMag);

    if (this.magBuffer.length > this.WINDOW_SIZE) {
      this.magBuffer.shift();
      this.gyroBuffer.shift();
    }

    if (this.magBuffer.length === this.WINDOW_SIZE) {
      const mean = this.magBuffer.reduce((a, b) => a + b, 0) / this.WINDOW_SIZE;
      const variance = this.magBuffer.reduce((a, b) => a + (b - mean)**2, 0) / this.WINDOW_SIZE;
      const meanGyro = this.gyroBuffer.reduce((a, b) => a + b, 0) / this.WINDOW_SIZE;

      let physicallyStationary = variance < this.ACCEL_VAR_THRESH && meanGyro < this.GYRO_MAG_THRESH;
      
      // If GNSS is available, it provides a strong prior
      if (frame.gnss && frame.gnss.speed !== null) {
        if (frame.gnss.speed > 1.0) {
          physicallyStationary = false; // Definitely moving
        } else if (frame.gnss.speed < 0.2 && physicallyStationary) {
          this.state.confidence = 0.95; 
        }
      }

      if (physicallyStationary) {
        if (this.state.status === 'MOVING') {
          this.state.status = 'POSSIBLY_STATIONARY';
          this.stationaryStartTime = frame.timestamp;
        } else if (this.state.status === 'POSSIBLY_STATIONARY') {
          if (frame.timestamp - this.stationaryStartTime! > 1000) {
            this.state.status = 'STATIONARY';
            this.state.isStationary = true;
          }
        }
      } else {
        this.state.status = 'MOVING';
        this.state.isStationary = false;
        this.stationaryStartTime = null;
        this.state.durationMs = 0;
      }
    }

    if (this.state.isStationary && this.stationaryStartTime) {
      this.state.durationMs = frame.timestamp - this.stationaryStartTime;
      // Confidence grows with duration up to 1.0
      this.state.confidence = Math.min(1.0, 0.5 + (this.state.durationMs / 5000));
    } else if (!this.state.isStationary) {
      this.state.confidence = 0.0;
    }

    return this.state;
  }
}
