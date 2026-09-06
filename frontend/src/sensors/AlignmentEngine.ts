import type { SensorFrame } from '../types/sensor';

export type AlignmentStatus = 
  | 'UNINITIALIZED'
  | 'WAITING_FOR_STATIONARY'
  | 'LEVEL_ESTIMATION'
  | 'WAITING_FOR_MOTION'
  | 'YAW_ALIGNMENT'
  | 'ALIGNED';

export interface AlignmentState {
  status: AlignmentStatus;
  confidence: number;
  rotationMatrix: number[][] | null; // 3x3 matrix mapping Phone -> Vehicle
  z_v: number[] | null;
  y_v: number[] | null;
  x_v: number[] | null;
}

export class AlignmentEngine {
  private state: AlignmentState = {
    status: 'UNINITIALIZED',
    confidence: 0,
    rotationMatrix: null,
    z_v: null,
    y_v: null,
    x_v: null
  };

  private gravityBuffer: number[][] = [];
  private readonly GRAVITY_SAMPLES_REQUIRED = 20; // 2s at 10Hz

  private horizontalAccelBuffer: number[][] = [];
  private readonly YAW_SAMPLES_REQUIRED = 15; // 1.5s of forward acceleration
  private lastGnssSpeed: number | null = null;
  
  public getState(): AlignmentState {
    return this.state;
  }

  public reset() {
    this.state = {
      status: 'UNINITIALIZED',
      confidence: 0,
      rotationMatrix: null,
      z_v: null,
      y_v: null,
      x_v: null
    };
    this.gravityBuffer = [];
    this.horizontalAccelBuffer = [];
    this.lastGnssSpeed = null;
  }

  public processFrame(frame: SensorFrame): SensorFrame {
    this.updateCalibration(frame);

    if (this.state.status === 'ALIGNED' && frame.ax !== null) {
      return this.rotateFrame(frame, this.state.rotationMatrix!);
    }

    return frame; 
  }

  private updateCalibration(frame: SensorFrame) {
    if (this.state.status === 'ALIGNED') return;

    if (this.state.status === 'UNINITIALIZED') {
      this.state.status = 'WAITING_FOR_STATIONARY';
    }

    if (this.state.status === 'WAITING_FOR_STATIONARY' || this.state.status === 'LEVEL_ESTIMATION') {
      if (frame.ax === null || frame.ay === null || frame.az === null) return;
      
      this.state.status = 'LEVEL_ESTIMATION';
      this.gravityBuffer.push([frame.ax, frame.ay, frame.az]);
      
      if (this.gravityBuffer.length > this.GRAVITY_SAMPLES_REQUIRED) {
        this.gravityBuffer.shift();
      }

      if (this.gravityBuffer.length === this.GRAVITY_SAMPLES_REQUIRED) {
        const mean = this.getMean(this.gravityBuffer);
        const variance = this.getVariance(this.gravityBuffer, mean);
        
        // Loosened variance threshold to 5.0 for handheld/bicycle stability
        if (variance < 5.0) {
          const norm = Math.sqrt(mean[0]**2 + mean[1]**2 + mean[2]**2);
          // Only accept if we actually have a vector (covers both 1.0g and 9.81 m/s^2 device APIs)
          if (norm > 0.5) {
            this.state.z_v = [mean[0]/norm, mean[1]/norm, mean[2]/norm];
            this.state.status = 'WAITING_FOR_MOTION';
            this.state.confidence = 0.3;
          }
        }
      }
    }

    if (this.state.status === 'WAITING_FOR_MOTION' || this.state.status === 'YAW_ALIGNMENT') {
      if (!this.state.z_v) return;

      if (frame.gnss && frame.gnss.speed !== null && frame.ax !== null && frame.ay !== null && frame.az !== null) {
        if (this.lastGnssSpeed === null) {
          this.lastGnssSpeed = frame.gnss.speed;
          return;
        }

        const dv = frame.gnss.speed - this.lastGnssSpeed;
        this.lastGnssSpeed = frame.gnss.speed;

        // Vehicle must be moving (> 1.0 m/s) and accelerating
        if (dv > 0.02 && frame.gnss.speed > 1.0) {
          this.state.status = 'YAW_ALIGNMENT';
          
          const a = [frame.ax, frame.ay, frame.az];
          const z = this.state.z_v;
          const dot = a[0]*z[0] + a[1]*z[1] + a[2]*z[2];
          
          const a_horiz = [
            a[0] - dot * z[0],
            a[1] - dot * z[1],
            a[2] - dot * z[2]
          ];

          this.horizontalAccelBuffer.push(a_horiz);

          if (this.horizontalAccelBuffer.length >= this.YAW_SAMPLES_REQUIRED) {
            const meanY = this.getMean(this.horizontalAccelBuffer);
            const normY = Math.sqrt(meanY[0]**2 + meanY[1]**2 + meanY[2]**2);
            
            if (normY > 0.1) {
              this.state.y_v = [meanY[0]/normY, meanY[1]/normY, meanY[2]/normY];
              
              const y = this.state.y_v;
              this.state.x_v = [
                y[1]*z[2] - y[2]*z[1],
                y[2]*z[0] - y[0]*z[2],
                y[0]*z[1] - y[1]*z[0]
              ];

              this.state.rotationMatrix = [
                this.state.x_v,
                this.state.y_v,
                this.state.z_v
              ];
              this.state.status = 'ALIGNED';
              this.state.confidence = 1.0;
            } else {
              this.horizontalAccelBuffer = [];
              this.state.status = 'WAITING_FOR_MOTION';
            }
          }
        }
      }
    }
  }

  public rotateFrame(frame: SensorFrame, R: number[][]): SensorFrame {
    const rotate = (v: number[]) => [
      R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
      R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
      R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2]
    ];

    let ax=frame.ax, ay=frame.ay, az=frame.az;
    let gx=frame.gx, gy=frame.gy, gz=frame.gz;

    if (ax !== null && ay !== null && az !== null) {
      const a_rot = rotate([ax, ay, az]);
      ax = a_rot[0]; ay = a_rot[1]; az = a_rot[2];
    }
    if (gx !== null && gy !== null && gz !== null) {
      const g_rot = rotate([gx, gy, gz]);
      gx = g_rot[0]; gy = g_rot[1]; gz = g_rot[2];
    }

    return {
      ...frame,
      ax, ay, az,
      gx, gy, gz
    };
  }

  private getMean(buffer: number[][]): number[] {
    const sum = [0, 0, 0];
    for (const v of buffer) {
      sum[0] += v[0];
      sum[1] += v[1];
      sum[2] += v[2];
    }
    return [sum[0]/buffer.length, sum[1]/buffer.length, sum[2]/buffer.length];
  }

  private getVariance(buffer: number[][], mean: number[]): number {
    let vSum = 0;
    for (const v of buffer) {
      vSum += (v[0]-mean[0])**2 + (v[1]-mean[1])**2 + (v[2]-mean[2])**2;
    }
    return vSum / buffer.length;
  }
}
