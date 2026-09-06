import type { RawSensorSample, SensorFrame, GNSSSample } from '../types/sensor';

export class SensorResampler {
  private intervalMs: number;
  
  private imuBuffer: RawSensorSample[] = [];
  private latestGNSS: GNSSSample | null = null;
  
  private nextEmitTime: number | null = null;
  private maxGapMs = 500; // If gap > 500ms, reset

  constructor(targetRateHz: number = 10) {
    this.intervalMs = 1000 / targetRateHz;
  }

  public pushIMU(sample: RawSensorSample) {
    // Only accept samples with full data (or convert nulls to 0)
    // The browser might return null if the device doesn't have the sensor
    const cleanSample: RawSensorSample = {
      timestamp: sample.timestamp,
      ax: sample.ax ?? 0,
      ay: sample.ay ?? 0,
      az: sample.az ?? 0,
      gx: sample.gx ?? 0,
      gy: sample.gy ?? 0,
      gz: sample.gz ?? 0,
    };
    
    // Maintain monotonic timestamps
    if (this.imuBuffer.length > 0 && cleanSample.timestamp <= this.imuBuffer[this.imuBuffer.length - 1].timestamp) {
        // Discard out-of-order or duplicate timestamps
        return;
    }
    
    this.imuBuffer.push(cleanSample);
  }

  public pushGNSS(sample: GNSSSample) {
    this.latestGNSS = sample;
  }

  public clearGNSS() {
    this.latestGNSS = null;
  }

  public process(): SensorFrame[] {
    const frames: SensorFrame[] = [];

    // Insufficient data to interpolate
    if (this.imuBuffer.length < 2) {
      return frames;
    }

    // Startup behavior: Set the first emit time
    if (this.nextEmitTime === null) {
      this.nextEmitTime = this.imuBuffer[0].timestamp;
    }

    while (this.imuBuffer.length >= 2) {
      // Check for large gaps
      const first = this.imuBuffer[0];
      const second = this.imuBuffer[1];
      const gap = second.timestamp - first.timestamp;

      if (gap > this.maxGapMs) {
        // Gap detected. Reset emit time to the second sample.
        this.nextEmitTime = second.timestamp;
        this.imuBuffer.shift(); // discard first
        continue;
      }

      if (this.nextEmitTime < first.timestamp) {
         // Next emit time fell behind (should rarely happen), catch up
         this.nextEmitTime = first.timestamp;
      }

      // Can we emit at nextEmitTime?
      if (this.nextEmitTime >= first.timestamp && this.nextEmitTime <= second.timestamp) {
        // Interpolate
        const t1 = first.timestamp;
        const t2 = second.timestamp;
        const fraction = (this.nextEmitTime - t1) / (t2 - t1);

        const frame: SensorFrame = {
          timestamp: this.nextEmitTime,
          ax: first.ax! + (second.ax! - first.ax!) * fraction,
          ay: first.ay! + (second.ay! - first.ay!) * fraction,
          az: first.az! + (second.az! - first.az!) * fraction,
          gx: first.gx! + (second.gx! - first.gx!) * fraction,
          gy: first.gy! + (second.gy! - first.gy!) * fraction,
          gz: first.gz! + (second.gz! - first.gz!) * fraction,
          gnss: this.latestGNSS,
        };

        frames.push(frame);
        this.nextEmitTime += this.intervalMs;

        // Note: we DO NOT shift here, because the next emit time might ALSO fall between first and second
        // (if interval is smaller than the gap between first and second).
      } else if (this.nextEmitTime > second.timestamp) {
        // The next emit time is in the future relative to the first two samples.
        // We can safely discard the first sample.
        this.imuBuffer.shift();
      } else {
        // nextEmitTime < first.timestamp. This is handled by the catch up block above.
        break; 
      }
    }

    return frames;
  }
  
  public getBufferLength(): number {
      return this.imuBuffer.length;
  }
}
