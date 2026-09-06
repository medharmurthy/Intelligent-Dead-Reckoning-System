import { describe, it, expect, beforeEach } from 'vitest';
import { SensorResampler } from './Resampler';
import type { GNSSSample } from '../types/sensor';

describe('SensorResampler', () => {
  let resampler: SensorResampler;

  beforeEach(() => {
    resampler = new SensorResampler(10); // 10Hz target
  });

  it('should interpolate samples correctly at 10Hz (100ms interval)', () => {
    // 0ms
    resampler.pushIMU({ timestamp: 0, ax: 0, ay: 0, az: 0, gx: 0, gy: 0, gz: 0 });
    // 150ms
    resampler.pushIMU({ timestamp: 150, ax: 15, ay: 30, az: 45, gx: 0, gy: 0, gz: 0 });
    
    const frames = resampler.process();
    // It should emit at 0ms and 100ms
    expect(frames.length).toBe(2);
    expect(frames[0].timestamp).toBe(0);
    expect(frames[1].timestamp).toBe(100);
    expect(frames[1].ax).toBeCloseTo(10);
    
    // Now push 250ms
    resampler.pushIMU({ timestamp: 250, ax: 25, ay: 50, az: 75, gx: 0, gy: 0, gz: 0 });
    const frames2 = resampler.process();
    
    // We should get the 200ms frame
    expect(frames2.length).toBe(1);
    expect(frames2[0].timestamp).toBe(200);
    expect(frames2[0].ax).toBeCloseTo(20);
  });

  it('should handle large gaps by resetting the next emit time', () => {
    resampler.pushIMU({ timestamp: 0, ax: 0, ay: 0, az: 0, gx: 0, gy: 0, gz: 0 });
    resampler.pushIMU({ timestamp: 600, ax: 10, ay: 10, az: 10, gx: 0, gy: 0, gz: 0 }); // > 500ms gap
    
    const frames = resampler.process();
    
    // Gap detected, first sample discarded, 600ms becomes the new start
    expect(frames.length).toBe(0);
    
    resampler.pushIMU({ timestamp: 650, ax: 15, ay: 15, az: 15, gx: 0, gy: 0, gz: 0 });
    const frames2 = resampler.process();
    
    // The resampler should emit at 600ms now
    expect(frames2.length).toBe(1);
    expect(frames2[0].timestamp).toBe(600);
  });

  it('should pass through the latest GNSS sample', () => {
    const gnss: GNSSSample = {
      timestamp: 50,
      latitude: 10,
      longitude: 20,
      speed: 5,
      heading: 90,
      accuracy: 2,
    };
    resampler.pushGNSS(gnss);
    
    resampler.pushIMU({ timestamp: 0, ax: 0, ay: 0, az: 0, gx: 0, gy: 0, gz: 0 });
    resampler.pushIMU({ timestamp: 100, ax: 10, ay: 10, az: 10, gx: 0, gy: 0, gz: 0 });
    
    const frames = resampler.process();
    expect(frames.length).toBe(2); // 0ms and 100ms
    expect(frames[0].gnss).toEqual(gnss);
    expect(frames[1].gnss).toEqual(gnss);
  });
});

