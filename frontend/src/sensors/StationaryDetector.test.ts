import { describe, it, expect } from 'vitest';
import { StationaryDetector } from './StationaryDetector';

describe('StationaryDetector', () => {
  it('should detect stationary after 1.5s of low variance', () => {
    const detector = new StationaryDetector();
    
    for (let i = 0; i < 30; i++) {
      detector.processFrame({
        timestamp: i * 100,
        ax: 0.0 + (Math.random() * 0.01), 
        ay: 0.0 + (Math.random() * 0.01), 
        az: 9.81 + (Math.random() * 0.01),
        gx: 0, gy: 0, gz: 0,
        gnss: null
      });
    }

    const state = detector.getState();
    expect(state.isStationary).toBe(true);
    expect(state.status).toBe('STATIONARY');
  });

  it('should reset stationary status upon movement', () => {
    const detector = new StationaryDetector();
    
    // Become stationary
    for (let i = 0; i < 30; i++) {
      detector.processFrame({
        timestamp: i * 100,
        ax: 0, ay: 0, az: 9.81, gx: 0, gy: 0, gz: 0, gnss: null
      });
    }

    expect(detector.getState().isStationary).toBe(true);

    // Sudden movement
    for (let i = 0; i < 5; i++) {
      detector.processFrame({
        timestamp: 3000 + i * 100,
        ax: 10.0, ay: 0, az: 9.81, gx: 0, gy: 0, gz: 0, gnss: null
      });
    }

    const state = detector.getState();
    expect(state.isStationary).toBe(false);
    expect(state.status).toBe('MOVING');
  });

  it('should use GNSS to confirm motion even if variance is low', () => {
    const detector = new StationaryDetector();
    
    // Low variance IMU, but GNSS says moving at 10 m/s
    for (let i = 0; i < 20; i++) {
      detector.processFrame({
        timestamp: i * 100,
        ax: 0, ay: 0, az: 9.81, gx: 0, gy: 0, gz: 0,
        gnss: {
          timestamp: i * 100,
          latitude: 0, longitude: 0, accuracy: 5, heading: 0,
          speed: 10.0 // 36 km/h (cruising, very smooth)
        }
      });
    }

    const state = detector.getState();
    expect(state.isStationary).toBe(false);
    expect(state.status).toBe('MOVING');
  });
});
