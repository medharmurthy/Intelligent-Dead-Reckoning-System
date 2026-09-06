import { describe, it, expect } from 'vitest';
import { AlignmentEngine } from './AlignmentEngine';

describe('AlignmentEngine', () => {
  it('should correctly determine Z (Pitch/Roll) for perfectly flat phone', () => {
    const engine = new AlignmentEngine();
    
    // Simulate 25 frames of flat phone
    for (let i = 0; i < 25; i++) {
      engine.processFrame({
        timestamp: i * 100,
        ax: 0, ay: 0, az: 9.81,
        gx: 0, gy: 0, gz: 0,
        gnss: null
      });
    }

    const state = engine.getState();
    expect(state.status).toBe('WAITING_FOR_MOTION');
    expect(state.z_v?.[0]).toBeCloseTo(0);
    expect(state.z_v?.[1]).toBeCloseTo(0);
    expect(state.z_v?.[2]).toBeCloseTo(1);
  });

  it('should compute full rotation matrix after motion (flat phone)', () => {
    const engine = new AlignmentEngine();
    
    // 1. Leveling
    for (let i = 0; i < 25; i++) {
      engine.processFrame({
        timestamp: i * 100,
        ax: 0, ay: 0, az: 9.81,
        gx: 0, gy: 0, gz: 0,
        gnss: null
      });
    }

    // 2. Yaw alignment (Driving forward)
    // When driving forward, accelerometer reads +y acceleration if phone is facing forward.
    // GNSS speed increases.
    for (let i = 0; i < 35; i++) {
      engine.processFrame({
        timestamp: 3000 + i * 100,
        ax: 0, ay: 2.0, az: 9.81, // Forward accel
        gx: 0, gy: 0, gz: 0,
        gnss: {
          timestamp: 3000 + i * 100,
          latitude: 0, longitude: 0, accuracy: 5, heading: 0,
          speed: 3.0 + i * 0.1 // Speed increasing
        }
      });
    }

    const state = engine.getState();
    expect(state.status).toBe('ALIGNED');
    
    // For a perfectly flat and forward-facing phone, Rotation matrix should be Identity
    const R = state.rotationMatrix!;
    expect(R[0][0]).toBeCloseTo(1); // X_v.x
    expect(R[1][1]).toBeCloseTo(1); // Y_v.y
    expect(R[2][2]).toBeCloseTo(1); // Z_v.z
  });

  it('should correct a phone rotated +90 yaw (pointing Right)', () => {
    const engine = new AlignmentEngine();
    
    for (let i = 0; i < 25; i++) {
      engine.processFrame({
        timestamp: i * 100,
        ax: 0, ay: 0, az: 9.81,
        gx: 0, gy: 0, gz: 0,
        gnss: null
      });
    }

    // If phone points Right, forward vehicle acceleration is felt on phone's -X axis.
    for (let i = 0; i < 35; i++) {
      engine.processFrame({
        timestamp: 3000 + i * 100,
        ax: -2.0, ay: 0.0, az: 9.81, // Felt on -X
        gx: 0, gy: 0, gz: 0,
        gnss: {
          timestamp: 3000 + i * 100,
          latitude: 0, longitude: 0, accuracy: 5, heading: 0,
          speed: 3.0 + i * 0.1
        }
      });
    }

    const state = engine.getState();
    expect(state.status).toBe('ALIGNED');

    // Test a rotated frame
    const frame = engine.processFrame({
      timestamp: 10000,
      ax: -2.0, ay: 0.0, az: 9.81, // Phone's -X accel
      gx: 0, gy: 0, gz: 0,
      gnss: null
    });

    // Vehicle should perceive this as +Y (forward) acceleration
    expect(frame.ax).toBeCloseTo(0);
    expect(frame.ay).toBeCloseTo(2.0);
    expect(frame.az).toBeCloseTo(9.81);
  });

  it('should correct a phone tilted 90 degrees UP (Screen facing backwards)', () => {
    const engine = new AlignmentEngine();
    
    // If phone is standing straight up, Gravity (+Z vehicle) is felt on phone's -Y axis (bottom of phone).
    for (let i = 0; i < 25; i++) {
      engine.processFrame({
        timestamp: i * 100,
        ax: 0, ay: -9.81, az: 0,
        gx: 0, gy: 0, gz: 0,
        gnss: null
      });
    }

    // Vehicle accelerates forward (+Y vehicle). 
    // This pushes phone backwards, into the screen, so phone feels it on +Z axis.
    for (let i = 0; i < 35; i++) {
      engine.processFrame({
        timestamp: 3000 + i * 100,
        ax: 0, ay: -9.81, az: 2.0,
        gx: 0, gy: 0, gz: 0,
        gnss: {
          timestamp: 3000 + i * 100,
          latitude: 0, longitude: 0, accuracy: 5, heading: 0,
          speed: 3.0 + i * 0.1
        }
      });
    }

    const state = engine.getState();
    expect(state.status).toBe('ALIGNED');

    // Test rotation
    const frame = engine.processFrame({
      timestamp: 10000,
      ax: 0, ay: -9.81, az: 2.0,
      gx: 0, gy: 0, gz: 0,
      gnss: null
    });

    // Should output [0, 2.0, 9.81]
    expect(frame.ay).toBeCloseTo(2.0);
    expect(frame.az).toBeCloseTo(9.81);
  });
});
