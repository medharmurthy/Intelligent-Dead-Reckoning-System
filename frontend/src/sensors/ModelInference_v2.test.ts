import { describe, it, expect, beforeAll } from 'vitest';
import { ModelPreprocessor } from './ModelPreprocessor';
import normData from '../../public/data/normalization_v2.json';

describe('V2 Model Integration Test', () => {
  let preprocessor: ModelPreprocessor;

  beforeAll(async () => {
    preprocessor = new ModelPreprocessor(normData);
  });

  it('Preprocessor generates correctly shaped [1, 9, 20] tensor using V2 config', () => {
    // Push 20 synthetic frames
    for (let i = 0; i < 20; i++) {
      preprocessor.push({
        timestamp: i * 100,
        channels: [0, 0, 9.8, 0, 0, 0, 9.8, 0, 0] // Stationary-like data
      });
    }

    expect(preprocessor.isReady()).toBe(true);

    const tensor = preprocessor.getNormalizedTensor();
    expect(tensor.length).toBe(9 * 20); // 180 floats
    
    // Check specific clamping / z-score normalization logic
    // accel_mag (index 6) = 9.8
    // v2 mean[6] = 9.9604, std[6] = 0.5227
    // expected = (9.8 - 9.9604) / 0.5227 = -0.3068
    const accel_mag_index = 6 * 20 + 0; // Channel 6, frame 0
    expect(tensor[accel_mag_index]).toBeCloseTo(-0.3068, 2);
  });
});
