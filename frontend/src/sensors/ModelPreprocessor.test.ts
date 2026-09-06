import { describe, it, expect, beforeAll } from 'vitest';
import { ModelPreprocessor } from './ModelPreprocessor';
import type { FeatureFrame } from './FeatureExtractor';
import testData from '../../public/data/test_data.json';
import normData from '../../public/data/normalization.json';

describe('ModelPreprocessor Deterministic Validation', () => {
  beforeAll(() => {
    // Data is already imported
  });

  it('should perfectly match python normalization and tensor shape [1, 9, 20]', () => {
    const preprocessor = new ModelPreprocessor(normData);
    
    // raw_window is [20, 9] in python
    const rawWindow = testData.raw_window;
    
    for (let i = 0; i < 20; i++) {
      const frame: FeatureFrame = {
        timestamp: i * 100,
        channels: rawWindow[i] as [number, number, number, number, number, number, number, number, number]
      };
      preprocessor.push(frame);
    }
    
    expect(preprocessor.isReady()).toBe(true);
    
    const tensor = preprocessor.getNormalizedTensor();
    const expectedTensor = testData.expected_tensor;
    
    expect(tensor.length).toBe(expectedTensor.length);
    
    // Check numerical equality within a small epsilon due to Float32 precision
    for (let i = 0; i < tensor.length; i++) {
      expect(tensor[i]).toBeCloseTo(expectedTensor[i], 5);
    }
  });
});
