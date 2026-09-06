import type { FeatureFrame } from './FeatureExtractor';

export interface NormalizationParams {
  channel_names: string[];
  mean: number[];
  std: number[];
  window_size: number;
  sampling_rate_hz: number;
}

export class ModelPreprocessor {
  private buffer: FeatureFrame[] = [];
  private readonly WINDOW_SIZE = 20;
  private readonly NUM_CHANNELS = 9;
  private normParams: NormalizationParams | null = null;

  constructor(normParams?: NormalizationParams) {
    if (normParams) {
      this.normParams = normParams;
    }
  }

  public async loadNormalization(url: string = '/data/normalization.json') {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load normalization params from ${url}`);
    }
    this.normParams = await response.json();
  }

  public push(frame: FeatureFrame) {
    this.buffer.push(frame);
    if (this.buffer.length > this.WINDOW_SIZE) {
      this.buffer.shift();
    }
  }

  public isReady(): boolean {
    return this.buffer.length === this.WINDOW_SIZE && this.normParams !== null;
  }

  /**
   * Produces a flattened Float32Array representing [1, 9, 20] tensor.
   * Format: Channel major. [Channel 0 (20 frames), Channel 1 (20 frames), ...]
   */
  public getNormalizedTensor(): Float32Array {
    if (!this.isReady() || !this.normParams) {
      throw new Error("Preprocessor is not ready. Buffer size: " + this.buffer.length);
    }

    const { mean, std } = this.normParams;
    const tensor = new Float32Array(this.NUM_CHANNELS * this.WINDOW_SIZE);

    // Channel-major loop to build [1, 9, 20]
    for (let c = 0; c < this.NUM_CHANNELS; c++) {
      const c_mean = mean[c];
      const c_std = std[c] === 0 ? 1e-6 : std[c]; // Avoid division by zero

      for (let t = 0; t < this.WINDOW_SIZE; t++) {
        const raw_val = this.buffer[t].channels[c];
        const norm_val = (raw_val - c_mean) / c_std;
        
        // Clamp to [-5.0, 5.0] to protect the model from out-of-distribution bicycle vibrations
        const clamped_val = Math.max(-5.0, Math.min(5.0, norm_val));
        
        // Index calculation for shape [1, 9, 20] -> index = c * 20 + t
        tensor[c * this.WINDOW_SIZE + t] = clamped_val;
      }
    }

    return tensor;
  }
  
  public getBufferLength(): number {
      return this.buffer.length;
  }
}
