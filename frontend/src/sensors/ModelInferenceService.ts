import * as ort from 'onnxruntime-web';

// Explicitly tell ONNX where to find the .wasm binaries in the public root
ort.env.wasm.wasmPaths = '/';

// Determine the motion class labels from python code or standard conventions
export const MotionClass = {
  STATIONARY: 0,
  ACCELERATING: 1,
  CRUISING: 2,
  BRAKING: 3,
  TURNING: 4
} as const;

export type MotionClassType = typeof MotionClass[keyof typeof MotionClass];

export interface AIOutput {
  speed_mps: number;
  variance: number; // e^(log_var)
  motionClassId: number;
  motionConfidence: number;
}

export class ModelInferenceService {
  private session: ort.InferenceSession | null = null;
  public isLoaded = false;

  public async loadModel(url: string = '/models/idr_tcn.onnx') {
    // Configure execution providers to use webassembly (wasm)
    ort.env.wasm.wasmPaths = '/';
    
    try {
      this.session = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
      this.isLoaded = true;
    } catch (err) {
      console.error("Failed to load ONNX model:", err);
      throw err;
    }
  }

  public async predict(inputTensorData: Float32Array): Promise<AIOutput> {
    if (!this.session) {
      throw new Error("Model not loaded");
    }

    // Prepare input tensor: [batch_size=1, channels=9, window_size=20]
    const tensor = new ort.Tensor('float32', inputTensorData, [1, 9, 20]);
    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.session.inputNames[0]] = tensor; // 'imu_window'

    // Run inference
    const results = await this.session.run(feeds);

    // Outputs expected: "speed", "log_var", "motion_logits"
    const speedTensor = results['speed'];
    const logVarTensor = results['log_var'];
    const motionTensor = results['motion_logits'];

    const speed_mps = speedTensor.data[0] as number;
    const log_var = logVarTensor.data[0] as number;
    const variance = Math.exp(log_var);

    // Parse motion logits [1, 5]
    const logits = motionTensor.data as Float32Array;
    let maxIdx = 0;
    let maxVal = logits[0];
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > maxVal) {
        maxVal = logits[i];
        maxIdx = i;
      }
    }

    // Softmax for confidence
    let sumExp = 0;
    for (let i = 0; i < logits.length; i++) {
      sumExp += Math.exp(logits[i]);
    }
    const motionConfidence = Math.exp(maxVal) / sumExp;

    return {
      speed_mps,
      variance,
      motionClassId: maxIdx,
      motionConfidence
    };
  }
}
