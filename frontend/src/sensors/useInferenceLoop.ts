import { useState, useEffect, useRef } from 'react';
import type { SensorFrame } from '../types/sensor';
import { FeatureExtractor } from './FeatureExtractor';
import { ModelPreprocessor } from './ModelPreprocessor';
import { ModelInferenceService } from './ModelInferenceService';
import type { AIOutput } from './ModelInferenceService';

export interface InferenceDiagnostics {
  modelLoaded: boolean;
  modelError: string | null;
  inferenceCount: number;
  lastLatencyMs: number;
  bufferSize: number;
  latestOutput: AIOutput | null;
}

export function useInferenceLoop(latestFrame: SensorFrame | undefined) {
  const [diagnostics, setDiagnostics] = useState<InferenceDiagnostics>({
    modelLoaded: false,
    modelError: null,
    inferenceCount: 0,
    lastLatencyMs: 0,
    bufferSize: 0,
    latestOutput: null
  });

  const extractorRef = useRef(new FeatureExtractor());
  const preprocessorRef = useRef(new ModelPreprocessor());
  const serviceRef = useRef(new ModelInferenceService());
  const isRunningRef = useRef(false);

  useEffect(() => {
    // Initialization: Load ONNX Model and Normalization Params
    const init = async () => {
      try {
        await Promise.all([
          preprocessorRef.current.loadNormalization('/data/normalization_v2.json'),
          serviceRef.current.loadModel('/models/idr_tcn_v2.onnx')
        ]);
        setDiagnostics(prev => ({ ...prev, modelLoaded: true, modelError: null }));
      } catch (err: any) {
        setDiagnostics(prev => ({ ...prev, modelLoaded: false, modelError: err.message }));
      }
    };
    init();
  }, []);

  useEffect(() => {
    if (!latestFrame || !diagnostics.modelLoaded || isRunningRef.current) {
      // Keep track of buffer size even if not running inference
      if (!latestFrame) return;
      // We don't push here if it's already running to avoid concurrency issues, 
      // but actually we must push every frame to maintain the buffer correctly!
    }

    if (!latestFrame) return;

    // 1. Extract features and push to buffer
    const featureFrame = extractorRef.current.extract(latestFrame);
    preprocessorRef.current.push(featureFrame);

    setDiagnostics(prev => ({ ...prev, bufferSize: preprocessorRef.current.getBufferLength() }));

    // 2. Run inference if ready
    if (preprocessorRef.current.isReady() && !isRunningRef.current) {
      isRunningRef.current = true;
      const start = performance.now();
      
      try {
        const tensorData = preprocessorRef.current.getNormalizedTensor();
        
        serviceRef.current.predict(tensorData).then(output => {
          const latency = performance.now() - start;
          setDiagnostics(prev => ({
            ...prev,
            inferenceCount: prev.inferenceCount + 1,
            lastLatencyMs: latency,
            latestOutput: output
          }));
        }).catch(err => {
          console.error("Inference error:", err);
          setDiagnostics(prev => ({ ...prev, modelError: err.message }));
        }).finally(() => {
          isRunningRef.current = false;
        });

      } catch (err: any) {
        console.error("Preprocessing error:", err);
        setDiagnostics(prev => ({ ...prev, modelError: err.message }));
        isRunningRef.current = false;
      }
    }

  }, [latestFrame]);

  return { diagnostics };
}
