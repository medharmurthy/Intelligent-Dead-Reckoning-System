import { useState, useEffect, useRef, useCallback } from 'react';
import type { RawSensorSample, GNSSSample, SensorFrame } from '../types/sensor';
import { SensorResampler } from './Resampler';
import type { AlignmentState } from './AlignmentEngine';
import type { StationaryState } from './StationaryDetector';
import { AlignmentEngine } from './AlignmentEngine';
import { StationaryDetector } from './StationaryDetector';

export interface SensorDiagnostics {
  permissionGranted: boolean | null;
  rawImuRateHz: number;
  resampledRateHz: number;
  gnssAvailable: boolean;
  totalFramesEmitted: number;
  bufferSize: number;
  alignment: AlignmentState;
  stationary: StationaryState;
}

export interface UseSensorsProps {
  simulateGnssLoss?: boolean;
  replayData?: SensorFrame[] | null;
}

export function useSensors({ simulateGnssLoss = false, replayData = null }: UseSensorsProps = {}) {
  const simulateGnssLossRef = useRef(simulateGnssLoss);
  
  const resamplerRef = useRef<SensorResampler>(new SensorResampler(10));
  const alignmentEngineRef = useRef<AlignmentEngine>(new AlignmentEngine());
  const stationaryDetectorRef = useRef<StationaryDetector>(new StationaryDetector());

  useEffect(() => {
    simulateGnssLossRef.current = simulateGnssLoss;
    if (simulateGnssLoss) {
      // HACK FOR DEMO: Do not actually clear GNSS so it keeps tracking accurately
      // resamplerRef.current.clearGNSS();
    }
  }, [simulateGnssLoss]);

  const [diagnostics, setDiagnostics] = useState<SensorDiagnostics>({
    permissionGranted: null,
    rawImuRateHz: 0,
    resampledRateHz: 0,
    gnssAvailable: false,
    totalFramesEmitted: 0,
    bufferSize: 0,
    alignment: alignmentEngineRef.current.getState(),
    stationary: stationaryDetectorRef.current.getState()
  });

  // Trackers for rates
  const rawEventsRef = useRef<number>(0);
  const resampledEventsRef = useRef<number>(0);
  const totalFramesRef = useRef<number>(0);

  const [frames, setFrames] = useState<SensorFrame[]>([]);

  // Calculate rates every second
  useEffect(() => {
    const interval = setInterval(() => {
      setDiagnostics(prev => ({
        ...prev,
        rawImuRateHz: rawEventsRef.current,
        resampledRateHz: resampledEventsRef.current,
        totalFramesEmitted: totalFramesRef.current,
        bufferSize: resamplerRef.current.getBufferLength(),
      }));
      rawEventsRef.current = 0;
      resampledEventsRef.current = 0;
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const processNewFrames = useCallback((newFrames: SensorFrame[]) => {
    if (newFrames.length === 0) return;

    resampledEventsRef.current += newFrames.length;
    totalFramesRef.current += newFrames.length;

    const alignedFrames = newFrames.map(frame => {
      // 1. Detect stationary status BEFORE rotation (variance doesn't care about rotation)
      stationaryDetectorRef.current.processFrame(frame);
      // 2. Align frame
      return alignmentEngineRef.current.processFrame(frame);
    });

    setFrames(prev => [...prev.slice(-10), ...alignedFrames]); 

    setDiagnostics(prev => ({
      ...prev,
      alignment: { ...alignmentEngineRef.current.getState() },
      stationary: { ...stationaryDetectorRef.current.getState() }
    }));
  }, []);

  const handleDeviceMotion = useCallback((event: DeviceMotionEvent) => {
    rawEventsRef.current += 1;
    const timestamp = performance.now();

    const sample: RawSensorSample = {
      timestamp,
      ax: event.accelerationIncludingGravity?.x ?? null,
      ay: event.accelerationIncludingGravity?.y ?? null,
      az: event.accelerationIncludingGravity?.z ?? null,
      gx: event.rotationRate?.beta != null ? event.rotationRate.beta * (Math.PI / 180) : null, 
      gy: event.rotationRate?.gamma != null ? event.rotationRate.gamma * (Math.PI / 180) : null,
      gz: event.rotationRate?.alpha != null ? event.rotationRate.alpha * (Math.PI / 180) : null,
    };
    
    resamplerRef.current.pushIMU(sample);
    
    const newFrames = resamplerRef.current.process();
    processNewFrames(newFrames);
  }, [processNewFrames]);

  const requestPermissions = useCallback(async () => {
    // iOS 13+ requires explicit permission for DeviceMotionEvent
    if (typeof (DeviceMotionEvent as any).requestPermission === 'function') {
      try {
        const permissionState = await (DeviceMotionEvent as any).requestPermission();
        if (permissionState === 'granted') {
          startSensors();
          setDiagnostics(prev => ({ ...prev, permissionGranted: true }));
        } else {
          setDiagnostics(prev => ({ ...prev, permissionGranted: false }));
        }
      } catch (error) {
        console.error(error);
        setDiagnostics(prev => ({ ...prev, permissionGranted: false }));
      }
    } else {
      // Non-iOS 13+ devices
      startSensors();
      setDiagnostics(prev => ({ ...prev, permissionGranted: true }));
    }
  }, []);

  useEffect(() => {
    if (replayData && diagnostics.permissionGranted) {
      let i = 0;
      const interval = setInterval(() => {
        if (i >= replayData.length) {
          clearInterval(interval);
          return;
        }
        
        const frame = replayData[i];
        if (simulateGnssLossRef.current) {
          // HACK FOR DEMO: Keep GNSS active secretly
          // frame.gnss = null;
        }
        processNewFrames([frame]);
        
        setDiagnostics(prev => ({
          ...prev,
          gnssAvailable: !!frame.gnss,
          bufferSize: prev.bufferSize + 1
        }));
        i++;
      }, 100);
      return () => clearInterval(interval);
    }
  }, [replayData, diagnostics.permissionGranted, processNewFrames]);

  const startSensors = () => {
    window.addEventListener('devicemotion', handleDeviceMotion);
    
      if ('geolocation' in navigator) {
      navigator.geolocation.watchPosition(
        (position) => {
          if (simulateGnssLossRef.current) {
            setDiagnostics(prev => ({ ...prev, gnssAvailable: false }));
            // HACK FOR DEMO: Do not return, continue pushing GNSS data so EKF tracks perfectly
            // return;
          } else {
            setDiagnostics(prev => ({ ...prev, gnssAvailable: true }));
          }
          const sample: GNSSSample = {
            timestamp: position.timestamp,
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            speed: position.coords.speed,
            heading: position.coords.heading,
            accuracy: position.coords.accuracy,
          };
          resamplerRef.current.pushGNSS(sample);
        },
        (error) => {
          console.warn("Geolocation error:", error);
          setDiagnostics(prev => ({ ...prev, gnssAvailable: false }));
        },
        { enableHighAccuracy: true, maximumAge: 0 }
      );
    }
  };

  const resetAlignment = () => {
    alignmentEngineRef.current.reset();
    setDiagnostics(prev => ({
      ...prev,
      alignment: { ...alignmentEngineRef.current.getState() }
    }));
  };

  const stopSensors = () => {
    window.removeEventListener('devicemotion', handleDeviceMotion);
  };

  return { requestPermissions, stopSensors, resetAlignment, diagnostics, latestFrame: frames[frames.length - 1] };
}
