import { useState, useRef, useEffect } from 'react';
import { NavigationEngine, type NavigationState } from '../navigation/NavigationEngine';
import type { SensorFrame } from '../types/sensor';
import type { AIOutput } from '../sensors/ModelInferenceService';

export function useNavigation(latestFrame: SensorFrame | undefined, latestAiOutput: AIOutput | null, isStationary: boolean) {
  const [navState, setNavState] = useState<NavigationState | null>(null);
  const [history, setHistory] = useState<NavigationState[]>([]);
  
  const engineRef = useRef(new NavigationEngine());

  useEffect(() => {
    if (!latestFrame) return;

    // Process the frame through the engine
    const newState = engineRef.current.process(latestFrame, latestAiOutput, isStationary);
    if (!newState) return;
    
    setNavState(newState);
    setHistory(prev => {
      const newHistory = [...prev, newState];
      // Keep last 500 points for trajectory (~50 seconds at 10Hz)
      if (newHistory.length > 500) {
        return newHistory.slice(newHistory.length - 500);
      }
      return newHistory;
    });
  }, [latestFrame, latestAiOutput]);

  return { navState, history };
}
