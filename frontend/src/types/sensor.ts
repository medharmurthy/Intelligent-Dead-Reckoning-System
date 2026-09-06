export interface RawSensorSample {
  timestamp: number; // in milliseconds
  ax: number | null;
  ay: number | null;
  az: number | null;
  gx: number | null;
  gy: number | null;
  gz: number | null;
}

export interface GNSSSample {
  timestamp: number; // in milliseconds
  latitude: number | null;
  longitude: number | null;
  speed: number | null;
  heading: number | null;
  accuracy: number | null;
}

export interface SensorFrame {
  timestamp: number;
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
  gnss: GNSSSample | null;
}
