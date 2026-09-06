import type { GNSSSample } from '../types/sensor';

export const NavigationStateMode = {
  GNSS_AIDED: 'GNSS_AIDED',
  DEGRADED: 'DEGRADED',
  DEAD_RECKONING: 'DEAD_RECKONING',
  REACQUISITION: 'REACQUISITION'
} as const;

export type NavigationStateModeType = typeof NavigationStateMode[keyof typeof NavigationStateMode];

export class GNSSQualityMonitor {
  public mode: NavigationStateModeType = NavigationStateMode.GNSS_AIDED;
  public gnssQuality: number = 1.0;
  
  private nisThreshold: number;
  private timeoutSec: number;
  private lastGnssTime: number | null = null;
  private consecutiveBadNis: number = 0;
  private consecutiveGoodGnss: number = 0;

  constructor(nisThreshold: number = 15.0, timeoutSec: number = 2.0) {
    this.nisThreshold = nisThreshold;
    this.timeoutSec = timeoutSec;
  }

  public update(currentTimestampSec: number, gnssSample: GNSSSample | null, nis: number | null): { mode: NavigationStateModeType, gnssQuality: number } {
    const hasSignal = gnssSample !== null && (gnssSample.accuracy !== null && gnssSample.accuracy < 50.0);
    
    if (hasSignal && gnssSample) {
      this.lastGnssTime = currentTimestampSec;
      const acc = gnssSample.accuracy !== null ? gnssSample.accuracy : 5.0;
      
      const baseQ = Math.max(0.0, Math.min(1.0, 1.0 - (acc - 2.0) / 20.0));
      let qPenalty = 0.0;

      if (nis !== null && nis > this.nisThreshold) {
        this.consecutiveBadNis += 1;
        this.consecutiveGoodGnss = 0;
        qPenalty = Math.min(0.5, 0.1 * this.consecutiveBadNis);
      } else {
        this.consecutiveBadNis = 0;
        this.consecutiveGoodGnss += 1;
        qPenalty = 0.0;
      }
      
      this.gnssQuality = Math.max(0.0, baseQ - qPenalty);
    } else {
      this.consecutiveGoodGnss = 0;
      if (this.lastGnssTime === null || (currentTimestampSec - this.lastGnssTime) > this.timeoutSec) {
        this.gnssQuality = 0.0;
      } else {
        this.gnssQuality = Math.max(0.0, this.gnssQuality - 0.2);
      }
    }

    if (this.mode === NavigationStateMode.GNSS_AIDED) {
      if (this.gnssQuality === 0.0) {
        this.mode = NavigationStateMode.DEAD_RECKONING;
      } else if (this.gnssQuality < 0.5) {
        this.mode = NavigationStateMode.DEGRADED;
      }
    } else if (this.mode === NavigationStateMode.DEGRADED) {
      if (this.gnssQuality === 0.0) {
        this.mode = NavigationStateMode.DEAD_RECKONING;
      } else if (this.gnssQuality >= 0.7) {
        this.mode = NavigationStateMode.GNSS_AIDED;
      }
    } else if (this.mode === NavigationStateMode.DEAD_RECKONING) {
      if (hasSignal && this.gnssQuality > 0.3) {
        this.mode = NavigationStateMode.REACQUISITION;
      }
    } else if (this.mode === NavigationStateMode.REACQUISITION) {
      if (this.consecutiveGoodGnss >= 3 && this.gnssQuality >= 0.7) {
        this.mode = NavigationStateMode.GNSS_AIDED;
      } else if (!hasSignal) {
        this.mode = NavigationStateMode.DEAD_RECKONING;
      }
    }

    return { mode: this.mode, gnssQuality: this.gnssQuality };
  }
}
