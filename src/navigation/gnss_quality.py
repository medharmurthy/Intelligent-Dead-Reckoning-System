import time

class NavigationStateMode:
    GNSS_AIDED = "GNSS_AIDED"
    DEGRADED = "DEGRADED"
    DEAD_RECKONING = "DEAD_RECKONING"
    REACQUISITION = "REACQUISITION"

class GNSSQualityMonitor:
    """
    Monitors GNSS Signal Quality and manages the Navigation State Machine (Sections 47-49 of plan.pdf).
    """
    def __init__(self, nis_threshold=15.0, timeout_sec=2.0):
        self.mode = NavigationStateMode.GNSS_AIDED
        self.gnss_quality = 1.0  # Quality metric in [0.0, 1.0]
        self.nis_threshold = nis_threshold
        self.timeout_sec = timeout_sec
        self.last_gnss_time = None
        self.consecutive_bad_nis = 0
        self.consecutive_good_gnss = 0

    def update(self, current_timestamp, gnss_sample=None, nis=None):
        """
        Updates GNSS quality state and evaluates Navigation State Machine transitions.
        
        Args:
          current_timestamp: float timestamp in seconds
          gnss_sample: dict or None containing lat, lon, speed, accuracy
          nis: Normalized Innovation Squared metric from EKF GNSS update
        """
        has_signal = gnss_sample is not None and gnss_sample.get("accuracy", 999) < 50.0
        
        if has_signal:
            self.last_gnss_time = current_timestamp
            acc = gnss_sample.get("accuracy", 5.0)
            
            # Base quality derived from reported horizontal accuracy (m)
            base_q = max(0.0, min(1.0, 1.0 - (acc - 2.0) / 20.0))
            
            # NIS anomaly penalty
            if nis is not None and nis > self.nis_threshold:
                self.consecutive_bad_nis += 1
                self.consecutive_good_gnss = 0
                q_penalty = min(0.5, 0.1 * self.consecutive_bad_nis)
            else:
                self.consecutive_bad_nis = 0
                self.consecutive_good_gnss += 1
                q_penalty = 0.0
                
            self.gnss_quality = max(0.0, base_q - q_penalty)
        else:
            self.consecutive_good_gnss = 0
            # Decay quality on missing signal
            if self.last_gnss_time is None or (current_timestamp - self.last_gnss_time) > self.timeout_sec:
                self.gnss_quality = 0.0
            else:
                self.gnss_quality = max(0.0, self.gnss_quality - 0.2)

        # State Machine Transitions (Section 49 of plan.pdf)
        if self.mode == NavigationStateMode.GNSS_AIDED:
            if self.gnss_quality == 0.0:
                self.mode = NavigationStateMode.DEAD_RECKONING
            elif self.gnss_quality < 0.5:
                self.mode = NavigationStateMode.DEGRADED

        elif self.mode == NavigationStateMode.DEGRADED:
            if self.gnss_quality == 0.0:
                self.mode = NavigationStateMode.DEAD_RECKONING
            elif self.gnss_quality >= 0.7:
                self.mode = NavigationStateMode.GNSS_AIDED

        elif self.mode == NavigationStateMode.DEAD_RECKONING:
            if has_signal and self.gnss_quality > 0.3:
                self.mode = NavigationStateMode.REACQUISITION

        elif self.mode == NavigationStateMode.REACQUISITION:
            if self.consecutive_good_gnss >= 3 and self.gnss_quality >= 0.7:
                self.mode = NavigationStateMode.GNSS_AIDED
            elif not has_signal:
                self.mode = NavigationStateMode.DEAD_RECKONING

        return self.mode, self.gnss_quality
