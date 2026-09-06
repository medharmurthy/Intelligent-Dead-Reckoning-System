import math
import numpy as np

class EKF2D:
    """
    2D Extended Kalman Filter (EKF) for Vehicle Inertial Dead Reckoning.
    State Vector x (8 dimensions):
      x[0]: p_x    - Position East/X (m)
      x[1]: p_y    - Position North/Y (m)
      x[2]: v_x    - Velocity X (m/s)
      x[3]: v_y    - Velocity Y (m/s)
      x[4]: psi    - Vehicle Yaw / Heading angle (rad)
      x[5]: b_g    - Gyroscope Bias (rad/s)
      x[6]: b_ax   - Accelerometer X Bias (m/s^2)
      x[7]: b_ay   - Accelerometer Y Bias (m/s^2)
    """
    def __init__(self, init_x=0.0, init_y=0.0, init_heading=0.0):
        self.dim_x = 8
        self.x = np.zeros((8, 1), dtype=np.float64)
        self.x[0, 0] = init_x
        self.x[1, 0] = init_y
        self.x[4, 0] = init_heading
        
        # State Covariance Matrix P (8x8)
        self.P = np.eye(8, dtype=np.float64) * 0.1
        self.P[0, 0] = 1.0; self.P[1, 1] = 1.0 # Initial position uncertainty
        self.P[4, 4] = 0.05                    # Initial heading uncertainty
        
        # Process Noise Covariance Matrix Q (8x8)
        self.Q = np.eye(8, dtype=np.float64) * 1e-4
        self.Q[0, 0] = 1e-3; self.Q[1, 1] = 1e-3
        self.Q[2, 2] = 1e-2; self.Q[3, 3] = 1e-2
        self.Q[4, 4] = 1e-3

    def predict(self, dt, ax_raw, ay_raw, gz_raw):
        """
        IMU State Propagation Step (INS Prediction):
        Correct raw IMU measurements using estimated biases and integrate motion equations.
        """
        if dt <= 0:
            return
            
        psi = self.x[4, 0]
        b_g = self.x[5, 0]
        b_ax = self.x[6, 0]
        b_ay = self.x[7, 0]
        
        # Corrected sensor values
        ax = ax_raw - b_ax
        ay = ay_raw - b_ay
        gz = gz_raw - b_g
        
        # Transform accelerations from vehicle frame to navigation frame (ENU)
        cos_p = math.cos(psi)
        sin_p = math.sin(psi)
        
        a_nav_x = ax * cos_p - ay * sin_p
        a_nav_y = ax * sin_p + ay * cos_p
        
        # Update State Vector x_k = f(x_{k-1}, u)
        self.x[0, 0] += self.x[2, 0] * dt + 0.5 * a_nav_x * (dt ** 2)
        self.x[1, 0] += self.x[3, 0] * dt + 0.5 * a_nav_y * (dt ** 2)
        self.x[2, 0] += a_nav_x * dt
        self.x[3, 0] += a_nav_y * dt
        self.x[4, 0] = (self.x[4, 0] + gz * dt + math.pi) % (2 * math.pi) - math.pi # Wrap to [-pi, pi]
        
        # Jacobian Matrix F_k = df/dx
        F = np.eye(8, dtype=np.float64)
        F[0, 2] = dt; F[1, 3] = dt
        F[0, 4] = -0.5 * (dt ** 2) * (ax * sin_p + ay * cos_p)
        F[1, 4] = 0.5 * (dt ** 2) * (ax * cos_p - ay * sin_p)
        F[2, 4] = -dt * (ax * sin_p + ay * cos_p)
        F[3, 4] = dt * (ax * cos_p - ay * sin_p)
        
        F[4, 5] = -dt
        F[2, 6] = -dt * cos_p; F[2, 7] = dt * sin_p
        F[3, 6] = -dt * sin_p; F[3, 7] = -dt * cos_p
        
        # Propagate Covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q * dt

    def update_ai_velocity(self, v_forward_ai, variance_ai=0.2):
        """
        AI Measurement Update Step (Section 45 of plan.pdf):
        Measurement z_AI = v_forward_ai
        Observation equation: h(x) = sqrt(v_x^2 + v_y^2) (or forward projection in vehicle frame)
        Measurement variance R_AI = max(variance_ai, 1e-4)
        """
        psi = self.x[4, 0]
        vx = self.x[2, 0]
        vy = self.x[3, 0]
        
        # Velocity in vehicle forward frame
        v_forward_est = vx * math.cos(psi) + vy * math.sin(psi)
        
        # Measurement matrix H (1x8)
        H = np.zeros((1, 8), dtype=np.float64)
        H[0, 2] = math.cos(psi)
        H[0, 3] = math.sin(psi)
        H[0, 4] = -vx * math.sin(psi) + vy * math.cos(psi)
        
        # Innovation residual r = z - h(x)
        r = np.array([[v_forward_ai - v_forward_est]], dtype=np.float64)
        
        R = np.array([[max(variance_ai, 1e-4)]], dtype=np.float64)
        
        # Kalman Gain K = P * H^T * (H * P * H^T + R)^-1
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update State & Covariance
        self.x = self.x + K @ r
        self.x[4, 0] = (self.x[4, 0] + math.pi) % (2 * math.pi) - math.pi
        I = np.eye(8, dtype=np.float64)
        self.P = (I - K @ H) @ self.P
        return float(r[0, 0]), float(S[0, 0])

    def update_nhc(self, R_nhc=0.01):
        """
        Non-Holonomic Constraint (NHC) Pseudo-Measurement Update (Section 50 of plan.pdf):
        For road vehicles: lateral velocity v_y_vehicle approx 0.
        """
        psi = self.x[4, 0]
        vx = self.x[2, 0]
        vy = self.x[3, 0]
        
        v_lat_est = -vx * math.sin(psi) + vy * math.cos(psi)
        
        H = np.zeros((1, 8), dtype=np.float64)
        H[0, 2] = -math.sin(psi)
        H[0, 3] = math.cos(psi)
        H[0, 4] = -vx * math.cos(psi) - vy * math.sin(psi)
        
        r = np.array([[0.0 - v_lat_est]], dtype=np.float64)
        R = np.array([[R_nhc]], dtype=np.float64)
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ r
        self.x[4, 0] = (self.x[4, 0] + math.pi) % (2 * math.pi) - math.pi
        I = np.eye(8, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

    def update_gnss(self, px_gnss, py_gnss, vx_gnss, vy_gnss, pos_std=2.0, vel_std=0.5):
        """
        GNSS Measurement Update (Section 46 of plan.pdf):
        Measurement z_GNSS = [p_x, p_y, v_x, v_y]^T
        """
        z = np.array([[px_gnss], [py_gnss], [vx_gnss], [vy_gnss]], dtype=np.float64)
        h = self.x[:4, :] # Estimated position and velocity
        
        H = np.zeros((4, 8), dtype=np.float64)
        H[0, 0] = 1.0; H[1, 1] = 1.0
        H[2, 2] = 1.0; H[3, 3] = 1.0
        
        r = z - h
        
        R = np.diag([pos_std**2, pos_std**2, vel_std**2, vel_std**2]).astype(np.float64)
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ r
        self.x[4, 0] = (self.x[4, 0] + math.pi) % (2 * math.pi) - math.pi
        I = np.eye(8, dtype=np.float64)
        self.P = (I - K @ H) @ self.P
        
        # Calculate Normalized Innovation Squared (NIS)
        nis = float(r.T @ np.linalg.inv(S) @ r)
        return nis, r

    @property
    def speed(self):
        return float(math.sqrt(self.x[2, 0]**2 + self.x[3, 0]**2))
        
    @property
    def heading_deg(self):
        heading_rad = self.x[4, 0]
        deg = math.degrees(heading_rad)
        return float((deg + 360) % 360)
