import { Matrix, inverse } from 'ml-matrix';

export class EKF2D {
  public x: Matrix;
  public P: Matrix;
  public Q: Matrix;

  constructor(initX: number = 0.0, initY: number = 0.0, initHeading: number = 0.0) {
    this.x = Matrix.zeros(8, 1);
    this.x.set(0, 0, initX);
    this.x.set(1, 0, initY);
    this.x.set(4, 0, initHeading);

    // Initial Covariance P
    this.P = Matrix.eye(8).mul(0.1);
    this.P.set(0, 0, 1.0);
    this.P.set(1, 1, 1.0);
    this.P.set(4, 4, 0.05);

    // Process Noise Q
    this.Q = Matrix.eye(8).mul(1e-4);
    this.Q.set(0, 0, 1e-3);
    this.Q.set(1, 1, 1e-3);
    this.Q.set(2, 2, 1e-2);
    this.Q.set(3, 3, 1e-2);
    this.Q.set(4, 4, 1e-3);
  }

  public predict(dt: number, ax_raw: number, ay_raw: number, gz_raw: number) {
    if (dt <= 0) return;

    const psi = this.x.get(4, 0);
    const b_g = this.x.get(5, 0);
    const b_ax = this.x.get(6, 0);
    const b_ay = this.x.get(7, 0);

    const ax = ax_raw - b_ax;
    const ay = ay_raw - b_ay;
    const gz = gz_raw - b_g;

    const cos_p = Math.cos(psi);
    const sin_p = Math.sin(psi);

    // Vehicle X is Right (psi - 90 deg), Vehicle Y is Forward (psi)
    const a_nav_x = ax * sin_p + ay * cos_p;
    const a_nav_y = -ax * cos_p + ay * sin_p;

    const vx = this.x.get(2, 0);
    const vy = this.x.get(3, 0);

    // Update state vector
    this.x.set(0, 0, this.x.get(0, 0) + vx * dt + 0.5 * a_nav_x * dt * dt);
    this.x.set(1, 0, this.x.get(1, 0) + vy * dt + 0.5 * a_nav_y * dt * dt);
    this.x.set(2, 0, vx + a_nav_x * dt);
    this.x.set(3, 0, vy + a_nav_y * dt);

    let new_psi = (psi + gz * dt + Math.PI) % (2 * Math.PI);
    if (new_psi < 0) new_psi += 2 * Math.PI;
    new_psi -= Math.PI;
    this.x.set(4, 0, new_psi);

    // Jacobian F
    const F = Matrix.eye(8);
    F.set(0, 2, dt);
    F.set(1, 3, dt);
    
    const d_anav_x_dpsi = ax * cos_p - ay * sin_p;
    const d_anav_y_dpsi = ax * sin_p + ay * cos_p;
    
    F.set(0, 4, 0.5 * dt * dt * d_anav_x_dpsi);
    F.set(1, 4, 0.5 * dt * dt * d_anav_y_dpsi);
    F.set(2, 4, dt * d_anav_x_dpsi);
    F.set(3, 4, dt * d_anav_y_dpsi);

    F.set(4, 5, -dt);
    // dvx / db_ax = -dt * sin(psi)
    F.set(2, 6, -dt * sin_p);
    // dvx / db_ay = -dt * cos(psi)
    F.set(2, 7, -dt * cos_p);
    // dvy / db_ax = dt * cos(psi)
    F.set(3, 6, dt * cos_p);
    // dvy / db_ay = -dt * sin(psi)
    F.set(3, 7, -dt * sin_p);

    // P = F * P * F^T + Q * dt
    const F_P_FT = F.mmul(this.P).mmul(F.transpose());
    this.P = F_P_FT.add(this.Q.clone().mul(dt));
  }

  public updateAIVelocity(v_forward_ai: number, variance_ai: number = 0.2): { r: number; S: number } {
    const psi = this.x.get(4, 0);
    const vx = this.x.get(2, 0);
    const vy = this.x.get(3, 0);

    const v_forward_est = vx * Math.cos(psi) + vy * Math.sin(psi);

    const H = Matrix.zeros(1, 8);
    H.set(0, 2, Math.cos(psi));
    H.set(0, 3, Math.sin(psi));
    H.set(0, 4, -vx * Math.sin(psi) + vy * Math.cos(psi));

    const r = new Matrix([[v_forward_ai - v_forward_est]]);
    const R = new Matrix([[Math.max(variance_ai, 1e-4)]]);

    const S = H.mmul(this.P).mmul(H.transpose()).add(R);
    // S is 1x1, invert it easily or use library
    const S_inv = inverse(S);

    const K = this.P.mmul(H.transpose()).mmul(S_inv);

    this.x = this.x.add(K.mmul(r));

    let new_psi = (this.x.get(4, 0) + Math.PI) % (2 * Math.PI);
    if (new_psi < 0) new_psi += 2 * Math.PI;
    new_psi -= Math.PI;
    this.x.set(4, 0, new_psi);

    const I = Matrix.eye(8);
    this.P = I.sub(K.mmul(H)).mmul(this.P);

    return { r: r.get(0, 0), S: S.get(0, 0) };
  }

  public updateNHC(R_nhc: number = 0.01) {
    const psi = this.x.get(4, 0);
    const vx = this.x.get(2, 0);
    const vy = this.x.get(3, 0);

    const v_lat_est = -vx * Math.sin(psi) + vy * Math.cos(psi);

    const H = Matrix.zeros(1, 8);
    H.set(0, 2, -Math.sin(psi));
    H.set(0, 3, Math.cos(psi));
    H.set(0, 4, -vx * Math.cos(psi) - vy * Math.sin(psi));

    const r = new Matrix([[-v_lat_est]]);
    const R = new Matrix([[R_nhc]]);

    const S = H.mmul(this.P).mmul(H.transpose()).add(R);
    const S_inv = inverse(S);

    const K = this.P.mmul(H.transpose()).mmul(S_inv);

    this.x = this.x.add(K.mmul(r));

    let new_psi = (this.x.get(4, 0) + Math.PI) % (2 * Math.PI);
    if (new_psi < 0) new_psi += 2 * Math.PI;
    new_psi -= Math.PI;
    this.x.set(4, 0, new_psi);

    const I = Matrix.eye(8);
    this.P = I.sub(K.mmul(H)).mmul(this.P);
  }

  public updateGNSS(px_gnss: number, py_gnss: number, vx_gnss: number, vy_gnss: number, pos_std: number = 2.0, vel_std: number = 0.5): { nis: number; r: Matrix } {
    const z = new Matrix([[px_gnss], [py_gnss], [vx_gnss], [vy_gnss]]);
    const h = new Matrix([
      [this.x.get(0, 0)],
      [this.x.get(1, 0)],
      [this.x.get(2, 0)],
      [this.x.get(3, 0)]
    ]);

    const H = Matrix.zeros(4, 8);
    H.set(0, 0, 1.0);
    H.set(1, 1, 1.0);
    H.set(2, 2, 1.0);
    H.set(3, 3, 1.0);

    const r = z.sub(h);

    const R = Matrix.zeros(4, 4);
    R.set(0, 0, pos_std * pos_std);
    R.set(1, 1, pos_std * pos_std);
    R.set(2, 2, vel_std * vel_std);
    R.set(3, 3, vel_std * vel_std);

    const S = H.mmul(this.P).mmul(H.transpose()).add(R);
    const S_inv = inverse(S);

    const K = this.P.mmul(H.transpose()).mmul(S_inv);

    this.x = this.x.add(K.mmul(r));

    let new_psi = (this.x.get(4, 0) + Math.PI) % (2 * Math.PI);
    if (new_psi < 0) new_psi += 2 * Math.PI;
    new_psi -= Math.PI;
    this.x.set(4, 0, new_psi);

    const I = Matrix.eye(8);
    this.P = I.sub(K.mmul(H)).mmul(this.P);

    // NIS = r^T * S^-1 * r
    const nisMatrix = r.transpose().mmul(S_inv).mmul(r);
    return { nis: nisMatrix.get(0, 0), r };
  }

  get speed(): number {
    return Math.sqrt(Math.pow(this.x.get(2, 0), 2) + Math.pow(this.x.get(3, 0), 2));
  }

  get headingDeg(): number {
    const psiRad = this.x.get(4, 0);
    const psiDeg = (psiRad * 180.0) / Math.PI;
    // Convert math angle (CCW from East) to Nav heading (CW from North)
    return (90 - psiDeg + 360) % 360;
  }
}
