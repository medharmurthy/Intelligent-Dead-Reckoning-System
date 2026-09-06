import { describe, it, expect } from 'vitest';
import { EKF2D } from './ins_ekf';
import testDataRaw from '../../public/data/test_ekf.json';

const testData: any = testDataRaw;

describe('EKF2D Deterministic Validation', () => {
  it('should match python EKF states exactly', () => {
    const ekf = new EKF2D();
    
    // helper to compare arrays
    const expectArraysClose = (actual: number[], expected: number[], step: string) => {
      expect(actual.length).toBe(expected.length);
      for (let i = 0; i < actual.length; i++) {
        try {
            expect(actual[i]).toBeCloseTo(expected[i], 4);
        } catch (e) {
            throw new Error(`Failed at step ${step}, index ${i}. Expected ${expected[i]}, got ${actual[i]}`);
        }
      }
    };

    // helper to extract array from Matrix
    const getXArray = (e: EKF2D) => {
      const arr = [];
      for (let i = 0; i < 8; i++) arr.push(e.x.get(i, 0));
      return arr;
    };
    
    const getPDiag = (e: EKF2D) => {
      const arr = [];
      for (let i = 0; i < 8; i++) arr.push(e.P.get(i, i));
      return arr;
    };

    // 0: init
    expectArraysClose(getXArray(ekf), testData[0].x, 'init x');
    
    // 1: predict
    ekf.predict(0.1, 0.5, 0.1, 0.02);
    expectArraysClose(getXArray(ekf), testData[1].x, 'predict1 x');
    expectArraysClose(getPDiag(ekf), testData[1].P_diag, 'predict1 P');
    
    // 2: predict again
    ekf.predict(0.1, 0.5, 0.1, 0.02);
    expectArraysClose(getXArray(ekf), testData[2].x, 'predict2 x');
    expectArraysClose(getPDiag(ekf), testData[2].P_diag, 'predict2 P');

    // 3: AI update
    const resAi = ekf.updateAIVelocity(1.0, 0.1);
    expectArraysClose(getXArray(ekf), testData[3].x, 'ai x');
    expectArraysClose(getPDiag(ekf), testData[3].P_diag, 'ai P');
    expect(resAi.r).toBeCloseTo(testData[3].r, 4);
    expect(resAi.S).toBeCloseTo(testData[3].S, 4);

    // 4: NHC update
    ekf.updateNHC(0.01);
    expectArraysClose(getXArray(ekf), testData[4].x, 'nhc x');

    // 5: GNSS update
    const resGnss = ekf.updateGNSS(0.1, 0.1, 1.0, 0.1, 2.0, 0.5);
    expectArraysClose(getXArray(ekf), testData[5].x, 'gnss x');
    expect(resGnss.nis).toBeCloseTo(testData[5].nis, 4);
  });
});
