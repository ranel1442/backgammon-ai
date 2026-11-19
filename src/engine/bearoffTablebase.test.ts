import { describe, expect, it } from 'vitest';
import {
  BEAROFF_NUM_STATES,
  decodeHomeBoardIndex,
  evaluateExactBearOff,
  getBearOffDatabase,
  indexHomeBoard,
  isPureBearOffPosition,
} from './bearoffTablebase';
import { evaluatePosition } from './evaluator';
import type { BoardState } from './types';

describe('Exact 54,264-State Bear-Off Tablebase', () => {
  it('verifies combinatorial state count C(15+6, 6) = 54,264 and exact index bijection', () => {
    expect(BEAROFF_NUM_STATES).toBe(54264);
    expect(indexHomeBoard([0, 0, 0, 0, 0, 0])).toBe(0);
    expect(decodeHomeBoardIndex(0)).toEqual([0, 0, 0, 0, 0, 0]);

    const testDistributions = [
      [1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 1],
      [2, 2, 2, 2, 2, 2],
      [3, 3, 3, 2, 2, 2],
      [15, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 15],
      [4, 1, 0, 3, 2, 5],
    ];

    for (const dist of testDistributions) {
      const idx = indexHomeBoard(dist);
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(BEAROFF_NUM_STATES);
      expect(decodeHomeBoardIndex(idx)).toEqual(dist);
    }

    // Verify bijection across a stride of all 54,264 indices
    for (let idx = 0; idx < BEAROFF_NUM_STATES; idx += 137) {
      const decoded = decodeHomeBoardIndex(idx);
      const sum = decoded.reduce((a, b) => a + b, 0);
      expect(sum).toBeLessThanOrEqual(15);
      expect(indexHomeBoard(decoded)).toBe(idx);
    }
    expect(decodeHomeBoardIndex(BEAROFF_NUM_STATES - 1)).toEqual([
      0, 0, 0, 0, 0, 15,
    ]);
  });

  it('gives 100% win probability and 1.0 expected turns for 1 checker on the 1-point on roll', () => {
    const db = getBearOffDatabase();
    const oneCheckerIdx = indexHomeBoard([1, 0, 0, 0, 0, 0]);
    expect(db.expectedTurns[oneCheckerIdx]).toBeCloseTo(1.0, 5);

    const res = evaluateExactBearOff(
      [1, 0, 0, 0, 0, 0],
      [2, 2, 1, 0, 0, 0],
      'white'
    );
    expect(res.winProb).toBeCloseTo(100.0, 4);
    expect(res.whiteExpectedTurns).toBeCloseTo(1.0, 4);
    expect(res.equity).toBeGreaterThanOrEqual(1.0);
  });

  it('guarantees 50.0% winProb and 0.00 equity on symmetric home boards', () => {
    const home = [2, 2, 2, 1, 1, 0];
    const sym = evaluateExactBearOff(home, home);
    expect(sym.winProb).toBeCloseTo(50.0, 4);
    expect(sym.equity).toBeCloseTo(0.0, 4);
    expect(sym.whiteExpectedTurns).toBeCloseTo(sym.blackExpectedTurns, 5);
  });

  it('integrates with evaluatePosition on pure bear-off boards', () => {
    const points = new Array(24).fill(0);
    // White has 2 checkers on 1-point (index 0)
    points[0] = 2;
    // Black has 6 checkers on 24-point (index 18, which is 6 pips from bearing off)
    points[18] = -6;

    const state: BoardState = {
      points,
      bar: { white: 0, black: 0 },
      borneOff: { white: 13, black: 9 },
      currentTurn: 'white',
      dice: [],
      remainingMoves: [],
    };

    expect(isPureBearOffPosition(state)).toBe(true);
    const evalRes = evaluatePosition(state);
    expect(evalRes.winProb).toBeGreaterThan(95);
    expect(evalRes.equity).toBeGreaterThan(0.9);
  });
});
