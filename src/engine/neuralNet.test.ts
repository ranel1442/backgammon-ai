import { describe, expect, it } from 'vitest';
import {
  classifyGamePhase,
  createUntrainedDualNN,
  extractFeatures210,
  extractFeatures260,
  FEATURE_VECTOR_SIZE,
  NN_INPUT_DIM,
  NN_INPUT_DIM_V2,
} from './neuralNet';
import { runMonteCarloRollout } from './rollout';
import { createInitialBoard, generateLegalTurnMoves } from './rules';
import { swapBoardColors } from './visionScanner';

describe('260-Feature Grandmaster Extractor, 3-Network Residual NN (v3) & Rollout Engine', () => {
  it('extracts a 260-element Float32Array (and legacy 210-element) with exact White/Black perspective invariance', () => {
    const initial = createInitialBoard();
    // Create an asymmetric tactical board with stripped points (2), spare builders (3, 4), and primes
    initial.points[4] = 2; // Stripped golden 5-point for White
    initial.points[5] = 4; // Spare builder on 6-point for White
    initial.points[7] = 3; // Spare builder on 8-point for White
    initial.points[23] = 2;
    initial.points[19] = -2; // Stripped 5-point for Black
    initial.points[18] = -4; // Spare builder on 6-point for Black

    const mirrored = swapBoardColors(initial);

    const whiteFeats260 = extractFeatures260(initial, 'white');
    const blackMirroredFeats260 = extractFeatures260(mirrored, 'black');

    expect(whiteFeats260.length).toBe(FEATURE_VECTOR_SIZE);
    expect(FEATURE_VECTOR_SIZE).toBe(260);
    expect(NN_INPUT_DIM).toBe(260);

    for (let i = 0; i < FEATURE_VECTOR_SIZE; i++) {
      expect(whiteFeats260[i]).toBeGreaterThanOrEqual(-1.5);
      expect(whiteFeats260[i]).toBeLessThanOrEqual(1.5);
      expect(whiteFeats260[i]).toBeCloseTo(blackMirroredFeats260[i], 5);
    }

    // Legacy 210-feature backward compatibility check
    const whiteFeats210 = extractFeatures210(initial, 'white');
    const blackMirroredFeats210 = extractFeatures210(mirrored, 'black');
    expect(whiteFeats210.length).toBe(NN_INPUT_DIM_V2);
    for (let i = 0; i < NN_INPUT_DIM_V2; i++) {
      expect(whiteFeats210[i]).toBeCloseTo(blackMirroredFeats210[i], 5);
    }
  });

  it('deterministically classifies Contact, Crashed, and Race game phases', () => {
    const initial = createInitialBoard();
    expect(classifyGamePhase(initial)).toBe('contact');

    // Crashed position: contact still exists, but White has only 4 checkers left (11 borne off)
    const crashed = createInitialBoard();
    crashed.points.fill(0);
    crashed.points[23] = 2; // White back anchor
    crashed.points[1] = 2;
    crashed.borneOff.white = 11; // 4 active checkers left (<= 5)
    crashed.points[20] = -8;
    crashed.points[18] = -7;
    crashed.borneOff.black = 0;
    expect(classifyGamePhase(crashed)).toBe('crashed');

    // Pure Race position: White in 0..5, Black in 18..23, no bar checkers
    const race = createInitialBoard();
    race.points.fill(0);
    race.points[4] = 3;
    race.points[1] = 2;
    race.borneOff.white = 10;
    race.points[19] = -3;
    race.points[22] = -2;
    race.borneOff.black = 10;
    expect(classifyGamePhase(race)).toBe('race');
  });

  it('normalizes 6-class Softmax probabilities to 1.0 and preserves antisymmetry across Contact (2xRes512), Crashed (Res384), and Race (Res384)', () => {
    const nn = createUntrainedDualNN(42);
    expect(nn.getVersion()).toBe(3);

    const initial = createInitialBoard();
    const mirrored = swapBoardColors(initial);

    const out = nn.evaluateWhitePerspective(initial);
    const probSum =
      out.pWinSingle +
      out.pWinGammonOnly +
      out.pWinBg +
      out.pLoseSingle +
      out.pLoseGammonOnly +
      out.pLoseBg;

    expect(probSum).toBeCloseTo(1.0, 5);
    expect(out.equity).toBeGreaterThanOrEqual(-3.0);
    expect(out.equity).toBeLessThanOrEqual(3.0);

    for (const phase of ['contact', 'crashed', 'race'] as const) {
      const symInitial = nn.evaluateSymmetric(initial, phase);
      const symMirrored = nn.evaluateSymmetric(mirrored, phase);
      expect(symInitial.equity).toBeCloseTo(-symMirrored.equity, 5);
      expect(symInitial.winProb + symMirrored.winProb).toBeCloseTo(100.0, 4);
    }
  });

  it('runs Common Random Numbers Monte Carlo Rollouts with 95% confidence intervals', () => {
    const initial = createInitialBoard();
    const dice: [number, number] = [3, 1];
    const moves = generateLegalTurnMoves(initial, dice).slice(0, 3);

    let lastProgress = 0;
    const results = runMonteCarloRollout(
      initial,
      dice,
      moves,
      36,
      (pct) => {
        lastProgress = pct;
      }
    );

    expect(lastProgress).toBe(100);
    expect(results.length).toBe(moves.length);
    for (const r of results) {
      expect(r.trials).toBe(36);
      expect(r.ci95).toBeGreaterThanOrEqual(0);
      expect(r.winProb).toBeGreaterThanOrEqual(0);
      expect(r.winProb).toBeLessThanOrEqual(100);
    }
  });
});
