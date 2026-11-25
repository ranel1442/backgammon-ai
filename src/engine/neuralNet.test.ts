import { describe, expect, it } from 'vitest';
import {
  classifyGamePhase,
  createUntrainedDualNN,
  DualBackgammonNN,
  extractFeatures210,
  extractFeatures260,
  FEATURE_VECTOR_SIZE,
  NN_INPUT_DIM,
  NN_INPUT_DIM_V2,
  type ResidualBlockWeightsJSON,
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

  it('guarantees exact numerical parity (< 1e-5) between flattened zero-allocation Float32Array forward pass (j += 4 unrolled gemv) and naive 2D matrix multiplication', () => {
    let prng = 1337;
    const nextRand = (scale: number): number => {
      prng = (Math.imul(prng, 1664525) + 1013904223) | 0;
      return (((prng >>> 0) / 4294967296) - 0.5) * scale;
    };

    const make2DMatrix = (
      outDim: number,
      inDim: number,
      scale: number
    ): number[][] => {
      const mat: number[][] = [];
      for (let r = 0; r < outDim; r++) {
        const row: number[] = [];
        for (let c = 0; c < inDim; c++) {
          row.push(nextRand(scale));
        }
        mat.push(row);
      }
      return mat;
    };

    const makeBias = (dim: number, scale = 0.05): number[] =>
      Array.from({ length: dim }, () => nextRand(scale));

    const naive2DLinear = (
      w2d: number[][],
      bias: number[],
      x: number[],
      applySiLU: boolean
    ): number[] => {
      const out = new Array<number>(w2d.length);
      for (let o = 0; o < w2d.length; o++) {
        let sum = bias[o];
        const row = w2d[o];
        for (let j = 0; j < row.length; j++) {
          sum += row[j] * x[j];
        }
        out[o] = applySiLU ? sum / (1.0 + Math.exp(-sum)) : sum;
      }
      return out;
    };

    const build2DSubNetSpec = (
      inDim: number,
      resDim: number,
      neckDim: number,
      numBlocks: number
    ) => {
      const inW = make2DMatrix(resDim, inDim, Math.sqrt(2 / inDim) * 0.25);
      const inB = makeBias(resDim);
      const blocks: Array<{
        dim: number;
        w1: number[][];
        b1: number[];
        w2: number[][];
        b2: number[];
      }> = [];
      for (let b = 0; b < numBlocks; b++) {
        blocks.push({
          dim: resDim,
          w1: make2DMatrix(resDim, resDim, Math.sqrt(2 / resDim) * 0.2),
          b1: makeBias(resDim),
          w2: make2DMatrix(resDim, resDim, Math.sqrt(2 / resDim) * 0.2),
          b2: makeBias(resDim),
        });
      }
      const neckW = make2DMatrix(neckDim, resDim, Math.sqrt(2 / resDim) * 0.25);
      const neckB = makeBias(neckDim);
      const outW = make2DMatrix(6, neckDim, Math.sqrt(2 / neckDim) * 0.25);
      const outB = makeBias(6);

      const referenceForward = (featArray: Float32Array) => {
        let x = naive2DLinear(inW, inB, Array.from(featArray), true);
        for (const blk of blocks) {
          const h1 = naive2DLinear(blk.w1, blk.b1, x, true);
          const h2 = naive2DLinear(blk.w2, blk.b2, h1, true);
          x = x.map((v, idx) => v + h2[idx]);
        }
        const neck = naive2DLinear(neckW, neckB, x, true);
        const logits = naive2DLinear(outW, outB, neck, false);
        const maxL = Math.max(...logits);
        const exps = logits.map((l) => Math.exp(l - maxL));
        const denom = exps.reduce((a, b) => a + b, 0);
        const probs = exps.map((e) => e / denom);
        const pWinGammon = probs[1] + probs[2];
        const pLoseGammon = probs[4] + probs[5];
        const winProb01 = probs[0] + pWinGammon;
        const equity = Math.max(
          -3.0,
          Math.min(
            3.0,
            winProb01 -
              (1 - winProb01) +
              (pWinGammon - pLoseGammon) +
              (probs[2] - probs[5])
          )
        );
        return { probs, winProb01, equity };
      };

      const json = {
        inDim,
        resDim,
        bottleneckDim: neckDim,
        outDim: 6,
        inW,
        inB,
        resBlocks: blocks as ResidualBlockWeightsJSON[],
        neckW,
        neckB,
        outW,
        outB,
      };

      return { json, referenceForward };
    };

    const contactSpec = build2DSubNetSpec(NN_INPUT_DIM, 512, 256, 2);
    const crashedSpec = build2DSubNetSpec(NN_INPUT_DIM, 384, 192, 1);
    const raceSpec = build2DSubNetSpec(NN_INPUT_DIM, 384, 192, 1);

    // Load 2D weight matrices directly into DualBackgammonNN (flattened once at load time)
    const compiledNN = new DualBackgammonNN({
      version: 3,
      trainedGames: 15000512,
      contact: contactSpec.json,
      crashed: crashedSpec.json,
      race: raceSpec.json,
    });

    const boardA = createInitialBoard();
    boardA.points[4] = 3;
    boardA.points[9] = 1;
    boardA.bar.black = 1;
    boardA.points[23] = 1;

    const boardB = createInitialBoard();
    boardB.points.fill(0);
    boardB.points[2] = 4;
    boardB.points[5] = 3;
    boardB.borneOff.white = 8;
    boardB.points[19] = -4;
    boardB.points[22] = -3;
    boardB.borneOff.black = 8;

    for (const board of [boardA, boardB, boardA]) {
      for (const [phase, spec] of [
        ['contact', contactSpec],
        ['crashed', crashedSpec],
        ['race', raceSpec],
      ] as const) {
        for (const perspective of ['white', 'black'] as const) {
          const feats = extractFeatures260(board, perspective);
          const ref = spec.referenceForward(feats);
          const fast = compiledNN.forwardPerspective(board, perspective, phase);

          expect(Math.abs(fast.winProb01 - ref.winProb01)).toBeLessThan(1e-5);
          expect(Math.abs(fast.equity - ref.equity)).toBeLessThan(1e-5);
          expect(Math.abs(fast.pWinSingle - ref.probs[0])).toBeLessThan(1e-5);
          expect(Math.abs(fast.pWinGammonOnly - ref.probs[1])).toBeLessThan(
            1e-5
          );
          expect(Math.abs(fast.pWinBg - ref.probs[2])).toBeLessThan(1e-5);
          expect(Math.abs(fast.pLoseSingle - ref.probs[3])).toBeLessThan(1e-5);
          expect(Math.abs(fast.pLoseGammonOnly - ref.probs[4])).toBeLessThan(
            1e-5
          );
          expect(Math.abs(fast.pLoseBg - ref.probs[5])).toBeLessThan(1e-5);
        }
      }
    }
  });
});
