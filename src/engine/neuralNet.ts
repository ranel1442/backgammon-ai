import { getExactExpectedBearOffTurns } from './bearoffTablebase';
import { getPipCount } from './rules';
import type { BoardState, Player } from './types';

export const NN_INPUT_DIM_V2 = 210;
export const NN_INPUT_DIM = 260;
export const FEATURE_VECTOR_SIZE = NN_INPUT_DIM;
export const NN_OUTPUT_DIM = 6;

export type GamePhase = 'race' | 'crashed' | 'contact';

/**
 * Deterministic 3-way game-phase classification shared identically between TypeScript and Python:
 * - 'race': no checkers on the bar AND White's highest occupied point index < Black's lowest occupied point index.
 * - 'crashed': contact position where either player has <= 5 active checkers remaining (15 - borneOff <= 5).
 * - 'contact': all other contact positions.
 */
export function classifyGamePhase(state: BoardState): GamePhase {
  if (state.bar.white === 0 && state.bar.black === 0) {
    let highestWhite = -1;
    let lowestBlack = 24;
    for (let i = 0; i < 24; i++) {
      const v = state.points[i];
      if (v > 0 && i > highestWhite) highestWhite = i;
      if (v < 0 && i < lowestBlack) lowestBlack = i;
    }
    if (highestWhite < lowestBlack) {
      return 'race';
    }
  }

  const whiteActive = 15 - state.borneOff.white;
  const blackActive = 15 - state.borneOff.black;
  if (whiteActive <= 5 || blackActive <= 5) {
    return 'crashed';
  }

  return 'contact';
}

export interface LayerWeightsJSON {
  inFeatures: number;
  outFeatures: number;
  /** Row-major weights of length `outFeatures * inFeatures` (1D flat or 2D `[outFeatures][inFeatures]`) */
  weights: number[] | number[][];
  /** Bias vector of length `outFeatures` */
  bias: number[];
}

export interface ResidualBlockWeightsJSON {
  dim: number;
  w1: number[] | number[][];
  b1: number[];
  w2: number[] | number[][];
  b2: number[];
}

export interface ResidualSubNetWeightsJSON {
  inDim: number;
  resDim: number;
  bottleneckDim: number;
  outDim: number;
  inW: number[] | number[][];
  inB: number[];
  resBlock?: ResidualBlockWeightsJSON;
  resBlocks?: ResidualBlockWeightsJSON[];
  neckW: number[] | number[][];
  neckB: number[];
  outW: number[] | number[][];
  outB: number[];
}

export interface NetworkWeightsJSON {
  layers?: LayerWeightsJSON[];
  inDim?: number;
  h1Dim?: number;
  h2Dim?: number;
  outDim?: number;
  w1?: number[] | number[][];
  b1?: number[];
  w2?: number[] | number[][];
  b2?: number[];
  w3?: number[] | number[][];
  b3?: number[];
  // Version 2 / Version 3 Residual fields
  resDim?: number;
  bottleneckDim?: number;
  inW?: number[] | number[][];
  inB?: number[];
  resBlock?: ResidualBlockWeightsJSON;
  resBlocks?: ResidualBlockWeightsJSON[];
  neckW?: number[] | number[][];
  neckB?: number[];
  outW?: number[] | number[][];
  outB?: number[];
}

export interface BackgammonNNWeightsFile {
  version: number;
  trainedGames?: number;
  gamesTrained?: number;
  eloEstimate?: number;
  trainedAt?: string;
  device?: string;
  // Version 2 & 3 keys
  contact?: NetworkWeightsJSON;
  crashed?: NetworkWeightsJSON;
  race?: NetworkWeightsJSON;
  // Version 1 backward-compatible keys
  contactNet?: NetworkWeightsJSON;
  crashedNet?: NetworkWeightsJSON;
  raceNet?: NetworkWeightsJSON;
}

export interface NNOutcomeProbabilities {
  /** Total win probability (0..1) = pWinSingle + pWinGammonOnly + pWinBg */
  winProb01: number;
  /** Probability of winning a single game (0..1) */
  pWinSingle: number;
  /** Probability of winning a 2x gammon (excluding backgammon) (0..1) */
  pWinGammonOnly: number;
  /** Probability of winning a gammon (including backgammon) (0..1) */
  pWinGammon: number;
  /** Probability of winning a backgammon (0..1) */
  pWinBg: number;
  /** Probability of losing a single game (0..1) */
  pLoseSingle: number;
  /** Probability of losing a 2x gammon (excluding backgammon) (0..1) */
  pLoseGammonOnly: number;
  /** Probability of losing a gammon (including backgammon) (0..1) */
  pLoseGammon: number;
  /** Probability of losing a backgammon (0..1) */
  pLoseBg: number;
  /** Cubeless equity in [-3.0, +3.0] from perspective player's viewpoint */
  equity: number;
}

interface CompiledLayer {
  inFeatures: number;
  outFeatures: number;
  weights: Float32Array;
  bias: Float32Array;
}

interface CompiledResBlock {
  layer1: CompiledLayer;
  layer2: CompiledLayer;
}

interface CompiledResidualSubNet {
  kind: 'residual' | 'mlp-v1';
  inDim: number;
  // For mlp-v1
  layers?: CompiledLayer[];
  // For residual (v2 / v3): inDim -> resDim (SiLU) -> [ResidualBlock(resDim) * K] -> bottleneckDim (SiLU) -> outDim (Softmax)
  inLayer?: CompiledLayer;
  resBlocks?: CompiledResBlock[];
  neckLayer?: CompiledLayer;
  outLayer?: CompiledLayer;
}

function getRelPointCount(
  points: readonly number[],
  isWhite: boolean,
  rel: number
): number {
  const raw = points[isWhite ? rel : 23 - rel];
  return isWhite ? raw : -raw;
}

/**
 * Zero-allocation in-place extractor for the Ultimate 260-Feature Grandmaster canonical representation (`version: 3`).
 * Writes directly into a pre-allocated `Float32Array(260)` buffer (`out`).
 */
export function extractFeatures260Into(
  state: BoardState,
  perspective: Player,
  out: Float32Array
): Float32Array {
  out.fill(0);
  const isWhite = perspective === 'white';
  const opponent: Player = isWhite ? 'black' : 'white';
  const points = state.points;

  // 1. [0..191]: 192 Point features + [210..257]: 48 Stripped & Spare Builder features
  for (let step = 0; step < 24; step++) {
    const rel = 23 - step;
    const count = getRelPointCount(points, isWhite, rel);
    const base = step * 8;

    if (count > 0) {
      out[base] = 1.0;
      if (count >= 2) out[base + 1] = 1.0;
      if (count >= 3) out[base + 2] = 1.0;
      if (count > 3) out[base + 3] = (count - 3) * 0.5;

      if (count === 2) {
        out[210 + step] = 1.0;
      } else if (count === 3 || count === 4) {
        out[234 + step] = 1.0;
      }
    } else if (count < 0) {
      const oppCount = -count;
      out[base + 4] = 1.0;
      if (oppCount >= 2) out[base + 5] = 1.0;
      if (oppCount >= 3) out[base + 6] = 1.0;
      if (oppCount > 3) out[base + 7] = (oppCount - 3) * 0.5;

      if (oppCount === 2) {
        out[210 + step] = -1.0;
      } else if (oppCount === 3 || oppCount === 4) {
        out[234 + step] = -1.0;
      }
    }
  }

  // 2. Global tactical, timing, lethality & prime-trapping metrics
  const barSelf = state.bar[perspective];
  const barOpp = state.bar[opponent];
  const borneOffSelf = state.borneOff[perspective];
  const borneOffOpp = state.borneOff[opponent];

  const pipSelf = getPipCount(state, perspective);
  const pipOpp = getPipCount(state, opponent);

  const exactTurnsSelf = getExactExpectedBearOffTurns(state, perspective);
  const exactTurnsOpp = getExactExpectedBearOffTurns(state, opponent);

  let highestSelfRel = barSelf > 0 ? 24 : -1;
  let lowestOppRel = barOpp > 0 ? -1 : 24;

  let curPrimeSelf = 0;
  let maxPrimeSelf = 0;
  let curPrimeOpp = 0;
  let maxPrimeOpp = 0;

  let homePointsSelf = 0;
  let homePointsOpp = 0;
  let outfieldAndBackPipSelf = barSelf * 25;
  let anchorsInOppHomeSelf = 0;
  let selfBackCheckers = barSelf;
  let oppBackCheckers = barOpp;

  for (let rel = 0; rel < 24; rel++) {
    const c = getRelPointCount(points, isWhite, rel);
    if (c > 0) {
      if (rel > highestSelfRel) highestSelfRel = rel;
      if (rel >= 6) {
        outfieldAndBackPipSelf += c * (rel + 1);
      }
      if (rel >= 18) {
        selfBackCheckers += c;
        if (c >= 2) anchorsInOppHomeSelf++;
      }
    } else if (c < 0) {
      if (rel < lowestOppRel) lowestOppRel = rel;
      if (rel <= 5) {
        oppBackCheckers += -c;
      }
    }

    if (c >= 2) {
      curPrimeSelf++;
      if (curPrimeSelf > maxPrimeSelf) maxPrimeSelf = curPrimeSelf;
      if (rel <= 5) homePointsSelf++;
    } else {
      curPrimeSelf = 0;
    }

    if (c <= -2) {
      curPrimeOpp++;
      if (curPrimeOpp > maxPrimeOpp) maxPrimeOpp = curPrimeOpp;
      if (rel >= 18) homePointsOpp++;
    } else {
      curPrimeOpp = 0;
    }
  }

  const isContact = highestSelfRel > lowestOppRel || barSelf > 0 || barOpp > 0;
  let blotExposureSelf = 0;
  let blotExposureOpp = 0;

  if (isContact) {
    for (let rel = 0; rel < 24; rel++) {
      const c = getRelPointCount(points, isWhite, rel);
      if (c === 1 && rel > lowestOppRel) {
        blotExposureSelf++;
      } else if (c === -1 && rel < highestSelfRel) {
        blotExposureOpp++;
      }
    }
  }

  const closedRatioSelf = homePointsSelf / 6.0;
  const closedRatioOpp = homePointsOpp / 6.0;
  const blotRatioSelf = Math.min(6, blotExposureSelf) / 6.0;
  const blotRatioOpp = Math.min(6, blotExposureOpp) / 6.0;
  const golden20Owned = getRelPointCount(points, isWhite, 19) >= 2 ? 1.0 : 0.0;
  const doubleAnchorIndicator = anchorsInOppHomeSelf >= 2 ? 1.0 : 0.0;
  const timingRatio = Math.min(
    2.0,
    outfieldAndBackPipSelf / Math.max(1.0, pipOpp)
  );

  out[192] = barSelf / 2.0;
  out[193] = barOpp / 2.0;
  out[194] = borneOffSelf / 15.0;
  out[195] = borneOffOpp / 15.0;
  out[196] = pipSelf / 167.0;
  out[197] = pipOpp / 167.0;
  out[198] = (pipOpp - pipSelf) / 50.0;
  out[199] = exactTurnsSelf / 20.0;
  out[200] = exactTurnsOpp / 20.0;
  out[201] = Math.min(6, maxPrimeSelf) / 6.0;
  out[202] = Math.min(6, maxPrimeOpp) / 6.0;
  out[203] = closedRatioSelf * closedRatioSelf; // Exact bar blockage prob by Self
  out[204] = closedRatioOpp * closedRatioOpp; // Exact bar blockage prob by Opp
  out[205] = blotRatioSelf * closedRatioOpp; // Hit lethality against Self
  out[206] = blotRatioOpp * closedRatioSelf; // Hit lethality against Opp
  out[207] = isContact ? 1.0 : 0.0;
  out[208] = timingRatio / 2.0; // Timing / Backgame viability ratio
  out[209] = 0.5 * doubleAnchorIndicator + 0.5 * golden20Owned;

  // [258..259]: Prime Trapping & Lethality products
  const primeTrapSelf = Math.min(6, maxPrimeSelf) * Math.min(3, oppBackCheckers);
  const primeTrapOpp = Math.min(6, maxPrimeOpp) * Math.min(3, selfBackCheckers);
  out[258] = primeTrapSelf / 18.0;
  out[259] = primeTrapOpp / 18.0;

  return out;
}

/**
 * Extracts the Ultimate 260-Feature Grandmaster canonical representation (`version: 3`)
 * for `perspective` ('white' | 'black').
 */
export function extractFeatures260(
  state: BoardState,
  perspective: Player
): Float32Array {
  return extractFeatures260Into(
    state,
    perspective,
    new Float32Array(NN_INPUT_DIM)
  );
}

/**
 * Zero-allocation in-place legacy 210-dimensional feature extractor (`version: 1` / `version: 2`).
 */
export function extractFeatures210Into(
  state: BoardState,
  perspective: Player,
  out: Float32Array
): Float32Array {
  out.fill(0);
  const isWhite = perspective === 'white';
  const opponent: Player = isWhite ? 'black' : 'white';
  const points = state.points;

  for (let step = 0; step < 24; step++) {
    const rel = 23 - step;
    const count = getRelPointCount(points, isWhite, rel);
    const base = step * 8;

    if (count > 0) {
      out[base] = 1.0;
      if (count >= 2) out[base + 1] = 1.0;
      if (count >= 3) out[base + 2] = 1.0;
      if (count > 3) out[base + 3] = (count - 3) * 0.5;
    } else if (count < 0) {
      const oppCount = -count;
      out[base + 4] = 1.0;
      if (oppCount >= 2) out[base + 5] = 1.0;
      if (oppCount >= 3) out[base + 6] = 1.0;
      if (oppCount > 3) out[base + 7] = (oppCount - 3) * 0.5;
    }
  }

  const barSelf = state.bar[perspective];
  const barOpp = state.bar[opponent];
  const borneOffSelf = state.borneOff[perspective];
  const borneOffOpp = state.borneOff[opponent];

  const pipSelf = getPipCount(state, perspective);
  const pipOpp = getPipCount(state, opponent);

  const exactTurnsSelf = getExactExpectedBearOffTurns(state, perspective);
  const exactTurnsOpp = getExactExpectedBearOffTurns(state, opponent);

  let highestSelfRel = barSelf > 0 ? 24 : -1;
  let lowestOppRel = barOpp > 0 ? -1 : 24;

  let curPrimeSelf = 0;
  let maxPrimeSelf = 0;
  let curPrimeOpp = 0;
  let maxPrimeOpp = 0;

  let homePointsSelf = 0;
  let homePointsOpp = 0;

  for (let rel = 0; rel < 24; rel++) {
    const c = getRelPointCount(points, isWhite, rel);
    if (c > 0 && rel > highestSelfRel) highestSelfRel = rel;
    if (c < 0 && rel < lowestOppRel) lowestOppRel = rel;

    if (c >= 2) {
      curPrimeSelf++;
      if (curPrimeSelf > maxPrimeSelf) maxPrimeSelf = curPrimeSelf;
      if (rel <= 5) homePointsSelf++;
    } else {
      curPrimeSelf = 0;
    }

    if (c <= -2) {
      curPrimeOpp++;
      if (curPrimeOpp > maxPrimeOpp) maxPrimeOpp = curPrimeOpp;
      if (rel >= 18) homePointsOpp++;
    } else {
      curPrimeOpp = 0;
    }
  }

  const isContact = highestSelfRel > lowestOppRel || barSelf > 0 || barOpp > 0;

  let blotExposureSelf = 0;
  let blotExposureOpp = 0;
  let trappedSelf = 0;
  let trappedOpp = 0;

  if (isContact) {
    for (let rel = 0; rel < 24; rel++) {
      const c = getRelPointCount(points, isWhite, rel);
      if (c === 1 && rel > lowestOppRel) {
        blotExposureSelf++;
      } else if (c === -1 && rel < highestSelfRel) {
        blotExposureOpp++;
      }
    }

    for (let rel = 18; rel <= 23; rel++) {
      const c = getRelPointCount(points, isWhite, rel);
      if (c > 0) {
        let blockedFront = 0;
        for (let s = 1; s <= 6; s++) {
          if (rel - s >= 0 && getRelPointCount(points, isWhite, rel - s) <= -2)
            blockedFront++;
        }
        if (blockedFront >= 3) trappedSelf += c;
      }
    }

    for (let rel = 0; rel <= 5; rel++) {
      const c = getRelPointCount(points, isWhite, rel);
      if (c < 0) {
        let blockedFront = 0;
        for (let s = 1; s <= 6; s++) {
          if (rel + s < 24 && getRelPointCount(points, isWhite, rel + s) >= 2)
            blockedFront++;
        }
        if (blockedFront >= 3) trappedOpp += -c;
      }
    }
  }

  out[192] = barSelf / 2.0;
  out[193] = barOpp / 2.0;
  out[194] = borneOffSelf / 15.0;
  out[195] = borneOffOpp / 15.0;
  out[196] = pipSelf / 167.0;
  out[197] = pipOpp / 167.0;
  out[198] = (pipOpp - pipSelf) / 50.0;
  out[199] = exactTurnsSelf / 20.0;
  out[200] = exactTurnsOpp / 20.0;
  out[201] = Math.min(6, maxPrimeSelf) / 6.0;
  out[202] = Math.min(6, maxPrimeOpp) / 6.0;
  out[203] = homePointsSelf / 6.0;
  out[204] = homePointsOpp / 6.0;
  out[205] = Math.min(6, blotExposureSelf) / 6.0;
  out[206] = Math.min(6, blotExposureOpp) / 6.0;
  out[207] = isContact ? 1.0 : 0.0;
  out[208] = Math.min(4, trappedSelf) / 4.0;
  out[209] = Math.min(4, trappedOpp) / 4.0;

  return out;
}

/**
 * Legacy 210-dimensional feature extractor preserved for backward compatibility
 * with Version 1 and Version 2 weight files.
 */
export function extractFeatures210(
  state: BoardState,
  perspective: Player
): Float32Array {
  return extractFeatures210Into(
    state,
    perspective,
    new Float32Array(NN_INPUT_DIM_V2)
  );
}

function isWeightSequence(
  val: unknown
): val is number[] | number[][] | Float32Array {
  return Array.isArray(val) || val instanceof Float32Array;
}

/**
 * Flattens a 2D row-major `[outFeatures][inFeatures]` matrix, 1D `number[]`, or `Float32Array`
 * into a single contiguous 1D row-major `Float32Array` buffer once at model load time
 * for sequential CPU L1/L2 cache reads.
 */
export function flattenWeightMatrix(
  weights: number[] | number[][] | Float32Array,
  inFeatures: number,
  outFeatures: number
): Float32Array {
  const expectedLen = inFeatures * outFeatures;

  if (weights instanceof Float32Array) {
    if (weights.length !== expectedLen) {
      throw new Error(
        `Layer Float32Array weight length mismatch: expected ${expectedLen}, got ${weights.length}`
      );
    }
    return new Float32Array(weights);
  }

  if (weights.length > 0 && Array.isArray(weights[0])) {
    const matrix2D = weights as number[][];
    if (matrix2D.length !== outFeatures) {
      throw new Error(
        `2D weight row count mismatch: expected ${outFeatures}, got ${matrix2D.length}`
      );
    }
    const flat = new Float32Array(expectedLen);
    let offset = 0;
    for (let r = 0; r < outFeatures; r++) {
      const row = matrix2D[r];
      if (row.length !== inFeatures) {
        throw new Error(
          `2D weight col count mismatch at row ${r}: expected ${inFeatures}, got ${row.length}`
        );
      }
      for (let c = 0; c < inFeatures; c++) {
        const v = row[c];
        flat[offset++] = Number.isFinite(v) ? v : 0;
      }
    }
    return flat;
  }

  const flatInput = weights as number[];
  if (flatInput.length !== expectedLen) {
    throw new Error(
      `Layer weight length mismatch: expected ${expectedLen}, got ${flatInput.length}`
    );
  }
  const out = new Float32Array(expectedLen);
  for (let i = 0; i < expectedLen; i++) {
    const v = flatInput[i];
    out[i] = Number.isFinite(v) ? v : 0;
  }
  return out;
}

function makeCompiledLayer(
  inFeatures: number,
  outFeatures: number,
  weights: number[] | number[][] | Float32Array,
  bias: number[] | Float32Array
): CompiledLayer {
  if (bias.length !== outFeatures) {
    throw new Error(
      `Layer bias length mismatch: expected ${outFeatures}, got ${bias.length}`
    );
  }
  const biasBuf = new Float32Array(outFeatures);
  for (let i = 0; i < outFeatures; i++) {
    const b = bias[i];
    biasBuf[i] = Number.isFinite(b) ? b : 0;
  }
  return {
    inFeatures,
    outFeatures,
    weights: flattenWeightMatrix(weights, inFeatures, outFeatures),
    bias: biasBuf,
  };
}

function compileSubNetwork(netJson: NetworkWeightsJSON): CompiledResidualSubNet {
  // Check Version 3 / Version 2 Residual MLP format:
  // inDim -> resDim (SiLU) -> [ResidualBlock(resDim) * K] -> bottleneckDim (SiLU) -> outDim
  const rawBlocks =
    Array.isArray(netJson.resBlocks) && netJson.resBlocks.length > 0
      ? netJson.resBlocks
      : netJson.resBlock
        ? [netJson.resBlock]
        : null;

  if (
    typeof netJson.inDim === 'number' &&
    typeof netJson.resDim === 'number' &&
    typeof netJson.bottleneckDim === 'number' &&
    typeof netJson.outDim === 'number' &&
    isWeightSequence(netJson.inW) &&
    isWeightSequence(netJson.inB) &&
    rawBlocks &&
    isWeightSequence(netJson.neckW) &&
    isWeightSequence(netJson.neckB) &&
    isWeightSequence(netJson.outW) &&
    isWeightSequence(netJson.outB)
  ) {
    const { inDim, resDim, bottleneckDim, outDim } = netJson;
    const compiledBlocks: CompiledResBlock[] = rawBlocks.map((rb) => ({
      layer1: makeCompiledLayer(resDim, resDim, rb.w1, rb.b1),
      layer2: makeCompiledLayer(resDim, resDim, rb.w2, rb.b2),
    }));

    return {
      kind: 'residual',
      inDim,
      inLayer: makeCompiledLayer(
        inDim,
        resDim,
        netJson.inW,
        netJson.inB as number[]
      ),
      resBlocks: compiledBlocks,
      neckLayer: makeCompiledLayer(
        resDim,
        bottleneckDim,
        netJson.neckW,
        netJson.neckB as number[]
      ),
      outLayer: makeCompiledLayer(
        bottleneckDim,
        outDim,
        netJson.outW,
        netJson.outB as number[]
      ),
    };
  }

  // Version 1 flat w1..w3 format
  if (
    typeof netJson.inDim === 'number' &&
    typeof netJson.h1Dim === 'number' &&
    typeof netJson.h2Dim === 'number' &&
    typeof netJson.outDim === 'number' &&
    Array.isArray(netJson.w1) &&
    Array.isArray(netJson.b1) &&
    Array.isArray(netJson.w2) &&
    Array.isArray(netJson.b2) &&
    Array.isArray(netJson.w3) &&
    Array.isArray(netJson.b3)
  ) {
    return {
      kind: 'mlp-v1',
      inDim: netJson.inDim,
      layers: [
        makeCompiledLayer(netJson.inDim, netJson.h1Dim, netJson.w1, netJson.b1),
        makeCompiledLayer(netJson.h1Dim, netJson.h2Dim, netJson.w2, netJson.b2),
        makeCompiledLayer(netJson.h2Dim, netJson.outDim, netJson.w3, netJson.b3),
      ],
    };
  }

  // Version 1 layers array format
  const layers = (netJson.layers ?? []).map((layer) =>
    makeCompiledLayer(
      layer.inFeatures,
      layer.outFeatures,
      layer.weights,
      layer.bias
    )
  );
  return {
    kind: 'mlp-v1',
    inDim: layers[0]?.inFeatures ?? NN_INPUT_DIM_V2,
    layers,
  };
}

/**
 * Lossless 4-way loop-unrolled (`j += 4`) row-major `Float32Array` matrix-vector multiplication (`gemv`):
 * computes `out[o] = activation(bias[o] + dot(weights[o], input))` (or `out[o] += ...` when `accumulateResidual` is true)
 * with sequential L1/L2 cache reads, zero heap allocations, and exact IEEE-754 left-to-right addition order.
 */
export function gemv(
  weights: Float32Array,
  bias: Float32Array,
  input: Float32Array,
  out: Float32Array,
  inFeatures: number,
  outFeatures: number,
  applySiLU: boolean,
  accumulateResidual = false
): void {
  const unrollLimit = inFeatures - 3;
  for (let o = 0; o < outFeatures; o++) {
    const rowOffset = o * inFeatures;
    let sum = bias[o];
    let j = 0;
    for (; j < unrollLimit; j += 4) {
      const wIdx = rowOffset + j;
      sum =
        sum +
        weights[wIdx] * input[j] +
        weights[wIdx + 1] * input[j + 1] +
        weights[wIdx + 2] * input[j + 2] +
        weights[wIdx + 3] * input[j + 3];
    }
    for (; j < inFeatures; j++) {
      sum += weights[rowOffset + j] * input[j];
    }
    const val = applySiLU
      ? sum < -40
        ? 0
        : sum > 40
          ? sum
          : sum / (1.0 + Math.exp(-sum))
      : sum;
    if (accumulateResidual) {
      out[o] += val;
    } else {
      out[o] = val;
    }
  }
}

function createEmptyOutcomeProbabilities(): NNOutcomeProbabilities {
  return {
    winProb01: 0.5,
    pWinSingle: 0.5,
    pWinGammonOnly: 0,
    pWinGammon: 0,
    pWinBg: 0,
    pLoseSingle: 0.5,
    pLoseGammonOnly: 0,
    pLoseGammon: 0,
    pLoseBg: 0,
    equity: 0,
  };
}

/**
 * Converts raw output logits (`outLen = 6` or `5`) in `logitsBuf` into normalized `NNOutcomeProbabilities`
 * into `target` (zero-allocation when `target` is a pre-allocated scratch object).
 */
function logitsToOutcomeProbabilitiesInto(
  logitsBuf: Float32Array,
  outLen: number,
  target: NNOutcomeProbabilities
): NNOutcomeProbabilities {
  let lWinSingle = 0;
  let lWinGammonOnly = 0;
  let lWinBg = 0;
  let lLoseSingle = 0;
  let lLoseGammonOnly = 0;
  let lLoseBg = 0;

  if (outLen >= 6) {
    lWinSingle = logitsBuf[0];
    lWinGammonOnly = logitsBuf[1];
    lWinBg = logitsBuf[2];
    lLoseSingle = logitsBuf[3];
    lLoseGammonOnly = logitsBuf[4];
    lLoseBg = logitsBuf[5];
  } else {
    lWinSingle = logitsBuf[0] ?? 0;
    lWinGammonOnly = logitsBuf[1] ?? 0;
    lWinBg = logitsBuf[2] ?? 0;
    lLoseSingle = 0.0;
    lLoseGammonOnly = logitsBuf[3] ?? 0;
    lLoseBg = logitsBuf[4] ?? 0;
  }

  const maxLogit = Math.max(
    lWinSingle,
    lWinGammonOnly,
    lWinBg,
    lLoseSingle,
    lLoseGammonOnly,
    lLoseBg
  );

  const eWinS = Math.exp(lWinSingle - maxLogit);
  const eWinG = Math.exp(lWinGammonOnly - maxLogit);
  const eWinB = Math.exp(lWinBg - maxLogit);
  const eLoseS = Math.exp(lLoseSingle - maxLogit);
  const eLoseG = Math.exp(lLoseGammonOnly - maxLogit);
  const eLoseB = Math.exp(lLoseBg - maxLogit);

  const denom = eWinS + eWinG + eWinB + eLoseS + eLoseG + eLoseB;
  if (!Number.isFinite(denom) || denom <= 0) {
    target.winProb01 = 0.5;
    target.pWinSingle = 0.5;
    target.pWinGammonOnly = 0;
    target.pWinGammon = 0;
    target.pWinBg = 0;
    target.pLoseSingle = 0.5;
    target.pLoseGammonOnly = 0;
    target.pLoseGammon = 0;
    target.pLoseBg = 0;
    target.equity = 0;
    return target;
  }

  const pWinSingle = eWinS / denom;
  const pWinGammonOnly = eWinG / denom;
  const pWinBg = eWinB / denom;
  const pLoseSingle = eLoseS / denom;
  const pLoseGammonOnly = eLoseG / denom;
  const pLoseBg = eLoseB / denom;

  const pWinGammon = pWinGammonOnly + pWinBg;
  const pLoseGammon = pLoseGammonOnly + pLoseBg;
  const winProb01 = pWinSingle + pWinGammon;

  const equity =
    winProb01 -
    (1.0 - winProb01) +
    (pWinGammon - pLoseGammon) +
    (pWinBg - pLoseBg);

  target.winProb01 = winProb01;
  target.pWinSingle = pWinSingle;
  target.pWinGammonOnly = pWinGammonOnly;
  target.pWinGammon = pWinGammon;
  target.pWinBg = pWinBg;
  target.pLoseSingle = pLoseSingle;
  target.pLoseGammonOnly = pLoseGammonOnly;
  target.pLoseGammon = pLoseGammon;
  target.pLoseBg = pLoseBg;
  target.equity = Math.max(-3.0, Math.min(3.0, equity));

  return target;
}

/**
 * 3-Network Residual Grandmaster Backgammon Neural Network (`version: 3`, backward-compatible with `version: 2` and `version: 1`):
 * - `contact`: 260 -> 512 (SiLU) -> ResidualBlock(512) -> ResidualBlock(512) -> 256 (SiLU) -> 6 (Softmax)
 * - `crashed`: 260 -> 384 (SiLU) -> ResidualBlock(384) -> 192 (SiLU) -> 6 (Softmax)
 * - `race`:    260 -> 384 (SiLU) -> ResidualBlock(384) -> 192 (SiLU) -> 6 (Softmax)
 *
 * Pre-allocates contiguous `Float32Array` scratchpads (`buf260`, `buf512a`, `buf512b`, `buf384a`, `buf384b`,
 * `buf256`, `buf192`, `buf6`) once at construction time so every forward pass executes with **zero heap allocations**.
 */
export class DualBackgammonNN {
  private readonly contactNet: CompiledResidualSubNet;
  private readonly crashedNet: CompiledResidualSubNet;
  private readonly raceNet: CompiledResidualSubNet;

  // Pre-allocated reusable zero-GC scratchpad Float32Array buffers
  private readonly buf260 = new Float32Array(260);
  private readonly buf210 = new Float32Array(210);
  private readonly buf512a = new Float32Array(512);
  private readonly buf512b = new Float32Array(512);
  private readonly buf384a = new Float32Array(384);
  private readonly buf384b = new Float32Array(384);
  private readonly buf256 = new Float32Array(256);
  private readonly buf192 = new Float32Array(192);
  private readonly buf6 = new Float32Array(6);

  // Pre-allocated fallback buffers for any non-standard custom layer dimensions (e.g., v1 MLPs or custom unit tests)
  private readonly customBufA = new Map<number, Float32Array>();
  private readonly customBufB = new Map<number, Float32Array>();

  // Reusable outcome probability scratchpads for symmetric dual-perspective evaluation
  private readonly scratchWhiteOut = createEmptyOutcomeProbabilities();
  private readonly scratchBlackOut = createEmptyOutcomeProbabilities();

  public readonly metadata: {
    version: number;
    trainedGames: number;
    eloEstimate: number;
    trainedAt?: string;
    device?: string;
  };

  constructor(weightsFile: BackgammonNNWeightsFile) {
    const rawContact = weightsFile.contact ?? weightsFile.contactNet;
    const rawRace = weightsFile.race ?? weightsFile.raceNet;
    const rawCrashed =
      weightsFile.crashed ?? weightsFile.crashedNet ?? rawContact;

    if (!rawContact || !rawRace || !rawCrashed) {
      throw new Error(
        'Invalid BackgammonNNWeightsFile: missing contact/race network definitions.'
      );
    }

    this.contactNet = compileSubNetwork(rawContact);
    this.crashedNet = compileSubNetwork(rawCrashed);
    this.raceNet = compileSubNetwork(rawRace);

    this.preallocateCustomDims(this.contactNet);
    this.preallocateCustomDims(this.crashedNet);
    this.preallocateCustomDims(this.raceNet);

    const trainedGames =
      weightsFile.trainedGames ?? weightsFile.gamesTrained ?? 0;

    this.metadata = {
      version: weightsFile.version ?? 3,
      trainedGames,
      eloEstimate:
        weightsFile.eloEstimate ??
        (trainedGames > 0
          ? Math.round(1650 + Math.min(450, Math.log10(trainedGames + 1) * 85))
          : 0),
      trainedAt: weightsFile.trainedAt,
      device: weightsFile.device,
    };
  }

  private ensureCustomPair(dim: number): void {
    if (
      dim === 512 ||
      dim === 384 ||
      dim === 256 ||
      dim === 192 ||
      dim === 6 ||
      dim === 260 ||
      dim === 210
    ) {
      return;
    }
    if (!this.customBufA.has(dim)) {
      this.customBufA.set(dim, new Float32Array(dim));
      this.customBufB.set(dim, new Float32Array(dim));
    }
  }

  private preallocateCustomDims(subNet: CompiledResidualSubNet): void {
    this.ensureCustomPair(subNet.inDim);
    if (subNet.inLayer) this.ensureCustomPair(subNet.inLayer.outFeatures);
    if (subNet.neckLayer) this.ensureCustomPair(subNet.neckLayer.outFeatures);
    if (subNet.outLayer) this.ensureCustomPair(subNet.outLayer.outFeatures);
    if (subNet.layers) {
      for (const l of subNet.layers) {
        this.ensureCustomPair(l.inFeatures);
        this.ensureCustomPair(l.outFeatures);
      }
    }
  }

  private getBufA(dim: number): Float32Array {
    if (dim === 512) return this.buf512a;
    if (dim === 384) return this.buf384a;
    if (dim === 256) return this.buf256;
    if (dim === 192) return this.buf192;
    if (dim === 6) return this.buf6;
    let buf = this.customBufA.get(dim);
    if (!buf) {
      buf = new Float32Array(dim);
      this.customBufA.set(dim, buf);
    }
    return buf;
  }

  private getBufB(dim: number): Float32Array {
    if (dim === 512) return this.buf512b;
    if (dim === 384) return this.buf384b;
    let buf = this.customBufB.get(dim);
    if (!buf) {
      buf = new Float32Array(dim);
      this.customBufB.set(dim, buf);
    }
    return buf;
  }

  public getVersion(): number {
    return this.metadata.version;
  }

  private selectSubNet(phase: GamePhase): CompiledResidualSubNet {
    if (phase === 'race') return this.raceNet;
    if (phase === 'crashed') return this.crashedNet;
    return this.contactNet;
  }

  private runSubNetForwardZeroAlloc(
    subNet: CompiledResidualSubNet,
    input: Float32Array,
    target: NNOutcomeProbabilities
  ): NNOutcomeProbabilities {
    if (
      subNet.kind === 'residual' &&
      subNet.inLayer &&
      subNet.resBlocks &&
      subNet.neckLayer &&
      subNet.outLayer
    ) {
      const inLayer = subNet.inLayer;
      const neckLayer = subNet.neckLayer;
      const outLayer = subNet.outLayer;

      const resDim = inLayer.outFeatures;
      const neckDim = neckLayer.outFeatures;
      const outDim = outLayer.outFeatures;

      const xBuf = this.getBufA(resDim);
      const h1Buf = this.getBufB(resDim);
      const neckBuf =
        neckDim === resDim ? h1Buf : this.getBufA(neckDim);
      const outBuf = outDim <= 6 ? this.buf6 : this.getBufA(outDim);

      // 1. Input projection: inDim -> resDim (SiLU)
      gemv(
        inLayer.weights,
        inLayer.bias,
        input,
        xBuf,
        inLayer.inFeatures,
        resDim,
        true,
        false
      );

      // 2. Stacked ResidualBlock(resDim)s: xBuf = xBuf + SiLU(W2 * SiLU(W1 * xBuf + b1) + b2)
      const blocks = subNet.resBlocks;
      for (let b = 0; b < blocks.length; b++) {
        const block = blocks[b];
        gemv(
          block.layer1.weights,
          block.layer1.bias,
          xBuf,
          h1Buf,
          resDim,
          resDim,
          true,
          false
        );
        gemv(
          block.layer2.weights,
          block.layer2.bias,
          h1Buf,
          xBuf,
          resDim,
          resDim,
          true,
          true
        );
      }

      // 3. Bottleneck projection: resDim -> bottleneckDim (SiLU)
      gemv(
        neckLayer.weights,
        neckLayer.bias,
        xBuf,
        neckBuf,
        resDim,
        neckDim,
        true,
        false
      );

      // 4. Output head: bottleneckDim -> outDim logits
      gemv(
        outLayer.weights,
        outLayer.bias,
        neckBuf,
        outBuf,
        neckDim,
        outDim,
        false,
        false
      );

      return logitsToOutcomeProbabilitiesInto(outBuf, outDim, target);
    }

    // Fallback for Version 1 MLP layers using pre-allocated buffers
    const layers = subNet.layers ?? [];
    let current = input;
    let lastLen = input.length;
    for (let l = 0; l < layers.length; l++) {
      const layer = layers[l];
      const isLast = l === layers.length - 1;
      const candidateOut = isLast && layer.outFeatures <= 6
        ? this.buf6
        : this.getBufA(layer.outFeatures);
      const nextBuf =
        candidateOut === current
          ? this.getBufB(layer.outFeatures)
          : candidateOut;
      gemv(
        layer.weights,
        layer.bias,
        current,
        nextBuf,
        layer.inFeatures,
        layer.outFeatures,
        !isLast,
        false
      );
      current = nextBuf;
      lastLen = layer.outFeatures;
    }
    return logitsToOutcomeProbabilitiesInto(current, lastLen, target);
  }

  private forwardPerspectiveInto(
    state: BoardState,
    perspective: Player,
    phase: GamePhase,
    target: NNOutcomeProbabilities
  ): NNOutcomeProbabilities {
    const subNet = this.selectSubNet(phase);
    const features =
      subNet.inDim === NN_INPUT_DIM
        ? extractFeatures260Into(state, perspective, this.buf260)
        : extractFeatures210Into(state, perspective, this.buf210);
    return this.runSubNetForwardZeroAlloc(subNet, features, target);
  }

  private resolvePhase(
    state: BoardState,
    phaseOrIsPureRace?: GamePhase | boolean
  ): GamePhase {
    if (typeof phaseOrIsPureRace === 'string') {
      return phaseOrIsPureRace;
    }
    if (typeof phaseOrIsPureRace === 'boolean') {
      if (phaseOrIsPureRace) return 'race';
      return classifyGamePhase(state) === 'crashed' ? 'crashed' : 'contact';
    }
    return classifyGamePhase(state);
  }

  /**
   * Evaluates a board state from `perspective`'s viewpoint using the specialized
   * Residual sub-network for the position's game phase (`contact` | `crashed` | `race`)
   * with zero typed-array heap allocations.
   */
  public forwardPerspective(
    state: BoardState,
    perspective: Player,
    phaseOrIsPureRace?: GamePhase | boolean,
    outTarget?: NNOutcomeProbabilities
  ): NNOutcomeProbabilities {
    const phase = this.resolvePhase(state, phaseOrIsPureRace);
    return this.forwardPerspectiveInto(
      state,
      perspective,
      phase,
      outTarget ?? createEmptyOutcomeProbabilities()
    );
  }

  /**
   * Alias for `evaluateSymmetric` / `forwardPerspective` zero-allocation inference.
   */
  public evaluate(
    state: BoardState,
    phaseOrIsPureRace?: GamePhase | boolean
  ): {
    equity: number;
    winProb: number;
    gammonProb: number;
    backgammonProb: number;
    phase: GamePhase;
  } {
    return this.evaluateSymmetric(state, phaseOrIsPureRace);
  }

  /**
   * Evaluates a board state from White's canonical viewpoint.
   */
  public evaluateWhitePerspective(
    state: BoardState,
    phaseOrIsPureRace?: GamePhase | boolean
  ): NNOutcomeProbabilities {
    return this.forwardPerspective(state, 'white', phaseOrIsPureRace);
  }

  /**
   * Evaluates a board state with 100% White/Black symmetry by averaging
   * White's perspective forward pass and Black's inverted perspective forward pass
   * using pre-allocated reusable scratchpad buffers (zero typed-array heap allocations).
   */
  public evaluateSymmetric(
    state: BoardState,
    phaseOrIsPureRace?: GamePhase | boolean
  ): {
    equity: number;
    winProb: number;
    gammonProb: number;
    backgammonProb: number;
    phase: GamePhase;
  } {
    const phase = this.resolvePhase(state, phaseOrIsPureRace);

    const whitePred = this.forwardPerspectiveInto(
      state,
      'white',
      phase,
      this.scratchWhiteOut
    );
    const blackPred = this.forwardPerspectiveInto(
      state,
      'black',
      phase,
      this.scratchBlackOut
    );

    const symEquity = 0.5 * (whitePred.equity - blackPred.equity);
    const symWinProb01 =
      0.5 * (whitePred.winProb01 + (1.0 - blackPred.winProb01));
    const symGammon01 =
      0.5 *
      (whitePred.pWinGammon +
        whitePred.pLoseGammon +
        blackPred.pWinGammon +
        blackPred.pLoseGammon);
    const symBg01 =
      0.5 *
      (whitePred.pWinBg +
        whitePred.pLoseBg +
        blackPred.pWinBg +
        blackPred.pLoseBg);

    const clampedEq = Math.max(-3.0, Math.min(3.0, symEquity));
    const equity =
      Math.abs(clampedEq) < 1e-6 ? 0 : Number(clampedEq.toFixed(2));
    const winProb =
      equity === 0
        ? 50
        : Number(Math.max(0.5, Math.min(99.5, symWinProb01 * 100)).toFixed(1));

    return {
      equity,
      winProb,
      gammonProb: Number(Math.min(95, symGammon01 * 100).toFixed(1)),
      backgammonProb: Number(Math.min(45, symBg01 * 100).toFixed(1)),
      phase,
    };
  }
}

export const TripleBackgammonNN = DualBackgammonNN;

function createSeededArray(
  length: number,
  scale: number,
  seed?: number
): number[] {
  if (seed === undefined) {
    return new Array<number>(length).fill(0);
  }
  let s = seed | 0;
  const arr = new Array<number>(length);
  for (let i = 0; i < length; i++) {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    const u = ((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5;
    arr[i] = u * scale;
  }
  return arr;
}

function createResidualSubNetJSON(
  inDim: number,
  resDim: number,
  bottleneckDim: number,
  outDim: number,
  numBlocks: number,
  seed?: number
): NetworkWeightsJSON {
  const s = seed;
  const resBlocks: ResidualBlockWeightsJSON[] = [];
  for (let b = 0; b < numBlocks; b++) {
    resBlocks.push({
      dim: resDim,
      w1: createSeededArray(
        resDim * resDim,
        Math.sqrt(2 / resDim) * 0.2,
        s !== undefined ? s + 10 + b * 2 : undefined
      ),
      b1: new Array<number>(resDim).fill(0),
      w2: createSeededArray(
        resDim * resDim,
        Math.sqrt(2 / resDim) * 0.2,
        s !== undefined ? s + 11 + b * 2 : undefined
      ),
      b2: new Array<number>(resDim).fill(0),
    });
  }

  return {
    inDim,
    resDim,
    bottleneckDim,
    outDim,
    inW: createSeededArray(
      inDim * resDim,
      Math.sqrt(2 / inDim) * 0.25,
      s !== undefined ? s + 1 : undefined
    ),
    inB: new Array<number>(resDim).fill(0),
    resBlock: resBlocks[0],
    resBlocks,
    neckW: createSeededArray(
      resDim * bottleneckDim,
      Math.sqrt(2 / resDim) * 0.25,
      s !== undefined ? s + 4 : undefined
    ),
    neckB: new Array<number>(bottleneckDim).fill(0),
    outW: createSeededArray(
      bottleneckDim * outDim,
      Math.sqrt(2 / bottleneckDim) * 0.2,
      s !== undefined ? s + 5 : undefined
    ),
    outB: new Array<number>(outDim).fill(0),
  };
}

/**
 * Helper to construct a valid `DualBackgammonNN` instance (`version: 3` 260-Feature 3-Network Residual MLP)
 * with exact grandmaster architecture dimensions:
 * - `contact`: 260 -> 512 (SiLU) -> ResidualBlock(512) -> ResidualBlock(512) -> 256 (SiLU) -> 6 (Softmax)
 * - `crashed`: 260 -> 384 (SiLU) -> ResidualBlock(384) -> 192 (SiLU) -> 6 (Softmax)
 * - `race`:    260 -> 384 (SiLU) -> ResidualBlock(384) -> 192 (SiLU) -> 6 (Softmax)
 */
export function createUntrainedDualNN(
  seed?: number,
  trainedGames: number = 0
): DualBackgammonNN {
  return new DualBackgammonNN({
    version: 3,
    trainedGames,
    eloEstimate: 0,
    contact: createResidualSubNetJSON(
      NN_INPUT_DIM,
      512,
      256,
      6,
      2,
      seed !== undefined ? seed * 10 : undefined
    ),
    crashed: createResidualSubNetJSON(
      NN_INPUT_DIM,
      384,
      192,
      6,
      1,
      seed !== undefined ? seed * 10 + 100 : undefined
    ),
    race: createResidualSubNetJSON(
      NN_INPUT_DIM,
      384,
      192,
      6,
      1,
      seed !== undefined ? seed * 10 + 200 : undefined
    ),
  });
}

let activeNeuralNet: DualBackgammonNN | null = null;
let loadAttempted = false;
let inFlightWeightsPromise: Promise<BackgammonNNWeightsFile | null> | null =
  null;

export function getActiveNeuralNet(): DualBackgammonNN | null {
  return activeNeuralNet;
}

export function setActiveNeuralNet(nn: DualBackgammonNN | null): void {
  activeNeuralNet = nn;
}

/**
 * Initializes the active `DualBackgammonNN` directly from an already-fetched weights JSON object
 * (used to share weights across `gameplayWorker` and `analysisWorker` without duplicate downloads).
 */
export function initNeuralNetFromWeightsJSON(
  json: BackgammonNNWeightsFile | null | undefined
): DualBackgammonNN | null {
  if (!json) return activeNeuralNet;
  if ((json.contact || json.contactNet) && (json.race || json.raceNet)) {
    activeNeuralNet = new DualBackgammonNN(json);
    loadAttempted = true;
    return activeNeuralNet;
  }
  return activeNeuralNet;
}

/**
 * Fetches `/weights/backgammon_nn.json` once (deduplicating concurrent calls in the same context)
 * using a Network-First cache-busted request (`?t=timestamp`, falling back to cached URL offline).
 */
export async function fetchNeuralNetWeightsJSON(
  url = '/weights/backgammon_nn.json'
): Promise<BackgammonNNWeightsFile | null> {
  if (inFlightWeightsPromise) {
    return inFlightWeightsPromise;
  }
  if (typeof fetch === 'undefined') {
    return null;
  }

  inFlightWeightsPromise = (async () => {
    try {
      const separator = url.includes('?') ? '&' : '?';
      const bustUrl = `${url}${separator}t=${Date.now()}`;
      let res: Response;
      try {
        res = await fetch(bustUrl, { cache: 'no-store' });
      } catch {
        res = await fetch(url);
      }
      if (!res.ok) return null;
      const json = (await res.json()) as BackgammonNNWeightsFile;
      if (
        json &&
        (json.contact || json.contactNet) &&
        (json.race || json.raceNet)
      ) {
        initNeuralNetFromWeightsJSON(json);
        return json;
      }
    } catch {
      // Weights file not present yet; use hybrid heuristic + exact tablebase
    }
    return null;
  })();

  return inFlightWeightsPromise;
}

/**
 * Attempts to asynchronously fetch and activate `/weights/backgammon_nn.json`.
 */
export async function tryLoadNeuralNetWeights(
  url = '/weights/backgammon_nn.json'
): Promise<DualBackgammonNN | null> {
  if (activeNeuralNet) return activeNeuralNet;
  if (loadAttempted) return null;
  loadAttempted = true;

  const json = await fetchNeuralNetWeightsJSON(url);
  return initNeuralNetFromWeightsJSON(json);
}
