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
  /** Flattened row-major weights of length `outFeatures * inFeatures` */
  weights: number[];
  /** Bias vector of length `outFeatures` */
  bias: number[];
}

export interface ResidualBlockWeightsJSON {
  dim: number;
  w1: number[];
  b1: number[];
  w2: number[];
  b2: number[];
}

export interface ResidualSubNetWeightsJSON {
  inDim: number;
  resDim: number;
  bottleneckDim: number;
  outDim: number;
  inW: number[];
  inB: number[];
  resBlock?: ResidualBlockWeightsJSON;
  resBlocks?: ResidualBlockWeightsJSON[];
  neckW: number[];
  neckB: number[];
  outW: number[];
  outB: number[];
}

export interface NetworkWeightsJSON {
  layers?: LayerWeightsJSON[];
  inDim?: number;
  h1Dim?: number;
  h2Dim?: number;
  outDim?: number;
  w1?: number[];
  b1?: number[];
  w2?: number[];
  b2?: number[];
  w3?: number[];
  b3?: number[];
  // Version 2 / Version 3 Residual fields
  resDim?: number;
  bottleneckDim?: number;
  inW?: number[];
  inB?: number[];
  resBlock?: ResidualBlockWeightsJSON;
  resBlocks?: ResidualBlockWeightsJSON[];
  neckW?: number[];
  neckB?: number[];
  outW?: number[];
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

/**
 * Extracts the Ultimate 260-Feature Grandmaster canonical representation (`version: 3`)
 * for `perspective` ('white' | 'black').
 *
 * Feature Layout (100% identical between TypeScript and Python Numba `@njit`):
 * - [0..191]   (192 features): 24 points * 8 units, ordered from `perspective`'s 24-point down to 1-point.
 * - [192..209] (18 features):  Upgraded global features including exact bar blockage `(closed/6)^2`,
 *                              hit lethality `blotExposure * (oppClosed/6)`, timing ratio, double-anchor & 20-pt ownership.
 * - [210..233] (24 features):  Stripped Point indicators (`+1` if self has `count === 2`, `-1` if opp has `count === 2`).
 * - [234..257] (24 features):  Spare Builder indicators (`+1` if self has `count === 3 || 4`, `-1` if opp has `count === 3 || 4`).
 * - [258..259] (2 features):   Prime Trapping & Lethality products (`primeLen * trappedBackCheckers / 18`).
 */
export function extractFeatures260(
  state: BoardState,
  perspective: Player
): Float32Array {
  const features = new Float32Array(NN_INPUT_DIM);
  const isWhite = perspective === 'white';
  const opponent: Player = isWhite ? 'black' : 'white';

  const getRelCount = (rel: number): number => {
    const absIdx = isWhite ? rel : 23 - rel;
    const raw = state.points[absIdx];
    return isWhite ? raw : -raw;
  };

  // 1. [0..191]: 192 Point features + [210..257]: 48 Stripped & Spare Builder features
  for (let step = 0; step < 24; step++) {
    const rel = 23 - step;
    const count = getRelCount(rel);
    const base = step * 8;

    if (count > 0) {
      features[base] = 1.0;
      if (count >= 2) features[base + 1] = 1.0;
      if (count >= 3) features[base + 2] = 1.0;
      if (count > 3) features[base + 3] = (count - 3) * 0.5;

      if (count === 2) {
        features[210 + step] = 1.0;
      } else if (count === 3 || count === 4) {
        features[234 + step] = 1.0;
      }
    } else if (count < 0) {
      const oppCount = -count;
      features[base + 4] = 1.0;
      if (oppCount >= 2) features[base + 5] = 1.0;
      if (oppCount >= 3) features[base + 6] = 1.0;
      if (oppCount > 3) features[base + 7] = (oppCount - 3) * 0.5;

      if (oppCount === 2) {
        features[210 + step] = -1.0;
      } else if (oppCount === 3 || oppCount === 4) {
        features[234 + step] = -1.0;
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
    const c = getRelCount(rel);
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
      const c = getRelCount(rel);
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
  const golden20Owned = getRelCount(19) >= 2 ? 1.0 : 0.0;
  const doubleAnchorIndicator = anchorsInOppHomeSelf >= 2 ? 1.0 : 0.0;
  const timingRatio = Math.min(
    2.0,
    outfieldAndBackPipSelf / Math.max(1.0, pipOpp)
  );

  features[192] = barSelf / 2.0;
  features[193] = barOpp / 2.0;
  features[194] = borneOffSelf / 15.0;
  features[195] = borneOffOpp / 15.0;
  features[196] = pipSelf / 167.0;
  features[197] = pipOpp / 167.0;
  features[198] = (pipOpp - pipSelf) / 50.0;
  features[199] = exactTurnsSelf / 20.0;
  features[200] = exactTurnsOpp / 20.0;
  features[201] = Math.min(6, maxPrimeSelf) / 6.0;
  features[202] = Math.min(6, maxPrimeOpp) / 6.0;
  features[203] = closedRatioSelf * closedRatioSelf; // Exact bar blockage prob by Self
  features[204] = closedRatioOpp * closedRatioOpp; // Exact bar blockage prob by Opp
  features[205] = blotRatioSelf * closedRatioOpp; // Hit lethality against Self
  features[206] = blotRatioOpp * closedRatioSelf; // Hit lethality against Opp
  features[207] = isContact ? 1.0 : 0.0;
  features[208] = timingRatio / 2.0; // Timing / Backgame viability ratio
  features[209] = 0.5 * doubleAnchorIndicator + 0.5 * golden20Owned;

  // [258..259]: Prime Trapping & Lethality products
  const primeTrapSelf = Math.min(6, maxPrimeSelf) * Math.min(3, oppBackCheckers);
  const primeTrapOpp = Math.min(6, maxPrimeOpp) * Math.min(3, selfBackCheckers);
  features[258] = primeTrapSelf / 18.0;
  features[259] = primeTrapOpp / 18.0;

  return features;
}

/**
 * Legacy 210-dimensional feature extractor preserved for backward compatibility
 * with Version 1 and Version 2 weight files.
 */
export function extractFeatures210(
  state: BoardState,
  perspective: Player
): Float32Array {
  const features = new Float32Array(NN_INPUT_DIM_V2);
  const isWhite = perspective === 'white';
  const opponent: Player = isWhite ? 'black' : 'white';

  const getRelCount = (rel: number): number => {
    const absIdx = isWhite ? rel : 23 - rel;
    const raw = state.points[absIdx];
    return isWhite ? raw : -raw;
  };

  for (let step = 0; step < 24; step++) {
    const rel = 23 - step;
    const count = getRelCount(rel);
    const base = step * 8;

    if (count > 0) {
      features[base] = 1.0;
      if (count >= 2) features[base + 1] = 1.0;
      if (count >= 3) features[base + 2] = 1.0;
      if (count > 3) features[base + 3] = (count - 3) * 0.5;
    } else if (count < 0) {
      const oppCount = -count;
      features[base + 4] = 1.0;
      if (oppCount >= 2) features[base + 5] = 1.0;
      if (oppCount >= 3) features[base + 6] = 1.0;
      if (oppCount > 3) features[base + 7] = (oppCount - 3) * 0.5;
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
    const c = getRelCount(rel);
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
      const c = getRelCount(rel);
      if (c === 1 && rel > lowestOppRel) {
        blotExposureSelf++;
      } else if (c === -1 && rel < highestSelfRel) {
        blotExposureOpp++;
      }
    }

    for (let rel = 18; rel <= 23; rel++) {
      const c = getRelCount(rel);
      if (c > 0) {
        let blockedFront = 0;
        for (let s = 1; s <= 6; s++) {
          if (rel - s >= 0 && getRelCount(rel - s) <= -2) blockedFront++;
        }
        if (blockedFront >= 3) trappedSelf += c;
      }
    }

    for (let rel = 0; rel <= 5; rel++) {
      const c = getRelCount(rel);
      if (c < 0) {
        let blockedFront = 0;
        for (let s = 1; s <= 6; s++) {
          if (rel + s < 24 && getRelCount(rel + s) >= 2) blockedFront++;
        }
        if (blockedFront >= 3) trappedOpp += -c;
      }
    }
  }

  features[192] = barSelf / 2.0;
  features[193] = barOpp / 2.0;
  features[194] = borneOffSelf / 15.0;
  features[195] = borneOffOpp / 15.0;
  features[196] = pipSelf / 167.0;
  features[197] = pipOpp / 167.0;
  features[198] = (pipOpp - pipSelf) / 50.0;
  features[199] = exactTurnsSelf / 20.0;
  features[200] = exactTurnsOpp / 20.0;
  features[201] = Math.min(6, maxPrimeSelf) / 6.0;
  features[202] = Math.min(6, maxPrimeOpp) / 6.0;
  features[203] = homePointsSelf / 6.0;
  features[204] = homePointsOpp / 6.0;
  features[205] = Math.min(6, blotExposureSelf) / 6.0;
  features[206] = Math.min(6, blotExposureOpp) / 6.0;
  features[207] = isContact ? 1.0 : 0.0;
  features[208] = Math.min(4, trappedSelf) / 4.0;
  features[209] = Math.min(4, trappedOpp) / 4.0;

  return features;
}

function makeCompiledLayer(
  inFeatures: number,
  outFeatures: number,
  weights: number[],
  bias: number[]
): CompiledLayer {
  const expectedLen = inFeatures * outFeatures;
  if (weights.length !== expectedLen) {
    throw new Error(
      `Layer weight length mismatch: expected ${expectedLen}, got ${weights.length}`
    );
  }
  if (bias.length !== outFeatures) {
    throw new Error(
      `Layer bias length mismatch: expected ${outFeatures}, got ${bias.length}`
    );
  }
  return {
    inFeatures,
    outFeatures,
    weights: new Float32Array(weights),
    bias: new Float32Array(bias),
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
    Array.isArray(netJson.inW) &&
    Array.isArray(netJson.inB) &&
    rawBlocks &&
    Array.isArray(netJson.neckW) &&
    Array.isArray(netJson.neckB) &&
    Array.isArray(netJson.outW) &&
    Array.isArray(netJson.outB)
  ) {
    const { inDim, resDim, bottleneckDim, outDim } = netJson;
    const compiledBlocks: CompiledResBlock[] = rawBlocks.map((rb) => ({
      layer1: makeCompiledLayer(resDim, resDim, rb.w1, rb.b1),
      layer2: makeCompiledLayer(resDim, resDim, rb.w2, rb.b2),
    }));

    return {
      kind: 'residual',
      inDim,
      inLayer: makeCompiledLayer(inDim, resDim, netJson.inW, netJson.inB),
      resBlocks: compiledBlocks,
      neckLayer: makeCompiledLayer(
        resDim,
        bottleneckDim,
        netJson.neckW,
        netJson.neckB
      ),
      outLayer: makeCompiledLayer(
        bottleneckDim,
        outDim,
        netJson.outW,
        netJson.outB
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

function applyLinear(
  layer: CompiledLayer,
  input: Float32Array,
  applySiLU: boolean
): Float32Array {
  const { inFeatures, outFeatures, weights, bias } = layer;
  const out = new Float32Array(outFeatures);
  for (let o = 0; o < outFeatures; o++) {
    let sum = bias[o];
    const rowOffset = o * inFeatures;
    for (let i = 0; i < inFeatures; i++) {
      sum += weights[rowOffset + i] * input[i];
    }
    out[o] = applySiLU ? sum / (1.0 + Math.exp(-sum)) : sum;
  }
  return out;
}

/**
 * Converts raw output logits (either 6 logits `[WinSingle, WinGammonOnly, WinBg, LoseSingle, LoseGammonOnly, LoseBg]`
 * or 5 logits with implicit `LoseSingle = 0.0`) into normalized `NNOutcomeProbabilities`.
 */
function logitsToOutcomeProbabilities(logits: Float32Array): NNOutcomeProbabilities {
  let lWinSingle = 0;
  let lWinGammonOnly = 0;
  let lWinBg = 0;
  let lLoseSingle = 0;
  let lLoseGammonOnly = 0;
  let lLoseBg = 0;

  if (logits.length >= 6) {
    lWinSingle = logits[0];
    lWinGammonOnly = logits[1];
    lWinBg = logits[2];
    lLoseSingle = logits[3];
    lLoseGammonOnly = logits[4];
    lLoseBg = logits[5];
  } else {
    lWinSingle = logits[0] ?? 0;
    lWinGammonOnly = logits[1] ?? 0;
    lWinBg = logits[2] ?? 0;
    lLoseSingle = 0.0;
    lLoseGammonOnly = logits[3] ?? 0;
    lLoseBg = logits[4] ?? 0;
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

  return {
    winProb01,
    pWinSingle,
    pWinGammonOnly,
    pWinGammon,
    pWinBg,
    pLoseSingle,
    pLoseGammonOnly,
    pLoseGammon,
    pLoseBg,
    equity: Math.max(-3.0, Math.min(3.0, equity)),
  };
}

/**
 * Executes a forward pass through a Version 3 / Version 2 Residual Sub-Network
 * (supporting 1, 2, or more stacked `ResidualBlock`s) or a Version 1 MLP Sub-Network.
 */
function runSubNetForward(
  subNet: CompiledResidualSubNet,
  input: Float32Array
): NNOutcomeProbabilities {
  if (
    subNet.kind === 'residual' &&
    subNet.inLayer &&
    subNet.resBlocks &&
    subNet.neckLayer &&
    subNet.outLayer
  ) {
    // 1. Input projection: inDim -> resDim (SiLU)
    let x = applyLinear(subNet.inLayer, input, true);

    // 2. Stacked ResidualBlock(resDim)s: x = x + SiLU(W2 * SiLU(W1 * x + b1) + b2)
    for (const block of subNet.resBlocks) {
      const h1 = applyLinear(block.layer1, x, true);
      const h2 = applyLinear(block.layer2, h1, true);
      const nextX = new Float32Array(x.length);
      for (let i = 0; i < x.length; i++) {
        nextX[i] = x[i] + h2[i];
      }
      x = nextX;
    }

    // 3. Bottleneck projection: resDim -> bottleneckDim (SiLU)
    const neck = applyLinear(subNet.neckLayer, x, true);

    // 4. Output head: bottleneckDim -> 6 logits
    const logits = applyLinear(subNet.outLayer, neck, false);
    return logitsToOutcomeProbabilities(logits);
  }

  // Fallback for Version 1 MLP layers
  let current = input;
  const layers = subNet.layers ?? [];
  for (let l = 0; l < layers.length; l++) {
    const isLast = l === layers.length - 1;
    current = applyLinear(layers[l], current, !isLast);
  }
  return logitsToOutcomeProbabilities(current);
}

/**
 * 3-Network Residual Grandmaster Backgammon Neural Network (`version: 3`, backward-compatible with `version: 2` and `version: 1`):
 * - `contact`: 260 -> 512 (SiLU) -> ResidualBlock(512) -> ResidualBlock(512) -> 256 (SiLU) -> 6 (Softmax)
 * - `crashed`: 260 -> 384 (SiLU) -> ResidualBlock(384) -> 192 (SiLU) -> 6 (Softmax)
 * - `race`:    260 -> 384 (SiLU) -> ResidualBlock(384) -> 192 (SiLU) -> 6 (Softmax)
 */
export class DualBackgammonNN {
  private readonly contactNet: CompiledResidualSubNet;
  private readonly crashedNet: CompiledResidualSubNet;
  private readonly raceNet: CompiledResidualSubNet;
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

  public getVersion(): number {
    return this.metadata.version;
  }

  private selectSubNet(phase: GamePhase): CompiledResidualSubNet {
    if (phase === 'race') return this.raceNet;
    if (phase === 'crashed') return this.crashedNet;
    return this.contactNet;
  }

  /**
   * Evaluates a board state from `perspective`'s viewpoint using the specialized
   * Residual sub-network for the position's game phase (`contact` | `crashed` | `race`).
   * Automatically dispatches 260-feature (v3) or 210-feature (v1/v2) extraction based on `subNet.inDim`.
   */
  public forwardPerspective(
    state: BoardState,
    perspective: Player,
    phaseOrIsPureRace?: GamePhase | boolean
  ): NNOutcomeProbabilities {
    const phase: GamePhase =
      typeof phaseOrIsPureRace === 'string'
        ? phaseOrIsPureRace
        : typeof phaseOrIsPureRace === 'boolean'
          ? phaseOrIsPureRace
            ? 'race'
            : classifyGamePhase(state) === 'crashed'
              ? 'crashed'
              : 'contact'
          : classifyGamePhase(state);

    const subNet = this.selectSubNet(phase);
    const features =
      subNet.inDim === NN_INPUT_DIM
        ? extractFeatures260(state, perspective)
        : extractFeatures210(state, perspective);
    return runSubNetForward(subNet, features);
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
   * White's perspective forward pass and Black's inverted perspective forward pass.
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
    const phase: GamePhase =
      typeof phaseOrIsPureRace === 'string'
        ? phaseOrIsPureRace
        : typeof phaseOrIsPureRace === 'boolean'
          ? phaseOrIsPureRace
            ? 'race'
            : classifyGamePhase(state) === 'crashed'
              ? 'crashed'
              : 'contact'
          : classifyGamePhase(state);

    const whitePred = this.forwardPerspective(state, 'white', phase);
    const blackPred = this.forwardPerspective(state, 'black', phase);

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
export function createUntrainedDualNN(seed?: number): DualBackgammonNN {
  return new DualBackgammonNN({
    version: 3,
    trainedGames: 0,
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

export function getActiveNeuralNet(): DualBackgammonNN | null {
  return activeNeuralNet;
}

export function setActiveNeuralNet(nn: DualBackgammonNN | null): void {
  activeNeuralNet = nn;
}

/**
 * Attempts to asynchronously fetch `/weights/backgammon_nn.json` using a Network-First
 * cache-busted request (`?t=timestamp`, falling back to cached URL when offline)
 * so newly exported weights from Python GPU training immediately appear in the UI.
 */
export async function tryLoadNeuralNetWeights(
  url = '/weights/backgammon_nn.json'
): Promise<DualBackgammonNN | null> {
  if (activeNeuralNet) return activeNeuralNet;
  if (loadAttempted) return null;
  loadAttempted = true;

  if (typeof fetch === 'undefined') {
    return null;
  }

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
      activeNeuralNet = new DualBackgammonNN(json);
      return activeNeuralNet;
    }
  } catch {
    // Weights file not present yet; use hybrid heuristic + exact tablebase
  }
  return null;
}
