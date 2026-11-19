import type { BoardState, Player } from './types';

/**
 * Total number of ways to distribute 0..15 checkers across 6 home-board points:
 * C(15 + 6, 6) = C(21, 6) = 54,264.
 */
export const BEAROFF_NUM_STATES = 54264;

/**
 * Maximum number of turns needed to bear off up to 15 checkers from the home board.
 */
export const BEAROFF_MAX_TURNS = 25;

export interface ExactBearOffEval {
  /** Cubeless equity from White's perspective in [-3.0, +3.0] */
  equity: number;
  /** White win probability percentage (0..100) */
  winProb: number;
  /** White gammon win probability percentage (0..100) */
  gammonProb: number;
  /** White backgammon win probability percentage (0..100, always 0 in pure bear-off) */
  backgammonProb: number;
  /** Black gammon win probability percentage (0..100) */
  loseGammonProb: number;
  /** Black backgammon win probability percentage (0..100, always 0 in pure bear-off) */
  loseBackgammonProb: number;
  /** Exact expected turns for White to bear off all remaining checkers */
  expectedTurnsWhite: number;
  /** Exact expected turns for Black to bear off all remaining checkers */
  expectedTurnsBlack: number;
  /** Alias for expectedTurnsWhite */
  whiteExpectedTurns: number;
  /** Alias for expectedTurnsBlack */
  blackExpectedTurns: number;
}

// Precomputed binomial coefficients C(n, k) for n <= 22, k <= 6
const BINOM: number[][] = (() => {
  const table: number[][] = Array.from({ length: 23 }, () =>
    new Array<number>(7).fill(0)
  );
  for (let n = 0; n <= 22; n++) {
    table[n][0] = 1;
    for (let k = 1; k <= Math.min(n, 6); k++) {
      table[n][k] = table[n - 1][k - 1] + table[n - 1][k];
    }
  }
  return table;
})();

// Cumulative state offset for each total checker count S = 0..16
// statesWithSum(S) = C(S + 5, 5)
const SUM_OFFSET: Int32Array = (() => {
  const offsets = new Int32Array(17);
  let acc = 0;
  for (let s = 0; s <= 15; s++) {
    offsets[s] = acc;
    acc += BINOM[s + 5][5];
  }
  offsets[16] = acc; // 54264
  return offsets;
})();

/**
 * Maps any 6-point home-board checker configuration `[c0, c1, c2, c3, c4, c5]`
 * (where `c0` is the 1-point and `c5` is the 6-point, and `sum(c) <= 15`)
 * to a unique canonical index in `0 .. 54263`.
 *
 * Ordering property:
 * States are ordered by total checkers `S = sum(c)` ascending (`0..15`),
 * and within the same `S`, lexicographically by `(c5, c4, c3, c2, c1)` ascending.
 * Every legal bear-off step strictly decreases the state index!
 */
export function indexHomeBoard(home: readonly number[]): number {
  const c0 = home[0] | 0;
  const c1 = home[1] | 0;
  const c2 = home[2] | 0;
  const c3 = home[3] | 0;
  const c4 = home[4] | 0;
  const c5 = home[5] | 0;

  const total = c0 + c1 + c2 + c3 + c4 + c5;
  if (total <= 0) return 0;
  if (total > 15) {
    throw new Error(`Invalid home board total checkers: ${total} > 15`);
  }

  let idx = SUM_OFFSET[total];
  let rem = total;

  // Rank c5 (0 .. rem), k = 4 remaining bars
  if (c5 > 0) {
    idx += BINOM[rem + 5][5] - BINOM[rem - c5 + 5][5];
    rem -= c5;
  }
  // Rank c4 (0 .. rem), k = 3 remaining bars
  if (c4 > 0) {
    idx += BINOM[rem + 4][4] - BINOM[rem - c4 + 4][4];
    rem -= c4;
  }
  // Rank c3 (0 .. rem), k = 2 remaining bars
  if (c3 > 0) {
    idx += BINOM[rem + 3][3] - BINOM[rem - c3 + 3][3];
    rem -= c3;
  }
  // Rank c2 (0 .. rem), k = 1 remaining bar
  if (c2 > 0) {
    idx += BINOM[rem + 2][2] - BINOM[rem - c2 + 2][2];
    rem -= c2;
  }
  // Rank c1 (0 .. rem)
  idx += c1;

  return idx;
}

/**
 * Inverse of `indexHomeBoard`: decodes a state index `0 .. 54263`
 * back into its 6-point home-board array `[c0, c1, c2, c3, c4, c5]`.
 */
export function decodeHomeBoardIndex(index: number): [
  number,
  number,
  number,
  number,
  number,
  number,
] {
  if (index <= 0) return [0, 0, 0, 0, 0, 0];

  let total = 0;
  while (total < 15 && SUM_OFFSET[total + 1] <= index) {
    total++;
  }

  let offset = index - SUM_OFFSET[total];
  let rem = total;

  const coords = [0, 0, 0, 0, 0]; // c5, c4, c3, c2, c1
  const dims = [4, 3, 2, 1];

  for (let dIdx = 0; dIdx < 4; dIdx++) {
    const k = dims[dIdx];
    let chosen = 0;
    for (let v = 0; v <= rem; v++) {
      const countForV = BINOM[rem - v + k][k];
      if (offset >= countForV) {
        offset -= countForV;
      } else {
        chosen = v;
        break;
      }
    }
    coords[dIdx] = chosen;
    rem -= chosen;
  }

  const c1 = offset;
  const c0 = rem - c1;

  return [c0, c1, coords[3], coords[2], coords[1], coords[0]];
}

interface BearOffDatabase {
  /** Flattened [54264 * 26] probability table P(T = k) for k = 0..25 */
  cdfTable: Float32Array;
  pdfTable: Float32Array;
  expectedTurns: Float32Array;
}

let cachedDatabase: BearOffDatabase | null = null;

/**
 * Generates all legal 1-die successor state indices from `counts` (`[c0..c5]`) for `die` (`1..6`).
 */
function getSingleDieSuccessors(
  c0: number,
  c1: number,
  c2: number,
  c3: number,
  c4: number,
  c5: number,
  die: number,
  outBuf: Int32Array
): number {
  const total = c0 + c1 + c2 + c3 + c4 + c5;
  if (total === 0) {
    outBuf[0] = 0;
    return 1;
  }

  const arr = [c0, c1, c2, c3, c4, c5];
  let highest = 5;
  while (highest >= 0 && arr[highest] === 0) {
    highest--;
  }

  let count = 0;
  for (let p = 0; p <= highest; p++) {
    if (arr[p] === 0) continue;

    if (p + 1 === die) {
      // Exact bear-off
      arr[p]--;
      outBuf[count++] = indexHomeBoard(arr);
      arr[p]++;
    } else if (p - die >= 0) {
      // Move checker within home board to p - die
      arr[p]--;
      arr[p - die]++;
      outBuf[count++] = indexHomeBoard(arr);
      arr[p - die]--;
      arr[p]++;
    } else if (p === highest && p + 1 < die) {
      // Overshoot bear-off from highest occupied point
      arr[p]--;
      outBuf[count++] = indexHomeBoard(arr);
      arr[p]++;
    }
  }

  return count;
}

/**
 * Lazily builds and caches the exact 54,264-state Dynamic Programming Bear-Off Tablebase.
 */
export function getBearOffDatabase(): BearOffDatabase {
  if (cachedDatabase) {
    return cachedDatabase;
  }

  const stride = BEAROFF_MAX_TURNS + 1; // 26
  const pdfTable = new Float32Array(BEAROFF_NUM_STATES * stride);
  const cdfTable = new Float32Array(BEAROFF_NUM_STATES * stride);
  const expectedTurns = new Float32Array(BEAROFF_NUM_STATES);

  // Best resulting state after playing 1, 2, or 3 identical dice d (0..5) from state s
  const best1 = new Int32Array(BEAROFF_NUM_STATES * 6);
  const best2 = new Int32Array(BEAROFF_NUM_STATES * 6);
  const best3 = new Int32Array(BEAROFF_NUM_STATES * 6);

  // Base case: state 0 (0 checkers left) takes 0 turns with probability 1.0
  pdfTable[0] = 1.0;
  for (let k = 0; k <= BEAROFF_MAX_TURNS; k++) {
    cdfTable[k] = 1.0;
  }
  expectedTurns[0] = 0;

  const succByDie = [
    new Int32Array(6),
    new Int32Array(6),
    new Int32Array(6),
    new Int32Array(6),
    new Int32Array(6),
    new Int32Array(6),
  ];
  const succCountByDie = new Int32Array(6);

  const chosenStates = new Int32Array(21);
  const chosenWeights = new Float32Array(21);

  let stateIdx = 1;
  for (let total = 1; total <= 15; total++) {
    for (let c5 = 0; c5 <= total; c5++) {
      const rem4 = total - c5;
      for (let c4 = 0; c4 <= rem4; c4++) {
        const rem3 = rem4 - c4;
        for (let c3 = 0; c3 <= rem3; c3++) {
          const rem2 = rem3 - c3;
          for (let c2 = 0; c2 <= rem2; c2++) {
            const rem1 = rem2 - c2;
            for (let c1 = 0; c1 <= rem1; c1++) {
              const c0 = rem1 - c1;

              // 1. Compute 1-die successors and best1 / best2 / best3 for each die d = 1..6
              for (let d = 0; d < 6; d++) {
                const sc = getSingleDieSuccessors(
                  c0,
                  c1,
                  c2,
                  c3,
                  c4,
                  c5,
                  d + 1,
                  succByDie[d]
                );
                succCountByDie[d] = sc;

                let b1 = 0;
                let b1Score = Infinity;
                let b2 = 0;
                let b2Score = Infinity;
                let b3 = 0;
                let b3Score = Infinity;

                const sArr = succByDie[d];
                for (let i = 0; i < sc; i++) {
                  const w1 = sArr[i];
                  const e1 = expectedTurns[w1];
                  if (e1 < b1Score) {
                    b1Score = e1;
                    b1 = w1;
                  }

                  const w2 = best1[w1 * 6 + d];
                  const e2 = expectedTurns[w2];
                  if (e2 < b2Score) {
                    b2Score = e2;
                    b2 = w2;
                  }

                  const w3 = best2[w1 * 6 + d];
                  const e3 = expectedTurns[w3];
                  if (e3 < b3Score) {
                    b3Score = e3;
                    b3 = w3;
                  }
                }

                const baseOffset = stateIdx * 6 + d;
                best1[baseOffset] = b1;
                best2[baseOffset] = b2;
                best3[baseOffset] = b3;
              }

              // 2. Find optimal target state for each of the 21 dice rolls
              let uniqueChosenCount = 0;

              for (let d1 = 0; d1 < 6; d1++) {
                for (let d2 = d1; d2 < 6; d2++) {
                  let bestTarget = 0;
                  let bestScore = Infinity;

                  if (d1 === d2) {
                    // Doubles: 4 moves of die d1
                    const sc = succCountByDie[d1];
                    const sArr = succByDie[d1];
                    for (let i = 0; i < sc; i++) {
                      const w1 = sArr[i];
                      const w4 = best3[w1 * 6 + d1];
                      const score = expectedTurns[w4];
                      if (score < bestScore) {
                        bestScore = score;
                        bestTarget = w4;
                      }
                    }
                  } else {
                    // Non-doubles: try d1 then d2, and d2 then d1
                    const sc1 = succCountByDie[d1];
                    const sArr1 = succByDie[d1];
                    for (let i = 0; i < sc1; i++) {
                      const w1 = sArr1[i];
                      const w2 = best1[w1 * 6 + d2];
                      const score = expectedTurns[w2];
                      if (score < bestScore) {
                        bestScore = score;
                        bestTarget = w2;
                      }
                    }

                    const sc2 = succCountByDie[d2];
                    const sArr2 = succByDie[d2];
                    for (let i = 0; i < sc2; i++) {
                      const w1 = sArr2[i];
                      const w2 = best1[w1 * 6 + d1];
                      const score = expectedTurns[w2];
                      if (score < bestScore) {
                        bestScore = score;
                        bestTarget = w2;
                      }
                    }
                  }

                  const prob = d1 === d2 ? 1 / 36 : 2 / 36;
                  let foundIdx = -1;
                  for (let u = 0; u < uniqueChosenCount; u++) {
                    if (chosenStates[u] === bestTarget) {
                      foundIdx = u;
                      break;
                    }
                  }
                  if (foundIdx !== -1) {
                    chosenWeights[foundIdx] += prob;
                  } else {
                    chosenStates[uniqueChosenCount] = bestTarget;
                    chosenWeights[uniqueChosenCount] = prob;
                    uniqueChosenCount++;
                  }
                }
              }

              // 3. Compute PDF P(T = k) = sum_v weight(v) * P_v(T = k - 1), CDF, and Expected Turns
              const rowOffset = stateIdx * stride;
              let expTurns = 1.0;

              for (let u = 0; u < uniqueChosenCount; u++) {
                const targetState = chosenStates[u];
                const weight = chosenWeights[u];
                expTurns += weight * expectedTurns[targetState];

                const targetRow = targetState * stride;
                for (let k = 1; k <= BEAROFF_MAX_TURNS; k++) {
                  pdfTable[rowOffset + k] += weight * pdfTable[targetRow + k - 1];
                }
              }

              expectedTurns[stateIdx] = expTurns;

              let runningCdf = 0;
              for (let k = 0; k <= BEAROFF_MAX_TURNS; k++) {
                runningCdf += pdfTable[rowOffset + k];
                cdfTable[rowOffset + k] = runningCdf > 1.0 ? 1.0 : runningCdf;
              }

              stateIdx++;
            }
          }
        }
      }
    }
  }

  cachedDatabase = {
    cdfTable,
    pdfTable,
    expectedTurns,
  };
  return cachedDatabase;
}

/**
 * Convolves White's and Black's exact bear-off turn distributions to return
 * 100% mathematically exact win, gammon, and cubeless equity values.
 */
export function evaluateExactBearOff(
  whiteHome: readonly number[],
  blackHome: readonly number[],
  turn?: Player
): ExactBearOffEval {
  const db = getBearOffDatabase();
  const stride = BEAROFF_MAX_TURNS + 1;

  const wIdx = indexHomeBoard(whiteHome);
  const bIdx = indexHomeBoard(blackHome);

  const wRow = wIdx * stride;
  const bRow = bIdx * stride;

  const expectedTurnsWhite = Number(db.expectedTurns[wIdx].toFixed(3));
  const expectedTurnsBlack = Number(db.expectedTurns[bIdx].toFixed(3));

  const whiteTotal =
    whiteHome[0] +
    whiteHome[1] +
    whiteHome[2] +
    whiteHome[3] +
    whiteHome[4] +
    whiteHome[5];
  const blackTotal =
    blackHome[0] +
    blackHome[1] +
    blackHome[2] +
    blackHome[3] +
    blackHome[4] +
    blackHome[5];

  if (whiteTotal === 0) {
    const isGammon = blackTotal === 15;
    return {
      equity: isGammon ? 2.0 : 1.0,
      winProb: 100,
      gammonProb: isGammon ? 100 : 0,
      backgammonProb: 0,
      loseGammonProb: 0,
      loseBackgammonProb: 0,
      expectedTurnsWhite: 0,
      expectedTurnsBlack,
      whiteExpectedTurns: 0,
      blackExpectedTurns: expectedTurnsBlack,
    };
  }

  if (blackTotal === 0) {
    const isGammon = whiteTotal === 15;
    return {
      equity: isGammon ? -2.0 : -1.0,
      winProb: 0,
      gammonProb: 0,
      backgammonProb: 0,
      loseGammonProb: isGammon ? 100 : 0,
      loseBackgammonProb: 0,
      expectedTurnsWhite,
      expectedTurnsBlack: 0,
      whiteExpectedTurns: expectedTurnsWhite,
      blackExpectedTurns: 0,
    };
  }

  // Check if home boards are identical (and > 1 checker so not an immediate 1-roll win)
  const isIdenticalMultiChecker =
    wIdx === bIdx && !(whiteTotal === 1 && whiteHome[0] === 1);

  let winProb01 = 0;

  if (!turn || isIdenticalMultiChecker) {
    // Symmetric simultaneous convolution: P(T_W < T_B) + 0.5 * P(T_W == T_B)
    for (let k = 1; k <= BEAROFF_MAX_TURNS; k++) {
      const pW = db.pdfTable[wRow + k];
      if (pW === 0) continue;
      const pB_greater = 1.0 - db.cdfTable[bRow + k];
      const pB_equal = db.pdfTable[bRow + k];
      winProb01 += pW * (pB_greater + 0.5 * pB_equal);
    }
  } else if (turn === 'white') {
    // White rolls first: White wins iff T_W <= T_B
    for (let k = 1; k <= BEAROFF_MAX_TURNS; k++) {
      const pW = db.pdfTable[wRow + k];
      if (pW === 0) continue;
      const pB_ge = 1.0 - db.cdfTable[bRow + k - 1];
      winProb01 += pW * pB_ge;
    }
  } else {
    // Black rolls first: White wins iff T_W < T_B
    for (let k = 1; k < BEAROFF_MAX_TURNS; k++) {
      const pW = db.pdfTable[wRow + k];
      if (pW === 0) continue;
      const pB_gt = 1.0 - db.cdfTable[bRow + k];
      winProb01 += pW * pB_gt;
    }
  }

  winProb01 = Math.max(0, Math.min(1, winProb01));

  // Gammon probability in pure bear-off:
  // Only possible if opponent still has all 15 checkers (0 borne off) and finishes in 1 turn while opponent cannot bear off
  const whiteGammon01 =
    blackTotal === 15 ? winProb01 * db.pdfTable[wRow + 1] : 0;
  const blackGammon01 =
    whiteTotal === 15 ? (1 - winProb01) * db.pdfTable[bRow + 1] : 0;

  const rawEquity =
    2 * winProb01 - 1 + (whiteGammon01 - blackGammon01);
  const clampedEquity = Math.max(-3.0, Math.min(3.0, rawEquity));
  const equity =
    Math.abs(clampedEquity) < 1e-6 ? 0 : Number(clampedEquity.toFixed(2));

  return {
    equity,
    winProb: Number((winProb01 * 100).toFixed(1)),
    gammonProb: Number((whiteGammon01 * 100).toFixed(1)),
    backgammonProb: 0,
    loseGammonProb: Number((blackGammon01 * 100).toFixed(1)),
    loseBackgammonProb: 0,
    expectedTurnsWhite,
    expectedTurnsBlack,
    whiteExpectedTurns: expectedTurnsWhite,
    blackExpectedTurns: expectedTurnsBlack,
  };
}

/**
 * Checks whether all remaining checkers of BOTH players are inside their respective home boards
 * (with 0 checkers on the bar and 0 checkers in the outer boards).
 */
export function isPureBearOffPosition(state: BoardState): boolean {
  if (state.bar.white > 0 || state.bar.black > 0) {
    return false;
  }
  // White checkers can only be on 0..5; Black checkers can only be on 18..23
  for (let i = 6; i < 24; i++) {
    if (state.points[i] > 0) return false;
  }
  for (let i = 0; i < 18; i++) {
    if (state.points[i] < 0) return false;
  }
  return true;
}

/**
 * Extracts a player's 6-element home-board checker counts `[c0, c1, c2, c3, c4, c5]`.
 */
export function extractPlayerHomeBoard(
  state: BoardState,
  player: Player
): [number, number, number, number, number, number] {
  if (player === 'white') {
    return [
      Math.max(0, state.points[0]),
      Math.max(0, state.points[1]),
      Math.max(0, state.points[2]),
      Math.max(0, state.points[3]),
      Math.max(0, state.points[4]),
      Math.max(0, state.points[5]),
    ];
  }
  return [
    Math.max(0, -state.points[23]),
    Math.max(0, -state.points[22]),
    Math.max(0, -state.points[21]),
    Math.max(0, -state.points[20]),
    Math.max(0, -state.points[19]),
    Math.max(0, -state.points[18]),
  ];
}

/**
 * Returns the exact expected number of turns required for `player` to bear off all remaining checkers:
 * - Uses exact 54,264-state tablebase lookup when all of `player`'s checkers are in the home board.
 * - Uses exact home-board projection + travel turns (`outsidePips / 8.167`) when checkers are still outside.
 */
export function getExactExpectedBearOffTurns(
  state: BoardState,
  player: Player
): number {
  if (state.borneOff[player] >= 15) {
    return 0;
  }

  const db = getBearOffDatabase();
  const home = extractPlayerHomeBoard(state, player);

  let outsideCheckers = state.bar[player];
  let outsidePips = state.bar[player] * 19;

  if (player === 'white') {
    for (let i = 6; i < 24; i++) {
      if (state.points[i] > 0) {
        outsideCheckers += state.points[i];
        outsidePips += state.points[i] * (i - 5);
      }
    }
  } else {
    for (let i = 0; i < 18; i++) {
      if (state.points[i] < 0) {
        const c = -state.points[i];
        outsideCheckers += c;
        outsidePips += c * (18 - i);
      }
    }
  }

  if (outsideCheckers === 0) {
    const idx = indexHomeBoard(home);
    return db.expectedTurns[idx];
  }

  // Distribute outsideCheckers onto the top home points (6-point, 5-point, 4-point) to get exact bear-off phase turns
  const projectedHome: [number, number, number, number, number, number] = [
    home[0],
    home[1],
    home[2],
    home[3],
    home[4],
    home[5],
  ];
  const homeSum =
    home[0] + home[1] + home[2] + home[3] + home[4] + home[5];
  let rem = Math.min(outsideCheckers, Math.max(0, 15 - homeSum));
  let pt = 5;
  while (rem > 0) {
    projectedHome[pt] += 1;
    rem -= 1;
    pt = pt === 3 ? 5 : pt - 1;
  }

  const idx = indexHomeBoard(projectedHome);
  const bearOffPhaseTurns = db.expectedTurns[idx];
  const travelTurns = outsidePips / 8.167;

  return bearOffPhaseTurns + travelTurns;
}
