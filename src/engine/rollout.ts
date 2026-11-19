import {
  evaluateExactBearOff,
  extractPlayerHomeBoard,
  isPureBearOffPosition,
} from './bearoffTablebase';
import { evaluatePosition } from './evaluator';
import { formatTurnNotation, generateLegalTurnMoves } from './rules';
import type { BoardState, Player, TurnMove } from './types';

export interface RolloutMoveResult {
  moveIndex: number;
  notation: string;
  equity: number;
  stdError: number;
  ci95: number;
  winProb: number;
  gammonProb: number;
  backgammonProb: number;
  trials: number;
}

/**
 * All 36 equiprobable ordered dice outcomes (d1, d2) for stratified Common Random Numbers.
 */
const ORDERED_DICE_36: ReadonlyArray<[number, number]> = (() => {
  const pairs: Array<[number, number]> = [];
  for (let d1 = 1; d1 <= 6; d1++) {
    for (let d2 = 1; d2 <= 6; d2++) {
      pairs.push([d1, d2]);
    }
  }
  return pairs;
})();

/**
 * Fast deterministic PRNG (Mulberry32) for Common Random Numbers across candidate moves.
 */
function createSeededRng(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Ultra-fast allocation-free board scorer (from White's perspective) used as the
 * rollout policy to pick strong moves at high speed inside Monte Carlo playouts.
 */
function scoreBoardFastForPolicy(state: BoardState): number {
  if (state.borneOff.white >= 15) return 1000;
  if (state.borneOff.black >= 15) return -1000;

  const pts = state.points;
  let score =
    (state.borneOff.white - state.borneOff.black) * 4.5 -
    (state.bar.white - state.bar.black) * 6.5;

  let whitePips = state.bar.white * 25;
  let blackPips = state.bar.black * 25;
  let whiteHomeMade = 0;
  let blackHomeMade = 0;
  let whiteRun = 0;
  let blackRun = 0;
  let whiteMaxPrime = 0;
  let blackMaxPrime = 0;

  for (let i = 0; i < 24; i++) {
    const v = pts[i];
    if (v > 0) {
      whitePips += v * (i + 1);
      whiteRun++;
      if (whiteRun > whiteMaxPrime) whiteMaxPrime = whiteRun;
      blackRun = 0;

      if (v >= 2) {
        if (i <= 5) whiteHomeMade += i === 4 || i === 3 ? 1.35 : 1.0;
        else if (i === 6) score += 0.9; // bar point
        else if (i >= 18) score += 0.75; // defensive anchor
      } else if (v === 1) {
        // Blot exposure penalty (higher in opponent home / outer board)
        score -= i >= 12 ? 1.15 : 0.65;
      }
    } else if (v < 0) {
      const count = -v;
      blackPips += count * (24 - i);
      blackRun++;
      if (blackRun > blackMaxPrime) blackMaxPrime = blackRun;
      whiteRun = 0;

      if (count >= 2) {
        if (i >= 18) blackHomeMade += i === 19 || i === 20 ? 1.35 : 1.0;
        else if (i === 17) score -= 0.9; // bar point
        else if (i <= 5) score -= 0.75; // defensive anchor
      } else if (count === 1) {
        score += i <= 11 ? 1.15 : 0.65;
      }
    } else {
      whiteRun = 0;
      blackRun = 0;
    }
  }

  score += (blackPips - whitePips) * 0.11;
  score += (whiteHomeMade - blackHomeMade) * 1.1;
  score += (Math.min(6, whiteMaxPrime) - Math.min(6, blackMaxPrime)) * 0.85;
  return score;
}

function selectRolloutTurnMove(moves: TurnMove[], player: Player): TurnMove {
  if (moves.length === 1) return moves[0];
  let bestMove = moves[0];
  let bestScore =
    player === 'white' ? -Infinity : Infinity;

  for (let i = 0; i < moves.length; i++) {
    const m = moves[i];
    const s = scoreBoardFastForPolicy(m.resultingState);
    if (player === 'white' ? s > bestScore : s < bestScore) {
      bestScore = s;
      bestMove = m;
    }
  }
  return bestMove;
}

interface SingleTrialOutcome {
  equity: number; // Always from White's perspective [-3.0, +3.0]
  whiteWinProb: number; // [0..100]
  gammonProb: number; // [0..100]
  backgammonProb: number; // [0..100]
}

function evaluateTerminalOutcome(state: BoardState): SingleTrialOutcome | null {
  if (state.borneOff.white >= 15) {
    const blackOff = state.borneOff.black;
    if (blackOff > 0) {
      return { equity: 1.0, whiteWinProb: 100, gammonProb: 0, backgammonProb: 0 };
    }
    const isBg =
      state.bar.black > 0 || state.points.slice(0, 6).some((p) => p < 0);
    return {
      equity: isBg ? 3.0 : 2.0,
      whiteWinProb: 100,
      gammonProb: 100,
      backgammonProb: isBg ? 100 : 0,
    };
  }

  if (state.borneOff.black >= 15) {
    const whiteOff = state.borneOff.white;
    if (whiteOff > 0) {
      return { equity: -1.0, whiteWinProb: 0, gammonProb: 0, backgammonProb: 0 };
    }
    const isBg =
      state.bar.white > 0 || state.points.slice(18, 24).some((p) => p > 0);
    return {
      equity: isBg ? -3.0 : -2.0,
      whiteWinProb: 0,
      gammonProb: 100,
      backgammonProb: isBg ? 100 : 0,
    };
  }

  return null;
}

/**
 * Simulates a single rollout trajectory starting after a candidate move has been applied,
 * using a deterministic dice sequence for Common Random Numbers variance reduction.
 */
function simulateSingleTrial(
  afterMoveState: BoardState,
  nextTurn: Player,
  trialIndex: number,
  maxPlies: number
): SingleTrialOutcome {
  const immediateTerminal = evaluateTerminalOutcome(afterMoveState);
  if (immediateTerminal) return immediateTerminal;

  let simState: BoardState = {
    points: [...afterMoveState.points],
    bar: { ...afterMoveState.bar },
    borneOff: { ...afterMoveState.borneOff },
    currentTurn: nextTurn,
    dice: [],
    remainingMoves: [],
  };

  if (isPureBearOffPosition(simState)) {
    const exact = evaluateExactBearOff(
      extractPlayerHomeBoard(simState, 'white'),
      extractPlayerHomeBoard(simState, 'black'),
      simState.currentTurn
    );
    return {
      equity: exact.equity,
      whiteWinProb: exact.winProb,
      gammonProb: exact.gammonProb,
      backgammonProb: 0,
    };
  }

  const rng = createSeededRng(104729 + trialIndex * 7919);

  for (let ply = 0; ply < maxPlies; ply++) {
    // Stratify the first opponent roll across all 36 ordered dice outcomes
    const dicePair: [number, number] =
      ply === 0
        ? ORDERED_DICE_36[trialIndex % 36]
        : ORDERED_DICE_36[Math.floor(rng() * 36)];

    const legalMoves = generateLegalTurnMoves(simState, dicePair);
    const activeTurn = simState.currentTurn;
    const nextPlayer: Player = activeTurn === 'white' ? 'black' : 'white';

    if (legalMoves.length > 0) {
      const chosen = selectRolloutTurnMove(legalMoves, activeTurn);
      simState = {
        points: [...chosen.resultingState.points],
        bar: { ...chosen.resultingState.bar },
        borneOff: { ...chosen.resultingState.borneOff },
        currentTurn: nextPlayer,
        dice: [],
        remainingMoves: [],
      };

      const term = evaluateTerminalOutcome(simState);
      if (term) return term;

      // As soon as the rollout reaches a pure bear-off state, resolve with the exact 54,264-state DP tablebase!
      if (isPureBearOffPosition(simState)) {
        const exact = evaluateExactBearOff(
          extractPlayerHomeBoard(simState, 'white'),
          extractPlayerHomeBoard(simState, 'black'),
          simState.currentTurn
        );
        return {
          equity: exact.equity,
          whiteWinProb: exact.winProb,
          gammonProb: exact.gammonProb,
          backgammonProb: 0,
        };
      }
    } else {
      simState = {
        ...simState,
        currentTurn: nextPlayer,
      };
    }
  }

  // Truncated rollout leaf evaluation (uses NeuralNet or Hybrid + Exact Bear-Off Tablebase)
  const leafEval = evaluatePosition(simState);
  return {
    equity: leafEval.equity,
    whiteWinProb: leafEval.winProb,
    gammonProb: leafEval.gammonProb,
    backgammonProb: 0,
  };
}

/**
 * Runs a variance-reduced Monte Carlo Rollout across the candidate moves using
 * Common Random Numbers (identical stratified dice sequences across all candidates)
 * and exact bear-off tablebase cutoffs.
 */
export function runMonteCarloRollout(
  state: BoardState,
  dice: [number, number],
  candidateMoves: TurnMove[],
  trialsPerMove: number = 1000,
  onProgress?: (
    progressPct: number,
    partialResults: RolloutMoveResult[]
  ) => void
): RolloutMoveResult[] {
  const numCandidates = candidateMoves.length;
  if (numCandidates === 0) return [];

  const totalTrials = Math.max(36, trialsPerMove);
  const player = state.currentTurn;
  const nextTurn: Player = player === 'white' ? 'black' : 'white';
  const maxPlies = 24;

  // Running Welford online variance accumulators per candidate
  const meanEquity = new Float64Array(numCandidates);
  const m2Equity = new Float64Array(numCandidates);
  const sumWinProb = new Float64Array(numCandidates);
  const sumGammonProb = new Float64Array(numCandidates);
  const sumBgProb = new Float64Array(numCandidates);
  const notations = candidateMoves.map((m) =>
    formatTurnNotation(player, dice, m.steps)
  );

  const buildSnapshot = (completedTrials: number): RolloutMoveResult[] => {
    const n = Math.max(1, completedTrials);
    return candidateMoves.map((_, idx) => {
      const variance = n > 1 ? m2Equity[idx] / (n - 1) : 0;
      const stdDev = Math.sqrt(Math.max(0, variance));
      const stdError = stdDev / Math.sqrt(n);
      const ci95 = 1.96 * stdError;
      const rawWhiteWin = sumWinProb[idx] / n;
      const playerWinProb =
        player === 'white' ? rawWhiteWin : 100 - rawWhiteWin;

      return {
        moveIndex: idx,
        notation: notations[idx],
        equity: Number(meanEquity[idx].toFixed(3)),
        stdError: Number(stdError.toFixed(3)),
        ci95: Number(ci95.toFixed(3)),
        winProb: Number(playerWinProb.toFixed(1)),
        gammonProb: Number((sumGammonProb[idx] / n).toFixed(1)),
        backgammonProb: Number((sumBgProb[idx] / n).toFixed(1)),
        trials: n,
      };
    });
  };

  const batchSize = Math.max(25, Math.floor(totalTrials / 10));

  for (let t = 0; t < totalTrials; t++) {
    const count = t + 1;
    for (let c = 0; c < numCandidates; c++) {
      const outcome = simulateSingleTrial(
        candidateMoves[c].resultingState,
        nextTurn,
        t,
        maxPlies
      );

      // Welford's algorithm for numerically stable online mean & variance
      const delta = outcome.equity - meanEquity[c];
      meanEquity[c] += delta / count;
      const delta2 = outcome.equity - meanEquity[c];
      m2Equity[c] += delta * delta2;

      sumWinProb[c] += outcome.whiteWinProb;
      sumGammonProb[c] += outcome.gammonProb;
      sumBgProb[c] += outcome.backgammonProb;
    }

    if (
      onProgress &&
      (count % batchSize === 0 || count === totalTrials)
    ) {
      const pct = Math.min(100, Math.round((count / totalTrials) * 100));
      onProgress(pct, buildSnapshot(count));
    }
  }

  return buildSnapshot(totalTrials);
}
