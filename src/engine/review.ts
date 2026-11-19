import type { HistoryEntry } from '../components/MoveHistory';
import type { MoveClassification } from './ai';
import type { Player } from './types';

export interface PlayerReviewStats {
  accuracy: number;
  avgEquityLoss: number;
  counts: Record<MoveClassification, number>;
  analyzedTurns: number;
}

export interface KeyMoment {
  turnIndex: number;
  turnNumber: number;
  player: Player;
  dice: [number, number];
  playedNotation: string;
  bestNotation: string;
  equityLoss: number;
  classification: MoveClassification;
  explanation: string[];
  isBiggestSwing?: boolean;
}

export interface EquityTimelinePoint {
  turnIndex: number;
  player: Player;
  equity: number;
  classification?: MoveClassification;
}

export interface GameReviewSummary {
  white: PlayerReviewStats;
  black: PlayerReviewStats;
  equityTimeline: EquityTimelinePoint[];
  keyMoments: KeyMoment[];
}

function createEmptyCounts(): Record<MoveClassification, number> {
  return {
    Best: 0,
    Good: 0,
    Inaccuracy: 0,
    Mistake: 0,
    Blunder: 0,
  };
}

/**
 * Computes comprehensive post-game / mid-game review statistics,
 * accuracy percentages, equity timeline, and key tactical swing moments.
 */
export function computeGameReview(history: HistoryEntry[]): GameReviewSummary {
  const counts: Record<Player, Record<MoveClassification, number>> = {
    white: createEmptyCounts(),
    black: createEmptyCounts(),
  };

  const totalLoss: Record<Player, number> = { white: 0, black: 0 };
  const nonForcedTurns: Record<Player, number> = { white: 0, black: 0 };
  const analyzedTurns: Record<Player, number> = { white: 0, black: 0 };

  const equityTimeline: EquityTimelinePoint[] = [];
  const keyMoments: KeyMoment[] = [];

  history.forEach((entry, turnIndex) => {
    const player = entry.player;
    const classification =
      entry.classification ?? entry.analysis?.playedMoveAnalysis?.classification;
    const equityLoss =
      entry.equityLoss ?? entry.analysis?.playedMoveAnalysis?.equityLoss ?? 0;

    equityTimeline.push({
      turnIndex,
      player,
      equity: entry.evalResult.equity,
      classification,
    });

    if (classification) {
      counts[player][classification] += 1;
      analyzedTurns[player] += 1;

      // Forced turns (where topMoves.length <= 1) do not penalize avgEquityLoss or accuracy
      const isForcedTurn =
        entry.analysis !== undefined
          ? entry.analysis.topMoves.length <= 1
          : entry.steps.length === 0;

      if (!isForcedTurn) {
        totalLoss[player] += equityLoss;
        nonForcedTurns[player] += 1;
      }

      if (
        classification === 'Inaccuracy' ||
        classification === 'Mistake' ||
        classification === 'Blunder'
      ) {
        const playedNotation =
          entry.analysis?.playedMoveAnalysis?.notation ?? entry.notation;
        const bestNotation =
          entry.analysis?.topMoves[0]?.notation ?? entry.notation;
        const explanation =
          entry.analysis?.playedMoveAnalysis?.explanation ??
          entry.analysis?.topMoves[0]?.explanation ??
          [];

        keyMoments.push({
          turnIndex,
          turnNumber: entry.turnNumber,
          player,
          dice: entry.dice,
          playedNotation,
          bestNotation,
          equityLoss: Number(equityLoss.toFixed(2)),
          classification,
          explanation,
        });
      }
    }
  });

  // Flag the single biggest equity swing among key moments
  if (keyMoments.length > 0) {
    let maxIdx = 0;
    for (let i = 1; i < keyMoments.length; i++) {
      if (keyMoments[i].equityLoss > keyMoments[maxIdx].equityLoss) {
        maxIdx = i;
      }
    }
    keyMoments[maxIdx].isBiggestSwing = true;
  }

  const buildPlayerStats = (player: Player): PlayerReviewStats => {
    const n = nonForcedTurns[player];
    const sumLoss = totalLoss[player];

    if (n === 0 || sumLoss === 0) {
      return {
        accuracy: 100,
        avgEquityLoss: 0,
        counts: counts[player],
        analyzedTurns: analyzedTurns[player],
      };
    }

    const rawAvg = sumLoss / n;
    const avgEquityLoss = Number(rawAvg.toFixed(3));
    const rawAccuracy = Math.max(
      0,
      Math.min(100, 100 * Math.exp(-3.2 * rawAvg))
    );
    const accuracy = Number(rawAccuracy.toFixed(1));

    return {
      accuracy,
      avgEquityLoss,
      counts: counts[player],
      analyzedTurns: analyzedTurns[player],
    };
  };

  return {
    white: buildPlayerStats('white'),
    black: buildPlayerStats('black'),
    equityTimeline,
    keyMoments,
  };
}
