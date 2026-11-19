import { describe, expect, it } from 'vitest';
import type { HistoryEntry } from '../components/MoveHistory';
import type { MoveClassification, TurnAnalysisResult } from './ai';
import { evaluatePosition } from './evaluator';
import { computeGameReview } from './review';
import { createInitialBoard } from './rules';
import type { Player } from './types';

function createMockHistoryEntry(params: {
  id: number;
  player: Player;
  dice: [number, number];
  notation: string;
  equity: number;
  classification: MoveClassification;
  equityLoss: number;
  candidateCount?: number;
  bestNotation?: string;
}): HistoryEntry {
  const baseBoard = createInitialBoard();
  const evalResult = {
    ...evaluatePosition(baseBoard),
    equity: params.equity,
  };

  const candidateCount = params.candidateCount ?? 3;
  const topMoves = Array.from({ length: candidateCount }).map((_, idx) => ({
    move: { steps: [], resultingState: baseBoard },
    notation: idx === 0 ? params.bestNotation || '24/18 13/9' : `13/${9 - idx}`,
    equity: params.equity + (idx === 0 ? params.equityLoss : 0),
    winProb: 50,
    gammonProb: 12,
    equityLoss: idx === 0 ? 0 : 0.05 * idx,
    combinatorics: {
      opponentHitRolls: 0,
      opponentHitPct: 0,
      opponentBarBlockRolls: 0,
      opponentBarBlockPct: 0,
      escapeRolls: 15,
      escapePct: 41.7,
    },
    explanation: ['Solid positional development'],
  }));

  const analysis: TurnAnalysisResult = {
    topMoves,
    playedMoveAnalysis: {
      notation: params.notation,
      equity: params.equity,
      equityLoss: params.equityLoss,
      classification: params.classification,
      explanation: ['Tactical move evaluation'],
    },
  };

  return {
    id: params.id,
    turnNumber: params.id,
    player: params.player,
    dice: params.dice,
    steps: [
      { from: 23, to: 17, dieUsed: params.dice[0], isHit: false },
    ],
    notation: `${params.dice[0]}-${params.dice[1]}: ${params.notation}`,
    turnStartState: baseBoard,
    resultingState: baseBoard,
    evalResult,
    classification: params.classification,
    equityLoss: params.equityLoss,
    analysis,
  };
}

describe('Game Review Suite (computeGameReview)', () => {
  it('returns 100% accuracy and 0 avgEquityLoss when a player makes zero equity loss', () => {
    const history: HistoryEntry[] = [
      createMockHistoryEntry({
        id: 1,
        player: 'white',
        dice: [3, 1],
        notation: '8/5 6/5',
        equity: 0.25,
        classification: 'Best',
        equityLoss: 0,
      }),
      createMockHistoryEntry({
        id: 2,
        player: 'black',
        dice: [6, 4],
        notation: '24/18 13/9',
        equity: 0.05,
        classification: 'Best',
        equityLoss: 0,
      }),
    ];

    const summary = computeGameReview(history);

    expect(summary.white.accuracy).toBe(100);
    expect(summary.white.avgEquityLoss).toBe(0);
    expect(summary.white.counts.Best).toBe(1);

    expect(summary.black.accuracy).toBe(100);
    expect(summary.black.avgEquityLoss).toBe(0);
    expect(summary.black.counts.Best).toBe(1);
    expect(summary.keyMoments).toHaveLength(0);
  });

  it('excludes forced turns (topMoves.length === 1) from penalizing avgEquityLoss and accuracy', () => {
    const history: HistoryEntry[] = [
      // Turn 1: White non-forced move with 0.20 loss
      createMockHistoryEntry({
        id: 1,
        player: 'white',
        dice: [5, 2],
        notation: '13/8 13/11',
        equity: -0.1,
        classification: 'Mistake',
        equityLoss: 0.2,
        candidateCount: 3,
      }),
      // Turn 2: White forced move (only 1 legal candidate)
      createMockHistoryEntry({
        id: 2,
        player: 'white',
        dice: [6, 6],
        notation: 'bar/18',
        equity: -0.1,
        classification: 'Best',
        equityLoss: 0,
        candidateCount: 1,
      }),
    ];

    const summaryWithoutForced = computeGameReview([history[0]]);
    const summaryWithForced = computeGameReview(history);

    // Forced turn must not dilute or change avgEquityLoss or accuracy
    expect(summaryWithForced.white.avgEquityLoss).toBe(
      summaryWithoutForced.white.avgEquityLoss
    );
    expect(summaryWithForced.white.accuracy).toBe(
      summaryWithoutForced.white.accuracy
    );
    // But the Best count tally still records the forced turn
    expect(summaryWithForced.white.counts.Best).toBe(1);
    expect(summaryWithForced.white.counts.Mistake).toBe(1);
    expect(summaryWithForced.white.analyzedTurns).toBe(2);
  });

  it('accurately tallies all 5 classifications and extracts keyMoments with biggest swing flagged', () => {
    const history: HistoryEntry[] = [
      createMockHistoryEntry({
        id: 1,
        player: 'white',
        dice: [3, 1],
        notation: '8/5 6/5',
        equity: 0.22,
        classification: 'Best',
        equityLoss: 0,
      }),
      createMockHistoryEntry({
        id: 2,
        player: 'black',
        dice: [4, 2],
        notation: '13/9 13/11',
        equity: 0.31,
        classification: 'Inaccuracy',
        equityLoss: 0.08,
        bestNotation: '8/4 6/4',
      }),
      createMockHistoryEntry({
        id: 3,
        player: 'white',
        dice: [6, 5],
        notation: '13/7 13/8',
        equity: 0.14,
        classification: 'Mistake',
        equityLoss: 0.16,
        bestNotation: '24/13',
      }),
      createMockHistoryEntry({
        id: 4,
        player: 'black',
        dice: [5, 3],
        notation: '24/19 24/21',
        equity: 0.55,
        classification: 'Blunder',
        equityLoss: 0.38,
        bestNotation: '8/3 6/3',
      }),
      createMockHistoryEntry({
        id: 5,
        player: 'white',
        dice: [4, 3],
        notation: '13/9 13/10',
        equity: 0.52,
        classification: 'Good',
        equityLoss: 0.03,
      }),
    ];

    const summary = computeGameReview(history);

    // Check White counts
    expect(summary.white.counts).toEqual({
      Best: 1,
      Good: 1,
      Inaccuracy: 0,
      Mistake: 1,
      Blunder: 0,
    });

    // Check Black counts
    expect(summary.black.counts).toEqual({
      Best: 0,
      Good: 0,
      Inaccuracy: 1,
      Mistake: 0,
      Blunder: 1,
    });

    // Check Key Moments (Inaccuracy on turn 2, Mistake on turn 3, Blunder on turn 4)
    expect(summary.keyMoments).toHaveLength(3);
    expect(summary.keyMoments.map((k) => k.turnNumber)).toEqual([2, 3, 4]);

    // Turn 4 (loss 0.38) should be marked as isBiggestSwing
    const biggest = summary.keyMoments.find((k) => k.isBiggestSwing);
    expect(biggest).toBeDefined();
    expect(biggest!.turnNumber).toBe(4);
    expect(biggest!.classification).toBe('Blunder');
    expect(biggest!.bestNotation).toBe('8/3 6/3');
  });
});
