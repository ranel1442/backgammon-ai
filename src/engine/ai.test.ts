import { describe, expect, it } from 'vitest';
import {
  ALL_21_DICE_OUTCOMES,
  analyzePosition,
  computeExactDiceCombinatorics,
  formatTurnNotation,
  getClassificationMeta,
} from './ai';
import { evaluatePosition } from './evaluator';
import {
  getOpeningBookRollEntries,
  isInitialBoardState,
  isSecondTurnReplyState,
} from './openingBook';
import { createInitialBoard } from './rules';
import type { BoardState, Player } from './types';

function createEmptyBoard(currentTurn: Player = 'white'): BoardState {
  return {
    points: new Array<number>(24).fill(0),
    bar: { white: 0, black: 0 },
    borneOff: { white: 0, black: 0 },
    currentTurn,
    dice: [],
    remainingMoves: [],
  };
}

/**
 * Mirrors a BoardState across the board (point i <-> 23 - i) and swaps White <-> Black.
 */
function mirrorBoard(state: BoardState): BoardState {
  const mirroredPoints = new Array<number>(24).fill(0);
  for (let i = 0; i < 24; i++) {
    const val = state.points[23 - i];
    mirroredPoints[i] = val === 0 ? 0 : -val;
  }

  return {
    points: mirroredPoints,
    bar: {
      white: state.bar.black,
      black: state.bar.white,
    },
    borneOff: {
      white: state.borneOff.black,
      black: state.borneOff.white,
    },
    currentTurn: state.currentTurn === 'white' ? 'black' : 'white',
    dice: [...state.dice],
    remainingMoves: [...state.remainingMoves],
  };
}

describe('Backgammon Multi-Factor Evaluator, Opening Book & Deep Expectiminimax AI', () => {
  describe('evaluatePosition symmetry, EPC & Kleinman race features', () => {
    it('evaluates the initial starting board at 0.00 equity and 50.0% winProb', () => {
      const board = createInitialBoard();
      const res = evaluatePosition(board);

      expect(res.equity).toBe(0);
      expect(res.winProb).toBe(50);
      expect(res.gammonProb).toBe(0);
      expect(res.backgammonProb).toBe(0);
      expect(res.features.isPureRace).toBe(false);
      expect(res.features.pipDiff).toBe(0);
      expect(res.features.effectivePipDiff).toBe(0);
    });

    it('is strictly antisymmetric when swapping and mirroring White and Black across all features', () => {
      const board = createEmptyBoard('white');
      // Asymmetric contact position with primes, blots, bar, builders, trapped back checkers, and borne-off checkers
      board.points[2] = 2;
      board.points[3] = 2;
      board.points[4] = 3;
      board.points[5] = 2; // White 4-prime in home board
      board.points[8] = 2; // White builder point in outfield
      board.points[10] = 1; // White blot
      board.points[20] = 2; // White defensive anchor
      board.borneOff.white = 1;

      board.points[19] = -2;
      board.points[18] = -2;
      board.points[14] = -1; // Black blot
      board.points[7] = -4;
      board.points[1] = -2; // Black back checkers trapped behind White's 4-prime
      board.bar.black = 1;
      board.borneOff.black = 2;

      const evalOriginal = evaluatePosition(board);
      const evalMirrored = evaluatePosition(mirrorBoard(board));

      expect(evalOriginal.equity).toBeGreaterThan(0);
      expect(evalMirrored.equity).toBe(-evalOriginal.equity);
      expect(evalMirrored.features.pipDiff).toBe(-evalOriginal.features.pipDiff);
      expect(evalMirrored.features.effectivePipDiff).toBe(
        -evalOriginal.features.effectivePipDiff
      );
      expect(evalMirrored.features.homePointsDiff).toBe(
        -evalOriginal.features.homePointsDiff
      );
      expect(evalMirrored.features.primeLengthDiff).toBe(
        -evalOriginal.features.primeLengthDiff
      );
      expect(evalMirrored.features.blotDangerDiff).toBe(
        -evalOriginal.features.blotDangerDiff
      );
      expect(evalMirrored.features.trappedCheckersDiff).toBe(
        -evalOriginal.features.trappedCheckersDiff
      );
      expect(evalMirrored.features.escapeChancesDiff).toBe(
        -evalOriginal.features.escapeChancesDiff
      );
      expect(evalMirrored.features.buildersDiff).toBe(
        -evalOriginal.features.buildersDiff
      );
      expect(evalMirrored.features.isPureRace).toBe(
        evalOriginal.features.isPureRace
      );
    });

    it('accurately detects Contact vs. Pure Race positions', () => {
      const contactBoard = createInitialBoard();
      expect(evaluatePosition(contactBoard).features.isPureRace).toBe(false);

      const raceBoard = createEmptyBoard('white');
      // White checkers in indices 0..5, Black checkers in indices 18..23 (completely past each other)
      raceBoard.points[4] = 5;
      raceBoard.points[2] = 5;
      raceBoard.borneOff.white = 5;

      raceBoard.points[19] = -5;
      raceBoard.points[21] = -5;
      raceBoard.borneOff.black = 5;

      expect(evaluatePosition(raceBoard).features.isPureRace).toBe(true);

      // Putting a checker on the bar turns it back into a contact position
      raceBoard.bar.white = 1;
      expect(evaluatePosition(raceBoard).features.isPureRace).toBe(false);
    });

    it('penalizes deep home-board wastage via Effective Pip Count (EPC) and Kleinman formula in pure race', () => {
      const raceBoard = createEmptyBoard('white');
      // Both players have 6 checkers left and identical raw pip count = 24 pips:
      // White: 2 checkers on 6-point (12), 2 on 4-point (8), 2 on 2-point (4) -> 24 pips, clean distribution
      raceBoard.points[5] = 2;
      raceBoard.points[3] = 2;
      raceBoard.points[1] = 2;
      raceBoard.borneOff.white = 9;

      // Black: 1 checker on 19-point (6 pips), 1 on 21-point (4 pips), 2 on 22-point (6 pips), 8 on 24-point (8 pips)
      // Wait: let's keep 9 borne off for Black too, and give Black 24 raw pips with heavy stacking on 1-point (index 23):
      // Black: 2 checkers on index 18 (6-pt = 12 pips), 1 on index 19 (5-pt = 5 pips), 1 on index 20 (4-pt = 4 pips),
      // Wait: 12 + 5 + 4 = 21 + 3 checkers on index 23 (1-pt = 3 pips) -> 6 checkers, 24 pips!
      // Or even simpler:
      // White: 4 checkers on 4-point (index 3 -> 4*4 = 16 pips), 2 checkers on 3-point (index 2 -> 2*3 = 6 pips), 1 on 2-point (index 1 -> 2 pips) = 7 checkers, 24 pips.
      // Black: 2 checkers on 6-point (index 18 -> 12 pips), 1 on 5-point (index 19 -> 5 pips), 7 on 1-point (index 23 -> 7 pips) = 10 checkers, 24 pips.
      const cleanWhiteVsStackedBlack = createEmptyBoard('white');
      // Give both exactly 8 checkers on board and 24 raw pips:
      // White: 2 on 5-pt (10), 2 on 4-pt (8), 2 on 2-pt (4), 2 on 1-pt (2) = 8 checkers, 24 pips (zero excess on 1,2,3!)
      cleanWhiteVsStackedBlack.points[4] = 2;
      cleanWhiteVsStackedBlack.points[3] = 2;
      cleanWhiteVsStackedBlack.points[1] = 2;
      cleanWhiteVsStackedBlack.points[0] = 2;
      cleanWhiteVsStackedBlack.borneOff.white = 7;

      // Black: 3 on 6-pt (index 18 -> 18 pips), 6 on 1-pt (index 23 -> 6 pips) = 9 checkers, 24 pips (4 excess buried on 1-pt + gaps on 4,5!)
      cleanWhiteVsStackedBlack.points[18] = -3;
      cleanWhiteVsStackedBlack.points[23] = -6;
      cleanWhiteVsStackedBlack.borneOff.black = 6;

      const evalRes = evaluatePosition(cleanWhiteVsStackedBlack);
      expect(evalRes.features.isPureRace).toBe(true);
      expect(evalRes.features.pipDiff).toBe(0); // Raw pips are both 24!
      expect(evalRes.features.effectivePipDiff).toBeGreaterThan(5); // White leads EPC significantly due to Black's wastage
      expect(evalRes.equity).toBeGreaterThan(0.25);
      expect(evalRes.winProb).toBeGreaterThan(60);
    });

    it('penalizes exposed blots facing direct opponent shots and rewards primes', () => {
      const safePrimeBoard = createEmptyBoard('white');
      safePrimeBoard.points[3] = 2;
      safePrimeBoard.points[4] = 2;
      safePrimeBoard.points[5] = 2;
      safePrimeBoard.points[6] = 2; // 4-prime
      safePrimeBoard.points[12] = -2;
      safePrimeBoard.points[2] = -2; // Black checkers trapped behind White's 4-prime

      const blotBoard = createEmptyBoard('white');
      blotBoard.points[1] = 2;
      blotBoard.points[3] = 1; // Blot exposed to Black at index 2!
      blotBoard.points[5] = 1; // Blot exposed to Black at index 2!
      blotBoard.points[6] = 1; // Blot exposed to Black at index 2!
      blotBoard.points[8] = 3;
      blotBoard.points[12] = -2;
      blotBoard.points[2] = -2;

      const safeEval = evaluatePosition(safePrimeBoard);
      const blotEval = evaluatePosition(blotBoard);

      expect(safeEval.features.primeLengthDiff).toBeGreaterThan(
        blotEval.features.primeLengthDiff
      );
      expect(safeEval.features.trappedCheckersDiff).toBeGreaterThan(0);
      expect(blotEval.features.blotDangerDiff).toBeGreaterThan(
        safeEval.features.blotDangerDiff
      );
      expect(safeEval.equity).toBeGreaterThan(blotEval.equity);
    });
  });

  describe('Opening Book & Deep Expectiminimax Search', () => {
    it('precomputes all 21 distinct dice rolls summing to probability 1.0', () => {
      expect(ALL_21_DICE_OUTCOMES).toHaveLength(21);
      const totalProb = ALL_21_DICE_OUTCOMES.reduce(
        (sum, item) => sum + item.probability,
        0
      );
      expect(totalProb).toBeCloseTo(1.0, 6);
    });

    it('covers all 15 non-double opening rolls in the theoretical Opening Book', () => {
      const board = createInitialBoard();
      expect(isInitialBoardState(board)).toBe(true);

      for (let d1 = 1; d1 <= 6; d1++) {
        for (let d2 = d1 + 1; d2 <= 6; d2++) {
          const entries = getOpeningBookRollEntries([d1, d2]);
          expect(entries.length).toBeGreaterThanOrEqual(2);
          expect(entries[0].rank).toBe(1);
          expect(entries[0].explanation.length).toBeGreaterThanOrEqual(1);
        }
      }
    });

    it('ranks opening 3-1, 6-1, 4-2, and 6-5 according to opening theory with Hebrew explanations', () => {
      const board = createInitialBoard();

      // 3-1 -> 8/5 6/5
      const res31 = analyzePosition(board, [3, 1], 2, [
        { from: 7, to: 4, dieUsed: 3, isHit: false },
        { from: 5, to: 4, dieUsed: 1, isHit: false },
      ]);
      expect(res31.topMoves[0].notation).toBe('8/5 6/5');
      expect(res31.topMoves[0].equityLoss).toBe(0);
      expect(/[\u0590-\u05FF]/.test(res31.topMoves[0].explanation[0])).toBe(
        true
      );
      expect(getClassificationMeta('Best').label).toBe('המהלך הטוב ביותר');
      expect(getClassificationMeta('Blunder').label).toBe('טעות גסה');
      expect(res31.playedMoveAnalysis?.classification).toBe('Best');

      // 6-1 -> 13/7 8/7 (making bar point)
      const res61 = analyzePosition(board, [6, 1], 2);
      expect(res61.topMoves[0].notation).toBe('13/7 8/7');

      // 4-2 -> 8/4 6/4 (making 4-point)
      const res42 = analyzePosition(board, [4, 2], 2);
      expect(res42.topMoves[0].notation).toBe('8/4 6/4');

      // 6-5 -> Lover's Leap (reaching point 13 from 24)
      const res65 = analyzePosition(board, [6, 5], 2);
      expect(['24/18 18/13', '24/19 19/13']).toContain(
        res65.topMoves[0].notation
      );
      expect(res65.topMoves[0].isBookMove).toBe(true);
      expect(res65.topMoves[0].bookBadgeLabel).toBe('ספר פתיחות');
    });

    it('matches 2nd-turn Reply Book moves canonically for both Black and White', () => {
      // Turn 1: White plays 3-1 (8/5 6/5 -> points[7]=2, points[5]=4, points[4]=2)
      const afterWhite31 = createInitialBoard();
      afterWhite31.points[7] = 2;
      afterWhite31.points[5] = 4;
      afterWhite31.points[4] = 2;
      afterWhite31.currentTurn = 'black';

      expect(isInitialBoardState(afterWhite31)).toBe(false);
      expect(isSecondTurnReplyState(afterWhite31)).toBe(true);

      // Black replies on Turn 2 with 6-1 (making bar point 13/7 8/7 in Black's relative notation)
      const blackReply61 = analyzePosition(afterWhite31, [6, 1], 2);
      expect(blackReply61.topMoves[0].notation).toBe('13/7 8/7');
      expect(blackReply61.topMoves[0].isBookMove).toBe(true);
      expect(blackReply61.topMoves[0].bookBadgeLabel).toBe('ספר פתיחות');

      // Mirror the position so Black opened 3-1 and now White is on Turn 2 reply with 6-1
      const afterBlack31 = mirrorBoard(afterWhite31);
      expect(afterBlack31.currentTurn).toBe('white');
      expect(isSecondTurnReplyState(afterBlack31)).toBe(true);

      const whiteReply61 = analyzePosition(afterBlack31, [6, 1], 2);
      expect(whiteReply61.topMoves[0].notation).toBe('13/7 8/7');
      expect(whiteReply61.topMoves[0].isBookMove).toBe(true);
      expect(whiteReply61.topMoves[0].equity).toBeCloseTo(
        -blackReply61.topMoves[0].equity,
        2
      );
    });

    it('classifies an obvious blunder when leaving blots and passing up a key hit/point', () => {
      const board = createEmptyBoard('white');
      board.points[4] = 1; // White blot on 5-point
      board.points[6] = 1; // White blot on 7-point
      board.points[20] = 2; // Safe anchor
      board.points[3] = -1; // Black blot on 4-point
      board.points[1] = -2;
      board.points[2] = -2;

      const blunderSteps = [
        { from: 20, to: 18, dieUsed: 2, isHit: false },
        { from: 20, to: 19, dieUsed: 1, isHit: false },
      ];

      const analysis = analyzePosition(board, [2, 1], 1, blunderSteps);
      expect(analysis.playedMoveAnalysis).toBeDefined();
      expect(analysis.playedMoveAnalysis!.equityLoss).toBeGreaterThan(0.1);
      expect(['Inaccuracy', 'Mistake', 'Blunder']).toContain(
        analysis.playedMoveAnalysis!.classification
      );
      expect(formatTurnNotation(blunderSteps, 'white')).toBe('21/19 21/20');
      expect(analysis.gamePhaseLabel?.length).toBeGreaterThan(0);
      expect(analysis.strategicPlan?.length).toBeGreaterThan(0);
      expect(analysis.runnerUpComparison).toBeDefined();
    });

    it('computes exact 36-roll dice combinatorics for opponent hit shots, bar blocks, and escapes', () => {
      const initial = createInitialBoard();
      const combInit = computeExactDiceCombinatorics(initial, 'white');
      expect(combInit.opponentHitRolls).toBeGreaterThanOrEqual(0);
      expect(combInit.opponentHitRolls).toBeLessThanOrEqual(36);
      expect(combInit.opponentBarBlockRolls).toBe(0);
      expect(combInit.escapeRolls).toBeGreaterThanOrEqual(0);
      expect(combInit.escapeRolls).toBeLessThanOrEqual(36);

      // Closed home board for White (points 0..5 all have >= 2 White checkers) with Black on the bar -> 36/36 bar block
      const closedOut = createEmptyBoard('white');
      for (let pt = 0; pt <= 5; pt++) {
        closedOut.points[pt] = 2;
      }
      closedOut.borneOff.white = 3;
      closedOut.bar.black = 1;
      closedOut.points[18] = -14;

      const combClosed = computeExactDiceCombinatorics(closedOut, 'white');
      expect(combClosed.opponentBarBlockRolls).toBe(36);
      expect(combClosed.opponentBarBlockPct).toBe(100);
      expect(combClosed.opponentHitRolls).toBe(0);
    });

    it('supports all 5 search depths (1, 2+, 3+, 4+, 5+) with Transposition Table and Anytime Cutoff', () => {
      const raceBoard = createEmptyBoard('white');
      raceBoard.points[4] = 2;
      raceBoard.points[2] = 2;
      raceBoard.borneOff.white = 11;
      raceBoard.points[19] = -2;
      raceBoard.points[21] = -2;
      raceBoard.borneOff.black = 11;

      for (const depth of [1, 2, 3, 4, 5] as const) {
        const res = analyzePosition(raceBoard, [4, 2], depth);
        expect(res.requestedDepth).toBe(depth);
        expect(res.completedDepth).toBeGreaterThanOrEqual(1);
        expect(res.topMoves.length).toBeGreaterThan(0);
        expect(res.elapsedMs).toBeLessThan(9600);
      }
    });
  });
});
