import { describe, expect, it } from 'vitest';
import { evaluateBaseline } from './evaluator';
import {
  createInitialBoard,
  formatTurnNotation,
  generateLegalTurnMoves,
  getPipCount,
  getValidDestinationsForSource,
  getValidNextSteps,
} from './rules';
import type { BoardState } from './types';

function createEmptyBoard(currentTurn: 'white' | 'black' = 'white'): BoardState {
  return {
    points: new Array<number>(24).fill(0),
    bar: { white: 0, black: 0 },
    borneOff: { white: 0, black: 0 },
    currentTurn,
    dice: [],
    remainingMoves: [],
  };
}

describe('Backgammon Engine & Rules', () => {
  describe('createInitialBoard', () => {
    it('sets up standard starting positions for white and black', () => {
      const board = createInitialBoard();

      expect(board.points).toHaveLength(24);
      expect(board.currentTurn).toBe('white');
      expect(board.bar).toEqual({ white: 0, black: 0 });
      expect(board.borneOff).toEqual({ white: 0, black: 0 });
      expect(board.dice).toEqual([]);
      expect(board.remainingMoves).toEqual([]);

      // White positions (positive)
      expect(board.points[23]).toBe(2);
      expect(board.points[12]).toBe(5);
      expect(board.points[7]).toBe(3);
      expect(board.points[5]).toBe(5);

      // Black positions (negative)
      expect(board.points[0]).toBe(-2);
      expect(board.points[11]).toBe(-5);
      expect(board.points[16]).toBe(-3);
      expect(board.points[18]).toBe(-5);

      // Total checker counts
      const whiteCount = board.points
        .filter((p) => p > 0)
        .reduce((sum, p) => sum + p, 0);
      const blackCount = board.points
        .filter((p) => p < 0)
        .reduce((sum, p) => sum + Math.abs(p), 0);

      expect(whiteCount).toBe(15);
      expect(blackCount).toBe(15);
    });
  });

  describe('getPipCount', () => {
    it('returns 167 for both players on the initial board', () => {
      const board = createInitialBoard();
      expect(getPipCount(board, 'white')).toBe(167);
      expect(getPipCount(board, 'black')).toBe(167);
    });

    it('includes checkers on the bar as 25 pips each', () => {
      const board = createEmptyBoard('white');
      board.points[0] = 2; // 2 * 1 = 2 pips for white
      board.bar.white = 1; // 1 * 25 = 25 pips for white

      board.points[22] = -3; // 3 * (24 - 22) = 6 pips for black
      board.bar.black = 2; // 2 * 25 = 50 pips for black

      expect(getPipCount(board, 'white')).toBe(27);
      expect(getPipCount(board, 'black')).toBe(56);
    });
  });

  describe('Blocked points & Hitting / Entering from the bar', () => {
    it('blocks landing on points with 2 or more opponent checkers', () => {
      const board = createEmptyBoard('white');
      board.points[10] = 1;
      board.points[8] = -2; // Blocked for die 2 (10 -> 8)
      board.points[6] = -3; // Blocked for die 4 (10 -> 6)

      const moves = generateLegalTurnMoves(board, [2, 4]);
      expect(moves).toHaveLength(0);
    });

    it('hits a single opponent blot to the bar', () => {
      const board = createEmptyBoard('white');
      board.points[10] = 1;
      board.points[7] = -1; // Black blot at index 7 (10 - 3)

      const moves = generateLegalTurnMoves(board, [3, 5]);
      // Possible paths for single checker at 10:
      // 1) 10 -> 7 (hit!) -> 2
      // 2) 10 -> 5 -> 2 (no hit!)
      // Both result in different board states because path 1 puts black on the bar!
      expect(moves).toHaveLength(2);

      const hitMove = moves.find((m) => m.steps.some((s) => s.isHit));
      const nonHitMove = moves.find((m) => m.steps.every((s) => !s.isHit));

      expect(hitMove).toBeDefined();
      expect(hitMove!.resultingState.bar.black).toBe(1);
      expect(hitMove!.resultingState.points[7]).toBe(0);
      expect(hitMove!.resultingState.points[2]).toBe(1);

      expect(nonHitMove).toBeDefined();
      expect(nonHitMove!.resultingState.bar.black).toBe(0);
      expect(nonHitMove!.resultingState.points[7]).toBe(-1);
      expect(nonHitMove!.resultingState.points[2]).toBe(1);
    });

    it('forces checkers on the bar to enter before any other checker can move', () => {
      const board = createEmptyBoard('white');
      board.bar.white = 1;
      board.points[10] = 1;
      // Block entry on 24 - 6 = 18, leave entry on 24 - 2 = 22 open
      board.points[18] = -2;

      const moves = generateLegalTurnMoves(board, [6, 2]);
      expect(moves.length).toBeGreaterThan(0);

      // Every legal turn move MUST start by entering from the bar with die 2 to index 22
      for (const move of moves) {
        expect(move.steps).toHaveLength(2);
        expect(move.steps[0]).toEqual({
          from: 'bar',
          to: 22,
          dieUsed: 2,
          isHit: false,
        });
        expect(move.steps[1].dieUsed).toBe(6);
      }
    });

    it('prevents moving board checkers when a second checker remains stuck on the bar', () => {
      const board = createEmptyBoard('black');
      board.bar.black = 2;
      board.points[10] = -1;
      // Black enters at die - 1. Leave index 0 (die 1) open, block index 3 (die 4).
      board.points[3] = 2;

      const moves = generateLegalTurnMoves(board, [1, 4]);
      expect(moves).toHaveLength(1);
      expect(moves[0].steps).toHaveLength(1);
      expect(moves[0].steps[0]).toEqual({
        from: 'bar',
        to: 0,
        dieUsed: 1,
        isHit: false,
      });
      expect(moves[0].resultingState.bar.black).toBe(1);
    });
  });

  describe('Doubles', () => {
    it('gives 4 moves of the rolled die value and deduplicates identical resulting states', () => {
      const board = createEmptyBoard('white');
      board.points[23] = 2;

      const moves = generateLegalTurnMoves(board, [3, 3]);
      // 2 checkers at 23 with four 3s can partition 4 moves between the 2 checkers:
      // (4, 0) -> 1 checker at 11, 1 at 23
      // (3, 1) -> 1 checker at 14, 1 at 20
      // (2, 2) -> 2 checkers at 17
      expect(moves).toHaveLength(3);
      for (const move of moves) {
        expect(move.steps).toHaveLength(4);
        expect(move.steps.every((s) => s.dieUsed === 3)).toBe(true);
      }
    });

    it('uses as many of the 4 dice as possible when partially blocked', () => {
      const board = createEmptyBoard('white');
      board.points[10] = 1;
      // Die is 2: 10 -> 8 -> 6 are open, but index 4 is blocked by black
      board.points[4] = -2;

      const moves = generateLegalTurnMoves(board, [2, 2]);
      expect(moves).toHaveLength(1);
      expect(moves[0].steps).toHaveLength(2);
      expect(moves[0].resultingState.points[6]).toBe(1);
    });
  });

  describe('Bearing off (exact and overshoot)', () => {
    it('disallows bearing off when a checker is outside the home board or on the bar', () => {
      const board = createEmptyBoard('white');
      board.points[0] = 1; // In home board (1-point)
      board.points[6] = 1; // Outside home board (7-point)
      // Block index 6 from moving with 2 or 3 (indices 4 and 3 blocked)
      board.points[4] = -2;
      board.points[3] = -2;

      const moves = generateLegalTurnMoves(board, [2, 3]);
      // Neither checker can move (index 0 cannot bear off while index 6 is outside home board)
      expect(moves).toHaveLength(0);
    });

    it('allows exact bear-off and overshoot bear-off from the highest occupied point', () => {
      const board = createEmptyBoard('white');
      board.points[3] = 1; // 4-point (exact for die 4)
      board.points[1] = 1; // 2-point (overshoot for die 6 once 4-point is cleared)

      const moves = generateLegalTurnMoves(board, [4, 6]);
      expect(moves).toHaveLength(1);
      expect(moves[0].steps).toHaveLength(2);
      expect(moves[0].resultingState.borneOff.white).toBe(2);
    });

    it('forbids overshoot bear-off from a lower point when a higher point is occupied', () => {
      const board = createEmptyBoard('white');
      board.points[4] = 1; // 5-point
      board.points[1] = 1; // 2-point
      // Roll [4, 4] or [4, 1]: with die 4, white MUST move index 4 -> index 0, cannot bear off index 1 with die 4
      const moves = generateLegalTurnMoves(board, [4, 1]);

      // Possible legal sequences for [4, 1]:
      // - 4 moves 4->0, 1 bears off 0->off (leaves checker on 1)
      // - 4 moves 4->0, 1 moves 1->0 (leaves 2 checkers on 0)
      // - 1 moves 4->3, 4 bears off 3->off (leaves checker on 1; same state as first!)
      // In NO legal move can BOTH checkers be borne off or index 1 be borne off with die 4 while index 4 is occupied
      for (const move of moves) {
        expect(move.resultingState.borneOff.white).toBeLessThanOrEqual(1);
        const illegalOvershoot = move.steps.some(
          (s) => s.from === 1 && s.to === 'off' && s.dieUsed === 4
        );
        expect(illegalOvershoot).toBe(false);
      }
    });

    it('supports black bearing off (indices 18-23) with exact and overshoot dice', () => {
      const board = createEmptyBoard('black');
      board.points[20] = -1; // 4 pips away (exact for die 4)
      board.points[22] = -1; // 2 pips away (overshoot for die 5 once index 20 is borne off)

      const moves = generateLegalTurnMoves(board, [4, 5]);
      expect(moves).toHaveLength(1);
      expect(moves[0].resultingState.borneOff.black).toBe(2);
    });
  });

  describe('Forced two-dice-usage & Forced higher-die rules', () => {
    it('enforces using both dice when one move choice blocks the second die', () => {
      const board = createEmptyBoard('white');
      board.points[10] = 1; // Checker A
      board.points[5] = 1; // Checker B

      // Roll [4, 2]:
      // If Checker A plays 4 (10 -> 6), then 2 is blocked for both A (6 -> 4 blocked) and B (5 -> 3 blocked).
      // If Checker A plays 2 (10 -> 8) and Checker B plays 4 (5 -> 1), BOTH dice are used!
      board.points[4] = -2; // Blocks 6 -> 4
      board.points[3] = -2; // Blocks 5 -> 3
      // Also block 10 -> 8 -> 4 (already blocked at 4) and 5 -> 1 -> -1 (cannot bear off since A is at 10/8)

      const moves = generateLegalTurnMoves(board, [4, 2]);
      expect(moves).toHaveLength(1);
      expect(moves[0].steps).toHaveLength(2);
      expect(moves[0].resultingState.points[8]).toBe(1);
      expect(moves[0].resultingState.points[1]).toBe(1);
    });

    it('enforces the forced higher-die rule when only 1 die can be played', () => {
      const board = createEmptyBoard('white');
      board.points[10] = 1;
      // Roll [2, 5]:
      // 10 -> 8 (die 2) is open
      // 10 -> 5 (die 5) is open
      // 10 -> 3 (die 2 + 5) is BLOCKED so only 1 die can be played!
      board.points[3] = -2;

      const moves = generateLegalTurnMoves(board, [2, 5]);
      expect(moves).toHaveLength(1);
      expect(moves[0].steps).toHaveLength(1);
      expect(moves[0].steps[0].dieUsed).toBe(5);
      expect(moves[0].steps[0].to).toBe(5);
    });

    it('allows playing the lower die if only 1 die can be played and the higher die is blocked', () => {
      const board = createEmptyBoard('white');
      board.points[10] = 1;
      // Roll [2, 5]:
      // 10 -> 5 (die 5) is BLOCKED
      // 10 -> 8 (die 2) is open, and 8 -> 3 (die 5) is BLOCKED
      board.points[5] = -2;
      board.points[3] = -2;

      const moves = generateLegalTurnMoves(board, [2, 5]);
      expect(moves).toHaveLength(1);
      expect(moves[0].steps).toHaveLength(1);
      expect(moves[0].steps[0].dieUsed).toBe(2);
      expect(moves[0].steps[0].to).toBe(8);
    });
  });

  describe('Step-by-step interactive move filtering & Evaluator', () => {
    it('filters valid next steps so player cannot violate forced-both-dice rule', () => {
      const board = createEmptyBoard('white');
      board.points[10] = 1; // Checker A
      board.points[5] = 1; // Checker B
      board.points[4] = -2; // Blocks 6 -> 4
      board.points[3] = -2; // Blocks 5 -> 3

      // Roll [4, 2]: playing 10 -> 6 (die 4) is illegal because then die 2 has no move.
      const firstSteps = getValidNextSteps(board, [4, 2], []);
      const illegalStep = firstSteps.find(
        (s) => s.from === 10 && s.to === 6 && s.dieUsed === 4
      );
      expect(illegalStep).toBeUndefined();
      expect(firstSteps.length).toBeGreaterThan(0);
    });

    it('formats standard Backgammon move notation accurately', () => {
      const notation = formatTurnNotation('white', [6, 4], [
        { from: 23, to: 17, dieUsed: 6, isHit: false },
        { from: 12, to: 8, dieUsed: 4, isHit: true },
      ]);
      expect(notation).toBe('6-4: 24/18 13/9*');
    });

    it('evaluates the initial board as 0.00 equity and 50% win probability', () => {
      const board = createInitialBoard();
      const evalRes = evaluateBaseline(board);
      expect(evalRes.equity).toBe(0);
      expect(evalRes.winProb).toBe(50);
      expect(evalRes.details.pipDiff).toBe(0);
    });

    it('computes single-die and combined multi-step destinations (3+4, doubles, bar entry + continuation, intermediate hit)', () => {
      const board = createInitialBoard();
      board.dice = [3, 4];
      board.remainingMoves = [3, 4];

      // From White's 13-point (index 12) with 3-4: can move 3->9, 4->8, or combined 3+4->5
      const opts = getValidDestinationsForSource(board, board, [3, 4], [], 12);
      const single9 = opts.find((o) => o.to === 9);
      const single8 = opts.find((o) => o.to === 8);
      const combined5 = opts.find((o) => o.to === 5);

      expect(single9?.isCombined).toBe(false);
      expect(single8?.isCombined).toBe(false);
      expect(combined5?.isCombined).toBe(true);
      expect(combined5?.steps).toHaveLength(2);
      expect(['3+4', '4+3']).toContain(combined5?.diceBadge);

      // Test intermediate blot hit preference when moving 3+4 from index 12 -> 5
      const hitIntermediateBoard = createEmptyBoard('white');
      hitIntermediateBoard.points[12] = 2;
      hitIntermediateBoard.points[9] = -1; // Black blot on intermediate point 9 (12 - 3 = 9)!
      hitIntermediateBoard.points[18] = -2;
      hitIntermediateBoard.dice = [3, 4];
      hitIntermediateBoard.remainingMoves = [3, 4];

      const hitOpts = getValidDestinationsForSource(
        hitIntermediateBoard,
        hitIntermediateBoard,
        [3, 4],
        [],
        12
      );
      const combinedHit5 = hitOpts.find((o) => o.to === 5);
      expect(combinedHit5).toBeDefined();
      expect(combinedHit5?.includesHit).toBe(true);
      expect(combinedHit5?.steps[0].to).toBe(9);
      expect(combinedHit5?.steps[0].isHit).toBe(true);
    });
  });
});
