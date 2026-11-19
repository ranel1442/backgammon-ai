export type Player = 'white' | 'black';

export interface BoardState {
  /**
   * 24 points (indices 0-23).
   * Positive numbers = white checkers, negative numbers = black checkers, 0 = empty.
   */
  points: number[];
  bar: {
    white: number;
    black: number;
  };
  borneOff: {
    white: number;
    black: number;
  };
  currentTurn: Player;
  dice: number[];
  remainingMoves: number[];
}

export interface MoveStep {
  from: number | 'bar';
  to: number | 'off';
  dieUsed: number;
  isHit: boolean;
}

export interface TurnMove {
  steps: MoveStep[];
  resultingState: BoardState;
}
