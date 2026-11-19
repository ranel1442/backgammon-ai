import type { BoardState, MoveStep, Player, TurnMove } from './types';

/**
 * Creates the standard Backgammon starting board state.
 * White moves from index 23 down to 0 and bears off.
 * Black moves from index 0 up to 23 and bears off.
 */
export function createInitialBoard(): BoardState {
  const points = new Array<number>(24).fill(0);

  // White checkers (positive) - moving 23 -> 0
  points[23] = 2;
  points[12] = 5;
  points[7] = 3;
  points[5] = 5;

  // Black checkers (negative) - moving 0 -> 23
  points[0] = -2;
  points[11] = -5;
  points[16] = -3;
  points[18] = -5;

  return {
    points,
    bar: { white: 0, black: 0 },
    borneOff: { white: 0, black: 0 },
    currentTurn: 'white',
    dice: [],
    remainingMoves: [],
  };
}

/**
 * Calculates the total pip count (total distance to bear off all remaining checkers)
 * for the specified player.
 */
export function getPipCount(state: BoardState, player: Player): number {
  let pips = 0;

  if (player === 'white') {
    for (let i = 0; i < 24; i++) {
      if (state.points[i] > 0) {
        pips += state.points[i] * (i + 1);
      }
    }
    pips += state.bar.white * 25;
  } else {
    for (let i = 0; i < 24; i++) {
      if (state.points[i] < 0) {
        pips += Math.abs(state.points[i]) * (24 - i);
      }
    }
    pips += state.bar.black * 25;
  }

  return pips;
}

/**
 * Checks whether all of the player's remaining checkers are in their home board
 * (and none are on the bar), allowing them to bear off.
 */
export function canBearOff(state: BoardState, player: Player): boolean {
  if (player === 'white') {
    if (state.bar.white > 0) return false;
    for (let i = 6; i < 24; i++) {
      if (state.points[i] > 0) return false;
    }
    return true;
  } else {
    if (state.bar.black > 0) return false;
    for (let i = 0; i < 18; i++) {
      if (state.points[i] < 0) return false;
    }
    return true;
  }
}

/**
 * Generates all legal single-die MoveSteps from the current state for a specific die value.
 */
export function getLegalStepsForDie(state: BoardState, die: number): MoveStep[] {
  const player = state.currentTurn;
  const steps: MoveStep[] = [];

  if (player === 'white') {
    // Rule b: Checkers on the bar MUST enter before any other checker can move.
    if (state.bar.white > 0) {
      const target = 24 - die;
      if (target >= 0 && target < 24 && state.points[target] >= -1) {
        steps.push({
          from: 'bar',
          to: target,
          dieUsed: die,
          isHit: state.points[target] === -1,
        });
      }
      return steps;
    }

    const mayBearOff = canBearOff(state, 'white');

    for (let from = 23; from >= 0; from--) {
      if (state.points[from] <= 0) continue;

      const target = from - die;
      if (target >= 0) {
        // Normal move on the board (blocked if opponent has 2 or more checkers: <= -2)
        if (state.points[target] >= -1) {
          steps.push({
            from,
            to: target,
            dieUsed: die,
            isHit: state.points[target] === -1,
          });
        }
      } else if (mayBearOff) {
        if (target === -1) {
          // Exact bear-off
          steps.push({
            from,
            to: 'off',
            dieUsed: die,
            isHit: false,
          });
        } else {
          // Overshoot bear-off: allowed only if no white checkers exist on higher points (from + 1 .. 5)
          let hasHigherChecker = false;
          for (let higher = from + 1; higher <= 5; higher++) {
            if (state.points[higher] > 0) {
              hasHigherChecker = true;
              break;
            }
          }
          if (!hasHigherChecker) {
            steps.push({
              from,
              to: 'off',
              dieUsed: die,
              isHit: false,
            });
          }
        }
      }
    }
  } else {
    // Black's turn (moving 0 -> 23)
    if (state.bar.black > 0) {
      const target = die - 1;
      if (target >= 0 && target < 24 && state.points[target] <= 1) {
        steps.push({
          from: 'bar',
          to: target,
          dieUsed: die,
          isHit: state.points[target] === 1,
        });
      }
      return steps;
    }

    const mayBearOff = canBearOff(state, 'black');

    for (let from = 0; from < 24; from++) {
      if (state.points[from] >= 0) continue;

      const target = from + die;
      if (target < 24) {
        // Normal move on the board (blocked if opponent has 2 or more checkers: >= 2)
        if (state.points[target] <= 1) {
          steps.push({
            from,
            to: target,
            dieUsed: die,
            isHit: state.points[target] === 1,
          });
        }
      } else if (mayBearOff) {
        if (target === 24) {
          // Exact bear-off
          steps.push({
            from,
            to: 'off',
            dieUsed: die,
            isHit: false,
          });
        } else {
          // Overshoot bear-off: allowed only if no black checkers exist on higher points (18 .. from - 1)
          let hasHigherChecker = false;
          for (let higher = 18; higher < from; higher++) {
            if (state.points[higher] < 0) {
              hasHigherChecker = true;
              break;
            }
          }
          if (!hasHigherChecker) {
            steps.push({
              from,
              to: 'off',
              dieUsed: die,
              isHit: false,
            });
          }
        }
      }
    }
  }

  return steps;
}

/**
 * Pure function that applies a single MoveStep to a BoardState and returns the new BoardState.
 */
export function applyMoveStep(state: BoardState, step: MoveStep): BoardState {
  const points = [...state.points];
  const bar = { ...state.bar };
  const borneOff = { ...state.borneOff };
  const remainingMoves = [...state.remainingMoves];
  const player = state.currentTurn;

  // Remove checker from origin
  if (step.from === 'bar') {
    bar[player] -= 1;
  } else {
    if (player === 'white') {
      points[step.from] -= 1;
    } else {
      points[step.from] += 1;
    }
  }

  // Place checker at destination
  if (step.to === 'off') {
    borneOff[player] += 1;
  } else {
    if (player === 'white') {
      if (step.isHit) {
        points[step.to] = 1;
        bar.black += 1;
      } else {
        points[step.to] += 1;
      }
    } else {
      if (step.isHit) {
        points[step.to] = -1;
        bar.white += 1;
      } else {
        points[step.to] -= 1;
      }
    }
  }

  // Consume one matching die from remainingMoves
  const dieIdx = remainingMoves.indexOf(step.dieUsed);
  if (dieIdx !== -1) {
    remainingMoves.splice(dieIdx, 1);
  }

  return {
    points,
    bar,
    borneOff,
    currentTurn: player,
    dice: [...state.dice],
    remainingMoves,
  };
}

/**
 * Generates a canonical string key for a BoardState's physical configuration
 * to deduplicate identical resulting states.
 */
function getBoardStateKey(state: BoardState): string {
  return `${state.currentTurn}|${state.bar.white},${state.bar.black}|${state.borneOff.white},${state.borneOff.black}|${state.points.join(',')}`;
}

/**
 * Generates all legal complete TurnMoves for the current player given a dice roll.
 * Enforces all official Backgammon rules:
 * a) Doubles give 4 moves of that die value.
 * b) Checkers on the bar must enter first.
 * c) Blocked points (>=2 opponent checkers) vs hits (1 opponent checker).
 * d) Bearing off (exact and overshoot only when no higher point is occupied).
 * e) Maximize number of dice used.
 * f) Forced higher die when only 1 of 2 unequal dice can be played.
 * g) Deduplicate identical resulting board states.
 */
export function generateLegalTurnMoves(
  state: BoardState,
  dice: [number, number]
): TurnMove[] {
  const isDoubles = dice[0] === dice[1];
  const expandedDice = isDoubles
    ? [dice[0], dice[0], dice[0], dice[0]]
    : [dice[0], dice[1]];

  const baseState: BoardState = {
    points: [...state.points],
    bar: { ...state.bar },
    borneOff: { ...state.borneOff },
    currentTurn: state.currentTurn,
    dice: [...expandedDice],
    remainingMoves: [...expandedDice],
  };

  // For non-doubles, explore both dice orders: [d0, d1] and [d1, d0].
  // For doubles, there is only one sequence of 4 identical dice.
  const diceOrders: number[][] = isDoubles
    ? [expandedDice]
    : [
        [dice[0], dice[1]],
        [dice[1], dice[0]],
      ];

  const rawTurnMoves: TurnMove[] = [];
  const visitedIntermediate = new Set<string>();

  function dfs(
    currentState: BoardState,
    diceQueue: number[],
    stepsSoFar: MoveStep[]
  ): void {
    if (diceQueue.length === 0) {
      rawTurnMoves.push({
        steps: stepsSoFar,
        resultingState: currentState,
      });
      return;
    }

    const visitKey = `${getBoardStateKey(currentState)}#${diceQueue.join(',')}`;
    if (visitedIntermediate.has(visitKey)) {
      return;
    }
    visitedIntermediate.add(visitKey);

    const currentDie = diceQueue[0];
    const nextDiceQueue = diceQueue.slice(1);
    const legalSteps = getLegalStepsForDie(currentState, currentDie);

    if (legalSteps.length === 0) {
      // Cannot play currentDie from this state.
      // Record partial sequence if at least 1 step was already made.
      if (stepsSoFar.length > 0) {
        rawTurnMoves.push({
          steps: stepsSoFar,
          resultingState: currentState,
        });
      }
      return;
    }

    for (const step of legalSteps) {
      const nextState = applyMoveStep(currentState, step);
      dfs(nextState, nextDiceQueue, [...stepsSoFar, step]);
    }
  }

  for (const order of diceOrders) {
    dfs(baseState, order, []);
  }

  if (rawTurnMoves.length === 0) {
    return [];
  }

  // Rule e: Must maximize the number of dice used
  let maxSteps = 0;
  for (const move of rawTurnMoves) {
    if (move.steps.length > maxSteps) {
      maxSteps = move.steps.length;
    }
  }

  let filteredMoves = rawTurnMoves.filter(
    (move) => move.steps.length === maxSteps
  );

  // Rule f: If only 1 die can be played (and not both on a non-double roll),
  // the player MUST play the higher die if a legal move exists for it.
  if (!isDoubles && maxSteps === 1) {
    const higherDie = Math.max(dice[0], dice[1]);
    const movesWithHigherDie = filteredMoves.filter(
      (move) => move.steps[0].dieUsed === higherDie
    );
    if (movesWithHigherDie.length > 0) {
      filteredMoves = movesWithHigherDie;
    }
  }

  // Rule g: Deduplicate identical resulting board states
  const uniqueMoves: TurnMove[] = [];
  const seenStates = new Set<string>();

  for (const move of filteredMoves) {
    const key = getBoardStateKey(move.resultingState);
    if (!seenStates.has(key)) {
      seenStates.add(key);
      uniqueMoves.push(move);
    }
  }

  return uniqueMoves;
}

/**
 * Derives the valid next MoveSteps strictly from `generateLegalTurnMoves(turnStartState, dice)`
 * filtered by the steps already taken in the current turn.
 * Ensures the player can never make a step that violates forced-both-dice or forced-higher-die rules,
 * while permitting any valid step ordering that reaches a legal resulting turn state.
 */
export function getValidNextSteps(
  turnStartState: BoardState,
  dice: [number, number],
  stepsTaken: MoveStep[]
): MoveStep[] {
  const legalTurnMoves = generateLegalTurnMoves(turnStartState, dice);
  if (legalTurnMoves.length === 0) {
    return [];
  }

  const maxSteps = legalTurnMoves[0].steps.length;
  if (stepsTaken.length >= maxSteps) {
    return [];
  }

  const isDoubles = dice[0] === dice[1];
  const expandedDice = isDoubles
    ? [dice[0], dice[0], dice[0], dice[0]]
    : [dice[0], dice[1]];

  let currentState: BoardState = {
    points: [...turnStartState.points],
    bar: { ...turnStartState.bar },
    borneOff: { ...turnStartState.borneOff },
    currentTurn: turnStartState.currentTurn,
    dice: [...expandedDice],
    remainingMoves: [...expandedDice],
  };

  for (const step of stepsTaken) {
    currentState = applyMoveStep(currentState, step);
  }

  const allowedFinalKeys = new Set(
    legalTurnMoves.map((m) => getBoardStateKey(m.resultingState))
  );

  function canReachAllowedFinal(
    state: BoardState,
    stepsRemaining: number
  ): boolean {
    if (stepsRemaining === 0) {
      return allowedFinalKeys.has(getBoardStateKey(state));
    }

    const distinctDice = Array.from(new Set(state.remainingMoves));
    for (const die of distinctDice) {
      const nextSteps = getLegalStepsForDie(state, die);
      for (const s of nextSteps) {
        const nextState = applyMoveStep(state, s);
        if (canReachAllowedFinal(nextState, stepsRemaining - 1)) {
          return true;
        }
      }
    }
    return false;
  }

  const validSteps: MoveStep[] = [];
  const seenSteps = new Set<string>();
  const distinctRemainingDice = Array.from(
    new Set(currentState.remainingMoves)
  );
  const stepsRemainingAfterThis = maxSteps - stepsTaken.length - 1;

  for (const die of distinctRemainingDice) {
    const candidates = getLegalStepsForDie(currentState, die);
    for (const candidate of candidates) {
      const stepKey = `${candidate.from}->${candidate.to}:${candidate.dieUsed}`;
      if (seenSteps.has(stepKey)) continue;

      const nextState = applyMoveStep(currentState, candidate);
      if (canReachAllowedFinal(nextState, stepsRemainingAfterThis)) {
        seenSteps.add(stepKey);
        validSteps.push(candidate);
      }
    }
  }

  return validSteps;
}

export interface ValidDestinationOption {
  to: number | 'off';
  /** Ordered sequence of 1..4 MoveSteps moving the SAME checker from `from` to `to` */
  steps: MoveStep[];
  /** True when this destination chains 2, 3, or 4 dice steps together */
  isCombined: boolean;
  /** Compact dice label (e.g. '3', '4', '3+4', '6+6') */
  diceBadge: string;
  /** True if any step in the chain hits an opponent blot (including intermediate points) */
  includesHit: boolean;
}

/**
 * Computes all legal single-die AND combined multi-step destinations for a checker selected at `from`.
 * Strictly derived from `generateLegalTurnMoves(turnStartState, dice)`:
 * - Respects bar entry priority (including entering 1 checker from the bar with die 1 and continuing with die 2).
 * - Requires a legal unblocked intermediate landing point.
 * - Prefers intermediate hitting paths (`isHit: true`) when both dice orders reach the same destination.
 */
export function getValidDestinationsForSource(
  turnStartState: BoardState,
  currentState: BoardState,
  dice: [number, number],
  stepsTaken: MoveStep[],
  from: number | 'bar'
): ValidDestinationOption[] {
  const legalTurnMoves = generateLegalTurnMoves(turnStartState, dice);
  if (legalTurnMoves.length === 0) {
    return [];
  }

  const maxSteps = legalTurnMoves[0].steps.length;
  const stepsRemainingTotal = maxSteps - stepsTaken.length;
  if (stepsRemainingTotal <= 0) {
    return [];
  }

  const allowedFinalKeys = new Set(
    legalTurnMoves.map((m) => getBoardStateKey(m.resultingState))
  );

  function canReachAllowedFinal(
    state: BoardState,
    stepsRemaining: number
  ): boolean {
    if (stepsRemaining === 0) {
      return allowedFinalKeys.has(getBoardStateKey(state));
    }

    const distinctDice = Array.from(new Set(state.remainingMoves));
    for (const die of distinctDice) {
      const nextSteps = getLegalStepsForDie(state, die);
      for (const s of nextSteps) {
        const nextState = applyMoveStep(state, s);
        if (canReachAllowedFinal(nextState, stepsRemaining - 1)) {
          return true;
        }
      }
    }
    return false;
  }

  const optionsByTarget = new Map<number | 'off', ValidDestinationOption>();

  function exploreSameCheckerChains(
    stateCursor: BoardState,
    currentPos: number | 'bar',
    chainSoFar: MoveStep[]
  ): void {
    const k = chainSoFar.length;
    if (k >= stepsRemainingTotal) {
      return;
    }

    const distinctDice = Array.from(new Set(stateCursor.remainingMoves));
    for (const die of distinctDice) {
      const candidates = getLegalStepsForDie(stateCursor, die).filter(
        (s) => s.from === currentPos
      );

      for (const step of candidates) {
        const nextState = applyMoveStep(stateCursor, step);
        const nextChain = [...chainSoFar, step];
        const remainingAfterChain = stepsRemainingTotal - nextChain.length;

        if (canReachAllowedFinal(nextState, remainingAfterChain)) {
          const target = step.to;
          const includesHit = nextChain.some((s) => s.isHit);
          const candidateOption: ValidDestinationOption = {
            to: target,
            steps: nextChain,
            isCombined: nextChain.length > 1,
            diceBadge: nextChain.map((s) => s.dieUsed).join('+'),
            includesHit,
          };

          const existing = optionsByTarget.get(target);
          if (
            !existing ||
            (existing.steps.length > candidateOption.steps.length) ||
            (existing.steps.length === candidateOption.steps.length &&
              candidateOption.includesHit &&
              !existing.includesHit)
          ) {
            optionsByTarget.set(target, candidateOption);
          }
        }

        if (
          typeof step.to === 'number' &&
          nextChain.length < stepsRemainingTotal
        ) {
          exploreSameCheckerChains(nextState, step.to, nextChain);
        }
      }
    }
  }

  exploreSameCheckerChains(currentState, from, []);
  return Array.from(optionsByTarget.values());
}

/**
 * Formats a turn's dice and steps into standard Backgammon notation
 * (e.g., `6-4: 24/18 13/9`, using `*` for hits like `13/7*` and `off` for bearing off).
 */
export function formatTurnNotation(
  player: Player,
  dice: [number, number],
  steps: MoveStep[]
): string {
  const diceStr = `${dice[0]}-${dice[1]}`;
  if (steps.length === 0) {
    return `${diceStr}: No Move`;
  }

  const formatPoint = (pt: number | 'bar' | 'off'): string => {
    if (pt === 'bar') return 'bar';
    if (pt === 'off') return 'off';
    return player === 'white' ? String(pt + 1) : String(24 - pt);
  };

  const formattedSteps = steps.map((step) => {
    const fromStr = formatPoint(step.from);
    const toStr = formatPoint(step.to);
    return `${fromStr}/${toStr}${step.isHit ? '*' : ''}`;
  });

  return `${diceStr}: ${formattedSteps.join(' ')}`;
}

