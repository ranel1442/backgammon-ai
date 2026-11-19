import { evaluatePosition, type EvalResult } from './evaluator';
import { classifyGamePhase, type GamePhase } from './neuralNet';
import {
  isOpeningOrReplyBookState,
  lookupOpeningBookMove,
} from './openingBook';
import { applyMoveStep, generateLegalTurnMoves, getPipCount } from './rules';
import type { BoardState, MoveStep, Player, TurnMove } from './types';

export type SearchDepth = 1 | 2 | 3 | 4 | 5;

export type MoveClassification =
  | 'Best'
  | 'Good'
  | 'Inaccuracy'
  | 'Mistake'
  | 'Blunder';

export interface ClassificationMeta {
  label: string;
  bgClass: string;
  textClass: string;
  borderClass: string;
  dotColor: string;
}

export const CLASSIFICATION_META_MAP: Record<
  MoveClassification,
  ClassificationMeta
> = {
  Best: {
    label: 'המהלך הטוב ביותר',
    bgClass: 'bg-emerald-500/15',
    textClass: 'text-emerald-300',
    borderClass: 'border-emerald-500/40',
    dotColor: '#34d399',
  },
  Good: {
    label: 'מהלך טוב',
    bgClass: 'bg-teal-500/15',
    textClass: 'text-teal-300',
    borderClass: 'border-teal-500/40',
    dotColor: '#2dd4bf',
  },
  Inaccuracy: {
    label: 'אי-דיוק',
    bgClass: 'bg-sky-500/15',
    textClass: 'text-sky-300',
    borderClass: 'border-sky-500/40',
    dotColor: '#facc15',
  },
  Mistake: {
    label: 'טעות',
    bgClass: 'bg-amber-500/15',
    textClass: 'text-amber-300',
    borderClass: 'border-amber-500/40',
    dotColor: '#f97316',
  },
  Blunder: {
    label: 'טעות גסה',
    bgClass: 'bg-rose-500/15',
    textClass: 'text-rose-300',
    borderClass: 'border-rose-500/40',
    dotColor: '#f43f5e',
  },
};

export function getClassificationMeta(
  classification: MoveClassification
): ClassificationMeta {
  return CLASSIFICATION_META_MAP[classification];
}

export interface ExactDiceCombinatorics {
  /** Number of opponent's 36 next rolls (0..36) that can legally hit at least one of player's blots */
  opponentHitRolls: number;
  /** Percentage of opponent's rolls (0..100) that can hit at least one blot */
  opponentHitPct: number;
  /** When opponent has a checker on the bar, how many of their 36 rolls are completely blocked ("dancing") */
  opponentBarBlockRolls: number;
  /** Percentage of opponent's 36 rolls that dance on the bar */
  opponentBarBlockPct: number;
  /** How many of player's 36 rolls allow a back checker in opponent's home board to advance/escape */
  escapeRolls: number;
  /** Percentage of player's 36 rolls that advance/escape a back checker */
  escapePct: number;
}

export interface CandidateMoveAnalysis {
  move: TurnMove;
  notation: string;
  /** Equity from White's perspective (+ favors White, - favors Black) */
  equity: number;
  /** Active player's win probability (0..100) or White winProb */
  winProb: number;
  /** Gammon probability percentage (0..100) */
  gammonProb: number;
  /** Equity lost relative to the #1 best move (>= 0) */
  equityLoss: number;
  /** Exact 36-roll dice combinatorics for the resulting position */
  combinatorics: ExactDiceCombinatorics;
  /** True when the move matches an entry in the Opening & Reply Book */
  isBookMove?: boolean;
  /** Hebrew badge label for book moves ('ספר פתיחות') */
  bookBadgeLabel?: string;
  /** Head-to-head comparison vs #2 move (populated on #1 move) */
  runnerUpComparison?: string;
  /** 2-4 tactical insights in Hebrew explaining why this move works or falls short */
  explanation: string[];
}

export interface TurnAnalysisResult {
  topMoves: CandidateMoveAnalysis[];
  gamePhase?: GamePhase;
  gamePhaseLabel?: string;
  strategicPlan?: string;
  runnerUpComparison?: string;
  requestedDepth?: SearchDepth;
  completedDepth?: SearchDepth;
  elapsedMs?: number;
  playedMoveAnalysis?: {
    notation: string;
    equity: number;
    equityLoss: number;
    classification: MoveClassification;
    explanation: string[];
  };
}

export interface DiceRollOutcome {
  dice: [number, number];
  probability: number;
  countOutOf36: 1 | 2;
}

/**
 * All 21 distinct 2-dice outcomes in Backgammon with exact probabilities
 * (1/36 for the 6 doubles, 2/36 for the 15 non-doubles; sum = 36/36 = 1.0).
 */
export const ALL_21_DICE_OUTCOMES: readonly DiceRollOutcome[] = (() => {
  const outcomes: DiceRollOutcome[] = [];
  for (let d1 = 1; d1 <= 6; d1++) {
    for (let d2 = d1; d2 <= 6; d2++) {
      const isDouble = d1 === d2;
      outcomes.push({
        dice: [d1, d2],
        probability: isDouble ? 1 / 36 : 2 / 36,
        countOutOf36: isDouble ? 1 : 2,
      });
    }
  }
  return outcomes;
})();

/**
 * Representative strategic dice rolls used during the Selective 3-Ply Extension
 * to evaluate the moving player's follow-up potential without blocking the UI.
 */
const SELECTIVE_3PLY_ROLLS: readonly DiceRollOutcome[] = [
  { dice: [3, 1], probability: 2 / 14, countOutOf36: 2 },
  { dice: [4, 2], probability: 2 / 14, countOutOf36: 2 },
  { dice: [5, 3], probability: 2 / 14, countOutOf36: 2 },
  { dice: [6, 1], probability: 2 / 14, countOutOf36: 2 },
  { dice: [6, 4], probability: 2 / 14, countOutOf36: 2 },
  { dice: [6, 5], probability: 2 / 14, countOutOf36: 2 },
  { dice: [5, 2], probability: 2 / 14, countOutOf36: 2 },
];

/**
 * Shared helper to format a sequence of MoveSteps into standard Backgammon notation
 * (e.g., `24/18 13/9`, using `*` for hits like `13/7*` and `off` for bearing off).
 */
export function formatTurnNotation(steps: MoveStep[], player: Player): string {
  if (steps.length === 0) {
    return 'ללא מהלך';
  }

  const formatPt = (pt: number | 'bar' | 'off'): string => {
    if (pt === 'bar') return 'bar';
    if (pt === 'off') return 'off';
    return player === 'white' ? String(pt + 1) : String(24 - pt);
  };

  return steps
    .map((s) => `${formatPt(s.from)}/${formatPt(s.to)}${s.isHit ? '*' : ''}`)
    .join(' ');
}

/**
 * Classifies a move based on the equity lost compared to the engine's best move.
 */
export function classifyEquityLoss(equityLoss: number): MoveClassification {
  if (equityLoss <= 0.02) return 'Best';
  if (equityLoss <= 0.05) return 'Good';
  if (equityLoss <= 0.1) return 'Inaccuracy';
  if (equityLoss <= 0.2) return 'Mistake';
  return 'Blunder';
}

/**
 * Computes exact 36-roll dice combinatorics for `player` on `state`:
 * - `opponentHitRolls` (0..36): how many of opponent's 36 next rolls can legally hit at least one of `player`'s blots.
 * - `opponentBarBlockRolls` (0..36): when opponent is on the bar, how many of their 36 rolls are completely blocked ("dancing").
 * - `escapeRolls` (0..36): when `player` has back checkers in opponent's home board, how many of `player`'s 36 rolls advance/escape them.
 */
export function computeExactDiceCombinatorics(
  state: BoardState,
  player: Player
): ExactDiceCombinatorics {
  const opponent: Player = player === 'white' ? 'black' : 'white';
  const isWhite = player === 'white';

  // Check if player has any blots (single checkers) on the board
  const hasPlayerBlots = state.points.some((v) =>
    isWhite ? v === 1 : v === -1
  );

  let opponentHitRolls = 0;
  let opponentBarBlockRolls = 0;

  const oppState: BoardState = {
    points: [...state.points],
    bar: { ...state.bar },
    borneOff: { ...state.borneOff },
    currentTurn: opponent,
    dice: [],
    remainingMoves: [],
  };

  if (hasPlayerBlots || state.bar[opponent] > 0) {
    for (const outcome of ALL_21_DICE_OUTCOMES) {
      const oppMoves = generateLegalTurnMoves(oppState, outcome.dice);

      if (state.bar[opponent] > 0 && oppMoves.length === 0) {
        opponentBarBlockRolls += outcome.countOutOf36;
      }

      if (hasPlayerBlots && oppMoves.length > 0) {
        const canHit = oppMoves.some(
          (m) =>
            m.resultingState.bar[player] > state.bar[player] ||
            m.steps.some((s) => s.isHit)
        );
        if (canHit) {
          opponentHitRolls += outcome.countOutOf36;
        }
      }
    }
  }

  // Compute escapeRolls for `player`'s back checkers in opponent's home board (or bar)
  const backIndices = isWhite
    ? [18, 19, 20, 21, 22, 23]
    : [0, 1, 2, 3, 4, 5];
  const hasBackCheckers =
    state.bar[player] > 0 ||
    backIndices.some((idx) =>
      isWhite ? state.points[idx] > 0 : state.points[idx] < 0
    );

  let escapeRolls = 36;
  if (hasBackCheckers) {
    escapeRolls = 0;
    const playerTurnState: BoardState = {
      points: [...state.points],
      bar: { ...state.bar },
      borneOff: { ...state.borneOff },
      currentTurn: player,
      dice: [],
      remainingMoves: [],
    };

    const backSet = new Set<number>(backIndices);

    for (const outcome of ALL_21_DICE_OUTCOMES) {
      const pMoves = generateLegalTurnMoves(playerTurnState, outcome.dice);
      if (pMoves.length === 0) continue;

      const canAdvanceBack = pMoves.some((m) =>
        m.steps.some(
          (s) =>
            s.from === 'bar' ||
            (typeof s.from === 'number' && backSet.has(s.from))
        )
      );
      if (canAdvanceBack) {
        escapeRolls += outcome.countOutOf36;
      }
    }
  }

  return {
    opponentHitRolls,
    opponentHitPct: Number(((opponentHitRolls / 36) * 100).toFixed(1)),
    opponentBarBlockRolls,
    opponentBarBlockPct: Number(
      ((opponentBarBlockRolls / 36) * 100).toFixed(1)
    ),
    escapeRolls,
    escapePct: Number(((escapeRolls / 36) * 100).toFixed(1)),
  };
}

/**
 * Generates the Hebrew strategic phase label and Grandmaster plan for a board state.
 */
export function getPhaseAndStrategicPlan(
  state: BoardState,
  player: Player
): {
  gamePhase: GamePhase;
  gamePhaseLabel: string;
  strategicPlan: string;
} {
  const gamePhase = classifyGamePhase(state);
  const opponent: Player = player === 'white' ? 'black' : 'white';

  if (gamePhase === 'race') {
    const playerPips = getPipCount(state, player);
    const oppPips = getPipCount(state, opponent);
    const lead = oppPips - playerPips;
    const raceLeadStr =
      lead > 0
        ? `יתרון מרוץ של ${lead} פיפס`
        : lead < 0
          ? `פיגור מרוץ של ${Math.abs(lead)} פיפס`
          : 'מרוץ שקול לחלוטין';
    return {
      gamePhase,
      gamePhaseLabel: `שלב: מרוץ טהור (Race • ${raceLeadStr})`,
      strategicPlan:
        'המטרה: מזעור בזבוז פיפס (EPC), הימנעות מערימות על נקודות 1–2 והוצאת כלים יעילה לפי טבלת הסיומים המדויקת.',
    };
  }

  if (gamePhase === 'crashed') {
    return {
      gamePhase,
      gamePhaseLabel: 'שלב: מצב מרוסק (Crashed Contact)',
      strategicPlan:
        'המטרה: תזמון סבלני (Timing), שמירה על מבנה הבית מפני קריסה וניצול כל הזדמנות לחילוץ או הכאת כלי חשוף.',
    };
  }

  // Contact phase: tailor strategic plan to position structure
  if (state.bar[opponent] > 0) {
    return {
      gamePhase,
      gamePhaseLabel: 'שלב: קרב מגע — לחץ התקפי (Blitz / כלי יריב על הבר)',
      strategicPlan:
        'המטרה: סגירת נקודות נוספות בבית כדי לנעול את היריב על הבר ולהגדיל סיכויי מארס (Gammon).',
    };
  }

  return {
    gamePhase,
    gamePhaseLabel: 'שלב: קרב מגע וחסימות (Contact & Priming)',
    strategicPlan:
      'המטרה: השלמת חומה רצופה (Prime), בניית נקודות הזהב בבית (נקודות 5 ו-4) ושמירה על עוגן הגנתי.',
  };
}

/**
 * Builds a concise Hebrew head-to-head comparison explaining why #1 beats #2.
 */
export function buildRunnerUpComparison(
  best: {
    notation: string;
    equity: number;
    winProb: number;
    gammonProb: number;
    combinatorics: ExactDiceCombinatorics;
  },
  runnerUp: {
    notation: string;
    equity: number;
    winProb: number;
    gammonProb: number;
    combinatorics: ExactDiceCombinatorics;
  },
  player: Player
): string {
  const eqGain =
    player === 'white'
      ? Math.max(0, best.equity - runnerUp.equity)
      : Math.max(0, runnerUp.equity - best.equity);

  const bestPlayerWin = player === 'white' ? best.winProb : 100 - best.winProb;
  const runnerPlayerWin =
    player === 'white' ? runnerUp.winProb : 100 - runnerUp.winProb;
  const winDelta = bestPlayerWin - runnerPlayerWin;
  const gammonDelta = best.gammonProb - runnerUp.gammonProb;
  const hitReduction =
    runnerUp.combinatorics.opponentHitRolls -
    best.combinatorics.opponentHitRolls;
  const barBlockGain =
    best.combinatorics.opponentBarBlockRolls -
    runnerUp.combinatorics.opponentBarBlockRolls;

  const clauses: string[] = [];

  if (Math.abs(winDelta) >= 0.3) {
    clauses.push(
      winDelta > 0
        ? `מעלה את סיכוי הניצחון ב-${winDelta.toFixed(1)}%`
        : `שומר על מבנה עמדתי עדיף לטווח ארוך`
    );
  }

  if (gammonDelta >= 0.8) {
    clauses.push(`מגדיל סיכוי מארס ב-${gammonDelta.toFixed(1)}%`);
  }

  if (hitReduction > 0) {
    const pctDiff = ((hitReduction / 36) * 100).toFixed(1);
    clauses.push(
      `מקטין סיכון פגיעה ישירה ב-${hitReduction} מתוך 36 הטלות (${pctDiff}%-)`
    );
  } else if (barBlockGain > 0) {
    const pctGain = ((barBlockGain / 36) * 100).toFixed(1);
    clauses.push(
      `מוסיף ${barBlockGain}/36 הטלות שבהן היריב נשאר חסום על הבר (+${pctGain}%)`
    );
  }

  if (clauses.length === 0) {
    clauses.push(
      'מעניק פיזור בנאים גמיש יותר ומבנה חסימה יציב יותר מול כל 21 הטלות היריב'
    );
  }

  return `עדיף ב-+${eqGain.toFixed(2)} Equity על פני ${runnerUp.notation}: ${clauses.join(', ')}.`;
}

function getPhysicalStateSignature(state: BoardState): string {
  return `${state.bar.white},${state.bar.black}|${state.borneOff.white},${state.borneOff.black}|${state.points.join(',')}`;
}

function evalWithCache(
  state: BoardState,
  cache?: Map<string, EvalResult>
): EvalResult {
  if (!cache) {
    return evaluatePosition(state);
  }
  const key = getPhysicalStateSignature(state);
  const existing = cache.get(key);
  if (existing) {
    return existing;
  }
  const computed = evaluatePosition(state);
  cache.set(key, computed);
  return computed;
}

/**
 * Compares `EvalFeatures` and exact 36-roll combinatorics between `before`, `candidate`, and `best`
 * board states to generate 2-4 concise, natural Hebrew Grandmaster tactical explanations.
 */
export function generateTacticalExplanation(
  before: BoardState,
  candidate: BoardState,
  best: BoardState,
  player: Player,
  isBest: boolean,
  dice?: [number, number],
  cache?: Map<string, EvalResult>,
  candCombinatorics?: ExactDiceCombinatorics,
  bestCombinatorics?: ExactDiceCombinatorics
): string[] {
  const sign = player === 'white' ? 1 : -1;
  const opponent: Player = player === 'white' ? 'black' : 'white';

  const insights: string[] = [];

  // 0. Check Opening & Reply Book if on Turn 1 or Turn 2
  if (dice && isOpeningOrReplyBookState(before)) {
    const bookEntry = lookupOpeningBookMove(before, dice, candidate);
    if (bookEntry) {
      insights.push(...bookEntry.explanation);
    } else if (!isBest) {
      const bestBook = lookupOpeningBookMove(before, dice, best);
      if (bestBook) {
        insights.push(
          `סוטה מהמהלך המומלץ בספר הפתיחות (${bestBook.canonicalNotation})`
        );
      }
    }
  }

  const beforeEval = evalWithCache(before, cache);
  const candEval = evalWithCache(candidate, cache);
  const bestEval = evalWithCache(best, cache);

  const bf = beforeEval.features;
  const cf = candEval.features;
  const bestF = bestEval.features;

  const comb =
    candCombinatorics ?? computeExactDiceCombinatorics(candidate, player);
  const bestComb =
    bestCombinatorics ??
    (isBest ? comb : computeExactDiceCombinatorics(best, player));

  // 1. Check if opponent checkers were hit to the bar + exact bar-dancing odds
  const hitsMade = candidate.bar[opponent] - before.bar[opponent];
  const bestHitsMade = best.bar[opponent] - before.bar[opponent];
  if (hitsMade > 0) {
    const barBlockNote =
      comb.opponentBarBlockRolls > 0
        ? ` — היריב חסום על הבר ב-${comb.opponentBarBlockRolls}/36 הטלות (${comb.opponentBarBlockPct}%)`
        : '';
    insights.push(
      hitsMade > 1
        ? `מכה ${hitsMade} כלים חשופים של היריב לבר${barBlockNote}`
        : `מכה כלי חשוף של היריב לבר (Bar)${barBlockNote}`
    );
  } else if (!isBest && bestHitsMade > 0) {
    insights.push('מפספס הזדמנות טקטית להכות כלי חשוף של היריב לבר');
  }

  // 2. Check bar escape
  if (before.bar[player] > candidate.bar[player]) {
    insights.push('מכניס כלי מהבר חזרה למשחק ומשפר את ניידות הכלים');
  }

  // 3. Check bearing off progress
  const borneOffGain = candidate.borneOff[player] - before.borneOff[player];
  const bestBorneOffGain = best.borneOff[player] - before.borneOff[player];
  if (borneOffGain > 0) {
    insights.push(
      borneOffGain > 1
        ? `מוציא ${borneOffGain} כלים מהלוח ומקדם את המרוץ לניצחון`
        : 'מוציא כלי מהלוח ביעילות לקראת סיום המשחק'
    );
  } else if (!isBest && bestBorneOffGain > borneOffGain) {
    insights.push('אינו מוציא כלים ביעילות מרבית בהשוואה לקו המומלץ של המנוע');
  }

  // 4. Home board points (made points in 0..5 for White, 18..23 for Black)
  const homeGain = (cf.homePointsDiff - bf.homePointsDiff) * sign;
  const bestHomeGain = (bestF.homePointsDiff - bf.homePointsDiff) * sign;
  if (homeGain > 0) {
    const golden5Idx = player === 'white' ? 4 : 19;
    const golden4Idx = player === 'white' ? 3 : 20;
    const madeGolden =
      (Math.abs(before.points[golden5Idx]) < 2 &&
        Math.abs(candidate.points[golden5Idx]) >= 2) ||
      (Math.abs(before.points[golden4Idx]) < 2 &&
        Math.abs(candidate.points[golden4Idx]) >= 2);

    insights.push(
      madeGolden
        ? 'בונה נקודה חזקה בבית (נקודת ה-5 / ה-4) ומחזק את השליטה בלוח'
        : 'סוגר נקודת מפתח בבית ומחזק את השליטה בלוח'
    );
  } else if (!isBest && bestHomeGain > homeGain) {
    insights.push('מוותר על סגירת נקודת מפתח חשובה בבית');
  }

  // 5. Prime building & trapping opponent back checkers
  const primeGain = (cf.primeLengthDiff - bf.primeLengthDiff) * sign;
  const playerPrimeNow =
    player === 'white'
      ? Math.max(0, cf.primeLengthDiff)
      : Math.max(0, -cf.primeLengthDiff);
  if (primeGain > 0 && playerPrimeNow >= 3) {
    insights.push(
      `בונה חומה (Prime) של ${Math.min(6, playerPrimeNow)} נקודות רצופות שמגבילה את תנועת היריב`
    );
  } else if (primeGain > 0) {
    insights.push('מחבר נקודות סמוכות ליצירת חסימה אפקטיבית מול היריב');
  }

  // 6. Exact 36-roll blot exposure / safety combinatorics
  const extraHitRollsVsBest = comb.opponentHitRolls - bestComb.opponentHitRolls;
  const dangerDelta = (cf.blotDangerDiff - bf.blotDangerDiff) * sign;

  if (!isBest && extraHitRollsVsBest >= 2) {
    insights.push(
      `חושף כלים ל-${comb.opponentHitRolls}/36 הטלות פגיעה של היריב (${comb.opponentHitPct}%) לעומת ${bestComb.opponentHitRolls}/36 בקו המוביל`
    );
  } else if (comb.opponentHitRolls === 0 && !cf.isPureRace) {
    insights.push(
      'שומר על מבנה בטוח לחלוטין — 0/36 הטלות פגיעה ישירות ליריב'
    );
  } else if (dangerDelta < -0.08) {
    insights.push(
      `מצמצם חשיפה של כלים בודדים (סיכון פגיעה נוכחי: ${comb.opponentHitRolls}/36 הטלות, ${comb.opponentHitPct}%)`
    );
  } else if (comb.opponentHitRolls >= 11) {
    insights.push(
      `משאיר כלי חשוף (Blot) בסיכון של ${comb.opponentHitRolls}/36 הטלות (${comb.opponentHitPct}%) במאבק על העמדה`
    );
  }

  // 7. Back-checker escape & prime trapping differential
  const trappedDelta = (cf.trappedCheckersDiff - bf.trappedCheckersDiff) * sign;
  const escapeDelta = (cf.escapeChancesDiff - bf.escapeChancesDiff) * sign;
  if (trappedDelta > 0) {
    insights.push(
      'מחלץ כלי אחורי מסכנת חסימה או כולא כלי אחורי של היריב מאחורי חומה'
    );
  } else if (escapeDelta >= 6) {
    insights.push(
      `משפר את סיכויי היציאה של הכלים האחוריים ל-${comb.escapeRolls}/36 הטלות (${comb.escapePct}%)`
    );
  }

  // 8. Builder diversity & outfield control
  const builderGain = (cf.buildersDiff - bf.buildersDiff) * sign;
  if (builderGain > 0 && !cf.isPureRace) {
    insights.push(
      'מגדיל את גמישות הבנאים בלוח החיצוני (נקודות 7–11) לקראת התור הבא'
    );
  }

  // 9. Defensive anchors & Bar pressure
  const anchorGain = (cf.anchorsDiff - bf.anchorsDiff) * sign;
  if (anchorGain > 0) {
    insights.push('מבסס עוגן הגנתי בטוח בבית של היריב');
  }

  // 10. Pure race Effective Pip Count (EPC) & Kleinman optimization
  if (cf.isPureRace) {
    const epcGainVsBest =
      (cf.effectivePipDiff - bestF.effectivePipDiff) * sign;
    if (isBest || epcGainVsBest >= -0.2) {
      insights.push(
        'מצמצם בזבוז פיפס (EPC) ומשפר את מבנה הכלים במרוץ לפי נוסחת קליינמן'
      );
    } else {
      insights.push(
        'יוצר בזבוז פיפס (Wastage) על נקודות נמוכות בבית ופוגע ביעילות המרוץ'
      );
    }
  }

  const uniqueInsights = Array.from(new Set(insights));
  if (uniqueInsights.length < 2) {
    if (isBest) {
      uniqueInsights.push(
        'המהלך המאוזן והמדויק ביותר בעמדה הנוכחית',
        'מקדם כלים ביעילות במרוץ לסיום תוך שמירה על מבנה הגנתי יציב'
      );
    } else {
      uniqueInsights.push(
        'פיזור כלים פחות יעיל בהשוואה לקו המוביל של המנוע',
        'מאבד מעט מהיתרון העמדתי לקראת הטלות הקובייה הבאות'
      );
    }
  }

  return uniqueInsights.slice(0, 4);
}

/**
 * Evaluates a candidate board state using 2-Ply Expectiminimax:
 * computes the probability-weighted expected equity across all 21 opponent dice rolls,
 * assuming the opponent plays their best 1-Ply response for each roll.
 */
function evaluate2PlyExpectiminimax(
  afterPlayerMoveState: BoardState,
  playerWhoMoved: Player,
  cache: Map<string, EvalResult>
): { equity: number; winProb: number; gammonProb: number } {
  if (
    afterPlayerMoveState.borneOff.white >= 15 ||
    afterPlayerMoveState.borneOff.black >= 15
  ) {
    const termEval = evalWithCache(afterPlayerMoveState, cache);
    return {
      equity: termEval.equity,
      winProb: termEval.winProb,
      gammonProb: termEval.gammonProb,
    };
  }

  const opponent: Player = playerWhoMoved === 'white' ? 'black' : 'white';
  const oppTurnState: BoardState = {
    points: [...afterPlayerMoveState.points],
    bar: { ...afterPlayerMoveState.bar },
    borneOff: { ...afterPlayerMoveState.borneOff },
    currentTurn: opponent,
    dice: [],
    remainingMoves: [],
  };

  let expectedEquity = 0;
  let expectedGammon = 0;

  for (const outcome of ALL_21_DICE_OUTCOMES) {
    const oppMoves = generateLegalTurnMoves(oppTurnState, outcome.dice);

    if (oppMoves.length === 0) {
      const passEval = evalWithCache(oppTurnState, cache);
      expectedEquity += outcome.probability * passEval.equity;
      expectedGammon += outcome.probability * passEval.gammonProb;
    } else {
      let bestOppEquity = opponent === 'white' ? -Infinity : Infinity;
      let bestOppGammon = 0;

      for (const oppMove of oppMoves) {
        const ev = evalWithCache(oppMove.resultingState, cache);
        if (opponent === 'white') {
          if (ev.equity > bestOppEquity) {
            bestOppEquity = ev.equity;
            bestOppGammon = ev.gammonProb;
          }
        } else {
          if (ev.equity < bestOppEquity) {
            bestOppEquity = ev.equity;
            bestOppGammon = ev.gammonProb;
          }
        }
      }

      expectedEquity += outcome.probability * bestOppEquity;
      expectedGammon += outcome.probability * bestOppGammon;
    }
  }

  const clampedEquity = Math.max(-3.0, Math.min(3.0, expectedEquity));
  const equity =
    Math.abs(clampedEquity) < 1e-6 ? 0 : Number(clampedEquity.toFixed(2));
  const rawWinProb =
    equity === 0 ? 50 : 100 / (1 + Math.exp(-clampedEquity * 1.18));
  const winProb = Number(Math.max(0.5, Math.min(99.5, rawWinProb)).toFixed(1));
  const gammonProb = Number(Math.max(0, Math.min(95, expectedGammon)).toFixed(1));

  return { equity, winProb, gammonProb };
}

/**
 * Selective 3-Ply Extension for close tactical contact decisions:
 * Evaluates a candidate move one ply deeper by simulating the opponent's best reply
 * across representative dice rolls and then finding the moving player's best counter-reply.
 */
function evaluateSelective3PlyExtension(
  afterPlayerMoveState: BoardState,
  playerWhoMoved: Player,
  base2PlyEquity: number,
  baseGammonProb: number,
  cache: Map<string, EvalResult>
): { equity: number; winProb: number; gammonProb: number } {
  if (
    afterPlayerMoveState.borneOff.white >= 15 ||
    afterPlayerMoveState.borneOff.black >= 15
  ) {
    const termEval = evalWithCache(afterPlayerMoveState, cache);
    return {
      equity: termEval.equity,
      winProb: termEval.winProb,
      gammonProb: termEval.gammonProb,
    };
  }

  const opponent: Player = playerWhoMoved === 'white' ? 'black' : 'white';
  const oppTurnState: BoardState = {
    points: [...afterPlayerMoveState.points],
    bar: { ...afterPlayerMoveState.bar },
    borneOff: { ...afterPlayerMoveState.borneOff },
    currentTurn: opponent,
    dice: [],
    remainingMoves: [],
  };

  let ply3ExpectedEquity = 0;

  for (const oppRoll of SELECTIVE_3PLY_ROLLS) {
    const oppMoves = generateLegalTurnMoves(oppTurnState, oppRoll.dice);
    let bestOppState = oppTurnState;

    if (oppMoves.length > 0) {
      let bestOppEq = opponent === 'white' ? -Infinity : Infinity;
      for (const om of oppMoves) {
        const eq = evalWithCache(om.resultingState, cache).equity;
        if (opponent === 'white' ? eq > bestOppEq : eq < bestOppEq) {
          bestOppEq = eq;
          bestOppState = om.resultingState;
        }
      }
    }

    const playerReTurnState: BoardState = {
      points: [...bestOppState.points],
      bar: { ...bestOppState.bar },
      borneOff: { ...bestOppState.borneOff },
      currentTurn: playerWhoMoved,
      dice: [],
      remainingMoves: [],
    };

    let playerCounterExpected = 0;
    for (const counterRoll of SELECTIVE_3PLY_ROLLS) {
      const counterMoves = generateLegalTurnMoves(
        playerReTurnState,
        counterRoll.dice
      );
      if (counterMoves.length === 0) {
        playerCounterExpected +=
          counterRoll.probability *
          evalWithCache(playerReTurnState, cache).equity;
      } else {
        let bestCounterEq = playerWhoMoved === 'white' ? -Infinity : Infinity;
        for (const cm of counterMoves) {
          const eq = evalWithCache(cm.resultingState, cache).equity;
          if (
            playerWhoMoved === 'white' ? eq > bestCounterEq : eq < bestCounterEq
          ) {
            bestCounterEq = eq;
          }
        }
        playerCounterExpected += counterRoll.probability * bestCounterEq;
      }
    }

    ply3ExpectedEquity += oppRoll.probability * playerCounterExpected;
  }

  const blended = base2PlyEquity * 0.7 + ply3ExpectedEquity * 0.3;
  const clampedEquity = Math.max(-3.0, Math.min(3.0, blended));
  const equity =
    Math.abs(clampedEquity) < 1e-6 ? 0 : Number(clampedEquity.toFixed(2));
  const rawWinProb =
    equity === 0 ? 50 : 100 / (1 + Math.exp(-clampedEquity * 1.18));
  const winProb = Number(Math.max(0.5, Math.min(99.5, rawWinProb)).toFixed(1));

  return { equity, winProb, gammonProb: baseGammonProb };
}

const ANYTIME_CUTOFF_MS = 9200;

class SearchCutoffReachedError extends Error {
  constructor() {
    super('ANYTIME_CUTOFF_REACHED');
  }
}

/**
 * Progressive beam widths [K1, K2, K3, K4] for Depths 3, 4, and 5:
 * Depth 4+ and Depth 5+ never use a smaller root beam K1 or ply-2 beam K2 than Depth 3+:
 * - Depth 3: K1=8 -> K2=4
 * - Depth 4: K1=8 -> K2=4 -> K3=3
 * - Depth 5: K1=8 -> K2=4 -> K3=3 -> K4=2
 */
const BEAM_SCHEDULE_BY_DEPTH: Record<3 | 4 | 5, number[]> = {
  3: [8, 4],
  4: [8, 4, 3],
  5: [8, 4, 3, 2],
};

/**
 * Recursive beam-pruned Expectiminimax evaluator with Transposition Table,
 * Pre-Turn / Post-Turn Leaf-Averaging (eliminating the Odd-Even Horizon Effect),
 * and 9,200ms Anytime Cutoff.
 */
function evaluateBeamExpectiminimaxNode(
  afterMoveState: BoardState,
  nextTurnPlayer: Player,
  remainingPlies: number,
  beamSchedule: number[],
  plyIndex: number,
  evalCache: Map<string, EvalResult>,
  transpositionTable: Map<string, number>,
  deadlineMs: number
): number {
  if (performance.now() > deadlineMs) {
    throw new SearchCutoffReachedError();
  }

  if (
    remainingPlies <= 0 ||
    afterMoveState.borneOff.white >= 15 ||
    afterMoveState.borneOff.black >= 15
  ) {
    return evalWithCache(afterMoveState, evalCache).equity;
  }

  const ttKey = `${getPhysicalStateSignature(afterMoveState)}|${nextTurnPlayer}|${remainingPlies}`;
  const cachedEq = transpositionTable.get(ttKey);
  if (cachedEq !== undefined) {
    return cachedEq;
  }

  // At the leaf chance ply (remainingPlies === 1), use exact 2-ply Expectiminimax over all 21 rolls
  // combined with Pre-Turn / Post-Turn Leaf-Averaging to eliminate the Odd-Even Horizon Effect.
  if (remainingPlies === 1) {
    const preTurnEq = evalWithCache(afterMoveState, evalCache).equity;
    const prevPlayer: Player = nextTurnPlayer === 'white' ? 'black' : 'white';
    const res2 = evaluate2PlyExpectiminimax(
      afterMoveState,
      prevPlayer,
      evalCache
    );
    const leafAvgEq = Math.max(
      -3.0,
      Math.min(3.0, 0.5 * preTurnEq + 0.5 * res2.equity)
    );
    transpositionTable.set(ttKey, leafAvgEq);
    return leafAvgEq;
  }

  const turnState: BoardState = {
    points: [...afterMoveState.points],
    bar: { ...afterMoveState.bar },
    borneOff: { ...afterMoveState.borneOff },
    currentTurn: nextTurnPlayer,
    dice: [],
    remainingMoves: [],
  };

  const opponent: Player = nextTurnPlayer === 'white' ? 'black' : 'white';
  const beamK = beamSchedule[Math.min(plyIndex, beamSchedule.length - 1)] ?? 2;
  const rolls = ALL_21_DICE_OUTCOMES;
  let totalProb = 0;
  let expectedEquity = 0;

  for (const outcome of rolls) {
    if (performance.now() > deadlineMs) {
      throw new SearchCutoffReachedError();
    }

    const moves = generateLegalTurnMoves(turnState, outcome.dice);
    if (moves.length === 0) {
      const passEq = evaluateBeamExpectiminimaxNode(
        turnState,
        opponent,
        remainingPlies - 1,
        beamSchedule,
        plyIndex + 1,
        evalCache,
        transpositionTable,
        deadlineMs
      );
      expectedEquity += outcome.probability * passEq;
      totalProb += outcome.probability;
      continue;
    }

    // Score moves at 1-ply to select top `beamK` candidates
    const scored = moves.map((m) => ({
      move: m,
      eq1: evalWithCache(m.resultingState, evalCache).equity,
    }));
    scored.sort((a, b) =>
      nextTurnPlayer === 'white' ? b.eq1 - a.eq1 : a.eq1 - b.eq1
    );
    const beamMoves = scored.slice(0, beamK);

    let bestMoveEq = nextTurnPlayer === 'white' ? -Infinity : Infinity;
    for (const cand of beamMoves) {
      const childEq = evaluateBeamExpectiminimaxNode(
        cand.move.resultingState,
        opponent,
        remainingPlies - 1,
        beamSchedule,
        plyIndex + 1,
        evalCache,
        transpositionTable,
        deadlineMs
      );
      if (
        nextTurnPlayer === 'white'
          ? childEq > bestMoveEq
          : childEq < bestMoveEq
      ) {
        bestMoveEq = childEq;
      }
    }

    expectedEquity += outcome.probability * bestMoveEq;
    totalProb += outcome.probability;
  }

  const normalizedEq =
    totalProb > 0
      ? Math.max(-3.0, Math.min(3.0, expectedEquity / totalProb))
      : evalWithCache(afterMoveState, evalCache).equity;
  transpositionTable.set(ttKey, normalizedEq);
  return normalizedEq;
}

/**
 * Analyzes a board position for a given dice roll at Depths 1..5 (`SearchDepth = 1 | 2 | 3 | 4 | 5`)
 * with Opening & Reply Book lookup, Transposition Table, Progressive Beam Pruning,
 * 9,200ms Anytime Cutoff, exact 36-roll combinatorics, and Grandmaster Hebrew comparisons.
 */
export function analyzePosition(
  state: BoardState,
  dice: [number, number],
  depth: SearchDepth = 2,
  playedSteps?: MoveStep[]
): TurnAnalysisResult {
  const startTime = performance.now();
  const deadlineMs = startTime + ANYTIME_CUTOFF_MS;
  const player = state.currentTurn;
  const opponent: Player = player === 'white' ? 'black' : 'white';
  const sign = player === 'white' ? 1 : -1;
  const evalCache = new Map<string, EvalResult>();
  const transpositionTable = new Map<string, number>();
  const legalMoves = generateLegalTurnMoves(state, dice);
  const isOpeningBoard = isOpeningOrReplyBookState(state);
  const { gamePhase, gamePhaseLabel, strategicPlan } = getPhaseAndStrategicPlan(
    state,
    player
  );

  if (legalMoves.length === 0) {
    const baseEval = evalWithCache(state, evalCache);
    return {
      topMoves: [],
      gamePhase,
      gamePhaseLabel,
      strategicPlan,
      requestedDepth: depth,
      completedDepth: depth,
      elapsedMs: Math.round(performance.now() - startTime),
      playedMoveAnalysis: playedSteps
        ? {
            notation: 'ללא מהלך',
            equity: baseEval.equity,
            equityLoss: 0,
            classification: 'Best',
            explanation: ['כל המהלכים החוקיים חסומים בהטלת קוביות זו'],
          }
        : undefined,
    };
  }

  // Step 1: Score all legal moves at 1-Ply (plus Opening & Reply Book theoretical bonus)
  const scored1Ply = legalMoves.map((move) => {
    const eval1 = evalWithCache(move.resultingState, evalCache);
    const bookEntry = isOpeningBoard
      ? lookupOpeningBookMove(state, dice, move.resultingState)
      : null;
    const bookBonus = bookEntry ? bookEntry.equityAdjustment * sign : 0;
    const adjustedEquity = Number(
      Math.max(-3.0, Math.min(3.0, eval1.equity + bookBonus)).toFixed(2)
    );

    return {
      move,
      equity: adjustedEquity,
      winProb: eval1.winProb,
      gammonProb: eval1.gammonProb,
      bookBonus,
    };
  });

  scored1Ply.sort((a, b) =>
    player === 'white' ? b.equity - a.equity : a.equity - b.equity
  );

  let finalCandidates = scored1Ply;
  let completedDepth: SearchDepth = 1;

  // Step 2: Refine with 2-Ply Expectiminimax (top 16 candidates) if depth >= 2
  if (depth >= 2 && scored1Ply.length > 1) {
    const topPool = scored1Ply.slice(0, 16).map((item) => {
      const eval2 = evaluate2PlyExpectiminimax(
        item.move.resultingState,
        player,
        evalCache
      );
      const adjustedEq = Number(
        Math.max(-3.0, Math.min(3.0, eval2.equity + item.bookBonus)).toFixed(2)
      );
      return {
        move: item.move,
        equity: adjustedEq,
        winProb: eval2.winProb,
        gammonProb: eval2.gammonProb,
        bookBonus: item.bookBonus,
      };
    });

    topPool.sort((a, b) =>
      player === 'white' ? b.equity - a.equity : a.equity - b.equity
    );
    completedDepth = 2;

    // For depth === 2: Selective 3-Ply Extension when in contact and top 2 moves are within 0.06 equity
    const basePositionEval = evalWithCache(state, evalCache);
    if (
      depth === 2 &&
      !basePositionEval.features.isPureRace &&
      !isOpeningBoard &&
      topPool.length >= 2 &&
      Math.abs(topPool[0].equity - topPool[1].equity) <= 0.06
    ) {
      const top3Extended = topPool.slice(0, 3).map((item) => {
        const eval3 = evaluateSelective3PlyExtension(
          item.move.resultingState,
          player,
          item.equity,
          item.gammonProb,
          evalCache
        );
        return {
          ...item,
          equity: eval3.equity,
          winProb: eval3.winProb,
          gammonProb: eval3.gammonProb,
        };
      });

      const remainingPool = topPool.slice(3);
      const merged = [...top3Extended, ...remainingPool];
      merged.sort((a, b) =>
        player === 'white' ? b.equity - a.equity : a.equity - b.equity
      );
      finalCandidates = merged;
    } else {
      finalCandidates = topPool;
    }

    // Step 3: Iterative Deepening for Depths 3+, 4+, and 5+ with Progressive Beam Pruning & Anytime Cutoff
    if (depth >= 3) {
      const targetDepths: Array<3 | 4 | 5> =
        depth === 3 ? [3] : depth === 4 ? [3, 4] : [3, 4, 5];

      for (const d of targetDepths) {
        if (performance.now() > deadlineMs) {
          break;
        }
        try {
          const schedule = BEAM_SCHEDULE_BY_DEPTH[d];
          const k1 = schedule[0];
          const rootBeam = finalCandidates.slice(0, k1);
          const rootRest = finalCandidates.slice(k1);

          const refinedBeam = rootBeam.map((item) => {
            const deepEq = evaluateBeamExpectiminimaxNode(
              item.move.resultingState,
              opponent,
              d - 1,
              schedule,
              1,
              evalCache,
              transpositionTable,
              deadlineMs
            );
            // Eliminate the Odd-Even Horizon Effect (GNU Backgammon damping):
            // combine 0.75 * E_ply(d) + 0.25 * E_ply(d-1) at every deep layer.
            const combinedRaw =
              0.75 * (deepEq + item.bookBonus) + 0.25 * item.equity;
            const blendedEq = Math.max(-3.0, Math.min(3.0, combinedRaw));
            const roundedEq =
              Math.abs(blendedEq) < 1e-6 ? 0 : Number(blendedEq.toFixed(2));
            const rawWin =
              roundedEq === 0 ? 50 : 100 / (1 + Math.exp(-blendedEq * 1.18));
            const winProb = Number(
              Math.max(0.5, Math.min(99.5, rawWin)).toFixed(1)
            );
            return {
              ...item,
              equity: roundedEq,
              winProb,
            };
          });

          const nextCandidates = [...refinedBeam, ...rootRest];
          nextCandidates.sort((a, b) =>
            player === 'white' ? b.equity - a.equity : a.equity - b.equity
          );
          finalCandidates = nextCandidates;
          completedDepth = d;
        } catch (err) {
          if (err instanceof SearchCutoffReachedError) {
            // Anytime cutoff reached: keep the deepest completed layer
            break;
          }
          throw err;
        }
      }
    }
  }

  const bestCandidate = finalCandidates[0];
  const bestState = bestCandidate.move.resultingState;
  const bestCombinatorics = computeExactDiceCombinatorics(bestState, player);

  const topMoves: CandidateMoveAnalysis[] = finalCandidates
    .slice(0, 3)
    .map((cand, idx) => {
      const rawLoss =
        player === 'white'
          ? Math.max(0, bestCandidate.equity - cand.equity)
          : Math.max(0, cand.equity - bestCandidate.equity);
      const equityLoss = Number(rawLoss.toFixed(2));
      const combinatorics =
        idx === 0
          ? bestCombinatorics
          : computeExactDiceCombinatorics(cand.move.resultingState, player);
      const bookEntry = isOpeningBoard
        ? lookupOpeningBookMove(state, dice, cand.move.resultingState)
        : null;

      return {
        move: cand.move,
        notation: formatTurnNotation(cand.move.steps, player),
        equity: cand.equity,
        winProb: cand.winProb,
        gammonProb: cand.gammonProb,
        equityLoss,
        combinatorics,
        isBookMove: Boolean(bookEntry),
        bookBadgeLabel: bookEntry ? 'ספר פתיחות' : undefined,
        explanation: generateTacticalExplanation(
          state,
          cand.move.resultingState,
          bestState,
          player,
          idx === 0 || equityLoss <= 0.02,
          dice,
          evalCache,
          combinatorics,
          bestCombinatorics
        ),
      };
    });

  let runnerUpComparison: string | undefined = undefined;
  if (topMoves.length >= 2) {
    runnerUpComparison = buildRunnerUpComparison(
      topMoves[0],
      topMoves[1],
      player
    );
    topMoves[0].runnerUpComparison = runnerUpComparison;
  }

  let playedMoveAnalysis: TurnAnalysisResult['playedMoveAnalysis'] = undefined;

  if (playedSteps) {
    let playedState: BoardState = {
      ...state,
      points: [...state.points],
      bar: { ...state.bar },
      borneOff: { ...state.borneOff },
      dice: [...state.dice],
      remainingMoves: [...state.remainingMoves],
    };
    for (const s of playedSteps) {
      playedState = applyMoveStep(playedState, s);
    }

    const playedSig = getPhysicalStateSignature(playedState);
    const matchedInFinal = finalCandidates.find(
      (c) => getPhysicalStateSignature(c.move.resultingState) === playedSig
    );

    let playedEquity: number;
    if (matchedInFinal) {
      playedEquity = matchedInFinal.equity;
    } else if (depth >= 2 && scored1Ply.length > 1) {
      const bookEntry = isOpeningBoard
        ? lookupOpeningBookMove(state, dice, playedState)
        : null;
      const bookBonus = bookEntry ? bookEntry.equityAdjustment * sign : 0;
      const eval2 = evaluate2PlyExpectiminimax(playedState, player, evalCache);
      playedEquity = Number(
        Math.max(-3.0, Math.min(3.0, eval2.equity + bookBonus)).toFixed(2)
      );
    } else {
      const bookEntry = isOpeningBoard
        ? lookupOpeningBookMove(state, dice, playedState)
        : null;
      const bookBonus = bookEntry ? bookEntry.equityAdjustment * sign : 0;
      playedEquity = Number(
        Math.max(
          -3.0,
          Math.min(3.0, evalWithCache(playedState, evalCache).equity + bookBonus)
        ).toFixed(2)
      );
    }

    const rawLoss =
      player === 'white'
        ? Math.max(0, bestCandidate.equity - playedEquity)
        : Math.max(0, playedEquity - bestCandidate.equity);
    const equityLoss = Number(rawLoss.toFixed(2));
    const classification = classifyEquityLoss(equityLoss);
    const playedComb = computeExactDiceCombinatorics(playedState, player);

    playedMoveAnalysis = {
      notation: formatTurnNotation(playedSteps, player),
      equity: playedEquity,
      equityLoss,
      classification,
      explanation: generateTacticalExplanation(
        state,
        playedState,
        bestState,
        player,
        classification === 'Best',
        dice,
        evalCache,
        playedComb,
        bestCombinatorics
      ),
    };
  }

  return {
    topMoves,
    gamePhase,
    gamePhaseLabel,
    strategicPlan,
    runnerUpComparison,
    requestedDepth: depth,
    completedDepth,
    elapsedMs: Math.round(performance.now() - startTime),
    playedMoveAnalysis,
  };
}
