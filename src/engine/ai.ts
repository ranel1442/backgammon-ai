import { evaluatePosition, type EvalResult } from './evaluator';
import { classifyGamePhase, type GamePhase } from './neuralNet';
import {
  isOpeningOrReplyBookState,
  lookupOpeningBookMove,
} from './openingBook';
import { applyMoveStep, generateLegalTurnMoves, getPipCount } from './rules';
import type { BoardState, MoveStep, Player, TurnMove } from './types';

export type SearchDepth = 1 | 2 | 3 | 4 | 5;

/** Champion search depth (Depth 3+ won the 10,000-game league 4-0-0) */
export const CHAMPION_SEARCH_DEPTH: SearchDepth = 3;

/** Hard safety deadline for live gameplay / hints (`gameplayWorker`) */
export const LIVE_SEARCH_CUTOFF_MS = 28000;

/** Hard safety deadline for background move review (`analysisWorker`) */
export const BACKGROUND_ANALYSIS_CUTOFF_MS = 22000;

/** Default safety cutoff alias (`28,000ms`) */
export const DEFAULT_ANYTIME_CUTOFF_MS = LIVE_SEARCH_CUTOFF_MS;

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
  neuralEvals?: number;
  completedPly3?: number;
  targetPly3?: number;
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

/**
 * Fast zero-allocation 53-bit IEEE-754 integer board hash combining dual orthogonal 32-bit FNV-1a / Murmur-style
 * integer hashes across all 24 points, bars, borneOff counts, active turn (`turnCode`), and search depth (`plyCode`).
 * Eliminates string concatenations and array joins in `evalCache` and `transpositionTable`.
 */
export function hashBoardStateNumeric(
  state: BoardState,
  turnCode: number = 0,
  plyCode: number = 0
): number {
  const pts = state.points;
  let h1 = 0x811c9dc5 | 0;
  let h2 = 0x9e3779b9 | 0;

  for (let i = 0; i < 24; i++) {
    const v = (pts[i] + 32) | 0;
    h1 = Math.imul(h1 ^ v, 0x01000193);
    h2 = Math.imul(h2 ^ (v + Math.imul(i + 1, 0x9e3779b1)), 0x85ebca6b);
  }

  const bw = state.bar.white | 0;
  const bb = state.bar.black | 0;
  const ow = state.borneOff.white | 0;
  const ob = state.borneOff.black | 0;
  const meta1 = bw | (bb << 5) | (ow << 10) | (ob << 15);
  const meta2 = (turnCode & 3) | ((plyCode & 31) << 2);

  h1 = Math.imul(h1 ^ meta1, 0x01000193);
  h1 = Math.imul(h1 ^ meta2, 0x01000193);
  h1 ^= h1 >>> 16;

  h2 = Math.imul(h2 ^ meta1, 0xc2b2ae35);
  h2 = Math.imul(h2 ^ Math.imul(meta2 + 1, 0x27d4eb2d), 0x85ebca6b);
  h2 ^= h2 >>> 15;

  // Top 21 bits of h1 * 2^32 + unsigned 32 bits of h2 -> exact 53-bit integer in [0, Number.MAX_SAFE_INTEGER]
  return (h1 >>> 11) * 4294967296 + (h2 >>> 0);
}

/**
 * Deduplicates move permutations that produce the exact same resulting physical `BoardState`,
 * returning each unique move alongside its precomputed 53-bit `boardHash`.
 */
function deduplicateMovesByResultingState(
  moves: TurnMove[]
): Array<{ move: TurnMove; boardHash: number }> {
  const len = moves.length;
  if (len === 0) return [];
  if (len === 1) {
    return [
      {
        move: moves[0],
        boardHash: hashBoardStateNumeric(moves[0].resultingState),
      },
    ];
  }

  const seen = new Set<number>();
  const unique: Array<{ move: TurnMove; boardHash: number }> = [];
  for (let i = 0; i < len; i++) {
    const m = moves[i];
    const h = hashBoardStateNumeric(m.resultingState);
    if (!seen.has(h)) {
      seen.add(h);
      unique.push({ move: m, boardHash: h });
    }
  }
  return unique;
}

interface SearchStats {
  neuralEvals: number;
}

class SearchCutoffReachedError extends Error {
  constructor() {
    super('SEARCH_DEADLINE_REACHED');
  }
}

function evalWithCache(
  state: BoardState,
  cache?: Map<number, EvalResult>,
  precomputedHash?: number,
  stats?: SearchStats
): EvalResult {
  if (!cache) {
    if (stats) stats.neuralEvals += 1;
    return evaluatePosition(state, true);
  }
  const key =
    precomputedHash !== undefined
      ? precomputedHash
      : hashBoardStateNumeric(state);
  const existing = cache.get(key);
  if (existing !== undefined) {
    return existing;
  }
  if (stats) stats.neuralEvals += 1;
  const computed = evaluatePosition(state, true);
  cache.set(key, computed);
  return computed;
}

/**
 * Fast-ranks legal moves for `movingPlayer` using cached full v3 ResNet equities when available
 * or the `< 5µs` Tactical Heuristic + Exact Bear-Off Tablebase (`evaluatePosition(s, false)`),
 * returning the top `limit` moves for direct full v3 ResNet evaluation.
 */
function selectTopMovesForResNet(
  uniqueMoves: Array<{ move: TurnMove; boardHash: number }>,
  movingPlayer: Player,
  limit: number,
  evalCache: Map<number, EvalResult>,
  fastRankCache: Map<number, number>
): Array<{ move: TurnMove; boardHash: number }> {
  if (uniqueMoves.length <= limit) {
    return uniqueMoves;
  }

  const isWhite = movingPlayer === 'white';
  const scored = new Array<{
    move: TurnMove;
    boardHash: number;
    fastScore: number;
  }>(uniqueMoves.length);

  for (let i = 0; i < uniqueMoves.length; i++) {
    const item = uniqueMoves[i];
    const cachedFull = evalCache.get(item.boardHash);
    let fastScore: number;
    if (cachedFull !== undefined) {
      fastScore = cachedFull.equity;
    } else {
      const cachedFast = fastRankCache.get(item.boardHash);
      if (cachedFast !== undefined) {
        fastScore = cachedFast;
      } else {
        fastScore = evaluatePosition(item.move.resultingState, false).equity;
        fastRankCache.set(item.boardHash, fastScore);
      }
    }
    scored[i] = {
      move: item.move,
      boardHash: item.boardHash,
      fastScore,
    };
  }

  scored.sort((a, b) =>
    isWhite ? b.fastScore - a.fastScore : a.fastScore - b.fastScore
  );

  const top = new Array<{ move: TurnMove; boardHash: number }>(limit);
  for (let i = 0; i < limit; i++) {
    top[i] = { move: scored[i].move, boardHash: scored[i].boardHash };
  }
  return top;
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
  cache?: Map<number, EvalResult>,
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

interface OptimalOpponentReply {
  state: BoardState;
  boardHash: number;
  equity: number;
  probability: number;
}

interface StageA2PlyResult {
  equity: number;
  rawEquity: number;
  winProb: number;
  gammonProb: number;
  optimalOppReplies: OptimalOpponentReply[];
}

const STAGE_A_ROOT_BASE_K1 = 8;
const STAGE_A_ROOT_MAX_K1 = 10;
const STAGE_A_ROOT_EXPAND_MARGIN = 0.06;
const STAGE_A_OPP_TOP_REPLIES = 6;

const STAGE_B_DEEP_BASE_K = 6;
const STAGE_B_DEEP_MAX_K = 8;
const STAGE_B_DEEP_EXPAND_MARGIN = 0.05;
const STAGE_B_COUNTER_TOP_MOVES = 4;

/**
 * Stage A (Ply 2 — Opponent's 21 rolls):
 * For each of the 21 opponent dice rolls, fast-ranks legal replies and evaluates the
 * **top 6** replies directly with the full v3 ResNet to identify the opponent's true
 * single optimal reply (`K2 = 1`), checking `deadlineMs` inside every Ply-2 dice loop.
 */
function evaluateStageA2PlyFoundation(
  afterPlayerMoveState: BoardState,
  playerWhoMoved: Player,
  evalCache: Map<number, EvalResult>,
  fastRankCache: Map<number, number>,
  stats: SearchStats,
  deadlineMs: number,
  precomputedBaseHash?: number
): StageA2PlyResult {
  const baseHash =
    precomputedBaseHash !== undefined
      ? precomputedBaseHash
      : hashBoardStateNumeric(afterPlayerMoveState);

  if (
    afterPlayerMoveState.borneOff.white >= 15 ||
    afterPlayerMoveState.borneOff.black >= 15
  ) {
    const termEval = evalWithCache(
      afterPlayerMoveState,
      evalCache,
      baseHash,
      stats
    );
    return {
      equity: termEval.equity,
      rawEquity: termEval.equity,
      winProb: termEval.winProb,
      gammonProb: termEval.gammonProb,
      optimalOppReplies: [],
    };
  }

  const opponent: Player = playerWhoMoved === 'white' ? 'black' : 'white';
  const oppIsWhite = opponent === 'white';
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
  const optimalOppReplies = new Array<OptimalOpponentReply>(
    ALL_21_DICE_OUTCOMES.length
  );

  for (let r = 0; r < ALL_21_DICE_OUTCOMES.length; r++) {
    if (performance.now() >= deadlineMs) {
      throw new SearchCutoffReachedError();
    }

    const outcome = ALL_21_DICE_OUTCOMES[r];
    const uniqueOppMoves = deduplicateMovesByResultingState(
      generateLegalTurnMoves(oppTurnState, outcome.dice)
    );

    if (uniqueOppMoves.length === 0) {
      const passEval = evalWithCache(oppTurnState, evalCache, baseHash, stats);
      expectedEquity += outcome.probability * passEval.equity;
      expectedGammon += outcome.probability * passEval.gammonProb;
      optimalOppReplies[r] = {
        state: oppTurnState,
        boardHash: baseHash,
        equity: passEval.equity,
        probability: outcome.probability,
      };
    } else {
      const topReplies = selectTopMovesForResNet(
        uniqueOppMoves,
        opponent,
        STAGE_A_OPP_TOP_REPLIES,
        evalCache,
        fastRankCache
      );

      let bestOppEquity = oppIsWhite ? -Infinity : Infinity;
      let bestOppGammon = 0;
      let bestOppState = topReplies[0].move.resultingState;
      let bestOppHash = topReplies[0].boardHash;

      for (let m = 0; m < topReplies.length; m++) {
        const rep = topReplies[m];
        const ev = evalWithCache(
          rep.move.resultingState,
          evalCache,
          rep.boardHash,
          stats
        );
        if (oppIsWhite ? ev.equity > bestOppEquity : ev.equity < bestOppEquity) {
          bestOppEquity = ev.equity;
          bestOppGammon = ev.gammonProb;
          bestOppState = rep.move.resultingState;
          bestOppHash = rep.boardHash;
        }
      }

      expectedEquity += outcome.probability * bestOppEquity;
      expectedGammon += outcome.probability * bestOppGammon;
      optimalOppReplies[r] = {
        state: bestOppState,
        boardHash: bestOppHash,
        equity: bestOppEquity,
        probability: outcome.probability,
      };
    }
  }

  const clampedEquity = Math.max(-3.0, Math.min(3.0, expectedEquity));
  const equity =
    Math.abs(clampedEquity) < 1e-6 ? 0 : Number(clampedEquity.toFixed(2));
  const rawWinProb =
    equity === 0 ? 50 : 100 / (1 + Math.exp(-clampedEquity * 1.18));
  const winProb = Number(Math.max(0.5, Math.min(99.5, rawWinProb)).toFixed(1));
  const gammonProb = Number(
    Math.max(0, Math.min(95, expectedGammon)).toFixed(1)
  );

  return {
    equity,
    rawEquity: clampedEquity,
    winProb,
    gammonProb,
    optimalOppReplies,
  };
}

/**
 * Stage B (Ply 3 — Our 21 follow-up rolls after Opponent's optimal reply `K2 = 1`):
 * Uses the 53-bit Numeric Transposition Table across all opponent optimal replies,
 * fast-ranks our legal follow-up moves on each of the 21 dice outcomes, and evaluates
 * the **top 4** moves directly with the full v3 ResNet, checking `deadlineMs` inside every Ply-3 dice loop.
 */
function evaluateStageB3PlyRefinement(
  optimalOppReplies: OptimalOpponentReply[],
  playerWhoMoved: Player,
  evalCache: Map<number, EvalResult>,
  fastRankCache: Map<number, number>,
  ply3TranspositionTable: Map<number, number>,
  stats: SearchStats,
  deadlineMs: number,
  fallbackEquity: number
): number {
  if (optimalOppReplies.length === 0) {
    return fallbackEquity;
  }

  const playerIsWhite = playerWhoMoved === 'white';
  const playerTurnCode = playerIsWhite ? 1 : 2;
  let totalExpectedPly3 = 0;

  for (let r = 0; r < optimalOppReplies.length; r++) {
    if (performance.now() >= deadlineMs) {
      throw new SearchCutoffReachedError();
    }

    const oppReply = optimalOppReplies[r];
    const afterOppState = oppReply.state;

    if (
      afterOppState.borneOff.white >= 15 ||
      afterOppState.borneOff.black >= 15
    ) {
      const termEq = evalWithCache(
        afterOppState,
        evalCache,
        oppReply.boardHash,
        stats
      ).equity;
      totalExpectedPly3 += oppReply.probability * termEq;
      continue;
    }

    const ttKey = hashBoardStateNumeric(afterOppState, playerTurnCode, 1);
    const cachedCounterExpectation = ply3TranspositionTable.get(ttKey);
    if (cachedCounterExpectation !== undefined) {
      totalExpectedPly3 += oppReply.probability * cachedCounterExpectation;
      continue;
    }

    const counterTurnState: BoardState = {
      points: [...afterOppState.points],
      bar: { ...afterOppState.bar },
      borneOff: { ...afterOppState.borneOff },
      currentTurn: playerWhoMoved,
      dice: [],
      remainingMoves: [],
    };

    let counterRollExpected = 0;

    for (let c = 0; c < ALL_21_DICE_OUTCOMES.length; c++) {
      if (performance.now() >= deadlineMs) {
        throw new SearchCutoffReachedError();
      }

      const counterOutcome = ALL_21_DICE_OUTCOMES[c];
      const uniqueCounterMoves = deduplicateMovesByResultingState(
        generateLegalTurnMoves(counterTurnState, counterOutcome.dice)
      );

      if (uniqueCounterMoves.length === 0) {
        const passEq = evalWithCache(
          counterTurnState,
          evalCache,
          oppReply.boardHash,
          stats
        ).equity;
        counterRollExpected += counterOutcome.probability * passEq;
      } else {
        const top4CounterMoves = selectTopMovesForResNet(
          uniqueCounterMoves,
          playerWhoMoved,
          STAGE_B_COUNTER_TOP_MOVES,
          evalCache,
          fastRankCache
        );

        let bestCounterEq = playerIsWhite ? -Infinity : Infinity;
        for (let m = 0; m < top4CounterMoves.length; m++) {
          const cm = top4CounterMoves[m];
          const ev = evalWithCache(
            cm.move.resultingState,
            evalCache,
            cm.boardHash,
            stats
          );
          if (
            playerIsWhite
              ? ev.equity > bestCounterEq
              : ev.equity < bestCounterEq
          ) {
            bestCounterEq = ev.equity;
          }
        }
        counterRollExpected += counterOutcome.probability * bestCounterEq;
      }
    }

    const clampedCounterExp = Math.max(
      -3.0,
      Math.min(3.0, counterRollExpected)
    );
    ply3TranspositionTable.set(ttKey, clampedCounterExp);
    totalExpectedPly3 += oppReply.probability * clampedCounterExp;
  }

  return Math.max(-3.0, Math.min(3.0, totalExpectedPly3));
}

/**
 * Analyzes a board position for a given dice roll (locked by default to Champion `Depth 3+`)
 * using Iterative Deepening calibrated for maximum Grandmaster accuracy in a 20–28s budget:
 * - Stage A: Complete Deep 2-Ply Foundation (all root moves with v3 ResNet -> top K1=8..10 -> top 6 v3 ResNet opponent replies per roll, K2=1).
 * - Stage B: Wide 3-Ply Grandmaster Refinement (top K_deep=6..8 -> top 4 v3 ResNet counter-moves per roll + 0.75*E3 + 0.25*E2 GNU horizon damping).
 * - Strict safety cutoff (`28,000ms` Live / `22,000ms` Background) checked inside every Ply-2 and Ply-3 dice loop.
 */
export function analyzePosition(
  state: BoardState,
  dice: [number, number],
  depth: SearchDepth = CHAMPION_SEARCH_DEPTH,
  playedSteps?: MoveStep[],
  cutoffMs: number = LIVE_SEARCH_CUTOFF_MS
): TurnAnalysisResult {
  const startTime = performance.now();
  const deadlineMs = startTime + Math.max(200, cutoffMs);
  const player = state.currentTurn;
  const sign = player === 'white' ? 1 : -1;
  const stats: SearchStats = { neuralEvals: 0 };
  const evalCache = new Map<number, EvalResult>();
  const fastRankCache = new Map<number, number>();
  const ply3TranspositionTable = new Map<number, number>();

  const uniqueLegalMoves = deduplicateMovesByResultingState(
    generateLegalTurnMoves(state, dice)
  );
  const isOpeningBoard = isOpeningOrReplyBookState(state);
  const { gamePhase, gamePhaseLabel, strategicPlan } = getPhaseAndStrategicPlan(
    state,
    player
  );

  if (uniqueLegalMoves.length === 0) {
    const baseEval = evalWithCache(state, evalCache, undefined, stats);
    const elapsedMs = Math.round(performance.now() - startTime);
    if (depth >= 3) {
      console.log(
        `[AI] Depth 3+ completed in ${elapsedMs}ms (${stats.neuralEvals} neural evals, completedPly3=0/0)`
      );
    }
    return {
      topMoves: [],
      gamePhase,
      gamePhaseLabel,
      strategicPlan,
      requestedDepth: depth,
      completedDepth: depth,
      elapsedMs,
      neuralEvals: stats.neuralEvals,
      completedPly3: 0,
      targetPly3: 0,
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

  // Stage A (Ply 1 — Root): Evaluate ALL unique legal root moves with full v3 ResNet
  interface SearchCandidate {
    move: TurnMove;
    boardHash: number;
    equity: number;
    rawStageAEq: number;
    winProb: number;
    gammonProb: number;
    bookBonus: number;
    optimalOppReplies: OptimalOpponentReply[];
    completedStageA: boolean;
    completedStageB: boolean;
  }

  const scored1Ply: SearchCandidate[] = uniqueLegalMoves.map(
    ({ move, boardHash }) => {
      const eval1 = evalWithCache(
        move.resultingState,
        evalCache,
        boardHash,
        stats
      );
      const bookEntry = isOpeningBoard
        ? lookupOpeningBookMove(state, dice, move.resultingState)
        : null;
      const bookBonus = bookEntry ? bookEntry.equityAdjustment * sign : 0;
      const clamped1 = Math.max(-3.0, Math.min(3.0, eval1.equity + bookBonus));
      const adjustedEquity =
        Math.abs(clamped1) < 1e-6 ? 0 : Number(clamped1.toFixed(2));

      return {
        move,
        boardHash,
        equity: adjustedEquity,
        rawStageAEq: clamped1,
        winProb: eval1.winProb,
        gammonProb: eval1.gammonProb,
        bookBonus,
        optimalOppReplies: [],
        completedStageA: false,
        completedStageB: false,
      };
    }
  );

  scored1Ply.sort((a, b) =>
    player === 'white' ? b.equity - a.equity : a.equity - b.equity
  );

  let finalCandidates: SearchCandidate[] = scored1Ply;
  let completedDepth: SearchDepth = 1;
  let completedPly3 = 0;
  let targetPly3 = 0;

  if (depth >= 2 && scored1Ply.length === 1) {
    completedDepth = depth;
    if (depth >= 3) {
      completedPly3 = 1;
      targetPly3 = 1;
    }
  } else if (depth >= 2 && scored1Ply.length > 1) {
    // Stage A (Ply 2): Keep top K1 = 8 root candidates (expand up to 10 if within 0.06 equity of #1)
    let k1Count = Math.min(STAGE_A_ROOT_BASE_K1, scored1Ply.length);
    const best1PlyEq = scored1Ply[0].equity;
    while (
      k1Count < Math.min(STAGE_A_ROOT_MAX_K1, scored1Ply.length) &&
      Math.abs(best1PlyEq - scored1Ply[k1Count].equity) <=
        STAGE_A_ROOT_EXPAND_MARGIN + 1e-9
    ) {
      k1Count++;
    }

    const stageACandidates: SearchCandidate[] = [];
    let stageACutoffHit = false;

    for (let i = 0; i < k1Count; i++) {
      const item = scored1Ply[i];
      try {
        const eval2 = evaluateStageA2PlyFoundation(
          item.move.resultingState,
          player,
          evalCache,
          fastRankCache,
          stats,
          deadlineMs,
          item.boardHash
        );
        const rawAdj = Math.max(
          -3.0,
          Math.min(3.0, eval2.rawEquity + item.bookBonus)
        );
        const adjustedEq =
          Math.abs(rawAdj) < 1e-6 ? 0 : Number(rawAdj.toFixed(2));
        stageACandidates.push({
          ...item,
          equity: adjustedEq,
          rawStageAEq: rawAdj,
          winProb: eval2.winProb,
          gammonProb: eval2.gammonProb,
          optimalOppReplies: eval2.optimalOppReplies,
          completedStageA: true,
        });
      } catch (err) {
        if (err instanceof SearchCutoffReachedError) {
          stageACutoffHit = true;
          break;
        }
        throw err;
      }
    }

    if (stageACandidates.length > 0) {
      stageACandidates.sort((a, b) =>
        player === 'white'
          ? b.rawStageAEq - a.rawStageAEq
          : a.rawStageAEq - b.rawStageAEq
      );
      const unsearchedTail = scored1Ply.slice(stageACandidates.length);
      finalCandidates = [...stageACandidates, ...unsearchedTail];
      if (!stageACutoffHit || stageACandidates.length >= 2) {
        completedDepth = 2;
      }
    }

    // Stage B — Wide 3-Ply Grandmaster Refinement (when depth >= 3)
    if (depth >= 3 && !stageACutoffHit && stageACandidates.length > 0) {
      let kDeep = Math.min(STAGE_B_DEEP_BASE_K, stageACandidates.length);
      const bestStageAEq = stageACandidates[0].rawStageAEq;
      while (
        kDeep < Math.min(STAGE_B_DEEP_MAX_K, stageACandidates.length) &&
        Math.abs(bestStageAEq - stageACandidates[kDeep].rawStageAEq) <=
          STAGE_B_DEEP_EXPAND_MARGIN + 1e-9
      ) {
        kDeep++;
      }

      targetPly3 = kDeep;
      const refinedStageB: SearchCandidate[] = [...stageACandidates];

      for (let i = 0; i < kDeep; i++) {
        const cand = refinedStageB[i];
        try {
          const e3Raw = evaluateStageB3PlyRefinement(
            cand.optimalOppReplies,
            player,
            evalCache,
            fastRankCache,
            ply3TranspositionTable,
            stats,
            deadlineMs,
            cand.rawStageAEq
          );

          // Apply GNU-style horizon damping: 0.75 * E3 + 0.25 * E2
          const dampedRaw =
            0.75 * (e3Raw + cand.bookBonus) + 0.25 * cand.rawStageAEq;
          const clampedDamped = Math.max(-3.0, Math.min(3.0, dampedRaw));
          const roundedEq =
            Math.abs(clampedDamped) < 1e-6
              ? 0
              : Number(clampedDamped.toFixed(2));
          const rawWin =
            roundedEq === 0 ? 50 : 100 / (1 + Math.exp(-clampedDamped * 1.18));
          const winProb = Number(
            Math.max(0.5, Math.min(99.5, rawWin)).toFixed(1)
          );

          refinedStageB[i] = {
            ...cand,
            equity: roundedEq,
            rawStageAEq: clampedDamped,
            winProb,
            completedStageB: true,
          };
          completedPly3 += 1;
        } catch (err) {
          if (err instanceof SearchCutoffReachedError) {
            break;
          }
          throw err;
        }
      }

      if (completedPly3 > 0) {
        // Sort completed Stage B candidates cleanly at the top, followed by remaining Stage A baseline candidates
        const finishedB = refinedStageB.slice(0, completedPly3);
        const restStageA = refinedStageB.slice(completedPly3);
        finishedB.sort((a, b) =>
          player === 'white'
            ? b.rawStageAEq - a.rawStageAEq
            : a.rawStageAEq - b.rawStageAEq
        );
        finalCandidates = [
          ...finishedB,
          ...restStageA,
          ...scored1Ply.slice(stageACandidates.length),
        ];
        completedDepth = 3;
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

    const playedHash = hashBoardStateNumeric(playedState);
    const matchedCandidate = finalCandidates.find(
      (c) =>
        c.boardHash === playedHash &&
        ((completedDepth >= 3 && c.completedStageB) ||
          (completedDepth === 2 && c.completedStageA) ||
          completedDepth === 1)
    );

    let playedEquity: number;
    if (matchedCandidate) {
      playedEquity = matchedCandidate.equity;
    } else {
      const bookEntry = isOpeningBoard
        ? lookupOpeningBookMove(state, dice, playedState)
        : null;
      const bookBonus = bookEntry ? bookEntry.equityAdjustment * sign : 0;

      if (
        depth >= 2 &&
        scored1Ply.length > 1 &&
        performance.now() + 25 < deadlineMs
      ) {
        try {
          const eval2 = evaluateStageA2PlyFoundation(
            playedState,
            player,
            evalCache,
            fastRankCache,
            stats,
            deadlineMs,
            playedHash
          );
          let eqVal = eval2.rawEquity + bookBonus;
          if (completedPly3 > 0 && performance.now() + 25 < deadlineMs) {
            const e3Played = evaluateStageB3PlyRefinement(
              eval2.optimalOppReplies,
              player,
              evalCache,
              fastRankCache,
              ply3TranspositionTable,
              stats,
              deadlineMs,
              eqVal
            );
            eqVal = 0.75 * (e3Played + bookBonus) + 0.25 * eqVal;
          }
          playedEquity = Number(
            Math.max(-3.0, Math.min(3.0, eqVal)).toFixed(2)
          );
        } catch (err) {
          if (err instanceof SearchCutoffReachedError) {
            const fallback1 = evalWithCache(
              playedState,
              evalCache,
              playedHash,
              stats
            );
            playedEquity = Number(
              Math.max(
                -3.0,
                Math.min(3.0, fallback1.equity + bookBonus)
              ).toFixed(2)
            );
          } else {
            throw err;
          }
        }
      } else {
        const eval1 = evalWithCache(playedState, evalCache, playedHash, stats);
        playedEquity = Number(
          Math.max(-3.0, Math.min(3.0, eval1.equity + bookBonus)).toFixed(2)
        );
      }
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

  const elapsedMs = Math.round(performance.now() - startTime);
  if (depth >= 3) {
    console.log(
      `[AI] Depth 3+ completed in ${elapsedMs}ms (${stats.neuralEvals} neural evals, completedPly3=${completedPly3}/${targetPly3})`
    );
  }

  return {
    topMoves,
    gamePhase,
    gamePhaseLabel,
    strategicPlan,
    runnerUpComparison,
    requestedDepth: depth,
    completedDepth,
    elapsedMs,
    neuralEvals: stats.neuralEvals,
    completedPly3,
    targetPly3,
    playedMoveAnalysis,
  };
}
