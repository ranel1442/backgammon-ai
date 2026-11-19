import {
  evaluateExactBearOff,
  extractPlayerHomeBoard,
  getExactExpectedBearOffTurns,
  isPureBearOffPosition,
} from './bearoffTablebase';
import {
  classifyGamePhase,
  getActiveNeuralNet,
  type GamePhase,
} from './neuralNet';
import { getPipCount } from './rules';
import type { BoardState, Player } from './types';

export interface EngineModelInfo {
  mode: 'nn' | 'hybrid-tablebase';
  hasNeuralNet: boolean;
  labelHe: string;
  trainedGames: number;
  eloEstimate: number;
  activeSubNet: 'Contact' | 'Crashed' | 'Race';
  activeSubNetHe: string;
  phase: GamePhase;
}

/**
 * Returns the active evaluation backend status, sub-network, and training metadata in Hebrew for UI display.
 */
export function getEngineModelInfo(state?: BoardState): EngineModelInfo {
  const phase: GamePhase = state ? classifyGamePhase(state) : 'contact';
  const activeSubNet: 'Contact' | 'Crashed' | 'Race' =
    phase === 'race' ? 'Race' : phase === 'crashed' ? 'Crashed' : 'Contact';
  const activeSubNetHe =
    phase === 'race'
      ? 'RaceNet (מרוץ)'
      : phase === 'crashed'
        ? 'CrashedNet (מצב מרוסק)'
        : 'ContactNet (קרב מגע)';

  const nn = getActiveNeuralNet();
  if (nn && nn.metadata.trainedGames > 0) {
    const gamesFmt = nn.metadata.trainedGames.toLocaleString('en-US');
    return {
      mode: 'nn',
      hasNeuralNet: true,
      labelHe: `רשת ResNet מאומנת (${gamesFmt} משחקים • ${activeSubNet})`,
      trainedGames: nn.metadata.trainedGames,
      eloEstimate: nn.metadata.eloEstimate,
      activeSubNet,
      activeSubNetHe,
      phase,
    };
  }
  return {
    mode: 'hybrid-tablebase',
    hasNeuralNet: false,
    labelHe: `מנוע היברידי + טבלת סיומים מדויקת (${activeSubNetHe})`,
    trainedGames: 0,
    eloEstimate: 1850,
    activeSubNet,
    activeSubNetHe,
    phase,
  };
}

export interface EvalFeatures {
  /** Positive when White leads the raw pip race (blackPips - whitePips) */
  pipDiff: number;
  /** Positive when White leads the Effective Pip Count race (blackEPC - whiteEPC) */
  effectivePipDiff: number;
  /** Difference in made home-board points (whiteHomePoints - blackHomePoints) */
  homePointsDiff: number;
  /** Difference in longest consecutive prime length (whitePrime - blackPrime) */
  primeLengthDiff: number;
  /** Difference in blot exposure danger (whiteBlotDanger - blackBlotDanger; lower is safer for White) */
  blotDangerDiff: number;
  /** Difference in defensive anchors in opponent's home board (whiteAnchors - blackAnchors) */
  anchorsDiff: number;
  /** Difference in opponent checkers on the bar (blackBar - whiteBar; positive favors White) */
  barDiff: number;
  /** Difference in trapped back checkers behind opponent primes (blackTrapped - whiteTrapped; positive favors White) */
  trappedCheckersDiff: number;
  /** Difference in 36-roll back-checker escape chances (whiteEscapeRolls - blackEscapeRolls; positive favors White) */
  escapeChancesDiff: number;
  /** Difference in flexible outfield builders on relative points 7..11 (whiteBuilders - blackBuilders) */
  buildersDiff: number;
  /** True when no checkers are on the bar and White's highest point < Black's lowest point */
  isPureRace: boolean;
}

export interface EvalResult {
  /** Cubeless equity clamped to [-3.0, +3.0], positive = White advantage, negative = Black advantage */
  equity: number;
  /** White win probability percentage (0 to 100) */
  winProb: number;
  /** Estimated gammon probability percentage (0 to 100) */
  gammonProb: number;
  /** Estimated backgammon probability percentage (0 to 100) */
  backgammonProb: number;
  /** Effective Pip Count (EPC) for both players including wastage */
  effectivePips: {
    white: number;
    black: number;
  };
  /** Detailed positional and race features */
  features: EvalFeatures;
  /** Summary details for UI readouts */
  details: {
    pipDiff: number;
    effectivePipDiff: number;
  };
}

interface PlayerPositionalMetrics {
  pips: number;
  effectivePips: number;
  wastage: number;
  homePoints: number;
  homeStrengthScore: number;
  maxPrimeLength: number;
  primeScore: number;
  blotDanger: number;
  anchors: number;
  anchorScore: number;
  barCount: number;
  barPenalty: number;
  stackPenalty: number;
  trappedCheckers: number;
  escapeRolls: number;
  trappedPenalty: number;
  buildersCount: number;
  builderScore: number;
  inOpponentHomeCount: number;
  borneOff: number;
}

const PRIME_REWARDS = [0, 0, 0.04, 0.11, 0.25, 0.44, 0.72];

/**
 * Detects whether the board is a pure non-contact race.
 * Pure race occurs when neither player has checkers on the bar and
 * White's highest occupied point index is strictly less than Black's lowest occupied point index.
 */
export function isPureRacePosition(state: BoardState): boolean {
  if (state.bar.white > 0 || state.bar.black > 0) {
    return false;
  }

  let highestWhite = -1;
  let lowestBlack = 24;

  for (let i = 0; i < 24; i++) {
    if (state.points[i] > 0 && i > highestWhite) {
      highestWhite = i;
    }
    if (state.points[i] < 0 && i < lowestBlack) {
      lowestBlack = i;
    }
  }

  return highestWhite < lowestBlack;
}

/**
 * Computes positional and race metrics for `player` using player-relative coordinates:
 * `rel = 0..23` where `rel = 0` is `player`'s 1-point and `rel = 23` is `player`'s 24-point.
 * In relative coordinates, `player` moves 23 -> 0 and `opponent` moves 0 -> 23.
 */
function computePlayerMetrics(
  state: BoardState,
  player: Player,
  isPureRace: boolean
): PlayerPositionalMetrics {
  const isWhite = player === 'white';
  const opponent: Player = isWhite ? 'black' : 'white';

  const getRelCount = (rel: number): number => {
    const absIdx = isWhite ? rel : 23 - rel;
    const raw = state.points[absIdx];
    return isWhite ? raw : -raw;
  };

  const pips = getPipCount(state, player);
  const borneOff = state.borneOff[player];
  const checkersOnBoard = 15 - borneOff;

  // 1. Effective Pip Count (EPC) & Wastage calculation
  const c0 = Math.max(0, getRelCount(0));
  const c1 = Math.max(0, getRelCount(1));
  const c2 = Math.max(0, getRelCount(2));

  let highestOccupiedRel = -1;
  for (let rel = 23; rel >= 0; rel--) {
    if (getRelCount(rel) > 0) {
      highestOccupiedRel = rel;
      break;
    }
  }

  let wastage = 0;
  if (c0 > 2) wastage += (c0 - 2) * 1.5;
  if (c1 > 2) wastage += (c1 - 2) * 1.0;
  if (c2 > 2) wastage += (c2 - 2) * 0.6;

  for (let rel = 3; rel <= 5; rel++) {
    if (getRelCount(rel) <= 0 && highestOccupiedRel > rel) {
      wastage += 0.85;
    }
  }

  wastage += checkersOnBoard * 0.5;
  const effectivePips = Number((pips + wastage).toFixed(2));

  // 2. Home board strength (rel 0..5) & opponent home board made points (rel 18..23)
  let homePoints = 0;
  let homeStrengthScore = 0;
  let opponentHomeMadePoints = 0;

  for (let rel = 0; rel <= 5; rel++) {
    const c = getRelCount(rel);
    if (c >= 2) {
      homePoints++;
      if (rel === 4) homeStrengthScore += 0.135;
      else if (rel === 3) homeStrengthScore += 0.115;
      else if (rel === 5) homeStrengthScore += 0.105;
      else if (rel === 2) homeStrengthScore += 0.085;
      else homeStrengthScore += 0.06;
    }
  }

  for (let rel = 18; rel <= 23; rel++) {
    if (getRelCount(rel) <= -2) {
      opponentHomeMadePoints++;
    }
  }

  // 3. Prime length (longest consecutive streak of made points >= 2 across rel 0..23)
  let currentStreak = 0;
  let maxPrimeLength = 0;
  for (let rel = 0; rel < 24; rel++) {
    if (getRelCount(rel) >= 2) {
      currentStreak++;
      if (currentStreak > maxPrimeLength) {
        maxPrimeLength = currentStreak;
      }
    } else {
      currentStreak = 0;
    }
  }
  const cappedPrime = Math.min(6, maxPrimeLength);
  const primeScore = isPureRace ? 0 : PRIME_REWARDS[cappedPrime];

  // 4. Defensive anchors in opponent's home board (rel 18..23) & back-checker count
  let anchors = 0;
  let anchorScore = 0;
  let inOpponentHomeCount = 0;
  let furthestBackRel = -1;

  for (let rel = 18; rel <= 23; rel++) {
    const c = getRelCount(rel);
    if (c > 0) {
      inOpponentHomeCount += c;
      if (rel > furthestBackRel) {
        furthestBackRel = rel;
      }
    }
    if (c >= 2) {
      anchors++;
      if (!isPureRace) {
        if (rel === 19 || rel === 20) anchorScore += 0.115;
        else if (rel === 18) anchorScore += 0.095;
        else anchorScore += 0.06;
      }
    }
  }

  // 5. Back-Checker Escape & Prime Trapping evaluation
  let trappedCheckers = 0;
  let escapeRolls = 36;
  let trappedPenalty = 0;

  if (!isPureRace && inOpponentHomeCount > 0) {
    for (let rel = 18; rel <= 23; rel++) {
      const c = getRelCount(rel);
      if (c <= 0) continue;

      let blockedInFront = 0;
      let consecutiveBlockade = 0;
      let maxConsecutiveBlockade = 0;
      for (let step = 1; step <= 6; step++) {
        const frontRel = rel - step;
        if (frontRel >= 0 && getRelCount(frontRel) <= -2) {
          blockedInFront++;
          consecutiveBlockade++;
          if (consecutiveBlockade > maxConsecutiveBlockade) {
            maxConsecutiveBlockade = consecutiveBlockade;
          }
        } else {
          consecutiveBlockade = 0;
        }
      }

      if (maxConsecutiveBlockade >= 3 || blockedInFront >= 4) {
        trappedCheckers += c;
        const severity =
          maxConsecutiveBlockade >= 6
            ? 0.22
            : maxConsecutiveBlockade === 5
              ? 0.16
              : maxConsecutiveBlockade === 4
                ? 0.11
                : 0.065;
        trappedPenalty += c * severity;
      }
    }

    if (furthestBackRel >= 18) {
      let validEscapeCount = 0;
      for (let d1 = 1; d1 <= 6; d1++) {
        for (let d2 = 1; d2 <= 6; d2++) {
          const t1 = furthestBackRel - d1;
          const t2 = furthestBackRel - d2;
          const open1 = t1 >= 0 && getRelCount(t1) >= -1;
          const open2 = t2 >= 0 && getRelCount(t2) >= -1;

          let canEscapeOutfield = false;
          if (open1 && t1 <= 17) {
            canEscapeOutfield = true;
          } else if (open2 && t2 <= 17) {
            canEscapeOutfield = true;
          } else if (open1 || open2) {
            const combined = furthestBackRel - d1 - d2;
            if (combined >= 0 && combined <= 17 && getRelCount(combined) >= -1) {
              canEscapeOutfield = true;
            }
          }

          if (canEscapeOutfield) {
            validEscapeCount++;
          }
        }
      }
      escapeRolls = validEscapeCount;
      if (escapeRolls < 16) {
        trappedPenalty += ((16 - escapeRolls) / 36) * 0.14 * inOpponentHomeCount;
      }
    }
  }

  // 6. Builder Diversity & Outfield Control (relative indices 6..10)
  let buildersCount = 0;
  let builderScore = 0;
  if (!isPureRace) {
    for (let rel = 6; rel <= 10; rel++) {
      const c = getRelCount(rel);
      if (c >= 1) {
        buildersCount++;
        if (c === 2) {
          builderScore += rel === 6 ? 0.075 : 0.045;
        } else if (c === 3 || c === 4) {
          builderScore += rel === 6 ? 0.09 : 0.06;
        } else if (c === 1) {
          builderScore += 0.025;
        }
      }
    }
  }

  // 7. Stacking penalty (points with > 5 checkers)
  let stackPenalty = 0;
  for (let rel = 0; rel < 24; rel++) {
    const c = getRelCount(rel);
    if (c > 5) {
      stackPenalty += (c - 5) * 0.028;
    }
  }

  // 8. Blot exposure danger (single checkers === 1)
  let blotDanger = 0;
  if (!isPureRace) {
    const oppBar = state.bar[opponent];

    let lowestOppRel = oppBar > 0 ? -1 : 24;
    if (lowestOppRel === 24) {
      for (let rel = 0; rel < 24; rel++) {
        if (getRelCount(rel) < 0) {
          lowestOppRel = rel;
          break;
        }
      }
    }

    for (let rel = 0; rel < 24; rel++) {
      if (getRelCount(rel) !== 1) continue;
      if (lowestOppRel >= rel) continue;

      let hittingRolls = 0;

      for (let d1 = 1; d1 <= 6; d1++) {
        for (let d2 = 1; d2 <= 6; d2++) {
          let canHitOnRoll = false;

          if (oppBar > 0) {
            const entry1 = d1 - 1;
            const entry2 = d2 - 1;
            if (entry1 === rel || entry2 === rel) {
              canHitOnRoll = true;
            } else if (oppBar === 1) {
              if (entry1 + d2 === rel && getRelCount(entry1) <= 1) {
                canHitOnRoll = true;
              } else if (entry2 + d1 === rel && getRelCount(entry2) <= 1) {
                canHitOnRoll = true;
              }
            }
          } else {
            const from1 = rel - d1;
            const from2 = rel - d2;
            if (from1 >= 0 && getRelCount(from1) < 0) {
              canHitOnRoll = true;
            } else if (from2 >= 0 && getRelCount(from2) < 0) {
              canHitOnRoll = true;
            } else {
              const fromBoth = rel - d1 - d2;
              if (fromBoth >= 0 && getRelCount(fromBoth) < 0) {
                const mid1 = rel - d1;
                const mid2 = rel - d2;
                if (
                  (mid1 >= 0 && getRelCount(mid1) <= 1) ||
                  (mid2 >= 0 && getRelCount(mid2) <= 1)
                ) {
                  canHitOnRoll = true;
                }
              } else if (d1 === d2) {
                for (let mult = 3; mult <= 4; mult++) {
                  const fromMult = rel - d1 * mult;
                  if (fromMult >= 0 && getRelCount(fromMult) < 0) {
                    let pathOpen = true;
                    for (let step = 1; step < mult; step++) {
                      if (getRelCount(fromMult + d1 * step) >= 2) {
                        pathOpen = false;
                        break;
                      }
                    }
                    if (pathOpen) {
                      canHitOnRoll = true;
                      break;
                    }
                  }
                }
              }
            }
          }

          if (canHitOnRoll) {
            hittingRolls++;
          }
        }
      }

      const distanceLossFactor = 0.65 + ((24 - rel) / 24) * 0.85;
      const oppHomeClosureFactor = 1 + opponentHomeMadePoints * 0.1;
      const shotDanger =
        (hittingRolls / 36) * distanceLossFactor * oppHomeClosureFactor;
      const indirectDanger = 0.04 * distanceLossFactor;

      blotDanger += shotDanger + indirectDanger;
    }
  }

  // 9. Bar penalty
  const playerBarCount = state.bar[player];
  const barPenalty =
    playerBarCount * (0.28 + opponentHomeMadePoints * 0.068);

  return {
    pips,
    effectivePips,
    wastage,
    homePoints,
    homeStrengthScore,
    maxPrimeLength,
    primeScore,
    blotDanger,
    anchors,
    anchorScore,
    barCount: playerBarCount,
    barPenalty,
    stackPenalty,
    trappedCheckers,
    escapeRolls,
    trappedPenalty,
    buildersCount,
    builderScore,
    inOpponentHomeCount,
    borneOff,
  };
}

function computeGammonAndBackgammonRates(
  playerWinProb01: number,
  player: PlayerPositionalMetrics,
  opponent: PlayerPositionalMetrics,
  isPureRace: boolean
): { gammon01: number; backgammon01: number } {
  if (opponent.borneOff > 0 || playerWinProb01 <= 0.5) {
    return { gammon01: 0, backgammon01: 0 };
  }

  const epcLead = Math.max(0, opponent.effectivePips - player.effectivePips);

  let gammonPotential = 0;
  if (isPureRace) {
    gammonPotential = Math.max(0, (epcLead - 24) * 0.015);
  } else {
    gammonPotential =
      player.homePoints * 0.038 +
      opponent.barCount * 0.095 +
      opponent.trappedCheckers * 0.065 +
      epcLead * 0.005;
  }

  const winExcess = Math.max(0, playerWinProb01 - 0.5);
  const gammon01 = Math.min(
    playerWinProb01 * 0.78,
    winExcess * 1.45 * Math.min(1.1, gammonPotential)
  );

  const deepStragglers = opponent.inOpponentHomeCount + opponent.barCount;
  const backgammon01 =
    deepStragglers > 0
      ? Math.min(gammon01 * 0.35, gammon01 * (0.06 + deepStragglers * 0.045))
      : 0;

  return { gammon01, backgammon01 };
}

/**
 * Master Backgammon Position Evaluator:
 * - Priority 1: Terminal win detection (±1.0 / ±2.0 / ±3.0).
 * - Priority 2: Exact 54,264-state Mathematical Bear-Off Tablebase when both players are in their home boards.
 * - Priority 3: Trained Dual Neural Network (ContactNet / RaceNet) when loaded.
 * - Priority 4: Hybrid Multi-Factor Positional + Exact Bear-Off Expected Turns & Kleinman Race Evaluator.
 */
export function evaluatePosition(state: BoardState): EvalResult {
  const isPureRace = isPureRacePosition(state);
  const white = computePlayerMetrics(state, 'white', isPureRace);
  const black = computePlayerMetrics(state, 'black', isPureRace);

  const pipDiff = black.pips - white.pips;
  const rawEffectiveDiff = black.effectivePips - white.effectivePips;
  const effectivePipDiff =
    Math.abs(rawEffectiveDiff) < 1e-6
      ? 0
      : Number(rawEffectiveDiff.toFixed(1));
  const homePointsDiff = white.homePoints - black.homePoints;
  const primeLengthDiff = white.maxPrimeLength - black.maxPrimeLength;
  const rawBlotDiff = white.blotDanger - black.blotDanger;
  const blotDangerDiff =
    Math.abs(rawBlotDiff) < 1e-6 ? 0 : Number(rawBlotDiff.toFixed(3));
  const anchorsDiff = white.anchors - black.anchors;
  const barDiff = state.bar.black - state.bar.white;
  const trappedCheckersDiff = black.trappedCheckers - white.trappedCheckers;
  const escapeChancesDiff = white.escapeRolls - black.escapeRolls;
  const buildersDiff = white.buildersCount - black.buildersCount;

  const features: EvalFeatures = {
    pipDiff,
    effectivePipDiff,
    homePointsDiff,
    primeLengthDiff,
    blotDangerDiff,
    anchorsDiff,
    barDiff,
    trappedCheckersDiff,
    escapeChancesDiff,
    buildersDiff,
    isPureRace,
  };

  const effectivePips = {
    white: white.effectivePips,
    black: black.effectivePips,
  };

  // Priority 1: Terminal state check (White wins)
  if (state.borneOff.white >= 15) {
    const isGammon = state.borneOff.black === 0;
    const isBackgammon =
      isGammon &&
      (state.bar.black > 0 || state.points.slice(0, 6).some((p) => p < 0));
    const equity = isBackgammon ? 3.0 : isGammon ? 2.0 : 1.0;
    return {
      equity,
      winProb: 100,
      gammonProb: isGammon ? 100 : 0,
      backgammonProb: isBackgammon ? 100 : 0,
      effectivePips,
      features,
      details: { pipDiff, effectivePipDiff },
    };
  }

  // Priority 1: Terminal state check (Black wins)
  if (state.borneOff.black >= 15) {
    const isGammon = state.borneOff.white === 0;
    const isBackgammon =
      isGammon &&
      (state.bar.white > 0 || state.points.slice(18, 24).some((p) => p > 0));
    const equity = isBackgammon ? -3.0 : isGammon ? -2.0 : -1.0;
    return {
      equity,
      winProb: 0,
      gammonProb: isGammon ? 100 : 0,
      backgammonProb: isBackgammon ? 100 : 0,
      effectivePips,
      features,
      details: { pipDiff, effectivePipDiff },
    };
  }

  // Priority 2: Pure Bear-Off Tablebase (100% mathematical perfection)
  if (isPureBearOffPosition(state)) {
    const whiteHome = extractPlayerHomeBoard(state, 'white');
    const blackHome = extractPlayerHomeBoard(state, 'black');
    const exact = evaluateExactBearOff(whiteHome, blackHome);
    return {
      equity: exact.equity,
      winProb: exact.winProb,
      gammonProb: Number(
        Math.min(95, exact.gammonProb + exact.loseGammonProb).toFixed(1)
      ),
      backgammonProb: 0,
      effectivePips,
      features,
      details: { pipDiff, effectivePipDiff },
    };
  }

  // Compute Heuristic Tactical Anchor (Exact Bear-Off Expected Turns + Multi-Factor Positional & Kleinman Race Evaluator)
  let winProb01 = 0.5;
  let positionalAdvantage = 0;

  const exactTurnsWhite = getExactExpectedBearOffTurns(state, 'white');
  const exactTurnsBlack = getExactExpectedBearOffTurns(state, 'black');
  const exactTurnsDiff = exactTurnsBlack - exactTurnsWhite;

  if (isPureRace) {
    const sumEPC = Math.max(16, white.effectivePips + black.effectivePips);
    const kleinmanRatio = rawEffectiveDiff / Math.sqrt(sumEPC);
    const borneOffAdv = (white.borneOff - black.borneOff) * 0.045;
    const stackAdv = -(white.stackPenalty - black.stackPenalty) * 0.35;

    positionalAdvantage =
      kleinmanRatio * 0.52 + exactTurnsDiff * 0.16 + borneOffAdv + stackAdv;
    winProb01 =
      Math.abs(positionalAdvantage) < 1e-9
        ? 0.5
        : 1 / (1 + Math.exp(-positionalAdvantage * 2.15));
  } else {
    positionalAdvantage += rawEffectiveDiff * 0.014;
    positionalAdvantage += exactTurnsDiff * 0.035;
    positionalAdvantage += (white.borneOff - black.borneOff) * 0.075;
    positionalAdvantage += white.homeStrengthScore - black.homeStrengthScore;
    positionalAdvantage += white.primeScore - black.primeScore;
    positionalAdvantage -= (white.blotDanger - black.blotDanger) * 0.42;
    positionalAdvantage += white.anchorScore - black.anchorScore;
    positionalAdvantage -= white.barPenalty - black.barPenalty;
    positionalAdvantage -= white.stackPenalty - black.stackPenalty;
    positionalAdvantage -= white.trappedPenalty - black.trappedPenalty;
    positionalAdvantage += white.builderScore - black.builderScore;

    winProb01 =
      Math.abs(positionalAdvantage) < 1e-9
        ? 0.5
        : 1 / (1 + Math.exp(-positionalAdvantage * 1.28));
  }

  const blackWinProb01 = 1 - winProb01;

  const whiteGammonRates = computeGammonAndBackgammonRates(
    winProb01,
    white,
    black,
    isPureRace
  );
  const blackGammonRates = computeGammonAndBackgammonRates(
    blackWinProb01,
    black,
    white,
    isPureRace
  );

  const cubelessEquity =
    winProb01 -
    blackWinProb01 +
    (whiteGammonRates.gammon01 - blackGammonRates.gammon01) +
    (whiteGammonRates.backgammon01 - blackGammonRates.backgammon01) +
    positionalAdvantage * 0.25;

  const clampedEquity = Math.max(-3.0, Math.min(3.0, cubelessEquity));
  const heurEquity =
    Math.abs(clampedEquity) < 1e-6 ? 0 : Number(clampedEquity.toFixed(2));

  const heurWinProb =
    heurEquity === 0
      ? 50
      : Number(Math.max(0.5, Math.min(99.5, winProb01 * 100)).toFixed(1));

  const totalGammonPct =
    (whiteGammonRates.gammon01 + blackGammonRates.gammon01) * 100;
  const totalBackgammonPct =
    (whiteGammonRates.backgammon01 + blackGammonRates.backgammon01) * 100;

  const heurGammonProb =
    totalGammonPct < 0.05 ? 0 : Number(Math.min(95, totalGammonPct).toFixed(1));
  const heurBackgammonProb =
    totalBackgammonPct < 0.05
      ? 0
      : Number(Math.min(45, totalBackgammonPct).toFixed(1));

  // Priority 3: 90% Trained v3 3-Network Residual Neural Network + 10% Tactical Feature Anchor
  const nn = getActiveNeuralNet();
  if (nn && nn.metadata.trainedGames > 0) {
    const gamePhase = classifyGamePhase(state);
    const nnEval = nn.evaluateSymmetric(state, gamePhase);
    const blendedEqRaw = Math.max(
      -3.0,
      Math.min(3.0, 0.9 * nnEval.equity + 0.1 * clampedEquity)
    );
    const blendedEquity =
      Math.abs(blendedEqRaw) < 1e-6 ? 0 : Number(blendedEqRaw.toFixed(2));
    const blendedWinProb = Number(
      Math.max(
        0.5,
        Math.min(99.5, 0.9 * nnEval.winProb + 0.1 * heurWinProb)
      ).toFixed(1)
    );
    const blendedGammonProb = Number(
      Math.max(
        0,
        Math.min(95, 0.9 * nnEval.gammonProb + 0.1 * heurGammonProb)
      ).toFixed(1)
    );
    const blendedBackgammonProb = Number(
      Math.max(
        0,
        Math.min(45, 0.9 * nnEval.backgammonProb + 0.1 * heurBackgammonProb)
      ).toFixed(1)
    );

    return {
      equity: blendedEquity,
      winProb: blendedWinProb,
      gammonProb: blendedGammonProb,
      backgammonProb: blendedBackgammonProb,
      effectivePips,
      features,
      details: { pipDiff, effectivePipDiff },
    };
  }

  return {
    equity: heurEquity,
    winProb: heurWinProb,
    gammonProb: heurGammonProb,
    backgammonProb: heurBackgammonProb,
    effectivePips,
    features,
    details: {
      pipDiff,
      effectivePipDiff,
    },
  };
}

/**
 * Alias for backward compatibility with Step 2 imports.
 */
export const evaluateBaseline = evaluatePosition;
