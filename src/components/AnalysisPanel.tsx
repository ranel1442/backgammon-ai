import React, { useState } from 'react';
import {
  Activity,
  Brain,
  CheckCircle2,
  Compass,
  Cpu,
  Dices,
  Eye,
  Lightbulb,
  Loader2,
  Navigation,
  Scale,
  ShieldAlert,
  Sparkles,
  Target,
} from 'lucide-react';
import {
  getClassificationMeta,
  type CandidateMoveAnalysis,
  type TurnAnalysisResult,
} from '../engine/ai';
import { getEngineModelInfo } from '../engine/evaluator';
import type { RolloutMoveResult } from '../engine/rollout';
import type { MoveStep, Player } from '../engine/types';

interface AnalysisPanelProps {
  analysis: TurnAnalysisResult | null;
  isThinking: boolean;
  thinkingElapsedMs?: number;
  activePlayer: Player;
  dice: [number, number] | null;
  isViewingHistory: boolean;
  isHistoryTurnAnalyzing?: boolean;
  historyTurnNumber?: number;
  isOpponentTurnOrAnimating?: boolean;
  hasGenuineNoLegalMoves?: boolean;
  showLiveHint: boolean;
  onToggleLiveHint: () => void;
  onHoverMoveSteps: (steps: MoveStep[] | null) => void;
  onApplyRecommendedMove?: (steps: MoveStep[]) => void;
  onRunRollout?: () => void;
  isRollingOut?: boolean;
  rolloutProgress?: number;
  rolloutResults?: RolloutMoveResult[] | null;
}

function formatSignedEquity(eq: number): string {
  if (eq === 0) return '0.00';
  return eq > 0 ? `+${eq.toFixed(2)}` : eq.toFixed(2);
}

export const AnalysisPanel: React.FC<AnalysisPanelProps> = ({
  analysis,
  isThinking,
  thinkingElapsedMs = 0,
  activePlayer,
  dice,
  isViewingHistory,
  isHistoryTurnAnalyzing = false,
  historyTurnNumber,
  isOpponentTurnOrAnimating = false,
  hasGenuineNoLegalMoves = false,
  showLiveHint,
  onToggleLiveHint,
  onHoverMoveSteps,
  onApplyRecommendedMove,
  onRunRollout,
  isRollingOut = false,
  rolloutProgress = 0,
  rolloutResults = null,
}) => {
  const [pinnedRankIndex, setPinnedRankIndex] = useState<number | null>(null);
  const shouldShowLines = isViewingHistory || showLiveHint;
  const playerHebrew = activePlayer === 'white' ? 'לבן' : 'שחור';
  const modelInfo = getEngineModelInfo(
    analysis?.topMoves[0]?.move.resultingState
  );

  const handleSelectCandidateLine = (idx: number, steps: MoveStep[]) => {
    if (pinnedRankIndex === idx) {
      setPinnedRankIndex(null);
      onHoverMoveSteps(null);
    } else {
      setPinnedRankIndex(idx);
      onHoverMoveSteps(steps);
    }
  };

  return (
    <section
      dir="rtl"
      className="rounded-2xl bg-slate-900/90 border border-slate-800 shadow-2xl overflow-hidden"
    >
      {/* Header Bar */}
      <div className="px-4 py-2.5 border-b border-slate-800 bg-slate-950/70 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          <Brain className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs sm:text-sm font-bold text-slate-100 tracking-tight">
            {isViewingHistory
              ? `ניתוח רב-אמן AI — תור #${historyTurnNumber} (הקווים המומלצים ביותר)`
              : 'ניתוח רב-אמן AI — הקווים המומלצים ביותר'}
          </h3>
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-cyan-500/15 text-cyan-200 border border-cyan-500/35 text-[10px] font-semibold">
            <Cpu className="w-3 h-3 text-cyan-400" />
            מנוע רב-אמן AI (עומק 3+)
          </span>
          <span
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold border ${
              modelInfo.hasNeuralNet
                ? 'bg-purple-500/15 text-purple-300 border-purple-500/35'
                : 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
            }`}
          >
            <Activity className="w-3 h-3" />
            {modelInfo.labelHe}
          </span>
          {isThinking ? (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-cyan-500/15 text-cyan-200 border border-cyan-500/40 text-[10px] font-mono font-bold">
              <Loader2 className="w-3 h-3 animate-spin text-cyan-400" />
              <span>
                מנתח עמדה (עומק 3+
                {thinkingElapsedMs >= 100
                  ? ` • ${(thinkingElapsedMs / 1000).toFixed(1)}s`
                  : ''}
                )...
              </span>
            </span>
          ) : (
            analysis?.elapsedMs !== undefined &&
            analysis.elapsedMs >= 50 && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-800/90 text-cyan-300 border border-slate-700 text-[10px] font-mono">
                עומק 3+ ({(analysis.elapsedMs / 1000).toFixed(1)}s)
              </span>
            )
          )}
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {shouldShowLines &&
            analysis &&
            analysis.topMoves.length > 0 &&
            onRunRollout && (
              <button
                type="button"
                onClick={onRunRollout}
                disabled={isRollingOut}
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-bold border transition-all ${
                  isRollingOut
                    ? 'bg-amber-500/20 text-amber-200 border-amber-500/50 cursor-wait'
                    : 'bg-gradient-to-r from-amber-500/20 to-emerald-500/20 hover:from-amber-500/30 hover:to-emerald-500/30 text-amber-200 border-amber-500/45 shadow-sm cursor-pointer'
                }`}
              >
                {isRollingOut ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-amber-300" />
                    <span>מריץ Rollout ({rolloutProgress}%)...</span>
                  </>
                ) : (
                  <>
                    <Dices className="w-3.5 h-3.5 text-amber-300" />
                    <span>הרץ Rollout מעמיק (1,000 סימולציות)</span>
                  </>
                )}
              </button>
            )}

          {shouldShowLines && analysis && analysis.topMoves.length > 0 && (
            <span className="hidden xl:inline-flex items-center gap-1 text-[11px] text-emerald-300/90 bg-emerald-500/10 border border-emerald-500/30 px-2.5 py-0.5 rounded-lg">
              <Navigation className="w-3 h-3" />
              רחף או לחץ על הסבר להצגת חצים על הלוח
            </span>
          )}

          {!isViewingHistory && (
            <button
              type="button"
              onClick={onToggleLiveHint}
              className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold border transition-all cursor-pointer ${
                showLiveHint
                  ? 'bg-cyan-500/20 text-cyan-200 border-cyan-500/50 shadow'
                  : 'bg-slate-800/80 text-slate-400 border-slate-700 hover:text-slate-200'
              }`}
            >
              <Lightbulb className="w-3.5 h-3.5" />
              {showLiveHint ? 'רמז חי: פעיל' : 'רמז חי: כבוי'}
            </button>
          )}
        </div>
      </div>

      {/* Live Monte Carlo Rollout Progress Bar */}
      {(isRollingOut || (rolloutResults && rolloutResults.length > 0)) && (
        <div className="px-4 py-1.5 bg-slate-950/90 border-b border-slate-800/80 flex items-center justify-between gap-3 text-[11px]">
          <div className="flex items-center gap-2 text-amber-300 font-semibold">
            <Dices className="w-3.5 h-3.5" />
            <span>
              {isRollingOut
                ? 'מבצע סימולציות מונטה-קרלו עם הפחתת שונות (CRN) וטבלת סיומים מדויקת...'
                : `Rollout הושלם (${rolloutResults?.[0]?.trials ?? 1000} סימולציות לכל מועמד, רווח בר-סמך 95%)`}
            </span>
          </div>
          <div className="flex items-center gap-2 w-40 sm:w-56">
            <div className="flex-1 h-1.5 rounded-full bg-slate-800 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-amber-400 to-emerald-400 transition-all duration-200"
                style={{
                  width: `${isRollingOut ? rolloutProgress : 100}%`,
                }}
              />
            </div>
            <span dir="ltr" className="font-mono text-[10px] text-slate-300">
              {isRollingOut ? `${rolloutProgress}%` : '100%'}
            </span>
          </div>
        </div>
      )}

      {/* Body Content */}
      <div className="p-3 sm:p-4">
        {!shouldShowLines ? (
          <div className="flex items-center justify-between gap-4 py-2 px-3 rounded-xl bg-slate-950/50 border border-slate-800/80 text-xs text-slate-400">
            <div className="flex items-center gap-2">
              <Eye className="w-4 h-4 text-slate-500 shrink-0" />
              <span>
                המלצות המנוע והחצים הטקטיים מוסתרים במהלך התור. הפעל את כפתור{' '}
                <strong className="text-slate-200">רמז חי</strong> למעלה או לחץ
                על תור בהיסטוריית המהלכים כדי לצפות ב-3 הקווים המובילים.
              </span>
            </div>
          </div>
        ) : !isViewingHistory && isOpponentTurnOrAnimating ? (
          <div className="flex items-center gap-2.5 py-3 px-3.5 rounded-xl bg-slate-950/55 border border-slate-800/80 text-xs text-slate-300">
            {isThinking ? (
              <Loader2 className="w-4 h-4 animate-spin text-cyan-400 shrink-0" />
            ) : (
              <Cpu className="w-4 h-4 text-sky-400 shrink-0" />
            )}
            <span>
              {isThinking
                ? 'היריב (שחור) מחשב מהלך בעומק 3+ — ממתין לסיום תור היריב...'
                : 'ממתין לסיום תור היריב...'}
            </span>
          </div>
        ) : !dice ? (
          <div className="flex items-center gap-2.5 py-3 px-3 rounded-xl bg-slate-950/40 border border-slate-800/70 text-xs text-slate-400">
            <Cpu className="w-4 h-4 text-cyan-400/70 shrink-0" />
            <span>
              הטל קוביות כדי להפעיל ניתוח עומק רב-שלבי (Expectiminimax + 3 רשתות
              ResNet ייעודיות + טבלת סיומים מדויקת) ולהציג חצים טקטיים על הלוח.
            </span>
          </div>
        ) : !isOpponentTurnOrAnimating && hasGenuineNoLegalMoves ? (
          <div className="py-3 px-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-xs text-rose-200">
            אין מהלכים חוקיים עבור הטלת הקוביות{' '}
            <span dir="ltr" className="inline-block font-mono font-bold">
              {dice[0]}-{dice[1]}
            </span>
            .
          </div>
        ) : !analysis ||
          analysis.topMoves.length === 0 ||
          ((isThinking || (isViewingHistory && isHistoryTurnAnalyzing)) &&
            !analysis) ? (
          <div className="flex items-center justify-center gap-2 py-6 text-xs font-mono text-cyan-300">
            <Loader2 className="w-4 h-4 animate-spin" />
            {isViewingHistory
              ? 'משלים ניתוח רב-אמן (עומק 3+) לתור זה ברקע...'
              : 'מנתח עמדה בעומק 3+ על פני 21 תוצאות קובייה ו-36 צירופי הסתברות...'}
          </div>
        ) : (
          <div className="space-y-3">
            {/* Strategic Phase & Grandmaster Plan Banner + Head-to-Head Comparison */}
            {(analysis.gamePhaseLabel || analysis.runnerUpComparison) && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-2.5">
                {analysis.gamePhaseLabel && (
                  <div className="px-3 py-2 rounded-xl bg-slate-950/75 border border-slate-800 flex items-start gap-2.5">
                    <Compass className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                    <div className="text-xs leading-snug">
                      <div className="font-bold text-cyan-300">
                        {analysis.gamePhaseLabel}
                      </div>
                      {analysis.strategicPlan && (
                        <div className="text-slate-300 text-[11px] mt-0.5">
                          {analysis.strategicPlan}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {analysis.runnerUpComparison && (
                  <div className="px-3 py-2 rounded-xl bg-emerald-950/25 border border-emerald-500/35 flex items-start gap-2.5">
                    <Scale className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                    <div className="text-xs leading-snug">
                      <div className="font-bold text-emerald-300">
                        למה הקו המוביל (#1) עדיף על החלופה הבאה (#2)?
                      </div>
                      <div className="text-slate-200 text-[11px] mt-0.5">
                        {analysis.runnerUpComparison}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* If inspecting history and playedMoveAnalysis is present, show Coach Summary Banner */}
            {isViewingHistory && analysis.playedMoveAnalysis && (
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="flex items-center gap-2.5 flex-wrap">
                  <span className="text-xs text-slate-400 font-medium">
                    המהלך ששוחק:
                  </span>
                  <span
                    dir="ltr"
                    className="inline-block font-mono text-xs sm:text-sm font-bold text-white bg-slate-800 px-2 py-0.5 rounded"
                  >
                    {dice[0]}-{dice[1]}: {analysis.playedMoveAnalysis.notation}
                  </span>
                  {(() => {
                    const meta = getClassificationMeta(
                      analysis.playedMoveAnalysis.classification
                    );
                    return (
                      <span
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-semibold border ${meta.bgClass} ${meta.textClass} ${meta.borderClass}`}
                      >
                        <CheckCircle2 className="w-3 h-3" />
                        {meta.label}
                      </span>
                    );
                  })()}
                  {analysis.playedMoveAnalysis.equityLoss > 0 && (
                    <span className="text-xs text-rose-400 font-semibold">
                      (איבוד Equity:{' '}
                      <span dir="ltr" className="inline-block font-mono">
                        -{analysis.playedMoveAnalysis.equityLoss.toFixed(2)}
                      </span>
                      )
                    </span>
                  )}
                </div>

                <div className="text-xs text-slate-300">
                  {analysis.playedMoveAnalysis.explanation[0]}
                </div>
              </div>
            )}

            {/* Top 3 Recommended Candidate Lines Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
              {analysis.topMoves.map(
                (cand: CandidateMoveAnalysis, index: number) => {
                  const isRank1 = index === 0;
                  const isPinned = pinnedRankIndex === index;
                  const rolloutStat = rolloutResults?.[index];
                  const comb = cand.combinatorics;
                  const diffLabel =
                    cand.equityLoss === 0
                      ? 'הקו המוביל'
                      : `-${cand.equityLoss.toFixed(2)}`;

                  return (
                    <div
                      key={index}
                      onClick={() =>
                        handleSelectCandidateLine(index, cand.move.steps)
                      }
                      onMouseEnter={() => onHoverMoveSteps(cand.move.steps)}
                      onMouseLeave={() => {
                        if (
                          pinnedRankIndex !== null &&
                          analysis.topMoves[pinnedRankIndex]
                        ) {
                          onHoverMoveSteps(
                            analysis.topMoves[pinnedRankIndex].move.steps
                          );
                        } else {
                          onHoverMoveSteps(null);
                        }
                      }}
                      className={`group relative rounded-xl p-3 border transition-all flex flex-col justify-between cursor-pointer ${
                        isPinned
                          ? 'bg-cyan-950/35 border-cyan-400 ring-1 ring-cyan-400/50 shadow-lg'
                          : isRank1
                            ? 'bg-emerald-950/20 border-emerald-500/40 hover:border-emerald-400 hover:bg-emerald-950/35'
                            : 'bg-slate-950/60 border-slate-800 hover:border-cyan-500/50 hover:bg-slate-900/90'
                      }`}
                    >
                      <div>
                        {/* Top Row: Rank Badge + Notation + Equity & Loss */}
                        <div className="flex items-center justify-between gap-2 mb-2">
                          <div className="flex items-center gap-1.5 min-w-0">
                            <span
                              dir="ltr"
                              className={`inline-block w-5 h-5 rounded-md text-center leading-5 font-mono text-[11px] font-bold shrink-0 ${
                                isRank1
                                  ? 'bg-emerald-400 text-slate-950'
                                  : 'bg-slate-800 text-slate-300 border border-slate-700'
                              }`}
                            >
                              #{index + 1}
                            </span>
                            <span
                              dir="ltr"
                              className="inline-block font-mono text-xs sm:text-sm font-bold text-slate-100 truncate"
                            >
                              {cand.notation}
                            </span>
                            {cand.isBookMove && (
                              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-amber-400/15 text-amber-300 border border-amber-400/40 text-[10px] font-bold shrink-0">
                                {cand.bookBadgeLabel ?? 'ספר פתיחות'}
                              </span>
                            )}
                          </div>

                          <div className="flex items-center gap-1.5 shrink-0">
                            <span
                              dir="ltr"
                              className={`inline-block font-mono text-xs font-bold px-1.5 py-0.5 rounded ${
                                cand.equity > 0
                                  ? 'bg-stone-100 text-slate-950'
                                  : cand.equity < 0
                                    ? 'bg-slate-800 text-stone-100 border border-slate-700'
                                    : 'bg-slate-800 text-slate-300'
                              }`}
                            >
                              {formatSignedEquity(cand.equity)}
                            </span>
                            <span
                              dir={cand.equityLoss === 0 ? 'rtl' : 'ltr'}
                              className={`inline-block font-mono text-[10px] font-semibold px-1.5 py-0.5 rounded ${
                                isRank1
                                  ? 'bg-emerald-500/20 text-emerald-300'
                                  : 'bg-rose-500/15 text-rose-300'
                              }`}
                            >
                              {diffLabel}
                            </span>
                          </div>
                        </div>

                        {/* Exact 36-Roll Dice Combinatorics Pills */}
                        {comb && (
                          <div className="flex flex-wrap items-center gap-1.5 mb-2">
                            <span
                              className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold border ${
                                comb.opponentHitRolls === 0
                                  ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
                                  : comb.opponentHitRolls <= 6
                                    ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                                    : 'bg-rose-500/15 text-rose-300 border-rose-500/35'
                              }`}
                            >
                              <ShieldAlert className="w-2.5 h-2.5" />
                              סיכון פגיעה:{' '}
                              <span dir="ltr" className="font-mono font-bold">
                                {comb.opponentHitRolls}/36 ({comb.opponentHitPct}
                                %)
                              </span>
                            </span>

                            {comb.opponentBarBlockRolls > 0 && (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-cyan-500/15 text-cyan-300 border border-cyan-500/35">
                                חסימת בר:{' '}
                                <span dir="ltr" className="font-mono font-bold">
                                  {comb.opponentBarBlockRolls}/36 (
                                  {comb.opponentBarBlockPct}%)
                                </span>
                              </span>
                            )}
                          </div>
                        )}

                        {/* Monte Carlo Rollout Result Badge (when Rollout is active or completed) */}
                        {rolloutStat && (
                          <div className="mb-2 px-2 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-between gap-2 text-[10px]">
                            <span className="text-amber-300 font-semibold">
                              Rollout ({rolloutStat.trials}):
                            </span>
                            <div className="flex items-center gap-2">
                              <span
                                dir="ltr"
                                className="inline-block font-mono font-bold text-amber-200"
                              >
                                {formatSignedEquity(rolloutStat.equity)} ±{' '}
                                {rolloutStat.ci95.toFixed(2)}
                              </span>
                              <span
                                dir="ltr"
                                className="inline-block font-mono text-slate-300"
                              >
                                W:{rolloutStat.winProb.toFixed(1)}% G:
                                {rolloutStat.gammonProb.toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        )}

                        {/* Tactical Explanation Bullet Points in Hebrew (linked to board arrows) */}
                        <ul className="space-y-1 mt-1.5">
                          {cand.explanation.map((bullet, bIdx) => (
                            <li
                              key={bIdx}
                              onMouseEnter={() =>
                                onHoverMoveSteps(cand.move.steps)
                              }
                              className="text-[11px] text-slate-300 hover:text-white leading-snug flex items-start gap-1.5 rounded px-1 py-0.5 hover:bg-slate-800/50 transition-colors"
                            >
                              <Sparkles
                                className={`w-3 h-3 mt-0.5 shrink-0 ${
                                  isRank1
                                    ? 'text-emerald-400'
                                    : 'text-cyan-400/70'
                                }`}
                              />
                              <span>{bullet}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Footer: Hover hint + Quick Apply button when in live human turn */}
                      <div className="mt-2.5 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] text-slate-400">
                        <span>
                          סיכוי ניצחון ({playerHebrew}):{' '}
                          <span dir="ltr" className="inline-block font-mono">
                            {(rolloutStat
                              ? rolloutStat.winProb
                              : cand.winProb
                            ).toFixed(1)}
                            %
                          </span>
                          {cand.gammonProb > 0.5 && (
                            <span className="mr-1.5 text-amber-300/90">
                              • מארס:{' '}
                              <span dir="ltr" className="inline-block font-mono">
                                {cand.gammonProb.toFixed(1)}%
                              </span>
                            </span>
                          )}
                        </span>
                        {!isViewingHistory && onApplyRecommendedMove ? (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              onApplyRecommendedMove(cand.move.steps);
                            }}
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-200 border border-cyan-500/40 font-semibold transition-colors cursor-pointer"
                          >
                            <Target className="w-2.5 h-2.5" />
                            בצע מהלך
                          </button>
                        ) : (
                          <span
                            className={`inline-flex items-center gap-1 font-medium transition-opacity ${
                              isPinned || isRank1
                                ? 'text-emerald-400 opacity-100'
                                : 'text-cyan-400/80 opacity-0 group-hover:opacity-100'
                            }`}
                          >
                            <Navigation className="w-2.5 h-2.5" />
                            {isPinned
                              ? 'חצים נעוצים בלוח'
                              : isRank1
                                ? 'חצים ירוקים בלוח'
                                : 'הצג חצים בלוח'}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                }
              )}
            </div>
          </div>
        )}
      </div>
    </section>
  );
};
