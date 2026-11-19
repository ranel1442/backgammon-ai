import React from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  FlaskConical,
  History,
  Loader2,
  Play,
  Sparkles,
  ThumbsUp,
  XCircle,
} from 'lucide-react';
import {
  getClassificationMeta,
  type MoveClassification,
  type TurnAnalysisResult,
} from '../engine/ai';
import type { EvalResult } from '../engine/evaluator';
import type { BoardState, MoveStep, Player } from '../engine/types';

export type { MoveClassification };

export interface HistoryEntry {
  id: number;
  turnNumber: number;
  player: Player;
  dice: [number, number];
  steps: MoveStep[];
  notation: string;
  turnStartState: BoardState;
  resultingState: BoardState;
  evalResult: EvalResult;
  classification?: MoveClassification;
  equityLoss?: number;
  analysis?: TurnAnalysisResult;
}

interface MoveHistoryProps {
  history: HistoryEntry[];
  viewingIndex: number | null;
  onSelectHistoryIndex: (index: number | null) => void;
  onEnterSandboxAtTurn?: (index: number) => void;
}

const CLASSIFICATION_ICONS: Record<MoveClassification, React.ReactNode> = {
  Best: <CheckCircle2 className="w-3 h-3 text-emerald-400" />,
  Good: <ThumbsUp className="w-3 h-3 text-teal-400" />,
  Inaccuracy: <Sparkles className="w-3 h-3 text-sky-400" />,
  Mistake: <AlertTriangle className="w-3 h-3 text-amber-400" />,
  Blunder: <XCircle className="w-3 h-3 text-rose-400" />,
};

export const MoveHistory: React.FC<MoveHistoryProps> = ({
  history,
  viewingIndex,
  onSelectHistoryIndex,
  onEnterSandboxAtTurn,
}) => {
  const isViewingHistory = viewingIndex !== null;

  return (
    <aside
      dir="rtl"
      className="w-full lg:w-80 xl:w-96 flex flex-col rounded-2xl bg-slate-900/90 border border-slate-800 shadow-2xl overflow-hidden shrink-0 h-[560px] lg:h-auto"
    >
      {/* Sidebar Header */}
      <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between bg-slate-950/60">
        <div className="flex items-center gap-2">
          <History className="w-4 h-4 text-emerald-400" />
          <h2 className="text-sm font-semibold text-slate-100 tracking-tight">
            היסטוריית מהלכים
          </h2>
        </div>
        <span className="text-xs font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded-md">
          {history.length} {history.length === 1 ? 'תור' : 'תורות'}
        </span>
      </div>

      {/* Historical Inspection Banner */}
      {isViewingHistory && viewingIndex !== null && (
        <div className="px-3 py-2.5 bg-amber-500/15 border-b border-amber-500/30 flex flex-col gap-2">
          <div className="flex items-center justify-between gap-2">
            <div className="text-xs text-amber-200 font-medium">
              צופה בעמדה היסטורית (תור #{viewingIndex + 1})
            </div>
            <button
              type="button"
              onClick={() => onSelectHistoryIndex(null)}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-400 hover:bg-amber-300 text-slate-950 font-semibold text-xs transition-colors shadow cursor-pointer"
            >
              <Play className="w-3 h-3 fill-current" />
              חזור למשחק החי
            </button>
          </div>
          {onEnterSandboxAtTurn && (
            <button
              type="button"
              onClick={() => onEnterSandboxAtTurn(viewingIndex)}
              className="w-full inline-flex items-center justify-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-200 border border-cyan-500/40 font-semibold text-xs transition-colors cursor-pointer"
            >
              <FlaskConical className="w-3.5 h-3.5" />
              בדוק חלופה (מצב חקירה)
            </button>
          )}
        </div>
      )}

      {/* Move List */}
      <div className="flex-1 overflow-y-auto divide-y divide-slate-800/60 p-2 space-y-1">
        {history.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-500">
            <History className="w-8 h-8 mb-2 opacity-40" />
            <p className="text-xs font-medium text-slate-300">
              טרם בוצעו מהלכים — הטל קוביות כדי להתחיל
            </p>
            <p className="text-[11px] text-slate-500 mt-1">
              כל תור מנותח אוטומטית על ידי מנוע ה-AI להערכת דיוק ואיבוד Equity.
            </p>
          </div>
        ) : (
          history.map((entry, idx) => {
            const isSelected = viewingIndex === idx;
            const isWhite = entry.player === 'white';
            const meta = entry.classification
              ? getClassificationMeta(entry.classification)
              : null;

            const equityStr =
              entry.evalResult.equity === 0
                ? '0.00'
                : entry.evalResult.equity > 0
                  ? `+${entry.evalResult.equity.toFixed(2)}`
                  : entry.evalResult.equity.toFixed(2);

            const hasLoss =
              entry.equityLoss !== undefined && entry.equityLoss > 0;

            return (
              <button
                key={entry.id}
                type="button"
                onClick={() =>
                  onSelectHistoryIndex(isSelected ? null : idx)
                }
                className={`w-full text-right px-3 py-2 rounded-xl transition-all flex items-center justify-between gap-2 cursor-pointer border ${
                  isSelected
                    ? 'bg-emerald-500/15 border-emerald-500/50 shadow-md'
                    : 'bg-slate-900/40 hover:bg-slate-800/70 border-transparent'
                }`}
              >
                <div className="flex items-center gap-2 min-w-0">
                  {/* Turn Number & Player Disc */}
                  <span
                    dir="ltr"
                    className="inline-block text-xs font-mono text-slate-500 w-5 shrink-0"
                  >
                    {entry.turnNumber}.
                  </span>
                  <span
                    className={`w-3 h-3 rounded-full shrink-0 border ${
                      isWhite
                        ? 'bg-stone-100 border-stone-400'
                        : 'bg-slate-900 border-slate-500'
                    }`}
                    title={isWhite ? 'לבן' : 'שחור'}
                  />

                  {/* Standard Backgammon Notation */}
                  <span
                    dir="ltr"
                    className="inline-block text-xs font-mono font-medium text-slate-100 truncate"
                  >
                    {entry.notation}
                  </span>
                </div>

                {/* AI Classification Badge + Equity Loss + Eval */}
                <div className="flex items-center gap-1.5 shrink-0">
                  {meta && entry.classification ? (
                    <span
                      className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[10px] font-semibold border ${meta.bgClass} ${meta.textClass} ${meta.borderClass}`}
                    >
                      {CLASSIFICATION_ICONS[entry.classification]}
                      <span>{meta.label}</span>
                      {hasLoss && (
                        <span
                          dir="ltr"
                          className="inline-block font-mono opacity-90"
                        >
                          -{entry.equityLoss!.toFixed(2)}
                        </span>
                      )}
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[10px] font-mono text-slate-400 bg-slate-800/60 border border-slate-700/50">
                      <Loader2 className="w-2.5 h-2.5 animate-spin text-cyan-400" />
                      AI
                    </span>
                  )}

                  <span
                    dir="ltr"
                    className={`inline-block text-[11px] font-mono font-semibold w-10 text-left ${
                      entry.evalResult.equity > 0
                        ? 'text-stone-200'
                        : entry.evalResult.equity < 0
                          ? 'text-slate-400'
                          : 'text-slate-500'
                    }`}
                  >
                    {equityStr}
                  </span>
                </div>
              </button>
            );
          })
        )}
      </div>

      {/* Footer Legend for AI Badges */}
      <div className="px-3 py-2 border-t border-slate-800 bg-slate-950/60 flex items-center justify-between text-[10px] text-slate-400">
        <span>לחץ על תור לצפייה בניתוח</span>
        <div className="flex items-center gap-1.5">
          <span className="text-emerald-400">● מומלץ</span>
          <span className="text-teal-400">● טוב</span>
          <span className="text-amber-400">● טעות</span>
          <span className="text-rose-400">● טעות גסה</span>
        </div>
      </div>
    </aside>
  );
};
