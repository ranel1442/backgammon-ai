import React, { useMemo, useRef, useState } from 'react';
import {
  Award,
  BarChart3,
  Download,
  FastForward,
  Filter,
  Flame,
  FlaskConical,
  Upload,
  X,
} from 'lucide-react';
import { getClassificationMeta, type MoveClassification } from '../engine/ai';
import type { GameReviewSummary, KeyMoment } from '../engine/review';
import { EvalGraph } from './EvalGraph';

interface GameReviewPanelProps {
  review: GameReviewSummary;
  selectedTurnIndex: number | null;
  onSelectTurnIndex: (turnIndex: number) => void;
  onEnterSandboxAtTurn: (turnIndex: number) => void;
  onExportMatch: () => void;
  onImportMatch: (file: File) => void;
  onClose: () => void;
  pendingAnalysisCount?: number;
}

type MistakeFilter = 'all' | 'white';

const CLASSIFICATION_ORDER: MoveClassification[] = [
  'Best',
  'Good',
  'Inaccuracy',
  'Mistake',
  'Blunder',
];

export const GameReviewPanel: React.FC<GameReviewPanelProps> = ({
  review,
  selectedTurnIndex,
  onSelectTurnIndex,
  onEnterSandboxAtTurn,
  onExportMatch,
  onImportMatch,
  onClose,
  pendingAnalysisCount = 0,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [mistakeFilter, setMistakeFilter] = useState<MistakeFilter>('all');

  const filteredKeyMoments = useMemo(() => {
    if (mistakeFilter === 'white') {
      return review.keyMoments.filter((km) => km.player === 'white');
    }
    return review.keyMoments;
  }, [review.keyMoments, mistakeFilter]);

  const handleNextMistake = () => {
    if (filteredKeyMoments.length === 0) return;

    if (selectedTurnIndex === null) {
      onSelectTurnIndex(filteredKeyMoments[0].turnIndex);
      return;
    }

    const next = filteredKeyMoments.find(
      (km) => km.turnIndex > selectedTurnIndex
    );
    if (next) {
      onSelectTurnIndex(next.turnIndex);
    } else {
      onSelectTurnIndex(filteredKeyMoments[0].turnIndex);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onImportMatch(file);
      e.target.value = '';
    }
  };

  return (
    <section
      dir="rtl"
      className="rounded-2xl bg-slate-900/95 border border-slate-700/80 shadow-2xl overflow-hidden"
    >
      {/* Top Header Bar */}
      <div className="px-4 py-3 border-b border-slate-800 bg-slate-950/80 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <BarChart3 className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              סיכום וניתוח משחק
              {pendingAnalysisCount > 0 && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 animate-pulse">
                  מנתח {pendingAnalysisCount} תורות...
                </span>
              )}
            </h2>
            <p className="text-[11px] text-slate-400">
              מדד דיוק דו-שלבי (2-Ply) • גרף יתרון ורגעי מפתח במשחק
            </p>
          </div>
        </div>

        {/* Action Buttons & Mistake Filter */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Mistake Filter Toggle: כל הטעויות vs הטעויות שלי בלבד */}
          <div className="inline-flex items-center rounded-xl bg-slate-900 p-0.5 border border-slate-800">
            <Filter className="w-3 h-3 text-slate-400 mx-1.5" />
            <button
              type="button"
              onClick={() => setMistakeFilter('all')}
              className={`px-2 py-1 rounded-lg text-[11px] font-semibold transition-all cursor-pointer ${
                mistakeFilter === 'all'
                  ? 'bg-slate-700 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              כל הטעויות
            </button>
            <button
              type="button"
              onClick={() => setMistakeFilter('white')}
              className={`px-2 py-1 rounded-lg text-[11px] font-semibold transition-all cursor-pointer ${
                mistakeFilter === 'white'
                  ? 'bg-emerald-500/25 text-emerald-200 border border-emerald-500/40 shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              הטעויות שלי בלבד
            </button>
          </div>

          <button
            type="button"
            onClick={handleNextMistake}
            disabled={filteredKeyMoments.length === 0}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
              filteredKeyMoments.length > 0
                ? 'bg-amber-400 hover:bg-amber-300 text-slate-950 shadow cursor-pointer'
                : 'bg-slate-800/60 text-slate-500 border border-slate-800 cursor-not-allowed'
            }`}
          >
            <FastForward className="w-3.5 h-3.5" />
            הטעות הבאה ({filteredKeyMoments.length})
          </button>

          {selectedTurnIndex !== null && (
            <button
              type="button"
              onClick={() => onEnterSandboxAtTurn(selectedTurnIndex)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-200 border border-cyan-500/40 text-xs font-semibold transition-colors cursor-pointer"
            >
              <FlaskConical className="w-3.5 h-3.5" />
              בדוק חלופה (מצב חקירה)
            </button>
          )}

          <button
            type="button"
            onClick={onExportMatch}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium transition-colors cursor-pointer"
            title="ייצא משחק (.json)"
          >
            <Download className="w-3.5 h-3.5" />
            ייצא משחק (.json)
          </button>

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium transition-colors cursor-pointer"
            title="טען משחק (.json)"
          >
            <Upload className="w-3.5 h-3.5" />
            טען משחק (.json)
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,application/json"
            onChange={handleFileChange}
            className="hidden"
          />

          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors cursor-pointer"
            title="סגור ניתוח משחק"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Body Grid */}
      <div className="p-4 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* White Stats Card */}
          <div className="rounded-xl bg-slate-950/70 border border-slate-800 p-3.5 flex flex-col justify-between gap-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <span className="w-5 h-5 rounded-full bg-stone-100 border-2 border-stone-400 shadow" />
                <div>
                  <div className="text-xs font-bold text-white">
                    לבן (שחקן 1)
                  </div>
                  <div className="text-[11px] text-slate-400">
                    איבוד Equity ממוצע:{' '}
                    <span dir="ltr" className="inline-block font-mono">
                      -{review.white.avgEquityLoss.toFixed(3)}
                    </span>
                  </div>
                </div>
              </div>

              <div className="text-left">
                <div
                  dir="ltr"
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-stone-100 text-slate-950 font-mono text-sm font-extrabold shadow"
                >
                  <Award className="w-3.5 h-3.5 text-emerald-600" />
                  {review.white.accuracy.toFixed(1)}%
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5 text-center">
                  מדד דיוק
                </div>
              </div>
            </div>

            <div className="grid grid-cols-5 gap-1.5 pt-2 border-t border-slate-800/80">
              {CLASSIFICATION_ORDER.map((cls) => {
                const meta = getClassificationMeta(cls);
                return (
                  <div
                    key={cls}
                    className="rounded-lg bg-slate-900/90 border border-slate-800/80 p-1.5 text-center"
                  >
                    <div
                      dir="ltr"
                      className={`inline-block text-xs font-mono font-bold ${meta.textClass}`}
                    >
                      {review.white.counts[cls]}
                    </div>
                    <div className="text-[9px] text-slate-400 truncate">
                      {meta.label}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Black Stats Card */}
          <div className="rounded-xl bg-slate-950/70 border border-slate-800 p-3.5 flex flex-col justify-between gap-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <span className="w-5 h-5 rounded-full bg-slate-900 border-2 border-slate-500 shadow" />
                <div>
                  <div className="text-xs font-bold text-slate-100">
                    שחור (שחקן 2 / AI)
                  </div>
                  <div className="text-[11px] text-slate-400">
                    איבוד Equity ממוצע:{' '}
                    <span dir="ltr" className="inline-block font-mono">
                      -{review.black.avgEquityLoss.toFixed(3)}
                    </span>
                  </div>
                </div>
              </div>

              <div className="text-left">
                <div
                  dir="ltr"
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-600 text-stone-100 font-mono text-sm font-extrabold shadow"
                >
                  <Award className="w-3.5 h-3.5 text-emerald-400" />
                  {review.black.accuracy.toFixed(1)}%
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5 text-center">
                  מדד דיוק
                </div>
              </div>
            </div>

            <div className="grid grid-cols-5 gap-1.5 pt-2 border-t border-slate-800/80">
              {CLASSIFICATION_ORDER.map((cls) => {
                const meta = getClassificationMeta(cls);
                return (
                  <div
                    key={cls}
                    className="rounded-lg bg-slate-900/90 border border-slate-800/80 p-1.5 text-center"
                  >
                    <div
                      dir="ltr"
                      className={`inline-block text-xs font-mono font-bold ${meta.textClass}`}
                    >
                      {review.black.counts[cls]}
                    </div>
                    <div className="text-[9px] text-slate-400 truncate">
                      {meta.label}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Equity Timeline Graph */}
        <EvalGraph
          timeline={review.equityTimeline}
          selectedTurnIndex={selectedTurnIndex}
          onSelectTurnIndex={onSelectTurnIndex}
        />

        {/* Key Moments List */}
        {filteredKeyMoments.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
              <span className="flex items-center gap-1.5">
                <Flame className="w-3.5 h-3.5 text-amber-400" />
                רגעי מפתח ותפניות במשחק ({filteredKeyMoments.length})
              </span>
              <span className="text-[11px] text-slate-500">
                לחץ על תור כדי לבחון את העמדה על הלוח
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 max-h-40 overflow-y-auto pr-1">
              {filteredKeyMoments.map((km: KeyMoment) => {
                const isSelected = selectedTurnIndex === km.turnIndex;
                const meta = getClassificationMeta(km.classification);

                return (
                  <div
                    key={km.turnIndex}
                    onClick={() => onSelectTurnIndex(km.turnIndex)}
                    className={`p-2.5 rounded-xl border transition-all cursor-pointer flex flex-col justify-between gap-1.5 ${
                      isSelected
                        ? 'bg-amber-500/15 border-amber-400/60 shadow'
                        : 'bg-slate-950/60 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-bold text-slate-200">
                        תור #{km.turnNumber} (
                        {km.player === 'white' ? 'לבן' : 'שחור'}{' '}
                        <span dir="ltr" className="inline-block font-mono">
                          {km.dice[0]}-{km.dice[1]}
                        </span>
                        )
                      </span>
                      <div className="flex items-center gap-1">
                        {km.isBiggestSwing && (
                          <span className="px-1.5 py-0.5 rounded bg-amber-400/20 text-amber-300 text-[9px] font-bold">
                            תפנית מרכזית
                          </span>
                        )}
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-semibold border ${meta.bgClass} ${meta.textClass} ${meta.borderClass}`}
                        >
                          {meta.label}{' '}
                          <span dir="ltr" className="inline-block font-mono">
                            (-{km.equityLoss.toFixed(2)})
                          </span>
                        </span>
                      </div>
                    </div>

                    <div className="text-[11px] text-slate-400 flex items-center justify-between gap-2">
                      <span className="truncate">
                        שוחק:{' '}
                        <span dir="ltr" className="inline-block font-mono">
                          {km.playedNotation}
                        </span>
                      </span>
                      <span className="text-emerald-400 shrink-0">
                        מומלץ:{' '}
                        <span dir="ltr" className="inline-block font-mono">
                          {km.bestNotation}
                        </span>
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </section>
  );
};
