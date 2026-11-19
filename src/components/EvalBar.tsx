import React from 'react';
import type { EvalResult } from '../engine/evaluator';

interface EvalBarProps {
  evalResult: EvalResult;
}

const SCALE_TICKS = [
  { label: '-2.0', topPercent: 16.7 },
  { label: '-1.0', topPercent: 33.3 },
  { label: '0.0', topPercent: 50.0 },
  { label: '+1.0', topPercent: 66.7 },
  { label: '+2.0', topPercent: 83.3 },
];

export const EvalBar: React.FC<EvalBarProps> = ({ evalResult }) => {
  const {
    equity,
    winProb,
    gammonProb,
    backgammonProb = 0,
    effectivePips,
    details,
  } = evalResult;

  const whiteHeightPercent = Math.max(6, Math.min(94, winProb));
  const blackWinProb = Number((100 - winProb).toFixed(1));
  const leaderWinProb = Math.max(winProb, blackWinProb);
  const singleWinProb = Number(
    Math.max(0, leaderWinProb - gammonProb).toFixed(1)
  );

  const formattedEquity =
    equity === 0
      ? '0.00'
      : equity > 0
        ? `+${equity.toFixed(2)}`
        : equity.toFixed(2);

  const shortStatus =
    equity > 0.05 ? 'לבן מוביל' : equity < -0.05 ? 'שחור מוביל' : 'שקול';

  const advantageTitle =
    equity > 0.05
      ? 'יתרון הלבן'
      : equity < -0.05
        ? 'יתרון השחור'
        : 'עמדה שקולה';

  const pipDiffLabel =
    details.pipDiff === 0
      ? 'מרוץ שקול (0)'
      : details.pipDiff > 0
        ? `לבן +${details.pipDiff}`
        : `שחור +${Math.abs(details.pipDiff)}`;

  const epcDiff = details.effectivePipDiff ?? details.pipDiff;
  const epcDiffLabel =
    epcDiff === 0
      ? 'שקול (0.0)'
      : epcDiff > 0
        ? `לבן +${epcDiff.toFixed(1)}`
        : `שחור +${Math.abs(epcDiff).toFixed(1)}`;

  return (
    <div
      dir="ltr"
      className="relative group flex flex-col w-14 sm:w-16 h-full min-h-[460px] rounded-2xl overflow-visible border-2 border-slate-700/90 bg-slate-950 shadow-2xl select-none shrink-0"
    >
      {/* Inner clipped bar fills */}
      <div className="relative flex-1 w-full h-full rounded-[14px] overflow-hidden flex flex-col">
        {/* Black Top Fill */}
        <div
          className="w-full bg-gradient-to-b from-slate-950 via-slate-900 to-slate-800 transition-all duration-500 ease-out"
          style={{ height: `${100 - whiteHeightPercent}%` }}
        />

        {/* White Bottom Fill */}
        <div
          className="w-full bg-gradient-to-t from-stone-100 via-stone-200 to-stone-300 transition-all duration-500 ease-out"
          style={{ height: `${whiteHeightPercent}%` }}
        />

        {/* Scale Tick Lines (-2.0, -1.0, 0.0, +1.0, +2.0) */}
        {SCALE_TICKS.map((tick) => {
          const isZero = tick.label === '0.0';
          return (
            <div
              key={tick.label}
              className="absolute inset-x-0 pointer-events-none flex items-center justify-between px-1 z-10"
              style={{ top: `${tick.topPercent}%` }}
            >
              <div
                className={`w-full ${
                  isZero
                    ? 'h-[2px] bg-emerald-500/75 shadow-[0_0_6px_rgba(16,185,129,0.7)]'
                    : 'h-[1px] bg-slate-500/35'
                }`}
              />
              <span
                dir="ltr"
                className={`absolute right-1 -translate-y-1/2 text-[8px] font-mono font-bold px-0.5 rounded ${
                  isZero
                    ? 'bg-emerald-950/90 text-emerald-300 border border-emerald-500/40'
                    : 'text-slate-400/70'
                }`}
              >
                {tick.label}
              </span>
            </div>
          );
        })}

        {/* Top Readout: Black Win % */}
        <div className="absolute top-2 inset-x-0 flex flex-col items-center z-20 pointer-events-none px-1">
          <span
            dir="rtl"
            className="text-[9px] font-bold text-slate-400 leading-none mb-0.5"
          >
            שחור
          </span>
          <span
            dir="ltr"
            className="inline-block text-[11px] sm:text-xs font-mono font-extrabold text-stone-100 drop-shadow"
          >
            {blackWinProb.toFixed(0)}%
          </span>
        </div>

        {/* Center / Dynamic Prominent Equity & Hebrew Status Pill */}
        <div className="absolute inset-x-1 top-1/2 -translate-y-1/2 flex flex-col items-center z-20 pointer-events-none">
          <div className="w-full rounded-lg bg-slate-950/90 border border-slate-700/90 py-1 px-1 shadow-lg flex flex-col items-center">
            <span
              dir="ltr"
              className={`inline-block text-[11px] sm:text-xs font-mono font-extrabold tracking-tight leading-tight ${
                equity > 0.05
                  ? 'text-emerald-400'
                  : equity < -0.05
                    ? 'text-sky-400'
                    : 'text-slate-200'
              }`}
            >
              {formattedEquity}
            </span>
            <span
              dir="rtl"
              className="text-[8px] sm:text-[9px] font-bold text-slate-300 leading-tight text-center mt-0.5"
            >
              {shortStatus}
            </span>
          </div>
        </div>

        {/* Bottom Readout: White Win % */}
        <div className="absolute bottom-2 inset-x-0 flex flex-col items-center z-20 pointer-events-none px-1">
          <span
            dir="ltr"
            className="inline-block text-[11px] sm:text-xs font-mono font-extrabold text-slate-950 drop-shadow-sm"
          >
            {winProb.toFixed(0)}%
          </span>
          <span
            dir="rtl"
            className="text-[9px] font-bold text-slate-800 leading-none mt-0.5"
          >
            לבן
          </span>
        </div>
      </div>

      {/* Hover Tooltip in Hebrew (RTL) */}
      <div
        dir="rtl"
        className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none absolute left-full ml-3 top-1/2 -translate-y-1/2 z-50 w-64 rounded-2xl bg-slate-900/95 backdrop-blur-md border border-slate-700 p-3.5 shadow-2xl text-xs text-slate-200 text-right"
      >
        <div className="flex items-center justify-between border-b border-slate-800 pb-2 mb-2">
          <span className="text-slate-100 font-bold text-sm">
            {advantageTitle}
          </span>
          <span
            dir="ltr"
            className={`inline-block font-mono font-extrabold text-sm px-1.5 py-0.5 rounded bg-slate-950 border ${
              equity > 0
                ? 'text-emerald-400 border-emerald-500/40'
                : equity < 0
                  ? 'text-rose-400 border-rose-500/40'
                  : 'text-slate-300 border-slate-700'
            }`}
          >
            {formattedEquity}
          </span>
        </div>

        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <span className="text-stone-300">סיכוי ניצחון (לבן)</span>
            <span
              dir="ltr"
              className="inline-block font-mono font-bold text-stone-100"
            >
              {winProb.toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-slate-400">סיכוי ניצחון (שחור)</span>
            <span
              dir="ltr"
              className="inline-block font-mono font-bold text-slate-300"
            >
              {blackWinProb.toFixed(1)}%
            </span>
          </div>

          <div className="pt-1.5 mt-1.5 border-t border-slate-800/90 space-y-1">
            <div className="flex justify-between items-center text-[11px]">
              <span className="text-slate-400">ניצחון רגיל (Single)</span>
              <span
                dir="ltr"
                className="inline-block font-mono text-emerald-300"
              >
                {singleWinProb.toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between items-center text-[11px]">
              <span className="text-slate-400">סיכוי מארס (Gammon)</span>
              <span dir="ltr" className="inline-block font-mono text-amber-400">
                {gammonProb.toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between items-center text-[11px]">
              <span className="text-slate-400">מארס טורקי (Backgammon)</span>
              <span dir="ltr" className="inline-block font-mono text-rose-400">
                {backgammonProb.toFixed(1)}%
              </span>
            </div>
          </div>

          <div className="pt-1.5 mt-1.5 border-t border-slate-800/90 space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-slate-400">הפרש פיפס (רגיל)</span>
              <span className="font-mono font-semibold text-cyan-400">
                {pipDiffLabel}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-400">פיפס אפקטיבי (EPC)</span>
              <span className="font-mono font-semibold text-teal-300">
                {epcDiffLabel}
              </span>
            </div>
            {effectivePips && (
              <div className="flex justify-between items-center text-[10px] text-slate-500">
                <span>EPC לבן / שחור</span>
                <span dir="ltr" className="inline-block font-mono">
                  {effectivePips.white.toFixed(1)} /{' '}
                  {effectivePips.black.toFixed(1)}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
