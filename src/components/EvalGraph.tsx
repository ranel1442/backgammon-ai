import React from 'react';
import { getClassificationMeta, type MoveClassification } from '../engine/ai';
import type { EquityTimelinePoint } from '../engine/review';

interface EvalGraphProps {
  timeline: EquityTimelinePoint[];
  selectedTurnIndex: number | null;
  onSelectTurnIndex: (turnIndex: number) => void;
}

const DOT_COLORS: Partial<Record<MoveClassification, string>> = {
  Inaccuracy: '#facc15', // yellow-400
  Mistake: '#f97316', // orange-500
  Blunder: '#f43f5e', // rose-500
};

export const EvalGraph: React.FC<EvalGraphProps> = ({
  timeline,
  selectedTurnIndex,
  onSelectTurnIndex,
}) => {
  const width = 660;
  const height = 160;
  const padX = 28;
  const padY = 16;
  const midY = height / 2;
  const plotW = width - padX * 2;
  const plotH = height - padY * 2;

  const equityToY = (eq: number): number => {
    const clamped = Math.max(-3.0, Math.min(3.0, eq));
    return midY - (clamped / 3.0) * (plotH / 2);
  };

  const indexToX = (idx: number, total: number): number => {
    if (total <= 1) return padX + plotW / 2;
    return padX + (idx / (total - 1)) * plotW;
  };

  if (timeline.length === 0) {
    return (
      <div
        dir="rtl"
        className="h-36 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center justify-center text-xs text-slate-500"
      >
        שחק מספר תורות כדי להציג את גרף התפתחות היתרון במשחק
      </div>
    );
  }

  const coords = timeline.map((pt, idx) => ({
    x: indexToX(idx, timeline.length),
    y: equityToY(pt.equity),
    pt,
  }));

  const polylinePoints = coords
    .map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`)
    .join(' ');

  const firstX = coords[0].x.toFixed(1);
  const lastX = coords[coords.length - 1].x.toFixed(1);
  const areaPath = `M ${firstX},${midY} L ${coords
    .map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`)
    .join(' L ')} L ${lastX},${midY} Z`;

  const activeCoord =
    selectedTurnIndex !== null && coords[selectedTurnIndex]
      ? coords[selectedTurnIndex]
      : null;

  return (
    <div
      dir="rtl"
      className="relative rounded-xl bg-slate-950/90 border border-slate-800 p-2.5 select-none"
    >
      <div className="flex items-center justify-between text-[10px] text-slate-400 mb-1 px-1">
        <span>
          יתרון הלבן (<span dir="ltr" className="inline-block font-mono">+3.0</span>)
        </span>
        <div className="flex items-center gap-3">
          <span className="inline-flex items-center gap-1 text-yellow-400">
            ● אי-דיוק
          </span>
          <span className="inline-flex items-center gap-1 text-orange-400">
            ● טעות
          </span>
          <span className="inline-flex items-center gap-1 text-rose-400">
            ● טעות גסה
          </span>
        </div>
        <span>
          יתרון השחור (<span dir="ltr" className="inline-block font-mono">-3.0</span>)
        </span>
      </div>

      <div dir="ltr">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-36 sm:h-40 overflow-visible"
        >
        <defs>
          <clipPath id="white-above-zero">
            <rect x="0" y="0" width={width} height={midY} />
          </clipPath>
          <clipPath id="black-below-zero">
            <rect x="0" y={midY} width={width} height={midY} />
          </clipPath>
          <linearGradient id="white-area-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#34d399" stopOpacity="0.45" />
            <stop offset="100%" stopColor="#e7e5e4" stopOpacity="0.08" />
          </linearGradient>
          <linearGradient id="black-area-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.08" />
            <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.4" />
          </linearGradient>
        </defs>

        <line
          x1={padX}
          y1={padY}
          x2={width - padX}
          y2={padY}
          stroke="#1e293b"
          strokeDasharray="3 3"
          strokeWidth="1"
        />
        <line
          x1={padX}
          y1={midY}
          x2={width - padX}
          y2={midY}
          stroke="#475569"
          strokeWidth="1.25"
        />
        <line
          x1={padX}
          y1={height - padY}
          x2={width - padX}
          y2={height - padY}
          stroke="#1e293b"
          strokeDasharray="3 3"
          strokeWidth="1"
        />

        {coords.length > 1 && (
          <>
            <path
              d={areaPath}
              fill="url(#white-area-grad)"
              clipPath="url(#white-above-zero)"
            />
            <path
              d={areaPath}
              fill="url(#black-area-grad)"
              clipPath="url(#black-below-zero)"
            />
          </>
        )}

        {coords.length > 1 && (
          <polyline
            fill="none"
            stroke="#38bdf8"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={polylinePoints}
          />
        )}

        {activeCoord && (
          <line
            x1={activeCoord.x}
            y1={padY}
            x2={activeCoord.x}
            y2={height - padY}
            stroke="#fbbf24"
            strokeWidth="1.5"
            strokeDasharray="2 2"
          />
        )}

        {coords.map(({ x, y, pt }) => {
          const isSelected = selectedTurnIndex === pt.turnIndex;
          const dotColor = pt.classification
            ? DOT_COLORS[pt.classification]
            : undefined;
          const labelHe = pt.classification
            ? getClassificationMeta(pt.classification).label
            : '';

          return (
            <g
              key={pt.turnIndex}
              className="cursor-pointer group/pt"
              onClick={() => onSelectTurnIndex(pt.turnIndex)}
            >
              <circle cx={x} cy={y} r={10} fill="transparent" />

              {isSelected && (
                <circle
                  cx={x}
                  cy={y}
                  r={7}
                  fill="none"
                  stroke="#fbbf24"
                  strokeWidth="2"
                />
              )}

              <circle
                cx={x}
                cy={y}
                r={dotColor ? 4.5 : isSelected ? 4 : 2.5}
                fill={dotColor || (isSelected ? '#fbbf24' : '#e2e8f0')}
                stroke="#0f172a"
                strokeWidth="1.2"
              >
                <title>
                  תור #{pt.turnIndex + 1} ({pt.player === 'white' ? 'לבן' : 'שחור'}):{' '}
                  {pt.equity >= 0
                    ? `+${pt.equity.toFixed(2)}`
                    : pt.equity.toFixed(2)}
                  {labelHe ? ` [${labelHe}]` : ''}
                </title>
              </circle>
            </g>
          );
        })}
        </svg>
      </div>
    </div>
  );
};
