import React, { useEffect, useLayoutEffect, useState } from 'react';
import type { MoveStep, Player } from '../engine/types';

export interface BoardArrowsProps {
  boardRef: React.RefObject<HTMLDivElement | null>;
  primarySteps?: MoveStep[] | null;
  primaryPlayer?: Player;
  primaryVariant?: 'best' | 'preview';
  secondarySteps?: MoveStep[] | null;
  secondaryPlayer?: Player;
  secondaryVariant?: 'opponent' | 'mistake';
}

interface Point2D {
  x: number;
  y: number;
}

interface MeasuredBoardGeometry {
  width: number;
  height: number;
  anchors: Record<string, Point2D>;
}

const DEFAULT_WIDTH = 1000;
const DEFAULT_HEIGHT = 500;

/**
 * Computes deterministic proportional coordinates on a 1000x500 canvas
 * as an immediate fallback before DOM measurement completes.
 */
function getFallbackCoordinate(
  pos: number | 'bar' | 'off',
  player: Player
): Point2D {
  if (pos === 'bar') {
    return {
      x: 462,
      y: player === 'white' ? 120 : 380,
    };
  }

  if (pos === 'off') {
    return {
      x: 960,
      y: player === 'black' ? 120 : 380,
    };
  }

  const isTop = pos >= 12;
  const y = isTop ? 95 : 405;

  // Left quadrant: 12..17 (top) or 11..6 (bottom) -> x in [12..432]
  if (pos >= 12 && pos <= 17) {
    const col = pos - 12;
    return { x: 47 + col * 70, y };
  }
  if (pos >= 6 && pos <= 11) {
    const col = 11 - pos;
    return { x: 47 + col * 70, y };
  }

  // Right quadrant: 18..23 (top) or 5..0 (bottom) -> x in [492..912]
  if (pos >= 18 && pos <= 23) {
    const col = pos - 18;
    return { x: 527 + col * 70, y };
  }
  const col = 5 - pos;
  return { x: 527 + col * 70, y };
}

function getAnchorKey(pos: number | 'bar' | 'off', player: Player): string {
  if (pos === 'bar') {
    return player === 'white' ? 'bar-white' : 'bar-black';
  }
  if (pos === 'off') {
    return player === 'white' ? 'off-white' : 'off-black';
  }
  return `point-${pos}`;
}

export const BoardArrows: React.FC<BoardArrowsProps> = ({
  boardRef,
  primarySteps = null,
  primaryPlayer = 'white',
  primaryVariant = 'best',
  secondarySteps = null,
  secondaryPlayer = 'black',
  secondaryVariant = 'opponent',
}) => {
  const [geometry, setGeometry] = useState<MeasuredBoardGeometry | null>(null);

  const measureAnchors = () => {
    const container = boardRef.current;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const anchors: Record<string, Point2D> = {};
    const elements = container.querySelectorAll<HTMLElement>(
      '[data-board-anchor]'
    );

    elements.forEach((el) => {
      const key = el.getAttribute('data-board-anchor');
      if (!key) return;
      const elRect = el.getBoundingClientRect();
      const cx = elRect.left - rect.left + elRect.width / 2;

      let cy = elRect.top - rect.top + elRect.height / 2;
      if (key.startsWith('point-')) {
        const ptIdx = Number(key.replace('point-', ''));
        const isTop = ptIdx >= 12;
        cy =
          elRect.top -
          rect.top +
          (isTop ? elRect.height * 0.34 : elRect.height * 0.66);
      }

      anchors[key] = { x: cx, y: cy };
    });

    setGeometry({
      width: rect.width,
      height: rect.height,
      anchors,
    });
  };

  useLayoutEffect(() => {
    measureAnchors();
  }, [primarySteps, secondarySteps, primaryPlayer, secondaryPlayer]);

  useEffect(() => {
    const container = boardRef.current;
    if (!container || typeof ResizeObserver === 'undefined') return;

    const observer = new ResizeObserver(() => {
      measureAnchors();
    });
    observer.observe(container);
    window.addEventListener('resize', measureAnchors);

    return () => {
      observer.disconnect();
      window.removeEventListener('resize', measureAnchors);
    };
  }, [boardRef]);

  const hasPrimary = Boolean(primarySteps && primarySteps.length > 0);
  const hasSecondary = Boolean(secondarySteps && secondarySteps.length > 0);

  if (!hasPrimary && !hasSecondary) {
    return null;
  }

  const width = geometry?.width ?? DEFAULT_WIDTH;
  const height = geometry?.height ?? DEFAULT_HEIGHT;

  const resolvePoint = (
    pos: number | 'bar' | 'off',
    player: Player
  ): Point2D => {
    const key = getAnchorKey(pos, player);
    if (geometry && geometry.anchors[key]) {
      return geometry.anchors[key];
    }
    return getFallbackCoordinate(pos, player);
  };

  const renderStepSeries = (
    steps: MoveStep[],
    player: Player,
    seriesType: 'primary' | 'secondary'
  ) => {
    const isPrimary = seriesType === 'primary';
    const strokeColor = isPrimary
      ? primaryVariant === 'preview'
        ? '#06b6d4'
        : '#10b981'
      : secondaryVariant === 'mistake'
        ? '#f59e0b'
        : '#38bdf8';

    const markerId = isPrimary
      ? primaryVariant === 'preview'
        ? 'arrow-preview'
        : 'arrow-best'
      : secondaryVariant === 'mistake'
        ? 'arrow-mistake'
        : 'arrow-opponent';

    const isDashed = !isPrimary && secondaryVariant === 'mistake';

    // Track repeated directed segments so parallel steps (e.g., doubles) curve distinctly
    const pairCounters = new Map<string, number>();

    return steps.map((step, idx) => {
      const start = resolvePoint(step.from, player);
      const end = resolvePoint(step.to, player);

      const pairKey = `${step.from}->${step.to}`;
      const occurrence = pairCounters.get(pairKey) ?? 0;
      pairCounters.set(pairKey, occurrence + 1);

      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const dist = Math.max(1, Math.hypot(dx, dy));

      // Shorten start and end slightly so arrow tips land cleanly on checker edges
      const startPad = Math.min(14, dist * 0.12);
      const endPad = Math.min(20, dist * 0.18);

      const ux = dx / dist;
      const uy = dy / dist;

      const x1 = start.x + ux * startPad;
      const y1 = start.y + uy * startPad;
      const x2 = end.x - ux * endPad;
      const y2 = end.y - uy * endPad;

      // Normal vector for smooth quadratic Bezier curvature
      const nx = -uy;
      const ny = ux;

      const baseSign = isPrimary ? 1 : -1;
      const arcMagnitude =
        (22 + occurrence * 18 + (idx % 2 === 0 ? 6 : -6)) * baseSign;

      const cx = (x1 + x2) / 2 + nx * arcMagnitude;
      const cy = (y1 + y2) / 2 + ny * arcMagnitude;

      // Quadratic Bezier midpoint at t = 0.5: B(0.5) = 0.25*P0 + 0.5*C + 0.25*P2
      const badgeX = 0.25 * x1 + 0.5 * cx + 0.25 * x2;
      const badgeY = 0.25 * y1 + 0.5 * cy + 0.25 * y2;

      const pathData = `M ${x1.toFixed(1)} ${y1.toFixed(1)} Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`;

      return (
        <g key={`${seriesType}-${idx}-${step.from}-${step.to}`}>
          {/* Dark halo under-stroke for high contrast over any checker color */}
          <path
            d={pathData}
            fill="none"
            stroke="rgba(2, 6, 23, 0.8)"
            strokeWidth={isPrimary ? 7.5 : 6.5}
            strokeLinecap="round"
          />

          {/* Main tactical arrow path */}
          <path
            d={pathData}
            fill="none"
            stroke={strokeColor}
            strokeWidth={isPrimary ? 4.2 : 3.5}
            strokeLinecap="round"
            strokeDasharray={isDashed ? '8 6' : undefined}
            markerEnd={`url(#${markerId})`}
            opacity={isPrimary ? 0.96 : 0.88}
          />

          {/* Origin pulse ring */}
          <circle
            cx={start.x}
            cy={start.y}
            r={isPrimary ? 6 : 5}
            fill={strokeColor}
            stroke="#020617"
            strokeWidth={2}
            opacity={0.9}
          />

          {/* Hit burst badge (*) when step.isHit is true */}
          {step.isHit && (
            <g transform={`translate(${end.x.toFixed(1)}, ${end.y.toFixed(1)})`}>
              <circle
                r={12}
                fill="#f43f5e"
                stroke="#fff1f2"
                strokeWidth={2}
              />
              <text
                x={0}
                y={4.5}
                textAnchor="middle"
                fill="#ffffff"
                fontSize="14"
                fontWeight="900"
                fontFamily="monospace"
              >
                *
              </text>
            </g>
          )}

          {/* Step Order Badge (1, 2, 3, 4) */}
          <g
            transform={`translate(${badgeX.toFixed(1)}, ${badgeY.toFixed(1)})`}
          >
            <circle
              r={10.5}
              fill="#090d16"
              stroke={strokeColor}
              strokeWidth={2.2}
            />
            <text
              x={0}
              y={3.8}
              textAnchor="middle"
              fill={strokeColor}
              fontSize="11"
              fontWeight="800"
              fontFamily="monospace"
            >
              {idx + 1}
            </text>
          </g>
        </g>
      );
    });
  };

  return (
    <div
      dir="ltr"
      className="pointer-events-none absolute inset-0 z-30 overflow-hidden"
    >
      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-full"
      >
        <defs>
          {/* Primary Best Move Arrowhead (Emerald) */}
          <marker
            id="arrow-best"
            viewBox="0 0 10 10"
            refX="7"
            refY="5"
            markerWidth="5.5"
            markerHeight="5.5"
            orient="auto-start-reverse"
          >
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#10b981" />
          </marker>

          {/* Primary Hover Preview Arrowhead (Cyan) */}
          <marker
            id="arrow-preview"
            viewBox="0 0 10 10"
            refX="7"
            refY="5"
            markerWidth="5.5"
            markerHeight="5.5"
            orient="auto-start-reverse"
          >
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#06b6d4" />
          </marker>

          {/* Secondary Opponent Last Move Arrowhead (Sky Blue) */}
          <marker
            id="arrow-opponent"
            viewBox="0 0 10 10"
            refX="7"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8" />
          </marker>

          {/* Secondary Played Mistake Arrowhead (Amber) */}
          <marker
            id="arrow-mistake"
            viewBox="0 0 10 10"
            refX="7"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#f59e0b" />
          </marker>
        </defs>

        {hasSecondary &&
          secondarySteps &&
          renderStepSeries(secondarySteps, secondaryPlayer, 'secondary')}

        {hasPrimary &&
          primarySteps &&
          renderStepSeries(primarySteps, primaryPlayer, 'primary')}
      </svg>

      {/* Subtle Hebrew Legend Pill on Center Strip */}
      <div
        dir="rtl"
        className="absolute bottom-1.5 left-2.5 flex items-center gap-2 px-2.5 py-1 rounded-full bg-slate-950/85 border border-slate-800/90 backdrop-blur-sm text-[10px] font-semibold text-slate-300 shadow-lg"
      >
        {hasPrimary && (
          <span className="inline-flex items-center gap-1">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                primaryVariant === 'preview' ? 'bg-cyan-400' : 'bg-emerald-400'
              }`}
            />
            {primaryVariant === 'preview' ? 'תצוגת קו נבחר' : 'קו מומלץ (AI)'}
          </span>
        )}
        {hasSecondary && (
          <span className="inline-flex items-center gap-1">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                secondaryVariant === 'mistake' ? 'bg-amber-400' : 'bg-sky-400'
              }`}
            />
            {secondaryVariant === 'mistake'
              ? 'המהלך ששוחק (להשוואה)'
              : 'מהלך היריב'}
          </span>
        )}
      </div>
    </div>
  );
};
