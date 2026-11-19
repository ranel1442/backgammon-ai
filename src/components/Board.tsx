import React, { useRef } from 'react';
import type { ValidDestinationOption } from '../engine/rules';
import type { BoardState, MoveStep, Player } from '../engine/types';
import { BoardArrows } from './BoardArrows';

interface BoardProps {
  boardState: BoardState;
  validNextSteps: MoveStep[];
  validDestinationOptions?: ValidDestinationOption[];
  selectedFrom: number | 'bar' | null;
  onSelectFrom: (from: number | 'bar' | null) => void;
  onExecuteStep: (step: MoveStep) => void;
  onExecuteStepSequence?: (steps: MoveStep[]) => void;
  isInteractive: boolean;
  rolledDice: [number, number] | null;
  stepsTaken: MoveStep[];
  lastTurnSteps?: MoveStep[];
  previewSteps?: MoveStep[] | null;
  primaryArrowSteps?: MoveStep[] | null;
  primaryArrowPlayer?: Player;
  primaryArrowVariant?: 'best' | 'preview';
  secondaryArrowSteps?: MoveStep[] | null;
  secondaryArrowPlayer?: Player;
  secondaryArrowVariant?: 'opponent' | 'mistake';
  aiStatusBanner?: string | null;
}

const TOP_LEFT_POINTS = [12, 13, 14, 15, 16, 17];
const TOP_RIGHT_POINTS = [18, 19, 20, 21, 22, 23];
const BOTTOM_LEFT_POINTS = [11, 10, 9, 8, 7, 6];
const BOTTOM_RIGHT_POINTS = [5, 4, 3, 2, 1, 0];

const PIP_POSITIONS: Record<number, number[]> = {
  1: [4],
  2: [0, 8],
  3: [0, 4, 8],
  4: [0, 2, 6, 8],
  5: [0, 2, 4, 6, 8],
  6: [0, 2, 3, 5, 6, 8],
};

interface DieFaceProps {
  value: number;
  used: boolean;
  player: Player;
}

const DieFace: React.FC<DieFaceProps> = ({ value, used, player }) => {
  const pips = PIP_POSITIONS[value] || [];
  const isWhite = player === 'white';

  return (
    <div
      dir="ltr"
      className={`relative w-10 h-10 sm:w-11 sm:h-11 rounded-xl p-1.5 grid grid-cols-3 grid-rows-3 gap-0.5 transition-all duration-300 shadow-lg border ${
        used
          ? 'opacity-30 scale-90 bg-slate-800/80 border-slate-700'
          : isWhite
            ? 'bg-gradient-to-br from-stone-50 via-stone-100 to-stone-300 border-stone-300 shadow-emerald-950/50 ring-1 ring-white/60'
            : 'bg-gradient-to-br from-slate-800 via-slate-900 to-slate-950 border-slate-600 shadow-black/70 ring-1 ring-slate-500/40'
      }`}
      title={`קובייה ${value} (${used ? 'נוצלה' : 'זמינה'})`}
    >
      {Array.from({ length: 9 }).map((_, idx) => {
        const hasPip = pips.includes(idx);
        return (
          <div key={idx} className="flex items-center justify-center">
            {hasPip && (
              <span
                className={`w-2 h-2 rounded-full ${
                  used
                    ? 'bg-slate-400'
                    : isWhite
                      ? 'bg-slate-900 shadow-inner'
                      : 'bg-stone-100 shadow-sm'
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
};

interface CheckerPieceProps {
  owner: Player;
  countBadge?: number;
  isTopOfStack?: boolean;
  isMovable?: boolean;
  isSelected?: boolean;
  size?: 'normal' | 'bar';
}

const CheckerPiece: React.FC<CheckerPieceProps> = ({
  owner,
  countBadge,
  isTopOfStack = false,
  isMovable = false,
  isSelected = false,
  size = 'normal',
}) => {
  const isWhite = owner === 'white';
  const sizeClasses =
    size === 'bar'
      ? 'w-8 h-8 sm:w-9 sm:h-9'
      : 'w-8 h-8 sm:w-10 sm:h-10';

  return (
    <div
      className={`relative rounded-full flex items-center justify-center transition-all duration-200 select-none ${sizeClasses} ${
        isWhite
          ? 'bg-gradient-to-b from-stone-50 via-stone-200 to-stone-300 border-2 border-stone-300 shadow-[0_3px_8px_rgba(0,0,0,0.65),inset_0_2px_3px_rgba(255,255,255,0.9)] text-slate-900'
          : 'bg-gradient-to-b from-slate-700 via-slate-900 to-slate-950 border-2 border-slate-600 shadow-[0_3px_8px_rgba(0,0,0,0.85),inset_0_1px_2px_rgba(255,255,255,0.25)] text-stone-100'
      } ${
        isSelected && isTopOfStack
          ? 'ring-4 ring-amber-400 scale-105 z-30 shadow-[0_0_18px_rgba(251,191,36,0.7)]'
          : isMovable && isTopOfStack
            ? 'ring-2 ring-emerald-400/90 group-hover/point:ring-emerald-300 group-hover/point:scale-105 z-20'
            : ''
      }`}
    >
      <div
        className={`w-[72%] h-[72%] rounded-full border flex items-center justify-center ${
          isWhite
            ? 'border-stone-400/60 bg-gradient-to-t from-stone-100 to-stone-200/60'
            : 'border-slate-700/90 bg-gradient-to-t from-slate-950 to-slate-800/70'
        }`}
      >
        {countBadge !== undefined && countBadge > 5 && (
          <span
            dir="ltr"
            className={`inline-block text-xs sm:text-sm font-extrabold font-mono leading-none tracking-tighter ${
              isWhite ? 'text-slate-900' : 'text-amber-300'
            }`}
          >
            {countBadge}
          </span>
        )}
      </div>
    </div>
  );
};

export const Board: React.FC<BoardProps> = ({
  boardState,
  validNextSteps,
  validDestinationOptions,
  selectedFrom,
  onSelectFrom,
  onExecuteStep,
  onExecuteStepSequence,
  isInteractive,
  rolledDice,
  stepsTaken,
  lastTurnSteps = [],
  previewSteps = null,
  primaryArrowSteps = null,
  primaryArrowPlayer = boardState.currentTurn,
  primaryArrowVariant = 'best',
  secondaryArrowSteps = null,
  secondaryArrowPlayer = 'black',
  secondaryArrowVariant = 'opponent',
  aiStatusBanner = null,
}) => {
  const boardContainerRef = useRef<HTMLDivElement | null>(null);

  const effectivePrimaryArrows = previewSteps ?? primaryArrowSteps;
  const effectivePrimaryVariant: 'best' | 'preview' = previewSteps
    ? 'preview'
    : primaryArrowVariant;

  // Engine hover preview sets
  const previewFromSet = new Set(
    previewSteps ? previewSteps.map((s) => s.from) : []
  );
  const previewToMap = new Map<number | 'off', MoveStep>();
  if (previewSteps) {
    for (const s of previewSteps) {
      previewToMap.set(s.to, s);
    }
  }

  // Determine which origins have legal moves right now
  const movableOrigins = new Set<number | 'bar'>(
    isInteractive ? validNextSteps.map((s) => s.from) : []
  );

  // Determine valid target destinations (single-die + combined multi-step) from the currently selected origin
  const resolvedDestOptions: ValidDestinationOption[] =
    isInteractive && selectedFrom !== null
      ? validDestinationOptions && validDestinationOptions.length > 0
        ? validDestinationOptions
        : validNextSteps
            .filter((s) => s.from === selectedFrom)
            .map((s) => ({
              to: s.to,
              steps: [s],
              isCombined: false,
              diceBadge: String(s.dieUsed),
              includesHit: s.isHit,
            }))
      : [];

  const targetOptionByPoint = new Map<number, ValidDestinationOption>();
  let bearOffTargetOption: ValidDestinationOption | null = null;

  for (const opt of resolvedDestOptions) {
    if (opt.to === 'off') {
      if (!bearOffTargetOption || (!opt.isCombined && bearOffTargetOption.isCombined)) {
        bearOffTargetOption = opt;
      }
    } else if (!targetOptionByPoint.has(opt.to)) {
      targetOptionByPoint.set(opt.to, opt);
    }
  }

  const executeOption = (opt: ValidDestinationOption) => {
    if (opt.steps.length > 1 && onExecuteStepSequence) {
      onExecuteStepSequence(opt.steps);
    } else {
      for (const s of opt.steps) {
        onExecuteStep(s);
      }
    }
  };

  // Recent move highlights
  const recentFromSet = new Set(lastTurnSteps.map((s) => s.from));
  const recentToSet = new Set(lastTurnSteps.map((s) => s.to));

  // Build dice display array (showing used vs available dice)
  const diceDisplayList: { value: number; used: boolean }[] = [];
  if (rolledDice) {
    const isDoubles = rolledDice[0] === rolledDice[1];
    const totalDice = isDoubles
      ? [rolledDice[0], rolledDice[0], rolledDice[0], rolledDice[0]]
      : [rolledDice[0], rolledDice[1]];

    const usedCounts = new Map<number, number>();
    for (const step of stepsTaken) {
      usedCounts.set(step.dieUsed, (usedCounts.get(step.dieUsed) || 0) + 1);
    }

    for (const val of totalDice) {
      const remainingUsed = usedCounts.get(val) || 0;
      if (remainingUsed > 0) {
        diceDisplayList.push({ value: val, used: true });
        usedCounts.set(val, remainingUsed - 1);
      } else {
        diceDisplayList.push({ value: val, used: false });
      }
    }
  }

  const handlePointClick = (pointIndex: number) => {
    if (!isInteractive) return;

    const targetOpt = targetOptionByPoint.get(pointIndex);
    if (targetOpt) {
      executeOption(targetOpt);
      return;
    }

    if (movableOrigins.has(pointIndex)) {
      if (selectedFrom === pointIndex) {
        onSelectFrom(null);
      } else {
        onSelectFrom(pointIndex);
      }
      return;
    }

    if (selectedFrom !== null) {
      onSelectFrom(null);
    }
  };

  const handleBarClick = () => {
    if (!isInteractive) return;
    if (movableOrigins.has('bar')) {
      onSelectFrom(selectedFrom === 'bar' ? null : 'bar');
    }
  };

  const handleBearOffClick = (trayOwner: Player) => {
    if (!isInteractive) return;
    if (trayOwner === boardState.currentTurn && bearOffTargetOption) {
      executeOption(bearOffTargetOption);
    }
  };

  const renderPointColumn = (pointIndex: number, isTopRow: boolean) => {
    const rawCount = boardState.points[pointIndex];
    const owner: Player | null =
      rawCount > 0 ? 'white' : rawCount < 0 ? 'black' : null;
    const count = Math.abs(rawCount);
    const visibleCheckers = Math.min(5, count);

    const isMovableOrigin = movableOrigins.has(pointIndex);
    const isSelectedOrigin = selectedFrom === pointIndex;
    const targetOption = targetOptionByPoint.get(pointIndex);
    const isTarget = Boolean(targetOption);
    const isCombinedTarget = Boolean(targetOption?.isCombined);
    const isRecentOrigin = recentFromSet.has(pointIndex);
    const isRecentTarget = recentToSet.has(pointIndex);
    const isPreviewOrigin = previewFromSet.has(pointIndex);
    const previewTargetStep = previewToMap.get(pointIndex);
    const isPreviewTarget = Boolean(previewTargetStep);

    const isEvenPoint = pointIndex % 2 === 0;
    const triangleColorClass = isEvenPoint
      ? 'bg-emerald-900/75'
      : 'bg-slate-700/75';

    const pointNumberLabel = pointIndex + 1;

    return (
      <div
        key={pointIndex}
        data-board-anchor={`point-${pointIndex}`}
        onClick={() => handlePointClick(pointIndex)}
        className={`group/point relative flex-1 flex flex-col items-center justify-between h-[195px] sm:h-[225px] select-none transition-colors ${
          isTarget || isMovableOrigin ? 'cursor-pointer' : 'cursor-default'
        }`}
      >
        {isTopRow && (
          <div
            dir="ltr"
            className={`w-full text-center text-[10px] font-mono py-0.5 border-b border-slate-800/80 ${
              isSelectedOrigin
                ? 'text-amber-300 font-bold bg-amber-500/10'
                : isCombinedTarget
                  ? 'text-amber-300 font-bold bg-amber-500/20'
                  : isTarget
                    ? 'text-emerald-300 font-bold bg-emerald-500/15'
                    : 'text-slate-400 bg-slate-950/60'
            }`}
          >
            {pointNumberLabel}
          </div>
        )}

        <div
          className={`relative w-full flex-1 flex ${
            isTopRow
              ? 'flex-col justify-start pt-1'
              : 'flex-col-reverse justify-start pb-1'
          } items-center`}
        >
          {/* Crisp Triangle Spike */}
          <div
            className={`absolute inset-x-0.5 ${
              isTopRow ? 'top-0 bottom-2' : 'bottom-0 top-2'
            } transition-all duration-200 ${triangleColorClass} ${
              isCombinedTarget
                ? 'ring-2 ring-amber-400/95 shadow-[0_0_14px_rgba(251,191,36,0.45)] brightness-125'
                : isTarget
                  ? 'ring-2 ring-emerald-400/90 brightness-125'
                  : isPreviewTarget || isPreviewOrigin
                    ? 'ring-2 ring-cyan-400/90 brightness-125'
                    : isSelectedOrigin
                      ? 'brightness-125'
                      : isRecentTarget || isRecentOrigin
                        ? 'brightness-110'
                        : ''
            }`}
            style={{
              clipPath: isTopRow
                ? 'polygon(0% 0%, 100% 0%, 50% 100%)'
                : 'polygon(50% 0%, 0% 100%, 100% 100%)',
            }}
          />

          {(isRecentOrigin || isRecentTarget) &&
            !isTarget &&
            !isSelectedOrigin && (
              <div
                className={`absolute ${
                  isTopRow ? 'top-0.5' : 'bottom-0.5'
                } w-1.5 h-1.5 rounded-full bg-cyan-400/70 z-10 pointer-events-none`}
              />
            )}

          {/* Checkers Stack */}
          <div
            className={`relative z-10 flex ${
              isTopRow ? 'flex-col' : 'flex-col-reverse'
            } items-center -space-y-1 sm:space-y-0.5`}
          >
            {owner &&
              Array.from({ length: visibleCheckers }).map((_, idx) => {
                const isTopChecker = idx === visibleCheckers - 1;
                return (
                  <CheckerPiece
                    key={idx}
                    owner={owner}
                    countBadge={isTopChecker && count > 5 ? count : undefined}
                    isTopOfStack={isTopChecker}
                    isMovable={isMovableOrigin}
                    isSelected={isSelectedOrigin}
                  />
                );
              })}
          </div>

          {/* Glowing Target Indicator when this point is a valid destination */}
          {isTarget && targetOption && (
            <div
              className="relative z-30 my-1 flex flex-col items-center justify-center animate-pulse"
              title={
                targetOption.isCombined
                  ? `מהלך משולב (${targetOption.diceBadge}) — לחץ לביצוע ברצף`
                  : 'בחר יעד למהלך'
              }
            >
              <div
                dir="ltr"
                className={`min-w-8 h-8 sm:min-w-9 sm:h-9 px-1.5 rounded-full flex items-center justify-center font-mono text-[11px] sm:text-xs font-extrabold border-2 shadow-lg transition-transform group-hover/point:scale-110 ${
                  targetOption.includesHit
                    ? targetOption.isCombined
                      ? 'bg-gradient-to-br from-rose-500 to-amber-500 border-amber-100 ring-2 ring-amber-300/90 text-white shadow-rose-500/60'
                      : 'bg-rose-500/90 border-rose-200 text-white shadow-rose-500/50'
                    : targetOption.isCombined
                      ? 'bg-gradient-to-br from-amber-300 via-emerald-400 to-teal-400 border-amber-100 ring-2 ring-emerald-300/90 text-slate-950 shadow-amber-400/50'
                      : 'bg-emerald-500/90 border-emerald-200 text-slate-950 shadow-emerald-500/50'
                }`}
              >
                {targetOption.includesHit
                  ? `${targetOption.diceBadge}*`
                  : targetOption.diceBadge}
              </div>
            </div>
          )}

          {/* Engine Line Hover Preview Badge */}
          {!isTarget && (isPreviewTarget || isPreviewOrigin) && (
            <div className="relative z-30 my-1 flex flex-col items-center justify-center">
              <div
                dir="ltr"
                className={`px-1.5 py-0.5 rounded-md font-mono text-[10px] font-bold border shadow-md ${
                  isPreviewTarget
                    ? 'bg-cyan-400 text-slate-950 border-cyan-200 shadow-cyan-500/40'
                    : 'bg-cyan-950/90 text-cyan-300 border-cyan-400/70'
                }`}
              >
                {isPreviewTarget && previewTargetStep
                  ? `→${previewTargetStep.dieUsed}${previewTargetStep.isHit ? '*' : ''}`
                  : 'מוצא'}
              </div>
            </div>
          )}
        </div>

        {!isTopRow && (
          <div
            dir="ltr"
            className={`w-full text-center text-[10px] font-mono py-0.5 border-t border-slate-800/80 ${
              isSelectedOrigin
                ? 'text-amber-300 font-bold bg-amber-500/10'
                : isCombinedTarget
                  ? 'text-amber-300 font-bold bg-amber-500/20'
                  : isTarget
                    ? 'text-emerald-300 font-bold bg-emerald-500/15'
                    : 'text-slate-400 bg-slate-950/60'
            }`}
          >
            {pointNumberLabel}
          </div>
        )}
      </div>
    );
  };

  const isBarMovable = movableOrigins.has('bar');
  const isBarSelected = selectedFrom === 'bar';

  return (
    /* Keep physical board container dir="ltr" so point geometry (12-17, 18-23, 11-6, 5-0) is preserved */
    <div
      ref={boardContainerRef}
      dir="ltr"
      className="relative flex-1 flex rounded-2xl bg-slate-950 border-2 border-slate-800 shadow-[0_20px_50px_rgba(0,0,0,0.75)] overflow-hidden"
    >
      {/* Tactical SVG Arrows Overlay */}
      <BoardArrows
        boardRef={boardContainerRef}
        primarySteps={effectivePrimaryArrows}
        primaryPlayer={primaryArrowPlayer}
        primaryVariant={effectivePrimaryVariant}
        secondarySteps={secondaryArrowSteps}
        secondaryPlayer={secondaryArrowPlayer}
        secondaryVariant={secondaryArrowVariant}
      />

      {/* Main 4-Quadrant Board Area */}
      <div className="flex-1 flex flex-col justify-between bg-[#0b1220] relative">
        {/* TOP HALF: Top-Left (12..17) | Central Bar | Top-Right Black Home (18..23) */}
        <div className="flex items-stretch w-full">
          {/* Top-Left Quadrant (Points 12 to 17) */}
          <div className="flex-1 flex px-1.5 sm:px-2">
            {TOP_LEFT_POINTS.map((pt) => renderPointColumn(pt, true))}
          </div>

          {/* Central Bar (Top Section - White hit checkers waiting to enter 23..18) */}
          <div
            data-board-anchor="bar-white"
            onClick={handleBarClick}
            className={`w-12 sm:w-14 bg-slate-900/95 border-x-2 border-slate-800 flex flex-col items-center justify-between py-2 select-none transition-colors ${
              isBarMovable ? 'cursor-pointer hover:bg-slate-800/90' : ''
            }`}
            title="בר (כלים שהוכו)"
          >
            <span
              dir="rtl"
              className="text-[10px] font-bold tracking-wide text-slate-400"
            >
              בר
            </span>
            <div className="flex flex-col items-center gap-1 my-auto">
              {boardState.bar.white > 0 && (
                <div className="relative flex flex-col items-center">
                  <CheckerPiece
                    owner="white"
                    size="bar"
                    isTopOfStack={true}
                    isMovable={
                      isBarMovable && boardState.currentTurn === 'white'
                    }
                    isSelected={
                      isBarSelected && boardState.currentTurn === 'white'
                    }
                  />
                  {boardState.bar.white > 1 && (
                    <span
                      dir="ltr"
                      className="inline-block mt-0.5 px-1.5 py-0.2 rounded-full bg-amber-400 text-slate-950 font-mono text-[10px] font-extrabold shadow"
                    >
                      ×{boardState.bar.white}
                    </span>
                  )}
                </div>
              )}
            </div>
            <div className="w-6 h-[1px] bg-slate-800" />
          </div>

          {/* Top-Right Quadrant: Black Home Board (Points 18 to 23) */}
          <div className="flex-1 flex px-1.5 sm:px-2 bg-slate-900/20">
            {TOP_RIGHT_POINTS.map((pt) => renderPointColumn(pt, true))}
          </div>
        </div>

        {/* CENTER BOARD STRIP: Active 2D Dice & Turn Status */}
        <div className="relative h-14 sm:h-16 flex items-center justify-between px-4 border-y border-slate-800/60 bg-slate-950/40">
          <div
            dir="rtl"
            className="text-[11px] sm:text-xs font-medium text-slate-500"
          >
            לוח חיצוני
          </div>

          {/* 2D Dice Display & Optional Paced AI Banner */}
          <div className="flex items-center gap-3 z-20">
            {aiStatusBanner && (
              <div
                dir="rtl"
                className="px-3 py-1 rounded-xl bg-sky-500/20 border border-sky-400/50 text-sky-200 text-xs font-bold shadow-lg animate-pulse"
              >
                {aiStatusBanner}
              </div>
            )}

            {diceDisplayList.length > 0 ? (
              <div className="flex items-center gap-2">
                {diceDisplayList.map((d, i) => (
                  <DieFace
                    key={i}
                    value={d.value}
                    used={d.used}
                    player={boardState.currentTurn}
                  />
                ))}
              </div>
            ) : (
              !aiStatusBanner && (
                <div
                  dir="rtl"
                  className="px-3 py-1 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-400"
                >
                  ממתין להטלת קוביות
                </div>
              )
            )}
          </div>

          <div
            dir="rtl"
            className="text-[11px] sm:text-xs font-medium text-slate-500"
          >
            בית (הוצאה)
          </div>
        </div>

        {/* BOTTOM HALF: Bottom-Left (11..6) | Central Bar | Bottom-Right White Home (5..0) */}
        <div className="flex items-stretch w-full">
          {/* Bottom-Left Quadrant (Points 11 down to 6) */}
          <div className="flex-1 flex px-1.5 sm:px-2">
            {BOTTOM_LEFT_POINTS.map((pt) => renderPointColumn(pt, false))}
          </div>

          {/* Central Bar (Bottom Section - Black hit checkers waiting to enter 0..5) */}
          <div
            data-board-anchor="bar-black"
            onClick={handleBarClick}
            className={`w-12 sm:w-14 bg-slate-900/95 border-x-2 border-slate-800 flex flex-col items-center justify-between py-2 select-none transition-colors ${
              isBarMovable ? 'cursor-pointer hover:bg-slate-800/90' : ''
            }`}
            title="בר (כלים שהוכו)"
          >
            <div className="w-6 h-[1px] bg-slate-800" />
            <div className="flex flex-col items-center gap-1 my-auto">
              {boardState.bar.black > 0 && (
                <div className="relative flex flex-col items-center">
                  <CheckerPiece
                    owner="black"
                    size="bar"
                    isTopOfStack={true}
                    isMovable={
                      isBarMovable && boardState.currentTurn === 'black'
                    }
                    isSelected={
                      isBarSelected && boardState.currentTurn === 'black'
                    }
                  />
                  {boardState.bar.black > 1 && (
                    <span
                      dir="ltr"
                      className="inline-block mt-0.5 px-1.5 py-0.2 rounded-full bg-amber-400 text-slate-950 font-mono text-[10px] font-extrabold shadow"
                    >
                      ×{boardState.bar.black}
                    </span>
                  )}
                </div>
              )}
            </div>
            <span
              dir="rtl"
              className="text-[10px] font-bold tracking-wide text-slate-400"
            >
              בר
            </span>
          </div>

          {/* Bottom-Right Quadrant: White Home Board (Points 5 down to 0) */}
          <div className="flex-1 flex px-1.5 sm:px-2 bg-slate-900/20">
            {BOTTOM_RIGHT_POINTS.map((pt) => renderPointColumn(pt, false))}
          </div>
        </div>
      </div>

      {/* RIGHT BEAR-OFF TRAYS (Top: Black Borne-Off, Bottom: White Borne-Off) */}
      <div className="w-16 sm:w-20 bg-slate-950 border-l-2 border-slate-800 flex flex-col justify-between p-1.5 sm:p-2 select-none shrink-0">
        {/* Black Bear-Off Tray (Top) */}
        {(() => {
          const isBlackBearOffTarget =
            isInteractive &&
            boardState.currentTurn === 'black' &&
            Boolean(bearOffTargetOption);
          return (
            <div
              data-board-anchor="off-black"
              onClick={() => handleBearOffClick('black')}
              title={isBlackBearOffTarget ? 'לחץ להוצאת כלי' : 'הוצאה (שחור)'}
              className={`flex-1 rounded-xl border p-1.5 flex flex-col items-center justify-between transition-all ${
                isBlackBearOffTarget
                  ? 'border-emerald-400 bg-emerald-500/15 ring-2 ring-emerald-400/70 cursor-pointer animate-pulse'
                  : 'border-slate-800 bg-slate-900/60'
              }`}
            >
              <div className="text-center">
                <div
                  dir="rtl"
                  className="text-[10px] font-semibold text-slate-400"
                >
                  הוצאה
                </div>
                <div
                  dir="ltr"
                  className="inline-block text-xs font-mono font-bold text-slate-200"
                >
                  {boardState.borneOff.black}
                  <span className="text-slate-500 text-[10px]">/15</span>
                </div>
              </div>

              <div className="w-full flex flex-col items-center gap-0.5 my-1 overflow-hidden">
                {Array.from({ length: boardState.borneOff.black }).map(
                  (_, i) => (
                    <div
                      key={i}
                      className="w-9 sm:w-11 h-2 rounded-sm bg-gradient-to-r from-slate-800 via-slate-600 to-slate-800 border border-slate-500/60 shadow-sm"
                    />
                  )
                )}
              </div>

              {isBlackBearOffTarget && bearOffTargetOption && (
                <div
                  dir="rtl"
                  className="px-1.5 py-0.5 rounded-full bg-emerald-400 text-slate-950 text-[10px] font-extrabold shadow text-center"
                >
                  הוצא ({bearOffTargetOption.diceBadge})
                </div>
              )}
            </div>
          );
        })()}

        <div className="h-3" />

        {/* White Bear-Off Tray (Bottom) */}
        {(() => {
          const isWhiteBearOffTarget =
            isInteractive &&
            boardState.currentTurn === 'white' &&
            Boolean(bearOffTargetOption);
          return (
            <div
              data-board-anchor="off-white"
              onClick={() => handleBearOffClick('white')}
              title={isWhiteBearOffTarget ? 'לחץ להוצאת כלי' : 'הוצאה (לבן)'}
              className={`flex-1 rounded-xl border p-1.5 flex flex-col items-center justify-between transition-all ${
                isWhiteBearOffTarget
                  ? 'border-emerald-400 bg-emerald-500/15 ring-2 ring-emerald-400/70 cursor-pointer animate-pulse'
                  : 'border-slate-800 bg-slate-900/60'
              }`}
            >
              {isWhiteBearOffTarget && bearOffTargetOption && (
                <div
                  dir="rtl"
                  className="px-1.5 py-0.5 rounded-full bg-emerald-400 text-slate-950 text-[10px] font-extrabold shadow text-center"
                >
                  הוצא ({bearOffTargetOption.diceBadge})
                </div>
              )}

              <div className="w-full flex flex-col-reverse items-center gap-0.5 my-1 overflow-hidden">
                {Array.from({ length: boardState.borneOff.white }).map(
                  (_, i) => (
                    <div
                      key={i}
                      className="w-9 sm:w-11 h-2 rounded-sm bg-gradient-to-r from-stone-300 via-stone-100 to-stone-300 border border-stone-400 shadow-sm"
                    />
                  )
                )}
              </div>

              <div className="text-center">
                <div
                  dir="ltr"
                  className="inline-block text-xs font-mono font-bold text-stone-100"
                >
                  {boardState.borneOff.white}
                  <span className="text-slate-500 text-[10px]">/15</span>
                </div>
                <div
                  dir="rtl"
                  className="text-[10px] font-semibold text-slate-400"
                >
                  הוצאה
                </div>
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
};
