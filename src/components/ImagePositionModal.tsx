import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  ArrowLeftRight,
  Camera,
  Check,
  Dices,
  Eraser,
  KeyRound,
  Loader2,
  PlusCircle,
  RefreshCw,
  RotateCcw,
  Sparkles,
  Trash2,
  Upload,
  Wand2,
  X,
} from 'lucide-react';
import { createInitialBoard } from '../engine/rules';
import type { BoardState, Player } from '../engine/types';
import {
  flipBoard180,
  scanBackgammonImage,
  ScanCancelledError,
  swapBoardColors,
} from '../engine/visionScanner';

export interface ImagePositionModalProps {
  isOpen: boolean;
  initialBoardState: BoardState;
  onClose: () => void;
  onConfirmPosition: (
    newState: BoardState,
    dice: [number, number] | null,
    sourceLabel: string
  ) => void;
}

type EditBrush = 'add-white' | 'add-black' | 'remove';

const TOP_LEFT_POINTS = [12, 13, 14, 15, 16, 17];
const TOP_RIGHT_POINTS = [18, 19, 20, 21, 22, 23];
const BOTTOM_LEFT_POINTS = [11, 10, 9, 8, 7, 6];
const BOTTOM_RIGHT_POINTS = [5, 4, 3, 2, 1, 0];

const STORAGE_API_KEY = 'gemini_api_key';

function createEmptyEditableBoard(): BoardState {
  return {
    points: new Array<number>(24).fill(0),
    bar: { white: 0, black: 0 },
    borneOff: { white: 0, black: 0 },
    currentTurn: 'white',
    dice: [],
    remainingMoves: [],
  };
}

export const ImagePositionModal: React.FC<ImagePositionModalProps> = ({
  isOpen,
  initialBoardState,
  onClose,
  onConfirmPosition,
}) => {
  const envKey = (
    (import.meta.env?.VITE_GEMINI_API_KEY as string | undefined) || ''
  ).trim();
  const hasEnvKey = envKey.length > 0;

  const [apiKey, setApiKey] = useState<string>(() => {
    if (hasEnvKey) return envKey;
    try {
      return localStorage.getItem(STORAGE_API_KEY) || '';
    } catch {
      return '';
    }
  });
  const [showKeyInput, setShowKeyInput] = useState<boolean>(
    () => !hasEnvKey && !apiKey
  );
  const [keySavedToast, setKeySavedToast] = useState<boolean>(false);

  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [scanElapsedSec, setScanElapsedSec] = useState<number>(0);
  const [scanError, setScanError] = useState<string | null>(null);
  const [scanNotes, setScanNotes] = useState<string | null>(null);

  const scanAbortRef = useRef<AbortController | null>(null);

  const [draftBoard, setDraftBoard] = useState<BoardState>(() => ({
    ...initialBoardState,
    points: [...initialBoardState.points],
    bar: { ...initialBoardState.bar },
    borneOff: { ...initialBoardState.borneOff },
  }));
  const [brush, setBrush] = useState<EditBrush>('add-white');
  const [selectedTurn, setSelectedTurn] = useState<Player>(
    initialBoardState.currentTurn
  );
  const [noDiceMode, setNoDiceMode] = useState<boolean>(false);
  const [die1, setDie1] = useState<number>(6);
  const [die2, setDie2] = useState<number>(4);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Sync draft board when modal opens
  useEffect(() => {
    if (isOpen) {
      setDraftBoard({
        ...initialBoardState,
        points: [...initialBoardState.points],
        bar: { ...initialBoardState.bar },
        borneOff: { ...initialBoardState.borneOff },
      });
      setSelectedTurn(initialBoardState.currentTurn);
      if (initialBoardState.dice.length >= 2) {
        setDie1(initialBoardState.dice[0]);
        setDie2(initialBoardState.dice[1]);
        setNoDiceMode(false);
      }
      setScanError(null);
    }
  }, [isOpen, initialBoardState]);

  // Support global Ctrl+V clipboard paste while modal is open
  useEffect(() => {
    if (!isOpen) return;

    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;

      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (file) {
            e.preventDefault();
            readFileToDataUrl(file);
            break;
          }
        }
      }
    };

    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [isOpen]);

  // Live elapsed-seconds timer while scanning
  useEffect(() => {
    if (!isOpen || !isScanning) {
      setScanElapsedSec(0);
      return;
    }
    const start = Date.now();
    const timer = window.setInterval(() => {
      setScanElapsedSec(Math.floor((Date.now() - start) / 1000));
    }, 500);
    return () => window.clearInterval(timer);
  }, [isOpen, isScanning]);

  // Abort any in-flight scan when the modal closes
  useEffect(() => {
    if (!isOpen && scanAbortRef.current) {
      scanAbortRef.current.abort();
      scanAbortRef.current = null;
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const readFileToDataUrl = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setScanError('נא לבחור קובץ תמונה תקין (PNG, JPG, WebP).');
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        setImageDataUrl(reader.result);
        setScanError(null);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleSaveApiKey = () => {
    try {
      localStorage.setItem(STORAGE_API_KEY, apiKey.trim());
      setKeySavedToast(true);
      setTimeout(() => setKeySavedToast(false), 2000);
      if (apiKey.trim()) {
        setShowKeyInput(false);
      }
    } catch {
      // Ignore storage errors
    }
  };

  const handleCancelScan = () => {
    if (scanAbortRef.current) {
      scanAbortRef.current.abort();
      scanAbortRef.current = null;
    }
    setIsScanning(false);
    setScanError(null);
  };

  const handleScanWithAI = async () => {
    if (!imageDataUrl) {
      setScanError('נא להעלות או להדביק תמונת לוח שש-בש תחילה.');
      return;
    }
    if (!apiKey.trim()) {
      setShowKeyInput(true);
      setScanError(
        'נא להזין מפתח Gemini API כדי להפעיל זיהוי תמונה אוטומטי, או ערוך את הלוח ידנית למטה.'
      );
      return;
    }

    scanAbortRef.current?.abort();
    const controller = new AbortController();
    scanAbortRef.current = controller;

    setIsScanning(true);
    setScanElapsedSec(0);
    setScanError(null);
    setScanNotes(null);

    try {
      try {
        localStorage.setItem(STORAGE_API_KEY, apiKey.trim());
      } catch {
        // Ignore storage errors
      }

      const scanned = await scanBackgammonImage(
        imageDataUrl,
        apiKey,
        controller.signal
      );
      if (controller.signal.aborted) return;

      setDraftBoard({
        points: [...scanned.points],
        bar: { ...scanned.bar },
        borneOff: { ...scanned.borneOff },
        currentTurn: scanned.detectedTurn ?? selectedTurn,
        dice: scanned.dice ? [...scanned.dice] : [],
        remainingMoves: scanned.dice ? [...scanned.dice] : [],
      });

      if (scanned.detectedTurn) {
        setSelectedTurn(scanned.detectedTurn);
      }
      if (scanned.dice) {
        setDie1(scanned.dice[0]);
        setDie2(scanned.dice[1]);
        setNoDiceMode(false);
      }
      setScanNotes(scanned.notes);
    } catch (err) {
      if (err instanceof ScanCancelledError || controller.signal.aborted) {
        setScanError(null);
        return;
      }
      setScanError(
        err instanceof Error
          ? err.message
          : 'אירעה שגיאה בסריקת התמונה. ניתן לכייל את הלוח ידנית.'
      );
    } finally {
      if (scanAbortRef.current === controller) {
        scanAbortRef.current = null;
      }
      setIsScanning(false);
    }
  };

  // Compute live checker counts
  const whiteOnPoints = draftBoard.points.reduce(
    (sum, v) => (v > 0 ? sum + v : sum),
    0
  );
  const blackOnPoints = draftBoard.points.reduce(
    (sum, v) => (v < 0 ? sum + Math.abs(v) : sum),
    0
  );

  const totalWhite =
    whiteOnPoints + draftBoard.bar.white + draftBoard.borneOff.white;
  const totalBlack =
    blackOnPoints + draftBoard.bar.black + draftBoard.borneOff.black;

  const isWhiteValid = totalWhite <= 15 && totalWhite > 0;
  const isBlackValid = totalBlack <= 15 && totalBlack > 0;
  const canConfirm = totalWhite <= 15 && totalBlack <= 15 && (totalWhite > 0 || totalBlack > 0);

  const handlePointEdit = (
    pointIndex: number,
    overrideAction?: EditBrush
  ) => {
    const activeBrush = overrideAction ?? brush;
    setDraftBoard((prev) => {
      const nextPoints = [...prev.points];
      const current = nextPoints[pointIndex];

      if (activeBrush === 'add-white') {
        if (current < 0) {
          nextPoints[pointIndex] = 1;
        } else if (current < 15) {
          nextPoints[pointIndex] = current + 1;
        }
      } else if (activeBrush === 'add-black') {
        if (current > 0) {
          nextPoints[pointIndex] = -1;
        } else if (current > -15) {
          nextPoints[pointIndex] = current - 1;
        }
      } else {
        if (current > 0) nextPoints[pointIndex] = current - 1;
        else if (current < 0) nextPoints[pointIndex] = current + 1;
      }

      return {
        ...prev,
        points: nextPoints,
      };
    });
  };

  const handleBarEdit = (player: Player, overrideAction?: EditBrush) => {
    const activeBrush = overrideAction ?? brush;
    setDraftBoard((prev) => {
      const nextBar = { ...prev.bar };
      if (activeBrush === 'remove') {
        nextBar[player] = Math.max(0, nextBar[player] - 1);
      } else if (
        (activeBrush === 'add-white' && player === 'white') ||
        (activeBrush === 'add-black' && player === 'black')
      ) {
        nextBar[player] = Math.min(15, nextBar[player] + 1);
      } else {
        nextBar[player] = Math.max(0, nextBar[player] - 1);
      }
      return { ...prev, bar: nextBar };
    });
  };

  const handleBorneOffEdit = (player: Player, overrideAction?: EditBrush) => {
    const activeBrush = overrideAction ?? brush;
    setDraftBoard((prev) => {
      const nextOff = { ...prev.borneOff };
      if (activeBrush === 'remove') {
        nextOff[player] = Math.max(0, nextOff[player] - 1);
      } else if (
        (activeBrush === 'add-white' && player === 'white') ||
        (activeBrush === 'add-black' && player === 'black')
      ) {
        nextOff[player] = Math.min(15, nextOff[player] + 1);
      } else {
        nextOff[player] = Math.max(0, nextOff[player] - 1);
      }
      return { ...prev, borneOff: nextOff };
    });
  };

  const handleAutoBalanceBorneOff = () => {
    setDraftBoard((prev) => {
      const wActive =
        prev.points.reduce((s, v) => (v > 0 ? s + v : s), 0) + prev.bar.white;
      const bActive =
        prev.points.reduce((s, v) => (v < 0 ? s + Math.abs(v) : s), 0) +
        prev.bar.black;
      return {
        ...prev,
        borneOff: {
          white: Math.max(0, 15 - wActive),
          black: Math.max(0, 15 - bActive),
        },
      };
    });
  };

  const handleConfirm = () => {
    if (!canConfirm) return;

    // Auto-fill borneOff if user left borneOff at 0 while fewer than 15 checkers are on board+bar
    const wActive = whiteOnPoints + draftBoard.bar.white;
    const bActive = blackOnPoints + draftBoard.bar.black;

    const finalBorneWhite =
      draftBoard.borneOff.white > 0
        ? Math.min(15 - wActive, draftBoard.borneOff.white)
        : wActive > 0 && wActive < 15
          ? 15 - wActive
          : 0;

    const finalBorneBlack =
      draftBoard.borneOff.black > 0
        ? Math.min(15 - bActive, draftBoard.borneOff.black)
        : bActive > 0 && bActive < 15
          ? 15 - bActive
          : 0;

    const finalDice: [number, number] | null = noDiceMode
      ? null
      : [die1, die2];

    const expandedDice = finalDice
      ? finalDice[0] === finalDice[1]
        ? [finalDice[0], finalDice[0], finalDice[0], finalDice[0]]
        : [finalDice[0], finalDice[1]]
      : [];

    const finalState: BoardState = {
      points: [...draftBoard.points],
      bar: { ...draftBoard.bar },
      borneOff: {
        white: finalBorneWhite,
        black: finalBorneBlack,
      },
      currentTurn: selectedTurn,
      dice: [...expandedDice],
      remainingMoves: [...expandedDice],
    };

    onConfirmPosition(
      finalState,
      finalDice,
      imageDataUrl ? 'עמדה מיובאת מתמונה' : 'עמדה מיובאת מתמונה (עריכה ידנית)'
    );
  };

  const renderEditablePoint = (pointIndex: number, isTopRow: boolean) => {
    const val = draftBoard.points[pointIndex];
    const owner: Player | null = val > 0 ? 'white' : val < 0 ? 'black' : null;
    const count = Math.abs(val);
    const visible = Math.min(5, count);
    const isEven = pointIndex % 2 === 0;
    const spikeColor = isEven ? 'bg-emerald-900/70' : 'bg-slate-700/70';

    return (
      <div
        key={pointIndex}
        onClick={() => handlePointEdit(pointIndex)}
        onContextMenu={(e) => {
          e.preventDefault();
          handlePointEdit(pointIndex, 'remove');
        }}
        title={`נקודה ${pointIndex + 1} — לחץ לעריכה (קליק ימני להסרת כלי)`}
        className="group/pt relative flex-1 flex flex-col items-center justify-between h-[135px] sm:h-[150px] cursor-pointer select-none hover:bg-cyan-500/10 transition-colors"
      >
        {isTopRow && (
          <div
            dir="ltr"
            className="w-full text-center text-[9px] font-mono text-slate-400 bg-slate-950/70 py-0.5 border-b border-slate-800"
          >
            {pointIndex + 1}
          </div>
        )}

        <div
          className={`relative w-full flex-1 flex ${
            isTopRow
              ? 'flex-col justify-start pt-1'
              : 'flex-col-reverse justify-start pb-1'
          } items-center`}
        >
          <div
            className={`absolute inset-x-0.5 ${
              isTopRow ? 'top-0 bottom-2' : 'bottom-0 top-2'
            } ${spikeColor} group-hover/pt:brightness-125 transition-all`}
            style={{
              clipPath: isTopRow
                ? 'polygon(0% 0%, 100% 0%, 50% 100%)'
                : 'polygon(50% 0%, 0% 100%, 100% 100%)',
            }}
          />

          <div
            className={`relative z-10 flex ${
              isTopRow ? 'flex-col' : 'flex-col-reverse'
            } items-center -space-y-1`}
          >
            {owner &&
              Array.from({ length: visible }).map((_, idx) => {
                const isTop = idx === visible - 1;
                const isWhite = owner === 'white';
                return (
                  <div
                    key={idx}
                    className={`w-6 h-6 sm:w-7 sm:h-7 rounded-full flex items-center justify-center text-[10px] font-mono font-extrabold border shadow ${
                      isWhite
                        ? 'bg-gradient-to-b from-stone-50 to-stone-300 border-stone-400 text-slate-950'
                        : 'bg-gradient-to-b from-slate-700 to-slate-950 border-slate-500 text-amber-300'
                    }`}
                  >
                    {isTop && count > 1 ? count : ''}
                  </div>
                );
              })}
          </div>
        </div>

        {!isTopRow && (
          <div
            dir="ltr"
            className="w-full text-center text-[9px] font-mono text-slate-400 bg-slate-950/70 py-0.5 border-t border-slate-800"
          >
            {pointIndex + 1}
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      dir="rtl"
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/85 backdrop-blur-md p-2 sm:p-4 overflow-y-auto"
    >
      <div className="relative w-full max-w-6xl rounded-2xl bg-slate-900 border border-slate-700/90 shadow-[0_25px_70px_rgba(0,0,0,0.85)] overflow-hidden my-auto max-h-[95vh] flex flex-col">
        {/* Modal Header */}
        <div className="px-4 sm:px-6 py-3.5 border-b border-slate-800 bg-slate-950/90 flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-400 to-emerald-500 flex items-center justify-center text-slate-950 shadow-lg shadow-cyan-500/20">
              <Camera className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-sm sm:text-base font-bold text-white flex items-center gap-2">
                סריקת לוח מתמונה ועורך עמדות חכם
                <span className="hidden sm:inline-block text-[10px] font-semibold px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                  Gemini Vision AI + כיול ידני
                </span>
              </h2>
              <p className="text-[11px] text-slate-400">
                העלה צילום מסך מאפליקציית שש-בש או תמונה של לוח אמיתי, כייל
                בלחיצה וקבל את המהלך המנצח עם חצים על הלוח
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors cursor-pointer"
            title="סגור"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Scrollable Modal Content */}
        <div className="p-4 sm:p-6 overflow-y-auto space-y-5 flex-1">
          {/* Top Row: Image Upload / Paste + Gemini API Key Bar */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
            {/* Image Dropzone & Preview (5 cols) */}
            <div className="lg:col-span-5 flex flex-col justify-between rounded-2xl bg-slate-950/70 border border-slate-800 p-3.5">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                  <Upload className="w-3.5 h-3.5 text-cyan-400" />
                  1. העלאת תמונת הלוח (או הדבקה ב-Ctrl+V)
                </span>
                {imageDataUrl && (
                  <button
                    type="button"
                    onClick={() => {
                      setImageDataUrl(null);
                      setScanNotes(null);
                    }}
                    className="text-[11px] text-rose-400 hover:text-rose-300 flex items-center gap-1 cursor-pointer"
                  >
                    <Trash2 className="w-3 h-3" />
                    הסר תמונה
                  </button>
                )}
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) readFileToDataUrl(file);
                }}
              />

              {imageDataUrl ? (
                <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-slate-900 flex items-center justify-center max-h-52">
                  <img
                    src={imageDataUrl}
                    alt="לוח שש-בש לסריקה"
                    className="max-h-52 w-auto object-contain"
                  />
                </div>
              ) : (
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setIsDragging(false);
                    const file = e.dataTransfer.files?.[0];
                    if (file) readFileToDataUrl(file);
                  }}
                  onClick={() => fileInputRef.current?.click()}
                  className={`flex-1 min-h-[150px] rounded-xl border-2 border-dashed flex flex-col items-center justify-center p-4 text-center cursor-pointer transition-all ${
                    isDragging
                      ? 'border-cyan-400 bg-cyan-500/10'
                      : 'border-slate-700 hover:border-cyan-500/60 bg-slate-900/50 hover:bg-slate-900'
                  }`}
                >
                  <Camera className="w-8 h-8 text-cyan-400 mb-2" />
                  <p className="text-xs font-bold text-slate-200">
                    גרור תמונה לכאן, לחץ לבחירת קובץ, או הקש{' '}
                    <kbd
                      dir="ltr"
                      className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-600 font-mono text-[10px] text-cyan-300"
                    >
                      Ctrl+V
                    </kbd>
                  </p>
                  <p className="text-[11px] text-slate-400 mt-1">
                    תומך בצילומי מסך מאפליקציות ובצילום לוח שש-בש פיזי מהטלפון
                  </p>
                </div>
              )}

              {/* Scan Action Buttons */}
              <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={handleScanWithAI}
                  disabled={isScanning || !imageDataUrl}
                  className={`flex-1 inline-flex items-center justify-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all ${
                    imageDataUrl && !isScanning
                      ? 'bg-gradient-to-r from-cyan-400 to-emerald-400 hover:from-cyan-300 hover:to-emerald-300 text-slate-950 shadow-lg shadow-cyan-500/20 cursor-pointer'
                      : 'bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed'
                  }`}
                >
                  {isScanning ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>
                        סורק לוח עם AI... (
                        <span dir="ltr" className="inline-block font-mono">
                          {scanElapsedSec}s
                        </span>
                        )
                      </span>
                    </>
                  ) : (
                    <>
                      <Wand2 className="w-4 h-4" />
                      סרוק לוח עם AI
                    </>
                  )}
                </button>

                {isScanning && (
                  <button
                    type="button"
                    onClick={handleCancelScan}
                    className="px-3 py-2 rounded-xl bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 border border-rose-500/40 text-xs font-bold transition-colors cursor-pointer"
                  >
                    ביטול
                  </button>
                )}

                <button
                  type="button"
                  onClick={() => {
                    setScanError(null);
                    setScanNotes(
                      'מצב עריכה ידנית פעיל — לחץ על הנקודות, הבר או מגשי ההוצאה בלוח כדי להציב כלים.'
                    );
                  }}
                  className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-emerald-300 border border-slate-700 text-xs font-semibold transition-colors cursor-pointer"
                >
                  ערוך לוח ידנית (ללא סריקה)
                </button>

                <button
                  type="button"
                  onClick={() => setShowKeyInput((prev) => !prev)}
                  className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold transition-colors cursor-pointer"
                  title="הגדרת מפתח Gemini API"
                >
                  <KeyRound className="w-3.5 h-3.5 text-amber-400" />
                  {hasEnvKey ? 'החלף מפתח ידנית' : 'מפתח Gemini API'}
                </button>
              </div>

              {hasEnvKey && !showKeyInput && (
                <div className="mt-2 px-2.5 py-1.5 rounded-xl bg-emerald-500/15 border border-emerald-500/35 text-emerald-300 text-[11px] font-semibold flex items-center justify-between gap-2">
                  <span className="inline-flex items-center gap-1.5">
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                    מפתח Gemini מחובר אוטומטית מקובץ .env
                  </span>
                  <button
                    type="button"
                    onClick={() => setShowKeyInput(true)}
                    className="text-[10px] underline text-emerald-200 hover:text-white cursor-pointer"
                  >
                    החלף מפתח ידנית
                  </button>
                </div>
              )}

              {/* Collapsible Gemini API Key Bar */}
              {showKeyInput && (
                <div className="mt-2.5 p-2.5 rounded-xl bg-slate-900 border border-slate-700/80 space-y-2">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="font-semibold text-slate-300">
                      מפתח Gemini API (נשמר מקומית בדפדפן):
                    </span>
                    {keySavedToast && (
                      <span className="text-emerald-400 font-bold">
                        ✓ המפתח נשמר!
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      dir="ltr"
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder="AIzaSy..."
                      className="flex-1 px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-xs font-mono text-slate-100 focus:outline-none focus:border-cyan-400"
                    />
                    <button
                      type="button"
                      onClick={handleSaveApiKey}
                      className="px-3 py-1.5 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-200 border border-cyan-500/40 text-xs font-bold cursor-pointer"
                    >
                      שמור מפתח
                    </button>
                  </div>
                </div>
              )}

              {scanError && (
                <div className="mt-2.5 p-2.5 rounded-xl bg-rose-500/15 border border-rose-500/40 text-rose-200 text-xs flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                  <span>{scanError}</span>
                </div>
              )}

              {scanNotes && (
                <div className="mt-2.5 p-2.5 rounded-xl bg-emerald-500/15 border border-emerald-500/40 text-emerald-200 text-xs flex items-start gap-2">
                  <Sparkles className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  <span>{scanNotes}</span>
                </div>
              )}
            </div>

            {/* Interactive Calibration & Click-to-Edit Board (7 cols) */}
            <div className="lg:col-span-7 flex flex-col justify-between rounded-2xl bg-slate-950/70 border border-slate-800 p-3.5">
              {/* Top Calibration Toolbar */}
              <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-slate-200">
                    2. אימות וכיול הלוח (לחץ על כל נקודה לעריכה):
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded-full text-[11px] font-bold border ${
                      isWhiteValid
                        ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40'
                        : 'bg-rose-500/20 text-rose-300 border-rose-500/50'
                    }`}
                  >
                    לבן:{' '}
                    <span dir="ltr" className="inline-block font-mono">
                      {totalWhite}/15
                    </span>
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded-full text-[11px] font-bold border ${
                      isBlackValid
                        ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40'
                        : 'bg-rose-500/20 text-rose-300 border-rose-500/50'
                    }`}
                  >
                    שחור:{' '}
                    <span dir="ltr" className="inline-block font-mono">
                      {totalBlack}/15
                    </span>
                  </span>
                </div>

                {/* Orientation & Reset Quick Buttons */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setDraftBoard((b) => flipBoard180(b))}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-[11px] font-semibold cursor-pointer"
                    title="סובב את הלוח ב-180 מעלות (החלף למעלה/למטה ושמאל/ימין)"
                  >
                    <RefreshCw className="w-3 h-3 text-cyan-400" />
                    הפוך כיוון (180°)
                  </button>
                  <button
                    type="button"
                    onClick={() => setDraftBoard((b) => swapBoardColors(b))}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-[11px] font-semibold cursor-pointer"
                    title="החלף בין הכלים הלבנים והשחורים והפוך כיוון בית"
                  >
                    <ArrowLeftRight className="w-3 h-3 text-amber-400" />
                    החלף צבעים (לבן ↔ שחור)
                  </button>
                  <button
                    type="button"
                    onClick={handleAutoBalanceBorneOff}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-emerald-300 border border-slate-700 text-[11px] font-semibold cursor-pointer"
                    title="השלם אוטומטית ל-15 כלים באמצעות מגש ההוצאה"
                  >
                    השלם הוצאה ל-15
                  </button>
                  <button
                    type="button"
                    onClick={() => setDraftBoard(createInitialBoard())}
                    className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-[11px] cursor-pointer"
                  >
                    <RotateCcw className="w-3 h-3" />
                    עמדת פתיחה
                  </button>
                  <button
                    type="button"
                    onClick={() => setDraftBoard(createEmptyEditableBoard())}
                    className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-rose-300 border border-slate-700 text-[11px] cursor-pointer"
                  >
                    <Trash2 className="w-3 h-3" />
                    נקה לוח
                  </button>
                </div>
              </div>

              {/* Brush Selector Bar */}
              <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5 px-2.5 py-1.5 rounded-xl bg-slate-900 border border-slate-800">
                <span className="text-[11px] text-slate-400 font-medium">
                  כלי עריכה בלחיצה על נקודה / בר / הוצאה:
                </span>
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setBrush('add-white')}
                    className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                      brush === 'add-white'
                        ? 'bg-stone-100 text-slate-950 shadow'
                        : 'bg-slate-800 text-stone-200 hover:bg-slate-700'
                    }`}
                  >
                    <PlusCircle className="w-3.5 h-3.5" />
                    הוסף לבן
                  </button>
                  <button
                    type="button"
                    onClick={() => setBrush('add-black')}
                    className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                      brush === 'add-black'
                        ? 'bg-amber-400 text-slate-950 shadow'
                        : 'bg-slate-800 text-slate-200 hover:bg-slate-700'
                    }`}
                  >
                    <PlusCircle className="w-3.5 h-3.5" />
                    הוסף שחור
                  </button>
                  <button
                    type="button"
                    onClick={() => setBrush('remove')}
                    className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                      brush === 'remove'
                        ? 'bg-rose-500 text-white shadow'
                        : 'bg-slate-800 text-rose-300 hover:bg-slate-700'
                    }`}
                  >
                    <Eraser className="w-3.5 h-3.5" />
                    הסר כלי
                  </button>
                </div>
              </div>

              {/* Interactive Mini Backgammon Board (dir="ltr" to match exact physical board coordinates) */}
              <div
                dir="ltr"
                className="flex rounded-xl bg-slate-950 border-2 border-slate-800 overflow-hidden shadow-inner"
              >
                <div className="flex-1 flex flex-col justify-between bg-[#0b1220]">
                  {/* Top Row: 12..17 | Bar (White) | 18..23 */}
                  <div className="flex items-stretch">
                    <div className="flex-1 flex px-1">
                      {TOP_LEFT_POINTS.map((pt) =>
                        renderEditablePoint(pt, true)
                      )}
                    </div>

                    {/* Top Bar (White Bar) */}
                    <div
                      onClick={() => handleBarEdit('white')}
                      onContextMenu={(e) => {
                        e.preventDefault();
                        handleBarEdit('white', 'remove');
                      }}
                      title="בר לבן (לחץ להוספה/הסרה)"
                      className="w-10 bg-slate-900 border-x border-slate-800 flex flex-col items-center justify-center gap-1 cursor-pointer hover:bg-slate-800/80 select-none"
                    >
                      <span className="text-[8px] text-slate-400 font-bold">
                        בר לבן
                      </span>
                      <div className="w-6 h-6 rounded-full bg-stone-100 text-slate-950 font-mono text-xs font-extrabold flex items-center justify-center border border-stone-400">
                        {draftBoard.bar.white}
                      </div>
                    </div>

                    <div className="flex-1 flex px-1 bg-slate-900/25">
                      {TOP_RIGHT_POINTS.map((pt) =>
                        renderEditablePoint(pt, true)
                      )}
                    </div>
                  </div>

                  {/* Center Strip */}
                  <div
                    dir="rtl"
                    className="h-6 bg-slate-950/80 border-y border-slate-800/80 px-3 flex items-center justify-between text-[10px] text-slate-400"
                  >
                    <span>ימין-למטה: בית הלבן (1–6)</span>
                    <span>ימין-למעלה: בית השחור (19–24)</span>
                  </div>

                  {/* Bottom Row: 11..6 | Bar (Black) | 5..0 */}
                  <div className="flex items-stretch">
                    <div className="flex-1 flex px-1">
                      {BOTTOM_LEFT_POINTS.map((pt) =>
                        renderEditablePoint(pt, false)
                      )}
                    </div>

                    {/* Bottom Bar (Black Bar) */}
                    <div
                      onClick={() => handleBarEdit('black')}
                      onContextMenu={(e) => {
                        e.preventDefault();
                        handleBarEdit('black', 'remove');
                      }}
                      title="בר שחור (לחץ להוספה/הסרה)"
                      className="w-10 bg-slate-900 border-x border-slate-800 flex flex-col items-center justify-center gap-1 cursor-pointer hover:bg-slate-800/80 select-none"
                    >
                      <div className="w-6 h-6 rounded-full bg-slate-800 text-amber-300 font-mono text-xs font-extrabold flex items-center justify-center border border-slate-500">
                        {draftBoard.bar.black}
                      </div>
                      <span className="text-[8px] text-slate-400 font-bold">
                        בר שחור
                      </span>
                    </div>

                    <div className="flex-1 flex px-1 bg-slate-900/25">
                      {BOTTOM_RIGHT_POINTS.map((pt) =>
                        renderEditablePoint(pt, false)
                      )}
                    </div>
                  </div>
                </div>

                {/* Right Bear-Off Trays */}
                <div className="w-14 bg-slate-950 border-l border-slate-800 flex flex-col justify-between p-1.5 select-none">
                  <div
                    onClick={() => handleBorneOffEdit('black')}
                    onContextMenu={(e) => {
                      e.preventDefault();
                      handleBorneOffEdit('black', 'remove');
                    }}
                    title="הוצאה שחור (לחץ להוספה/הסרה)"
                    className="flex-1 rounded-lg bg-slate-900/80 hover:bg-slate-800 border border-slate-800 flex flex-col items-center justify-center cursor-pointer p-1"
                  >
                    <span dir="rtl" className="text-[9px] text-slate-400">
                      הוצאה שחור
                    </span>
                    <span className="font-mono text-xs font-extrabold text-amber-300 mt-1">
                      {draftBoard.borneOff.black}
                    </span>
                  </div>

                  <div className="h-2" />

                  <div
                    onClick={() => handleBorneOffEdit('white')}
                    onContextMenu={(e) => {
                      e.preventDefault();
                      handleBorneOffEdit('white', 'remove');
                    }}
                    title="הוצאה לבן (לחץ להוספה/הסרה)"
                    className="flex-1 rounded-lg bg-slate-900/80 hover:bg-slate-800 border border-slate-800 flex flex-col items-center justify-center cursor-pointer p-1"
                  >
                    <span dir="rtl" className="text-[9px] text-slate-400">
                      הוצאה לבן
                    </span>
                    <span className="font-mono text-xs font-extrabold text-stone-100 mt-1">
                      {draftBoard.borneOff.white}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Row: Turn & Dice Configuration */}
          <div className="rounded-2xl bg-slate-950/80 border border-slate-800 p-4 flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
            {/* Who is on roll? */}
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-xs font-bold text-slate-200">
                3. תור מי עכשיו?
              </span>
              <div className="inline-flex rounded-xl bg-slate-900 p-1 border border-slate-800">
                <button
                  type="button"
                  onClick={() => setSelectedTurn('white')}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                    selectedTurn === 'white'
                      ? 'bg-stone-100 text-slate-950 shadow'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <span className="w-3 h-3 rounded-full bg-stone-200 border border-slate-500" />
                  תור הלבן
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedTurn('black')}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                    selectedTurn === 'black'
                      ? 'bg-amber-400 text-slate-950 shadow'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <span className="w-3 h-3 rounded-full bg-slate-900 border border-slate-400" />
                  תור השחור
                </button>
              </div>
            </div>

            {/* Dice Selection */}
            <div className="flex flex-wrap items-center gap-4">
              <label className="inline-flex items-center gap-2 text-xs font-semibold text-slate-300 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={noDiceMode}
                  onChange={(e) => setNoDiceMode(e.target.checked)}
                  className="rounded border-slate-600 text-emerald-500 focus:ring-emerald-500"
                />
                ללא קוביות (הצג רק מד יתרון והערכת עמדה)
              </label>

              {!noDiceMode && (
                <div className="flex flex-wrap items-center gap-3 bg-slate-900 px-3 py-2 rounded-xl border border-slate-800">
                  <div className="flex items-center gap-1.5">
                    <Dices className="w-4 h-4 text-emerald-400" />
                    <span className="text-xs font-bold text-slate-300">
                      קובייה 1:
                    </span>
                    <div dir="ltr" className="flex items-center gap-1">
                      {[1, 2, 3, 4, 5, 6].map((val) => (
                        <button
                          key={val}
                          type="button"
                          onClick={() => setDie1(val)}
                          className={`w-7 h-7 rounded-lg font-mono text-xs font-extrabold border transition-all cursor-pointer ${
                            die1 === val
                              ? 'bg-emerald-400 text-slate-950 border-emerald-200 scale-105 shadow'
                              : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
                          }`}
                        >
                          {val}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="h-5 w-[1px] bg-slate-800 hidden sm:block" />

                  <div className="flex items-center gap-1.5">
                    <span className="text-xs font-bold text-slate-300">
                      קובייה 2:
                    </span>
                    <div dir="ltr" className="flex items-center gap-1">
                      {[1, 2, 3, 4, 5, 6].map((val) => (
                        <button
                          key={val}
                          type="button"
                          onClick={() => setDie2(val)}
                          className={`w-7 h-7 rounded-lg font-mono text-xs font-extrabold border transition-all cursor-pointer ${
                            die2 === val
                              ? 'bg-emerald-400 text-slate-950 border-emerald-200 scale-105 shadow'
                              : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
                          }`}
                        >
                          {val}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-4 sm:px-6 py-3.5 border-t border-slate-800 bg-slate-950/90 flex flex-wrap items-center justify-between gap-3 shrink-0">
          <div className="text-xs text-slate-400">
            {totalWhite > 15 || totalBlack > 15 ? (
              <span className="text-rose-400 font-semibold">
                שים לב: מספר הכלים של אחד השחקנים עולה על 15. נא להסיר כלים
                עודפים לפני הטעינה.
              </span>
            ) : (
              <span>
                לאחר האישור, המנוע יציג מיד את מד היתרון (Eval Bar) ויצייר חצים
                ירוקים ממוספרים של המהלך הטוב ביותר על הלוח.
              </span>
            )}
          </div>

          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition-colors cursor-pointer"
            >
              ביטול
            </button>

            <button
              type="button"
              onClick={handleConfirm}
              disabled={!canConfirm}
              className={`inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs sm:text-sm font-bold transition-all ${
                canConfirm
                  ? 'bg-emerald-400 hover:bg-emerald-300 text-slate-950 shadow-lg shadow-emerald-500/25 cursor-pointer'
                  : 'bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed'
              }`}
            >
              <Check className="w-4 h-4" />
              טען עמדה ונתח עכשיו
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
