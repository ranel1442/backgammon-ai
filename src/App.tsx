import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  BarChart3,
  Bot,
  Brain,
  Camera,
  Check,
  Cpu,
  Dices,
  Download,
  FlaskConical,
  History,
  Lightbulb,
  Maximize2,
  Menu,
  Minimize2,
  Play,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  Trophy,
  Undo2,
  Users,
  X,
} from 'lucide-react';
import { AnalysisPanel } from './components/AnalysisPanel';
import { Board } from './components/Board';
import { ErrorBoundary } from './components/ErrorBoundary';
import { EvalBar } from './components/EvalBar';
import { GameReviewPanel } from './components/GameReviewPanel';
import { ImagePositionModal } from './components/ImagePositionModal';
import { MoveHistory, type HistoryEntry } from './components/MoveHistory';
import { RotatePhoneOverlay } from './components/RotatePhoneOverlay';
import type { TurnAnalysisResult } from './engine/ai';
import { evaluatePosition } from './engine/evaluator';
import { computeGameReview } from './engine/review';
import {
  applyMoveStep,
  createInitialBoard,
  formatTurnNotation,
  generateLegalTurnMoves,
  getPipCount,
  getValidDestinationsForSource,
  getValidNextSteps,
} from './engine/rules';
import type { BoardState, MoveStep, Player } from './engine/types';
import { useBackgammonAI } from './hooks/useBackgammonAI';
import { useMobileLayout } from './hooks/useMobileLayout';

type GameMode = 'ai' | 'local';

function normalizeGameMode(raw: unknown): GameMode {
  return raw === 'local' ? 'local' : 'ai';
}

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

interface WinSummary {
  winner: Player;
  winType: 'Single' | 'Gammon' | 'Backgammon';
  multiplier: 1 | 2 | 3;
}

interface LastOpponentTurnRecord {
  turnStartState: BoardState;
  resultingState: BoardState;
  dice: [number, number];
  steps: MoveStep[];
  notation: string;
}

function cloneBoardState(state: BoardState): BoardState {
  return {
    points: [...state.points],
    bar: { ...state.bar },
    borneOff: { ...state.borneOff },
    currentTurn: state.currentTurn,
    dice: [...state.dice],
    remainingMoves: [...state.remainingMoves],
  };
}

function detectGameOver(state: BoardState): WinSummary | null {
  if (state.borneOff.white >= 15) {
    const blackOff = state.borneOff.black;
    if (blackOff > 0) {
      return { winner: 'white', winType: 'Single', multiplier: 1 };
    }
    const blackInWhiteHomeOrBar =
      state.bar.black > 0 || state.points.slice(0, 6).some((p) => p < 0);
    return blackInWhiteHomeOrBar
      ? { winner: 'white', winType: 'Backgammon', multiplier: 3 }
      : { winner: 'white', winType: 'Gammon', multiplier: 2 };
  }

  if (state.borneOff.black >= 15) {
    const whiteOff = state.borneOff.white;
    if (whiteOff > 0) {
      return { winner: 'black', winType: 'Single', multiplier: 1 };
    }
    const whiteInBlackHomeOrBar =
      state.bar.white > 0 || state.points.slice(18, 24).some((p) => p > 0);
    return whiteInBlackHomeOrBar
      ? { winner: 'black', winType: 'Backgammon', multiplier: 3 }
      : { winner: 'black', winType: 'Gammon', multiplier: 2 };
  }

  return null;
}

function getHebrewVictoryTitle(win: WinSummary): string {
  const winnerHe = win.winner === 'white' ? 'הלבן ניצח' : 'השחור ניצח';
  if (win.winType === 'Backgammon') {
    return `${winnerHe} — ניצחון מארס טורקי (Backgammon)!`;
  }
  if (win.winType === 'Gammon') {
    return `${winnerHe} — ניצחון מארס (Gammon)!`;
  }
  return `${winnerHe} — ניצחון רגיל (סינגל)!`;
}

function vibrateLight(ms = 12): void {
  try {
    if (
      typeof navigator !== 'undefined' &&
      typeof navigator.vibrate === 'function'
    ) {
      navigator.vibrate(ms);
    }
  } catch {
    // Ignore unsupported environments such as iOS Safari
  }
}

export function App() {
  const [gameMode, setGameMode] = useState<GameMode>('ai');
  const turnSequenceRef = useRef(0);
  const [turnStartState, setTurnStartState] = useState<BoardState>(() =>
    createInitialBoard()
  );
  const [currentState, setCurrentState] = useState<BoardState>(() =>
    createInitialBoard()
  );
  const [rolledDice, setRolledDice] = useState<[number, number] | null>(null);
  const [stepsTaken, setStepsTaken] = useState<MoveStep[]>([]);
  const [undoStack, setUndoStack] = useState<BoardState[]>([]);
  const [selectedFrom, setSelectedFrom] = useState<number | 'bar' | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [viewingHistoryIndex, setViewingHistoryIndex] = useState<number | null>(
    null
  );
  const [showLiveHint, setShowLiveHint] = useState<boolean>(true);
  const [liveAnalysis, setLiveAnalysis] = useState<TurnAnalysisResult | null>(
    null
  );
  const [previewSteps, setPreviewSteps] = useState<MoveStep[] | null>(null);
  const [showGameReview, setShowGameReview] = useState<boolean>(false);
  const [isScannerModalOpen, setIsScannerModalOpen] = useState<boolean>(false);
  const [importedPositionLabel, setImportedPositionLabel] = useState<
    string | null
  >(null);
  const [holdAIForImportedTurn, setHoldAIForImportedTurn] =
    useState<boolean>(false);

  // Paced Opponent Playback & Replay State
  const [aiStatusBanner, setAiStatusBanner] = useState<string | null>(null);
  const [aiPlannedSteps, setAiPlannedSteps] = useState<MoveStep[] | null>(null);
  const [isAIAnimating, setIsAIAnimating] = useState<boolean>(false);
  const [lastOpponentTurn, setLastOpponentTurn] =
    useState<LastOpponentTurnRecord | null>(null);
  const [showLastOpponentArrows, setShowLastOpponentArrows] =
    useState<boolean>(false);
  const [isReplayingOpponent, setIsReplayingOpponent] =
    useState<boolean>(false);
  const [replayBoardState, setReplayBoardState] = useState<BoardState | null>(
    null
  );
  const [replayStepsTaken, setReplayStepsTaken] = useState<MoveStep[]>([]);
  const playbackTimersRef = useRef<Array<ReturnType<typeof setTimeout>>>([]);

  // "What-If" Sandbox Mode State
  const [isSandboxMode, setIsSandboxMode] = useState<boolean>(false);
  const [sandboxOriginTurnIndex, setSandboxOriginTurnIndex] = useState<
    number | null
  >(null);
  const [sandboxTurnStartState, setSandboxTurnStartState] =
    useState<BoardState>(() => createInitialBoard());
  const [sandboxCurrentState, setSandboxCurrentState] = useState<BoardState>(
    () => createInitialBoard()
  );
  const [sandboxRolledDice, setSandboxRolledDice] = useState<
    [number, number] | null
  >(null);
  const [sandboxStepsTaken, setSandboxStepsTaken] = useState<MoveStep[]>([]);
  const [sandboxUndoStack, setSandboxUndoStack] = useState<BoardState[]>([]);
  const [sandboxAnalysis, setSandboxAnalysis] =
    useState<TurnAnalysisResult | null>(null);
  const [deferredInstallPrompt, setDeferredInstallPrompt] =
    useState<BeforeInstallPromptEvent | null>(null);

  // Mobile Landscape Cockpit & Drawer State
  const { isPortrait, isMobileLandscape, isCompactMobile } = useMobileLayout();
  const [mobileDrawer, setMobileDrawer] = useState<
    'analysis' | 'history' | null
  >(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [rotatePromptDismissed, setRotatePromptDismissed] = useState<boolean>(
    () => {
      try {
        return sessionStorage.getItem('rotate_prompt_dismissed') === '1';
      } catch {
        return false;
      }
    }
  );

  const {
    isThinking,
    thinkingElapsedMs,
    computeAIMove,
    analyzeLivePosition,
    enqueueTurnAnalysis,
    pendingAnalysisCount: workerQueueCount,
    clearAnalysisQueue,
    isRollingOut,
    rolloutProgress,
    rolloutResults,
    runRollout,
    clearRolloutResults,
    nnLoaded,
  } = useBackgammonAI();

  useEffect(() => {
    const onBeforeInstallPrompt = (e: Event) => {
      e.preventDefault();
      setDeferredInstallPrompt(e as BeforeInstallPromptEvent);
    };
    window.addEventListener('beforeinstallprompt', onBeforeInstallPrompt);
    return () =>
      window.removeEventListener('beforeinstallprompt', onBeforeInstallPrompt);
  }, []);

  useEffect(() => {
    const onFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener('fullscreenchange', onFullscreenChange);
    return () =>
      document.removeEventListener('fullscreenchange', onFullscreenChange);
  }, []);

  const canUseFullscreen =
    typeof document !== 'undefined' &&
    Boolean(document.fullscreenEnabled) &&
    typeof document.documentElement?.requestFullscreen === 'function';

  const handleToggleFullscreen = async () => {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await document.documentElement.requestFullscreen();
      }
    } catch {
      // Ignore browser fullscreen restrictions
    }
  };

  const handleDismissRotatePrompt = () => {
    setRotatePromptDismissed(true);
    try {
      sessionStorage.setItem('rotate_prompt_dismissed', '1');
    } catch {
      // Ignore storage errors
    }
  };

  const handleInstallPwa = async () => {
    if (!deferredInstallPrompt) return;
    await deferredInstallPrompt.prompt();
    await deferredInstallPrompt.userChoice.catch(() => null);
    setDeferredInstallPrompt(null);
  };

  const clearPlaybackTimers = () => {
    for (const t of playbackTimersRef.current) {
      clearTimeout(t);
    }
    playbackTimersRef.current = [];
  };

  useEffect(() => {
    return () => clearPlaybackTimers();
  }, []);

  const isViewingHistory = viewingHistoryIndex !== null && !isSandboxMode;
  const inspectedEntry: HistoryEntry | null =
    viewingHistoryIndex !== null && history[viewingHistoryIndex]
      ? history[viewingHistoryIndex]
      : null;

  const activeTurnStartState = isSandboxMode
    ? sandboxTurnStartState
    : turnStartState;
  const activeCurrentState = isSandboxMode ? sandboxCurrentState : currentState;
  const activeRolledDice = isSandboxMode ? sandboxRolledDice : rolledDice;
  const activeStepsTaken = isSandboxMode ? sandboxStepsTaken : stepsTaken;
  const activeUndoStack = isSandboxMode ? sandboxUndoStack : undoStack;

  const displayedState: BoardState = isReplayingOpponent && replayBoardState
    ? replayBoardState
    : isSandboxMode
      ? sandboxCurrentState
      : inspectedEntry
        ? inspectedEntry.resultingState
        : currentState;

  const gameOver = useMemo(() => detectGameOver(currentState), [currentState]);
  const sandboxGameOver = useMemo(
    () => (isSandboxMode ? detectGameOver(sandboxCurrentState) : null),
    [isSandboxMode, sandboxCurrentState]
  );
  const activeGameOver = isSandboxMode ? sandboxGameOver : gameOver;

  useEffect(() => {
    if (gameOver) {
      setShowGameReview(true);
    }
  }, [gameOver]);

  const handleBackgroundTurnAnalyzed = useCallback(
    (turnId: string, analysisRes: TurnAnalysisResult) => {
      setHistory((prev) =>
        prev.map((item) => {
          if (item.turnId !== turnId) return item;
          const playedEval = analysisRes.playedMoveAnalysis;
          const refinedEq = playedEval?.equity ?? item.evalResult.equity;
          const refinedWinProb =
            refinedEq === 0
              ? 50
              : Number(
                  Math.max(
                    0.5,
                    Math.min(99.5, 100 / (1 + Math.exp(-refinedEq * 1.18)))
                  ).toFixed(1)
                );
          return {
            ...item,
            isAnalyzing: false,
            classification: playedEval?.classification ?? 'Best',
            equityLoss: playedEval?.equityLoss ?? 0,
            playedEvaluation: playedEval,
            analysis: analysisRes,
            evalResult: {
              ...item.evalResult,
              equity: refinedEq,
              winProb: refinedWinProb,
            },
          };
        })
      );
    },
    []
  );

  // Queue any unanalyzed imported turns on `analysisWorker` when Game Review is opened
  useEffect(() => {
    if (!showGameReview) return;

    history.forEach((entry) => {
      if (!entry.analysis && !entry.classification && !entry.isAnalyzing) {
        const tid = entry.turnId ?? `turn-import-${entry.id}`;
        setHistory((prev) =>
          prev.map((item) =>
            item.id === entry.id
              ? { ...item, turnId: tid, isAnalyzing: true }
              : item
          )
        );
        void enqueueTurnAnalysis(
          tid,
          entry.turnStartState,
          entry.dice,
          entry.steps,
          handleBackgroundTurnAnalyzed
        );
      }
    });
  }, [
    showGameReview,
    history,
    enqueueTurnAnalysis,
    handleBackgroundTurnAnalyzed,
  ]);

  const gameReviewSummary = useMemo(
    () => computeGameReview(history),
    [history]
  );

  const pendingAnalysisCount = useMemo(() => {
    const historyPending = history.filter(
      (h) => Boolean(h.isAnalyzing) || !h.classification
    ).length;
    return Math.max(workerQueueCount, historyPending);
  }, [history, workerQueueCount]);

  const legalTurnMoves = useMemo(() => {
    if (!activeRolledDice || activeGameOver) return [];
    return generateLegalTurnMoves(activeTurnStartState, activeRolledDice);
  }, [activeTurnStartState, activeRolledDice, activeGameOver]);

  const validNextSteps = useMemo(() => {
    if (
      !activeRolledDice ||
      activeGameOver ||
      isViewingHistory ||
      isReplayingOpponent
    ) {
      return [];
    }
    return getValidNextSteps(
      activeTurnStartState,
      activeRolledDice,
      activeStepsTaken
    );
  }, [
    activeTurnStartState,
    activeRolledDice,
    activeStepsTaken,
    activeGameOver,
    isViewingHistory,
    isReplayingOpponent,
  ]);

  const validDestinationOptions = useMemo(() => {
    if (
      selectedFrom === null ||
      !activeRolledDice ||
      activeGameOver ||
      isViewingHistory ||
      isReplayingOpponent
    ) {
      return [];
    }
    return getValidDestinationsForSource(
      activeTurnStartState,
      activeCurrentState,
      activeRolledDice,
      activeStepsTaken,
      selectedFrom
    );
  }, [
    selectedFrom,
    activeTurnStartState,
    activeCurrentState,
    activeRolledDice,
    activeStepsTaken,
    activeGameOver,
    isViewingHistory,
    isReplayingOpponent,
  ]);

  const evalResult = useMemo(() => {
    void nnLoaded;
    if (isSandboxMode || isReplayingOpponent) {
      return evaluatePosition(displayedState);
    }
    if (inspectedEntry) {
      return inspectedEntry.evalResult;
    }
    if (stepsTaken.length === 0 && history.length > 0) {
      // Uses instant <2ms 1-ply eval immediately on turn commit, then smoothly
      // refines to Depth 3+ equity if background analysis finishes before the next move!
      return history[history.length - 1].evalResult;
    }
    return evaluatePosition(displayedState);
  }, [
    displayedState,
    nnLoaded,
    isSandboxMode,
    isReplayingOpponent,
    inspectedEntry,
    stepsTaken.length,
    history,
  ]);
  const whitePips = useMemo(
    () => getPipCount(displayedState, 'white'),
    [displayedState]
  );
  const blackPips = useMemo(
    () => getPipCount(displayedState, 'black'),
    [displayedState]
  );

  const maxStepsRequired =
    legalTurnMoves.length > 0 ? legalTurnMoves[0].steps.length : 0;
  const hasNoLegalMoves =
    activeRolledDice !== null &&
    legalTurnMoves.length === 0 &&
    !activeGameOver;
  const isTurnComplete =
    activeRolledDice !== null &&
    maxStepsRequired > 0 &&
    (activeStepsTaken.length === maxStepsRequired || Boolean(activeGameOver));

  const isAITurn =
    !isSandboxMode &&
    !holdAIForImportedTurn &&
    gameMode === 'ai' &&
    currentState.currentTurn === 'black' &&
    !gameOver &&
    !isViewingHistory;

  useEffect(() => {
    if (
      !isAITurn &&
      !isViewingHistory &&
      !isReplayingOpponent &&
      validNextSteps.length > 0 &&
      validNextSteps.every((s) => s.from === 'bar')
    ) {
      setSelectedFrom('bar');
    }
  }, [validNextSteps, isAITurn, isViewingHistory, isReplayingOpponent]);

  useEffect(() => {
    if (isSandboxMode || !rolledDice || isAITurn || gameOver || !showLiveHint) {
      if (!isSandboxMode && (!rolledDice || isAITurn || gameOver)) {
        setLiveAnalysis(null);
      }
      return;
    }

    let cancelled = false;
    analyzeLivePosition(turnStartState, rolledDice).then((res) => {
      if (!cancelled) {
        setLiveAnalysis(res);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [
    isSandboxMode,
    turnStartState,
    rolledDice,
    isAITurn,
    gameOver,
    showLiveHint,
    analyzeLivePosition,
  ]);

  useEffect(() => {
    if (!isSandboxMode || !sandboxRolledDice || sandboxGameOver) {
      setSandboxAnalysis(null);
      return;
    }

    let cancelled = false;
    analyzeLivePosition(sandboxTurnStartState, sandboxRolledDice).then(
      (res) => {
        if (!cancelled) {
          setSandboxAnalysis(res);
        }
      }
    );

    return () => {
      cancelled = true;
    };
  }, [
    isSandboxMode,
    sandboxTurnStartState,
    sandboxRolledDice,
    sandboxGameOver,
    analyzeLivePosition,
  ]);

  const handleEnterSandboxAtTurn = (turnIndex: number) => {
    const entry = history[turnIndex];
    if (!entry) return;

    clearPlaybackTimers();
    setIsAIAnimating(false);
    setIsReplayingOpponent(false);
    setReplayBoardState(null);
    setAiStatusBanner(null);

    const startClone = cloneBoardState(entry.turnStartState);
    setViewingHistoryIndex(turnIndex);
    setIsSandboxMode(true);
    setSandboxOriginTurnIndex(turnIndex);
    setSandboxTurnStartState(startClone);
    setSandboxCurrentState(startClone);
    setSandboxRolledDice([entry.dice[0], entry.dice[1]]);
    setSandboxStepsTaken([]);
    setSandboxUndoStack([]);
    setSelectedFrom(null);
    setPreviewSteps(null);
  };

  const handleResetSandboxPosition = () => {
    if (sandboxOriginTurnIndex === null) return;
    const entry = history[sandboxOriginTurnIndex];
    if (!entry) return;

    const startClone = cloneBoardState(entry.turnStartState);
    setSandboxTurnStartState(startClone);
    setSandboxCurrentState(startClone);
    setSandboxRolledDice([entry.dice[0], entry.dice[1]]);
    setSandboxStepsTaken([]);
    setSandboxUndoStack([]);
    setSelectedFrom(null);
    setPreviewSteps(null);
  };

  const handleExitSandbox = () => {
    setIsSandboxMode(false);
    setSandboxOriginTurnIndex(null);
    setSandboxAnalysis(null);
    setSelectedFrom(null);
    setPreviewSteps(null);
  };

  const startNewGame = (mode: GameMode = gameMode) => {
    clearPlaybackTimers();
    clearAnalysisQueue();
    const initial = createInitialBoard();
    setGameMode(mode);
    setTurnStartState(initial);
    setCurrentState(initial);
    setRolledDice(null);
    setStepsTaken([]);
    setUndoStack([]);
    setSelectedFrom(null);
    setHistory([]);
    setViewingHistoryIndex(null);
    setLiveAnalysis(null);
    setPreviewSteps(null);
    setShowGameReview(false);
    setIsSandboxMode(false);
    setSandboxOriginTurnIndex(null);
    setAiStatusBanner(null);
    setAiPlannedSteps(null);
    setIsAIAnimating(false);
    setLastOpponentTurn(null);
    setShowLastOpponentArrows(false);
    setIsReplayingOpponent(false);
    setReplayBoardState(null);
    setImportedPositionLabel(null);
    setHoldAIForImportedTurn(false);
  };

  const handleConfirmScannedPosition = (
    newState: BoardState,
    dice: [number, number] | null,
    sourceLabel: string
  ) => {
    clearPlaybackTimers();
    clearAnalysisQueue();
    setIsScannerModalOpen(false);
    setIsSandboxMode(false);
    setSandboxOriginTurnIndex(null);
    setViewingHistoryIndex(null);
    setShowGameReview(false);
    setHistory([]);
    setTurnStartState(cloneBoardState(newState));
    setCurrentState(cloneBoardState(newState));
    setRolledDice(dice);
    setStepsTaken([]);
    setUndoStack([]);
    setSelectedFrom(null);
    setPreviewSteps(null);
    setLastOpponentTurn(null);
    setShowLastOpponentArrows(false);
    setIsReplayingOpponent(false);
    setReplayBoardState(null);
    setAiPlannedSteps(null);
    setIsAIAnimating(false);
    setAiStatusBanner(null);
    setImportedPositionLabel(sourceLabel || 'עמדה מיובאת מתמונה');
    // Pause AI auto-execution on the imported turn if Black is on roll so the user can study Black's optimal move arrows first
    setHoldAIForImportedTurn(newState.currentTurn === 'black');

    if (dice) {
      setShowLiveHint(true);
    } else {
      setLiveAnalysis(null);
    }
  };

  const handleRollDice = () => {
    if (
      (!isSandboxMode && activeRolledDice !== null) ||
      activeGameOver ||
      isViewingHistory ||
      isReplayingOpponent
    ) {
      return;
    }

    vibrateLight(12);
    const d1 = Math.floor(Math.random() * 6) + 1;
    const d2 = Math.floor(Math.random() * 6) + 1;
    const dicePair: [number, number] = [d1, d2];
    const expanded = d1 === d2 ? [d1, d1, d1, d1] : [d1, d2];

    const baseState =
      isSandboxMode && sandboxStepsTaken.length > 0
        ? sandboxTurnStartState
        : activeCurrentState;

    const stateWithDice: BoardState = {
      ...baseState,
      points: [...baseState.points],
      bar: { ...baseState.bar },
      borneOff: { ...baseState.borneOff },
      dice: [...expanded],
      remainingMoves: [...expanded],
    };

    if (isSandboxMode) {
      setSandboxTurnStartState(stateWithDice);
      setSandboxCurrentState(stateWithDice);
      setSandboxRolledDice(dicePair);
      setSandboxStepsTaken([]);
      setSandboxUndoStack([]);
    } else {
      // Clear persistent opponent arrows once the human rolls their next dice
      setShowLastOpponentArrows(false);
      setTurnStartState(stateWithDice);
      setCurrentState(stateWithDice);
      setRolledDice(dicePair);
      setStepsTaken([]);
      setUndoStack([]);
    }
    setSelectedFrom(null);
    setPreviewSteps(null);
  };

  const handleExecuteStep = (step: MoveStep) => {
    if (
      isViewingHistory ||
      isAITurn ||
      isReplayingOpponent ||
      !activeRolledDice
    ) {
      return;
    }

    if (isSandboxMode) {
      setSandboxUndoStack((prev) => [...prev, sandboxCurrentState]);
      const nextState = applyMoveStep(sandboxCurrentState, step);
      setSandboxCurrentState(nextState);
      setSandboxStepsTaken((prev) => [...prev, step]);
    } else {
      setUndoStack((prev) => [...prev, currentState]);
      const nextState = applyMoveStep(currentState, step);
      setCurrentState(nextState);
      setStepsTaken((prev) => [...prev, step]);
    }
    setSelectedFrom(null);
    setPreviewSteps(null);
  };

  const handleExecuteStepSequence = (steps: MoveStep[]) => {
    if (
      steps.length === 0 ||
      isViewingHistory ||
      isAITurn ||
      isReplayingOpponent ||
      !activeRolledDice
    ) {
      return;
    }

    if (isSandboxMode) {
      let stateCursor = sandboxCurrentState;
      const undoAdditions: BoardState[] = [];
      for (const s of steps) {
        undoAdditions.push(stateCursor);
        stateCursor = applyMoveStep(stateCursor, s);
      }
      setSandboxUndoStack((prev) => [...prev, ...undoAdditions]);
      setSandboxCurrentState(stateCursor);
      setSandboxStepsTaken((prev) => [...prev, ...steps]);
    } else {
      let stateCursor = currentState;
      const undoAdditions: BoardState[] = [];
      for (const s of steps) {
        undoAdditions.push(stateCursor);
        stateCursor = applyMoveStep(stateCursor, s);
      }
      setUndoStack((prev) => [...prev, ...undoAdditions]);
      setCurrentState(stateCursor);
      setStepsTaken((prev) => [...prev, ...steps]);
    }
    setSelectedFrom(null);
    setPreviewSteps(null);
  };

  const handleApplyRecommendedMove = (recommendedSteps: MoveStep[]) => {
    if (
      isViewingHistory ||
      isAITurn ||
      isReplayingOpponent ||
      !activeRolledDice
    ) {
      return;
    }
    const newStack: BoardState[] = [];
    let stateCursor = activeTurnStartState;

    for (const step of recommendedSteps) {
      newStack.push(stateCursor);
      stateCursor = applyMoveStep(stateCursor, step);
    }

    if (isSandboxMode) {
      setSandboxUndoStack(newStack);
      setSandboxCurrentState(stateCursor);
      setSandboxStepsTaken(recommendedSteps);
    } else {
      setUndoStack(newStack);
      setCurrentState(stateCursor);
      setStepsTaken(recommendedSteps);
    }
    setSelectedFrom(null);
    setPreviewSteps(null);
  };

  const handleUndoStep = () => {
    if (
      activeUndoStack.length === 0 ||
      isViewingHistory ||
      isAITurn ||
      isReplayingOpponent
    ) {
      return;
    }
    const previousState = activeUndoStack[activeUndoStack.length - 1];

    if (isSandboxMode) {
      setSandboxCurrentState(previousState);
      setSandboxUndoStack((prev) => prev.slice(0, -1));
      setSandboxStepsTaken((prev) => prev.slice(0, -1));
    } else {
      setCurrentState(previousState);
      setUndoStack((prev) => prev.slice(0, -1));
      setStepsTaken((prev) => prev.slice(0, -1));
    }
    setSelectedFrom(null);
  };

  const commitTurnWithSteps = (
    player: Player,
    dice: [number, number],
    steps: MoveStep[],
    startStateForTurn: BoardState,
    finalStateBeforeTurnSwitch: BoardState,
    precomputedAnalysis?: TurnAnalysisResult
  ) => {
    const nextPlayer: Player = player === 'white' ? 'black' : 'white';
    const committedState: BoardState = {
      ...finalStateBeforeTurnSwitch,
      points: [...finalStateBeforeTurnSwitch.points],
      bar: { ...finalStateBeforeTurnSwitch.bar },
      borneOff: { ...finalStateBeforeTurnSwitch.borneOff },
      currentTurn: nextPlayer,
      dice: [],
      remainingMoves: [],
    };

    if (isSandboxMode) {
      setSandboxTurnStartState(committedState);
      setSandboxCurrentState(committedState);
      setSandboxRolledDice(null);
      setSandboxStepsTaken([]);
      setSandboxUndoStack([]);
      setSandboxAnalysis(null);
      setSelectedFrom(null);
      setPreviewSteps(null);
      return;
    }

    const notation = formatTurnNotation(player, dice, steps);

    if (player === 'black') {
      setLastOpponentTurn({
        turnStartState: cloneBoardState(startStateForTurn),
        resultingState: cloneBoardState(committedState),
        dice: [dice[0], dice[1]],
        steps: [...steps],
        notation,
      });
      setShowLastOpponentArrows(steps.length > 0);
    }

    const instantEval = evaluatePosition(committedState);
    const entryId = ++turnSequenceRef.current;
    const turnId = `turn-${Date.now()}-${entryId}`;

    const initialEntry: HistoryEntry = {
      id: entryId,
      turnId,
      turnNumber: history.length + 1,
      player,
      dice: [dice[0], dice[1]],
      steps: [...steps],
      notation,
      turnStartState: cloneBoardState(startStateForTurn),
      resultingState: cloneBoardState(committedState),
      evalResult: instantEval,
      isAnalyzing: !precomputedAnalysis,
      classification: precomputedAnalysis?.playedMoveAnalysis?.classification,
      equityLoss: precomputedAnalysis?.playedMoveAnalysis?.equityLoss,
      playedEvaluation: precomputedAnalysis?.playedMoveAnalysis,
      analysis: precomputedAnalysis,
    };

    setHistory((prev) => [...prev, initialEntry]);

    // Dispatch player's turn into dedicated background `analysisWorker` FIFO queue (zero gameplay blocking!)
    if (!precomputedAnalysis) {
      void enqueueTurnAnalysis(
        turnId,
        cloneBoardState(startStateForTurn),
        [dice[0], dice[1]],
        [...steps],
        handleBackgroundTurnAnalyzed
      );
    }

    setStepsTaken([]);
    setUndoStack([]);
    setSelectedFrom(null);
    setLiveAnalysis(null);
    setPreviewSteps(null);
    setAiPlannedSteps(null);
    setIsAIAnimating(false);
    setHoldAIForImportedTurn(false);

    // Simultaneously, with ZERO delay, roll AI's dice immediately when transitioning to AI turn
    const isNextAITurn =
      nextPlayer === 'black' &&
      gameMode === 'ai' &&
      !detectGameOver(committedState);

    if (isNextAITurn) {
      const d1 = Math.floor(Math.random() * 6) + 1;
      const d2 = Math.floor(Math.random() * 6) + 1;
      const aiDice: [number, number] = [d1, d2];
      const expanded = d1 === d2 ? [d1, d1, d1, d1] : [d1, d2];
      const aiStateWithDice: BoardState = {
        ...committedState,
        points: [...committedState.points],
        bar: { ...committedState.bar },
        borneOff: { ...committedState.borneOff },
        dice: [...expanded],
        remainingMoves: [...expanded],
      };
      setShowLastOpponentArrows(false);
      setTurnStartState(aiStateWithDice);
      setCurrentState(aiStateWithDice);
      setRolledDice(aiDice);
      setAiStatusBanner(
        `היריב (שחור) הטיל ${d1}-${d2} — מחשב מהלך...`
      );
    } else {
      setTurnStartState(committedState);
      setCurrentState(committedState);
      setRolledDice(null);
      setAiStatusBanner(null);
    }
  };

  const handleConfirmTurn = () => {
    if (
      !activeRolledDice ||
      !isTurnComplete ||
      isViewingHistory ||
      isReplayingOpponent
    ) {
      return;
    }
    vibrateLight(12);
    commitTurnWithSteps(
      activeCurrentState.currentTurn,
      activeRolledDice,
      activeStepsTaken,
      activeTurnStartState,
      activeCurrentState
    );
  };

  const handlePassTurn = () => {
    if (
      !activeRolledDice ||
      !hasNoLegalMoves ||
      isViewingHistory ||
      isReplayingOpponent
    ) {
      return;
    }
    vibrateLight(12);
    commitTurnWithSteps(
      activeCurrentState.currentTurn,
      activeRolledDice,
      [],
      activeTurnStartState,
      activeCurrentState
    );
  };

  // Replay Opponent's Last Turn step-by-step on demand
  const handleReplayOpponentTurn = () => {
    if (!lastOpponentTurn || isAITurn || isReplayingOpponent) return;

    clearPlaybackTimers();
    setViewingHistoryIndex(null);
    setIsSandboxMode(false);
    setIsReplayingOpponent(true);
    setShowLastOpponentArrows(true);

    const startBoard = cloneBoardState(lastOpponentTurn.turnStartState);
    setReplayBoardState(startBoard);
    setReplayStepsTaken([]);
    setAiPlannedSteps(lastOpponentTurn.steps);
    setAiStatusBanner(
      `שידור חוזר — היריב (שחור) משחק: ${lastOpponentTurn.notation}`
    );

    const steps = lastOpponentTurn.steps;
    let cursor = cloneBoardState(startBoard);

    steps.forEach((step, idx) => {
      const stepTimer = setTimeout(
        () => {
          cursor = applyMoveStep(cursor, step);
          setReplayBoardState(cloneBoardState(cursor));
          setReplayStepsTaken(steps.slice(0, idx + 1));
        },
        650 + (idx + 1) * 450
      );
      playbackTimersRef.current.push(stepTimer);
    });

    const finishTimer = setTimeout(
      () => {
        setIsReplayingOpponent(false);
        setReplayBoardState(null);
        setReplayStepsTaken([]);
        setAiPlannedSteps(null);
        setAiStatusBanner(null);
      },
      650 + steps.length * 450 + 400
    );
    playbackTimersRef.current.push(finishTimer);
  };

  const handleExportMatch = () => {
    const payload = {
      version: 1,
      exportedAt: new Date().toISOString(),
      gameMode,
      currentState,
      history,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `backgammon-match-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImportMatch = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result));
        if (Array.isArray(parsed.history) && parsed.currentState) {
          clearPlaybackTimers();
          clearAnalysisQueue();
          setHistory(parsed.history);
          setCurrentState(parsed.currentState);
          setTurnStartState(parsed.currentState);
          if (parsed.gameMode) {
            setGameMode(normalizeGameMode(parsed.gameMode));
          }
          setRolledDice(null);
          setStepsTaken([]);
          setUndoStack([]);
          setViewingHistoryIndex(null);
          setIsSandboxMode(false);
          setShowGameReview(true);
        }
      } catch {
        // Ignore malformed JSON file
      }
    };
    reader.readAsText(file);
  };

  // Instant AI Opponent Turn Execution on Dedicated `gameplayWorker` (Depth 3+):
  // - Rolls dice immediately with zero artificial delay if not already rolled
  // - Computes move on `gameplayWorker` concurrently while `analysisWorker` reviews human move
  // - Shows planned move arrows (650ms) and animates checkers step-by-step (450ms/step)
  useEffect(() => {
    if (!isAITurn) return;

    if (rolledDice === null) {
      handleRollDice();
      return;
    }

    let cancelled = false;
    setAiStatusBanner(
      `היריב (שחור) הטיל ${rolledDice[0]}-${rolledDice[1]} — מחשב מהלך...`
    );

    computeAIMove(turnStartState, rolledDice).then((analysisRes) => {
      if (cancelled) return;

      if (analysisRes.topMoves.length === 0) {
        setIsAIAnimating(true);
        setAiStatusBanner(
          `היריב (שחור): ${rolledDice[0]}-${rolledDice[1]} — אין מהלכים חוקיים`
        );
        const passTimer = setTimeout(() => {
          if (cancelled) return;
          setIsAIAnimating(false);
          commitTurnWithSteps(
            'black',
            rolledDice,
            [],
            turnStartState,
            currentState
          );
        }, 650);
        playbackTimersRef.current.push(passTimer);
        return;
      }

      const bestChoice = analysisRes.topMoves[0];
      const chosenSteps = bestChoice.move.steps;
      const fullNotation = `${rolledDice[0]}-${rolledDice[1]}: ${bestChoice.notation}`;

      // Stage 2: Show planned move arrows and banner for 650ms before moving checkers
      setIsAIAnimating(true);
      setAiPlannedSteps(chosenSteps);
      setAiStatusBanner(`היריב (שחור) משחק: ${fullNotation}`);

      const enrichedAnalysis: TurnAnalysisResult = {
        ...analysisRes,
        playedMoveAnalysis: {
          notation: bestChoice.notation,
          equity: bestChoice.equity,
          equityLoss: 0,
          classification: 'Best',
          explanation: bestChoice.explanation,
        },
      };

      // Stage 3: Execute each MoveStep one by one (450ms per step)
      let stepStateCursor = cloneBoardState(turnStartState);
      chosenSteps.forEach((step, idx) => {
        const stepTimer = setTimeout(
          () => {
            if (cancelled) return;
            stepStateCursor = applyMoveStep(stepStateCursor, step);
            setCurrentState(cloneBoardState(stepStateCursor));
            setStepsTaken(chosenSteps.slice(0, idx + 1));
          },
          650 + (idx + 1) * 450
        );
        playbackTimersRef.current.push(stepTimer);
      });

      const commitTimer = setTimeout(
        () => {
          if (cancelled) return;
          commitTurnWithSteps(
            'black',
            rolledDice,
            chosenSteps,
            turnStartState,
            bestChoice.move.resultingState,
            enrichedAnalysis
          );
        },
        650 + chosenSteps.length * 450 + 350
      );
      playbackTimersRef.current.push(commitTimer);
    });

    return () => {
      cancelled = true;
      setIsAIAnimating(false);
      clearPlaybackTimers();
    };
  }, [isAITurn, rolledDice, turnStartState, computeAIMove]);

  const lastTurnSteps = useMemo(() => {
    if (isReplayingOpponent && lastOpponentTurn) {
      return replayStepsTaken;
    }
    if (isSandboxMode) {
      return sandboxStepsTaken;
    }
    if (inspectedEntry) {
      return inspectedEntry.steps;
    }
    if (stepsTaken.length > 0) {
      return stepsTaken;
    }
    return history.length > 0 ? history[history.length - 1].steps : [];
  }, [
    isReplayingOpponent,
    lastOpponentTurn,
    replayStepsTaken,
    isSandboxMode,
    sandboxStepsTaken,
    inspectedEntry,
    stepsTaken,
    history,
  ]);

  const panelAnalysis: TurnAnalysisResult | null = isSandboxMode
    ? sandboxAnalysis
    : isViewingHistory
      ? inspectedEntry?.analysis || null
      : liveAnalysis;

  const panelDice: [number, number] | null = isSandboxMode
    ? sandboxRolledDice
    : isViewingHistory
      ? inspectedEntry?.dice || null
      : rolledDice;

  const panelPlayer: Player = isSandboxMode
    ? sandboxCurrentState.currentTurn
    : isViewingHistory
      ? inspectedEntry?.player || 'white'
      : currentState.currentTurn;

  const panelStartState: BoardState = isSandboxMode
    ? sandboxTurnStartState
    : isViewingHistory && inspectedEntry
      ? inspectedEntry.turnStartState
      : turnStartState;

  useEffect(() => {
    clearRolloutResults();
  }, [panelAnalysis, clearRolloutResults]);

  const handleRunRollout = () => {
    if (!panelAnalysis || !panelDice || panelAnalysis.topMoves.length === 0) {
      return;
    }
    const candidateMoves = panelAnalysis.topMoves.map((c) => c.move);
    void runRollout(panelStartState, panelDice, candidateMoves, 1000);
  };

  // Compute Primary & Secondary Tactical SVG Arrows for the Board
  const {
    primaryArrowSteps,
    primaryArrowPlayer,
    secondaryArrowSteps,
    secondaryArrowPlayer,
    secondaryArrowVariant,
  } = useMemo(() => {
    // 1. Historical Turn Inspection Mode
    if (isViewingHistory && inspectedEntry) {
      const bestMoveSteps =
        inspectedEntry.analysis?.topMoves[0]?.move.steps ?? null;
      const isSuboptimal =
        inspectedEntry.classification &&
        inspectedEntry.classification !== 'Best' &&
        inspectedEntry.classification !== 'Good';

      return {
        primaryArrowSteps: bestMoveSteps,
        primaryArrowPlayer: inspectedEntry.player,
        secondaryArrowSteps: isSuboptimal ? inspectedEntry.steps : null,
        secondaryArrowPlayer: inspectedEntry.player,
        secondaryArrowVariant: 'mistake' as const,
      };
    }

    // 2. Sandbox Mode
    if (isSandboxMode) {
      const bestSandboxSteps =
        showLiveHint && sandboxStepsTaken.length === 0
          ? (sandboxAnalysis?.topMoves[0]?.move.steps ?? null)
          : null;
      return {
        primaryArrowSteps: bestSandboxSteps,
        primaryArrowPlayer: sandboxCurrentState.currentTurn,
        secondaryArrowSteps: null,
        secondaryArrowPlayer: 'black' as Player,
        secondaryArrowVariant: 'opponent' as const,
      };
    }

    // 3. Live Play Mode (including AI Turn / Replay / Human Turn)
    const bestLiveSteps =
      !isAITurn &&
      !isReplayingOpponent &&
      showLiveHint &&
      stepsTaken.length === 0
        ? (liveAnalysis?.topMoves[0]?.move.steps ?? null)
        : null;

    const opponentSteps =
      aiPlannedSteps ??
      (showLastOpponentArrows && lastOpponentTurn
        ? lastOpponentTurn.steps
        : null);

    return {
      primaryArrowSteps: bestLiveSteps,
      primaryArrowPlayer: currentState.currentTurn,
      secondaryArrowSteps: opponentSteps,
      secondaryArrowPlayer: 'black' as Player,
      secondaryArrowVariant: 'opponent' as const,
    };
  }, [
    isViewingHistory,
    inspectedEntry,
    isSandboxMode,
    showLiveHint,
    sandboxStepsTaken.length,
    sandboxAnalysis,
    sandboxCurrentState.currentTurn,
    isAITurn,
    isReplayingOpponent,
    stepsTaken.length,
    liveAnalysis,
    currentState.currentTurn,
    aiPlannedSteps,
    showLastOpponentArrows,
    lastOpponentTurn,
  ]);

  const isOpponentTurnOrAnimating =
    !isViewingHistory &&
    !isSandboxMode &&
    ((gameMode === 'ai' &&
      currentState.currentTurn === 'black' &&
      !holdAIForImportedTurn) ||
      isAIAnimating ||
      isReplayingOpponent);

  const hasGenuineNoLegalMovesForPanel = isViewingHistory
    ? Boolean(
        inspectedEntry &&
          (inspectedEntry.steps.length === 0 ||
            (inspectedEntry.analysis !== undefined &&
              inspectedEntry.analysis.topMoves.length === 0 &&
              !inspectedEntry.isAnalyzing))
      )
    : !isOpponentTurnOrAnimating && hasNoLegalMoves;

  const analysisPanelElement = (
    <AnalysisPanel
      analysis={panelAnalysis}
      isThinking={isThinking && !isViewingHistory}
      thinkingElapsedMs={thinkingElapsedMs}
      activePlayer={panelPlayer}
      dice={panelDice}
      isViewingHistory={isViewingHistory}
      isHistoryTurnAnalyzing={Boolean(inspectedEntry?.isAnalyzing)}
      historyTurnNumber={inspectedEntry?.turnNumber}
      isOpponentTurnOrAnimating={isOpponentTurnOrAnimating}
      hasGenuineNoLegalMoves={hasGenuineNoLegalMovesForPanel}
      showLiveHint={showLiveHint}
      onToggleLiveHint={() => setShowLiveHint((prev) => !prev)}
      onHoverMoveSteps={setPreviewSteps}
      onApplyRecommendedMove={
        isSandboxMode
          ? !sandboxGameOver
            ? handleApplyRecommendedMove
            : undefined
          : !isViewingHistory &&
              !isAITurn &&
              !isAIAnimating &&
              !isReplayingOpponent &&
              !gameOver
            ? handleApplyRecommendedMove
            : undefined
      }
      onRunRollout={handleRunRollout}
      isRollingOut={isRollingOut}
      rolloutProgress={rolloutProgress}
      rolloutResults={rolloutResults}
    />
  );

  const canRollBoardDice =
    !isViewingHistory &&
    !isAITurn &&
    !isReplayingOpponent &&
    !activeGameOver &&
    (isSandboxMode || activeRolledDice === null);

  if (isCompactMobile) {
    const showRotateOverlay =
      isPortrait &&
      !rotatePromptDismissed &&
      !isScannerModalOpen &&
      !showGameReview;

    return (
      <div
        dir="rtl"
        className="h-[100dvh] w-screen overflow-hidden bg-[#070b14] text-slate-100 flex flex-col font-sans safe-area-pad touch-board-surface"
      >
        {/* Main Mobile Cockpit: Landscape Side-by-Side OR Portrait Fallback Stack */}
        {isMobileLandscape || !isPortrait ? (
          <div className="flex-1 min-h-0 w-full flex flex-row items-stretch">
            {/* Board + Slim Vertical EvalBar (~82-85% width, ~96% height) */}
            <div
              dir="ltr"
              className="flex-1 min-w-0 h-full p-1.5 flex items-stretch gap-1.5"
            >
              <EvalBar evalResult={evalResult} compact={true} />
              <Board
                compact={true}
                boardState={displayedState}
                validNextSteps={validNextSteps}
                validDestinationOptions={validDestinationOptions}
                selectedFrom={selectedFrom}
                onSelectFrom={setSelectedFrom}
                onExecuteStep={handleExecuteStep}
                onExecuteStepSequence={handleExecuteStepSequence}
                isInteractive={
                  isSandboxMode
                    ? !sandboxGameOver
                    : !isViewingHistory &&
                      !isAITurn &&
                      !isReplayingOpponent &&
                      !gameOver
                }
                rolledDice={
                  isReplayingOpponent && lastOpponentTurn
                    ? lastOpponentTurn.dice
                    : isSandboxMode
                      ? sandboxRolledDice
                      : inspectedEntry
                        ? inspectedEntry.dice
                        : rolledDice
                }
                stepsTaken={
                  isReplayingOpponent
                    ? replayStepsTaken
                    : isSandboxMode
                      ? sandboxStepsTaken
                      : inspectedEntry
                        ? inspectedEntry.steps
                        : stepsTaken
                }
                lastTurnSteps={lastTurnSteps}
                previewSteps={previewSteps}
                primaryArrowSteps={primaryArrowSteps}
                primaryArrowPlayer={primaryArrowPlayer}
                secondaryArrowSteps={secondaryArrowSteps}
                secondaryArrowPlayer={secondaryArrowPlayer}
                secondaryArrowVariant={secondaryArrowVariant}
                aiStatusBanner={aiStatusBanner}
                onRollDice={handleRollDice}
                canRollDice={canRollBoardDice}
              />
            </div>

            {/* Thumb-Reachable Side Action Rail (126px) */}
            <aside className="w-[126px] shrink-0 h-full bg-slate-950/95 border-r border-slate-800/90 p-1.5 flex flex-col justify-between gap-1.5">
              {/* Compact Pips + Active Turn */}
              <div className="space-y-1">
                <div
                  className={`flex items-center justify-between px-2 py-1 rounded-lg border text-[10px] ${
                    displayedState.currentTurn === 'black' && !activeGameOver
                      ? 'bg-slate-800/90 border-emerald-500/50 text-white'
                      : 'bg-slate-900/70 border-slate-800 text-slate-300'
                  }`}
                >
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-slate-900 border border-slate-400 shrink-0" />
                    <span className="font-bold">שחור</span>
                  </div>
                  <span dir="ltr" className="font-mono font-bold">
                    {blackPips}
                  </span>
                </div>

                <div
                  className={`flex items-center justify-between px-2 py-1 rounded-lg border text-[10px] ${
                    displayedState.currentTurn === 'white' && !activeGameOver
                      ? 'bg-stone-100/10 border-emerald-500/50 text-white'
                      : 'bg-slate-900/70 border-slate-800 text-slate-300'
                  }`}
                >
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-stone-100 border border-stone-400 shrink-0" />
                    <span className="font-bold">לבן</span>
                  </div>
                  <span dir="ltr" className="font-mono font-bold">
                    {whitePips}
                  </span>
                </div>

                {isThinking && (
                  <div className="px-2 py-1 rounded-lg bg-cyan-500/15 border border-cyan-500/35 text-[10px] text-cyan-200 font-mono flex items-center justify-center gap-1">
                    <Brain className="w-3 h-3 animate-pulse text-cyan-400 shrink-0" />
                    <span>{(thinkingElapsedMs / 1000).toFixed(1)}s</span>
                  </div>
                )}
              </div>

              {/* Primary Thumb Action Area */}
              <div className="flex flex-col gap-1.5 my-auto">
                {isViewingHistory && viewingHistoryIndex !== null ? (
                  <>
                    <button
                      type="button"
                      onClick={() => setViewingHistoryIndex(null)}
                      className="w-full min-h-[48px] px-2 py-2 rounded-xl bg-amber-400 hover:bg-amber-300 text-slate-950 font-bold text-xs shadow-lg cursor-pointer"
                    >
                      חזור לחי
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        handleEnterSandboxAtTurn(viewingHistoryIndex)
                      }
                      className="w-full min-h-[44px] px-2 py-1.5 rounded-xl bg-cyan-500/20 text-cyan-200 border border-cyan-500/40 font-semibold text-[11px] cursor-pointer"
                    >
                      חקירה
                    </button>
                  </>
                ) : isAITurn ? (
                  <div className="w-full min-h-[48px] px-2 py-2 rounded-xl bg-slate-900 border border-slate-700 text-[11px] text-sky-300 font-semibold flex flex-col items-center justify-center text-center">
                    <span>תור AI</span>
                    <span className="text-[10px] text-slate-400">מחשב...</span>
                  </div>
                ) : hasNoLegalMoves ? (
                  <button
                    type="button"
                    onClick={handlePassTurn}
                    className="w-full min-h-[48px] px-2 py-2 rounded-xl bg-rose-500 text-white font-bold text-xs shadow-lg shadow-rose-500/30 animate-pulse cursor-pointer"
                  >
                    העבר תור
                  </button>
                ) : isTurnComplete ? (
                  <>
                    <button
                      type="button"
                      onClick={handleConfirmTurn}
                      disabled={Boolean(activeGameOver) || isReplayingOpponent}
                      className="w-full min-h-[48px] px-2 py-2 rounded-xl bg-emerald-400 hover:bg-emerald-300 text-slate-950 font-extrabold text-xs shadow-[0_0_20px_rgba(16,185,129,0.45)] flex items-center justify-center gap-1 cursor-pointer"
                    >
                      <Check className="w-4 h-4" />
                      אשר תור
                    </button>
                    {activeStepsTaken.length > 0 && (
                      <button
                        type="button"
                        onClick={handleUndoStep}
                        className="w-full min-h-[44px] px-2 py-1.5 rounded-xl bg-slate-800 text-slate-200 border border-slate-700 font-semibold text-xs flex items-center justify-center gap-1 cursor-pointer"
                      >
                        <Undo2 className="w-3.5 h-3.5" />
                        בטל
                      </button>
                    )}
                  </>
                ) : activeRolledDice === null || isSandboxMode ? (
                  <button
                    type="button"
                    onClick={handleRollDice}
                    disabled={Boolean(activeGameOver) || isReplayingOpponent}
                    className="w-full min-h-[48px] px-2 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-extrabold text-xs shadow-lg shadow-emerald-500/25 flex items-center justify-center gap-1.5 cursor-pointer"
                  >
                    <Dices className="w-4 h-4" />
                    הטל קוביות
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleUndoStep}
                    disabled={activeStepsTaken.length === 0}
                    className={`w-full min-h-[48px] px-2 py-2 rounded-xl font-bold text-xs border flex items-center justify-center gap-1 ${
                      activeStepsTaken.length > 0
                        ? 'bg-slate-800 text-slate-100 border-slate-600 cursor-pointer'
                        : 'bg-slate-900/60 text-slate-400 border-slate-800'
                    }`}
                  >
                    <Undo2 className="w-3.5 h-3.5" />
                    {activeStepsTaken.length > 0
                      ? `בטל (${activeStepsTaken.length})`
                      : 'בחר כלי'}
                  </button>
                )}
              </div>

              {/* 2x2 Quick-Access Icon Grid */}
              <div className="grid grid-cols-2 gap-1.5">
                <button
                  type="button"
                  onClick={() =>
                    setMobileDrawer((prev) =>
                      prev === 'analysis' ? null : 'analysis'
                    )
                  }
                  className={`relative min-h-[44px] min-w-[44px] rounded-xl border flex flex-col items-center justify-center gap-0.5 text-[10px] font-semibold cursor-pointer ${
                    mobileDrawer === 'analysis'
                      ? 'bg-cyan-500/25 text-cyan-200 border-cyan-400'
                      : 'bg-slate-900 text-slate-200 border-slate-800'
                  }`}
                >
                  <Lightbulb className="w-4 h-4 text-cyan-400" />
                  <span>רמז</span>
                  {showLiveHint && (
                    <span className="absolute top-1 left-1 w-2 h-2 rounded-full bg-cyan-400" />
                  )}
                </button>

                <button
                  type="button"
                  onClick={() =>
                    setMobileDrawer((prev) =>
                      prev === 'history' ? null : 'history'
                    )
                  }
                  className={`relative min-h-[44px] min-w-[44px] rounded-xl border flex flex-col items-center justify-center gap-0.5 text-[10px] font-semibold cursor-pointer ${
                    mobileDrawer === 'history'
                      ? 'bg-emerald-500/25 text-emerald-200 border-emerald-400'
                      : 'bg-slate-900 text-slate-200 border-slate-800'
                  }`}
                >
                  <History className="w-4 h-4 text-emerald-400" />
                  <span>היסטוריה</span>
                  {history.length > 0 && (
                    <span
                      dir="ltr"
                      className="absolute top-0.5 left-0.5 px-1 rounded-full bg-slate-800 text-[9px] font-mono text-slate-200"
                    >
                      {history.length}
                    </span>
                  )}
                  {pendingAnalysisCount > 0 && (
                    <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => setIsScannerModalOpen(true)}
                  className="min-h-[44px] min-w-[44px] rounded-xl bg-slate-900 text-slate-200 border border-slate-800 flex flex-col items-center justify-center gap-0.5 text-[10px] font-semibold cursor-pointer"
                >
                  <Camera className="w-4 h-4 text-cyan-300" />
                  <span>סרוק</span>
                </button>

                <button
                  type="button"
                  onClick={() => setMobileMenuOpen(true)}
                  className="min-h-[44px] min-w-[44px] rounded-xl bg-slate-900 text-slate-200 border border-slate-800 flex flex-col items-center justify-center gap-0.5 text-[10px] font-semibold cursor-pointer"
                >
                  <Menu className="w-4 h-4 text-slate-300" />
                  <span>תפריט</span>
                </button>
              </div>
            </aside>
          </div>
        ) : (
          /* Clean Portrait Fallback when user dismissed Rotate Prompt */
          <div className="flex-1 min-h-0 flex flex-col justify-between p-1.5 gap-1.5">
            <div className="flex items-center justify-between gap-1.5 px-2 py-1 rounded-xl bg-slate-950/90 border border-slate-800 shrink-0">
              <div className="flex items-center gap-2 text-[11px] font-mono">
                <span className="text-slate-300">שחור: {blackPips}</span>
                <span className="text-slate-600">|</span>
                <span className="text-stone-100">לבן: {whitePips}</span>
              </div>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => setMobileDrawer('analysis')}
                  className="min-w-[44px] min-h-[44px] rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center"
                  title="רמז"
                >
                  <Lightbulb className="w-4 h-4 text-cyan-400" />
                </button>
                <button
                  type="button"
                  onClick={() => setMobileDrawer('history')}
                  className="min-w-[44px] min-h-[44px] rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center"
                  title="היסטוריה"
                >
                  <History className="w-4 h-4 text-emerald-400" />
                </button>
                <button
                  type="button"
                  onClick={() => setIsScannerModalOpen(true)}
                  className="min-w-[44px] min-h-[44px] rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center"
                  title="סרוק לוח"
                >
                  <Camera className="w-4 h-4 text-cyan-300" />
                </button>
                <button
                  type="button"
                  onClick={() => setMobileMenuOpen(true)}
                  className="min-w-[44px] min-h-[44px] rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center"
                  title="תפריט"
                >
                  <Menu className="w-4 h-4 text-slate-200" />
                </button>
              </div>
            </div>

            <div dir="ltr" className="flex-1 min-h-0 flex items-stretch gap-1">
              <EvalBar evalResult={evalResult} compact={true} />
              <Board
                compact={true}
                boardState={displayedState}
                validNextSteps={validNextSteps}
                validDestinationOptions={validDestinationOptions}
                selectedFrom={selectedFrom}
                onSelectFrom={setSelectedFrom}
                onExecuteStep={handleExecuteStep}
                onExecuteStepSequence={handleExecuteStepSequence}
                isInteractive={
                  isSandboxMode
                    ? !sandboxGameOver
                    : !isViewingHistory &&
                      !isAITurn &&
                      !isReplayingOpponent &&
                      !gameOver
                }
                rolledDice={
                  isReplayingOpponent && lastOpponentTurn
                    ? lastOpponentTurn.dice
                    : isSandboxMode
                      ? sandboxRolledDice
                      : inspectedEntry
                        ? inspectedEntry.dice
                        : rolledDice
                }
                stepsTaken={
                  isReplayingOpponent
                    ? replayStepsTaken
                    : isSandboxMode
                      ? sandboxStepsTaken
                      : inspectedEntry
                        ? inspectedEntry.steps
                        : stepsTaken
                }
                lastTurnSteps={lastTurnSteps}
                previewSteps={previewSteps}
                primaryArrowSteps={primaryArrowSteps}
                primaryArrowPlayer={primaryArrowPlayer}
                secondaryArrowSteps={secondaryArrowSteps}
                secondaryArrowPlayer={secondaryArrowPlayer}
                secondaryArrowVariant={secondaryArrowVariant}
                aiStatusBanner={aiStatusBanner}
                onRollDice={handleRollDice}
                canRollDice={canRollBoardDice}
              />
            </div>

            <div className="flex items-center gap-2 shrink-0">
              {isAITurn ? (
                <div className="flex-1 min-h-[48px] rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-xs text-sky-300 font-semibold">
                  {aiStatusBanner || 'היריב (שחור) מחשב מהלך...'}
                </div>
              ) : hasNoLegalMoves ? (
                <button
                  type="button"
                  onClick={handlePassTurn}
                  className="flex-1 min-h-[48px] rounded-xl bg-rose-500 text-white font-bold text-xs cursor-pointer"
                >
                  אין מהלכים חוקיים — העבר תור
                </button>
              ) : isTurnComplete ? (
                <>
                  <button
                    type="button"
                    onClick={handleConfirmTurn}
                    className="flex-1 min-h-[48px] rounded-xl bg-emerald-400 text-slate-950 font-extrabold text-xs cursor-pointer"
                  >
                    אשר תור
                  </button>
                  <button
                    type="button"
                    onClick={handleUndoStep}
                    className="min-h-[48px] px-4 rounded-xl bg-slate-800 text-slate-200 border border-slate-700 font-semibold text-xs cursor-pointer"
                  >
                    בטל
                  </button>
                </>
              ) : activeRolledDice === null ? (
                <button
                  type="button"
                  onClick={handleRollDice}
                  className="flex-1 min-h-[48px] rounded-xl bg-emerald-500 text-slate-950 font-extrabold text-xs cursor-pointer"
                >
                  הטל קוביות
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handleUndoStep}
                  disabled={activeStepsTaken.length === 0}
                  className="flex-1 min-h-[48px] rounded-xl bg-slate-800 text-slate-200 border border-slate-700 font-semibold text-xs"
                >
                  בטל צעד ({activeStepsTaken.length})
                </button>
              )}
            </div>
          </div>
        )}

        {/* Slide-Over Drawer: Analysis or Move History */}
        {mobileDrawer && (
          <div className="fixed inset-0 z-40 flex justify-end bg-slate-950/65 backdrop-blur-xs">
            <div
              className="flex-1"
              onClick={() => setMobileDrawer(null)}
              aria-hidden="true"
            />
            <div className="w-[88vw] max-w-lg h-full bg-slate-950 border-r border-slate-800 shadow-2xl flex flex-col overflow-hidden safe-area-pad">
              <div className="px-3 py-2 border-b border-slate-800 flex items-center justify-between bg-slate-900/90 shrink-0">
                <span className="text-xs font-bold text-white">
                  {mobileDrawer === 'analysis'
                    ? 'ניתוח רב-אמן (עומק 3+)'
                    : 'היסטוריית מהלכים'}
                </span>
                <button
                  type="button"
                  onClick={() => setMobileDrawer(null)}
                  className="min-w-[44px] min-h-[44px] rounded-xl bg-slate-800 text-slate-200 flex items-center justify-center cursor-pointer"
                  title="סגור"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="flex-1 overflow-y-auto p-2">
                {mobileDrawer === 'analysis' ? (
                  analysisPanelElement
                ) : (
                  <MoveHistory
                    history={history}
                    viewingIndex={viewingHistoryIndex}
                    onSelectHistoryIndex={(idx) => {
                      setIsSandboxMode(false);
                      setViewingHistoryIndex(idx);
                      setMobileDrawer(null);
                    }}
                    onEnterSandboxAtTurn={(idx) => {
                      handleEnterSandboxAtTurn(idx);
                      setMobileDrawer(null);
                    }}
                  />
                )}
              </div>
            </div>
          </div>
        )}

        {/* Compact Action Sheet Menu */}
        {mobileMenuOpen && (
          <div className="fixed inset-0 z-40 flex items-center justify-center bg-slate-950/75 backdrop-blur-sm p-4">
            <div
              className="absolute inset-0"
              onClick={() => setMobileMenuOpen(false)}
              aria-hidden="true"
            />
            <div className="relative w-full max-w-sm rounded-2xl bg-slate-900 border border-slate-700 p-4 space-y-2.5 shadow-2xl">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <span className="text-sm font-bold text-white">תפריט משחק</span>
                <button
                  type="button"
                  onClick={() => setMobileMenuOpen(false)}
                  className="p-2 rounded-lg bg-slate-800 text-slate-300 cursor-pointer"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => {
                    setGameMode('ai');
                    setMobileMenuOpen(false);
                  }}
                  className={`min-h-[44px] rounded-xl text-xs font-bold border flex items-center justify-center gap-1.5 cursor-pointer ${
                    gameMode === 'ai'
                      ? 'bg-cyan-400 text-slate-950 border-cyan-300'
                      : 'bg-slate-800 text-slate-300 border-slate-700'
                  }`}
                >
                  <Bot className="w-4 h-4" />
                  נגד AI
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setGameMode('local');
                    setMobileMenuOpen(false);
                  }}
                  className={`min-h-[44px] rounded-xl text-xs font-bold border flex items-center justify-center gap-1.5 cursor-pointer ${
                    gameMode === 'local'
                      ? 'bg-emerald-500 text-slate-950 border-emerald-400'
                      : 'bg-slate-800 text-slate-300 border-slate-700'
                  }`}
                >
                  <Users className="w-4 h-4" />
                  שני שחקנים
                </button>
              </div>

              <button
                type="button"
                onClick={() => {
                  setShowGameReview(true);
                  setMobileMenuOpen(false);
                }}
                className="w-full min-h-[44px] px-3 rounded-xl bg-amber-400/15 text-amber-200 border border-amber-400/40 text-xs font-bold flex items-center justify-center gap-2 cursor-pointer"
              >
                <BarChart3 className="w-4 h-4" />
                סיכום וניתוח משחק ({history.length})
              </button>

              {canUseFullscreen && (
                <button
                  type="button"
                  onClick={() => {
                    void handleToggleFullscreen();
                    setMobileMenuOpen(false);
                  }}
                  className="w-full min-h-[44px] px-3 rounded-xl bg-slate-800 text-slate-200 border border-slate-700 text-xs font-bold flex items-center justify-center gap-2 cursor-pointer"
                >
                  {isFullscreen ? (
                    <Minimize2 className="w-4 h-4" />
                  ) : (
                    <Maximize2 className="w-4 h-4" />
                  )}
                  {isFullscreen ? 'צא ממסך מלא' : 'מסך מלא'}
                </button>
              )}

              {deferredInstallPrompt && (
                <button
                  type="button"
                  onClick={() => {
                    void handleInstallPwa();
                    setMobileMenuOpen(false);
                  }}
                  className="w-full min-h-[44px] px-3 rounded-xl bg-emerald-500/20 text-emerald-200 border border-emerald-500/40 text-xs font-bold flex items-center justify-center gap-2 cursor-pointer"
                >
                  <Download className="w-4 h-4" />
                  התקן כאפליקציה
                </button>
              )}

              <button
                type="button"
                onClick={() => {
                  startNewGame(gameMode);
                  setMobileMenuOpen(false);
                }}
                className="w-full min-h-[44px] px-3 rounded-xl bg-rose-500/15 text-rose-200 border border-rose-500/40 text-xs font-bold flex items-center justify-center gap-2 cursor-pointer"
              >
                <RotateCcw className="w-4 h-4" />
                משחק חדש
              </button>
            </div>
          </div>
        )}

        {/* Full-Screen Game Review Modal on Mobile */}
        {showGameReview && (
          <div className="fixed inset-0 z-40 bg-slate-950/90 backdrop-blur-md p-2 flex items-center justify-center safe-area-pad">
            <div className="w-full max-w-4xl">
              <GameReviewPanel
                review={gameReviewSummary}
                selectedTurnIndex={viewingHistoryIndex}
                onSelectTurnIndex={(idx) => {
                  setIsSandboxMode(false);
                  setViewingHistoryIndex(idx);
                  setShowGameReview(false);
                }}
                onEnterSandboxAtTurn={(idx) => {
                  handleEnterSandboxAtTurn(idx);
                  setShowGameReview(false);
                }}
                onExportMatch={handleExportMatch}
                onImportMatch={handleImportMatch}
                onClose={() => setShowGameReview(false)}
                pendingAnalysisCount={pendingAnalysisCount}
              />
            </div>
          </div>
        )}

        {/* Portrait Rotate Prompt Overlay */}
        {showRotateOverlay && (
          <RotatePhoneOverlay onDismiss={handleDismissRotatePrompt} />
        )}

        {/* Scan & Analyze Board from Image Modal */}
        <ErrorBoundary
          fallbackTitle="אירעה שגיאה בחלון סריקת התמונה"
          onClose={() => setIsScannerModalOpen(false)}
        >
          <ImagePositionModal
            isOpen={isScannerModalOpen}
            initialBoardState={displayedState}
            onClose={() => setIsScannerModalOpen(false)}
            onConfirmPosition={handleConfirmScannedPosition}
          />
        </ErrorBoundary>
      </div>
    );
  }

  return (
    <div
      dir="rtl"
      className="min-h-[100dvh] overflow-x-hidden bg-[#070b14] text-slate-100 flex flex-col font-sans selection:bg-emerald-500/30"
    >
      {/* Top Navigation & Controls Header */}
      <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-[1440px] mx-auto px-3 sm:px-6 min-h-16 py-2 flex flex-wrap items-center justify-between gap-2.5">
          {/* Brand Title */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-400 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Sparkles className="w-5 h-5 text-slate-950" />
            </div>
            <div>
              <h1 className="text-base sm:text-lg font-bold tracking-tight text-white flex items-center gap-2">
                שש-בש AI
                <span className="hidden sm:inline-flex items-center gap-1 text-[10px] font-semibold px-2.5 py-0.5 rounded-full bg-cyan-500/15 text-cyan-200 border border-cyan-500/35">
                  <Cpu className="w-3 h-3 text-cyan-400" />
                  מנוע רב-אמן AI (עומק 3+)
                </span>
              </h1>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                מנוע ניתוח רב-שלבי, חצים טקטיים ואימון בזמן אמת
              </p>
            </div>
          </div>

          {/* Mode Selector ("נגד AI" vs "שני שחקנים"), Static Engine Badge, Scan, Live Hint & Game Review */}
          <div className="flex items-center gap-1.5 sm:gap-2 flex-wrap">
            {/* Game Mode Toggle: נגד AI vs שני שחקנים */}
            <div
              role="group"
              aria-label="בחירת מצב משחק"
              className="inline-flex rounded-xl bg-slate-900 p-1 border border-slate-800 gap-1"
            >
              <button
                type="button"
                onClick={() => setGameMode('ai')}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                  gameMode === 'ai'
                    ? 'bg-cyan-400 text-slate-950 shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Bot className="w-3.5 h-3.5" />
                נגד AI
              </button>
              <button
                type="button"
                onClick={() => setGameMode('local')}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                  gameMode === 'local'
                    ? 'bg-emerald-500 text-slate-950 shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Users className="w-3.5 h-3.5" />
                שני שחקנים
              </button>
            </div>

            {deferredInstallPrompt && (
              <button
                type="button"
                onClick={handleInstallPwa}
                className="inline-flex items-center gap-1.5 px-2.5 py-2 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-200 border border-emerald-500/40 text-xs font-bold transition-all cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                התקן כאפליקציה
              </button>
            )}

            {/* Scan Board from Image Button */}
            <button
              type="button"
              onClick={() => setIsScannerModalOpen(true)}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-gradient-to-r from-cyan-500/20 to-emerald-500/20 hover:from-cyan-500/30 hover:to-emerald-500/30 text-cyan-200 border border-cyan-500/50 text-xs font-bold transition-all cursor-pointer shadow-sm"
            >
              <Camera className="w-3.5 h-3.5 text-cyan-300" />
              סרוק לוח מתמונה
            </button>

            {/* Live Hint Header Toggle */}
            <button
              type="button"
              onClick={() => setShowLiveHint((prev) => !prev)}
              className={`inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-bold border transition-all cursor-pointer ${
                showLiveHint
                  ? 'bg-cyan-500/20 text-cyan-200 border-cyan-500/50'
                  : 'bg-slate-800/80 text-slate-400 border-slate-700 hover:text-slate-200'
              }`}
            >
              <Lightbulb className="w-3.5 h-3.5" />
              רמז חי
            </button>

            {/* Game Review Toggle Button */}
            <button
              type="button"
              onClick={() => setShowGameReview((prev) => !prev)}
              className={`inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-bold border transition-all cursor-pointer ${
                showGameReview
                  ? 'bg-amber-400 text-slate-950 border-amber-300 shadow-lg shadow-amber-400/20'
                  : 'bg-slate-800 hover:bg-slate-700 text-amber-300 border-slate-700'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              סיכום וניתוח משחק
              {history.length > 0 && (
                <span
                  dir="ltr"
                  className={`inline-block px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                    showGameReview
                      ? 'bg-slate-950/20 text-slate-950'
                      : 'bg-amber-400/20 text-amber-300'
                  }`}
                >
                  {history.length}
                </span>
              )}
            </button>

            <button
              type="button"
              onClick={() => startNewGame(gameMode)}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-100 border border-slate-700 text-xs font-semibold transition-colors cursor-pointer shadow"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              משחק חדש
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 max-w-[1440px] w-full mx-auto px-3 sm:px-6 py-4 sm:py-5 flex flex-col lg:flex-row gap-5 items-stretch">
        {/* Main Column */}
        <div className="flex-1 flex flex-col gap-3 min-w-0">
          {/* Imported Scanned Position Badge Banner */}
          {importedPositionLabel && !isSandboxMode && (
            <div className="rounded-xl bg-cyan-950/35 border border-cyan-500/40 px-3.5 py-2 flex flex-wrap items-center justify-between gap-2 text-xs">
              <div className="flex items-center gap-2 text-cyan-200 font-semibold">
                <Camera className="w-4 h-4 text-cyan-400" />
                <span>{importedPositionLabel}</span>
                <span className="text-slate-400 font-normal hidden sm:inline">
                  — מוצג ניתוח חי וחצים ירוקים למהלך המיטבי על הלוח
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setIsScannerModalOpen(true)}
                  className="px-2.5 py-1 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-200 border border-cyan-500/40 font-semibold cursor-pointer"
                >
                  ערוך עמדה / קוביות
                </button>
                <button
                  type="button"
                  onClick={() => setImportedPositionLabel(null)}
                  className="text-slate-400 hover:text-slate-200 cursor-pointer p-1"
                  title="הסתר הודעה"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}

          {/* Interactive "What-If" Sandbox Mode Banner */}
          {isSandboxMode && (
            <div className="rounded-2xl bg-gradient-to-r from-amber-500/20 via-amber-500/10 to-cyan-500/15 border border-amber-400/60 px-4 py-3 flex flex-wrap items-center justify-between gap-3 shadow-lg">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-amber-400 text-slate-950 flex items-center justify-center shadow">
                  <FlaskConical className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-xs sm:text-sm font-bold text-amber-200 flex items-center gap-2">
                    מצב חקירה (Sandbox) — בודק קו משחק חלופי
                    {sandboxOriginTurnIndex !== null && (
                      <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-slate-950/70 text-amber-300 border border-amber-400/30">
                        התפצל מתור #{sandboxOriginTurnIndex + 1}
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-slate-300">
                    נסה מהלכים חלופיים או הטל קוביות לתורות הבאים בחופשיות.
                    היסטוריית המשחק המקורית נשמרת במלואה.
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleResetSandboxPosition}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-amber-200 border border-amber-400/40 text-xs font-semibold transition-colors cursor-pointer"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  אפס עמדה
                </button>
                <button
                  type="button"
                  onClick={handleExitSandbox}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-amber-400 hover:bg-amber-300 text-slate-950 text-xs font-bold transition-colors shadow cursor-pointer"
                >
                  <X className="w-3.5 h-3.5" />
                  צא ממצב חקירה
                </button>
              </div>
            </div>
          )}

          {/* End-of-Game Victory Banner */}
          {gameOver && !isSandboxMode && (
            <div className="rounded-2xl bg-gradient-to-r from-emerald-500/20 via-teal-500/15 to-amber-500/20 border border-emerald-400/50 p-4 flex flex-wrap items-center justify-between gap-4 shadow-xl">
              <div className="flex items-center gap-3">
                <div className="w-11 h-11 rounded-xl bg-amber-400/20 border border-amber-400/50 flex items-center justify-center text-amber-300">
                  <Trophy className="w-6 h-6" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-base sm:text-lg font-bold text-white">
                      {getHebrewVictoryTitle(gameOver)}
                    </h2>
                    <span
                      dir="ltr"
                      className="inline-block px-2 py-0.5 rounded-full text-xs font-mono font-bold bg-amber-400 text-slate-950"
                    >
                      {gameOver.multiplier}× נקודות
                    </span>
                  </div>
                  <p className="text-xs text-slate-300 mt-0.5">
                    {gameOver.winType === 'Backgammon'
                      ? 'ליריב לא הוצא אף כלי ונותרו לו כלים בבית המנצח או על הבר (ניקוד משולש).'
                      : gameOver.winType === 'Gammon'
                        ? 'היריב לא הספיק להוציא אף כלי מהלוח (ניקוד כפול).'
                        : 'ניצחון משחק רגיל — כל 15 הכלים הוצאו מהלוח.'}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowGameReview(true)}
                  className="px-4 py-2 rounded-xl bg-amber-400 hover:bg-amber-300 text-slate-950 font-bold text-xs transition-colors shadow-lg cursor-pointer"
                >
                  לסיכום המשחק
                </button>
                <button
                  type="button"
                  onClick={() => startNewGame(gameMode)}
                  className="px-4 py-2 rounded-xl bg-emerald-400 hover:bg-emerald-300 text-slate-950 font-bold text-xs transition-colors shadow-lg cursor-pointer"
                >
                  משחק חדש
                </button>
              </div>
            </div>
          )}

          {/* Collapsible Game Review Suite Panel */}
          {showGameReview && (
            <GameReviewPanel
              review={gameReviewSummary}
              selectedTurnIndex={viewingHistoryIndex}
              onSelectTurnIndex={(idx) => {
                setIsSandboxMode(false);
                setViewingHistoryIndex(idx);
              }}
              onEnterSandboxAtTurn={handleEnterSandboxAtTurn}
              onExportMatch={handleExportMatch}
              onImportMatch={handleImportMatch}
              onClose={() => setShowGameReview(false)}
              pendingAnalysisCount={pendingAnalysisCount}
            />
          )}

          {/* Top Player Info Strip (Black) */}
          <div className="flex items-center justify-between px-4 py-2.5 rounded-xl bg-slate-900/90 border border-slate-800">
            <div className="flex items-center gap-3 flex-wrap">
              <div className="relative">
                <div className="w-7 h-7 rounded-full bg-gradient-to-b from-slate-700 to-slate-950 border-2 border-slate-500 shadow" />
                {displayedState.currentTurn === 'black' && !activeGameOver && (
                  <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-emerald-400 ring-2 ring-slate-900 animate-ping" />
                )}
              </div>
              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-sm font-bold text-slate-100">
                    שחור
                  </span>
                  <span className="text-[11px] px-2 py-0.5 rounded-md bg-slate-800 text-slate-300 font-medium">
                    {isSandboxMode
                      ? 'ארגז חול'
                      : gameMode === 'local'
                        ? 'שחקן 2'
                        : 'מנוע רב-אמן AI (עומק 3+)'}
                  </span>
                  {displayedState.currentTurn === 'black' && !activeGameOver && (
                    <span className="text-[11px] text-emerald-400 font-semibold">
                      • תור השחור
                    </span>
                  )}
                  {lastOpponentTurn && !isAITurn && (
                    <span className="text-[11px] px-2 py-0.5 rounded-md bg-sky-500/15 border border-sky-500/30 text-sky-300">
                      מהלך אחרון:{' '}
                      <span dir="ltr" className="inline-block font-mono font-bold">
                        {lastOpponentTurn.notation}
                      </span>
                    </span>
                  )}
                </div>
                <div className="text-[11px] text-slate-400">
                  בית השחור: נקודות{' '}
                  <span dir="ltr" className="inline-block font-mono">
                    19–24
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="text-left">
                <div className="text-[10px] text-slate-400">פיפס</div>
                <div
                  dir="ltr"
                  className="inline-block text-sm sm:text-base font-mono font-bold text-slate-100"
                >
                  {blackPips}
                </div>
              </div>
              <div className="h-7 w-[1px] bg-slate-800" />
              <div className="text-left">
                <div className="text-[10px] text-slate-400">הוצאה</div>
                <div
                  dir="ltr"
                  className="inline-block text-sm sm:text-base font-mono font-bold text-emerald-400"
                >
                  {displayedState.borneOff.black}/15
                </div>
              </div>
            </div>
          </div>

          {/* Center Row: Keep dir="ltr" so EvalBar stays on the left of the physical Backgammon board */}
          <div dir="ltr" className="flex items-stretch gap-3">
            <EvalBar evalResult={evalResult} />

            <Board
              boardState={displayedState}
              validNextSteps={validNextSteps}
              validDestinationOptions={validDestinationOptions}
              selectedFrom={selectedFrom}
              onSelectFrom={setSelectedFrom}
              onExecuteStep={handleExecuteStep}
              onExecuteStepSequence={handleExecuteStepSequence}
              isInteractive={
                isSandboxMode
                  ? !sandboxGameOver
                  : !isViewingHistory &&
                    !isAITurn &&
                    !isReplayingOpponent &&
                    !gameOver
              }
              rolledDice={
                isReplayingOpponent && lastOpponentTurn
                  ? lastOpponentTurn.dice
                  : isSandboxMode
                    ? sandboxRolledDice
                    : inspectedEntry
                      ? inspectedEntry.dice
                      : rolledDice
              }
              stepsTaken={
                isReplayingOpponent
                  ? replayStepsTaken
                  : isSandboxMode
                    ? sandboxStepsTaken
                    : inspectedEntry
                      ? inspectedEntry.steps
                      : stepsTaken
              }
              lastTurnSteps={lastTurnSteps}
              previewSteps={previewSteps}
              primaryArrowSteps={primaryArrowSteps}
              primaryArrowPlayer={primaryArrowPlayer}
              secondaryArrowSteps={secondaryArrowSteps}
              secondaryArrowPlayer={secondaryArrowPlayer}
              secondaryArrowVariant={secondaryArrowVariant}
              aiStatusBanner={aiStatusBanner}
            />
          </div>

          {/* Bottom Player Info Strip (White) & Turn Action Controls */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 px-4 py-3 rounded-xl bg-slate-900/90 border border-slate-800">
            {/* White Player Info */}
            <div className="flex items-center justify-between sm:justify-start gap-4">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <div className="w-7 h-7 rounded-full bg-gradient-to-b from-stone-50 to-stone-300 border-2 border-stone-400 shadow" />
                  {displayedState.currentTurn === 'white' && !activeGameOver && (
                    <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-emerald-400 ring-2 ring-slate-900 animate-ping" />
                  )}
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-white">לבן</span>
                    <span className="text-[11px] px-2 py-0.5 rounded-md bg-slate-800 text-stone-200 font-medium">
                      {isSandboxMode ? 'ארגז חול' : 'שחקן 1 (אתה)'}
                    </span>
                    {displayedState.currentTurn === 'white' && !activeGameOver && (
                      <span className="text-[11px] text-emerald-400 font-semibold">
                        • תור הלבן
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] text-slate-400">
                    בית הלבן: נקודות{' '}
                    <span dir="ltr" className="inline-block font-mono">
                      6–1
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-4 sm:mr-4">
                <div className="text-left">
                  <div className="text-[10px] text-slate-400">פיפס</div>
                  <div
                    dir="ltr"
                    className="inline-block text-sm sm:text-base font-mono font-bold text-stone-100"
                  >
                    {whitePips}
                  </div>
                </div>
                <div className="h-7 w-[1px] bg-slate-800" />
                <div className="text-left">
                  <div className="text-[10px] text-slate-400">הוצאה</div>
                  <div
                    dir="ltr"
                    className="inline-block text-sm sm:text-base font-mono font-bold text-emerald-400"
                  >
                    {displayedState.borneOff.white}/15
                  </div>
                </div>
              </div>
            </div>

            {/* Action Controls Bar */}
            <div className="flex flex-wrap items-center justify-end gap-2">
              {/* Replay Opponent Turn Button */}
              {lastOpponentTurn &&
                !isAITurn &&
                !isViewingHistory &&
                !isSandboxMode && (
                  <button
                    type="button"
                    onClick={handleReplayOpponentTurn}
                    disabled={isReplayingOpponent}
                    className={`inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold border transition-all ${
                      isReplayingOpponent
                        ? 'bg-sky-500/20 text-sky-300 border-sky-500/40 animate-pulse cursor-wait'
                        : 'bg-sky-500/15 hover:bg-sky-500/25 text-sky-300 border-sky-500/35 cursor-pointer'
                    }`}
                    title="הפעל שוב אנימציה של תור היריב האחרון"
                  >
                    <Play className="w-3.5 h-3.5" />
                    הצג שוב תור יריב
                  </button>
                )}

              {isViewingHistory && viewingHistoryIndex !== null ? (
                <>
                  <button
                    type="button"
                    onClick={() => handleEnterSandboxAtTurn(viewingHistoryIndex)}
                    className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-200 border border-cyan-500/40 font-bold text-xs transition-colors cursor-pointer"
                  >
                    <FlaskConical className="w-3.5 h-3.5" />
                    בדוק חלופה (מצב חקירה)
                  </button>
                  <button
                    type="button"
                    onClick={() => setViewingHistoryIndex(null)}
                    className="px-4 py-2 rounded-xl bg-amber-400 hover:bg-amber-300 text-slate-950 font-bold text-xs transition-colors shadow cursor-pointer"
                  >
                    חזור למשחק החי
                  </button>
                </>
              ) : isAITurn ? (
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 border border-slate-700 text-xs text-sky-300 font-semibold">
                  <Brain className="w-4 h-4 animate-pulse text-sky-400" />
                  <span>
                    {aiStatusBanner || 'היריב (שחור) מבצע תור...'}
                    {isThinking && thinkingElapsedMs >= 100 && (
                      <span dir="ltr" className="mr-1.5 font-mono text-cyan-300">
                        (עומק 3+ • {(thinkingElapsedMs / 1000).toFixed(1)}s)
                      </span>
                    )}
                  </span>
                </div>
              ) : (
                <>
                  {/* Roll Dice Button */}
                  <button
                    type="button"
                    onClick={handleRollDice}
                    disabled={
                      (!isSandboxMode && activeRolledDice !== null) ||
                      Boolean(activeGameOver) ||
                      isReplayingOpponent
                    }
                    className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all ${
                      (isSandboxMode || activeRolledDice === null) &&
                      !activeGameOver &&
                      !isReplayingOpponent
                        ? 'bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-lg shadow-emerald-500/25 cursor-pointer'
                        : 'bg-slate-800/60 text-slate-500 border border-slate-800 cursor-not-allowed'
                    }`}
                  >
                    <Dices className="w-4 h-4" />
                    {isSandboxMode ? 'הטל קוביות חדשות' : 'הטל קוביות'}
                  </button>

                  {/* Undo Step Button */}
                  <button
                    type="button"
                    onClick={handleUndoStep}
                    disabled={
                      activeStepsTaken.length === 0 ||
                      Boolean(activeGameOver) ||
                      isReplayingOpponent
                    }
                    className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold border transition-all ${
                      activeStepsTaken.length > 0 &&
                      !activeGameOver &&
                      !isReplayingOpponent
                        ? 'bg-slate-800 hover:bg-slate-700 text-slate-100 border-slate-600 cursor-pointer'
                        : 'bg-slate-900/50 text-slate-600 border-slate-800 cursor-not-allowed'
                    }`}
                  >
                    <Undo2 className="w-3.5 h-3.5" />
                    בטל צעד ({activeStepsTaken.length})
                  </button>

                  {/* No Legal Moves — Pass Turn Button */}
                  {hasNoLegalMoves ? (
                    <button
                      type="button"
                      onClick={handlePassTurn}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-rose-500 hover:bg-rose-400 text-white font-bold text-xs transition-all shadow-lg shadow-rose-500/30 animate-pulse cursor-pointer"
                    >
                      <ShieldAlert className="w-4 h-4" />
                      אין מהלכים חוקיים — העבר תור
                    </button>
                  ) : (
                    /* Confirm Turn Button */
                    <button
                      type="button"
                      onClick={handleConfirmTurn}
                      disabled={
                        !isTurnComplete ||
                        Boolean(activeGameOver) ||
                        isReplayingOpponent
                      }
                      className={`inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold transition-all ${
                        isTurnComplete &&
                        !activeGameOver &&
                        !isReplayingOpponent
                          ? 'bg-amber-400 hover:bg-amber-300 text-slate-950 shadow-lg shadow-amber-400/25 cursor-pointer'
                          : 'bg-slate-800/60 text-slate-500 border border-slate-800 cursor-not-allowed'
                      }`}
                    >
                      <Check className="w-4 h-4" />
                      אשר תור
                    </button>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Stockfish-style Top-3 Engine Lines & Coach Analysis Panel */}
          {analysisPanelElement}
        </div>

        {/* Sidebar: Move History & AI Badges */}
        <MoveHistory
          history={history}
          viewingIndex={viewingHistoryIndex}
          onSelectHistoryIndex={(idx) => {
            setIsSandboxMode(false);
            setViewingHistoryIndex(idx);
          }}
          onEnterSandboxAtTurn={handleEnterSandboxAtTurn}
        />
      </main>

      {/* Scan & Analyze Board from Image Modal */}
      <ErrorBoundary
        fallbackTitle="אירעה שגיאה בחלון סריקת התמונה"
        onClose={() => setIsScannerModalOpen(false)}
      >
        <ImagePositionModal
          isOpen={isScannerModalOpen}
          initialBoardState={displayedState}
          onClose={() => setIsScannerModalOpen(false)}
          onConfirmPosition={handleConfirmScannedPosition}
        />
      </ErrorBoundary>
    </div>
  );
}

export default App;
