import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BarChart3,
  Bot,
  Brain,
  Camera,
  Check,
  Dices,
  Download,
  FlaskConical,
  Lightbulb,
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
import { EvalBar } from './components/EvalBar';
import { GameReviewPanel } from './components/GameReviewPanel';
import { ImagePositionModal } from './components/ImagePositionModal';
import { MoveHistory, type HistoryEntry } from './components/MoveHistory';
import type { SearchDepth, TurnAnalysisResult } from './engine/ai';
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

type GameMode =
  | 'local'
  | 'ai-1ply'
  | 'ai-2ply'
  | 'ai-3ply'
  | 'ai-4ply'
  | 'ai-5ply';

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

export function App() {
  const [gameMode, setGameMode] = useState<GameMode>('ai-2ply');
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

  const {
    isThinking,
    thinkingDepth,
    thinkingElapsedMs,
    analyzeTurn,
    isRollingOut,
    rolloutProgress,
    rolloutResults,
    runRollout,
    clearRolloutResults,
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

  // Queue any unanalyzed turns when Game Review is opened
  useEffect(() => {
    if (!showGameReview) return;

    history.forEach((entry) => {
      if (!entry.analysis || !entry.classification) {
        analyzeTurn(entry.turnStartState, entry.dice, 2, entry.steps, {
          isBackground: true,
        }).then((analysisRes) => {
          setHistory((prev) =>
            prev.map((item) =>
              item.id === entry.id
                ? {
                    ...item,
                    classification:
                      analysisRes.playedMoveAnalysis?.classification ?? 'Best',
                    equityLoss: analysisRes.playedMoveAnalysis?.equityLoss ?? 0,
                    analysis: analysisRes,
                  }
                : item
            )
          );
        });
      }
    });
  }, [showGameReview, history, analyzeTurn]);

  const gameReviewSummary = useMemo(
    () => computeGameReview(history),
    [history]
  );

  const pendingAnalysisCount = useMemo(
    () => history.filter((h) => !h.classification).length,
    [history]
  );

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

  const evalResult = useMemo(
    () => evaluatePosition(displayedState),
    [displayedState]
  );
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
    gameMode !== 'local' &&
    currentState.currentTurn === 'black' &&
    !gameOver &&
    !isViewingHistory;

  const aiDepth: SearchDepth =
    gameMode === 'ai-1ply'
      ? 1
      : gameMode === 'ai-3ply'
        ? 3
        : gameMode === 'ai-4ply'
          ? 4
          : gameMode === 'ai-5ply'
            ? 5
            : 2;

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
    if (isSandboxMode || !rolledDice || isAITurn || gameOver) {
      if (!isSandboxMode) setLiveAnalysis(null);
      return;
    }

    let cancelled = false;
    analyzeTurn(turnStartState, rolledDice, 2).then((res) => {
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
    analyzeTurn,
  ]);

  useEffect(() => {
    if (!isSandboxMode || !sandboxRolledDice || sandboxGameOver) {
      setSandboxAnalysis(null);
      return;
    }

    let cancelled = false;
    analyzeTurn(sandboxTurnStartState, sandboxRolledDice, 2).then((res) => {
      if (!cancelled) {
        setSandboxAnalysis(res);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [
    isSandboxMode,
    sandboxTurnStartState,
    sandboxRolledDice,
    sandboxGameOver,
    analyzeTurn,
  ]);

  const handleEnterSandboxAtTurn = (turnIndex: number) => {
    const entry = history[turnIndex];
    if (!entry) return;

    clearPlaybackTimers();
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

    const evaluation = evaluatePosition(committedState);
    const entryId = history.length + 1;

    const initialEntry: HistoryEntry = {
      id: entryId,
      turnNumber: entryId,
      player,
      dice,
      steps,
      notation,
      turnStartState: startStateForTurn,
      resultingState: committedState,
      evalResult: evaluation,
      classification: precomputedAnalysis?.playedMoveAnalysis?.classification,
      equityLoss: precomputedAnalysis?.playedMoveAnalysis?.equityLoss,
      analysis: precomputedAnalysis,
    };

    setHistory((prev) => [...prev, initialEntry]);
    setTurnStartState(committedState);
    setCurrentState(committedState);
    setRolledDice(null);
    setStepsTaken([]);
    setUndoStack([]);
    setSelectedFrom(null);
    setLiveAnalysis(null);
    setPreviewSteps(null);
    setAiPlannedSteps(null);
    setAiStatusBanner(null);
    setHoldAIForImportedTurn(false);

    if (!precomputedAnalysis) {
      analyzeTurn(startStateForTurn, dice, 2, steps, {
        isBackground: true,
      }).then((analysisRes) => {
        setHistory((prev) =>
          prev.map((item) =>
            item.id === entryId
              ? {
                  ...item,
                  classification:
                    analysisRes.playedMoveAnalysis?.classification ?? 'Best',
                  equityLoss: analysisRes.playedMoveAnalysis?.equityLoss ?? 0,
                  analysis: analysisRes,
                }
              : item
          )
        );
      });
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
          setHistory(parsed.history);
          setCurrentState(parsed.currentState);
          setTurnStartState(parsed.currentState);
          if (parsed.gameMode) setGameMode(parsed.gameMode);
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

  // Paced, Staged AI Opponent Turn Execution:
  // Stage 1 (500ms): "היריב (שחור) מטיל קוביות..." -> Roll dice
  // Stage 2 (650ms): "היריב (שחור) משחק: [notation]" + Sky-blue planned move arrows
  // Stage 3 (450ms per step): Apply each MoveStep one by one so the player sees every checker move
  useEffect(() => {
    if (!isAITurn) return;

    if (rolledDice === null) {
      setAiStatusBanner('היריב (שחור) מטיל קוביות...');
      const rollTimer = setTimeout(() => {
        handleRollDice();
      }, 500);
      return () => clearTimeout(rollTimer);
    }

    let cancelled = false;
    setAiStatusBanner(
      `היריב (שחור) הטיל ${rolledDice[0]}-${rolledDice[1]} — מחשב מהלך...`
    );

    analyzeTurn(turnStartState, rolledDice, aiDepth).then((analysisRes) => {
      if (cancelled) return;

      if (analysisRes.topMoves.length === 0) {
        setAiStatusBanner(
          `היריב (שחור): ${rolledDice[0]}-${rolledDice[1]} — אין מהלכים חוקיים`
        );
        const passTimer = setTimeout(() => {
          if (cancelled) return;
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
      clearPlaybackTimers();
    };
  }, [isAITurn, rolledDice, turnStartState, aiDepth, analyzeTurn]);

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
                <span className="hidden md:inline-block text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                  3-Net ResNet + עומק 5+
                </span>
              </h1>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                מנוע ניתוח רב-שלבי, חצים טקטיים ואימון בזמן אמת
              </p>
            </div>
          </div>

          {/* Mode Selector (All 5 AI Depths + Local), Live Hint, Game Review Toggle & New Game */}
          <div className="flex items-center gap-1.5 sm:gap-2 flex-wrap">
            {/* Compact Mobile/Tablet Select */}
            <div className="xl:hidden">
              <select
                aria-label="בחירת מצב משחק ורמת AI"
                value={gameMode}
                onChange={(e) => setGameMode(e.target.value as GameMode)}
                className="px-2.5 py-1.5 rounded-xl bg-slate-900 border border-slate-700 text-xs font-bold text-emerald-300 focus:outline-none focus:border-emerald-400 cursor-pointer"
              >
                <option value="local">משחק זוגי (מקומי)</option>
                <option value="ai-1ply">נגד מאמן AI (עומק 1)</option>
                <option value="ai-2ply">נגד מאסטר AI (עומק 2+)</option>
                <option value="ai-3ply">נגד רב-אמן AI (עומק 3+)</option>
                <option value="ai-4ply">נגד אלוף-על AI (עומק 4+)</option>
                <option value="ai-5ply">נגד אלגוריתם-על AI (עומק 5+)</option>
              </select>
            </div>

            {/* Full Desktop Button Strip for All 5 AI Levels */}
            <div className="hidden xl:inline-flex rounded-xl bg-slate-900 p-1 border border-slate-800 gap-0.5">
              <button
                type="button"
                onClick={() => setGameMode('local')}
                className={`inline-flex items-center gap-1 px-2 py-1.5 rounded-lg text-[11px] font-semibold transition-all cursor-pointer ${
                  gameMode === 'local'
                    ? 'bg-emerald-500 text-slate-950 shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Users className="w-3.5 h-3.5" />
                משחק זוגי
              </button>
              <button
                type="button"
                onClick={() => setGameMode('ai-1ply')}
                className={`inline-flex items-center gap-1 px-2 py-1.5 rounded-lg text-[11px] font-semibold transition-all cursor-pointer ${
                  gameMode === 'ai-1ply'
                    ? 'bg-emerald-500 text-slate-950 shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Bot className="w-3.5 h-3.5" />
                נגד מאמן AI (עומק 1)
              </button>
              <button
                type="button"
                onClick={() => setGameMode('ai-2ply')}
                className={`inline-flex items-center gap-1 px-2 py-1.5 rounded-lg text-[11px] font-semibold transition-all cursor-pointer ${
                  gameMode === 'ai-2ply'
                    ? 'bg-emerald-500 text-slate-950 shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Brain className="w-3.5 h-3.5" />
                נגד מאסטר AI (עומק 2+)
              </button>
              <button
                type="button"
                onClick={() => setGameMode('ai-3ply')}
                className={`inline-flex items-center gap-1 px-2 py-1.5 rounded-lg text-[11px] font-semibold transition-all cursor-pointer ${
                  gameMode === 'ai-3ply'
                    ? 'bg-cyan-400 text-slate-950 shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Brain className="w-3.5 h-3.5" />
                נגד רב-אמן AI (עומק 3+)
              </button>
              <button
                type="button"
                onClick={() => setGameMode('ai-4ply')}
                className={`inline-flex items-center gap-1 px-2 py-1.5 rounded-lg text-[11px] font-semibold transition-all cursor-pointer ${
                  gameMode === 'ai-4ply'
                    ? 'bg-purple-400 text-slate-950 shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Sparkles className="w-3.5 h-3.5" />
                נגד אלוף-על AI (עומק 4+)
              </button>
              <button
                type="button"
                onClick={() => setGameMode('ai-5ply')}
                className={`inline-flex items-center gap-1 px-2 py-1.5 rounded-lg text-[11px] font-semibold transition-all cursor-pointer ${
                  gameMode === 'ai-5ply'
                    ? 'bg-amber-400 text-slate-950 shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Trophy className="w-3.5 h-3.5" />
                נגד אלגוריתם-על AI (עומק 5+)
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
                        : gameMode === 'ai-1ply'
                          ? 'מאמן AI (עומק 1)'
                          : gameMode === 'ai-2ply'
                            ? 'מאסטר AI (עומק 2+)'
                            : gameMode === 'ai-3ply'
                              ? 'רב-אמן AI (עומק 3+)'
                              : gameMode === 'ai-4ply'
                                ? 'אלוף-על AI (עומק 4+)'
                                : 'אלגוריתם-על AI (עומק 5+)'}
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
                        (עומק {aiDepth}
                        {aiDepth >= 2 ? '+' : ''} •{' '}
                        {(thinkingElapsedMs / 1000).toFixed(1)}s)
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
          <AnalysisPanel
            analysis={panelAnalysis}
            isThinking={isThinking && !isViewingHistory}
            thinkingDepth={thinkingDepth}
            thinkingElapsedMs={thinkingElapsedMs}
            activePlayer={panelPlayer}
            dice={panelDice}
            isViewingHistory={isViewingHistory}
            historyTurnNumber={inspectedEntry?.turnNumber}
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
      <ImagePositionModal
        isOpen={isScannerModalOpen}
        initialBoardState={displayedState}
        onClose={() => setIsScannerModalOpen(false)}
        onConfirmPosition={handleConfirmScannedPosition}
      />
    </div>
  );
}

export default App;
