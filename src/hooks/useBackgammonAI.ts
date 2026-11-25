import { useCallback, useEffect, useRef, useState } from 'react';
import {
  analyzePosition,
  BACKGROUND_ANALYSIS_CUTOFF_MS,
  CHAMPION_SEARCH_DEPTH,
  LIVE_SEARCH_CUTOFF_MS,
  type SearchDepth,
  type TurnAnalysisResult,
} from '../engine/ai';
import {
  fetchNeuralNetWeightsJSON,
  initNeuralNetFromWeightsJSON,
} from '../engine/neuralNet';
import {
  runMonteCarloRollout,
  type RolloutMoveResult,
} from '../engine/rollout';
import type { BoardState, MoveStep, TurnMove } from '../engine/types';
import type {
  AIWorkerRequest,
  AIWorkerResponse,
} from '../workers/ai.worker';

interface PendingGameplayAnalyze {
  state: BoardState;
  dice: [number, number];
  playedSteps?: MoveStep[];
  resolve: (result: TurnAnalysisResult) => void;
}

interface PendingRolloutCallback {
  resolve: (results: RolloutMoveResult[]) => void;
}

interface QueuedTurnAnalysisItem {
  turnId: string;
  beforeState: BoardState;
  dice: [number, number];
  playedMove: MoveStep[];
  generation: number;
  onComplete?: (turnId: string, result: TurnAnalysisResult) => void;
  resolve: (result: TurnAnalysisResult | null) => void;
}

export function useBackgammonAI() {
  const gameplayWorkerRef = useRef<Worker | null>(null);
  const analysisWorkerRef = useRef<Worker | null>(null);
  const weightsReadyPromiseRef = useRef<Promise<void>>(Promise.resolve());

  const requestCounterRef = useRef(0);
  const latestGameplayRequestIdRef = useRef(0);
  const latestRolloutRequestIdRef = useRef(0);

  const pendingGameplayMapRef = useRef<Map<number, PendingGameplayAnalyze>>(
    new Map()
  );
  const pendingRolloutRef = useRef<Map<number, PendingRolloutCallback>>(
    new Map()
  );

  // Dedicated FIFO queue state for `analysisWorker`
  const analysisQueueRef = useRef<QueuedTurnAnalysisItem[]>([]);
  const activeAnalysisItemRef = useRef<{
    requestId: number;
    item: QueuedTurnAnalysisItem;
  } | null>(null);
  const queueGenerationRef = useRef(0);

  const [isThinking, setIsThinking] = useState(false);
  const [thinkingElapsedMs, setThinkingElapsedMs] = useState(0);
  const [pendingAnalysisCount, setPendingAnalysisCount] = useState(0);
  const [isRollingOut, setIsRollingOut] = useState(false);
  const [rolloutProgress, setRolloutProgress] = useState(0);
  const [rolloutResults, setRolloutResults] = useState<
    RolloutMoveResult[] | null
  >(null);
  const [nnLoaded, setNnLoaded] = useState(false);

  const syncPendingCount = useCallback(() => {
    const count =
      analysisQueueRef.current.length +
      (activeAnalysisItemRef.current ? 1 : 0);
    setPendingAnalysisCount(count);
  }, []);

  const pumpAnalysisQueue = useCallback(() => {
    if (activeAnalysisItemRef.current !== null) {
      return;
    }

    const nextItem = analysisQueueRef.current.shift();
    if (!nextItem) {
      setPendingAnalysisCount(0);
      return;
    }

    // Skip stale items from a cleared game session
    if (nextItem.generation !== queueGenerationRef.current) {
      nextItem.resolve(null);
      pumpAnalysisQueue();
      return;
    }

    const worker = analysisWorkerRef.current;
    if (!worker) {
      // Fallback for non-Worker test environments (SSR / Vitest)
      const syncRes = analyzePosition(
        nextItem.beforeState,
        nextItem.dice,
        CHAMPION_SEARCH_DEPTH,
        nextItem.playedMove,
        BACKGROUND_ANALYSIS_CUTOFF_MS
      );
      if (nextItem.generation === queueGenerationRef.current) {
        nextItem.onComplete?.(nextItem.turnId, syncRes);
        nextItem.resolve(syncRes);
      } else {
        nextItem.resolve(null);
      }
      syncPendingCount();
      pumpAnalysisQueue();
      return;
    }

    const requestId = ++requestCounterRef.current;
    activeAnalysisItemRef.current = { requestId, item: nextItem };
    syncPendingCount();

    void weightsReadyPromiseRef.current.then(() => {
      if (
        !analysisWorkerRef.current ||
        activeAnalysisItemRef.current?.requestId !== requestId
      ) {
        return;
      }
      const payload: AIWorkerRequest = {
        requestId,
        type: 'ANALYZE',
        state: nextItem.beforeState,
        dice: nextItem.dice,
        depth: CHAMPION_SEARCH_DEPTH,
        playedSteps: nextItem.playedMove,
        cutoffMs: BACKGROUND_ANALYSIS_CUTOFF_MS,
        turnId: nextItem.turnId,
      };
      analysisWorkerRef.current.postMessage(payload);
    });
  }, [syncPendingCount]);

  useEffect(() => {
    let disposed = false;

    const hasWorkerSupport = typeof Worker !== 'undefined';
    const gameplayWorker = hasWorkerSupport
      ? new Worker(new URL('../workers/ai.worker.ts', import.meta.url), {
          type: 'module',
          name: 'backgammon-gameplay-worker',
        })
      : null;

    const analysisWorker = hasWorkerSupport
      ? new Worker(new URL('../workers/ai.worker.ts', import.meta.url), {
          type: 'module',
          name: 'backgammon-analysis-worker',
        })
      : null;

    gameplayWorkerRef.current = gameplayWorker;
    analysisWorkerRef.current = analysisWorker;

    // Fetch `/weights/backgammon_nn.json` ONCE on the main thread and broadcast to both workers
    weightsReadyPromiseRef.current = fetchNeuralNetWeightsJSON(
      '/weights/backgammon_nn.json'
    )
      .then((weightsJson) => {
        if (disposed) return;
        const nn = initNeuralNetFromWeightsJSON(weightsJson);
        if (nn) {
          setNnLoaded(true);
        }
        const initMsg: AIWorkerRequest = {
          type: 'INIT_WEIGHTS',
          weights: weightsJson,
        };
        gameplayWorker?.postMessage(initMsg);
        analysisWorker?.postMessage(initMsg);
      })
      .catch(() => {
        // Fallback to hybrid heuristic + 54,264-state bear-off tablebase
      });

    if (gameplayWorker) {
      const failGameplayRequest = (errText: string, reqId?: number) => {
        console.error('[useBackgammonAI] gameplayWorker error:', errText);
        if (reqId !== undefined) {
          const pending = pendingGameplayMapRef.current.get(reqId);
          if (pending) {
            pendingGameplayMapRef.current.delete(reqId);
            if (reqId === latestGameplayRequestIdRef.current) {
              setIsThinking(false);
            }
            const fallback = analyzePosition(
              pending.state,
              pending.dice,
              1,
              pending.playedSteps,
              2000
            );
            pending.resolve(fallback);
          }
          const pendingRollout = pendingRolloutRef.current.get(reqId);
          if (pendingRollout) {
            pendingRolloutRef.current.delete(reqId);
            if (reqId === latestRolloutRequestIdRef.current) {
              setIsRollingOut(false);
            }
            pendingRollout.resolve([]);
          }
          return;
        }

        // Unscoped worker error: drain all in-flight gameplay requests safely so timer never freezes
        setIsThinking(false);
        setIsRollingOut(false);
        for (const [, pending] of pendingGameplayMapRef.current) {
          const fallback = analyzePosition(
            pending.state,
            pending.dice,
            1,
            pending.playedSteps,
            2000
          );
          pending.resolve(fallback);
        }
        pendingGameplayMapRef.current.clear();
        for (const [, pendingR] of pendingRolloutRef.current) {
          pendingR.resolve([]);
        }
        pendingRolloutRef.current.clear();
      };

      gameplayWorker.onerror = (ev) => {
        failGameplayRequest(ev.message || 'Gameplay worker crashed');
      };

      gameplayWorker.onmessage = (event: MessageEvent<AIWorkerResponse>) => {
        const msg = event.data;

        if (msg.type === 'WORKER_ERROR') {
          failGameplayRequest(msg.error, msg.requestId);
          return;
        }

        if (msg.type === 'ANALYZE_RESULT') {
          const { requestId, result } = msg;
          const pending = pendingGameplayMapRef.current.get(requestId);
          if (!pending) return;

          pendingGameplayMapRef.current.delete(requestId);
          if (requestId === latestGameplayRequestIdRef.current) {
            setIsThinking(false);
          }
          pending.resolve(result);
          return;
        }

        if (msg.type === 'ROLLOUT_PROGRESS') {
          if (msg.requestId === latestRolloutRequestIdRef.current) {
            setRolloutProgress(msg.progressPct);
            setRolloutResults(msg.partialResults);
          }
          return;
        }

        if (msg.type === 'ROLLOUT_RESULT') {
          const pendingRollout = pendingRolloutRef.current.get(msg.requestId);
          pendingRolloutRef.current.delete(msg.requestId);

          if (msg.requestId === latestRolloutRequestIdRef.current) {
            setIsRollingOut(false);
            setRolloutProgress(100);
            setRolloutResults(msg.rolloutResults);
          }
          if (pendingRollout) {
            pendingRollout.resolve(msg.rolloutResults);
          }
        }
      };
    }

    if (analysisWorker) {
      const failActiveAnalysis = (errText: string, reqId?: number) => {
        console.error('[useBackgammonAI] analysisWorker error:', errText);
        const active = activeAnalysisItemRef.current;
        if (!active) return;
        if (reqId !== undefined && active.requestId !== reqId) return;

        activeAnalysisItemRef.current = null;
        const { item } = active;
        if (item.generation === queueGenerationRef.current) {
          const fallback = analyzePosition(
            item.beforeState,
            item.dice,
            1,
            item.playedMove,
            2000
          );
          item.onComplete?.(item.turnId, fallback);
          item.resolve(fallback);
        } else {
          item.resolve(null);
        }
        syncPendingCount();
        pumpAnalysisQueue();
      };

      analysisWorker.onerror = (ev) => {
        failActiveAnalysis(ev.message || 'Analysis worker crashed');
      };

      analysisWorker.onmessage = (event: MessageEvent<AIWorkerResponse>) => {
        const msg = event.data;
        if (msg.type === 'WORKER_ERROR') {
          failActiveAnalysis(msg.error, msg.requestId);
          return;
        }
        if (msg.type !== 'ANALYZE_RESULT') return;

        const active = activeAnalysisItemRef.current;
        if (!active || active.requestId !== msg.requestId) {
          return;
        }

        activeAnalysisItemRef.current = null;
        const { item } = active;

        if (item.generation === queueGenerationRef.current) {
          item.onComplete?.(item.turnId, msg.result);
          item.resolve(msg.result);
        } else {
          item.resolve(null);
        }

        syncPendingCount();
        pumpAnalysisQueue();
      };
    }

    return () => {
      disposed = true;
      gameplayWorker?.terminate();
      analysisWorker?.terminate();
      gameplayWorkerRef.current = null;
      analysisWorkerRef.current = null;
      pendingGameplayMapRef.current.clear();
      pendingRolloutRef.current.clear();
      analysisQueueRef.current = [];
      activeAnalysisItemRef.current = null;
    };
  }, [pumpAnalysisQueue, syncPendingCount]);

  useEffect(() => {
    if (!isThinking) {
      setThinkingElapsedMs(0);
      return;
    }
    const start = performance.now();
    const timer = window.setInterval(() => {
      setThinkingElapsedMs(Math.round(performance.now() - start));
    }, 100);
    return () => window.clearInterval(timer);
  }, [isThinking]);

  const dispatchGameplayAnalyze = useCallback(
    (
      state: BoardState,
      dice: [number, number],
      playedSteps?: MoveStep[]
    ): Promise<TurnAnalysisResult> => {
      const requestId = ++requestCounterRef.current;
      latestGameplayRequestIdRef.current = requestId;
      setIsThinking(true);

      if (!gameplayWorkerRef.current) {
        const syncResult = analyzePosition(
          state,
          dice,
          CHAMPION_SEARCH_DEPTH,
          playedSteps,
          LIVE_SEARCH_CUTOFF_MS
        );
        setIsThinking(false);
        return Promise.resolve(syncResult);
      }

      return new Promise<TurnAnalysisResult>((resolve) => {
        pendingGameplayMapRef.current.set(requestId, {
          state,
          dice,
          playedSteps,
          resolve,
        });
        void weightsReadyPromiseRef.current.then(() => {
          if (!gameplayWorkerRef.current) {
            const fallback = analyzePosition(
              state,
              dice,
              CHAMPION_SEARCH_DEPTH,
              playedSteps,
              LIVE_SEARCH_CUTOFF_MS
            );
            setIsThinking(false);
            resolve(fallback);
            return;
          }
          const payload: AIWorkerRequest = {
            requestId,
            type: 'ANALYZE',
            state,
            dice,
            depth: CHAMPION_SEARCH_DEPTH,
            playedSteps,
            cutoffMs: LIVE_SEARCH_CUTOFF_MS,
          };
          gameplayWorkerRef.current.postMessage(payload);
        });
      });
    },
    []
  );

  /**
   * Computes the AI opponent's own move at Champion `Depth 3+` on the dedicated `gameplayWorker` thread.
   */
  const computeAIMove = useCallback(
    (state: BoardState, dice: [number, number]): Promise<TurnAnalysisResult> =>
      dispatchGameplayAnalyze(state, dice),
    [dispatchGameplayAnalyze]
  );

  /**
   * Computes on-demand Live Hint ("רמז חי"), Scanned Image, or Sandbox analysis at `Depth 3+` on `gameplayWorker`.
   */
  const analyzeLivePosition = useCallback(
    (
      state: BoardState,
      dice: [number, number],
      playedSteps?: MoveStep[]
    ): Promise<TurnAnalysisResult> =>
      dispatchGameplayAnalyze(state, dice, playedSteps),
    [dispatchGameplayAnalyze]
  );

  /**
   * Enqueues a completed turn into the dedicated background `analysisWorker` FIFO queue at `Depth 3+`.
   * Never blocks `gameplayWorker` or delays the AI's dice roll / move execution.
   */
  const enqueueTurnAnalysis = useCallback(
    (
      turnId: string,
      beforeState: BoardState,
      dice: [number, number],
      playedMove: MoveStep[],
      onComplete?: (turnId: string, result: TurnAnalysisResult) => void
    ): Promise<TurnAnalysisResult | null> => {
      // Avoid duplicate queueing for the same turnId
      const alreadyQueued =
        activeAnalysisItemRef.current?.item.turnId === turnId ||
        analysisQueueRef.current.some((q) => q.turnId === turnId);
      if (alreadyQueued) {
        return Promise.resolve(null);
      }

      const generation = queueGenerationRef.current;
      return new Promise<TurnAnalysisResult | null>((resolve) => {
        analysisQueueRef.current.push({
          turnId,
          beforeState,
          dice,
          playedMove,
          generation,
          onComplete,
          resolve,
        });
        syncPendingCount();
        pumpAnalysisQueue();
      });
    },
    [pumpAnalysisQueue, syncPendingCount]
  );

  /**
   * Clears all pending background analysis jobs (called on "משחק חדש" or loading a new position).
   */
  const clearAnalysisQueue = useCallback(() => {
    queueGenerationRef.current += 1;
    for (const queued of analysisQueueRef.current) {
      queued.resolve(null);
    }
    analysisQueueRef.current = [];
    latestGameplayRequestIdRef.current = ++requestCounterRef.current;
    setIsThinking(false);
    setPendingAnalysisCount(0);
  }, []);

  const runRollout = useCallback(
    (
      state: BoardState,
      dice: [number, number],
      candidateMoves: TurnMove[],
      trialsPerMove: number = 1000
    ): Promise<RolloutMoveResult[]> => {
      const requestId = ++requestCounterRef.current;
      latestRolloutRequestIdRef.current = requestId;
      setIsRollingOut(true);
      setRolloutProgress(0);
      setRolloutResults(null);

      if (!gameplayWorkerRef.current) {
        const syncRollout = runMonteCarloRollout(
          state,
          dice,
          candidateMoves,
          trialsPerMove,
          (pct, partial) => {
            setRolloutProgress(pct);
            setRolloutResults(partial);
          }
        );
        setIsRollingOut(false);
        setRolloutProgress(100);
        setRolloutResults(syncRollout);
        return Promise.resolve(syncRollout);
      }

      return new Promise<RolloutMoveResult[]>((resolve) => {
        pendingRolloutRef.current.set(requestId, { resolve });
        void weightsReadyPromiseRef.current.then(() => {
          if (!gameplayWorkerRef.current) return;
          const payload: AIWorkerRequest = {
            requestId,
            type: 'ROLLOUT',
            state,
            dice,
            candidateMoves,
            trialsPerMove,
          };
          gameplayWorkerRef.current.postMessage(payload);
        });
      });
    },
    []
  );

  const clearRolloutResults = useCallback(() => {
    latestRolloutRequestIdRef.current = ++requestCounterRef.current;
    setIsRollingOut(false);
    setRolloutProgress(0);
    setRolloutResults(null);
  }, []);

  const thinkingDepth: SearchDepth = CHAMPION_SEARCH_DEPTH;

  return {
    isThinking,
    thinkingDepth,
    thinkingElapsedMs,
    computeAIMove,
    analyzeLivePosition,
    enqueueTurnAnalysis,
    pendingAnalysisCount,
    clearAnalysisQueue,
    isRollingOut,
    rolloutProgress,
    rolloutResults,
    runRollout,
    clearRolloutResults,
    nnLoaded,
  };
}
