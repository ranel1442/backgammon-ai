import { useCallback, useEffect, useRef, useState } from 'react';
import {
  analyzePosition,
  type SearchDepth,
  type TurnAnalysisResult,
} from '../engine/ai';
import { tryLoadNeuralNetWeights } from '../engine/neuralNet';
import {
  runMonteCarloRollout,
  type RolloutMoveResult,
} from '../engine/rollout';
import type { BoardState, MoveStep, TurnMove } from '../engine/types';
import type {
  AIWorkerRequest,
  AIWorkerResponse,
} from '../workers/ai.worker';

interface PendingAnalyzeCallback {
  resolve: (result: TurnAnalysisResult) => void;
  isLiveRequest: boolean;
}

interface PendingRolloutCallback {
  resolve: (results: RolloutMoveResult[]) => void;
}

export function useBackgammonAI() {
  const workerRef = useRef<Worker | null>(null);
  const requestCounterRef = useRef(0);
  const latestLiveRequestIdRef = useRef(0);
  const latestRolloutRequestIdRef = useRef(0);
  const pendingMapRef = useRef<Map<number, PendingAnalyzeCallback>>(new Map());
  const pendingRolloutRef = useRef<Map<number, PendingRolloutCallback>>(
    new Map()
  );

  const [isThinking, setIsThinking] = useState(false);
  const [thinkingDepth, setThinkingDepth] = useState<SearchDepth>(2);
  const [thinkingElapsedMs, setThinkingElapsedMs] = useState(0);
  const [isRollingOut, setIsRollingOut] = useState(false);
  const [rolloutProgress, setRolloutProgress] = useState(0);
  const [rolloutResults, setRolloutResults] = useState<
    RolloutMoveResult[] | null
  >(null);
  const [nnLoaded, setNnLoaded] = useState(false);

  useEffect(() => {
    void tryLoadNeuralNetWeights('/weights/backgammon_nn.json').then(
      (loaded) => {
        if (loaded) setNnLoaded(true);
      }
    );

    if (typeof Worker === 'undefined') return;

    const worker = new Worker(
      new URL('../workers/ai.worker.ts', import.meta.url),
      { type: 'module' }
    );

    worker.onmessage = (event: MessageEvent<AIWorkerResponse>) => {
      const msg = event.data;

      if (msg.type === 'ANALYZE_RESULT') {
        const { requestId, result } = msg;
        const pending = pendingMapRef.current.get(requestId);
        if (!pending) return;

        pendingMapRef.current.delete(requestId);

        if (
          pending.isLiveRequest &&
          requestId === latestLiveRequestIdRef.current
        ) {
          setIsThinking(false);
        }

        if (
          !pending.isLiveRequest ||
          requestId === latestLiveRequestIdRef.current
        ) {
          pending.resolve(result);
        }
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

    workerRef.current = worker;

    return () => {
      worker.terminate();
      workerRef.current = null;
      pendingMapRef.current.clear();
      pendingRolloutRef.current.clear();
    };
  }, []);

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

  const analyzeTurn = useCallback(
    (
      state: BoardState,
      dice: [number, number],
      depth: SearchDepth = 2,
      playedSteps?: MoveStep[],
      options?: { isBackground?: boolean }
    ): Promise<TurnAnalysisResult> => {
      const requestId = ++requestCounterRef.current;
      const isLiveRequest = !options?.isBackground;

      if (isLiveRequest) {
        latestLiveRequestIdRef.current = requestId;
        setThinkingDepth(depth);
        setIsThinking(true);
      }

      if (!workerRef.current) {
        const syncResult = analyzePosition(state, dice, depth, playedSteps);
        if (isLiveRequest) {
          setIsThinking(false);
        }
        return Promise.resolve(syncResult);
      }

      return new Promise<TurnAnalysisResult>((resolve) => {
        pendingMapRef.current.set(requestId, { resolve, isLiveRequest });
        const payload: AIWorkerRequest = {
          requestId,
          type: 'ANALYZE',
          state,
          dice,
          depth,
          playedSteps,
        };
        workerRef.current!.postMessage(payload);
      });
    },
    []
  );

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

      if (!workerRef.current) {
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
        const payload: AIWorkerRequest = {
          requestId,
          type: 'ROLLOUT',
          state,
          dice,
          candidateMoves,
          trialsPerMove,
        };
        workerRef.current!.postMessage(payload);
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

  return {
    isThinking,
    thinkingDepth,
    thinkingElapsedMs,
    analyzeTurn,
    isRollingOut,
    rolloutProgress,
    rolloutResults,
    runRollout,
    clearRolloutResults,
    nnLoaded,
  };
}
