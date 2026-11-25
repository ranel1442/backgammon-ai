import {
  analyzePosition,
  CHAMPION_SEARCH_DEPTH,
  LIVE_SEARCH_CUTOFF_MS,
  type SearchDepth,
  type TurnAnalysisResult,
} from '../engine/ai';
import {
  initNeuralNetFromWeightsJSON,
  type BackgammonNNWeightsFile,
} from '../engine/neuralNet';
import {
  runMonteCarloRollout,
  type RolloutMoveResult,
} from '../engine/rollout';
import type { BoardState, MoveStep, TurnMove } from '../engine/types';

export type AIWorkerRequest =
  | {
      type: 'INIT_WEIGHTS';
      weights: BackgammonNNWeightsFile | null;
    }
  | {
      requestId: number;
      type: 'ANALYZE';
      state: BoardState;
      dice: [number, number];
      depth?: SearchDepth;
      playedSteps?: MoveStep[];
      cutoffMs?: number;
      turnId?: string;
    }
  | {
      requestId: number;
      type: 'ROLLOUT';
      state: BoardState;
      dice: [number, number];
      candidateMoves: TurnMove[];
      trialsPerMove: number;
    };

export type AIWorkerResponse =
  | {
      type: 'WEIGHTS_READY';
      loaded: boolean;
    }
  | {
      requestId: number;
      type: 'ANALYZE_RESULT';
      result: TurnAnalysisResult;
      turnId?: string;
    }
  | {
      requestId: number;
      type: 'ROLLOUT_PROGRESS';
      progressPct: number;
      partialResults: RolloutMoveResult[];
    }
  | {
      requestId: number;
      type: 'ROLLOUT_RESULT';
      rolloutResults: RolloutMoveResult[];
    }
  | {
      requestId?: number;
      type: 'WORKER_ERROR';
      error: string;
      turnId?: string;
    };

let activeRequestId: number | undefined = undefined;
let activeTurnId: string | undefined = undefined;

function formatWorkerError(err: unknown): string {
  if (err instanceof Error) {
    return `${err.name}: ${err.message}`;
  }
  return String(err);
}

function postWorkerError(
  err: unknown,
  requestId?: number,
  turnId?: string
): void {
  const message = formatWorkerError(err);
  console.error('[AI Worker] Exception:', err);
  const errorMsg: AIWorkerResponse = {
    requestId,
    type: 'WORKER_ERROR',
    error: message,
    turnId,
  };
  self.postMessage(errorMsg);
}

self.onerror = (event) => {
  const errText =
    typeof event === 'string'
      ? event
      : event instanceof ErrorEvent
        ? event.message
        : 'Uncaught worker error';
  postWorkerError(errText, activeRequestId, activeTurnId);
};

self.onunhandledrejection = (event: PromiseRejectionEvent) => {
  postWorkerError(event.reason, activeRequestId, activeTurnId);
};

self.onmessage = (event: MessageEvent<AIWorkerRequest>) => {
  const data = event.data;

  if (data.type === 'INIT_WEIGHTS') {
    try {
      const nn = initNeuralNetFromWeightsJSON(data.weights);
      const ack: AIWorkerResponse = {
        type: 'WEIGHTS_READY',
        loaded: Boolean(nn),
      };
      self.postMessage(ack);
    } catch (err) {
      postWorkerError(err);
      const fallbackAck: AIWorkerResponse = {
        type: 'WEIGHTS_READY',
        loaded: false,
      };
      self.postMessage(fallbackAck);
    }
    return;
  }

  if (data.type === 'ANALYZE') {
    const {
      requestId,
      state,
      dice,
      depth = CHAMPION_SEARCH_DEPTH,
      playedSteps,
      cutoffMs = LIVE_SEARCH_CUTOFF_MS,
      turnId,
    } = data;
    activeRequestId = requestId;
    activeTurnId = turnId;
    try {
      const result = analyzePosition(
        state,
        dice,
        depth,
        playedSteps,
        cutoffMs
      );
      const response: AIWorkerResponse = {
        requestId,
        type: 'ANALYZE_RESULT',
        result,
        turnId,
      };
      self.postMessage(response);
    } catch (err) {
      postWorkerError(err, requestId, turnId);
    } finally {
      activeRequestId = undefined;
      activeTurnId = undefined;
    }
    return;
  }

  if (data.type === 'ROLLOUT') {
    const { requestId, state, dice, candidateMoves, trialsPerMove } = data;
    activeRequestId = requestId;
    activeTurnId = undefined;
    try {
      const rolloutResults = runMonteCarloRollout(
        state,
        dice,
        candidateMoves,
        trialsPerMove,
        (progressPct, partialResults) => {
          const progressMsg: AIWorkerResponse = {
            requestId,
            type: 'ROLLOUT_PROGRESS',
            progressPct,
            partialResults,
          };
          self.postMessage(progressMsg);
        }
      );

      const finalMsg: AIWorkerResponse = {
        requestId,
        type: 'ROLLOUT_RESULT',
        rolloutResults,
      };
      self.postMessage(finalMsg);
    } catch (err) {
      postWorkerError(err, requestId);
    } finally {
      activeRequestId = undefined;
    }
  }
};
