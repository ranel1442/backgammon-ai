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

// Automatically load trained TD-Gammon neural network weights if available
void tryLoadNeuralNetWeights('/weights/backgammon_nn.json');

export type AIWorkerRequest =
  | {
      requestId: number;
      type: 'ANALYZE';
      state: BoardState;
      dice: [number, number];
      depth: SearchDepth;
      playedSteps?: MoveStep[];
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
      requestId: number;
      type: 'ANALYZE_RESULT';
      result: TurnAnalysisResult;
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
    };

self.onmessage = (event: MessageEvent<AIWorkerRequest>) => {
  const data = event.data;

  if (data.type === 'ANALYZE') {
    const { requestId, state, dice, depth, playedSteps } = data;
    const result = analyzePosition(state, dice, depth, playedSteps);
    const response: AIWorkerResponse = {
      requestId,
      type: 'ANALYZE_RESULT',
      result,
    };
    self.postMessage(response);
    return;
  }

  if (data.type === 'ROLLOUT') {
    const { requestId, state, dice, candidateMoves, trialsPerMove } = data;
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
  }
};
