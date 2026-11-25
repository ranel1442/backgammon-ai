import { describe, expect, it, vi } from 'vitest';
import { evaluatePosition } from './evaluator';
import { createInitialBoard } from './rules';
import {
  flipBoard180,
  GEMINI_REQUEST_TIMEOUT_MS,
  GEMINI_VISION_MODELS,
  normalizeScannedBoard,
  scanBackgammonImage,
  ScanCancelledError,
  swapBoardColors,
} from './visionScanner';

describe('Vision Scanner & Board Calibration Helpers', () => {
  describe('normalizeScannedBoard', () => {
    it('clamps extreme point counts, caps total checkers at 15, and normalizes dice', () => {
      const rawPoints = new Array<number>(24).fill(0);
      rawPoints[5] = 19; // Exceeds 15
      rawPoints[12] = 4; // Total white on points = 23 -> should cap to <= 15
      rawPoints[18] = -20; // Exceeds -15

      const state = normalizeScannedBoard({
        points: rawPoints,
        bar: { white: 2, black: 3 },
        borneOff: { white: 0, black: 0 },
        dice: [6, 4],
        detectedTurn: 'black',
      });

      const whiteOnPoints = state.points.reduce(
        (sum, v) => (v > 0 ? sum + v : sum),
        0
      );
      const blackOnPoints = state.points.reduce(
        (sum, v) => (v < 0 ? sum + Math.abs(v) : sum),
        0
      );

      expect(
        whiteOnPoints + state.bar.white + state.borneOff.white
      ).toBeLessThanOrEqual(15);
      expect(
        blackOnPoints + state.bar.black + state.borneOff.black
      ).toBeLessThanOrEqual(15);
      expect(state.currentTurn).toBe('black');
      expect(state.dice).toEqual([6, 4]);
    });

    it('automatically infers borneOff when borneOff is 0 and fewer than 15 checkers are on board + bar', () => {
      const rawPoints = new Array<number>(24).fill(0);
      // White has 10 checkers on board + 1 on bar = 11 -> borneOff should be inferred as 4
      rawPoints[0] = 4;
      rawPoints[2] = 3;
      rawPoints[4] = 3;

      // Black has 12 checkers on board + 0 on bar = 12 -> borneOff should be inferred as 3
      rawPoints[23] = -5;
      rawPoints[21] = -4;
      rawPoints[19] = -3;

      const state = normalizeScannedBoard({
        points: rawPoints,
        bar: { white: 1, black: 0 },
        borneOff: { white: 0, black: 0 },
        dice: [3, 3],
        detectedTurn: 'white',
      });

      expect(state.borneOff.white).toBe(4);
      expect(state.borneOff.black).toBe(3);
      expect(state.dice).toEqual([3, 3, 3, 3]);
    });
  });

  describe('flipBoard180 & swapBoardColors calibration helpers', () => {
    it('flipBoard180 rotates 180 degrees and double-flip returns the exact original board', () => {
      const board = createInitialBoard();
      board.points[0] = 3;
      board.points[12] = -4;
      board.bar.white = 1;

      const flippedOnce = flipBoard180(board);
      expect(flippedOnce.points[0]).toBe(-4);
      expect(flippedOnce.points[12]).toBe(3);
      expect(flippedOnce.bar.white).toBe(1);

      const flippedTwice = flipBoard180(flippedOnce);
      expect(flippedTwice).toEqual(board);
    });

    it('swapBoardColors swaps White/Black ownership, mirrors home boards, and preserves exact evaluation antisymmetry', () => {
      const board = createInitialBoard();
      // Create an asymmetric contact position
      board.points[4] = 3; // White made golden 5-point
      board.points[3] = 2; // White made 4-point
      board.points[10] = 1; // White blot
      board.points[19] = -2; // Black made 5-point
      board.bar.black = 1;

      const originalEval = evaluatePosition(board);
      const swappedBoard = swapBoardColors(board);
      const swappedEval = evaluatePosition(swappedBoard);

      expect(originalEval.equity).not.toBe(0);
      expect(swappedEval.equity).toBe(-originalEval.equity);
      expect(swappedBoard.currentTurn).toBe('black');
      expect(swapBoardColors(swappedBoard)).toEqual(board);
    });
  });

  describe('scanBackgammonImage multi-model fallback cascade', () => {
    it('cascades through gemini-flash-latest (thinkingBudget=512) -> gemini-flash-lite-latest (thinkingBudget=0) with 90000ms timeout', async () => {
      expect(GEMINI_REQUEST_TIMEOUT_MS).toBe(90000);
      expect(GEMINI_VISION_MODELS).toEqual([
        'gemini-flash-latest',
        'gemini-flash-lite-latest',
      ]);

      const calledUrls: string[] = [];
      const thinkingBudgets: Array<number | undefined> = [];
      const mockFetch = vi.fn(
        async (url: string | URL | Request, init?: RequestInit) => {
          const urlStr = String(url);
          calledUrls.push(urlStr);
          if (typeof init?.body === 'string') {
            const bodyJson = JSON.parse(init.body);
            thinkingBudgets.push(
              bodyJson?.generationConfig?.thinkingConfig?.thinkingBudget
            );
          }

          if (urlStr.includes('gemini-flash-latest:')) {
            return new Response('Rate limit exceeded', { status: 429 });
          }
          return new Response(
            JSON.stringify({
              candidates: [
                {
                  content: {
                    parts: [
                      {
                        text: JSON.stringify({
                          points: createInitialBoard().points,
                          bar: { white: 0, black: 0 },
                          borneOff: { white: 0, black: 0 },
                          dice: [6, 1],
                          detectedTurn: 'white',
                          notes: 'זוהתה עמדת פתיחה עם 6-1 ללבן',
                        }),
                      },
                    ],
                  },
                },
              ],
            }),
            { status: 200 }
          );
        }
      );

      vi.stubGlobal('fetch', mockFetch);
      try {
        const res = await scanBackgammonImage(
          'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
          'AIzaTestKey123'
        );

        expect(calledUrls).toHaveLength(2);
        expect(calledUrls[0]).toContain('gemini-flash-latest:generateContent');
        expect(calledUrls[1]).toContain(
          'gemini-flash-lite-latest:generateContent'
        );
        expect(thinkingBudgets).toEqual([512, 0]);
        expect(res.dice).toEqual([6, 1]);
        expect(res.detectedTurn).toBe('white');
      } finally {
        vi.unstubAllGlobals();
      }
    });

    it('strictly excludes -image, -tts, native-audio, robotics, and embedding models in Dynamic Model Discovery', async () => {
      const calledUrls: string[] = [];
      const mockFetch = vi.fn(async (url: string | URL | Request) => {
        const urlStr = String(url);
        calledUrls.push(urlStr);

        if (urlStr.endsWith('models?key=AIzaTestKey123')) {
          return new Response(
            JSON.stringify({
              models: [
                {
                  name: 'models/text-embedding-004',
                  supportedGenerationMethods: ['embedContent'],
                },
                {
                  name: 'models/gemini-2.5-flash-image',
                  supportedGenerationMethods: ['generateContent'],
                },
                {
                  name: 'models/gemini-2.5-flash-native-audio-latest',
                  supportedGenerationMethods: ['generateContent'],
                },
                {
                  name: 'models/gemini-robotics-er-1.5-preview',
                  supportedGenerationMethods: ['generateContent'],
                },
                {
                  name: 'models/gemini-2.5-flash-tts',
                  supportedGenerationMethods: ['generateContent'],
                },
                {
                  name: 'models/gemini-3.0-flash-preview',
                  supportedGenerationMethods: ['generateContent'],
                },
              ],
            }),
            { status: 200 }
          );
        }

        if (urlStr.includes('gemini-3.0-flash-preview:generateContent')) {
          return new Response(
            JSON.stringify({
              candidates: [
                {
                  content: {
                    parts: [
                      {
                        text: JSON.stringify({
                          points: createInitialBoard().points,
                          bar: { white: 0, black: 0 },
                          borneOff: { white: 0, black: 0 },
                          dice: [5, 3],
                          detectedTurn: 'black',
                          notes: 'סריקה דינמית הצליחה',
                        }),
                      },
                    ],
                  },
                },
              ],
            }),
            { status: 200 }
          );
        }

        return new Response('Model not found', { status: 404 });
      });

      vi.stubGlobal('fetch', mockFetch);
      try {
        const res = await scanBackgammonImage(
          'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
          'AIzaTestKey123'
        );

        expect(calledUrls).toHaveLength(4);
        expect(calledUrls[2]).toContain('/v1beta/models?key=AIzaTestKey123');
        expect(calledUrls[3]).toContain(
          'gemini-3.0-flash-preview:generateContent'
        );
        expect(
          calledUrls.some((u) => u.includes('gemini-2.5-flash-image'))
        ).toBe(false);
        expect(res.dice).toEqual([5, 3]);
        expect(res.detectedTurn).toBe('black');
      } finally {
        vi.unstubAllGlobals();
      }
    });

    it('retries the same model without thinkingConfig when HTTP 400 mentions thinkingBudget', async () => {
      const requestBodies: Array<Record<string, unknown>> = [];
      let callCount = 0;

      const mockFetch = vi.fn(
        async (_url: string | URL | Request, init?: RequestInit) => {
          callCount += 1;
          if (typeof init?.body === 'string') {
            requestBodies.push(JSON.parse(init.body));
          }

          if (callCount === 1) {
            return new Response(
              JSON.stringify({
                error: { message: 'thinkingBudget is not supported by model' },
              }),
              { status: 400 }
            );
          }

          return new Response(
            JSON.stringify({
              candidates: [
                {
                  content: {
                    parts: [
                      {
                        text: JSON.stringify({
                          points: createInitialBoard().points,
                          bar: { white: 0, black: 0 },
                          borneOff: { white: 0, black: 0 },
                          dice: [4, 2],
                          detectedTurn: 'white',
                          notes: 'הצליח ללא thinkingConfig',
                        }),
                      },
                    ],
                  },
                },
              ],
            }),
            { status: 200 }
          );
        }
      );

      vi.stubGlobal('fetch', mockFetch);
      try {
        const res = await scanBackgammonImage(
          'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
          'AIzaTestKey123'
        );

        expect(callCount).toBe(2);
        expect(
          (requestBodies[0].generationConfig as Record<string, unknown>)
            .thinkingConfig
        ).toBeDefined();
        expect(
          (requestBodies[1].generationConfig as Record<string, unknown>)
            .thinkingConfig
        ).toBeUndefined();
        expect(res.dice).toEqual([4, 2]);
      } finally {
        vi.unstubAllGlobals();
      }
    });

    it('aggregates errors across all attempted models so 429 quota is not masked behind a 404', async () => {
      const mockFetch = vi.fn(async (url: string | URL | Request) => {
        const urlStr = String(url);
        if (urlStr.includes('gemini-flash-lite-latest:')) {
          return new Response('Quota exceeded', { status: 429 });
        }
        return new Response('Model not found', { status: 404 });
      });

      vi.stubGlobal('fetch', mockFetch);
      try {
        await expect(
          scanBackgammonImage(
            'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
            'AIzaTestKey123'
          )
        ).rejects.toThrow(/429/);
      } finally {
        vi.unstubAllGlobals();
      }
    });

    it('aborts immediately when cancelled via external AbortSignal without cascading', async () => {
      const controller = new AbortController();
      controller.abort();

      const mockFetch = vi.fn();
      vi.stubGlobal('fetch', mockFetch);
      try {
        await expect(
          scanBackgammonImage(
            'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
            'AIzaTestKey123',
            controller.signal
          )
        ).rejects.toBeInstanceOf(ScanCancelledError);
        expect(mockFetch).not.toHaveBeenCalled();
      } finally {
        vi.unstubAllGlobals();
      }
    });
  });
});
