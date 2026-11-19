import type { BoardState, Player } from './types';

export interface ScannedBoardResponse {
  /**
   * 24 points (indices 0..23):
   * Positive integers (1..15) = White/Light checkers
   * Negative integers (-1..-15) = Black/Dark checkers
   * 0 = Empty point
   */
  points: number[];
  bar: {
    white: number;
    black: number;
  };
  borneOff: {
    white: number;
    black: number;
  };
  dice: [number, number] | null;
  detectedTurn: Player | null;
  notes: string;
}

const GEMINI_VISION_SYSTEM_PROMPT = `You are an expert Backgammon board computer-vision analyzer.
Analyze the provided image (which may be a screenshot from a digital Backgammon app or a photo of a real-life physical wooden Backgammon board) and extract the exact checker position into a JSON object.

CANONICAL BOARD COORDINATE MAPPING (24 points, array indices 0 to 23):
Orient the board so the Bear-Off / Home boards are on the RIGHT side (if the home boards are on the left in the photo, mentally mirror/orient them to the right):
1. Bottom-Right Quadrant (White's Home Board, 6 points from RIGHT to LEFT):
   - Index 0: Bottom-most right corner point (White's 1-point)
   - Index 1: White's 2-point
   - Index 2: White's 3-point
   - Index 3: White's 4-point
   - Index 4: White's 5-point
   - Index 5: Point adjacent to the central bar on Bottom-Right (White's 6-point)
2. Bottom-Left Quadrant (White's Outer Board, 6 points from RIGHT to LEFT, starting next to the central bar):
   - Index 6: Point immediately to the left of the central bar on the bottom row (White's 7-point / Bar point)
   - Index 7: White's 8-point
   - Index 8: White's 9-point
   - Index 9: White's 10-point
   - Index 10: White's 11-point
   - Index 11: Bottom-most left corner point (White's 12-point)
3. Top-Left Quadrant (Black's Outer Board, 6 points from LEFT to RIGHT):
   - Index 12: Top-most left corner point (White's 13-point)
   - Index 13: White's 14-point
   - Index 14: White's 15-point
   - Index 15: White's 16-point
   - Index 16: White's 17-point
   - Index 17: Point immediately to the left of the central bar on the top row (White's 18-point)
4. Top-Right Quadrant (Black's Home Board, 6 points from LEFT to RIGHT, starting next to the central bar):
   - Index 18: Point immediately to the right of the central bar on the top row (White's 19-point / Black's 6-point)
   - Index 19: White's 20-point (Black's 5-point)
   - Index 20: White's 21-point (Black's 4-point)
   - Index 21: White's 22-point (Black's 3-point)
   - Index 22: White's 23-point (Black's 2-point)
   - Index 23: Top-most right corner point (White's 24-point / Black's 1-point)

CHECKER ENCODING:
- Light-colored checkers (White, Ivory, Cream, Light Wood, Blue/Cyan in some apps) = POSITIVE integer (+1 to +15).
- Dark-colored checkers (Black, Red, Brown, Dark Wood, Charcoal) = NEGATIVE integer (-1 to -15).
- Empty point = 0.
- Count stacked checkers carefully (if a numeric badge like "6" or "7" is shown on a stack in an app screenshot, use that exact number).
- Total checkers per player across points + bar + borneOff cannot exceed 15.

OUTPUT JSON SCHEMA (Return ONLY valid JSON matching this exact structure):
{
  "points": [24 integers from index 0 to index 23],
  "bar": { "white": <integer 0..15>, "black": <integer 0..15> },
  "borneOff": { "white": <integer 0..15>, "black": <integer 0..15> },
  "dice": [<die1 1..6>, <die2 1..6>] or null if no dice are rolled/visible,
  "detectedTurn": "white" | "black" | null,
  "notes": "<Short Hebrew summary of what was detected, e.g. 'זוהתה עמדת מגע עם קוביות 6-4 לתור הלבן'>"
}`;

/**
 * Client-side canvas image preprocessor:
 * Automatically resizes large photos/screenshots to max `1600px` dimension and applies
 * mild contrast normalization before sending to Gemini Vision for faster, sharper recognition.
 * Safely falls back to the original data URL in headless/test environments.
 */
export async function preprocessBoardImage(
  base64DataUrl: string,
  maxDimension = 1600
): Promise<string> {
  if (
    typeof document === 'undefined' ||
    typeof Image === 'undefined' ||
    !base64DataUrl.startsWith('data:image/')
  ) {
    return base64DataUrl;
  }

  return new Promise<string>((resolve) => {
    const img = new Image();
    img.onload = () => {
      try {
        let width = img.naturalWidth || img.width;
        let height = img.naturalHeight || img.height;
        if (!width || !height) {
          resolve(base64DataUrl);
          return;
        }

        if (width > maxDimension || height > maxDimension) {
          const scale = maxDimension / Math.max(width, height);
          width = Math.round(width * scale);
          height = Math.round(height * scale);
        }

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          resolve(base64DataUrl);
          return;
        }

        ctx.filter = 'contrast(1.08) saturate(1.04)';
        ctx.drawImage(img, 0, 0, width, height);
        const outUrl = canvas.toDataURL('image/jpeg', 0.9);
        resolve(outUrl || base64DataUrl);
      } catch {
        resolve(base64DataUrl);
      }
    };
    img.onerror = () => resolve(base64DataUrl);
    img.src = base64DataUrl;
  });
}

function parseDataUrl(base64DataUrl: string): {
  mimeType: string;
  data: string;
} {
  const trimmed = base64DataUrl.trim();
  const match = trimmed.match(/^data:([^;]+);base64,(.+)$/s);
  if (match) {
    return {
      mimeType: match[1] || 'image/jpeg',
      data: match[2].replace(/\s/g, ''),
    };
  }
  return {
    mimeType: 'image/jpeg',
    data: trimmed.replace(/\s/g, ''),
  };
}

function extractJsonFromText(rawText: string): Partial<ScannedBoardResponse> {
  const cleaned = rawText
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();

  try {
    return JSON.parse(cleaned) as Partial<ScannedBoardResponse>;
  } catch {
    const firstBrace = cleaned.indexOf('{');
    const lastBrace = cleaned.lastIndexOf('}');
    if (firstBrace !== -1 && lastBrace > firstBrace) {
      return JSON.parse(
        cleaned.slice(firstBrace, lastBrace + 1)
      ) as Partial<ScannedBoardResponse>;
    }
    throw new Error('לא ניתן היה לפענח את תשובת ה-JSON ממנוע הסריקה.');
  }
}

/**
 * Normalizes and sanitizes a raw or partial `ScannedBoardResponse` into a strictly valid `BoardState`.
 * - Clamps all 24 points to integers in [-15, 15].
 * - Caps total checkers per player (on board + bar) at 15.
 * - Automatically infers `borneOff` when `borneOff` is 0/omitted and a player has between 1 and 14 checkers on board + bar
 *   (common when real-life photos crop out the external bear-off trays).
 */
export function normalizeScannedBoard(
  raw: Partial<ScannedBoardResponse>
): BoardState {
  const rawPoints = Array.isArray(raw.points) ? raw.points : [];
  const points = new Array<number>(24).fill(0);

  for (let i = 0; i < 24; i++) {
    const val = Number(rawPoints[i]);
    if (Number.isFinite(val) && val !== 0) {
      const clamped = Math.max(-15, Math.min(15, Math.round(val)));
      points[i] = clamped === 0 ? 0 : clamped;
    }
  }

  let barWhite = Math.max(
    0,
    Math.min(15, Math.round(Number(raw.bar?.white) || 0))
  );
  let barBlack = Math.max(
    0,
    Math.min(15, Math.round(Number(raw.bar?.black) || 0))
  );

  // Enforce maximum 15 checkers for White across points + bar
  let whiteActive =
    points.reduce((sum, v) => (v > 0 ? sum + v : sum), 0) + barWhite;
  if (whiteActive > 15) {
    let excess = whiteActive - 15;
    if (barWhite > 0) {
      const reduceBar = Math.min(barWhite, excess);
      barWhite -= reduceBar;
      excess -= reduceBar;
    }
    while (excess > 0) {
      let maxIdx = -1;
      let maxVal = 0;
      for (let i = 0; i < 24; i++) {
        if (points[i] > maxVal) {
          maxVal = points[i];
          maxIdx = i;
        }
      }
      if (maxIdx === -1) break;
      points[maxIdx] -= 1;
      excess -= 1;
    }
    whiteActive = 15;
  }

  // Enforce maximum 15 checkers for Black across points + bar
  let blackActive =
    points.reduce((sum, v) => (v < 0 ? sum + Math.abs(v) : sum), 0) + barBlack;
  if (blackActive > 15) {
    let excess = blackActive - 15;
    if (barBlack > 0) {
      const reduceBar = Math.min(barBlack, excess);
      barBlack -= reduceBar;
      excess -= reduceBar;
    }
    while (excess > 0) {
      let minIdx = -1;
      let minVal = 0;
      for (let i = 0; i < 24; i++) {
        if (points[i] < minVal) {
          minVal = points[i];
          minIdx = i;
        }
      }
      if (minIdx === -1) break;
      points[minIdx] += 1;
      if (points[minIdx] === 0) points[minIdx] = 0;
      excess -= 1;
    }
    blackActive = 15;
  }

  // Borne-off calculation:
  // If raw.borneOff is explicitly > 0, clamp to remaining capacity (15 - active).
  // If raw.borneOff is 0 or omitted and 0 < active < 15, infer borneOff = 15 - active.
  const rawBorneWhite = Math.max(0, Math.round(Number(raw.borneOff?.white) || 0));
  const rawBorneBlack = Math.max(0, Math.round(Number(raw.borneOff?.black) || 0));

  const borneWhite =
    rawBorneWhite > 0
      ? Math.min(15 - whiteActive, rawBorneWhite)
      : whiteActive > 0 && whiteActive < 15
        ? 15 - whiteActive
        : 0;

  const borneBlack =
    rawBorneBlack > 0
      ? Math.min(15 - blackActive, rawBorneBlack)
      : blackActive > 0 && blackActive < 15
        ? 15 - blackActive
        : 0;

  const currentTurn: Player =
    raw.detectedTurn === 'black' ? 'black' : 'white';

  let dice: number[] = [];
  let remainingMoves: number[] = [];

  if (
    Array.isArray(raw.dice) &&
    raw.dice.length >= 2 &&
    Number.isFinite(Number(raw.dice[0])) &&
    Number.isFinite(Number(raw.dice[1]))
  ) {
    const d1 = Math.max(1, Math.min(6, Math.round(Number(raw.dice[0]))));
    const d2 = Math.max(1, Math.min(6, Math.round(Number(raw.dice[1]))));
    dice = d1 === d2 ? [d1, d1, d1, d1] : [d1, d2];
    remainingMoves = [...dice];
  }

  return {
    points,
    bar: {
      white: barWhite,
      black: barBlack,
    },
    borneOff: {
      white: borneWhite,
      black: borneBlack,
    },
    currentTurn,
    dice,
    remainingMoves,
  };
}

/**
 * Rotates the physical Backgammon board 180 degrees in-plane:
 * - Bottom-Right (0..5) <-> Top-Left (12..17)
 * - Bottom-Left (6..11) <-> Top-Right (18..23)
 * Preserves checker colors and counts, and applying `flipBoard180` twice returns the exact original board.
 */
export function flipBoard180(state: BoardState): BoardState {
  const newPoints = new Array<number>(24).fill(0);
  for (let i = 0; i < 24; i++) {
    const val = state.points[(i + 12) % 24];
    newPoints[i] = val === 0 ? 0 : val;
  }

  return {
    points: newPoints,
    bar: { ...state.bar },
    borneOff: { ...state.borneOff },
    currentTurn: state.currentTurn,
    dice: [...state.dice],
    remainingMoves: [...state.remainingMoves],
  };
}

/**
 * Swaps White and Black checker ownership and mirrors home-board direction (point i <-> 23 - i)
 * so White always moves 23 -> 0 and Black always moves 0 -> 23.
 * Preserves exact evaluation antisymmetry: `evaluatePosition(swapBoardColors(S)).equity === -evaluatePosition(S).equity`.
 */
export function swapBoardColors(state: BoardState): BoardState {
  const mirroredPoints = new Array<number>(24).fill(0);
  for (let i = 0; i < 24; i++) {
    const val = state.points[23 - i];
    mirroredPoints[i] = val === 0 ? 0 : -val;
  }

  return {
    points: mirroredPoints,
    bar: {
      white: state.bar.black,
      black: state.bar.white,
    },
    borneOff: {
      white: state.borneOff.black,
      black: state.borneOff.white,
    },
    currentTurn: state.currentTurn === 'white' ? 'black' : 'white',
    dice: [...state.dice],
    remainingMoves: [...state.remainingMoves],
  };
}

export const GEMINI_VISION_MODELS = [
  'gemini-2.5-flash',
  'gemini-2.5-flash-lite',
  'gemini-2.0-flash',
] as const;

const RETRYABLE_HTTP_STATUSES = new Set([404, 429, 503]);

/**
 * Calls Google Gemini Vision REST API with a multi-model fallback cascade
 * (`gemini-2.5-flash` -> `gemini-2.5-flash-lite` -> `gemini-2.0-flash`) on HTTP 429/503/404 errors
 * to analyze a Backgammon board screenshot or photo and return a structured `ScannedBoardResponse`.
 */
export async function scanBackgammonImage(
  base64DataUrl: string,
  apiKey: string
): Promise<ScannedBoardResponse> {
  const cleanKey = apiKey.trim();
  if (!cleanKey) {
    throw new Error('נא להזין מפתח Gemini API תקין כדי לסרוק את התמונה.');
  }

  const processedUrl = await preprocessBoardImage(base64DataUrl);
  const { mimeType, data } = parseDataUrl(processedUrl);
  if (!data) {
    throw new Error('נתוני התמונה חסרים או לא תקינים.');
  }

  let lastError: Error | null = null;

  for (let i = 0; i < GEMINI_VISION_MODELS.length; i++) {
    const modelName = GEMINI_VISION_MODELS[i];
    const isLastModel = i === GEMINI_VISION_MODELS.length - 1;
    const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${encodeURIComponent(cleanKey)}`;

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contents: [
            {
              parts: [
                { text: GEMINI_VISION_SYSTEM_PROMPT },
                {
                  inlineData: {
                    mimeType,
                    data,
                  },
                },
              ],
            },
          ],
          generationConfig: {
            responseMimeType: 'application/json',
            temperature: 0.1,
          },
        }),
      });

      if (!response.ok) {
        const errBody = await response.text().catch(() => '');
        // Automatically cascade to the next fallback model on 429, 503, or 404
        if (RETRYABLE_HTTP_STATUSES.has(response.status) && !isLastModel) {
          continue;
        }
        if (response.status === 429) {
          throw new Error(
            'חריגה ממכסת הבקשות של Gemini API (429) בכל המודלים. המתן מספר שניות ונסה שוב, או ערוך את הלוח ידנית.'
          );
        }
        if (response.status === 503) {
          throw new Error(
            'שרתי Gemini Vision עמוסים כעת (503). נסה שוב בעוד מספר שניות או ערוך את הלוח ידנית.'
          );
        }
        if (response.status === 400 || response.status === 403) {
          throw new Error(
            `שגיאת אימות מול Gemini API (${response.status}). בדוק שמפתח ה-API תקין.`
          );
        }
        throw new Error(
          `שגיאה בקריאה ל-Gemini Vision (${response.status}): ${errBody.slice(0, 120)}`
        );
      }

      const payload = await response.json();
      const rawText: string | undefined =
        payload?.candidates?.[0]?.content?.parts?.[0]?.text;

      if (!rawText) {
        throw new Error('מנוע ה-AI לא החזיר תוצאת סריקה לתמונה זו.');
      }

      const parsed = extractJsonFromText(rawText);
      const normalizedState = normalizeScannedBoard(parsed);

      let parsedDice: [number, number] | null = null;
      if (
        Array.isArray(parsed.dice) &&
        parsed.dice.length >= 2 &&
        Number.isFinite(Number(parsed.dice[0])) &&
        Number.isFinite(Number(parsed.dice[1]))
      ) {
        parsedDice = [
          Math.max(1, Math.min(6, Math.round(Number(parsed.dice[0])))),
          Math.max(1, Math.min(6, Math.round(Number(parsed.dice[1])))),
        ];
      }

      const detectedTurn: Player | null =
        parsed.detectedTurn === 'white' || parsed.detectedTurn === 'black'
          ? parsed.detectedTurn
          : null;

      return {
        points: normalizedState.points,
        bar: normalizedState.bar,
        borneOff: normalizedState.borneOff,
        dice: parsedDice,
        detectedTurn,
        notes:
          typeof parsed.notes === 'string' && parsed.notes.trim().length > 0
            ? parsed.notes.trim()
            : 'הלוח נסרק בהצלחה. מומלץ לוודא את מיקום הכלים לפני טעינת הניתוח.',
      };
    } catch (err) {
      lastError =
        err instanceof Error
          ? err
          : new Error('אירעה שגיאה בלתי צפויה בסריקת התמונה.');
      if (isLastModel) {
        break;
      }
    }
  }

  throw (
    lastError ?? new Error('סריקת התמונה נכשלה. ניתן גם לערוך את הלוח ידנית.')
  );
}
