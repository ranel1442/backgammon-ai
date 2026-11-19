import type { BoardState, Player } from './types';

export interface OpeningBookMoveEntry {
  /** Rank among book responses for this roll (1 = best theoretical play) */
  rank: 1 | 2 | 3;
  /** Standard player-relative notation (e.g., '8/5 6/5', '13/7 8/7', '24/13') */
  canonicalNotation: string;
  /**
   * Sequence of steps in 1-based player-relative point numbers [fromPoint, toPoint]
   * where 24 is the player's furthest back point and 1 is the deepest home point.
   */
  stepsRel: Array<[number, number]>;
  /**
   * Theoretical rollout equity bonus (in player's favor) to ensure
   * rollout-proven book moves win close static/2-ply ties on move 1 & move 2.
   */
  equityAdjustment: number;
  /** Rich Hebrew strategic explanations for why this opening/reply move is played */
  explanation: string[];
  /** Whether this entry comes from the 1st-roll opening book or 2nd-turn reply book */
  bookPhase?: 'opening' | 'reply';
}

/**
 * Extracts the player-relative 24-point array from any `BoardState` for `player`
 * (positive = `player`'s checkers, negative = opponent's checkers, index 23 = player's 24-point, index 0 = player's 1-point).
 */
export function extractPlayerRelativePoints(
  state: BoardState,
  player: Player
): number[] {
  const relPoints = new Array<number>(24).fill(0);
  const isWhite = player === 'white';
  for (let rel = 0; rel < 24; rel++) {
    const absIdx = isWhite ? rel : 23 - rel;
    const raw = state.points[absIdx];
    relPoints[rel] = isWhite ? raw : raw === 0 ? 0 : -raw;
  }
  return relPoints;
}

/**
 * Checks whether `player`'s own checkers are in their untouched initial starting configuration:
 * 2 on relative 24-point (idx 23), 5 on 13-point (idx 12), 3 on 8-point (idx 7), 5 on 6-point (idx 5),
 * and 0 on the bar or borne off.
 */
function hasUntouchedStartingCheckers(
  state: BoardState,
  player: Player
): boolean {
  if (state.bar[player] !== 0 || state.borneOff[player] !== 0) {
    return false;
  }
  const rel = extractPlayerRelativePoints(state, player);
  if (rel[23] !== 2 || rel[12] !== 5 || rel[7] !== 3 || rel[5] !== 5) {
    return false;
  }
  for (let i = 0; i < 24; i++) {
    if (i !== 23 && i !== 12 && i !== 7 && i !== 5 && rel[i] > 0) {
      return false;
    }
  }
  return true;
}

/**
 * Checks whether a BoardState is the exact initial starting configuration
 * (Roll 1, before the first move of the match).
 */
export function isInitialBoardState(state: BoardState): boolean {
  return (
    hasUntouchedStartingCheckers(state, 'white') &&
    hasUntouchedStartingCheckers(state, 'black')
  );
}

/**
 * Checks whether a BoardState is a 2nd-Turn Reply position (Roll 2):
 * The active player (`state.currentTurn`) still has all 15 checkers on their initial starting points
 * (and 0 on the bar / borne off), while the opponent has already executed their 1st-turn opening move.
 */
export function isSecondTurnReplyState(state: BoardState): boolean {
  const player = state.currentTurn;
  const opponent: Player = player === 'white' ? 'black' : 'white';
  if (!hasUntouchedStartingCheckers(state, player)) {
    return false;
  }
  if (hasUntouchedStartingCheckers(state, opponent)) {
    return false; // Still Turn 1
  }
  if (state.borneOff[opponent] !== 0) {
    return false;
  }
  return true;
}

/**
 * Returns true when `state` is covered by either the 1st-Roll Opening Book or the 2nd-Turn Reply Book.
 */
export function isOpeningOrReplyBookState(state: BoardState): boolean {
  return isInitialBoardState(state) || isSecondTurnReplyState(state);
}

/**
 * Theoretical Backgammon Opening Book covering all 15 non-double opening rolls
 * (and standard double rolls on the initial board) with rollout-backed rankings
 * and idiomatic Hebrew coaching explanations.
 */
const OPENING_BOOK_BY_ROLL: Record<string, OpeningBookMoveEntry[]> = {
  '3-1': [
    {
      rank: 1,
      canonicalNotation: '8/5 6/5',
      stepsRel: [
        [8, 5],
        [6, 5],
      ],
      equityAdjustment: 0.14,
      bookPhase: 'opening',
      explanation: [
        'תיאוריית פתיחה (המהלך המושלם ל-3-1): סגירת נקודת ה-5 הקריטית בבית לבניית חומה מוקדמת',
        'יוצר בסיס לחומת חסימה (Prime) ומקשה מאוד על בריחת הכלים האחוריים של היריב',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '24/21 13/12',
      stepsRel: [
        [24, 21],
        [13, 12],
      ],
      equityAdjustment: 0.01,
      bookPhase: 'opening',
      explanation: [
        'מפצל את הכלים האחוריים לנקודה 21, אך מוותר על הזדמנות פז לסגור את נקודת ה-5 בבית',
      ],
    },
  ],
  '6-1': [
    {
      rank: 1,
      canonicalNotation: '13/7 8/7',
      stepsRel: [
        [13, 7],
        [8, 7],
      ],
      equityAdjustment: 0.13,
      bookPhase: 'opening',
      explanation: [
        'תיאוריית פתיחה (המהלך המיטבי ל-6-1): בניית נקודת הבר (נקודה 7) היוצרת רצף חסימה של 3 נקודות (8-7-6)',
        'שולט בלוח החיצוני וחוסם קוביות 6 של היריב בניסיונו לברוח מהבית שלנו',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '24/18 13/12',
      stepsRel: [
        [24, 18],
        [13, 12],
      ],
      equityAdjustment: 0.01,
      bookPhase: 'opening',
      explanation: [
        'מבריח כלי אחורי לנקודת הבר של היריב, אך פחות חזק מסגירת נקודת ה-7 הבטוחה',
      ],
    },
  ],
  '4-2': [
    {
      rank: 1,
      canonicalNotation: '8/4 6/4',
      stepsRel: [
        [8, 4],
        [6, 4],
      ],
      equityAdjustment: 0.12,
      bookPhase: 'opening',
      explanation: [
        'תיאוריית פתיחה (המהלך המיטבי ל-4-2): סגירת נקודת ה-4 החזקה בבית ללא השארת כלים חשופים',
        'מחזק משמעותית את מבנה הבית ומגדיל את האיום במקרה של הכאת כלי יריב בהמשך',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '24/20 13/11',
      stepsRel: [
        [24, 20],
        [13, 11],
      ],
      equityAdjustment: 0.01,
      bookPhase: 'opening',
      explanation: [
        'מפצל לנקודת ה-5 של היריב ומוריד בנאי, אך מוותר על סגירה מיידית של נקודת ה-4 בבית',
      ],
    },
  ],
  '6-5': [
    {
      rank: 1,
      canonicalNotation: '24/18 18/13',
      stepsRel: [
        [24, 18],
        [18, 13],
      ],
      equityAdjustment: 0.12,
      bookPhase: 'opening',
      explanation: [
        'תיאוריית פתיחה ("קפיצת האוהבים" — Lover\'s Leap): הברחת כלי אחורי כל הדרך מנקודה 24 לנקודת ה-13 הבטוחה',
        'מצמצם את הסכנה להילכד מאחורי חומה של היריב ואינו משאיר אף כלי חשוף (Blot) על הלוח',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '24/18 13/8',
      stepsRel: [
        [24, 18],
        [13, 8],
      ],
      equityAdjustment: 0.02,
      bookPhase: 'opening',
      explanation: [
        'מפצל כלי אחורי לנקודת הבר של היריב (18) ומתגבר את נקודת ה-8 במרכז הלוח',
      ],
    },
  ],
  '5-3': [
    {
      rank: 1,
      canonicalNotation: '8/3 6/3',
      stepsRel: [
        [8, 3],
        [6, 3],
      ],
      equityAdjustment: 0.1,
      bookPhase: 'opening',
      explanation: [
        'תיאוריית פתיחה (המהלך המוביל ל-5-3): סגירת נקודת ה-3 בבית לבניית מבנה פנימי יציב וללא חשיפת כלים',
        'מייצר נקודת נחיתה בטוחה לכלים מהלוח החיצוני ומחזק את כוח הסגירה בבית',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '13/8 13/10',
      stepsRel: [
        [13, 8],
        [13, 10],
      ],
      equityAdjustment: 0.04,
      bookPhase: 'opening',
      explanation: [
        'קו בנייה חלופי: הורדת שני בנאים מנקודת ה-13 ללוח החיצוני (נקודות 8 ו-10) לשליטה במרכז',
      ],
    },
  ],
  '6-4': [
    {
      rank: 1,
      canonicalNotation: '24/18 13/9',
      stepsRel: [
        [24, 18],
        [13, 9],
      ],
      equityAdjustment: 0.09,
      bookPhase: 'opening',
      explanation: [
        'תיאוריית פתיחה (הקו המודרני ל-6-4): פיצול כלי אחורי לנקודת הבר של היריב (18) והורדת בנאי אקטיבי לנקודה 9',
        'משלב לחץ על עוגן מתקדם עם גמישות מרבית לבניית נקודות ה-5 וה-7 בתור הבא',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '8/2 6/2',
      stepsRel: [
        [8, 2],
        [6, 2],
      ],
      equityAdjustment: 0.06,
      bookPhase: 'opening',
      explanation: [
        'סגירת נקודת ה-2 בבית: מהלך סולידי ובטוח לחלוטין שמונע השארת כלים חשופים',
      ],
    },
    {
      rank: 3,
      canonicalNotation: '24/18 18/14',
      stepsRel: [
        [24, 18],
        [18, 14],
      ],
      equityAdjustment: 0.04,
      bookPhase: 'opening',
      explanation: [
        'ריצה מהירה עם כלי אחורי לנקודה 14: מקטין את החשיפה לפגיעות ישירות ומקדם את המרוץ',
      ],
    },
  ],
  '6-3': [
    {
      rank: 1,
      canonicalNotation: '24/18 13/10',
      stepsRel: [
        [24, 18],
        [13, 10],
      ],
      equityAdjustment: 0.09,
      bookPhase: 'opening',
      explanation: [
        'תיאוריית פתיחה (המהלך המיטבי ל-6-3): יציאה עם כלי אחורי לנקודה 18 והורדת בנאי מרכזי לנקודה 10',
        'ממקם שני כלים בנקודות אסטרטגיות שמכסות הן בניית עוגן מתקדם והן סגירת נקודות בבית',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '24/18 18/15',
      stepsRel: [
        [24, 18],
        [18, 15],
      ],
      equityAdjustment: 0.04,
      bookPhase: 'opening',
      explanation: [
        'ריצה רציפה עם כלי אחורי לנקודה 15 כדי לצאת מטווח האש של הבית של היריב',
      ],
    },
  ],
  '6-2': [
    {
      rank: 1,
      canonicalNotation: '24/18 13/11',
      stepsRel: [
        [24, 18],
        [13, 11],
      ],
      equityAdjustment: 0.09,
      bookPhase: 'opening',
      explanation: [
        'תיאוריית פתיחה (המהלך המיטבי ל-6-2): פיצול כלי אחורי לנקודה 18 והורדת בנאי חיוני לנקודה 11',
        'הבנאי בנקודה 11 עובד בסינרגיה מושלמת עם נקודות 8 ו-6 לכיסוי נקודות ה-5 וה-7',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '13/7 13/11',
      stepsRel: [
        [13, 7],
        [13, 11],
      ],
      equityAdjustment: 0.03,
      bookPhase: 'opening',
      explanation: [
        'הורדת שני בנאים ללוח החיצוני (7 ו-11) ליצירת לחץ התקפי, תוך השארת הכלים האחוריים במקום',
      ],
    },
  ],
  '5-4': [
    {
      rank: 1,
      canonicalNotation: '13/8 13/9',
      stepsRel: [
        [13, 8],
        [13, 9],
      ],
      equityAdjustment: 0.09,
      bookPhase: 'opening',
      explanation: [
        'תיאוריית פתיחה (המהלך המוביל ל-5-4): הורדת שני כלים מנקודת ה-13 לנקודות 8 ו-9 בלוח החיצוני',
        'שומר על בטיחות מלאה מהכאה ישירה ומעניק שליטה אדירה בבנאים לקראת התור הבא',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '24/20 13/8',
      stepsRel: [
        [24, 20],
        [13, 8],
      ],
      equityAdjustment: 0.05,
      bookPhase: 'opening',
      explanation: [
        'פיצול כלי אחורי לנקודת ה-5 של היריב (נקודה 20) לצד תגבור נקודת ה-8 המרכזית',
      ],
    },
  ],
  '5-2': [
    {
      rank: 1,
      canonicalNotation: '13/8 13/11',
      stepsRel: [
        [13, 8],
        [13, 11],
      ],
      equityAdjustment: 0.09,
      bookPhase: 'opening',
      explanation: [
        'תיאוריית פתיחה (המהלך הקלאסי ל-5-2): פיתוח שני כלים מה-13 לנקודות 8 ו-11 ללא סיכון להכאה ישירה',
        'בונה עתודת בנאים גמישה בלוח החיצוני שתאפשר סגירת נקודות מפתח בתור הבא',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '24/22 13/8',
      stepsRel: [
        [24, 22],
        [13, 8],
      ],
      equityAdjustment: 0.04,
      bookPhase: 'opening',
      explanation: [
        'פיצול כלי אחורי לנקודה 22 ותגבור נקודת ה-8 במרכז הלוח',
      ],
    },
  ],
  '5-1': [
    {
      rank: 1,
      canonicalNotation: '24/23 13/8',
      stepsRel: [
        [24, 23],
        [13, 8],
      ],
      equityAdjustment: 0.08,
      bookPhase: 'opening',
      explanation: [
        'תיאוריית פתיחה (המהלך המאזן ל-5-1): פיצול הכלי האחורי לנקודה 23 ותגבור נקודת ה-8 מה-13',
        'מפתח גמישות בשני צידי הלוח תוך צמצום חשיפה למתקפה כפולה של היריב',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '13/8 6/5',
      stepsRel: [
        [13, 8],
        [6, 5],
      ],
      equityAdjustment: 0.05,
      bookPhase: 'opening',
      explanation: [
        'משחק התקפי (Slot): תגבור נקודת ה-8 והצבת כלי חלוץ על נקודת ה-5 הזהובה כדי לתפוס יוזמה',
      ],
    },
  ],
  '4-3': [
    {
      rank: 1,
      canonicalNotation: '13/9 13/10',
      stepsRel: [
        [13, 9],
        [13, 10],
      ],
      equityAdjustment: 0.09,
      bookPhase: 'opening',
      explanation: [
        'תיאוריית פתיחה (המהלך המיטבי ל-4-3): הורדת שני בנאים לנקודות 9 ו-10 בלוח החיצוני',
        'מייצר מקסימום קומבינציות קובייה לסגירת נקודות ה-5, ה-4 וה-7 בתור הבא',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '24/20 13/10',
      stepsRel: [
        [24, 20],
        [13, 10],
      ],
      equityAdjustment: 0.06,
      bookPhase: 'opening',
      explanation: [
        'פיצול כלי אחורי לנקודת ה-5 של היריב (20) והורדת בנאי גמיש לנקודה 10',
      ],
    },
    {
      rank: 3,
      canonicalNotation: '24/21 13/9',
      stepsRel: [
        [24, 21],
        [13, 9],
      ],
      equityAdjustment: 0.04,
      bookPhase: 'opening',
      explanation: [
        'פיצול כלי אחורי לנקודה 21 בשילוב הורדת בנאי לנקודה 9',
      ],
    },
  ],
  '4-1': [
    {
      rank: 1,
      canonicalNotation: '24/23 13/9',
      stepsRel: [
        [24, 23],
        [13, 9],
      ],
      equityAdjustment: 0.08,
      bookPhase: 'opening',
      explanation: [
        'תיאוריית פתיחה (המהלך המוביל ל-4-1): פיצול הכלי האחורי לנקודה 23 והורדת בנאי אקטיבי לנקודה 9',
        'משלב הכנה לעוגן הגנתי עם הצבת בנאי שמאיים לסגור את נקודת ה-5 בתור הבא',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '13/9 6/5',
      stepsRel: [
        [13, 9],
        [6, 5],
      ],
      equityAdjustment: 0.05,
      bookPhase: 'opening',
      explanation: [
        'הצבת כלי חלוץ (Slot) אגרסיבי על נקודת ה-5 הזהובה וגיבויו עם בנאי בנקודה 9',
      ],
    },
  ],
  '3-2': [
    {
      rank: 1,
      canonicalNotation: '24/21 13/11',
      stepsRel: [
        [24, 21],
        [13, 11],
      ],
      equityAdjustment: 0.08,
      bookPhase: 'opening',
      explanation: [
        'תיאוריית פתיחות (המהלך המודרני ל-3-2): פיצול כלי אחורי לנקודה 21 והורדת בנאי אסטרטגי לנקודה 11',
        'מאזן בין שחרור הכלים האחוריים לבין פיתוח בנאי מרכזי בלוח החיצוני',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '13/10 13/11',
      stepsRel: [
        [13, 10],
        [13, 11],
      ],
      equityAdjustment: 0.06,
      bookPhase: 'opening',
      explanation: [
        'פיתוח כפול בלוח החיצוני: הורדת שני בנאים לנקודות 10 ו-11 ללא פיצול העורף',
      ],
    },
  ],
  '2-1': [
    {
      rank: 1,
      canonicalNotation: '13/11 6/5',
      stepsRel: [
        [13, 11],
        [6, 5],
      ],
      equityAdjustment: 0.08,
      bookPhase: 'opening',
      explanation: [
        'תיאוריית פתיחה (המהלך היוזם ל-2-1): הורדת בנאי לנקודה 11 והצבת כלי חלוץ (Slot) על נקודת ה-5 הזהובה',
        'מגדיל משמעותית את הסיכוי לסגור את נקודת ה-5 החזקה כבר בתור הבא',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '24/23 13/11',
      stepsRel: [
        [24, 23],
        [13, 11],
      ],
      equityAdjustment: 0.06,
      bookPhase: 'opening',
      explanation: [
        'קו סולידי: פיצול הכלי האחורי לנקודה 23 והורדת בנאי לנקודה 11 עם פחות סיכון להכאה',
      ],
    },
  ],
};

/**
 * 2nd-Turn Reply Book (Roll 2):
 * Covers tactical hits, golden-point responses, splitting vs. slotting, and all 6 double rolls
 * from the replying player's canonical perspective.
 */
const SECOND_TURN_REPLY_BOOK_BY_ROLL: Record<string, OpeningBookMoveEntry[]> = {
  '3-1': [
    {
      rank: 1,
      canonicalNotation: '8/5 6/5',
      stepsRel: [
        [8, 5],
        [6, 5],
      ],
      equityAdjustment: 0.15,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (3-1): סגירת נקודת ה-5 הקריטית בבית (כולל הכאה על נקודת ה-5 אם היריב התפצל אליה)',
        'מעניק יתרון מבני מיידי ומציב חומת חסימה מול הכלים האחוריים של היריב',
      ],
    },
  ],
  '6-1': [
    {
      rank: 1,
      canonicalNotation: '13/7 8/7',
      stepsRel: [
        [13, 7],
        [8, 7],
      ],
      equityAdjustment: 0.14,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (6-1): סגירת נקודת הבר (נקודה 7) ובניית חומה של 3 נקודות (8-7-6)',
        'כולא את הכלים האחוריים של היריב ומונע ממנו לברוח בקוביות 6',
      ],
    },
  ],
  '4-2': [
    {
      rank: 1,
      canonicalNotation: '8/4 6/4',
      stepsRel: [
        [8, 4],
        [6, 4],
      ],
      equityAdjustment: 0.13,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (4-2): סגירת נקודת ה-4 בבית ליצירת מבנה תקיפה וחסימה יציב',
        'מחזק את הבית לקראת חילופי מהלומות ומונע מהיריב להתבסס על נקודת ה-4',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '24/20 13/11',
      stepsRel: [
        [24, 20],
        [13, 11],
      ],
      equityAdjustment: 0.06,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה: הכאה או תפיסת נקודת ה-5 של היריב (20) בשילוב הורדת בנאי לנקודה 11',
      ],
    },
  ],
  '6-5': [
    {
      rank: 1,
      canonicalNotation: '24/18 18/13',
      stepsRel: [
        [24, 18],
        [18, 13],
      ],
      equityAdjustment: 0.13,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (6-5): הברחת כלי אחורי מנקודה 24 כל הדרך לנקודת ה-13 הבטוחה (Lover\'s Leap)',
        'מנטרל את איומי החסימה של היריב ומאזן מיד את מרוץ הפיפס',
      ],
    },
  ],
  '5-3': [
    {
      rank: 1,
      canonicalNotation: '8/3 6/3',
      stepsRel: [
        [8, 3],
        [6, 3],
      ],
      equityAdjustment: 0.11,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (5-3): סגירת נקודת ה-3 בבית לבניית בית חזק מול פתיחת היריב',
        'מייצר נקודה שנייה בבית ללא חשיפת כלים בודדים',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '13/8 13/10',
      stepsRel: [
        [13, 8],
        [13, 10],
      ],
      equityAdjustment: 0.05,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה: הורדת שני בנאים לנקודות 8 ו-10 כאשר היריב פתח במהלך פסיבי',
      ],
    },
  ],
  '6-4': [
    {
      rank: 1,
      canonicalNotation: '24/18 13/9',
      stepsRel: [
        [24, 18],
        [13, 9],
      ],
      equityAdjustment: 0.1,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (6-4): פיצול כלי אחורי לנקודת הבר של היריב (18) והורדת בנאי לנקודה 9',
        'לוחץ על הלוח החיצוני של היריב ומכין סגירת נקודות מפתח בבית',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '8/2 6/2',
      stepsRel: [
        [8, 2],
        [6, 2],
      ],
      equityAdjustment: 0.08,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה: סגירת נקודת ה-2 בבית (במיוחד כאשר כולאים כלי יריב או כשנקודה 18 תפוסה)',
      ],
    },
    {
      rank: 3,
      canonicalNotation: '24/20 20/14',
      stepsRel: [
        [24, 20],
        [20, 14],
      ],
      equityAdjustment: 0.07,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה: הכאה בנקודה 20 ויציאה לנקודה 14 כאשר היריב הציב כלי חלוץ (Slot) בנקודת ה-5 שלו',
      ],
    },
  ],
  '6-3': [
    {
      rank: 1,
      canonicalNotation: '24/18 13/10',
      stepsRel: [
        [24, 18],
        [13, 10],
      ],
      equityAdjustment: 0.1,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (6-3): פיצול כלי אחורי לנקודה 18 והורדת בנאי מרכזי לנקודה 10',
        'משיג אחיזה בנקודת הבר של היריב ובונה גמישות טקטית במרכז הלוח',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '24/21 21/15',
      stepsRel: [
        [24, 21],
        [21, 15],
      ],
      equityAdjustment: 0.06,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה: יציאה עם כלי אחורי לנקודה 15 כאשר נקודה 18 של היריב חסומה',
      ],
    },
  ],
  '6-2': [
    {
      rank: 1,
      canonicalNotation: '24/18 13/11',
      stepsRel: [
        [24, 18],
        [13, 11],
      ],
      equityAdjustment: 0.1,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (6-2): פיצול לנקודה 18 והורדת בנאי לנקודה 11',
        'משלב מאבק על עוגן מתקדם עם בנאי מצוין לכיסוי נקודת ה-5 בבית',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '24/22 22/16',
      stepsRel: [
        [24, 22],
        [22, 16],
      ],
      equityAdjustment: 0.06,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה: בריחה עם כלי אחורי לנקודה 16 כאשר נקודת הבר של היריב (18) סגורה',
      ],
    },
  ],
  '5-4': [
    {
      rank: 1,
      canonicalNotation: '13/8 13/9',
      stepsRel: [
        [13, 8],
        [13, 9],
      ],
      equityAdjustment: 0.1,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (5-4): הורדת שני בנאים לנקודות 8 ו-9 לשליטה מלאה בלוח החיצוני',
        'מכין תשתית חזקה לסגירת נקודות בבית מבלי להשאיר כלים חשופים',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '24/20 13/8',
      stepsRel: [
        [24, 20],
        [13, 8],
      ],
      equityAdjustment: 0.08,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה: פיצול כלים אחוריים לנקודה 20 להשגת עוגן הגנתי מול בניית הבית של היריב',
      ],
    },
  ],
  '5-2': [
    {
      rank: 1,
      canonicalNotation: '13/8 13/11',
      stepsRel: [
        [13, 8],
        [13, 11],
      ],
      equityAdjustment: 0.09,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (5-2): פיתוח שני בנאים לנקודות 8 ו-11 בשליטה מלאה במרכז הלוח',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '24/22 13/8',
      stepsRel: [
        [24, 22],
        [13, 8],
      ],
      equityAdjustment: 0.07,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה: פיצול כלי אחורי לנקודה 22 להכנת עוגן הגנתי ותגבור נקודת ה-8',
      ],
    },
  ],
  '5-1': [
    {
      rank: 1,
      canonicalNotation: '24/23 13/8',
      stepsRel: [
        [24, 23],
        [13, 8],
      ],
      equityAdjustment: 0.09,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (5-1): פיצול כלים אחוריים לנקודה 23 ותגבור נקודת ה-8 מול פיתוח היריב',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '13/8 6/5',
      stepsRel: [
        [13, 8],
        [6, 5],
      ],
      equityAdjustment: 0.06,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה: תגבור נקודת ה-8 והצבת כלי חלוץ (Slot) או הכאה על נקודת ה-5 בבית',
      ],
    },
  ],
  '4-3': [
    {
      rank: 1,
      canonicalNotation: '24/20 13/10',
      stepsRel: [
        [24, 20],
        [13, 10],
      ],
      equityAdjustment: 0.09,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (4-3): פיצול כלי אחורי לנקודת ה-5 של היריב (20) והורדת בנאי לנקודה 10',
        'תופס עוגן מפתח בבית היריב (או מכה כלי חלוץ שלו) ומפתח בנאי גמיש',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '13/9 13/10',
      stepsRel: [
        [13, 9],
        [13, 10],
      ],
      equityAdjustment: 0.08,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה: הורדת שני בנאים לנקודות 9 ו-10 לבנייה התקפית מהירה',
      ],
    },
  ],
  '4-1': [
    {
      rank: 1,
      canonicalNotation: '24/20 13/12',
      stepsRel: [
        [24, 20],
        [13, 12],
      ],
      equityAdjustment: 0.09,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה: כניסה/הכאה בנקודת ה-5 של היריב (20) והורדת בנאי כאשר נקודה 20 פנויה או חשופה',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '24/23 13/9',
      stepsRel: [
        [24, 23],
        [13, 9],
      ],
      equityAdjustment: 0.08,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (4-1): פיצול הכלי האחורי לנקודה 23 והורדת בנאי אקטיבי לנקודה 9',
      ],
    },
  ],
  '3-2': [
    {
      rank: 1,
      canonicalNotation: '24/21 13/11',
      stepsRel: [
        [24, 21],
        [13, 11],
      ],
      equityAdjustment: 0.09,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (3-2): פיצול כלים אחוריים לנקודה 21 להשגת עוגן הגנתי והורדת בנאי לנקודה 11',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '13/10 13/11',
      stepsRel: [
        [13, 10],
        [13, 11],
      ],
      equityAdjustment: 0.07,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה: הורדת שני בנאים לנקודות 10 ו-11 לשליטה בלוח החיצוני',
      ],
    },
  ],
  '2-1': [
    {
      rank: 1,
      canonicalNotation: '24/23 13/11',
      stepsRel: [
        [24, 23],
        [13, 11],
      ],
      equityAdjustment: 0.08,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (2-1): פיצול כלי אחורי לנקודה 23 והורדת בנאי לנקודה 11 מול לחץ היריב',
      ],
    },
    {
      rank: 2,
      canonicalNotation: '13/11 6/5',
      stepsRel: [
        [13, 11],
        [6, 5],
      ],
      equityAdjustment: 0.08,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה: הורדת בנאי לנקודה 11 והצבת כלי חלוץ (Slot) או הכאה על נקודת ה-5 הזהובה',
      ],
    },
  ],
  '1-1': [
    {
      rank: 1,
      canonicalNotation: '8/7(2) 6/5(2)',
      stepsRel: [
        [8, 7],
        [8, 7],
        [6, 5],
        [6, 5],
      ],
      equityAdjustment: 0.18,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (דאבל 1-1 חלומי): סגירה כפולה של נקודת ה-5 הזהובה ונקודת הבר (7) ליצירת חומת 4 נקודות (8-7-6-5)!',
      ],
    },
  ],
  '2-2': [
    {
      rank: 1,
      canonicalNotation: '13/11(2) 6/4(2)',
      stepsRel: [
        [13, 11],
        [13, 11],
        [6, 4],
        [6, 4],
      ],
      equityAdjustment: 0.15,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (דאבל 2-2): סגירת נקודת ה-4 בבית והורדת זוג בנאים חזקים לנקודה 11',
      ],
    },
  ],
  '3-3': [
    {
      rank: 1,
      canonicalNotation: '8/5(2) 6/3(2)',
      stepsRel: [
        [8, 5],
        [8, 5],
        [6, 3],
        [6, 3],
      ],
      equityAdjustment: 0.16,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (דאבל 3-3): סגירה כפולה של נקודות ה-5 וה-3 בבית לבניית מבצר פנימי עוצמתי',
      ],
    },
  ],
  '4-4': [
    {
      rank: 1,
      canonicalNotation: '24/20(2) 13/9(2)',
      stepsRel: [
        [24, 20],
        [24, 20],
        [13, 9],
        [13, 9],
      ],
      equityAdjustment: 0.16,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (דאבל 4-4): תפיסת עוגן הזהב בנקודת ה-5 של היריב (20) ובניית נקודה 9 בלוח החיצוני',
      ],
    },
  ],
  '5-5': [
    {
      rank: 1,
      canonicalNotation: '13/8(2) 8/3(2)',
      stepsRel: [
        [13, 8],
        [13, 8],
        [8, 3],
        [8, 3],
      ],
      equityAdjustment: 0.15,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (דאבל 5-5): העברת זוג כלים מנקודה 13 לסגירת נקודת ה-3 בבית ותגבור מרכז הלוח',
      ],
    },
  ],
  '6-6': [
    {
      rank: 1,
      canonicalNotation: '24/18(2) 13/7(2)',
      stepsRel: [
        [24, 18],
        [24, 18],
        [13, 7],
        [13, 7],
      ],
      equityAdjustment: 0.18,
      bookPhase: 'reply',
      explanation: [
        'תיאוריית תגובה (דאבל 6-6): תפיסת שתי נקודות הבר על הלוח (נקודה 18 ונקודה 7) במקביל!',
      ],
    },
  ],
};

function getRollKey(dice: [number, number]): string {
  const high = Math.max(dice[0], dice[1]);
  const low = Math.min(dice[0], dice[1]);
  return `${high}-${low}`;
}

/**
 * Simulates applying `stepsRel` (1-based relative coordinates `fromPt -> toPt`)
 * onto `beforeState` from `player`'s perspective, returning the expected
 * player-relative 24-point array and expected opponent bar count, or `null` if blocked/illegal.
 */
function simulateRelativeStepsOnState(
  beforeState: BoardState,
  player: Player,
  stepsRel: Array<[number, number]>
): { relPoints: number[]; oppBar: number } | null {
  const opponent: Player = player === 'white' ? 'black' : 'white';
  const relPoints = extractPlayerRelativePoints(beforeState, player);
  let oppBar = beforeState.bar[opponent];

  for (const [fromPt, toPt] of stepsRel) {
    const fromRel = fromPt - 1;
    const toRel = toPt - 1;
    if (fromRel < 0 || fromRel >= 24 || toRel < 0 || toRel >= 24) {
      return null;
    }
    if (relPoints[fromRel] <= 0) {
      return null;
    }
    if (relPoints[toRel] <= -2) {
      // Blocked by opponent made point
      return null;
    }

    relPoints[fromRel] -= 1;
    if (relPoints[toRel] === -1) {
      // Hit opponent blot
      relPoints[toRel] = 1;
      oppBar += 1;
    } else {
      relPoints[toRel] += 1;
    }
  }

  return { relPoints, oppBar };
}

/**
 * Returns all opening book entries for a given dice roll `[d1, d2]`.
 * When `isReply` is true, returns the 2nd-turn reply book entries for that roll.
 */
export function getOpeningBookRollEntries(
  dice: [number, number],
  isReply = false
): OpeningBookMoveEntry[] {
  const key = getRollKey(dice);
  if (isReply) {
    return (
      SECOND_TURN_REPLY_BOOK_BY_ROLL[key] ?? OPENING_BOOK_BY_ROLL[key] ?? []
    );
  }
  return (
    OPENING_BOOK_BY_ROLL[key] ?? SECOND_TURN_REPLY_BOOK_BY_ROLL[key] ?? []
  );
}

/**
 * Looks up whether `candidateState` reached from `beforeState` with `dice`
 * matches a theoretical 1st-Roll Opening Book move or 2nd-Turn Reply Book move.
 * Matches canonically from `beforeState.currentTurn`'s perspective so both White and Black
 * benefit from the exact same opening & reply theory.
 */
export function lookupOpeningBookMove(
  beforeState: BoardState,
  dice: [number, number],
  candidateState: BoardState
): OpeningBookMoveEntry | null {
  const isTurn1 = isInitialBoardState(beforeState);
  const isTurn2 = !isTurn1 && isSecondTurnReplyState(beforeState);

  if (!isTurn1 && !isTurn2) {
    return null;
  }

  const entries = getOpeningBookRollEntries(dice, isTurn2);
  if (entries.length === 0) {
    return null;
  }

  const player = beforeState.currentTurn;
  const opponent: Player = player === 'white' ? 'black' : 'white';
  const candRelPoints = extractPlayerRelativePoints(candidateState, player);
  const candOppBar = candidateState.bar[opponent];

  for (const entry of entries) {
    const sim = simulateRelativeStepsOnState(
      beforeState,
      player,
      entry.stepsRel
    );
    if (!sim) {
      continue;
    }
    if (sim.oppBar !== candOppBar) {
      continue;
    }
    let matches = true;
    for (let i = 0; i < 24; i++) {
      if (candRelPoints[i] !== sim.relPoints[i]) {
        matches = false;
        break;
      }
    }
    if (matches) {
      return entry;
    }
  }

  return null;
}
