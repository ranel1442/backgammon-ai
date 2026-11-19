"""
Exact One-Sided Bear-Off Dynamic Programming Tablebase (54,264 states).
Mirrors src/engine/bearoffTablebase.ts for exact endgame supervision & TD targets.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

BEAROFF_NUM_STATES = 54264  # C(15 + 6, 6)
BEAROFF_MAX_TURNS = 25

# Precompute binomial coefficients C(n, k) for 0 <= n <= 25, 0 <= k <= 6
COMB = np.zeros((26, 7), dtype=np.int32)
for n in range(26):
    COMB[n, 0] = 1
    for k in range(1, min(n, 6) + 1):
        COMB[n, k] = COMB[n - 1, k - 1] + COMB[n - 1, k]

DICE_21: List[Tuple[int, int, float]] = []
for d1 in range(1, 7):
    for d2 in range(d1, 7):
        DICE_21.append((d1, d2, (1.0 / 36.0) if d1 == d2 else (2.0 / 36.0)))


SUM_OFFSET = np.zeros(17, dtype=np.int32)
_acc = 0
for _s in range(16):
    SUM_OFFSET[_s] = _acc
    _acc += int(COMB[_s + 5, 5])
SUM_OFFSET[16] = _acc


def index_home_board(c: List[int] | np.ndarray) -> int:
    """
    Maps a 6-element home-board count array [c0, c1, c2, c3, c4, c5]
    (where sum(c) <= 15) to its unique topological index in [0, 54263].
    Matches indexHomeBoard() in src/engine/bearoffTablebase.ts.
    """
    c0, c1, c2, c3, c4, c5 = int(c[0]), int(c[1]), int(c[2]), int(c[3]), int(c[4]), int(c[5])
    total = c0 + c1 + c2 + c3 + c4 + c5
    if total <= 0:
        return 0
    if total > 15:
        total = 15

    idx = int(SUM_OFFSET[total])
    rem = total

    if c5 > 0:
        idx += int(COMB[rem + 5, 5] - COMB[rem - c5 + 5, 5])
        rem -= c5
    if c4 > 0:
        idx += int(COMB[rem + 4, 4] - COMB[rem - c4 + 4, 4])
        rem -= c4
    if c3 > 0:
        idx += int(COMB[rem + 3, 3] - COMB[rem - c3 + 3, 3])
        rem -= c3
    if c2 > 0:
        idx += int(COMB[rem + 2, 2] - COMB[rem - c2 + 2, 2])
        rem -= c2
    idx += c1
    return idx


def decode_home_board_index(index: int) -> List[int]:
    """
    Inverse of index_home_board: decodes index in [0, 54263] -> [c0, c1, c2, c3, c4, c5].
    """
    if index <= 0:
        return [0, 0, 0, 0, 0, 0]

    total = 0
    while total < 15 and int(SUM_OFFSET[total + 1]) <= index:
        total += 1

    offset = index - int(SUM_OFFSET[total])
    rem = total
    coords = [0, 0, 0, 0]  # c5, c4, c3, c2
    dims = [4, 3, 2, 1]

    for d_idx in range(4):
        k = dims[d_idx]
        chosen = 0
        for v in range(rem + 1):
            count_for_v = int(COMB[rem - v + k, k])
            if offset >= count_for_v:
                offset -= count_for_v
            else:
                chosen = v
                break
        coords[d_idx] = chosen
        rem -= chosen

    c1 = offset
    c0 = rem - c1
    return [c0, c1, coords[3], coords[2], coords[1], coords[0]]


def _highest_occupied(counts: List[int]) -> int:
    for i in range(5, -1, -1):
        if counts[i] > 0:
            return i
    return -1


def _generate_single_die_steps(counts: List[int], die: int) -> List[List[int]]:
    highest = _highest_occupied(counts)
    if highest < 0:
        return []

    results: List[List[int]] = []
    for p in range(6):
        if counts[p] <= 0:
            continue
        target = p - die
        if target >= 0:
            nxt = list(counts)
            nxt[p] -= 1
            nxt[target] += 1
            results.append(nxt)
        elif target == -1 or (target < -1 and p == highest):
            nxt = list(counts)
            nxt[p] -= 1
            results.append(nxt)
    return results


def _get_reachable_indices(start_counts: List[int], d1: int, d2: int) -> List[int]:
    if d1 == d2:
        layer = [start_counts]
        for _ in range(4):
            nxt_map: Dict[int, List[int]] = {}
            for board in layer:
                if _highest_occupied(board) < 0:
                    nxt_map[0] = [0, 0, 0, 0, 0, 0]
                    continue
                for s in _generate_single_die_steps(board, d1):
                    nxt_map[index_home_board(s)] = s
            if not nxt_map:
                break
            layer = list(nxt_map.values())
        return list({index_home_board(b) for b in layer})

    two_step: set[int] = set()
    one_step: set[int] = set()

    for a, b in ((d1, d2), (d2, d1)):
        firsts = _generate_single_die_steps(start_counts, a)
        for f in firsts:
            f_idx = index_home_board(f)
            if f_idx == 0:
                two_step.add(0)
                continue
            seconds = _generate_single_die_steps(f, b)
            if not seconds:
                one_step.add(f_idx)
            else:
                for s in seconds:
                    two_step.add(index_home_board(s))

    if two_step:
        return list(two_step)
    if one_step:
        return list(one_step)
    return [index_home_board(start_counts)]


class BearOffDatabase:
    def __init__(self) -> None:
        cache_path = Path(__file__).resolve().parent / "checkpoints" / "bearoff_54264.npz"
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                self.pdf = data["pdf"]
                self.cdf = data["cdf"]
                self.expected_turns = data["expected_turns"]
                return
            except Exception:
                pass

        self.pdf = np.zeros((BEAROFF_NUM_STATES, BEAROFF_MAX_TURNS + 1), dtype=np.float32)
        self.cdf = np.zeros((BEAROFF_NUM_STATES, BEAROFF_MAX_TURNS + 1), dtype=np.float32)
        self.expected_turns = np.zeros(BEAROFF_NUM_STATES, dtype=np.float32)
        self._build()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                pdf=self.pdf,
                cdf=self.cdf,
                expected_turns=self.expected_turns,
            )
        except Exception:
            pass

    def _build(self) -> None:
        self.pdf[0, 0] = 1.0
        self.cdf[0, :] = 1.0
        self.expected_turns[0] = 0.0

        for state_idx in range(1, BEAROFF_NUM_STATES):
            counts = decode_home_board_index(state_idx)
            accum_pdf = np.zeros(BEAROFF_MAX_TURNS + 1, dtype=np.float32)

            for d1, d2, weight in DICE_21:
                candidates = _get_reachable_indices(counts, d1, d2)
                best_idx = min(candidates, key=lambda idx: float(self.expected_turns[idx]))
                accum_pdf[1 : BEAROFF_MAX_TURNS] += weight * self.pdf[best_idx, 0 : BEAROFF_MAX_TURNS - 1]
                accum_pdf[BEAROFF_MAX_TURNS] += weight * (
                    self.pdf[best_idx, BEAROFF_MAX_TURNS - 1] + self.pdf[best_idx, BEAROFF_MAX_TURNS]
                )

            self.pdf[state_idx] = accum_pdf
            self.cdf[state_idx] = np.cumsum(accum_pdf)
            turns = np.arange(BEAROFF_MAX_TURNS + 1, dtype=np.float32)
            self.expected_turns[state_idx] = float(np.dot(turns, accum_pdf))


_CACHED_DB: Optional[BearOffDatabase] = None


def get_bearoff_db() -> BearOffDatabase:
    global _CACHED_DB
    if _CACHED_DB is None:
        _CACHED_DB = BearOffDatabase()
    return _CACHED_DB


def evaluate_exact_bearoff(
    white_home: List[int],
    black_home: List[int],
    turn: Optional[str] = None,
) -> Dict[str, float]:
    db = get_bearoff_db()
    w_idx = index_home_board(white_home)
    b_idx = index_home_board(black_home)

    w_total = sum(white_home)
    b_total = sum(black_home)

    p_white_win = 0.0
    if turn is None or (w_idx == b_idx and w_total > 1):
        for k in range(BEAROFF_MAX_TURNS + 1):
            pw = float(db.pdf[w_idx, k])
            if pw == 0.0:
                continue
            p_less = float(db.cdf[b_idx, k - 1]) if k > 0 else 0.0
            p_black_later = max(0.0, 1.0 - float(db.cdf[b_idx, k]))
            p_black_same = float(db.pdf[b_idx, k])
            p_white_win += pw * (p_black_later + 0.5 * p_black_same)
    elif turn == "white":
        for k in range(BEAROFF_MAX_TURNS + 1):
            pw = float(db.pdf[w_idx, k])
            if pw == 0.0:
                continue
            p_black_not_done = 1.0 if k == 0 else max(0.0, 1.0 - float(db.cdf[b_idx, k - 1]))
            p_white_win += pw * p_black_not_done
    else:
        for k in range(1, BEAROFF_MAX_TURNS + 1):
            pw = float(db.pdf[w_idx, k])
            if pw == 0.0:
                continue
            p_black_not_done = max(0.0, 1.0 - float(db.cdf[b_idx, k]))
            p_white_win += pw * p_black_not_done

    p_white_win = max(0.0, min(1.0, p_white_win))
    p_black_win = 1.0 - p_white_win

    p_white_gammon = float(db.cdf[w_idx, 1]) * p_white_win * 0.25 if b_total == 15 else 0.0
    p_black_gammon = float(db.cdf[b_idx, 1]) * p_black_win * 0.25 if w_total == 15 else 0.0

    equity = (p_white_win - p_black_win) + (p_white_gammon - p_black_gammon)
    return {
        "equity": max(-3.0, min(3.0, equity)),
        "win_prob": p_white_win * 100.0,
        "gammon_prob": (p_white_gammon + p_black_gammon) * 100.0,
        "white_expected_turns": float(db.expected_turns[w_idx]),
        "black_expected_turns": float(db.expected_turns[b_idx]),
    }
