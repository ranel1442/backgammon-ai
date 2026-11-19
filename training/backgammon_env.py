"""
Numba-JIT Multi-Core Accelerated Backgammon Environment, Legal Move Generator,
3-Phase Classifier ('contact' | 'crashed' | 'race'), and 210-Feature Canonical Extractor.
Matches src/engine/rules.ts and src/engine/neuralNet.ts index-for-index.

Includes @njit(parallel=True, fastmath=True) batch kernels using numba.prange:
- batch_extract_features_210(boards, bars, borne_offs, turns)
- batch_classify_phases(boards, bars, borne_offs)
- batch_generate_legal_afterstates(boards, bars, borne_offs, turns, dice)
- batch_apply_selected_moves(...)
- batch_label_positions_teacher(boards, bars, borne_offs, turns)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range  # type: ignore

    def njit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

from bearoff_dp import (
    BEAROFF_MAX_TURNS,
    COMB,
    DICE_21 as ALL_21_DICE,
    SUM_OFFSET,
    get_bearoff_db,
    index_home_board,
)

FEATURE_VECTOR_SIZE = 260
FEATURE_VECTOR_SIZE_V2 = 210
MAX_CANDIDATES = 64

# Initialize cached 54,264-state bear-off tablebase arrays at import time for Numba kernels
_BEAROFF_DB = get_bearoff_db()
_BEAROFF_EXPECTED_TURNS = _BEAROFF_DB.expected_turns
_BEAROFF_PDF = _BEAROFF_DB.pdf
_BEAROFF_CDF = _BEAROFF_DB.cdf
_DICE_21_ARR = np.array(
    [[d1, d2, prob] for d1, d2, prob in ALL_21_DICE], dtype=np.float32
)


@dataclass
class BoardState:
    points: List[int]  # length 24: >0 white, <0 black
    bar_white: int = 0
    bar_black: int = 0
    off_white: int = 0
    off_black: int = 0
    current_turn: str = "white"  # 'white' | 'black'

    def clone(self) -> "BoardState":
        return BoardState(
            points=list(self.points),
            bar_white=self.bar_white,
            bar_black=self.bar_black,
            off_white=self.off_white,
            off_black=self.off_black,
            current_turn=self.current_turn,
        )


def create_initial_board(starting_turn: str = "white") -> BoardState:
    pts = [0] * 24
    # White moves 23 -> 0
    pts[23] = 2
    pts[12] = 5
    pts[7] = 3
    pts[5] = 5
    # Black moves 0 -> 23
    pts[0] = -2
    pts[11] = -5
    pts[16] = -3
    pts[18] = -5
    return BoardState(points=pts, current_turn=starting_turn)


@njit(fastmath=True)
def _index_home_board_jit(
    c0: int, c1: int, c2: int, c3: int, c4: int, c5: int
) -> int:
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


@njit(fastmath=True)
def _get_exact_expected_bearoff_turns_jit(
    points: np.ndarray,
    bar_white: int,
    bar_black: int,
    off_white: int,
    off_black: int,
    is_white: bool,
) -> float:
    off = off_white if is_white else off_black
    if off >= 15:
        return 0.0

    if is_white:
        h0 = points[0] if points[0] > 0 else 0
        h1 = points[1] if points[1] > 0 else 0
        h2 = points[2] if points[2] > 0 else 0
        h3 = points[3] if points[3] > 0 else 0
        h4 = points[4] if points[4] > 0 else 0
        h5 = points[5] if points[5] > 0 else 0
        bar = bar_white
    else:
        h0 = -points[23] if points[23] < 0 else 0
        h1 = -points[22] if points[22] < 0 else 0
        h2 = -points[21] if points[21] < 0 else 0
        h3 = -points[20] if points[20] < 0 else 0
        h4 = -points[19] if points[19] < 0 else 0
        h5 = -points[18] if points[18] < 0 else 0
        bar = bar_black

    outside_checkers = bar
    outside_pips = bar * 19

    if is_white:
        for i in range(6, 24):
            v = points[i]
            if v > 0:
                outside_checkers += v
                outside_pips += v * (i - 5)
    else:
        for i in range(0, 18):
            v = points[i]
            if v < 0:
                c = -v
                outside_checkers += c
                outside_pips += c * (18 - i)

    if outside_checkers == 0:
        idx = _index_home_board_jit(h0, h1, h2, h3, h4, h5)
        return float(_BEAROFF_EXPECTED_TURNS[idx])

    home_sum = h0 + h1 + h2 + h3 + h4 + h5
    rem = outside_checkers
    max_add = 15 - home_sum
    if max_add < 0:
        max_add = 0
    if rem > max_add:
        rem = max_add

    pt = 5
    while rem > 0:
        if pt == 5:
            h5 += 1
        elif pt == 4:
            h4 += 1
        elif pt == 3:
            h3 += 1
        rem -= 1
        pt = 5 if pt == 3 else pt - 1

    idx = _index_home_board_jit(h0, h1, h2, h3, h4, h5)
    return float(_BEAROFF_EXPECTED_TURNS[idx]) + (float(outside_pips) / 8.167)


@njit(fastmath=True)
def _classify_phase_jit(
    points: np.ndarray,
    bar_white: int,
    bar_black: int,
    off_white: int,
    off_black: int,
) -> int:
    """
    Returns:
      0 -> 'race'
      1 -> 'crashed'
      2 -> 'contact'
    Matches classifyGamePhase() in src/engine/neuralNet.ts.
    """
    if bar_white == 0 and bar_black == 0:
        highest_white = -1
        lowest_black = 24
        for i in range(24):
            v = points[i]
            if v > 0 and i > highest_white:
                highest_white = i
            if v < 0 and i < lowest_black:
                lowest_black = i
        if highest_white < lowest_black:
            return 0  # 'race'

    if (15 - off_white) <= 5 or (15 - off_black) <= 5:
        return 1  # 'crashed'

    return 2  # 'contact'


def classify_game_phase(state: BoardState) -> str:
    """
    Deterministic 3-way game-phase classification ('race' | 'crashed' | 'contact')
    shared identically between TypeScript and Python.
    """
    pts = np.asarray(state.points, dtype=np.int32)
    code = _classify_phase_jit(
        pts, state.bar_white, state.bar_black, state.off_white, state.off_black
    )
    if code == 0:
        return "race"
    if code == 1:
        return "crashed"
    return "contact"


def is_pure_race_state(state: BoardState) -> bool:
    return classify_game_phase(state) == "race"


def is_pure_bearoff_state(state: BoardState) -> bool:
    if state.bar_white > 0 or state.bar_black > 0:
        return False
    for i, val in enumerate(state.points):
        if val > 0 and i > 5:
            return False
        if val < 0 and i < 18:
            return False
    return True


def extract_home_board(state: BoardState, player: str) -> List[int]:
    if player == "white":
        return [max(0, state.points[i]) for i in range(6)]
    return [max(0, -state.points[23 - i]) for i in range(6)]


def check_winner(state: BoardState) -> Optional[Tuple[str, int]]:
    """
    Returns (winner, multiplier) where multiplier is 1 (Single), 2 (Gammon), or 3 (Backgammon),
    or None if the game is ongoing.
    """
    if state.off_white >= 15:
        if state.off_black > 0:
            return ("white", 1)
        in_home_or_bar = state.bar_black > 0 or any(state.points[i] < 0 for i in range(6))
        return ("white", 3 if in_home_or_bar else 2)

    if state.off_black >= 15:
        if state.off_white > 0:
            return ("black", 1)
        in_home_or_bar = state.bar_white > 0 or any(state.points[i] > 0 for i in range(18, 24))
        return ("black", 3 if in_home_or_bar else 2)

    return None


@njit(fastmath=True)
def _can_bear_off_jit(points: np.ndarray, bar_self: int, is_white: bool) -> bool:
    if bar_self > 0:
        return False
    if is_white:
        for i in range(6, 24):
            if points[i] > 0:
                return False
        return True
    else:
        for i in range(0, 18):
            if points[i] < 0:
                return False
        return True


def _can_bear_off(state: BoardState, player: str) -> bool:
    pts = np.asarray(state.points, dtype=np.int32)
    is_white = player == "white"
    return bool(
        _can_bear_off_jit(pts, state.bar_white if is_white else state.bar_black, is_white)
    )


def _generate_single_step_states(
    state: BoardState, player: str, die: int
) -> List[Tuple[BoardState, int]]:
    results: List[Tuple[BoardState, int]] = []
    if player == "white":
        if state.bar_white > 0:
            target = 24 - die
            if state.points[target] >= -1:
                nxt = state.clone()
                nxt.bar_white -= 1
                if nxt.points[target] == -1:
                    nxt.points[target] = 1
                    nxt.bar_black += 1
                else:
                    nxt.points[target] += 1
                results.append((nxt, die))
            return results

        can_off = _can_bear_off(state, "white")
        highest = -1
        if can_off:
            for i in range(5, -1, -1):
                if state.points[i] > 0:
                    highest = i
                    break

        for idx in range(23, -1, -1):
            if state.points[idx] <= 0:
                continue
            target = idx - die
            if target >= 0:
                if state.points[target] >= -1:
                    nxt = state.clone()
                    nxt.points[idx] -= 1
                    if nxt.points[target] == -1:
                        nxt.points[target] = 1
                        nxt.bar_black += 1
                    else:
                        nxt.points[target] += 1
                    results.append((nxt, die))
            elif can_off:
                if target == -1 or (target < -1 and idx == highest):
                    nxt = state.clone()
                    nxt.points[idx] -= 1
                    nxt.off_white += 1
                    results.append((nxt, die))
    else:
        if state.bar_black > 0:
            target = die - 1
            if state.points[target] <= 1:
                nxt = state.clone()
                nxt.bar_black -= 1
                if nxt.points[target] == 1:
                    nxt.points[target] = -1
                    nxt.bar_white += 1
                else:
                    nxt.points[target] -= 1
                results.append((nxt, die))
            return results

        can_off = _can_bear_off(state, "black")
        lowest = 24
        if can_off:
            for i in range(18, 24):
                if state.points[i] < 0:
                    lowest = i
                    break

        for idx in range(24):
            if state.points[idx] >= 0:
                continue
            target = idx + die
            if target <= 23:
                if state.points[target] <= 1:
                    nxt = state.clone()
                    nxt.points[idx] += 1
                    if nxt.points[target] == 1:
                        nxt.points[target] = -1
                        nxt.bar_white += 1
                    else:
                        nxt.points[target] -= 1
                    results.append((nxt, die))
            elif can_off:
                if target == 24 or (target > 24 and idx == lowest):
                    nxt = state.clone()
                    nxt.points[idx] += 1
                    nxt.off_black += 1
                    results.append((nxt, die))

    return results


def _board_key(state: BoardState) -> Tuple[int, ...]:
    return (
        *state.points,
        state.bar_white,
        state.bar_black,
        state.off_white,
        state.off_black,
    )


def generate_legal_afterstates(
    state: BoardState, dice: Tuple[int, int]
) -> List[BoardState]:
    """
    Returns deduplicated legal resulting BoardStates (with current_turn switched to opponent),
    enforcing maximum dice usage and higher-die rule.
    """
    player = state.current_turn
    next_player = "black" if player == "white" else "white"
    d1, d2 = dice

    if d1 == d2:
        layer = [state]
        for _ in range(4):
            nxt_dict = {}
            for s in layer:
                if s.off_white >= 15 or s.off_black >= 15:
                    continue
                for ns, _ in _generate_single_step_states(s, player, d1):
                    nxt_dict[_board_key(ns)] = ns
            if not nxt_dict:
                break
            layer = list(nxt_dict.values())

        out = []
        for s in layer:
            c = s.clone()
            c.current_turn = next_player
            out.append(c)
        return out

    two_die_map = {}
    one_die_map = {}

    for first_d, second_d in ((d1, d2), (d2, d1)):
        first_steps = _generate_single_step_states(state, player, first_d)
        for s1, used1 in first_steps:
            if s1.off_white >= 15 or s1.off_black >= 15:
                two_die_map[_board_key(s1)] = s1
                continue
            second_steps = _generate_single_step_states(s1, player, second_d)
            if not second_steps:
                k = _board_key(s1)
                if k not in one_die_map or used1 > one_die_map[k][1]:
                    one_die_map[k] = (s1, used1)
            else:
                for s2, _ in second_steps:
                    two_die_map[_board_key(s2)] = s2

    if two_die_map:
        chosen = list(two_die_map.values())
    elif one_die_map:
        max_die = max(u for _, u in one_die_map.values())
        chosen = [s for s, u in one_die_map.values() if u == max_die]
    else:
        c = state.clone()
        c.current_turn = next_player
        return [c]

    for s in chosen:
        s.current_turn = next_player
    return chosen


def get_exact_expected_bearoff_turns_py(state: BoardState, player: str) -> float:
    pts = np.asarray(state.points, dtype=np.int32)
    return float(
        _get_exact_expected_bearoff_turns_jit(
            pts,
            state.bar_white,
            state.bar_black,
            state.off_white,
            state.off_black,
            player == "white",
        )
    )


@njit(fastmath=True)
def _extract_features_260_into_jit(
    points: np.ndarray,
    bar_white: int,
    bar_black: int,
    off_white: int,
    off_black: int,
    is_white: bool,
    out_row: np.ndarray,
) -> None:
    """
    Writes the Ultimate 260-Feature Grandmaster canonical representation (`version: 3`)
    directly into `out_row` (length 260).
    100% Numba-compiled and matches `extractFeatures260` in `src/engine/neuralNet.ts` index-for-index.
    """
    for i in range(FEATURE_VECTOR_SIZE):
        out_row[i] = 0.0

    exact_turns_self = _get_exact_expected_bearoff_turns_jit(
        points, bar_white, bar_black, off_white, off_black, is_white
    )
    exact_turns_opp = _get_exact_expected_bearoff_turns_jit(
        points, bar_white, bar_black, off_white, off_black, not is_white
    )

    # 1. [0..191]: 192 Point features + [210..257]: 48 Stripped & Spare Builder features
    for step in range(24):
        rel = 23 - step
        abs_idx = rel if is_white else (23 - rel)
        raw = points[abs_idx]
        count = raw if is_white else -raw
        base = step * 8

        if count > 0:
            out_row[base] = 1.0
            if count >= 2:
                out_row[base + 1] = 1.0
            if count >= 3:
                out_row[base + 2] = 1.0
            if count > 3:
                out_row[base + 3] = (count - 3) * 0.5

            if count == 2:
                out_row[210 + step] = 1.0
            elif count == 3 or count == 4:
                out_row[234 + step] = 1.0
        elif count < 0:
            opp_count = -count
            out_row[base + 4] = 1.0
            if opp_count >= 2:
                out_row[base + 5] = 1.0
            if opp_count >= 3:
                out_row[base + 6] = 1.0
            if opp_count > 3:
                out_row[base + 7] = (opp_count - 3) * 0.5

            if opp_count == 2:
                out_row[210 + step] = -1.0
            elif opp_count == 3 or opp_count == 4:
                out_row[234 + step] = -1.0

    bar_self = bar_white if is_white else bar_black
    bar_opp = bar_black if is_white else bar_white
    off_self = off_white if is_white else off_black
    off_opp = off_black if is_white else off_white

    pip_self = bar_self * 25
    pip_opp = bar_opp * 25

    highest_self_rel = 24 if bar_self > 0 else -1
    lowest_opp_rel = -1 if bar_opp > 0 else 24

    cur_prime_self = 0
    max_prime_self = 0
    cur_prime_opp = 0
    max_prime_opp = 0
    home_points_self = 0
    home_points_opp = 0
    outfield_and_back_pip_self = bar_self * 25
    anchors_in_opp_home_self = 0
    self_back_checkers = bar_self
    opp_back_checkers = bar_opp

    for rel in range(24):
        abs_idx = rel if is_white else (23 - rel)
        raw = points[abs_idx]
        c = raw if is_white else -raw

        if c > 0:
            pip_self += c * (rel + 1)
            if rel > highest_self_rel:
                highest_self_rel = rel
            if rel >= 6:
                outfield_and_back_pip_self += c * (rel + 1)
            if rel >= 18:
                self_back_checkers += c
                if c >= 2:
                    anchors_in_opp_home_self += 1
        elif c < 0:
            pip_opp += (-c) * (24 - rel)
            if rel < lowest_opp_rel:
                lowest_opp_rel = rel
            if rel <= 5:
                opp_back_checkers += -c

        if c >= 2:
            cur_prime_self += 1
            if cur_prime_self > max_prime_self:
                max_prime_self = cur_prime_self
            if rel <= 5:
                home_points_self += 1
        else:
            cur_prime_self = 0

        if c <= -2:
            cur_prime_opp += 1
            if cur_prime_opp > max_prime_opp:
                max_prime_opp = cur_prime_opp
            if rel >= 18:
                home_points_opp += 1
        else:
            cur_prime_opp = 0

    is_contact = highest_self_rel > lowest_opp_rel or bar_self > 0 or bar_opp > 0
    blot_self = 0
    blot_opp = 0

    if is_contact:
        for rel in range(24):
            abs_idx = rel if is_white else (23 - rel)
            c = points[abs_idx] if is_white else -points[abs_idx]
            if c == 1 and rel > lowest_opp_rel:
                blot_self += 1
            elif c == -1 and rel < highest_self_rel:
                blot_opp += 1

    closed_ratio_self = home_points_self / 6.0
    closed_ratio_opp = home_points_opp / 6.0
    blot_ratio_self = min(6, blot_self) / 6.0
    blot_ratio_opp = min(6, blot_opp) / 6.0

    abs_19 = 19 if is_white else (23 - 19)
    c_19 = points[abs_19] if is_white else -points[abs_19]
    golden_20_owned = 1.0 if c_19 >= 2 else 0.0
    double_anchor_ind = 1.0 if anchors_in_opp_home_self >= 2 else 0.0
    denom_pip_opp = float(pip_opp) if pip_opp > 1 else 1.0
    timing_ratio = min(2.0, float(outfield_and_back_pip_self) / denom_pip_opp)

    out_row[192] = bar_self / 2.0
    out_row[193] = bar_opp / 2.0
    out_row[194] = off_self / 15.0
    out_row[195] = off_opp / 15.0
    out_row[196] = pip_self / 167.0
    out_row[197] = pip_opp / 167.0
    out_row[198] = (pip_opp - pip_self) / 50.0
    out_row[199] = exact_turns_self / 20.0
    out_row[200] = exact_turns_opp / 20.0
    out_row[201] = min(6, max_prime_self) / 6.0
    out_row[202] = min(6, max_prime_opp) / 6.0
    out_row[203] = closed_ratio_self * closed_ratio_self
    out_row[204] = closed_ratio_opp * closed_ratio_opp
    out_row[205] = blot_ratio_self * closed_ratio_opp
    out_row[206] = blot_ratio_opp * closed_ratio_self
    out_row[207] = 1.0 if is_contact else 0.0
    out_row[208] = timing_ratio / 2.0
    out_row[209] = 0.5 * double_anchor_ind + 0.5 * golden_20_owned

    # [258..259]: Prime Trapping & Lethality products
    prime_trap_self = min(6, max_prime_self) * min(3, opp_back_checkers)
    prime_trap_opp = min(6, max_prime_opp) * min(3, self_back_checkers)
    out_row[258] = prime_trap_self / 18.0
    out_row[259] = prime_trap_opp / 18.0


_extract_features_210_into_jit = _extract_features_260_into_jit


@njit(fastmath=True)
def _extract_features_260_single_jit(
    points: np.ndarray,
    bar_white: int,
    bar_black: int,
    off_white: int,
    off_black: int,
    is_white: bool,
) -> np.ndarray:
    out = np.empty(FEATURE_VECTOR_SIZE, dtype=np.float32)
    _extract_features_260_into_jit(
        points, bar_white, bar_black, off_white, off_black, is_white, out
    )
    return out


def extract_features_260(state: BoardState, perspective: str) -> np.ndarray:
    """
    Canonical 260-dimensional feature vector from `perspective`'s point of view.
    Accelerated via Numba @njit and matches extractFeatures260() in src/engine/neuralNet.ts.
    """
    pts = np.asarray(state.points, dtype=np.int32)
    return _extract_features_260_single_jit(
        pts,
        state.bar_white,
        state.bar_black,
        state.off_white,
        state.off_black,
        perspective == "white",
    )


extract_features_210 = extract_features_260


# ============================================================================
# MULTI-CORE PARALLEL NUMBA BATCH KERNELS (@njit(parallel=True, fastmath=True))
# ============================================================================


@njit(parallel=True, fastmath=True, cache=True)
def batch_extract_features_260(
    boards: np.ndarray,
    bars: np.ndarray,
    borne_offs: np.ndarray,
    turns: np.ndarray,
) -> np.ndarray:
    """
    Extracts (N, 260) canonical float32 features across all N environments in parallel
    using all CPU cores via numba.prange.
    - boards: (N, 24) int32
    - bars: (N, 2) int32 [bar_white, bar_black]
    - borne_offs: (N, 2) int32 [off_white, off_black]
    - turns: (N,) int32 (1 = White perspective, 0 = Black perspective)
    """
    n = boards.shape[0]
    out = np.empty((n, FEATURE_VECTOR_SIZE), dtype=np.float32)
    for i in prange(n):
        is_white = turns[i] == 1
        _extract_features_260_into_jit(
            boards[i],
            int(bars[i, 0]),
            int(bars[i, 1]),
            int(borne_offs[i, 0]),
            int(borne_offs[i, 1]),
            is_white,
            out[i],
        )
    return out


batch_extract_features_210 = batch_extract_features_260


@njit(parallel=True, fastmath=True, cache=True)
def batch_classify_phases(
    boards: np.ndarray,
    bars: np.ndarray,
    borne_offs: np.ndarray,
) -> np.ndarray:
    """
    Classifies game phase across N boards in parallel:
      0 = 'race', 1 = 'crashed', 2 = 'contact'
    """
    n = boards.shape[0]
    phases = np.empty(n, dtype=np.int32)
    for i in prange(n):
        phases[i] = _classify_phase_jit(
            boards[i],
            int(bars[i, 0]),
            int(bars[i, 1]),
            int(borne_offs[i, 0]),
            int(borne_offs[i, 1]),
        )
    return phases


@njit(fastmath=True)
def _step_one_die_jit(
    pts: np.ndarray,
    bw: int,
    bb: int,
    ow: int,
    ob: int,
    is_white: bool,
    die: int,
    out_pts: np.ndarray,
    out_bw: np.ndarray,
    out_bb: np.ndarray,
    out_ow: np.ndarray,
    out_ob: np.ndarray,
) -> int:
    """
    Generates all legal 1-die successor states for a single board state into pre-allocated buffers.
    Returns number of legal 1-die steps (at most 24).
    """
    count = 0
    if is_white:
        if bw > 0:
            target = 24 - die
            if pts[target] >= -1:
                for k in range(24):
                    out_pts[0, k] = pts[k]
                out_bw[0] = bw - 1
                out_bb[0] = bb
                out_ow[0] = ow
                out_ob[0] = ob
                if out_pts[0, target] == -1:
                    out_pts[0, target] = 1
                    out_bb[0] += 1
                else:
                    out_pts[0, target] += 1
                return 1
            return 0

        can_off = _can_bear_off_jit(pts, bw, True)
        highest = -1
        if can_off:
            for i in range(5, -1, -1):
                if pts[i] > 0:
                    highest = i
                    break

        for idx in range(23, -1, -1):
            if pts[idx] <= 0:
                continue
            target = idx - die
            if target >= 0:
                if pts[target] >= -1:
                    for k in range(24):
                        out_pts[count, k] = pts[k]
                    out_pts[count, idx] -= 1
                    out_bw[count] = bw
                    out_bb[count] = bb
                    out_ow[count] = ow
                    out_ob[count] = ob
                    if out_pts[count, target] == -1:
                        out_pts[count, target] = 1
                        out_bb[count] += 1
                    else:
                        out_pts[count, target] += 1
                    count += 1
            elif can_off:
                if target == -1 or (target < -1 and idx == highest):
                    for k in range(24):
                        out_pts[count, k] = pts[k]
                    out_pts[count, idx] -= 1
                    out_bw[count] = bw
                    out_bb[count] = bb
                    out_ow[count] = ow + 1
                    out_ob[count] = ob
                    count += 1
    else:
        if bb > 0:
            target = die - 1
            if pts[target] <= 1:
                for k in range(24):
                    out_pts[0, k] = pts[k]
                out_bw[0] = bw
                out_bb[0] = bb - 1
                out_ow[0] = ow
                out_ob[0] = ob
                if out_pts[0, target] == 1:
                    out_pts[0, target] = -1
                    out_bw[0] += 1
                else:
                    out_pts[0, target] -= 1
                return 1
            return 0

        can_off = _can_bear_off_jit(pts, bb, False)
        lowest = 24
        if can_off:
            for i in range(18, 24):
                if pts[i] < 0:
                    lowest = i
                    break

        for idx in range(24):
            if pts[idx] >= 0:
                continue
            target = idx + die
            if target <= 23:
                if pts[target] <= 1:
                    for k in range(24):
                        out_pts[count, k] = pts[k]
                    out_pts[count, idx] += 1
                    out_bw[count] = bw
                    out_bb[count] = bb
                    out_ow[count] = ow
                    out_ob[count] = ob
                    if out_pts[count, target] == 1:
                        out_pts[count, target] = -1
                        out_bw[count] += 1
                    else:
                        out_pts[count, target] -= 1
                    count += 1
            elif can_off:
                if target == 24 or (target > 24 and idx == lowest):
                    for k in range(24):
                        out_pts[count, k] = pts[k]
                    out_pts[count, idx] += 1
                    out_bw[count] = bw
                    out_bb[count] = bb
                    out_ow[count] = ow
                    out_ob[count] = ob + 1
                    count += 1

    return count


@njit(fastmath=True)
def _append_unique_state_jit(
    pts: np.ndarray,
    bw: int,
    bb: int,
    ow: int,
    ob: int,
    buf_pts: np.ndarray,
    buf_bars: np.ndarray,
    buf_offs: np.ndarray,
    cur_count: int,
    max_cap: int,
) -> int:
    for i in range(cur_count):
        if (
            buf_bars[i, 0] == bw
            and buf_bars[i, 1] == bb
            and buf_offs[i, 0] == ow
            and buf_offs[i, 1] == ob
        ):
            same = True
            for k in range(24):
                if buf_pts[i, k] != pts[k]:
                    same = False
                    break
            if same:
                return cur_count

    if cur_count >= max_cap:
        return cur_count

    for k in range(24):
        buf_pts[cur_count, k] = pts[k]
    buf_bars[cur_count, 0] = bw
    buf_bars[cur_count, 1] = bb
    buf_offs[cur_count, 0] = ow
    buf_offs[cur_count, 1] = ob
    return cur_count + 1


@njit(fastmath=True)
def _generate_env_afterstates_jit(
    pts: np.ndarray,
    bw: int,
    bb: int,
    ow: int,
    ob: int,
    is_white: bool,
    d1: int,
    d2: int,
    out_pts: np.ndarray,
    out_bars: np.ndarray,
    out_offs: np.ndarray,
) -> int:
    """
    Generates all deduplicated legal turn afterstates (up to MAX_CANDIDATES) for one environment.
    Returns the number of legal candidate afterstates (>= 1).
    """
    if ow >= 15 or ob >= 15:
        for k in range(24):
            out_pts[0, k] = pts[k]
        out_bars[0, 0] = bw
        out_bars[0, 1] = bb
        out_offs[0, 0] = ow
        out_offs[0, 1] = ob
        return 1

    step1_pts = np.empty((24, 24), dtype=np.int32)
    step1_bw = np.empty(24, dtype=np.int32)
    step1_bb = np.empty(24, dtype=np.int32)
    step1_ow = np.empty(24, dtype=np.int32)
    step1_ob = np.empty(24, dtype=np.int32)

    step2_pts = np.empty((24, 24), dtype=np.int32)
    step2_bw = np.empty(24, dtype=np.int32)
    step2_bb = np.empty(24, dtype=np.int32)
    step2_ow = np.empty(24, dtype=np.int32)
    step2_ob = np.empty(24, dtype=np.int32)

    if d1 == d2:
        layer_pts = np.empty((MAX_CANDIDATES, 24), dtype=np.int32)
        layer_bars = np.empty((MAX_CANDIDATES, 2), dtype=np.int32)
        layer_offs = np.empty((MAX_CANDIDATES, 2), dtype=np.int32)

        next_pts = np.empty((MAX_CANDIDATES, 24), dtype=np.int32)
        next_bars = np.empty((MAX_CANDIDATES, 2), dtype=np.int32)
        next_offs = np.empty((MAX_CANDIDATES, 2), dtype=np.int32)

        for k in range(24):
            layer_pts[0, k] = pts[k]
        layer_bars[0, 0] = bw
        layer_bars[0, 1] = bb
        layer_offs[0, 0] = ow
        layer_offs[0, 1] = ob
        layer_count = 1

        for _step in range(4):
            next_count = 0
            for idx in range(layer_count):
                if layer_offs[idx, 0] >= 15 or layer_offs[idx, 1] >= 15:
                    continue
                n1 = _step_one_die_jit(
                    layer_pts[idx],
                    int(layer_bars[idx, 0]),
                    int(layer_bars[idx, 1]),
                    int(layer_offs[idx, 0]),
                    int(layer_offs[idx, 1]),
                    is_white,
                    d1,
                    step1_pts,
                    step1_bw,
                    step1_bb,
                    step1_ow,
                    step1_ob,
                )
                for j in range(n1):
                    next_count = _append_unique_state_jit(
                        step1_pts[j],
                        int(step1_bw[j]),
                        int(step1_bb[j]),
                        int(step1_ow[j]),
                        int(step1_ob[j]),
                        next_pts,
                        next_bars,
                        next_offs,
                        next_count,
                        MAX_CANDIDATES,
                    )
            if next_count == 0:
                break
            for idx in range(next_count):
                for k in range(24):
                    layer_pts[idx, k] = next_pts[idx, k]
                layer_bars[idx, 0] = next_bars[idx, 0]
                layer_bars[idx, 1] = next_bars[idx, 1]
                layer_offs[idx, 0] = next_offs[idx, 0]
                layer_offs[idx, 1] = next_offs[idx, 1]
            layer_count = next_count

        for idx in range(layer_count):
            for k in range(24):
                out_pts[idx, k] = layer_pts[idx, k]
            out_bars[idx, 0] = layer_bars[idx, 0]
            out_bars[idx, 1] = layer_bars[idx, 1]
            out_offs[idx, 0] = layer_offs[idx, 0]
            out_offs[idx, 1] = layer_offs[idx, 1]
        return layer_count

    # Non-double roll (d1 != d2)
    two_count = 0
    one_pts = np.empty((48, 24), dtype=np.int32)
    one_bars = np.empty((48, 2), dtype=np.int32)
    one_offs = np.empty((48, 2), dtype=np.int32)
    one_die_used = np.empty(48, dtype=np.int32)
    one_count = 0

    for order in range(2):
        first_d = d1 if order == 0 else d2
        second_d = d2 if order == 0 else d1

        n1 = _step_one_die_jit(
            pts,
            bw,
            bb,
            ow,
            ob,
            is_white,
            first_d,
            step1_pts,
            step1_bw,
            step1_bb,
            step1_ow,
            step1_ob,
        )
        for i in range(n1):
            if step1_ow[i] >= 15 or step1_ob[i] >= 15:
                two_count = _append_unique_state_jit(
                    step1_pts[i],
                    int(step1_bw[i]),
                    int(step1_bb[i]),
                    int(step1_ow[i]),
                    int(step1_ob[i]),
                    out_pts,
                    out_bars,
                    out_offs,
                    two_count,
                    MAX_CANDIDATES,
                )
                continue

            n2 = _step_one_die_jit(
                step1_pts[i],
                int(step1_bw[i]),
                int(step1_bb[i]),
                int(step1_ow[i]),
                int(step1_ob[i]),
                is_white,
                second_d,
                step2_pts,
                step2_bw,
                step2_bb,
                step2_ow,
                step2_ob,
            )
            if n2 == 0:
                if one_count < 48:
                    for k in range(24):
                        one_pts[one_count, k] = step1_pts[i, k]
                    one_bars[one_count, 0] = step1_bw[i]
                    one_bars[one_count, 1] = step1_bb[i]
                    one_offs[one_count, 0] = step1_ow[i]
                    one_offs[one_count, 1] = step1_ob[i]
                    one_die_used[one_count] = first_d
                    one_count += 1
            else:
                for j in range(n2):
                    two_count = _append_unique_state_jit(
                        step2_pts[j],
                        int(step2_bw[j]),
                        int(step2_bb[j]),
                        int(step2_ow[j]),
                        int(step2_ob[j]),
                        out_pts,
                        out_bars,
                        out_offs,
                        two_count,
                        MAX_CANDIDATES,
                    )

    if two_count > 0:
        return two_count

    if one_count > 0:
        max_d = 0
        for i in range(one_count):
            if one_die_used[i] > max_d:
                max_d = int(one_die_used[i])
        final_count = 0
        for i in range(one_count):
            if one_die_used[i] == max_d:
                final_count = _append_unique_state_jit(
                    one_pts[i],
                    int(one_bars[i, 0]),
                    int(one_bars[i, 1]),
                    int(one_offs[i, 0]),
                    int(one_offs[i, 1]),
                    out_pts,
                    out_bars,
                    out_offs,
                    final_count,
                    MAX_CANDIDATES,
                )
        return final_count

    # Pass turn (unchanged state)
    for k in range(24):
        out_pts[0, k] = pts[k]
    out_bars[0, 0] = bw
    out_bars[0, 1] = bb
    out_offs[0, 0] = ow
    out_offs[0, 1] = ob
    return 1


@njit(parallel=True, fastmath=True, cache=True)
def batch_generate_legal_afterstates(
    boards: np.ndarray,
    bars: np.ndarray,
    borne_offs: np.ndarray,
    turns: np.ndarray,
    dice: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates legal afterstates across all N environments in parallel using numba.prange.
    Returns:
      cand_boards: (N, MAX_CANDIDATES, 24) int32
      cand_bars:   (N, MAX_CANDIDATES, 2) int32
      cand_offs:   (N, MAX_CANDIDATES, 2) int32
      cand_counts: (N,) int32
    """
    n = boards.shape[0]
    cand_boards = np.empty((n, MAX_CANDIDATES, 24), dtype=np.int32)
    cand_bars = np.empty((n, MAX_CANDIDATES, 2), dtype=np.int32)
    cand_offs = np.empty((n, MAX_CANDIDATES, 2), dtype=np.int32)
    cand_counts = np.empty(n, dtype=np.int32)

    for i in prange(n):
        cand_counts[i] = _generate_env_afterstates_jit(
            boards[i],
            int(bars[i, 0]),
            int(bars[i, 1]),
            int(borne_offs[i, 0]),
            int(borne_offs[i, 1]),
            turns[i] == 1,
            int(dice[i, 0]),
            int(dice[i, 1]),
            cand_boards[i],
            cand_bars[i],
            cand_offs[i],
        )
    return cand_boards, cand_bars, cand_offs, cand_counts


@njit(parallel=True, fastmath=True, cache=True)
def batch_pack_and_extract_candidates(
    cand_boards: np.ndarray,
    cand_bars: np.ndarray,
    cand_offs: np.ndarray,
    cand_counts: np.ndarray,
    offsets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Packs all candidate afterstates across N environments into a single contiguous
    (TotalCandidates, 210) float32 feature matrix and (TotalCandidates,) int32 phase vector
    in parallel across all CPU cores via numba.prange.
    """
    n = cand_counts.shape[0]
    total_cands = int(offsets[n])
    flat_feats = np.empty((total_cands, FEATURE_VECTOR_SIZE), dtype=np.float32)
    flat_phases = np.empty(total_cands, dtype=np.int32)

    for i in prange(n):
        base = int(offsets[i])
        cnt = int(cand_counts[i])
        for c in range(cnt):
            idx = base + c
            _extract_features_210_into_jit(
                cand_boards[i, c],
                int(cand_bars[i, c, 0]),
                int(cand_bars[i, c, 1]),
                int(cand_offs[i, c, 0]),
                int(cand_offs[i, c, 1]),
                True,  # Canonical White perspective for value networks
                flat_feats[idx],
            )
            flat_phases[idx] = _classify_phase_jit(
                cand_boards[i, c],
                int(cand_bars[i, c, 0]),
                int(cand_bars[i, c, 1]),
                int(cand_offs[i, c, 0]),
                int(cand_offs[i, c, 1]),
            )
    return flat_feats, flat_phases


@njit(parallel=True, fastmath=True, cache=True)
def batch_select_best_candidates(
    flat_equities: np.ndarray,
    offsets: np.ndarray,
    cand_counts: np.ndarray,
    turns: np.ndarray,
    epsilon: float,
    rng_states: np.ndarray,
) -> np.ndarray:
    """
    Selects the best candidate index (0 .. cand_counts[i]-1) for each of the N environments
    in parallel across CPU threads (argmax for White, argmin for Black, with epsilon exploration).
    """
    n = cand_counts.shape[0]
    selected = np.zeros(n, dtype=np.int32)

    for i in prange(n):
        cnt = int(cand_counts[i])
        if cnt <= 1:
            selected[i] = 0
            continue

        if epsilon > 0.0:
            s = _xorshift64star(rng_states[i])
            rng_states[i] = s
            u = float(s & np.uint64(0xFFFFFF)) / float(0x1000000)
            if u < epsilon:
                s2 = _xorshift64star(s)
                rng_states[i] = s2
                selected[i] = int(s2 % np.uint64(cnt))
                continue

        base = int(offsets[i])
        is_white = turns[i] == 1
        best_c = 0
        best_val = flat_equities[base]
        for c in range(1, cnt):
            v = flat_equities[base + c]
            if (is_white and v > best_val) or (not is_white and v < best_val):
                best_val = v
                best_c = c
        selected[i] = best_c

    return selected


@njit(fastmath=True)
def _xorshift64star(state: np.uint64) -> np.uint64:
    x = state
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    return x * np.uint64(2685821657736338717)


@njit(parallel=True, fastmath=True, cache=True)
def batch_apply_selected_moves(
    boards: np.ndarray,
    bars: np.ndarray,
    borne_offs: np.ndarray,
    turns: np.ndarray,
    dice: np.ndarray,
    cand_boards: np.ndarray,
    cand_bars: np.ndarray,
    cand_offs: np.ndarray,
    selected_indices: np.ndarray,
    rng_states: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the chosen candidate move for all N environments in parallel via numba.prange,
    detects terminal wins (computing 6-class terminal outcome vectors from White's perspective),
    flips current turn, and rolls fresh dice (1..6, 1..6) thread-safely.

    Returns:
      dones: (N,) bool array
      terminal_outcomes: (N, 6) float32 array
    """
    n = boards.shape[0]
    dones = np.zeros(n, dtype=np.bool_)
    terminal_outcomes = np.zeros((n, 6), dtype=np.float32)

    for i in prange(n):
        sel = int(selected_indices[i])
        for k in range(24):
            boards[i, k] = cand_boards[i, sel, k]
        bars[i, 0] = cand_bars[i, sel, 0]
        bars[i, 1] = cand_bars[i, sel, 1]
        borne_offs[i, 0] = cand_offs[i, sel, 0]
        borne_offs[i, 1] = cand_offs[i, sel, 1]

        # Switch active turn
        turns[i] = 1 - turns[i]

        # Check terminal win
        ow = int(borne_offs[i, 0])
        ob = int(borne_offs[i, 1])
        if ow >= 15:
            dones[i] = True
            if ob > 0:
                terminal_outcomes[i, 0] = 1.0  # WinSingle
            else:
                in_home_or_bar = bars[i, 1] > 0
                if not in_home_or_bar:
                    for p in range(6):
                        if boards[i, p] < 0:
                            in_home_or_bar = True
                            break
                if in_home_or_bar:
                    terminal_outcomes[i, 2] = 1.0  # WinBg
                else:
                    terminal_outcomes[i, 1] = 1.0  # WinGammon
        elif ob >= 15:
            dones[i] = True
            if ow > 0:
                terminal_outcomes[i, 3] = 1.0  # LoseSingle
            else:
                in_home_or_bar = bars[i, 0] > 0
                if not in_home_or_bar:
                    for p in range(18, 24):
                        if boards[i, p] > 0:
                            in_home_or_bar = True
                            break
                if in_home_or_bar:
                    terminal_outcomes[i, 5] = 1.0  # LoseBg
                else:
                    terminal_outcomes[i, 4] = 1.0  # LoseGammon

        # Thread-safe PRNG step to roll next dice
        s = _xorshift64star(rng_states[i])
        d1 = int(s % np.uint64(6)) + 1
        s = _xorshift64star(s)
        d2 = int(s % np.uint64(6)) + 1
        rng_states[i] = s
        dice[i, 0] = d1
        dice[i, 1] = d2

    return dones, terminal_outcomes


@njit(fastmath=True)
def _generate_hard_position_into_jit(
    board_row: np.ndarray,
    bar_row: np.ndarray,
    off_row: np.ndarray,
    rng_state: np.uint64,
    archetype: int,
) -> np.uint64:
    """
    Numba-compiled generator for 4 difficult Backgammon tactical archetypes:
      0 = 2-3 / 1-2 Backgame (defender holds 2 deep anchors, behind 35-70 pips)
      1 = Deep Holding Game (defender holds 1-pt or 2-pt anchor vs opponent 4/5-prime)
      2 = Mutual Prime Battle & Blitz (both sides hold 4+ prime with trapped/bar checkers)
      3 = Crunched Home Board / Crashed (<= 5 active checkers stacked on 1-pt & 2-pt)
    Strictly enforces 15 checkers per side and zero point collisions.
    """
    for k in range(24):
        board_row[k] = 0
    bar_row[0] = 0
    bar_row[1] = 0
    off_row[0] = 0
    off_row[1] = 0

    s = _xorshift64star(rng_state)
    variant = int(s % np.uint64(3))
    s = _xorshift64star(s)
    swap_colors = (s & np.uint64(1)) == np.uint64(1)

    arch = archetype % 4
    if arch == 0:
        # Archetype 0: 2-3 or 1-2 Backgame (White holds 2 deep anchors in 21..23, behind 40-65 pips)
        if variant == 0:
            # 2-3 Backgame (points 22 & 21)
            board_row[22] = 2
            board_row[21] = 2
        elif variant == 1:
            # 1-2 Backgame (points 23 & 22)
            board_row[23] = 2
            board_row[22] = 2
        else:
            # 1-3 Backgame with 5 back checkers (points 23 & 21)
            board_row[23] = 3
            board_row[21] = 2

        back_w = board_row[21] + board_row[22] + board_row[23]
        board_row[5] = 3
        board_row[4] = 2
        board_row[3] = 2
        board_row[7] = 2
        board_row[9] = 15 - (back_w + 9)  # 1 or 2 checkers on point 9

        # Black attacker far ahead in the race on disjoint points {14, 15, 16, 17, 18, 19}
        board_row[19] = -3
        board_row[18] = -3
        board_row[17] = -3
        board_row[16] = -2
        board_row[15] = -2
        if variant == 0:
            board_row[14] = -2
            off_row[1] = 0
        else:
            board_row[14] = -1
            off_row[1] = 1

    elif arch == 1:
        # Archetype 1: Deep Holding Game (White holds 1-pt or 2-pt anchor vs Black prime)
        anchor_idx = 23 if variant == 0 else (22 if variant == 1 else 20)
        board_row[anchor_idx] = 2
        board_row[5] = 3
        board_row[4] = 3
        board_row[3] = 2
        board_row[2] = 2
        board_row[7] = 2
        board_row[8] = 1  # Total White = 15

        # Black prime on {15, 16, 17, 18, 19} + builders on 12 & 10
        board_row[19] = -3
        board_row[18] = -3
        board_row[17] = -2
        board_row[16] = -2
        board_row[15] = -2
        board_row[12] = -2
        board_row[10] = -1  # Total Black = 15

    elif arch == 2:
        # Archetype 2: Mutual Prime Battle & Blitz
        board_row[6] = 2
        board_row[5] = 3
        board_row[4] = 3
        board_row[3] = 2
        board_row[2] = 2
        board_row[23] = 2
        if variant == 0:
            bar_row[0] = 1
        else:
            board_row[8] = 1  # Total White = 15

        board_row[17] = -2
        board_row[18] = -3
        board_row[19] = -3
        board_row[20] = -2
        board_row[21] = -2
        board_row[0] = -2
        if variant == 2:
            bar_row[1] = 1
        else:
            board_row[14] = -1  # Total Black = 15

    else:
        # Archetype 3: Crunched Home Board / Crashed (<= 5 active checkers for White)
        # White has 10 or 11 checkers borne off (4 or 5 active -> triggers CrashedNet!)
        if variant == 0:
            board_row[0] = 2
            board_row[1] = 2
            board_row[22] = 1
            off_row[0] = 10  # 5 active checkers
        elif variant == 1:
            board_row[0] = 2
            board_row[1] = 1
            board_row[23] = 1
            off_row[0] = 11  # 4 active checkers
        else:
            board_row[0] = 3
            board_row[1] = 1
            bar_row[0] = 1
            off_row[0] = 10  # 5 active checkers

        # Black holds home-board points containing White's back checker
        board_row[21] = -2
        board_row[20] = -2
        board_row[19] = -2
        board_row[18] = -2
        board_row[16] = -1
        off_row[1] = 6  # Total Black = 9 + 6 = 15

    if swap_colors:
        # Mirror & swap White <-> Black so both sides train on all 4 archetypes symmetrically
        tmp = np.empty(24, dtype=np.int32)
        for k in range(24):
            tmp[k] = -board_row[23 - k]
        for k in range(24):
            board_row[k] = tmp[k]

        bw = bar_row[0]
        bar_row[0] = bar_row[1]
        bar_row[1] = bw

        ow = off_row[0]
        off_row[0] = off_row[1]
        off_row[1] = ow

    return s


def generate_hard_position(
    rng_state: int, archetype: int
) -> Tuple[BoardState, int]:
    """
    Python wrapper around `_generate_hard_position_into_jit` returning a valid `BoardState`
    and updated PRNG state for the requested archetype (0=Backgame, 1=Holding, 2=PrimeBattle, 3=Crashed).
    """
    board_row = np.zeros(24, dtype=np.int32)
    bar_row = np.zeros(2, dtype=np.int32)
    off_row = np.zeros(2, dtype=np.int32)
    next_s = _generate_hard_position_into_jit(
        board_row,
        bar_row,
        off_row,
        np.uint64(max(1, rng_state)),
        int(archetype),
    )
    turn = "white" if (int(next_s) & 1) == 1 else "black"
    state = BoardState(
        points=[int(x) for x in board_row],
        bar_white=int(bar_row[0]),
        bar_black=int(bar_row[1]),
        off_white=int(off_row[0]),
        off_black=int(off_row[1]),
        current_turn=turn,
    )
    return state, int(next_s)


@njit(parallel=True, fastmath=True, cache=True)
def batch_reset_envs(
    boards: np.ndarray,
    bars: np.ndarray,
    borne_offs: np.ndarray,
    turns: np.ndarray,
    dice: np.ndarray,
    mask: np.ndarray,
    rng_states: np.ndarray,
    hard_ratio: float = 0.0,
) -> None:
    """
    Resets environments where mask[i] is True in parallel across CPU threads.
    With probability `1 - hard_ratio`, resets to the canonical initial Backgammon board;
    with probability `hard_ratio`, resets to a sampled hard curriculum position
    (Backgame / Deep Holding / Mutual Prime Battle / Crashed Home Board).
    """
    n = boards.shape[0]
    for i in prange(n):
        if not mask[i]:
            continue

        s = _xorshift64star(rng_states[i])
        use_hard = False
        if hard_ratio > 0.0:
            u = float(s & np.uint64(0xFFFFFF)) / float(0x1000000)
            if u < hard_ratio:
                use_hard = True

        if use_hard:
            s = _xorshift64star(s)
            arch = int(s & np.uint64(3))
            s = _generate_hard_position_into_jit(
                boards[i], bars[i], borne_offs[i], s, arch
            )
        else:
            for k in range(24):
                boards[i, k] = 0
            boards[i, 23] = 2
            boards[i, 12] = 5
            boards[i, 7] = 3
            boards[i, 5] = 5
            boards[i, 0] = -2
            boards[i, 11] = -5
            boards[i, 16] = -3
            boards[i, 18] = -5

            bars[i, 0] = 0
            bars[i, 1] = 0
            borne_offs[i, 0] = 0
            borne_offs[i, 1] = 0

        s = _xorshift64star(s)
        turns[i] = int(s & np.uint64(1))
        s = _xorshift64star(s)
        dice[i, 0] = int(s % np.uint64(6)) + 1
        s = _xorshift64star(s)
        dice[i, 1] = int(s % np.uint64(6)) + 1
        rng_states[i] = s


@njit(fastmath=True)
def _evaluate_heuristic_probs_into_jit(
    pts: np.ndarray,
    bw: int,
    bb: int,
    ow: int,
    ob: int,
    turn_is_white: bool,
    out_probs: np.ndarray,
) -> float:
    """
    Computes (equity_for_white, class_probs_for_white[6]) inside Numba,
    using exact 54,264-state bear-off tablebase when in pure bear-off.
    """
    # Check pure bear-off state
    pure_bo = bw == 0 and bb == 0
    if pure_bo:
        for i in range(24):
            v = pts[i]
            if v > 0 and i > 5:
                pure_bo = False
                break
            if v < 0 and i < 18:
                pure_bo = False
                break

    if pure_bo:
        w0 = pts[0] if pts[0] > 0 else 0
        w1 = pts[1] if pts[1] > 0 else 0
        w2 = pts[2] if pts[2] > 0 else 0
        w3 = pts[3] if pts[3] > 0 else 0
        w4 = pts[4] if pts[4] > 0 else 0
        w5 = pts[5] if pts[5] > 0 else 0

        b0 = -pts[23] if pts[23] < 0 else 0
        b1 = -pts[22] if pts[22] < 0 else 0
        b2 = -pts[21] if pts[21] < 0 else 0
        b3 = -pts[20] if pts[20] < 0 else 0
        b4 = -pts[19] if pts[19] < 0 else 0
        b5 = -pts[18] if pts[18] < 0 else 0

        w_idx = _index_home_board_jit(w0, w1, w2, w3, w4, w5)
        b_idx = _index_home_board_jit(b0, b1, b2, b3, b4, b5)
        w_total = w0 + w1 + w2 + w3 + w4 + w5
        b_total = b0 + b1 + b2 + b3 + b4 + b5

        p_white_win = 0.0
        if turn_is_white:
            for k in range(BEAROFF_MAX_TURNS + 1):
                pw = float(_BEAROFF_PDF[w_idx, k])
                if pw == 0.0:
                    continue
                p_b_not_done = 1.0 if k == 0 else max(0.0, 1.0 - float(_BEAROFF_CDF[b_idx, k - 1]))
                p_white_win += pw * p_b_not_done
        else:
            for k in range(1, BEAROFF_MAX_TURNS + 1):
                pw = float(_BEAROFF_PDF[w_idx, k])
                if pw == 0.0:
                    continue
                p_b_not_done = max(0.0, 1.0 - float(_BEAROFF_CDF[b_idx, k]))
                p_white_win += pw * p_b_not_done

        p_win = min(1.0, max(0.0, p_white_win))
        p_lose = 1.0 - p_win
        w_gam = float(_BEAROFF_CDF[w_idx, 1]) * p_win * 0.25 if b_total == 15 else 0.0
        b_gam = float(_BEAROFF_CDF[b_idx, 1]) * p_lose * 0.25 if w_total == 15 else 0.0

        out_probs[0] = max(1e-5, p_win - w_gam)
        out_probs[1] = max(1e-5, w_gam)
        out_probs[2] = 1e-5
        out_probs[3] = max(1e-5, p_lose - b_gam)
        out_probs[4] = max(1e-5, b_gam)
        out_probs[5] = 1e-5
        s = out_probs[0] + out_probs[1] + out_probs[2] + out_probs[3] + out_probs[4] + out_probs[5]
        for c in range(6):
            out_probs[c] /= s
        return (p_win - p_lose) + (w_gam - b_gam)

    w_feat = np.empty(FEATURE_VECTOR_SIZE, dtype=np.float32)
    _extract_features_260_into_jit(pts, bw, bb, ow, ob, True, w_feat)

    pip_diff = float(w_feat[198] * 50.0)
    turns_diff = float((w_feat[200] - w_feat[199]) * 20.0)
    prime_diff = float((w_feat[201] - w_feat[202]) * 6.0)
    home_diff = float((np.sqrt(max(0.0, w_feat[203])) - np.sqrt(max(0.0, w_feat[204]))) * 6.0)
    blot_diff = float((w_feat[206] - w_feat[205]) * 6.0)
    trapped_diff = float((w_feat[258] - w_feat[259]) * 6.0)
    bar_diff = float((w_feat[193] - w_feat[192]) * 2.0)
    off_diff = float((w_feat[194] - w_feat[195]) * 15.0)
    is_contact = bool(w_feat[207] > 0.5)

    if not is_contact:
        raw_eq = turns_diff * 0.22 + pip_diff * 0.025 + off_diff * 0.08
    else:
        raw_eq = (
            pip_diff * 0.022
            + turns_diff * 0.06
            + home_diff * 0.14
            + prime_diff * 0.16
            + blot_diff * 0.11
            + trapped_diff * 0.18
            + bar_diff * 0.36
            + off_diff * 0.08
        )

    equity = max(-2.8, min(2.8, raw_eq))
    p_win = 1.0 / (1.0 + np.exp(-1.55 * equity))
    p_lose = 1.0 - p_win

    w_gam = (0.18 * p_win) if ob == 0 and equity > 0.25 else 0.02 * p_win
    b_gam = (0.18 * p_lose) if ow == 0 and equity < -0.25 else 0.02 * p_lose
    w_bg = 0.08 * w_gam if bb > 0 else 0.01 * w_gam
    b_bg = 0.08 * b_gam if bw > 0 else 0.01 * b_gam

    out_probs[0] = max(1e-5, p_win - w_gam)
    out_probs[1] = max(1e-5, w_gam - w_bg)
    out_probs[2] = max(1e-5, w_bg)
    out_probs[3] = max(1e-5, p_lose - b_gam)
    out_probs[4] = max(1e-5, b_gam - b_bg)
    out_probs[5] = max(1e-5, b_bg)
    s = out_probs[0] + out_probs[1] + out_probs[2] + out_probs[3] + out_probs[4] + out_probs[5]
    for c in range(6):
        out_probs[c] /= s
    return equity


@njit(parallel=True, fastmath=True)
def batch_label_positions_teacher(
    boards: np.ndarray,
    bars: np.ndarray,
    borne_offs: np.ndarray,
    turns: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Labels N positions in parallel across all CPU cores using the exact 54,264-state
    bear-off tablebase (for pure bear-off) and multi-factor heuristic teacher.
    Returns:
      equities: (N,) float32
      probs:    (N, 6) float32
    """
    n = boards.shape[0]
    equities = np.empty(n, dtype=np.float32)
    probs = np.empty((n, 6), dtype=np.float32)
    for i in prange(n):
        equities[i] = _evaluate_heuristic_probs_into_jit(
            boards[i],
            int(bars[i, 0]),
            int(bars[i, 1]),
            int(borne_offs[i, 0]),
            int(borne_offs[i, 1]),
            turns[i] == 1,
            probs[i],
        )
    return equities, probs


def evaluate_heuristic_baseline(state: BoardState) -> Tuple[float, np.ndarray]:
    """
    Fast expert heuristic teacher returning (equity_for_white, class_probs_for_white[6]):
    [pWinSingle, pWinGammonOnly, pWinBg, pLoseSingle, pLoseGammonOnly, pLoseBg].
    """
    pts = np.asarray(state.points, dtype=np.int32)
    probs = np.empty(6, dtype=np.float32)
    eq = _evaluate_heuristic_probs_into_jit(
        pts,
        state.bar_white,
        state.bar_black,
        state.off_white,
        state.off_black,
        state.current_turn == "white",
        probs,
    )
    return float(eq), probs
