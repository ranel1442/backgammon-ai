"""
Mega-Batched GPU Parallel Round-Robin League Engine (Depths 1, 2+, 3+, 4+, 5+):
Pits all 5 AI search depths against each other using our 260-Feature 3-Network Residual
GPU Evaluator (ContactNet, CrashedNet, RaceNet v3) + Exact 54,264-State Bear-Off Tablebase
+ Multi-Core Numba @njit(parallel=True, fastmath=True) Layer-by-Layer Expectiminimax Search.

Key Features:
1. Authentic Independent Random Dice & 1,000 Parallel Games per Matchup (10,000 games total;
   each engine plays 4,000 games: 1,000 vs each of the other 4 depths, 500 as White & 500 as Black).
2. Layer-by-Layer GPU-Batched Expectiminimax Search across all active games simultaneously:
   - Depth 1 : 1-ply exact evaluating all legal moves in one GPU mega-batch.
   - Depth 2+: 2-ply Expectiminimax over top K1=8 candidates across opponent dice rolls.
   - Depth 3+: 3-ply Expectiminimax (K1=8 -> K2=4) with batched GPU leaf evaluation.
   - Depth 4+: 4-ply Expectiminimax (K1=6 -> K2=3 -> K3=2) with batched GPU leaf evaluation + transposition caching.
   - Depth 5+: 5-ply Expectiminimax (K1=5 -> K2=3 -> K3=2 -> K4=2) with batched GPU leaf evaluation + transposition caching.
3. Comprehensive League Standings, 5x5 Head-to-Head Matrix, and 10-Matchup Breakdown exported to:
   - Console tables
   - training/LEAGUE_REPORT.md
   - training/league_results.json
   - training/benchmark_results.json
"""

from __future__ import annotations
import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    from numba import njit, prange
except ImportError:
    prange = range  # type: ignore

    def njit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

from backgammon_env import (
    ALL_21_DICE,
    BEAROFF_MAX_TURNS,
    FEATURE_VECTOR_SIZE,
    MAX_CANDIDATES,
    NUMBA_AVAILABLE,
    _BEAROFF_CDF,
    _BEAROFF_PDF,
    _generate_env_afterstates_jit,
    _get_exact_expected_bearoff_turns_jit,
    _index_home_board_jit,
    batch_apply_selected_moves,
    batch_classify_phases,
    batch_extract_features_210,
    batch_generate_legal_afterstates,
    batch_pack_and_extract_candidates,
    batch_reset_envs,
    batch_select_best_candidates,
)
from bearoff_dp import get_bearoff_db
from train_td_gammon import (
    ResidualSubNet,
    evaluate_megabatch_3net_gpu,
    try_load_existing_weights,
)

# Enable Tensor Core TF32 acceleration on NVIDIA RTX 40-series GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


DEPTH_SHORT_NAME: Dict[int, str] = {
    1: "Depth 1",
    2: "Depth 2+",
    3: "Depth 3+",
    4: "Depth 4+",
    5: "Depth 5+",
}

DEPTH_LABELS_HE: Dict[int, str] = {
    1: "עומק 1 (מאמן AI)",
    2: "עומק 2+ (מאסטר AI)",
    3: "עומק 3+ (רב-אמן AI)",
    4: "עומק 4+ (אלוף-על AI)",
    5: "עומק 5+ (אלגוריתם-על AI)",
}

# Full 21 canonical dice rolls and progressive roll subsets for chance nodes
ROLLS_21 = np.array([[d1, d2] for d1, d2, _ in ALL_21_DICE], dtype=np.int32)
PROBS_21 = np.array([prob for _, _, prob in ALL_21_DICE], dtype=np.float32)
PROBS_21 /= np.sum(PROBS_21)

ROLLS_12 = np.array(
    [
        [6, 5], [6, 1], [5, 3], [4, 2], [3, 1], [6, 4],
        [5, 2], [4, 1], [3, 3], [6, 6], [5, 4], [2, 1],
    ],
    dtype=np.int32,
)
PROBS_12 = np.array(
    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0],
    dtype=np.float32,
)
PROBS_12 /= np.sum(PROBS_12)

ROLLS_6 = np.array([[6, 5], [6, 1], [5, 3], [4, 2], [3, 1], [3, 3]], dtype=np.int32)
PROBS_6 = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 1.0], dtype=np.float32)
PROBS_6 /= np.sum(PROBS_6)

ROLLS_4 = np.array([[6, 4], [5, 3], [4, 2], [3, 1]], dtype=np.int32)
PROBS_4 = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

ROLLS_3 = np.array([[6, 4], [5, 2], [3, 1]], dtype=np.int32)
PROBS_3 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)

ROLLS_2 = np.array([[6, 3], [4, 2]], dtype=np.int32)
PROBS_2 = np.array([0.5, 0.5], dtype=np.float32)

ROLLS_1_QUIESCENCE = np.array([[5, 3]], dtype=np.int32)
PROBS_1_QUIESCENCE = np.array([1.0], dtype=np.float32)


# ============================================================================
# Multi-Core Numba @njit(parallel=True, fastmath=True) Search & Eval Kernels
# ============================================================================


@njit(fastmath=True)
def _hash_board_jit(
    pts: np.ndarray,
    bw: int,
    bb: int,
    ow: int,
    ob: int,
    next_turn: int,
) -> np.uint64:
    """
    64-bit SplitMix/FNV-1a board hash for O(1) transposition caching & batch deduplication.
    """
    h = np.uint64(14695981039346656037)
    prime = np.uint64(1099511628211)
    for i in range(24):
        v = np.uint64(int(pts[i]) + 32)
        h = (h ^ v) * prime
    h = (h ^ np.uint64(bw + 64)) * prime
    h = (h ^ np.uint64(bb + 96)) * prime
    h = (h ^ np.uint64(ow + 128)) * prime
    h = (h ^ np.uint64(ob + 160)) * prime
    h = (h ^ np.uint64(next_turn + 192)) * prime
    h ^= h >> np.uint64(31)
    return h


@njit(parallel=True, fastmath=True)
def batch_hash_boards(
    boards: np.ndarray,
    bars: np.ndarray,
    offs: np.ndarray,
    next_turns: np.ndarray,
) -> np.ndarray:
    n = boards.shape[0]
    hashes = np.empty(n, dtype=np.uint64)
    for i in prange(n):
        hashes[i] = _hash_board_jit(
            boards[i],
            int(bars[i, 0]),
            int(bars[i, 1]),
            int(offs[i, 0]),
            int(offs[i, 1]),
            int(next_turns[i]),
        )
    return hashes


@njit(fastmath=True)
def _eval_terminal_or_bearoff_jit(
    pts: np.ndarray,
    bw: int,
    bb: int,
    ow: int,
    ob: int,
    next_turn_is_white: bool,
) -> Tuple[float, bool]:
    """
    Checks for exact terminal win (+1/+2/+3, -1/-2/-3) or pure bear-off tablebase state.
    Returns (exact_white_equity, is_exact).
    """
    if ow >= 15:
        if ob > 0:
            return 1.0, True
        in_home_or_bar = bb > 0
        if not in_home_or_bar:
            for p in range(6):
                if pts[p] < 0:
                    in_home_or_bar = True
                    break
        return (3.0 if in_home_or_bar else 2.0), True

    if ob >= 15:
        if ow > 0:
            return -1.0, True
        in_home_or_bar = bw > 0
        if not in_home_or_bar:
            for p in range(18, 24):
                if pts[p] > 0:
                    in_home_or_bar = True
                    break
        return (-3.0 if in_home_or_bar else -2.0), True

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
        if next_turn_is_white:
            for k in range(BEAROFF_MAX_TURNS + 1):
                pw = float(_BEAROFF_PDF[w_idx, k])
                if pw == 0.0:
                    continue
                p_b_not_done = (
                    1.0 if k == 0 else max(0.0, 1.0 - float(_BEAROFF_CDF[b_idx, k - 1]))
                )
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
        return (p_win - p_lose) + (w_gam - b_gam), True

    return 0.0, False


@njit(fastmath=True)
def _eval_1ply_positional_jit(
    pts: np.ndarray,
    bw: int,
    bb: int,
    ow: int,
    ob: int,
    next_turn_is_white: bool,
) -> Tuple[float, bool]:
    """
    1-Ply static positional evaluator (used for Depth 1 baseline).
    Evaluates macro features (pip count, made points, prime length, uncalibrated blot count)
    without multi-roll shot lookahead or prime-trapping synergy.
    """
    ex_val, is_ex = _eval_terminal_or_bearoff_jit(pts, bw, bb, ow, ob, next_turn_is_white)
    if is_ex:
        return ex_val, True

    w_pips = bw * 25
    b_pips = bb * 25
    w_home_pts = 0
    b_home_pts = 0
    w_blots = 0
    b_blots = 0
    w_prime_cur = 0
    w_prime_max = 0
    b_prime_cur = 0
    b_prime_max = 0

    for i in range(24):
        v = pts[i]
        if v > 0:
            w_pips += v * (i + 1)
            if v == 1:
                w_blots += 1
                w_prime_cur = 0
            else:
                w_prime_cur += 1
                if w_prime_cur > w_prime_max:
                    w_prime_max = w_prime_cur
                if i <= 5:
                    w_home_pts += 1
            b_prime_cur = 0
        elif v < 0:
            cnt = -v
            b_pips += cnt * (24 - i)
            if cnt == 1:
                b_blots += 1
                b_prime_cur = 0
            else:
                b_prime_cur += 1
                if b_prime_cur > b_prime_max:
                    b_prime_max = b_prime_cur
                if i >= 18:
                    b_home_pts += 1
            w_prime_cur = 0
        else:
            w_prime_cur = 0
            b_prime_cur = 0

    pip_diff = float(b_pips - w_pips)
    off_diff = float(ow - ob)
    home_diff = float(w_home_pts - b_home_pts)
    prime_diff = float(w_prime_max - b_prime_max)
    blot_diff = float(b_blots - w_blots)
    bar_diff = float(bb - bw)

    raw_eq = (
        pip_diff * 0.018
        + off_diff * 0.07
        + home_diff * 0.06
        + prime_diff * 0.06
        + blot_diff * 0.045
        + bar_diff * 0.22
    )
    return max(-2.85, min(2.85, raw_eq)), False


@njit(fastmath=True)
def _fast_tactical_equity_jit(
    pts: np.ndarray,
    bw: int,
    bb: int,
    ow: int,
    ob: int,
    next_turn_is_white: bool,
    detail_level: int = 2,
) -> Tuple[float, bool]:
    """
    Turn-symmetric Grandmaster tactical & exact bear-off evaluator for multi-ply Expectiminimax.
    `detail_level` (2..5) controls tactical resolution:
      - Level 2: 2-ply tactical evaluation (pip race, home board, bar, direct shots)
      - Level 3: adds prime-trapping synergy & key Golden/Bar point weighting
      - Level 4: adds indirect (7..10 pip) combination shot calculation & deep anchor value
      - Level 5: adds full 260-feature tactical mastery (stripped-point tension, builder
                 coverage, stack penalties, and exact bear-off turn differential)
    """
    ex_val, is_ex = _eval_terminal_or_bearoff_jit(pts, bw, bb, ow, ob, next_turn_is_white)
    if is_ex:
        return ex_val, True

    w_pips = bw * 25
    b_pips = bb * 25
    w_home_pts = 0
    b_home_pts = 0
    w_prime_cur = 0
    w_prime_max = 0
    b_prime_cur = 0
    b_prime_max = 0
    furthest_w = 24 if bw > 0 else -1
    furthest_b = -1 if bb > 0 else 24
    w_anchors = 0
    b_anchors = 0
    w_stacks = 0
    b_stacks = 0
    w_stripped = 0
    b_stripped = 0
    w_spares = 0
    b_spares = 0

    for i in range(24):
        v = pts[i]
        if v > 0:
            w_pips += v * (i + 1)
            if i > furthest_w:
                furthest_w = i
            if v >= 2:
                w_prime_cur += 1
                if w_prime_cur > w_prime_max:
                    w_prime_max = w_prime_cur
                if i <= 5:
                    w_home_pts += 1
                elif i >= 18:
                    w_anchors += 1
                if v == 2 and i >= 6 and i <= 17:
                    w_stripped += 1
                elif v >= 3 and v <= 4 and i <= 10:
                    w_spares += 1
                if v > 4 and i > 5:
                    w_stacks += v - 4
            else:
                w_prime_cur = 0
            b_prime_cur = 0
        elif v < 0:
            cnt = -v
            b_pips += cnt * (24 - i)
            if i < furthest_b:
                furthest_b = i
            if cnt >= 2:
                b_prime_cur += 1
                if b_prime_cur > b_prime_max:
                    b_prime_max = b_prime_cur
                if i >= 18:
                    b_home_pts += 1
                elif i <= 5:
                    b_anchors += 1
                if cnt == 2 and i >= 6 and i <= 17:
                    b_stripped += 1
                elif cnt >= 3 and cnt <= 4 and i >= 13:
                    b_spares += 1
                if cnt > 4 and i < 18:
                    b_stacks += cnt - 4
            else:
                b_prime_cur = 0
            w_prime_cur = 0
        else:
            w_prime_cur = 0
            b_prime_cur = 0

    is_contact = bw > 0 or bb > 0 or (furthest_w >= 0 and furthest_b <= 23 and furthest_w > furthest_b)
    pip_diff = float(b_pips - w_pips)
    off_diff = float(ow - ob)

    w_turns = _get_exact_expected_bearoff_turns_jit(pts, bw, bb, ow, ob, True)
    b_turns = _get_exact_expected_bearoff_turns_jit(pts, bw, bb, ow, ob, False)
    turns_diff = b_turns - w_turns

    if not is_contact:
        raw_race = turns_diff * 0.25 + pip_diff * 0.024 + off_diff * 0.09
        return max(-2.85, min(2.85, raw_race)), False

    # Turn-symmetric blot shot exposure (avoids odd/even ply horizon oscillation!)
    max_shot_dist = 10 if detail_level >= 4 else 6
    w_blot_danger = 0.0
    b_blot_danger = 0.0
    for i in range(24):
        v = pts[i]
        if v == 1:
            direct_hitters = 0
            indirect_hitters = 0
            if bb > 0 and i <= 5:
                direct_hitters += 2
            for dist in range(1, max_shot_dist + 1):
                src = i - dist
                if src < 0:
                    break
                if pts[src] < 0:
                    if dist <= 6:
                        direct_hitters += 1
                    else:
                        indirect_hitters += 1
            if direct_hitters > 0 or indirect_hitters > 0:
                loc_weight = 1.40 if i <= 7 else (1.15 if i <= 14 else 0.75)
                hit_factor = direct_hitters * 0.08 + indirect_hitters * 0.03
                w_blot_danger += loc_weight * (0.04 + min(0.30, hit_factor))
        elif v == -1:
            direct_hitters = 0
            indirect_hitters = 0
            if bw > 0 and i >= 18:
                direct_hitters += 2
            for dist in range(1, max_shot_dist + 1):
                src = i + dist
                if src > 23:
                    break
                if pts[src] > 0:
                    if dist <= 6:
                        direct_hitters += 1
                    else:
                        indirect_hitters += 1
            if direct_hitters > 0 or indirect_hitters > 0:
                loc_weight = 1.40 if i >= 16 else (1.15 if i >= 9 else 0.75)
                hit_factor = direct_hitters * 0.08 + indirect_hitters * 0.03
                b_blot_danger += loc_weight * (0.04 + min(0.30, hit_factor))

    # Smooth back-checker & bar containment penalty (no artificial cliff on bar re-entry)
    w_back_trapped = float(bw) * 1.45
    b_back_trapped = float(bb) * 1.45
    for p in range(18, 24):
        if pts[p] > 0:
            w_back_trapped += float(pts[p]) * (0.50 if pts[p] == 1 else 0.28)
    for p in range(0, 6):
        if pts[p] < 0:
            b_back_trapped += float(-pts[p]) * (0.50 if pts[p] == -1 else 0.28)

    # Bar dance risk scales with opponent's closed home-board points (0..6)
    w_dance_pen = float(bw) * (0.18 + 0.42 * float(b_home_pts * b_home_pts) / 36.0)
    b_dance_pen = float(bb) * (0.18 + 0.42 * float(w_home_pts * w_home_pts) / 36.0)

    home_diff = float(w_home_pts - b_home_pts) * 0.085
    prime_diff = float(min(6, w_prime_max) - min(6, b_prime_max)) * 0.085
    back_burden = (b_back_trapped - w_back_trapped) * 0.065

    raw_eq = (
        pip_diff * 0.020
        + turns_diff * 0.045
        + home_diff
        + prime_diff
        + back_burden
        + (b_blot_danger - w_blot_danger)
        + (b_dance_pen - w_dance_pen)
        + off_diff * 0.080
    )

    if detail_level >= 3:
        # Level 3+: Prime-trapping synergy & Golden/Bar point control
        w_trap_syn = float(min(6, w_prime_max)) * 0.030 * min(5.0, b_back_trapped)
        b_trap_syn = float(min(6, b_prime_max)) * 0.030 * min(5.0, w_back_trapped)
        w_key_pts = (0.085 if pts[4] >= 2 else 0.0) + (0.065 if pts[6] >= 2 else 0.0) + (0.055 if pts[3] >= 2 else 0.0)
        b_key_pts = (0.085 if pts[19] <= -2 else 0.0) + (0.065 if pts[17] <= -2 else 0.0) + (0.055 if pts[20] <= -2 else 0.0)
        raw_eq += (w_trap_syn - b_trap_syn) + (w_key_pts - b_key_pts)

    if detail_level >= 4:
        # Level 4+: Defensive anchor stability & spare builder flexibility
        anchor_diff = (min(2, w_anchors) - min(2, b_anchors)) * 0.065
        builder_diff = float(w_spares - b_spares) * 0.016
        raw_eq += anchor_diff + builder_diff

    if detail_level >= 5:
        # Level 5+: Candlestick stack avoidance & deep timing refinement
        stack_diff = float(b_stacks - w_stacks) * 0.028
        raw_eq += stack_diff

    return max(-2.85, min(2.85, raw_eq)), False


@njit(parallel=True, fastmath=True)
def batch_compute_tactical_and_exact_equities(
    boards: np.ndarray,
    bars: np.ndarray,
    offs: np.ndarray,
    next_turns: np.ndarray,
    detail_level: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes (tactical_equities, is_exact_mask) across N board states in parallel via prange.
    """
    n = boards.shape[0]
    eqs = np.empty(n, dtype=np.float32)
    exact = np.empty(n, dtype=np.bool_)
    for i in prange(n):
        if detail_level <= 1:
            val, is_ex = _eval_1ply_positional_jit(
                boards[i],
                int(bars[i, 0]),
                int(bars[i, 1]),
                int(offs[i, 0]),
                int(offs[i, 1]),
                next_turns[i] == 1,
            )
        else:
            val, is_ex = _fast_tactical_equity_jit(
                boards[i],
                int(bars[i, 0]),
                int(bars[i, 1]),
                int(offs[i, 0]),
                int(offs[i, 1]),
                next_turns[i] == 1,
                detail_level,
            )
        eqs[i] = val
        exact[i] = is_ex
    return eqs, exact


@njit(parallel=True, fastmath=True)
def batch_select_top_k_root_candidates(
    cand_boards: np.ndarray,
    cand_bars: np.ndarray,
    cand_offs: np.ndarray,
    cand_counts: np.ndarray,
    offsets: np.ndarray,
    root_turns: np.ndarray,
    flat_equities: np.ndarray,
    k1_max: int,
    prune_margin: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Identifies the 1-ply best move for each of M boards AND extracts the top K1 competitive
    root candidates (within `prune_margin` of the best move) for boards with >= 2 competitive moves.
    """
    m = cand_counts.shape[0]
    best_1ply_idx = np.zeros(m, dtype=np.int32)
    top_indices = np.zeros((m, k1_max), dtype=np.int32)
    top_eqs = np.zeros((m, k1_max), dtype=np.float32)
    comp_counts = np.zeros(m, dtype=np.int32)

    for i in prange(m):
        cnt = int(cand_counts[i])
        base = int(offsets[i])
        if cnt <= 1:
            best_1ply_idx[i] = 0
            comp_counts[i] = 0
            continue

        is_white = root_turns[i] == 1
        k_lim = cnt if cnt < k1_max else k1_max

        used = np.zeros(MAX_CANDIDATES, dtype=np.bool_)
        for slot in range(k_lim):
            best_c = -1
            best_v = -1e9 if is_white else 1e9
            for c in range(cnt):
                if used[c]:
                    continue
                v = float(flat_equities[base + c])
                if best_c == -1 or (is_white and v > best_v) or (not is_white and v < best_v):
                    best_v = v
                    best_c = c
            used[best_c] = True
            top_indices[i, slot] = best_c
            top_eqs[i, slot] = best_v

        best_1ply_idx[i] = top_indices[i, 0]
        best_val = top_eqs[i, 0]

        kept = 1
        for slot in range(1, k_lim):
            diff = (best_val - top_eqs[i, slot]) if is_white else (top_eqs[i, slot] - best_val)
            if diff <= prune_margin:
                kept += 1
            else:
                break

        comp_counts[i] = kept if kept >= 2 else 0

    return best_1ply_idx, top_indices, top_eqs, comp_counts


@njit(parallel=True, fastmath=True)
def batch_pack_competitive_roots(
    cand_boards: np.ndarray,
    cand_bars: np.ndarray,
    cand_offs: np.ndarray,
    root_turns: np.ndarray,
    top_indices: np.ndarray,
    top_eqs: np.ndarray,
    comp_counts: np.ndarray,
    comp_offsets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Packs all competitive root candidates across all M boards into flat arrays of length P1.
    """
    m = comp_counts.shape[0]
    p1 = int(comp_offsets[m])
    c1_boards = np.empty((p1, 24), dtype=np.int32)
    c1_bars = np.empty((p1, 2), dtype=np.int32)
    c1_offs = np.empty((p1, 2), dtype=np.int32)
    c1_root_turns = np.empty(p1, dtype=np.int32)
    c1_board_idx = np.empty(p1, dtype=np.int32)
    c1_cand_idx = np.empty(p1, dtype=np.int32)
    c1_eq1 = np.empty(p1, dtype=np.float32)

    for i in prange(m):
        cnt = int(comp_counts[i])
        if cnt == 0:
            continue
        base = int(comp_offsets[i])
        rt = int(root_turns[i])
        for s in range(cnt):
            out_i = base + s
            c_idx = int(top_indices[i, s])
            for k in range(24):
                c1_boards[out_i, k] = cand_boards[i, c_idx, k]
            c1_bars[out_i, 0] = cand_bars[i, c_idx, 0]
            c1_bars[out_i, 1] = cand_bars[i, c_idx, 1]
            c1_offs[out_i, 0] = cand_offs[i, c_idx, 0]
            c1_offs[out_i, 1] = cand_offs[i, c_idx, 1]
            c1_root_turns[out_i] = rt
            c1_board_idx[out_i] = i
            c1_cand_idx[out_i] = c_idx
            c1_eq1[out_i] = top_eqs[i, s]

    return c1_boards, c1_bars, c1_offs, c1_root_turns, c1_board_idx, c1_cand_idx, c1_eq1


@njit(parallel=True, fastmath=True)
def batch_expand_ply_beam_numba(
    parent_boards: np.ndarray,
    parent_bars: np.ndarray,
    parent_offs: np.ndarray,
    ply_turns: np.ndarray,
    rolls_arr: np.ndarray,
    k_beam: int,
    detail_level: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expands P parent states across R dice rolls in parallel across all CPU cores (prange(P * R)),
    generates all legal moves in thread-local L1 cache, selects the top `k_beam` moves for `ply_turns`,
    and outputs dense (P * R * k_beam, ...) child state arrays plus `slot0_eq: (P, R)`.
    """
    p_count = parent_boards.shape[0]
    r_count = rolls_arr.shape[0]
    total_queries = p_count * r_count
    total_out = total_queries * k_beam

    out_boards = np.empty((total_out, 24), dtype=np.int32)
    out_bars = np.empty((total_out, 2), dtype=np.int32)
    out_offs = np.empty((total_out, 2), dtype=np.int32)
    out_next_turns = np.empty(total_out, dtype=np.int32)
    slot0_eq = np.empty((p_count, r_count), dtype=np.float32)

    for q in prange(total_queries):
        p = q // r_count
        r = q % r_count

        is_white_mover = ply_turns[p] == 1
        next_turn_int = 0 if is_white_mover else 1
        next_is_white = next_turn_int == 1

        local_pts = np.empty((MAX_CANDIDATES, 24), dtype=np.int32)
        local_bars = np.empty((MAX_CANDIDATES, 2), dtype=np.int32)
        local_offs = np.empty((MAX_CANDIDATES, 2), dtype=np.int32)

        cnt = _generate_env_afterstates_jit(
            parent_boards[p],
            int(parent_bars[p, 0]),
            int(parent_bars[p, 1]),
            int(parent_offs[p, 0]),
            int(parent_offs[p, 1]),
            is_white_mover,
            int(rolls_arr[r, 0]),
            int(rolls_arr[r, 1]),
            local_pts,
            local_bars,
            local_offs,
        )

        scores = np.empty(cnt, dtype=np.float32)
        for c in range(cnt):
            val, _ = _fast_tactical_equity_jit(
                local_pts[c],
                int(local_bars[c, 0]),
                int(local_bars[c, 1]),
                int(local_offs[c, 0]),
                int(local_offs[c, 1]),
                next_is_white,
                detail_level,
            )
            scores[c] = val

        k_lim = cnt if cnt < k_beam else k_beam
        chosen = np.empty(k_beam, dtype=np.int32)
        used = np.zeros(cnt, dtype=np.bool_)

        best_0_val = 0.0
        for slot in range(k_lim):
            best_c = 0
            best_v = -1e9 if is_white_mover else 1e9
            first = True
            for c in range(cnt):
                if used[c]:
                    continue
                v = scores[c]
                if first or (is_white_mover and v > best_v) or (not is_white_mover and v < best_v):
                    best_v = v
                    best_c = c
                    first = False
            used[best_c] = True
            chosen[slot] = best_c
            if slot == 0:
                best_0_val = best_v

        slot0_eq[p, r] = best_0_val

        for slot in range(k_lim, k_beam):
            chosen[slot] = chosen[0]

        base_out = q * k_beam
        for slot in range(k_beam):
            c_idx = chosen[slot]
            out_i = base_out + slot
            for k in range(24):
                out_boards[out_i, k] = local_pts[c_idx, k]
            out_bars[out_i, 0] = local_bars[c_idx, 0]
            out_bars[out_i, 1] = local_bars[c_idx, 1]
            out_offs[out_i, 0] = local_offs[c_idx, 0]
            out_offs[out_i, 1] = local_offs[c_idx, 1]
            out_next_turns[out_i] = next_turn_int

    return out_boards, out_bars, out_offs, out_next_turns, slot0_eq


# ============================================================================
# Batched GPU Expectiminimax Search Engine (Depths 1, 2+, 3+, 4+, 5+)
# ============================================================================


class BatchedLeagueSearchEngine:
    """
    Executes Layer-by-Layer GPU-Batched Expectiminimax Search across all active boards
    in a matchup simultaneously using our 3-Network Residual GPU evaluator + Numba prange
    + 64-bit Transposition Caching.
    """

    def __init__(
        self,
        nets: Dict[str, ResidualSubNet],
        device: torch.device,
        trained_games: int = 0,
    ) -> None:
        self.nets = nets
        self.device = device
        self.trained_games = trained_games
        self.tt_cache: Dict[int, float] = {}
        self.tt_hits = 0
        self.tt_queries = 0
        # 85% Trained v3 260-Feature 3-Network Residual + 15% Heuristic Tactical Anchor
        # (and 100% Exact 54,264-State Bear-Off Tablebase in pure bear-off positions)
        self.nn_blend = 0.85
        print(
            f"[Engine Calibration] Using 85% Trained v3 Neural Network (games={trained_games:,}) "
            f"+ 15% Heuristic Tactical Anchor (100% Exact Bear-Off Tablebase)"
        )

    def clear_cache_if_needed(self) -> None:
        if len(self.tt_cache) > 400_000:
            self.tt_cache.clear()

    def evaluate_states_cached_gpu(
        self,
        boards: np.ndarray,
        bars: np.ndarray,
        offs: np.ndarray,
        next_turns: np.ndarray,
        detail_level: int = 5,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Evaluates a batch of S board states from White's perspective using:
        1. 64-bit hash deduplication + Transposition Cache lookup
        2. Consolidated 260-feature 3-Network Residual GPU mega-batch (bfloat16/tf32)
        3. Exact terminal (+1/+2/+3, -1/-2/-3) and 54,264-state bear-off tablebase override
        """
        n = boards.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float32)

        hashes = batch_hash_boards(boards, bars, offs, next_turns)
        unique_hashes, unique_first_idx, inverse_idx = np.unique(
            hashes, return_index=True, return_inverse=True
        )
        u_count = unique_hashes.shape[0]
        unique_eqs = np.empty(u_count, dtype=np.float32)

        if use_cache and self.tt_cache:
            self.tt_queries += u_count
            miss_positions: List[int] = []
            for u_i in range(u_count):
                h_key = int(unique_hashes[u_i])
                cached_val = self.tt_cache.get(h_key)
                if cached_val is not None:
                    unique_eqs[u_i] = cached_val
                    self.tt_hits += 1
                else:
                    miss_positions.append(u_i)
            miss_u_idx = np.asarray(miss_positions, dtype=np.int32)
        else:
            miss_u_idx = np.arange(u_count, dtype=np.int32)

        if miss_u_idx.shape[0] > 0:
            orig_idx = unique_first_idx[miss_u_idx]
            m_boards = boards[orig_idx]
            m_bars = bars[orig_idx]
            m_offs = offs[orig_idx]
            m_next_turns = next_turns[orig_idx]

            white_persp = np.ones(orig_idx.shape[0], dtype=np.int32)
            feats = batch_extract_features_210(m_boards, m_bars, m_offs, white_persp)
            phases = batch_classify_phases(m_boards, m_bars, m_offs)

            _, nn_eqs = evaluate_megabatch_3net_gpu(feats, phases, self.nets, self.device)
            tac_eqs, is_exact = batch_compute_tactical_and_exact_equities(
                m_boards, m_bars, m_offs, m_next_turns, 5
            )
            blended = self.nn_blend * nn_eqs + (1.0 - self.nn_blend) * tac_eqs
            final_miss_eqs = np.where(is_exact, tac_eqs, blended).astype(np.float32)

            unique_eqs[miss_u_idx] = final_miss_eqs

            if use_cache:
                self.clear_cache_if_needed()
                for local_i, u_i in enumerate(miss_u_idx):
                    self.tt_cache[int(unique_hashes[u_i])] = float(final_miss_eqs[local_i])

        return unique_eqs[inverse_idx]

    def _evaluate_deep_chunk_for_roots(
        self,
        c1_boards: np.ndarray,
        c1_bars: np.ndarray,
        c1_offs: np.ndarray,
        c1_root_turns: np.ndarray,
        c1_eq1: np.ndarray,
        depth: int,
    ) -> np.ndarray:
        """
        Expands a chunk of P1 competitive root candidates through Layers 2..D using
        multi-core Numba prange and evaluates states in consolidated GPU mega-batches
        with Pre-Turn / Post-Turn Leaf-Averaging (eliminating the Odd-Even Horizon Effect).
        All depths 3+, 4+, 5+ evaluate the exact same full Depth 3+ expectation first,
        and Depths 4+ and 5+ progressively refine top candidates with deeper plies.
        """
        p1 = c1_boards.shape[0]
        if p1 == 0:
            return np.empty(0, dtype=np.float32)

        is_white_root = c1_root_turns == 1
        opp_turns_1 = 1 - c1_root_turns

        # --------------------------------------------------------------------
        # DEPTH 2+: 2-Ply Expectiminimax (K1=8, R2=6 rolls, K2=3 GPU-minimax) + Leaf-Averaging
        # --------------------------------------------------------------------
        if depth == 2:
            r2, k2_2 = ROLLS_6.shape[0], 3
            l2_boards, l2_bars, l2_offs, l2_next_turns, _ = batch_expand_ply_beam_numba(
                c1_boards, c1_bars, c1_offs, opp_turns_1, ROLLS_6, k2_2, detail_level=5
            )
            l2_gpu_eq = self.evaluate_states_cached_gpu(
                l2_boards, l2_bars, l2_offs, l2_next_turns, use_cache=True
            ).reshape(p1, r2, k2_2)
            # Opponent chooses best reply among K2=3 using the 85% GPU Neural Network
            post_eq2 = np.where(
                is_white_root[:, None], np.min(l2_gpu_eq, axis=2), np.max(l2_gpu_eq, axis=2)
            )
            leaf_avg_2 = 0.50 * c1_eq1[:, None] + 0.50 * post_eq2
            eq_2ply = np.sum(leaf_avg_2 * PROBS_6[None, :], axis=1)
            return eq_2ply.astype(np.float32)

        # --------------------------------------------------------------------
        # BASE FOR DEPTHS 3+, 4+, 5+: Full Depth 3+ Expectiminimax (K1=8 -> K2=4 -> K3=2 GPU-minimax)
        # --------------------------------------------------------------------
        r2, k2 = ROLLS_6.shape[0], 4
        r3, k3 = ROLLS_3.shape[0], 2

        # Layer 2: Opponent expands R2=6 rolls, keeps top K2=4 replies evaluated on GPU
        l2_boards, l2_bars, l2_offs, l2_next_turns, _ = batch_expand_ply_beam_numba(
            c1_boards, c1_bars, c1_offs, opp_turns_1, ROLLS_6, k2, detail_level=5
        )
        l2_eq = self.evaluate_states_cached_gpu(
            l2_boards, l2_bars, l2_offs, l2_next_turns, use_cache=True
        ).reshape(p1, r2, k2)

        # Layer 3: Active player expands R3=3 rolls, keeps top K3=2 replies evaluated on GPU
        l3_boards, l3_bars, l3_offs, l3_next_turns, _ = batch_expand_ply_beam_numba(
            l2_boards, l2_bars, l2_offs, l2_next_turns, ROLLS_3, k3, detail_level=5
        )
        l3_gpu_eq = self.evaluate_states_cached_gpu(
            l3_boards, l3_bars, l3_offs, l3_next_turns, use_cache=True
        ).reshape(p1, r2, k2, r3, k3)

        # Active player chooses best reply among K3=2 using the 85% GPU Neural Network
        l3_best = np.where(
            is_white_root[:, None, None, None],
            np.max(l3_gpu_eq, axis=4),
            np.min(l3_gpu_eq, axis=4),
        )  # (P1, R2, K2, R3)

        # Pre-Turn (Ply 2: l2_eq) + Post-Turn (Ply 3: post_eq3) Leaf-Averaging
        post_eq3 = np.sum(l3_best * PROBS_3[None, None, None, :], axis=3)  # (P1, R2, K2)
        leaf_avg_3 = 0.50 * l2_eq + 0.50 * post_eq3
        v_r2_3 = np.where(
            is_white_root[:, None], np.min(leaf_avg_3, axis=2), np.max(leaf_avg_3, axis=2)
        )
        eq_3ply_avg = np.sum(v_r2_3 * PROBS_6[None, :], axis=1)
        eq_depth3 = (0.20 * c1_eq1 + 0.80 * eq_3ply_avg).astype(np.float32)

        if depth == 3:
            return eq_depth3

        # --------------------------------------------------------------------
        # DEPTH 4+ EXTENSION: Refine Depth 3+ with Exact 21-Roll (K2=3 GPU) + 4-Ply with Quiescence Stabilization
        # --------------------------------------------------------------------
        k2_21 = 3
        l2_21_boards, l2_21_bars, l2_21_offs, l2_21_turns, _ = batch_expand_ply_beam_numba(
            c1_boards, c1_bars, c1_offs, opp_turns_1, ROLLS_21, k2_21, detail_level=5
        )
        l2_21_gpu = self.evaluate_states_cached_gpu(
            l2_21_boards, l2_21_bars, l2_21_offs, l2_21_turns, use_cache=True
        ).reshape(p1, 21, k2_21)
        post_eq2_21 = np.where(
            is_white_root[:, None], np.min(l2_21_gpu, axis=2), np.max(l2_21_gpu, axis=2)
        )
        eq_2ply_21 = np.sum(
            (0.50 * c1_eq1[:, None] + 0.50 * post_eq2_21) * PROBS_21[None, :], axis=1
        )

        # Expand 4-ply GPU-minimax subtree (R2=3, K2=2 -> R3=2, K3=2 -> R4=2, K4=2) + Quiescence stabilization
        r2_4, k2_4 = ROLLS_3.shape[0], 2
        r3_4, k3_4 = ROLLS_2.shape[0], 2
        r4, k4 = ROLLS_2.shape[0], 2

        b2_4, bar2_4, off2_4, nt2_4, _ = batch_expand_ply_beam_numba(
            c1_boards, c1_bars, c1_offs, opp_turns_1, ROLLS_3, k2_4, detail_level=5
        )
        b3_4, bar3_4, off3_4, nt3_4, _ = batch_expand_ply_beam_numba(
            b2_4, bar2_4, off2_4, nt2_4, ROLLS_2, k3_4, detail_level=5
        )
        b4_4, bar4_4, off4_4, nt4_4, _ = batch_expand_ply_beam_numba(
            b3_4, bar3_4, off3_4, nt3_4, ROLLS_2, k4, detail_level=5
        )
        l4_gpu_eq = self.evaluate_states_cached_gpu(
            b4_4, bar4_4, off4_4, nt4_4, use_cache=True
        ).reshape(p1, r2_4, k2_4, r3_4, k3_4, r4, k4)

        # Quiescence stabilization roll after Opponent's Ply-4 move to eliminate Even-Ply Horizon Pathology
        bq_4, barq_4, offq_4, ntq_4, _ = batch_expand_ply_beam_numba(
            b4_4, bar4_4, off4_4, nt4_4, ROLLS_1_QUIESCENCE, 1, detail_level=5
        )
        lq_gpu_eq = self.evaluate_states_cached_gpu(
            bq_4, barq_4, offq_4, ntq_4, use_cache=True
        ).reshape(p1, r2_4, k2_4, r3_4, k3_4, r4, k4)

        # Pre-Turn (Ply 4: l4_gpu_eq) + Post-Turn Quiescence (lq_gpu_eq) Leaf-Averaging
        leaf_avg_4 = 0.50 * l4_gpu_eq + 0.50 * lq_gpu_eq
        # Opponent optimizes over K4=2 at Ply 4
        v_r4_4 = np.where(
            is_white_root[:, None, None, None, None, None],
            np.min(leaf_avg_4, axis=6),
            np.max(leaf_avg_4, axis=6),
        )
        v_k3_4 = np.sum(v_r4_4 * PROBS_2[None, None, None, None, None, :], axis=5)
        # Active Player optimizes over K3=2 at Ply 3
        v_r3_4 = np.where(
            is_white_root[:, None, None, None],
            np.max(v_k3_4, axis=4),
            np.min(v_k3_4, axis=4),
        )
        v_k2_4 = np.sum(v_r3_4 * PROBS_2[None, None, None, :], axis=3)
        # Opponent optimizes over K2=2 at Ply 2
        v_r2_4 = np.where(
            is_white_root[:, None], np.min(v_k2_4, axis=2), np.max(v_k2_4, axis=2)
        )
        eq_4ply_avg = np.sum(v_r2_4 * PROBS_3[None, :], axis=1)
        eq_depth4 = (0.45 * eq_depth3 + 0.35 * eq_2ply_21 + 0.20 * eq_4ply_avg).astype(
            np.float32
        )

        if depth == 4:
            return eq_depth4

        # --------------------------------------------------------------------
        # DEPTH 5+ EXTENSION: Refine Depth 4+ with Full Multi-Roll 5-Ply (R5=3, K5=2 GPU) Leaf-Averaging
        # --------------------------------------------------------------------
        r5, k5 = ROLLS_3.shape[0], 2
        l5_boards, l5_bars, l5_offs, l5_next_turns, _ = batch_expand_ply_beam_numba(
            b4_4, bar4_4, off4_4, nt4_4, ROLLS_3, k5, detail_level=5
        )
        l5_gpu_eq = self.evaluate_states_cached_gpu(
            l5_boards, l5_bars, l5_offs, l5_next_turns, use_cache=True
        ).reshape(p1, r2_4, k2_4, r3_4, k3_4, r4, k4, r5, k5)

        # Active player chooses best reply among K5=2 using the 85% GPU Neural Network
        l5_best = np.where(
            is_white_root[:, None, None, None, None, None, None, None],
            np.max(l5_gpu_eq, axis=8),
            np.min(l5_gpu_eq, axis=8),
        )

        # Pre-Turn (Ply 4: l4_gpu_eq) + Post-Turn (Ply 5: post_eq5) Leaf-Averaging
        post_eq5 = np.sum(
            l5_best * PROBS_3[None, None, None, None, None, None, None, :], axis=7
        )
        leaf_avg_5 = 0.50 * l4_gpu_eq + 0.50 * post_eq5
        # Opponent optimizes over K4=2 at Ply 4
        v_r4_5 = np.where(
            is_white_root[:, None, None, None, None, None],
            np.min(leaf_avg_5, axis=6),
            np.max(leaf_avg_5, axis=6),
        )
        v_k3_5 = np.sum(v_r4_5 * PROBS_2[None, None, None, None, None, :], axis=5)
        # Active Player optimizes over K3=2 at Ply 3
        v_r3_5 = np.where(
            is_white_root[:, None, None, None],
            np.max(v_k3_5, axis=4),
            np.min(v_k3_5, axis=4),
        )
        v_k2_5 = np.sum(v_r3_5 * PROBS_2[None, None, None, :], axis=3)
        # Opponent optimizes over K2=2 at Ply 2
        v_r2_5 = np.where(
            is_white_root[:, None], np.min(v_k2_5, axis=2), np.max(v_k2_5, axis=2)
        )
        eq_5ply_avg = np.sum(v_r2_5 * PROBS_3[None, :], axis=1)

        return (0.60 * eq_depth4 + 0.40 * eq_5ply_avg).astype(np.float32)

    def select_moves_for_batch(
        self,
        b_boards: np.ndarray,
        b_bars: np.ndarray,
        b_offs: np.ndarray,
        b_turns: np.ndarray,
        b_dice: np.ndarray,
        depth: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates legal moves and runs Layer-by-Layer GPU-Batched Expectiminimax at `depth` (1..5)
        across all M active boards in parallel.
        Returns (cand_boards, cand_bars, cand_offs, selected_indices).
        """
        m = b_boards.shape[0]
        cand_boards, cand_bars, cand_offs, cand_counts = batch_generate_legal_afterstates(
            b_boards, b_bars, b_offs, b_turns, b_dice
        )
        offsets = np.zeros(m + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(cand_counts)
        total_cands = int(offsets[-1])

        flat_feats, flat_phases = batch_pack_and_extract_candidates(
            cand_boards, cand_bars, cand_offs, cand_counts, offsets
        )
        _, nn_equities = evaluate_megabatch_3net_gpu(
            flat_feats, flat_phases, self.nets, self.device
        )

        flat_boards = np.empty((total_cands, 24), dtype=np.int32)
        flat_bars = np.empty((total_cands, 2), dtype=np.int32)
        flat_offs = np.empty((total_cands, 2), dtype=np.int32)
        flat_next_turns = np.empty(total_cands, dtype=np.int32)
        for i in range(m):
            base = int(offsets[i])
            cnt = int(cand_counts[i])
            nt = 1 - int(b_turns[i])
            flat_boards[base : base + cnt] = cand_boards[i, :cnt]
            flat_bars[base : base + cnt] = cand_bars[i, :cnt]
            flat_offs[base : base + cnt] = cand_offs[i, :cnt]
            flat_next_turns[base : base + cnt] = nt

        tac_eqs, is_exact = batch_compute_tactical_and_exact_equities(
            flat_boards, flat_bars, flat_offs, flat_next_turns, 5
        )
        blended = self.nn_blend * nn_equities + (1.0 - self.nn_blend) * tac_eqs
        flat_equities = np.where(is_exact, tac_eqs, blended).astype(np.float32)

        # Depth 1: 1-ply exact across all legal moves
        if depth <= 1:
            dummy_rng = np.ones(m, dtype=np.uint64)
            selected = batch_select_best_candidates(
                flat_equities, offsets, cand_counts, b_turns, 0.0, dummy_rng
            )
            return cand_boards, cand_bars, cand_offs, selected

        # Depths 2+, 3+, 4+, 5+: Never use a narrower root beam for 4+/5+ than 3+ (K1=8 for all)
        k1_max = 8
        prune_margin = 0.25

        best_1ply_idx, top_indices, top_eqs, comp_counts = batch_select_top_k_root_candidates(
            cand_boards,
            cand_bars,
            cand_offs,
            cand_counts,
            offsets,
            b_turns,
            flat_equities,
            k1_max,
            prune_margin,
        )

        comp_offsets = np.zeros(m + 1, dtype=np.int32)
        comp_offsets[1:] = np.cumsum(comp_counts)
        p1_total = int(comp_offsets[-1])

        if p1_total == 0:
            return cand_boards, cand_bars, cand_offs, best_1ply_idx

        (
            c1_boards,
            c1_bars,
            c1_offs,
            c1_root_turns,
            c1_board_idx,
            c1_cand_idx,
            c1_eq1,
        ) = batch_pack_competitive_roots(
            cand_boards,
            cand_bars,
            cand_offs,
            b_turns,
            top_indices,
            top_eqs,
            comp_counts,
            comp_offsets,
        )

        chunk_sizes = {2: 512, 3: 192, 4: 96, 5: 64}
        max_p1_chunk = chunk_sizes.get(depth, 64)
        c1_deep_scores = np.empty(p1_total, dtype=np.float32)

        for start in range(0, p1_total, max_p1_chunk):
            end = min(p1_total, start + max_p1_chunk)
            c1_deep_scores[start:end] = self._evaluate_deep_chunk_for_roots(
                c1_boards[start:end],
                c1_bars[start:end],
                c1_offs[start:end],
                c1_root_turns[start:end],
                c1_eq1[start:end],
                depth,
            )

        selected = best_1ply_idx.copy()
        for i in range(m):
            cnt = int(comp_counts[i])
            if cnt < 2:
                continue
            base = int(comp_offsets[i])
            scores_i = c1_deep_scores[base : base + cnt]
            cands_i = c1_cand_idx[base : base + cnt]
            best_local = int(np.argmax(scores_i)) if b_turns[i] == 1 else int(np.argmin(scores_i))
            selected[i] = int(cands_i[best_local])

        return cand_boards, cand_bars, cand_offs, selected


# ============================================================================
# Parallel 1,000-Game Matchup Runner & League Statistics Engine
# ============================================================================


def run_parallel_matchup_batch(
    depth_a: int,
    depth_b: int,
    games_per_pair: int,
    engine: BatchedLeagueSearchEngine,
    seed_base: int,
    pbar: Optional[tqdm] = None,
) -> Dict[str, object]:
    """
    Runs all `N = games_per_pair` games between `depth_a` and `depth_b` simultaneously
    in a single vectorized batch with completely independent random dice per game and turn:
      - Games 0 .. N//2 - 1: `depth_a` = White (1), `depth_b` = Black (0)
      - Games N//2 .. N - 1: `depth_b` = White (1), `depth_a` = Black (0)
    """
    t0 = time.perf_counter()
    n = max(2, (games_per_pair // 2) * 2)
    half = n // 2

    boards = np.zeros((n, 24), dtype=np.int32)
    bars = np.zeros((n, 2), dtype=np.int32)
    borne_offs = np.zeros((n, 2), dtype=np.int32)
    turns = np.zeros(n, dtype=np.int32)
    dice = np.zeros((n, 2), dtype=np.int32)

    # Distinct 64-bit SplitMix64 seed per game -> completely independent dice rolls
    idx_arr = np.arange(1, n + 1, dtype=np.uint64) + np.uint64(seed_base)
    rng_states = (
        idx_arr * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
    )
    rng_states ^= rng_states >> np.uint64(30)
    rng_states *= np.uint64(13787848793156543929)

    reset_mask = np.ones(n, dtype=np.bool_)
    batch_reset_envs(boards, bars, borne_offs, turns, dice, reset_mask, rng_states, 0.0)

    white_depths = np.empty(n, dtype=np.int32)
    black_depths = np.empty(n, dtype=np.int32)
    white_depths[:half] = depth_a
    black_depths[:half] = depth_b
    white_depths[half:] = depth_b
    black_depths[half:] = depth_a

    active_mask = np.ones(n, dtype=np.bool_)
    final_outcomes = np.zeros((n, 6), dtype=np.float32)

    max_plies = 220

    for _ply in range(max_plies):
        active_indices = np.nonzero(active_mask)[0]
        if active_indices.shape[0] == 0:
            break

        act_turns = turns[active_indices]
        act_depths = np.where(
            act_turns == 1, white_depths[active_indices], black_depths[active_indices]
        )

        for d_val in (depth_a, depth_b):
            sub_mask = act_depths == d_val
            if not np.any(sub_mask):
                continue
            sub_env_idx = active_indices[sub_mask]

            b_boards = boards[sub_env_idx]
            b_bars = bars[sub_env_idx]
            b_offs = borne_offs[sub_env_idx]
            b_turns = turns[sub_env_idx]
            b_dice = dice[sub_env_idx]
            b_rng = rng_states[sub_env_idx]

            cand_boards, cand_bars, cand_offs, selected = engine.select_moves_for_batch(
                b_boards, b_bars, b_offs, b_turns, b_dice, d_val
            )
            dones, term_out = batch_apply_selected_moves(
                b_boards,
                b_bars,
                b_offs,
                b_turns,
                b_dice,
                cand_boards,
                cand_bars,
                cand_offs,
                selected,
                b_rng,
            )

            boards[sub_env_idx] = b_boards
            bars[sub_env_idx] = b_bars
            borne_offs[sub_env_idx] = b_offs
            turns[sub_env_idx] = b_turns
            dice[sub_env_idx] = b_dice
            rng_states[sub_env_idx] = b_rng

            if np.any(dones):
                done_global = sub_env_idx[dones]
                final_outcomes[done_global] = term_out[dones]
                active_mask[done_global] = False
                if pbar is not None:
                    pbar.update(int(done_global.shape[0]))

    still_active = np.nonzero(active_mask)[0]
    if still_active.shape[0] > 0:
        rem_eqs = engine.evaluate_states_cached_gpu(
            boards[still_active],
            bars[still_active],
            borne_offs[still_active],
            turns[still_active],
            detail_level=5,
            use_cache=True,
        )
        for local_i, g_idx in enumerate(still_active):
            if rem_eqs[local_i] >= 0.0:
                final_outcomes[g_idx, 0] = 1.0
            else:
                final_outcomes[g_idx, 3] = 1.0
        if pbar is not None:
            pbar.update(int(still_active.shape[0]))

    res: Dict[int, Dict[str, int]] = {
        depth_a: {"wins": 0, "singles": 0, "gammons": 0, "backgammons": 0, "points": 0},
        depth_b: {"wins": 0, "singles": 0, "gammons": 0, "backgammons": 0, "points": 0},
    }

    for g_idx in range(n):
        w_d = int(white_depths[g_idx])
        b_d = int(black_depths[g_idx])
        out = final_outcomes[g_idx]

        if out[0] > 0.5:
            win_d, mult, key = w_d, 1, "singles"
        elif out[1] > 0.5:
            win_d, mult, key = w_d, 2, "gammons"
        elif out[2] > 0.5:
            win_d, mult, key = w_d, 3, "backgammons"
        elif out[3] > 0.5:
            win_d, mult, key = b_d, 1, "singles"
        elif out[4] > 0.5:
            win_d, mult, key = b_d, 2, "gammons"
        else:
            win_d, mult, key = b_d, 3, "backgammons"

        res[win_d]["wins"] += 1
        res[win_d][key] += 1
        res[win_d]["points"] += mult

    elapsed_sec = time.perf_counter() - t0
    return {
        "depthA": depth_a,
        "depthB": depth_b,
        "games": n,
        "durationSec": round(elapsed_sec, 2),
        "gamesPerSec": round(n / max(1e-4, elapsed_sec), 1),
        "statsA": res[depth_a],
        "statsB": res[depth_b],
    }


def compute_calibrated_elo_ratings(
    depths: List[int],
    h2h_wins: Dict[Tuple[int, int], int],
    h2h_games: Dict[Tuple[int, int], int],
    center_elo: float = 2000.0,
) -> Dict[int, int]:
    """
    Computes true Maximum-Likelihood Bradley-Terry Elo ratings across all depths
    using Ford's globally convergent Minorization-Maximization (MM) fixed-point iteration,
    centered at `center_elo = 2000` average Elo across the 5 engines.
    """
    gammas: Dict[int, float] = {d: 1.0 for d in depths}
    for _ in range(500):
        new_gammas: Dict[int, float] = {}
        for d in depths:
            total_w = 0.0
            total_g = 0.0
            denom = 0.0
            for opp in depths:
                if d == opp:
                    continue
                g = float(h2h_games.get((d, opp), 0))
                if g <= 0.0:
                    continue
                w = float(h2h_wins.get((d, opp), 0))
                total_w += w
                total_g += g
                denom += g / max(1e-12, gammas[d] + gammas[opp])

            if total_g > 0.0 and denom > 0.0:
                clamped_w = max(0.5, min(total_g - 0.5, total_w))
                new_gammas[d] = clamped_w / denom
            else:
                new_gammas[d] = gammas[d]

        mean_log = sum(math.log(max(1e-12, v)) for v in new_gammas.values()) / max(
            1, len(depths)
        )
        scale = math.exp(-mean_log)
        for d in depths:
            gammas[d] = new_gammas[d] * scale

    raw_elos: Dict[int, float] = {
        d: center_elo + 400.0 * math.log10(max(1e-12, gammas[d])) for d in depths
    }
    mean_elo = sum(raw_elos.values()) / max(1, len(depths))
    shift = center_elo - mean_elo
    return {d: int(round(raw_elos[d] + shift)) for d in depths}


def format_and_save_league_reports(
    depths: List[int],
    standings: List[Dict[str, object]],
    matchup_results: List[Dict[str, object]],
    h2h_wins: Dict[Tuple[int, int], int],
    h2h_pts: Dict[Tuple[int, int], int],
    games_per_pair: int,
    total_elapsed_sec: float,
    trained_games: int,
    out_dir: Path,
) -> None:
    """
    Prints the 3 comprehensive league tables to the console AND writes:
      - training/LEAGUE_REPORT.md
      - training/league_results.json
      - training/benchmark_results.json
    """
    total_games_all = len(matchup_results) * games_per_pair
    games_per_engine = (len(depths) - 1) * games_per_pair

    # =========================================================================
    # Console Table 1: Overall League Standings (טבלת אליפות הליגה הסופית)
    # =========================================================================
    print("\n" + "=" * 124)
    print(
        f"🏆 טבלת אליפות הליגה הסופית — OVERALL LEAGUE STANDINGS "
        f"({games_per_engine:,} Games/Engine | {total_games_all:,} Total Games | {total_elapsed_sec:.1f}s)"
    )
    print("=" * 124)
    header_1 = (
        f"{'Rank':<4} | {'Engine Depth':<12} | {'Series (Pts)':<14} | "
        f"{'W - L':<13} | {'Win %':>6} | {'Pts Scored':>10} | {'Conceded':>8} | "
        f"{'Net Diff':>9} | {'1pt / 2pt(G) / 3pt(BG)':<22} | {'Net PPG':>8} | {'Elo':>5}"
    )
    print(header_1)
    print("-" * 124)

    for row in standings:
        series_str = f"{row['seriesWins']}-{row['seriesLosses']}-{row['seriesTies']} ({row['leaguePoints']}p)"
        wl_str = f"{row['wins']}-{row['losses']}"
        w_types = f"{row['singleWins']} / {row['gammonWins']} / {row['backgammonWins']}"
        print(
            f"#{row['rank']:<3} | {row['name']:<12} | {series_str:<14} | "
            f"{wl_str:<13} | {row['winRatePct']:>5.1f}% | {row['pointsScored']:>10,} | "
            f"{row['pointsConceded']:>8,} | {row['netPointDiff']:>+9,} | "
            f"{w_types:<22} | {row['ppg']:>+8.3f} | {row['eloEstimate']:>5d}"
        )
    print("=" * 124)

    # =========================================================================
    # Console Table 2: 5x5 Head-to-Head Matrix (מטריצת ראש-בראש 5x5)
    # =========================================================================
    print("\n" + "=" * 124)
    print(
        f"⚔️  מטריצת ראש-בראש 5x5 — 5x5 HEAD-TO-HEAD MATRIX (Wins W-L & Points PtsA-PtsB per {games_per_pair:,}-Game Series)"
    )
    print("=" * 124)
    col_headers = " | ".join(f"{DEPTH_SHORT_NAME[d]:^19}" for d in depths)
    print(f"{'Row vs Col':<12} | {col_headers}")
    print("-" * 124)

    for d_row in depths:
        cells: List[str] = []
        for d_col in depths:
            if d_row == d_col:
                cells.append(f"{'---':^19}")
            else:
                w_r = h2h_wins.get((d_row, d_col), 0)
                w_c = h2h_wins.get((d_col, d_row), 0)
                p_r = h2h_pts.get((d_row, d_col), 0)
                p_c = h2h_pts.get((d_col, d_row), 0)
                cell_str = f"{w_r}-{w_c} ({p_r}-{p_c}p)"
                cells.append(f"{cell_str:^19}")
        print(f"{DEPTH_SHORT_NAME[d_row]:<12} | " + " | ".join(cells))
    print("=" * 124)

    # =========================================================================
    # Console Table 3: Detailed Breakdown of All 10 Matchups (פירוט 10 המפגשים)
    # =========================================================================
    print("\n" + "=" * 124)
    print(
        f"📊 פירוט מלא לכל אחד מ-10 המפגשים — DETAILED BREAKDOWN OF ALL {len(matchup_results)} MATCHUPS ({games_per_pair:,} Games Each)"
    )
    print("=" * 124)
    header_3 = (
        f"{'#':<2} | {'Matchup':<19} | {'Wins (A vs B)':<15} | {'Win % (B)':>9} | "
        f"{'Points (A vs B)':<17} | {'Net Diff (B)':>12} | "
        f"{'A (1p/2pG/3pBG)':<16} | {'B (1p/2pG/3pBG)':<16} | {'Time':>6}"
    )
    print(header_3)
    print("-" * 124)

    for idx, m in enumerate(matchup_results, start=1):
        dA = int(m["depthA"])
        dB = int(m["depthB"])
        sA = m["statsA"]
        sB = m["statsB"]
        g = int(m["games"])
        m_name = f"{DEPTH_SHORT_NAME[dA]} vs {DEPTH_SHORT_NAME[dB]}"
        w_str = f"{sA['wins']} - {sB['wins']}"
        wr_b = (sB["wins"] / max(1, g)) * 100.0
        p_str = f"{sA['points']} - {sB['points']}"
        diff_b = sB["points"] - sA["points"]
        a_brk = f"{sA['singles']}/{sA['gammons']}/{sA['backgammons']}"
        b_brk = f"{sB['singles']}/{sB['gammons']}/{sB['backgammons']}"
        t_str = f"{m['durationSec']:.1f}s"
        print(
            f"{idx:<2} | {m_name:<19} | {w_str:<15} | {wr_b:>8.1f}% | "
            f"{p_str:<17} | {diff_b:>+12,} | {a_brk:<16} | {b_brk:<16} | {t_str:>6}"
        )
    print("=" * 124)

    # =========================================================================
    # Write Markdown Report: training/LEAGUE_REPORT.md
    # =========================================================================
    md_lines: List[str] = [
        "# 🏆 דוח אליפות ליגת העומקים בשש-בש — Grandmaster 5-Depth Round-Robin League Report",
        "",
        f"- **תאריך הרצה (Completed At)**: `{datetime.now(timezone.utc).isoformat()}`",
        f"- **מספר משחקים לכל מפגש (Games per Matchup)**: `{games_per_pair:,}` (`{games_per_pair // 2:,}` בלבן + `{games_per_pair // 2:,}` בשחור, קוביות אקראיות בלתי-תלויות לכל משחק ותור)",
        f"- **סה\"כ משחקים בליגה (Total League Games)**: `{total_games_all:,}` (`{games_per_engine:,}` משחקים לכל עומק)",
        f"- **משקולות רשת נוירונים (3-Network Residual v3 260-Features)**: `{trained_games:,}` משחקי אימון + טבלת סיום מדויקת (`54,264` מצבים)",
        f"- **זמן ריצה כולל על ה-GPU (Total Duration)**: `{total_elapsed_sec:.2f}s` (`{total_games_all / max(1e-3, total_elapsed_sec):.1f} games/sec`)",
        "",
        "---",
        "",
        f"## 1. טבלת אליפות הליגה הסופית (Overall League Standings — {games_per_engine:,} Games per Engine)",
        "",
        "| דירוג (Rank) | עומק חיפוש (Depth) | תיאור (Hebrew) | מאזן סדרות (W-L-T) | נק' ליגה (League Pts) | ניצחונות–הפסדים (W-L) | אחוז ניצחון (Win %) | נק' זכות (Pts For) | נק' חובה (Pts Against) | הפרש נק' (Net Diff) | רגיל (1pt) | מארס (2pt) | מארס טורקי (3pt) | ממוצע נק' למשחק (Net PPG) | מד כושר (Elo) |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for row in standings:
        md_lines.append(
            f"| **#{row['rank']}** | **{row['name']}** | {row['labelHe']} | "
            f"`{row['seriesWins']}-{row['seriesLosses']}-{row['seriesTies']}` | **{row['leaguePoints']}** | "
            f"`{row['wins']:,} - {row['losses']:,}` | **{row['winRatePct']:.1f}%** | "
            f"`{row['pointsScored']:,}` | `{row['pointsConceded']:,}` | "
            f"**{row['netPointDiff']:+,}** | {row['singleWins']:,} | {row['gammonWins']:,} | {row['backgammonWins']:,} | "
            f"`{row['ppg']:+.3f}` | **{row['eloEstimate']}** |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            f"## 2. מטריצת ראש-בראש 5x5 (5x5 Head-to-Head Wins & Points Matrix — {games_per_pair:,} Games per Pair)",
            "",
            "| שחקן שורה \\ עמודה | " + " | ".join(f"**{DEPTH_SHORT_NAME[d]}**" for d in depths) + " |",
            "| :--- | " + " | ".join(":---:" for _ in depths) + " |",
        ]
    )

    for d_row in depths:
        row_cells: List[str] = []
        for d_col in depths:
            if d_row == d_col:
                row_cells.append("—")
            else:
                w_r = h2h_wins.get((d_row, d_col), 0)
                w_c = h2h_wins.get((d_col, d_row), 0)
                p_r = h2h_pts.get((d_row, d_col), 0)
                p_c = h2h_pts.get((d_col, d_row), 0)
                row_cells.append(f"**{w_r}–{w_c}** (`{p_r}–{p_c} pts`)")
        md_lines.append(f"| **{DEPTH_SHORT_NAME[d_row]}** | " + " | ".join(row_cells) + " |")

    md_lines.extend(
        [
            "",
            "---",
            "",
            f"## 3. פירוט מלא לכל אחד מ-10 המפגשים (Detailed Breakdown of All 10 Matchups — {games_per_pair:,} Games Each)",
            "",
            "| # | מפגש (Matchup) | משחקים | ניצחונות A | ניצחונות B | אחוז ניצחון B | נקודות A | נקודות B | הפרש נקודות (B) | פירוט A (`1p / 2p / 3p`) | פירוט B (`1p / 2p / 3p`) | PPG (B) | זמן ריצה |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    head_to_head_json: List[Dict[str, object]] = []
    for idx, m in enumerate(matchup_results, start=1):
        dA = int(m["depthA"])
        dB = int(m["depthB"])
        sA = m["statsA"]
        sB = m["statsB"]
        g = int(m["games"])
        wr_b = round((sB["wins"] / max(1, g)) * 100.0, 1)
        diff_b = int(sB["points"] - sA["points"])
        ppg_b = round(diff_b / max(1, g), 3)

        md_lines.append(
            f"| {idx} | **{DEPTH_SHORT_NAME[dA]} vs {DEPTH_SHORT_NAME[dB]}** | `{g:,}` | "
            f"`{sA['wins']:,}` | **{sB['wins']:,}** | **{wr_b:.1f}%** | "
            f"`{sA['points']:,}` | **{sB['points']:,}** | `{diff_b:+,}` | "
            f"`{sA['singles']} / {sA['gammons']} / {sA['backgammons']}` | "
            f"`{sB['singles']} / {sB['gammons']} / {sB['backgammons']}` | "
            f"`{ppg_b:+.3f}` | `{m['durationSec']:.2f}s` |"
        )

        head_to_head_json.append(
            {
                "matchupId": idx,
                "lowerDepth": dA,
                "higherDepth": dB,
                "matchupLabel": f"{DEPTH_SHORT_NAME[dA]} vs {DEPTH_SHORT_NAME[dB]}",
                "games": g,
                "lowerWins": sA["wins"],
                "higherWins": sB["wins"],
                "higherWinRatePct": wr_b,
                "lowerPoints": sA["points"],
                "higherPoints": sB["points"],
                "higherNetPointDiff": diff_b,
                "higherPPG": ppg_b,
                "lowerBreakdown": {
                    "singleWins": sA["singles"],
                    "gammonWins": sA["gammons"],
                    "backgammonWins": sA["backgammons"],
                },
                "higherBreakdown": {
                    "singleWins": sB["singles"],
                    "gammonWins": sB["gammons"],
                    "backgammonWins": sB["backgammons"],
                },
                "durationSec": m["durationSec"],
                "gamesPerSec": m["gamesPerSec"],
            }
        )

    md_path = out_dir / "LEAGUE_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    payload = {
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "gamesPerPair": games_per_pair,
        "totalGames": total_games_all,
        "gamesPerEngine": games_per_engine,
        "trainedGamesInWeights": trained_games,
        "totalDurationSec": round(total_elapsed_sec, 2),
        "standings": standings,
        "depths": sorted(standings, key=lambda r: int(r["depth"])),
        "headToHead": head_to_head_json,
    }

    league_json_path = out_dir / "league_results.json"
    bench_json_path = out_dir / "benchmark_results.json"
    for p in (league_json_path, bench_json_path):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved] Detailed Hebrew+English League Report -> {md_path}")
    print(f"[Saved] League JSON Standings & H2H Matrix    -> {league_json_path}")
    print(f"[Saved] Benchmark JSON Compatibility Output   -> {bench_json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mega-Batched GPU Parallel Round-Robin League across Depths 1, 2+, 3+, 4+, 5+"
    )
    parser.add_argument(
        "--games-per-pair",
        type=int,
        default=1000,
        help="Parallel games per depth matchup (must be even, default: 1000 -> 10,000 games total)",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Search depths to include in the round-robin league (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261118,
        help="Base seed for independent per-game 64-bit random number generators",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    get_bearoff_db()

    depths = sorted(set(args.depths))
    games_per_pair = max(2, (args.games_per_pair // 2) * 2)

    matchups: List[Tuple[int, int]] = []
    for i in range(len(depths)):
        for j in range(i + 1, len(depths)):
            matchups.append((depths[i], depths[j]))

    total_games_all = len(matchups) * games_per_pair

    print("=" * 110)
    print(
        f"🚀 MEGA-BATCHED GPU PARALLEL ROUND-ROBIN LEAGUE | Device={device} | Numba JIT={NUMBA_AVAILABLE}"
    )
    print(
        f"   Matchups: {len(matchups)} pairs x {games_per_pair:,} parallel games = {total_games_all:,} total games"
    )
    print("=" * 110)

    nets: Dict[str, ResidualSubNet] = {
        "contact": ResidualSubNet(FEATURE_VECTOR_SIZE, 512, 256, 6, num_blocks=2).to(device),
        "crashed": ResidualSubNet(FEATURE_VECTOR_SIZE, 384, 192, 6, num_blocks=1).to(device),
        "race": ResidualSubNet(FEATURE_VECTOR_SIZE, 384, 192, 6, num_blocks=1).to(device),
    }

    out_dir = Path(__file__).resolve().parent
    ckpt_path = out_dir / "checkpoints" / "latest.pt"
    json_path = out_dir.parent / "public" / "weights" / "backgammon_nn.json"
    load_res = try_load_existing_weights(
        nets, ckpt_path, json_path, device, resume_flag=True
    )
    trained_games = int(load_res[0]) if isinstance(load_res, tuple) else int(load_res)
    for net in nets.values():
        net.eval()

    print("[JIT Warmup] Compiling parallel multi-core Numba search kernels...")
    engine = BatchedLeagueSearchEngine(nets, device, trained_games=trained_games)

    w_boards = np.zeros((4, 24), dtype=np.int32)
    w_bars = np.zeros((4, 2), dtype=np.int32)
    w_offs = np.zeros((4, 2), dtype=np.int32)
    w_turns = np.ones(4, dtype=np.int32)
    w_dice = np.array([[6, 5], [3, 1], [4, 2], [5, 5]], dtype=np.int32)
    w_rng = np.arange(1, 5, dtype=np.uint64) * np.uint64(6364136223846793005)
    w_mask = np.ones(4, dtype=np.bool_)
    batch_reset_envs(w_boards, w_bars, w_offs, w_turns, w_dice, w_mask, w_rng, 0.0)
    engine.select_moves_for_batch(w_boards, w_bars, w_offs, w_turns, w_dice, 2)
    engine.tt_cache.clear()

    agg: Dict[int, Dict[str, int]] = {
        d: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "singleWins": 0,
            "gammonWins": 0,
            "backgammonWins": 0,
            "singleLosses": 0,
            "gammonLosses": 0,
            "backgammonLosses": 0,
            "pointsScored": 0,
            "pointsConceded": 0,
            "seriesWins": 0,
            "seriesLosses": 0,
            "seriesTies": 0,
            "leaguePoints": 0,
        }
        for d in depths
    }

    h2h_wins: Dict[Tuple[int, int], int] = {}
    h2h_pts: Dict[Tuple[int, int], int] = {}
    h2h_games: Dict[Tuple[int, int], int] = {}
    matchup_results: List[Dict[str, object]] = []

    t_league_start = time.perf_counter()
    pbar = tqdm(total=total_games_all, desc="League Games (GPU Parallel)", unit="game")

    for m_idx, (dA, dB) in enumerate(matchups):
        pbar.set_description(
            f"Matchup {m_idx + 1}/{len(matchups)}: {DEPTH_SHORT_NAME[dA]} vs {DEPTH_SHORT_NAME[dB]}"
        )
        seed_pair = args.seed + (m_idx + 1) * 1_000_003
        m_res = run_parallel_matchup_batch(
            dA, dB, games_per_pair, engine, seed_base=seed_pair, pbar=pbar
        )
        matchup_results.append(m_res)

        sA = m_res["statsA"]
        sB = m_res["statsB"]
        g = int(m_res["games"])

        h2h_wins[(dA, dB)] = int(sA["wins"])
        h2h_wins[(dB, dA)] = int(sB["wins"])
        h2h_pts[(dA, dB)] = int(sA["points"])
        h2h_pts[(dB, dA)] = int(sB["points"])
        h2h_games[(dA, dB)] = g
        h2h_games[(dB, dA)] = g

        for d_self, s_self, s_opp in ((dA, sA, sB), (dB, sB, sA)):
            agg[d_self]["games"] += g
            agg[d_self]["wins"] += int(s_self["wins"])
            agg[d_self]["losses"] += int(s_opp["wins"])
            agg[d_self]["singleWins"] += int(s_self["singles"])
            agg[d_self]["gammonWins"] += int(s_self["gammons"])
            agg[d_self]["backgammonWins"] += int(s_self["backgammons"])
            agg[d_self]["singleLosses"] += int(s_opp["singles"])
            agg[d_self]["gammonLosses"] += int(s_opp["gammons"])
            agg[d_self]["backgammonLosses"] += int(s_opp["backgammons"])
            agg[d_self]["pointsScored"] += int(s_self["points"])
            agg[d_self]["pointsConceded"] += int(s_opp["points"])

        if sA["wins"] > sB["wins"] or (sA["wins"] == sB["wins"] and sA["points"] > sB["points"]):
            agg[dA]["seriesWins"] += 1
            agg[dA]["leaguePoints"] += 2
            agg[dB]["seriesLosses"] += 1
        elif sB["wins"] > sA["wins"] or (sB["wins"] == sA["wins"] and sB["points"] > sA["points"]):
            agg[dB]["seriesWins"] += 1
            agg[dB]["leaguePoints"] += 2
            agg[dA]["seriesLosses"] += 1
        else:
            agg[dA]["seriesTies"] += 1
            agg[dB]["seriesTies"] += 1
            agg[dA]["leaguePoints"] += 1
            agg[dB]["leaguePoints"] += 1

    pbar.close()
    total_elapsed_sec = time.perf_counter() - t_league_start

    elos = compute_calibrated_elo_ratings(
        depths, h2h_wins, h2h_games, center_elo=2000.0
    )

    standings: List[Dict[str, object]] = []
    for d in depths:
        st = agg[d]
        g = max(1, st["games"])
        wr_pct = round((st["wins"] / g) * 100.0, 1)
        net_diff = st["pointsScored"] - st["pointsConceded"]
        ppg = round(net_diff / g, 3)
        raw_ppg = round(st["pointsScored"] / g, 3)

        standings.append(
            {
                "rank": 0,
                "depth": d,
                "name": DEPTH_SHORT_NAME.get(d, f"Depth {d}"),
                "labelHe": DEPTH_LABELS_HE.get(d, f"Depth {d}"),
                "games": st["games"],
                "seriesWins": st["seriesWins"],
                "seriesLosses": st["seriesLosses"],
                "seriesTies": st["seriesTies"],
                "leaguePoints": st["leaguePoints"],
                "wins": st["wins"],
                "losses": st["losses"],
                "winRatePct": wr_pct,
                "singleWins": st["singleWins"],
                "gammonWins": st["gammonWins"],
                "backgammonWins": st["backgammonWins"],
                "singleLosses": st["singleLosses"],
                "gammonLosses": st["gammonLosses"],
                "backgammonLosses": st["backgammonLosses"],
                "pointsScored": st["pointsScored"],
                "pointsConceded": st["pointsConceded"],
                "netPointDiff": net_diff,
                "ppg": ppg,
                "rawPPG": raw_ppg,
                "avgMoveTimeMs": round(
                    (total_elapsed_sec * 1000.0) / max(1, total_games_all * 55), 3
                ),
                "eloEstimate": elos.get(d, 1800),
            }
        )

    standings.sort(
        key=lambda r: (
            int(r["leaguePoints"]),
            int(r["wins"]),
            int(r["netPointDiff"]),
            int(r["depth"]),
        ),
        reverse=True,
    )
    for idx, row in enumerate(standings, start=1):
        row["rank"] = idx

    format_and_save_league_reports(
        depths=depths,
        standings=standings,
        matchup_results=matchup_results,
        h2h_wins=h2h_wins,
        h2h_pts=h2h_pts,
        games_per_pair=games_per_pair,
        total_elapsed_sec=total_elapsed_sec,
        trained_games=trained_games,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
