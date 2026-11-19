"""
Duplicated-Dice Benchmark Suite for 3-Network Residual Backgammon AI vs. Heuristic Baseline.
Plays N pairs of duplicated games (identical dice sequences with colors swapped)
to eliminate dice variance and measure exact Win %, Gammon %, PPG, and Elo gain.
Supports both Version 2 (3-Network Residual: Contact, Crashed, Race) and Version 1 JSON weights.
"""

from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from backgammon_env import (
    BoardState,
    check_winner,
    classify_game_phase,
    create_initial_board,
    evaluate_heuristic_baseline,
    extract_features_210,
    generate_legal_afterstates,
)


def silu_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


class NumpyResidualSubNet:
    def __init__(self, d: Dict[str, object]) -> None:
        self.in_dim = int(d["inDim"])
        self.res_dim = int(d["resDim"])
        self.bottleneck_dim = int(d["bottleneckDim"])
        self.out_dim = int(d["outDim"])

        self.in_w = np.array(d["inW"], dtype=np.float32).reshape(self.res_dim, self.in_dim)
        self.in_b = np.array(d["inB"], dtype=np.float32)

        raw_blocks = d.get("resBlocks")
        if isinstance(raw_blocks, list) and len(raw_blocks) > 0:
            self.blocks = [
                (
                    np.array(rb["w1"], dtype=np.float32).reshape(self.res_dim, self.res_dim),
                    np.array(rb["b1"], dtype=np.float32),
                    np.array(rb["w2"], dtype=np.float32).reshape(self.res_dim, self.res_dim),
                    np.array(rb["b2"], dtype=np.float32),
                )
                for rb in raw_blocks
                if isinstance(rb, dict)
            ]
        else:
            rb = d["resBlock"]
            assert isinstance(rb, dict)
            self.blocks = [
                (
                    np.array(rb["w1"], dtype=np.float32).reshape(self.res_dim, self.res_dim),
                    np.array(rb["b1"], dtype=np.float32),
                    np.array(rb["w2"], dtype=np.float32).reshape(self.res_dim, self.res_dim),
                    np.array(rb["b2"], dtype=np.float32),
                )
            ]

        self.neck_w = np.array(d["neckW"], dtype=np.float32).reshape(self.bottleneck_dim, self.res_dim)
        self.neck_b = np.array(d["neckB"], dtype=np.float32)

        self.out_w = np.array(d["outW"], dtype=np.float32).reshape(self.out_dim, self.bottleneck_dim)
        self.out_b = np.array(d["outB"], dtype=np.float32)

    def evaluate_equity(self, feat: np.ndarray) -> float:
        h = silu_np(self.in_w @ feat + self.in_b)
        for w1, b1, w2, b2 in self.blocks:
            r1 = silu_np(w1 @ h + b1)
            r2 = silu_np(w2 @ r1 + b2)
            h = h + r2
        neck = silu_np(self.neck_w @ h + self.neck_b)
        logits = self.out_w @ neck + self.out_b
        if self.out_dim == 6:
            logits6 = logits.astype(np.float64)
        else:
            logits6 = np.array(
                [logits[0], logits[1], logits[2], 0.0, logits[3], logits[4]],
                dtype=np.float64,
            )
        logits6 -= np.max(logits6)
        probs = np.exp(logits6)
        probs /= np.sum(probs)
        weights = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0], dtype=np.float64)
        return float(np.dot(probs, weights))


class NumpyLegacySubNet:
    def __init__(self, d: Dict[str, object]) -> None:
        self.in_dim = int(d["inDim"])
        self.h1_dim = int(d["h1Dim"])
        self.h2_dim = int(d["h2Dim"])
        self.out_dim = int(d["outDim"])
        self.w1 = np.array(d["w1"], dtype=np.float32).reshape(self.h1_dim, self.in_dim)
        self.b1 = np.array(d["b1"], dtype=np.float32)
        self.w2 = np.array(d["w2"], dtype=np.float32).reshape(self.h2_dim, self.h1_dim)
        self.b2 = np.array(d["b2"], dtype=np.float32)
        self.w3 = np.array(d["w3"], dtype=np.float32).reshape(self.out_dim, self.h2_dim)
        self.b3 = np.array(d["b3"], dtype=np.float32)

    def evaluate_equity(self, feat: np.ndarray) -> float:
        h1 = silu_np(self.w1 @ feat + self.b1)
        h2 = silu_np(self.w2 @ h1 + self.b2)
        logits5 = self.w3 @ h2 + self.b3
        logits6 = np.array(
            [logits5[0], logits5[1], logits5[2], 0.0, logits5[3], logits5[4]],
            dtype=np.float64,
        )
        logits6 -= np.max(logits6)
        probs = np.exp(logits6)
        probs /= np.sum(probs)
        weights = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0], dtype=np.float64)
        return float(np.dot(probs, weights))


class TripleNumpyNN:
    def __init__(self, weights_path: Path) -> None:
        with open(weights_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.version = int(data.get("version", 1))
        if self.version >= 2:
            self.contact_net = NumpyResidualSubNet(data["contact"])
            self.crashed_net = NumpyResidualSubNet(data["crashed"])
            self.race_net = NumpyResidualSubNet(data["race"])
        else:
            self.contact_net = NumpyLegacySubNet(data["contactNet"])
            self.crashed_net = self.contact_net
            self.race_net = NumpyLegacySubNet(data["raceNet"])

    def evaluate_white_equity(self, state: BoardState) -> float:
        phase = classify_game_phase(state)
        if phase == "race":
            net = self.race_net
        elif phase == "crashed":
            net = self.crashed_net
        else:
            net = self.contact_net
        w_eq = net.evaluate_equity(extract_features_210(state, "white"))
        b_eq = net.evaluate_equity(extract_features_210(state, "black"))
        return 0.5 * (w_eq - b_eq)


def select_move_nn(nn_model: TripleNumpyNN, afterstates: List[BoardState], player: str) -> BoardState:
    if len(afterstates) == 1:
        return afterstates[0]
    scores = [nn_model.evaluate_white_equity(s) for s in afterstates]
    best_idx = int(np.argmax(scores)) if player == "white" else int(np.argmin(scores))
    return afterstates[best_idx]


def select_move_heuristic(afterstates: List[BoardState], player: str) -> BoardState:
    if len(afterstates) == 1:
        return afterstates[0]
    scores = [evaluate_heuristic_baseline(s)[0] for s in afterstates]
    best_idx = int(np.argmax(scores)) if player == "white" else int(np.argmin(scores))
    return afterstates[best_idx]


def play_single_game(
    nn_model: TripleNumpyNN,
    nn_color: str,
    dice_stream: List[Tuple[int, int]],
) -> Tuple[str, int]:
    state = create_initial_board(starting_turn="white")
    for _ply, dice in enumerate(dice_stream):
        win = check_winner(state)
        if win is not None:
            return win
        afterstates = generate_legal_afterstates(state, dice)
        if not afterstates:
            continue
        if state.current_turn == nn_color:
            state = select_move_nn(nn_model, afterstates, state.current_turn)
        else:
            state = select_move_heuristic(afterstates, state.current_turn)

    win = check_winner(state)
    if win is not None:
        return win
    eq, _ = evaluate_heuristic_baseline(state)
    return ("white" if eq >= 0 else "black", 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 3-Network Residual Backgammon NN vs Heuristic Baseline")
    parser.add_argument("--games", type=int, default=400, help="Total duplicated games (must be even)")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "public" / "weights" / "backgammon_nn.json"),
        help="Path to exported backgammon_nn.json",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Weights file not found at {weights_path}. Run train_td_gammon.py first.")
        return

    nn_model = TripleNumpyNN(weights_path)
    pairs = max(1, args.games // 2)
    total_games = pairs * 2

    nn_wins = 0
    nn_gammons = 0
    nn_points = 0

    for _ in tqdm(range(pairs), desc=f"Playing duplicated pairs (v{nn_model.version})", unit="pair"):
        dice_stream = [(random.randint(1, 6), random.randint(1, 6)) for _ in range(180)]
        for nn_color in ("white", "black"):
            winner, mult = play_single_game(nn_model, nn_color, dice_stream)
            if winner == nn_color:
                nn_wins += 1
                if mult >= 2:
                    nn_gammons += 1
                nn_points += mult
            else:
                nn_points -= mult

    win_rate = nn_wins / total_games
    gammon_rate = nn_gammons / total_games
    ppg = nn_points / total_games
    clamped_wr = max(0.01, min(0.99, win_rate))
    elo_diff = -400.0 * math.log10(1.0 / clamped_wr - 1.0)

    print("\n=== Duplicated-Dice Benchmark Results ===")
    print(f"Network Schema     : Version {nn_model.version}")
    print(f"Total Games Played : {total_games} ({pairs} duplicated dice pairs)")
    print(f"NN Win Rate        : {win_rate * 100:.1f}% ({nn_wins}/{total_games})")
    print(f"NN Gammon Rate     : {gammon_rate * 100:.1f}%")
    print(f"Points Per Game    : {ppg:+.3f} PPG")
    print(f"Estimated Elo Gain : {elo_diff:+.1f} Elo")


if __name__ == "__main__":
    main()
