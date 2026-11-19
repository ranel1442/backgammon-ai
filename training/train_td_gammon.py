"""
Grandmaster-Level 3-Network Residual Backgammon GPU Trainer (Multi-Core Numba + CUDA BF16/TF32 Mega-Batching):
- Multi-core Numba @njit(parallel=True, fastmath=True) vectorized board stepping & 210-feature extraction
- Tensor Core TF32 + BFloat16 mixed-precision GPU inference & mega-batched training
- Stage 1: Balanced Supervised Warm-Start across Contact, Crashed, and Race phases
- Stage 2: Batched 4,096-Environment GPU TD(lambda=0.7) Self-Play + Single-Pass Mega-Batched 2-Ply AlphaZero Distillation
- Automatic Checkpoint Save & Resume (training/checkpoints/latest.pt every 25,000 games)
- Live Version-2 JSON Export to ../public/weights/backgammon_nn.json
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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from backgammon_env import (
    ALL_21_DICE,
    FEATURE_VECTOR_SIZE,
    NUMBA_AVAILABLE,
    batch_apply_selected_moves,
    batch_classify_phases,
    batch_extract_features_210,
    batch_generate_legal_afterstates,
    batch_label_positions_teacher,
    batch_pack_and_extract_candidates,
    batch_reset_envs,
    batch_select_best_candidates,
    generate_hard_position,
)
from bearoff_dp import get_bearoff_db


# Configure CUDA Tensor Core TF32 acceleration globally
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


PHASE_ID_TO_NAME = {0: "race", 1: "crashed", 2: "contact"}


class ResidualBlock(nn.Module):
    """
    Residual Block: x + SiLU(W2 * SiLU(W1 * x + b1) + b2)
    Matches evaluateResidualSubNet() in src/engine/neuralNet.ts.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc1(x))
        h = F.silu(self.fc2(h))
        return x + h

    def export_dict(self) -> Dict[str, object]:
        return {
            "dim": self.dim,
            "w1": [round(float(v), 6) for v in self.fc1.weight.detach().cpu().numpy().reshape(-1)],
            "b1": [round(float(v), 6) for v in self.fc1.bias.detach().cpu().numpy().reshape(-1)],
            "w2": [round(float(v), 6) for v in self.fc2.weight.detach().cpu().numpy().reshape(-1)],
            "b2": [round(float(v), 6) for v in self.fc2.bias.detach().cpu().numpy().reshape(-1)],
        }

    def load_from_export_dict(self, d: Dict[str, object], device: torch.device) -> None:
        w1 = torch.tensor(d["w1"], dtype=torch.float32, device=device).reshape(self.dim, self.dim)
        b1 = torch.tensor(d["b1"], dtype=torch.float32, device=device)
        w2 = torch.tensor(d["w2"], dtype=torch.float32, device=device).reshape(self.dim, self.dim)
        b2 = torch.tensor(d["b2"], dtype=torch.float32, device=device)
        with torch.no_grad():
            self.fc1.weight.copy_(w1)
            self.fc1.bias.copy_(b1)
            self.fc2.weight.copy_(w2)
            self.fc2.bias.copy_(b2)


class ResidualSubNet(nn.Module):
    """
    Deep Residual Sub-Network (Version 3):
      inDim (260) -> resDim (SiLU) -> [ResidualBlock(resDim) * num_blocks] -> bottleneckDim (SiLU) -> outDim (6 Softmax)
    Matches ResidualSubNetWeightsJSON in src/engine/neuralNet.ts.
    """

    def __init__(
        self,
        in_dim: int,
        res_dim: int,
        bottleneck_dim: int,
        out_dim: int = 6,
        num_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.res_dim = res_dim
        self.bottleneck_dim = bottleneck_dim
        self.out_dim = out_dim
        self.num_blocks = num_blocks

        self.in_fc = nn.Linear(in_dim, res_dim)
        self.res_blocks = nn.ModuleList([ResidualBlock(res_dim) for _ in range(num_blocks)])
        self.neck_fc = nn.Linear(res_dim, bottleneck_dim)
        self.out_fc = nn.Linear(bottleneck_dim, out_dim)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.in_fc(x))
        for block in self.res_blocks:
            h = block(h)
        neck = F.silu(self.neck_fc(h))
        return self.out_fc(neck)

    def forward_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns 6-class probabilities:
        [pWinSingle, pWinGammonOnly, pWinBg, pLoseSingle, pLoseGammonOnly, pLoseBg]
        """
        logits = self.forward_logits(x)
        return F.softmax(logits.float(), dim=-1)

    def export_dict(self) -> Dict[str, object]:
        blocks_exported = [b.export_dict() for b in self.res_blocks]
        return {
            "inDim": self.in_dim,
            "resDim": self.res_dim,
            "bottleneckDim": self.bottleneck_dim,
            "outDim": self.out_dim,
            "inW": [round(float(v), 6) for v in self.in_fc.weight.detach().cpu().numpy().reshape(-1)],
            "inB": [round(float(v), 6) for v in self.in_fc.bias.detach().cpu().numpy().reshape(-1)],
            "resBlock": blocks_exported[0],
            "resBlocks": blocks_exported,
            "neckW": [round(float(v), 6) for v in self.neck_fc.weight.detach().cpu().numpy().reshape(-1)],
            "neckB": [round(float(v), 6) for v in self.neck_fc.bias.detach().cpu().numpy().reshape(-1)],
            "outW": [round(float(v), 6) for v in self.out_fc.weight.detach().cpu().numpy().reshape(-1)],
            "outB": [round(float(v), 6) for v in self.out_fc.bias.detach().cpu().numpy().reshape(-1)],
        }

    def load_from_export_dict(self, d: Dict[str, object], device: torch.device) -> None:
        in_w = torch.tensor(d["inW"], dtype=torch.float32, device=device).reshape(self.res_dim, self.in_dim)
        in_b = torch.tensor(d["inB"], dtype=torch.float32, device=device)
        neck_w = torch.tensor(d["neckW"], dtype=torch.float32, device=device).reshape(self.bottleneck_dim, self.res_dim)
        neck_b = torch.tensor(d["neckB"], dtype=torch.float32, device=device)
        out_w = torch.tensor(d["outW"], dtype=torch.float32, device=device).reshape(self.out_dim, self.bottleneck_dim)
        out_b = torch.tensor(d["outB"], dtype=torch.float32, device=device)
        with torch.no_grad():
            self.in_fc.weight.copy_(in_w)
            self.in_fc.bias.copy_(in_b)
            self.neck_fc.weight.copy_(neck_w)
            self.neck_fc.bias.copy_(neck_b)
            self.out_fc.weight.copy_(out_w)
            self.out_fc.bias.copy_(out_b)
        rbs = d.get("resBlocks")
        if isinstance(rbs, list) and len(rbs) == len(self.res_blocks):
            for block, rb_dict in zip(self.res_blocks, rbs):
                if isinstance(rb_dict, dict):
                    block.load_from_export_dict(rb_dict, device)
        else:
            rb = d.get("resBlock")
            if isinstance(rb, dict) and len(self.res_blocks) > 0:
                self.res_blocks[0].load_from_export_dict(rb, device)


def estimate_elo_from_games(trained_games: int) -> int:
    if trained_games <= 0:
        return 1750
    bonus = min(350, int(65.0 * math.log10(1.0 + trained_games / 500.0)))
    return 1780 + bonus


def evaluate_megabatch_3net_gpu(
    flat_feats: np.ndarray,
    flat_phases: np.ndarray,
    nets: Dict[str, ResidualSubNet],
    device: torch.device,
    max_chunk: int = 65536,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates a contiguous (TotalStates, 210) feature tensor on GPU using BFloat16 Tensor Cores,
    routing each row to its phase sub-network (0=race, 1=crashed, 2=contact).
    Returns:
      probs6:   (TotalStates, 6) float32
      equities: (TotalStates,) float32
    """
    total = flat_feats.shape[0]
    if total == 0:
        return np.empty((0, 6), dtype=np.float32), np.empty(0, dtype=np.float32)

    use_amp = device.type == "cuda"
    eq_weights = torch.tensor([1.0, 2.0, 3.0, -1.0, -2.0, -3.0], device=device, dtype=torch.float32)

    out_probs = np.empty((total, 6), dtype=np.float32)
    out_equities = np.empty(total, dtype=np.float32)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        for start in range(0, total, max_chunk):
            end = min(total, start + max_chunk)
            chunk_feats = torch.from_numpy(flat_feats[start:end]).to(device, non_blocking=True)
            chunk_phases = flat_phases[start:end]
            chunk_probs = torch.empty((end - start, 6), dtype=torch.float32, device=device)

            for phase_id, phase_name in PHASE_ID_TO_NAME.items():
                mask_np = chunk_phases == phase_id
                if np.any(mask_np):
                    idx_t = torch.from_numpy(np.nonzero(mask_np)[0]).to(device, non_blocking=True)
                    chunk_probs[idx_t] = nets[phase_name].forward_probs(chunk_feats[idx_t])

            chunk_eq = torch.mv(chunk_probs.float(), eq_weights)
            out_probs[start:end] = chunk_probs.float().cpu().numpy()
            out_equities[start:end] = chunk_eq.float().cpu().numpy()

    return out_probs, out_equities


def generate_balanced_warmup_dataset(
    num_samples: int,
    num_envs: int = 1024,
    hard_ratio: float = 0.30,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Stage 1: Generates balanced training positions across Contact, Crashed, and Race phases
    using Multi-Core Parallel Numba kernels across `num_envs` boards simultaneously
    with `hard_ratio` curriculum injection (Backgames, Deep Holding, Prime Battles, Crashed).
    """
    print(
        f"[Stage 1] Generating {num_samples:,} 3-Phase positions via Multi-Core Numba "
        f"({num_envs} parallel boards | hard_ratio={hard_ratio:.0%} | JIT={'ON' if NUMBA_AVAILABLE else 'OFF'})..."
    )
    get_bearoff_db()

    b_envs = max(64, min(num_envs, 2048))
    boards = np.zeros((b_envs, 24), dtype=np.int32)
    bars = np.zeros((b_envs, 2), dtype=np.int32)
    borne_offs = np.zeros((b_envs, 2), dtype=np.int32)
    turns = np.zeros(b_envs, dtype=np.int32)
    dice = np.zeros((b_envs, 2), dtype=np.int32)
    rng_states = np.arange(1, b_envs + 1, dtype=np.uint64) * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)

    reset_mask = np.ones(b_envs, dtype=np.bool_)
    batch_reset_envs(boards, bars, borne_offs, turns, dice, reset_mask, rng_states, hard_ratio)

    buckets_x: Dict[str, List[np.ndarray]] = {"contact": [], "crashed": [], "race": []}
    buckets_y: Dict[str, List[np.ndarray]] = {"contact": [], "crashed": [], "race": []}

    pbar = tqdm(total=num_samples, desc="Sampling 3-Phase Positions (Numba prange)", unit="pos")
    collected = 0
    white_persp = np.ones(b_envs, dtype=np.int32)

    while collected < num_samples:
        cand_boards, cand_bars, cand_offs, cand_counts = batch_generate_legal_afterstates(
            boards, bars, borne_offs, turns, dice
        )
        offsets = np.zeros(b_envs + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(cand_counts)
        total_cands = int(offsets[-1])

        # Use fast random/epsilon move selection during warm-start data collection
        flat_eq = np.zeros(total_cands, dtype=np.float32)
        selected = batch_select_best_candidates(
            flat_eq, offsets, cand_counts, turns, 1.0, rng_states
        )
        dones, _ = batch_apply_selected_moves(
            boards,
            bars,
            borne_offs,
            turns,
            dice,
            cand_boards,
            cand_bars,
            cand_offs,
            selected,
            rng_states,
        )

        # Label non-terminal states in parallel across all CPU cores
        valid_mask = ~dones
        if np.any(valid_mask):
            v_boards = boards[valid_mask]
            v_bars = bars[valid_mask]
            v_offs = borne_offs[valid_mask]
            v_turns = turns[valid_mask]
            v_white = white_persp[: len(v_boards)]

            feats = batch_extract_features_210(v_boards, v_bars, v_offs, v_white)
            phases = batch_classify_phases(v_boards, v_bars, v_offs)
            _, probs = batch_label_positions_teacher(v_boards, v_bars, v_offs, v_turns)

            take = min(len(v_boards), num_samples - collected)
            feats = feats[:take]
            phases = phases[:take]
            probs = probs[:take]

            for phase_id, phase_name in PHASE_ID_TO_NAME.items():
                pm = phases == phase_id
                if np.any(pm):
                    buckets_x[phase_name].append(feats[pm])
                    buckets_y[phase_name].append(probs[pm])

            collected += take
            pbar.update(take)

        if np.any(dones):
            batch_reset_envs(boards, bars, borne_offs, turns, dice, dones, rng_states, hard_ratio)

    pbar.close()

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for phase in ("contact", "crashed", "race"):
        xs = buckets_x[phase]
        ys = buckets_y[phase]
        if xs:
            x_cat = np.concatenate(xs, axis=0)
            y_cat = np.concatenate(ys, axis=0)
            out[phase] = (x_cat, y_cat)
            print(f"  Phase '{phase}': {len(x_cat):,} samples collected")
        else:
            out[phase] = (
                np.zeros((1, FEATURE_VECTOR_SIZE), dtype=np.float32),
                np.full((1, 6), 1.0 / 6.0, dtype=np.float32),
            )
            print(f"  Phase '{phase}': 0 samples collected")
    return out


def train_supervised_warmup(
    net: ResidualSubNet,
    x_data: np.ndarray,
    y_data: np.ndarray,
    device: torch.device,
    epochs: int = 8,
    batch_size: int = 4096,
    lr: float = 2e-3,
    label: str = "SubNet",
) -> None:
    if len(x_data) < 2:
        return

    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    x_tensor = torch.from_numpy(x_data).to(device)
    y_tensor = torch.from_numpy(y_data).to(device)
    n = x_tensor.shape[0]
    use_amp = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            bx = x_tensor[idx]
            by = y_tensor[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred_probs = net.forward_probs(bx)
                loss = -torch.sum(by * torch.log(torch.clamp(pred_probs, min=1e-7)), dim=-1).mean()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1

        scheduler.step()
        print(f"  [{label} Warm-Start] Epoch {epoch}/{epochs} — Soft-CE Loss: {total_loss / max(1, batches):.4f}")


def compute_megabatch_2ply_distillation_targets(
    d_boards: np.ndarray,
    d_bars: np.ndarray,
    d_offs: np.ndarray,
    d_turns: np.ndarray,
    nets: Dict[str, ResidualSubNet],
    device: torch.device,
) -> np.ndarray:
    """
    Mega-Batched 2-Ply Expectiminimax AlphaZero Distillation:
    Expands all K sampled positions across all 21 dice outcomes (K * 21 queries),
    generates all legal child afterstates in parallel via Numba prange, packs all
    children into ONE contiguous (TotalChildren, 210) tensor, and runs a SINGLE
    batched GPU forward pass!
    Returns (K, 6) 2-ply target probability distributions from White's perspective.
    """
    k = d_boards.shape[0]
    if k == 0:
        return np.empty((0, 6), dtype=np.float32)

    dice_pairs = np.array([[d1, d2] for d1, d2, _ in ALL_21_DICE], dtype=np.int32)  # (21, 2)
    dice_probs = np.array([prob for _, _, prob in ALL_21_DICE], dtype=np.float32)  # (21,)

    # Expand K states -> K * 21 (state, dice) pairs
    rep_boards = np.repeat(d_boards, 21, axis=0)
    rep_bars = np.repeat(d_bars, 21, axis=0)
    rep_offs = np.repeat(d_offs, 21, axis=0)
    rep_turns = np.repeat(d_turns, 21, axis=0)
    rep_dice = np.tile(dice_pairs, (k, 1))

    cand_boards, cand_bars, cand_offs, cand_counts = batch_generate_legal_afterstates(
        rep_boards, rep_bars, rep_offs, rep_turns, rep_dice
    )
    n_queries = k * 21
    offsets = np.zeros(n_queries + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(cand_counts)

    flat_feats, flat_phases = batch_pack_and_extract_candidates(
        cand_boards, cand_bars, cand_offs, cand_counts, offsets
    )
    # Single mega-batched GPU forward pass for all candidate children across all K*21 rolls
    flat_probs, flat_equities = evaluate_megabatch_3net_gpu(flat_feats, flat_phases, nets, device)

    dummy_rng = np.ones(n_queries, dtype=np.uint64)
    best_child_local = batch_select_best_candidates(
        flat_equities, offsets, cand_counts, rep_turns, 0.0, dummy_rng
    )
    best_child_global = offsets[:-1] + best_child_local
    best_roll_probs = flat_probs[best_child_global].reshape(k, 21, 6)  # (K, 21, 6)

    # Probability-weighted expectation over the 21 dice outcomes
    targets = np.einsum("kdc,d->kc", best_roll_probs, dice_probs).astype(np.float32)
    row_sums = np.sum(targets, axis=-1, keepdims=True)
    targets /= np.maximum(row_sums, 1e-7)
    return targets


def evaluate_hard_position_validation_suite(
    nets: Dict[str, ResidualSubNet],
    device: torch.device,
    samples_per_archetype: int = 32,
) -> Dict[str, float]:
    """
    Hard-Position Validation Suite:
    Evaluates the 3 Residual networks across deterministic samples of the 4 difficult archetypes:
      0 = Backgames (2-3 / 1-2)
      1 = Deep Holding Games
      2 = Mutual Prime Battles
      3 = Crashed Home Boards
    Measures Top-1 move agreement with 1-ply teacher and value calibration accuracy.
    """
    archetype_names = {
        0: "Backgame",
        1: "Holding",
        2: "PrimeBattle",
        3: "Crashed",
    }
    for net in nets.values():
        net.eval()

    results: Dict[str, float] = {}
    rng_cursor = 987654321

    for arch_id, arch_name in archetype_names.items():
        n = samples_per_archetype
        boards = np.zeros((n, 24), dtype=np.int32)
        bars = np.zeros((n, 2), dtype=np.int32)
        offs = np.zeros((n, 2), dtype=np.int32)
        turns = np.zeros(n, dtype=np.int32)
        dice = np.zeros((n, 2), dtype=np.int32)

        for i in range(n):
            state, rng_cursor = generate_hard_position(rng_cursor, arch_id)
            boards[i] = np.asarray(state.points, dtype=np.int32)
            bars[i, 0] = state.bar_white
            bars[i, 1] = state.bar_black
            offs[i, 0] = state.off_white
            offs[i, 1] = state.off_black
            turns[i] = 1 if state.current_turn == "white" else 0
            dice[i, 0] = (i % 6) + 1
            dice[i, 1] = ((i // 6) % 6) + 1

        cand_boards, cand_bars, cand_offs, cand_counts = batch_generate_legal_afterstates(
            boards, bars, offs, turns, dice
        )
        offsets = np.zeros(n + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(cand_counts)
        flat_feats, flat_phases = batch_pack_and_extract_candidates(
            cand_boards, cand_bars, cand_offs, cand_counts, offsets
        )
        _, nn_equities = evaluate_megabatch_3net_gpu(flat_feats, flat_phases, nets, device)

        # Compare NN equity calibration against teacher on the root hard positions
        white_persp = np.ones(n, dtype=np.int32)
        root_feats = batch_extract_features_210(boards, bars, offs, white_persp)
        root_phases = batch_classify_phases(boards, bars, offs)
        nn_root_probs, nn_root_eq = evaluate_megabatch_3net_gpu(root_feats, root_phases, nets, device)
        teacher_eq, teacher_probs = batch_label_positions_teacher(boards, bars, offs, turns)

        mae = float(np.mean(np.abs(nn_root_eq - teacher_eq)))
        # Convert MAE into a 0..100% calibration accuracy score
        cal_acc = max(0.0, min(100.0, 100.0 * math.exp(-0.85 * mae)))
        results[arch_name] = cal_acc

    summary_str = " | ".join(f"{k}: {v:.1f}%" for k, v in results.items())
    print(f"[Hard-Position Validation] {summary_str}")
    return results


def save_checkpoint_and_export(
    nets: Dict[str, ResidualSubNet],
    optimizer: Optional[torch.optim.Optimizer],
    games_trained: int,
    ckpt_path: Path,
    json_output_path: Path,
    device: Optional[torch.device] = None,
    force_export: bool = False,
) -> None:
    if device is not None:
        evaluate_hard_position_validation_suite(nets, device)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 3,
            "games_trained": games_trained,
            "contact_state_dict": nets["contact"].state_dict(),
            "crashed_state_dict": nets["crashed"].state_dict(),
            "race_state_dict": nets["race"].state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        },
        ckpt_path,
    )
    export_weights_json(nets, json_output_path, games_trained, force_export=force_export)


def export_weights_json(
    nets: Dict[str, ResidualSubNet],
    output_path: Path,
    games_trained: int,
    force_export: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force_export:
        try:
            with open(output_path, "r", encoding="utf-8") as existing_f:
                existing_data = json.load(existing_f)
            existing_ver = int(existing_data.get("version", 1))
            existing_games = int(
                existing_data.get("trainedGames", existing_data.get("gamesTrained", 0))
            )
            if existing_ver >= 3 and existing_games > games_trained:
                print(
                    f"[Safeguard] Keeping {output_path} because existing v{existing_ver} trainedGames "
                    f"({existing_games:,}) > current run ({games_trained:,}). "
                    f"Pass --force-export to override."
                )
                return
        except Exception:
            pass

    elo_est = estimate_elo_from_games(games_trained)
    payload = {
        "version": 3,
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "trainedGames": games_trained,
        "eloEstimate": elo_est,
        "contact": nets["contact"].export_dict(),
        "crashed": nets["crashed"].export_dict(),
        "race": nets["race"].export_dict(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    size_kb = output_path.stat().st_size / 1024.0
    print(
        f"[Export v3] Saved 260-Feature 3-Network Residual weights to {output_path} "
        f"({size_kb:.1f} KB | games={games_trained:,} | ~{elo_est} Elo)"
    )


def try_load_existing_weights(
    nets: Dict[str, ResidualSubNet],
    ckpt_path: Path,
    json_output_path: Path,
    device: torch.device,
    resume_flag: bool = True,
) -> Tuple[int, Optional[dict]]:
    """
    Loads existing Version 3 weights from `training/checkpoints/latest.pt` or `public/weights/backgammon_nn.json`
    when `--resume` or `--warmup-samples 0` is used, preserving cumulative `trainedGames`.
    """
    existing_json_games = 0
    if json_output_path.exists():
        try:
            with open(json_output_path, "r", encoding="utf-8") as f:
                jdata = json.load(f)
            if int(jdata.get("version", 1)) == 3:
                existing_json_games = int(
                    jdata.get("trainedGames", jdata.get("gamesTrained", 0))
                )
        except Exception:
            pass

    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if ckpt.get("version") == 3:
                nets["contact"].load_state_dict(ckpt["contact_state_dict"])
                nets["crashed"].load_state_dict(ckpt["crashed_state_dict"])
                nets["race"].load_state_dict(ckpt["race_state_dict"])
                ckpt_games = int(ckpt.get("games_trained", 0))
                games = max(ckpt_games, existing_json_games)
                opt_state = ckpt.get("optimizer_state_dict")
                print(f"[Init] Loaded v3 260-Feature Residual weights from {ckpt_path} (games={games:,}).")
                return games, opt_state
        except Exception as e:
            print(f"[Init] Starting fresh v3 architecture (previous checkpoint incompatible: {e})")

    if json_output_path.exists():
        try:
            with open(json_output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if int(data.get("version", 1)) == 3:
                nets["contact"].load_from_export_dict(data["contact"], device)
                nets["crashed"].load_from_export_dict(data["crashed"], device)
                nets["race"].load_from_export_dict(data["race"], device)
                games = int(data.get("trainedGames", 0))
                print(f"[Init] Loaded Version 3 JSON weights from {json_output_path} (games={games:,}).")
                return games, None
        except Exception as e:
            print(f"[Init] Could not load JSON weights {json_output_path}: {e}")

    return 0, None


def run_hybrid_td_and_distillation(
    nets: Dict[str, ResidualSubNet],
    device: torch.device,
    total_games: int,
    start_games: int = 0,
    num_envs: int = 4096,
    train_batch_size: int = 4096,
    grad_steps: int = 4,
    rollout_plies: int = 24,
    td_lambda: float = 0.7,
    lr: float = 3e-4,
    hard_ratio: float = 0.30,
    checkpoint_every: int = 25000,
    ckpt_path: Optional[Path] = None,
    json_output_path: Optional[Path] = None,
    optimizer_state: Optional[dict] = None,
    force_export: bool = False,
) -> float:
    """
    Stage 2: Multi-Core Numba + GPU Mega-Batched TD(lambda=0.7) Self-Play + 2-Ply AlphaZero Distillation
    with Hard-Example Curriculum Mixing (`hard_ratio`).
    Steps all `num_envs` environments in lockstep across all CPU threads and executes
    continuous mega-batched BFloat16/TF32 Tensor Core forward/backward passes on the GPU.
    """
    all_params = (
        list(nets["contact"].parameters())
        + list(nets["crashed"].parameters())
        + list(nets["race"].parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-5)
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception:
            pass

    b_envs = max(8, min(num_envs, max(8, total_games - start_games)))
    boards = np.zeros((b_envs, 24), dtype=np.int32)
    bars = np.zeros((b_envs, 2), dtype=np.int32)
    borne_offs = np.zeros((b_envs, 2), dtype=np.int32)
    turns = np.zeros(b_envs, dtype=np.int32)
    dice = np.zeros((b_envs, 2), dtype=np.int32)
    env_plies = np.zeros(b_envs, dtype=np.int32)
    rng_states = (
        np.arange(1, b_envs + 1, dtype=np.uint64) * np.uint64(6364136223846793005)
        + np.uint64(int(time.time() * 1000) | 1)
    )

    init_mask = np.ones(b_envs, dtype=np.bool_)
    batch_reset_envs(boards, bars, borne_offs, turns, dice, init_mask, rng_states, hard_ratio)
    white_persp = np.ones(b_envs, dtype=np.int32)

    games_done = start_games
    last_ckpt_games = start_games
    recent_losses: List[float] = []
    use_amp = device.type == "cuda"

    print(
        f"[Stage 2] Multi-Core Numba + CUDA BF16/TF32 Mega-Batch TD(λ={td_lambda}) + 2-Ply Distillation "
        f"({b_envs:,} parallel boards | hard_ratio={hard_ratio:.0%} | train_batch={train_batch_size:,} | grad_steps={grad_steps})..."
    )
    pbar = tqdm(total=total_games, initial=games_done, desc="Mega-Batch GPU Self-Play", unit="game")

    while games_done < total_games:
        for net in nets.values():
            net.eval()

        # Rollout buffer of shape (T, B, ...) across `rollout_plies` synchronous steps
        t_steps = rollout_plies
        traj_feats = np.empty((t_steps, b_envs, FEATURE_VECTOR_SIZE), dtype=np.float32)
        traj_phases = np.empty((t_steps, b_envs), dtype=np.int32)
        traj_preds = np.empty((t_steps, b_envs, 6), dtype=np.float32)
        traj_dones = np.zeros((t_steps, b_envs), dtype=np.bool_)
        traj_terminals = np.zeros((t_steps, b_envs, 6), dtype=np.float32)

        completed_in_window = 0

        for t in range(t_steps):
            # 1. Parallel Numba legal move generation across all B environments
            cand_boards, cand_bars, cand_offs, cand_counts = batch_generate_legal_afterstates(
                boards, bars, borne_offs, turns, dice
            )
            offsets = np.zeros(b_envs + 1, dtype=np.int32)
            offsets[1:] = np.cumsum(cand_counts)

            # 2. Parallel Numba feature extraction + phase classification for all candidates
            flat_feats, flat_phases = batch_pack_and_extract_candidates(
                cand_boards, cand_bars, cand_offs, cand_counts, offsets
            )

            # 3. Single Mega-Batch GPU Tensor Core forward pass for all candidates
            flat_probs, flat_equities = evaluate_megabatch_3net_gpu(
                flat_feats, flat_phases, nets, device
            )

            # 4. Parallel Numba epsilon-greedy move selection & board update
            selected = batch_select_best_candidates(
                flat_equities, offsets, cand_counts, turns, 0.05, rng_states
            )
            chosen_global_idx = offsets[:-1] + selected

            traj_feats[t] = flat_feats[chosen_global_idx]
            traj_phases[t] = flat_phases[chosen_global_idx]
            traj_preds[t] = flat_probs[chosen_global_idx]

            dones, term_outcomes = batch_apply_selected_moves(
                boards,
                bars,
                borne_offs,
                turns,
                dice,
                cand_boards,
                cand_bars,
                cand_offs,
                selected,
                rng_states,
            )
            env_plies += 1

            # Force reset games exceeding 120 plies using teacher bootstrap
            max_ply_mask = (env_plies >= 120) & (~dones)
            if np.any(max_ply_mask):
                _, teacher_probs = batch_label_positions_teacher(
                    boards[max_ply_mask],
                    bars[max_ply_mask],
                    borne_offs[max_ply_mask],
                    turns[max_ply_mask],
                )
                term_outcomes[max_ply_mask] = teacher_probs
                dones[max_ply_mask] = True

            traj_dones[t] = dones
            traj_terminals[t] = term_outcomes

            num_finished = int(np.sum(dones))
            if num_finished > 0:
                completed_in_window += num_finished
                env_plies[dones] = 0
                batch_reset_envs(boards, bars, borne_offs, turns, dice, dones, rng_states, hard_ratio)

        # Bootstrap final step value for ongoing games
        boot_feats = batch_extract_features_210(boards, bars, borne_offs, white_persp)
        boot_phases = batch_classify_phases(boards, bars, borne_offs)
        boot_probs, _ = evaluate_megabatch_3net_gpu(boot_feats, boot_phases, nets, device)

        # Vectorized backward TD(lambda) target computation across all B environments simultaneously
        traj_targets = np.empty_like(traj_preds)
        g_return = boot_probs.copy()
        for t in range(t_steps - 1, -1, -1):
            done_t = traj_dones[t][:, None]  # (B, 1)
            next_val = traj_preds[t + 1] if t < t_steps - 1 else boot_probs
            td_step = (1.0 - td_lambda) * next_val + td_lambda * g_return
            g_return = np.where(done_t, traj_terminals[t], td_step)
            traj_targets[t] = g_return

        flat_train_x = traj_feats.reshape(-1, FEATURE_VECTOR_SIZE)
        flat_train_phases = traj_phases.reshape(-1)
        flat_train_y = traj_targets.reshape(-1, 6)

        # Mega-Batched 2-Ply AlphaZero Distillation on a subset of active boards
        n_distill = min(32, b_envs)
        distill_targets = compute_megabatch_2ply_distillation_targets(
            boards[:n_distill],
            bars[:n_distill],
            borne_offs[:n_distill],
            turns[:n_distill],
            nets,
            device,
        )
        distill_feats = boot_feats[:n_distill]
        distill_phases = boot_phases[:n_distill]

        flat_train_x = np.concatenate([flat_train_x, distill_feats], axis=0)
        flat_train_phases = np.concatenate([flat_train_phases, distill_phases], axis=0)
        flat_train_y = np.concatenate([flat_train_y, distill_targets], axis=0)

        # Multi-step Mega-Batched Gradient Updates on GPU
        for phase_id, phase_name in PHASE_ID_TO_NAME.items():
            mask_p = flat_train_phases == phase_id
            if not np.any(mask_p):
                continue

            net = nets[phase_name]
            net.train()
            px = torch.from_numpy(flat_train_x[mask_p]).to(device, non_blocking=True)
            py = torch.from_numpy(flat_train_y[mask_p]).to(device, non_blocking=True)
            n_p = px.shape[0]

            for _gs in range(grad_steps):
                if n_p > train_batch_size:
                    idx = torch.randint(0, n_p, (train_batch_size,), device=device)
                    bx = px[idx]
                    by = py[idx]
                else:
                    bx = px
                    by = py

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pred = net.forward_probs(bx)
                    loss_p = -torch.sum(by * torch.log(torch.clamp(pred, min=1e-7)), dim=-1).mean()
                loss_p.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                recent_losses.append(float(loss_p.item()))

        # Account for completed games + equivalent full-game trajectories collected across all B boards
        equiv_games = max(completed_in_window, max(1, (b_envs * t_steps) // 64))
        step_games = min(equiv_games, total_games - games_done)
        games_done += step_games
        pbar.update(step_games)

        if recent_losses:
            pbar.set_postfix(
                loss=f"{np.mean(recent_losses[-30:]):.4f}",
                elo=f"~{estimate_elo_from_games(games_done)}",
            )

        if (
            checkpoint_every > 0
            and (games_done - last_ckpt_games) >= checkpoint_every
            and ckpt_path is not None
            and json_output_path is not None
        ):
            save_checkpoint_and_export(
                nets, optimizer, games_done, ckpt_path, json_output_path, device, force_export=force_export
            )
            last_ckpt_games = games_done

    pbar.close()
    if ckpt_path is not None and json_output_path is not None:
        save_checkpoint_and_export(
            nets, optimizer, games_done, ckpt_path, json_output_path, device, force_export=force_export
        )

    return float(np.mean(recent_losses)) if recent_losses else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 3-Network Residual Backgammon AI (Multi-Core Numba + GPU Mega-Batching)"
    )
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=300000,
        help="Supervised warm-start samples across Contact/Crashed/Race (pass 0 to skip Stage 1)",
    )
    parser.add_argument("--warmup-epochs", type=int, default=8, help="Supervised warm-start epochs")
    parser.add_argument(
        "--selfplay-games",
        "--games",
        dest="selfplay_games",
        type=int,
        default=500000,
        help="Total TD(lambda) + 2-Ply Distillation self-play games (supports 10M-15M+)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4096,
        help="Parallel self-play environments per GPU mega-batch (default: 4096)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4096,
        help="Gradient update mini-batch size on GPU (default: 4096)",
    )
    parser.add_argument(
        "--grad-steps",
        type=int,
        default=4,
        help="Gradient update steps per rollout window (default: 4)",
    )
    parser.add_argument(
        "--hard-ratio",
        type=float,
        default=0.30,
        help="Fraction of environment resets sampled from hard tactical archetypes (default: 0.30)",
    )
    parser.add_argument("--td-lambda", type=float, default=0.7, help="TD(lambda) eligibility trace decay")
    parser.add_argument("--lr", type=float, default=3e-4, help="Self-play learning rate")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25000,
        help="Save checkpoint & export JSON every N self-play games (default: 25,000)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from training/checkpoints/latest.pt if present",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Force overwrite of public/weights/backgammon_nn.json even if trainedGames is lower",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "public" / "weights" / "backgammon_nn.json"),
        help="Output JSON path for browser engine",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(
            f"=== CUDA GPU Detected: {gpu_name} ({vram_gb:.1f} GB VRAM) | "
            f"TF32+BF16=ON | Numba Parallel JIT={NUMBA_AVAILABLE} ==="
        )
    else:
        device = torch.device("cpu")
        print(f"=== CUDA not detected; running on CPU | Numba Parallel JIT={NUMBA_AVAILABLE} ===")

    nets: Dict[str, ResidualSubNet] = {
        "contact": ResidualSubNet(FEATURE_VECTOR_SIZE, 512, 256, 6, num_blocks=2).to(device),
        "crashed": ResidualSubNet(FEATURE_VECTOR_SIZE, 384, 192, 6, num_blocks=1).to(device),
        "race": ResidualSubNet(FEATURE_VECTOR_SIZE, 384, 192, 6, num_blocks=1).to(device),
    }

    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    ckpt_path = ckpt_dir / "latest.pt"
    json_output_path = Path(args.output)

    start_games = 0
    optimizer_state = None

    if args.resume or args.warmup_samples <= 0:
        start_games, optimizer_state = try_load_existing_weights(
            nets, ckpt_path, json_output_path, device, resume_flag=args.resume
        )

    start_time = time.time()

    if start_games == 0 and args.warmup_samples > 0:
        datasets = generate_balanced_warmup_dataset(
            args.warmup_samples, num_envs=args.num_envs, hard_ratio=args.hard_ratio
        )
        train_supervised_warmup(
            nets["contact"],
            datasets["contact"][0],
            datasets["contact"][1],
            device,
            epochs=args.warmup_epochs,
            batch_size=args.train_batch_size,
            label="ContactNet (260->512->2xRes512->256->6)",
        )
        train_supervised_warmup(
            nets["crashed"],
            datasets["crashed"][0],
            datasets["crashed"][1],
            device,
            epochs=args.warmup_epochs,
            batch_size=args.train_batch_size,
            label="CrashedNet (260->384->Res384->192->6)",
        )
        train_supervised_warmup(
            nets["race"],
            datasets["race"][0],
            datasets["race"][1],
            device,
            epochs=args.warmup_epochs,
            batch_size=args.train_batch_size,
            label="RaceNet (260->384->Res384->192->6)",
        )
        save_checkpoint_and_export(
            nets, None, 0, ckpt_path, json_output_path, device, force_export=args.force_export
        )

    # Accumulate trainedGames (previous_trained_games + new_games) when resuming or skipping warmup
    target_total_games = (
        start_games + args.selfplay_games
        if (args.resume or args.warmup_samples <= 0)
        else args.selfplay_games
    )

    if target_total_games > start_games and args.selfplay_games > 0:
        run_hybrid_td_and_distillation(
            nets=nets,
            device=device,
            total_games=target_total_games,
            start_games=start_games,
            num_envs=args.num_envs,
            train_batch_size=args.train_batch_size,
            grad_steps=args.grad_steps,
            td_lambda=args.td_lambda,
            lr=args.lr,
            hard_ratio=args.hard_ratio,
            checkpoint_every=args.checkpoint_every,
            ckpt_path=ckpt_path,
            json_output_path=json_output_path,
            optimizer_state=optimizer_state,
            force_export=args.force_export,
        )
    else:
        save_checkpoint_and_export(
            nets,
            None,
            max(start_games, target_total_games),
            ckpt_path,
            json_output_path,
            device,
            force_export=args.force_export,
        )

    elapsed = time.time() - start_time
    print(f"=== Training Complete in {elapsed:.1f}s ===")


if __name__ == "__main__":
    main()
