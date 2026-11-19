# Grandmaster 3-Network Residual GPU Training Suite (RTX 4070 SUPER / CUDA + Numba JIT)

This folder contains the local PyTorch GPU training and benchmarking suite for the **3-Network Residual Backgammon AI (`Contact`, `Crashed`, `Race`)** and the **Exact 54,264-State Bear-Off Tablebase**.

## Architecture

1. **210-Feature Canonical Board Representation (`backgammon_env.py` & `src/engine/neuralNet.ts`)**:
   - `192` point features (`24 points × 2 players × 4 thermometer/excess channels`), canonically ordered from the active player's 24-point down to their 1-point.
   - `18` global features: Bar counts, Borne-off counts, raw Pip counts, Pip difference, Effective Pip Counts (EPC with wastage), Longest consecutive primes, Home-board points made, Defensive anchors, Direct blot hit shots (`0..36`), and Pure-Race indicator.
   - Accelerated in Python via **Numba `@njit`** (`_extract_features_210_jit`, `_classify_phase_jit`).

2. **3-Network Residual Architecture (`version: 2`)**:
   - Deterministic 3-way game-phase classification (`classifyGamePhase` / `classify_game_phase`):
     - `race`: No checkers on the bar AND White's highest occupied point index < Black's lowest occupied point index.
     - `crashed`: Contact position where either player has `<= 5` active checkers remaining (`15 - borneOff <= 5`).
     - `contact`: All other contact positions.
   - **Sub-Network Topologies**:
     - `contact`: `210 -> 384 (SiLU) -> ResidualBlock(384) -> 192 (SiLU) -> 6 (Softmax)`
     - `crashed`: `210 -> 256 (SiLU) -> ResidualBlock(256) -> 128 (SiLU) -> 6 (Softmax)`
     - `race`: `210 -> 256 (SiLU) -> ResidualBlock(256) -> 128 (SiLU) -> 6 (Softmax)`
   - **Perspective-Symmetric Evaluation**: Averages White's prediction with the negation of Black's prediction so mirror symmetry holds with zero bias.

3. **Exact 54,264-State One-Sided Bear-Off Tablebase (`bearoff_dp.py` & `src/engine/bearoffTablebase.ts`)**:
   - Exact Dynamic Programming over all $\binom{15+6}{6} = 54,264$ home-board checker distributions across up to 25 turns.

---

## Quick Start on Windows (NVIDIA RTX 4070 SUPER)

### 1. Create a Python Virtual Environment & Install PyTorch with CUDA + Numba

```powershell
cd training
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Run Grandmaster Training (Stage 1 Balanced Warm-Start + Stage 2 1,024-Env TD(λ) + 2-Ply AlphaZero Distillation)

Default high-strength training run (`300,000` warm-start positions + `500,000` self-play games across `1,024` parallel boards, saving checkpoints every `25,000` games):

```powershell
python train_td_gammon.py
```

Extended 2,000,000-game Grandmaster run with automatic checkpoint resume (`training/checkpoints/latest.pt`):

```powershell
python train_td_gammon.py --selfplay-games 2000000 --num-envs 1024 --resume
```

Every `25,000` games (and on completion), the trainer automatically saves:
- `training/checkpoints/latest.pt` (PyTorch state dicts + optimizer state)
- `../public/weights/backgammon_nn.json` (`version: 2` weights with `trainedGames` and `eloEstimate`)

### 3. Run the 400-Game Duplicated-Dice Benchmark

```powershell
python benchmark.py --games 400
```
