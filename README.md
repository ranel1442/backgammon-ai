# 🎲 Backgammon AI Master — Deep RL Engine, Real-Time Analysis & Vision Scanner

![React](https://img.shields.io/badge/React-TypeScript-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-PWA-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.4-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Numba](https://img.shields.io/badge/Numba-JIT%20Parallel-00A3E0?style=for-the-badge)
![Gemini Vision](https://img.shields.io/badge/Gemini%20Vision-Board%20Scanner-8E75B2?style=for-the-badge)

An offline-first **Progressive Web App (PWA)** for playing, studying, and analyzing Backgammon at master strength—inspired by **Stockfish** and **Chess.com Game Review**. Under the hood, the engine combines a **3-Network Residual Neural Network trained over 15,000,512 self-play games on GPU**, a **54,264-state Exact Bear-Off Tablebase**, a **beam-pruned 5-ply Expectiminimax search engine** running inside a Web Worker, and a **Gemini Vision AI board scanner**.

---

## ✨ Key Features

* **📊 Live Stockfish-Style Evaluation Bar (`EvalBar`):** Real-time Cubeless Equity (`-3.00` to `+3.00`), exact win/gammon/backgammon probability breakdowns, and Effective Pip Count (`EPC`).
* **🎯 Numbered Tactical Arrows & Grandmaster Explanations:**
  * Displays the engine's top 3 candidate lines with numbered SVG move arrows (`1, 2, 3, 4`) directly on the board.
  * **Exact 36-Roll Combinatorics:** Enumerates all 36 opponent dice outcomes on the next turn to report exact blot-hit exposure, bar-entry blockage ("dancing") odds, and trapped-checker escape probabilities.
  * **Head-to-Head Move Deltas:** Quantitatively explains why the #1 recommended move outperforms the runner-up in Equity, gammon chances, and safety.
* **🎲 Monte Carlo CRN Rollouts (1,000 Games):** Runs 1,000 full-game rollouts per candidate inside a dedicated Web Worker using Common Random Numbers (CRN) variance reduction, displaying `Equity ± 95% CI`.
* **📈 Full Game Review & Interactive "What-If" Sandbox Mode:**
  * Calculates player `Accuracy %` (excluding forced 1-move turns), average equity loss, and per-turn move classifications (`Best`, `Good`, `Inaccuracy`, `Mistake`, `Blunder`).
  * **Interactive Momentum Chart (`EvalGraph`):** Click any turn on the timeline to inspect that exact position, or step through critical errors via **Next Mistake**.
  * **What-If Sandbox:** Branch off any historical turn to test alternative plays and dice rolls live without mutating match history, plus full `.json` match export/import.
* **📷 AI Vision Board Scanner (`Gemini Vision`):**
  * Supports digital app screenshots and real-life physical board photos via drag-and-drop, file upload, or `Ctrl+V` clipboard paste.
  * Client-side preprocessing (auto-resize & contrast normalization) paired with a 3-model automatic fallback cascade (`gemini-2.5-flash` -> `gemini-2.5-flash-lite` -> `gemini-2.0-flash`).
  * **Interactive Calibration Studio:** One-click `180°` board rotation, color swap, manual click-to-edit brush, turn/dice selection, and instant tactical analysis.
* **🎮 Gameplay UX & Offline PWA:**
  * Multi-step combined dice moves (e.g., `3+4 = 7` and chained doubles) with strict Backgammon legality enforcement and undo support.
  * Animated opponent turn playback with one-click turn replay.
  * Installable Hebrew RTL PWA with Service Worker caching—neural weights, tablebase, and all 5 AI depths execute **100% locally in-browser**.

---

## 🧠 Under the Hood: AI Architecture & Training Pipeline

Unlike deterministic board games such as Chess, Backgammon is stochastic—every ply branches across **21 weighted dice outcomes**. To achieve strong, well-calibrated play across all phases of the game, the engine integrates **4 evaluation and search layers**:

### 1. Exact Bear-Off Tablebase (54,264 States — 100% Perfect Play)
Once all 15 checkers enter a player's home board (6 points), there are exactly `C(21, 6) = 54,264` possible configurations.
Using Dynamic Programming and bidirectional combinatorial indexing (`src/engine/bearoffTablebase.ts` & `training/bearoff_dp.py`), the engine precomputes exact turn-to-bear-off probability distributions `P(T = k)` and convolves both players' distributions to deliver **exact mathematical equity** in every bear-off race.

### 2. 3-Network Residual Architecture (`v3`) — 260 Grandmaster Features
Every board state is deterministically classified into one of three distinct strategic phases (`Contact`, `Crashed`, `Race`) and evaluated by a dedicated Residual MLP (following TD-Gammon / GNU Backgammon principles) using a canonical, perspective-symmetric **260-feature vector**:
* **192 Board Features:** Thermometer-encoded point occupancy across all 24 points for both players.
* **48 Structural Flexibility Features:** Explicit separation between fragile stripped points (`count === 2`) and flexible spare builders (`count === 3 || 4`).
* **20 Strategic Features:** Bar/borne-off counts, raw pip counts, exact expected bear-off turns from the 54,264-state tablebase, prime-trapping product (`primeLength × opponentCheckersBehindPrime`), hit lethality (`blotExposure × opponentHomeStrength`), Backgame timing ratios, double-anchor detection, and Golden Point (`20-Point`) ownership.

| Specialized Network | Game Phase Specialization | Architecture (`SiLU` + `Residual`) | Output (`6-Class Softmax`) |
| :--- | :--- | :--- | :--- |
| **`ContactNet`** | Contact battles, primes, blitzes & Backgames | `260 -> 512 -> 2×Res(512) -> 256 -> 6` | `Win 1/2/3`, `Lose 1/2/3` |
| **`CrashedNet`** | Crunched home boards (<=5 active checkers) & timing | `260 -> 384 -> Res(384) -> 192 -> 6` | `Win 1/2/3`, `Lose 1/2/3` |
| **`RaceNet`** | Pure non-contact races & pip-wastage minimization | `260 -> 384 -> Res(384) -> 192 -> 6` | `Win 1/2/3`, `Lose 1/2/3` |

### 3. Massive Local GPU Training — 15,000,512 Self-Play Games
All three networks were trained locally using **PyTorch (CUDA 12.4)** on an **NVIDIA GeForce RTX 4070 SUPER (12GB VRAM)** with a high-throughput multi-core simulation pipeline:
* **Numba `@njit(parallel=True)` Engine:** Compiled Backgammon move generation, state transitions, and 260-feature extraction to native machine code across all CPU cores (`prange`), accelerating self-play throughput from ~20 games/sec to **1,323–1,465 games/sec (~70× speedup)**.
* **4,096-Environment Mega-Batching (`TF32` + `BFloat16`):**
  1. **Stage 1 — Supervised Warm-Start:** Stratified pre-training across balanced Contact, Crashed, and Race positions evaluated via 2-ply Expectiminimax and the exact bear-off tablebase.
  2. **Stage 2 — Hybrid `TD(λ=0.7)` + 2-Ply AlphaZero Distillation:** Combines temporal-difference eligibility traces with batched 2-ply Expectiminimax target distribution distillation.
  3. **Hard-Example Curriculum Injection (`--hard-ratio 0.30`):** 30% of environment resets inject rare, high-complexity tactical archetypes (`2-3 Backgames`, `Deep Holding Games`, `Mutual Prime Battles`, and `Crunched Boards`).
* **Final Training Metrics:**
  * **Total Self-Play Volume:** `15,000,512` full games (~825M board evaluations) completed in `3h 08m`.
  * **Estimated Strength:** `~2071 Elo`.
  * **Hard-Position Validation:** `Holding 94.5%` | `PrimeBattle 56.8%` | `Crashed 55.4%` | `Backgame 16.5%`.
  * Weights auto-export to `public/weights/backgammon_nn.json` and execute in-browser via zero-allocation `Float32Array` inference.

### 4. Multi-Ply Expectiminimax Search (`Depth 1` to `5+`) & League Benchmarking
* **5 Configurable Search Depths:** Combines hierarchical beam pruning, a board-hash **Transposition Table**, GNU-style **Odd-Even Horizon Damping**, and a strict **9.5s Anytime Cutoff** so deep searches never exceed 10 seconds per turn.
* **Canonical Opening & 2nd-Turn Reply Book (`src/engine/openingBook.ts`):** Symmetric coverage of all 15 opening rolls and 2nd-turn replies for both White and Black.
* **10,000-Game Parallel GPU Round-Robin League (`training/tournament_benchmark.py`):** Evaluates all 5 depths across 10 head-to-head matchups (`1,000` independent random-dice games per pair; `4,000` games per engine), generating a full standings matrix, gammon/backgammon breakdown, PPG, and Bradley-Terry Elo ratings (`training/LEAGUE_REPORT.md`).

---

## 🗂️ Project Structure

```text
backgammon-ai/
├── public/
│   ├── weights/backgammon_nn.json   # Trained v3 3-Network ResNet weights (15M games)
│   ├── manifest.webmanifest         # PWA manifest (Hebrew RTL standalone app)
│   ├── sw.js                        # Offline Service Worker (Network-First for weights)
│   └── icon-*.png / favicon.svg     # Custom PWA app icons
├── src/
│   ├── engine/
│   │   ├── rules.ts                 # Official Backgammon rules & multi-step combined moves
│   │   ├── bearoffTablebase.ts      # 54,264-state Exact Bear-Off Tablebase (DP)
│   │   ├── neuralNet.ts             # 260-feature extractor & Float32Array 3-Net inference
│   │   ├── evaluator.ts             # 4-layer hybrid evaluator & Kleinman EPC
│   │   ├── openingBook.ts           # Canonical 1st-roll & 2nd-turn reply book
│   │   ├── ai.ts                    # Expectiminimax (Depth 1..5+), 36-roll combinatorics & explanations
│   │   ├── rollout.ts               # Monte Carlo Rollout engine (CRN)
│   │   ├── review.ts                # Accuracy %, equity timeline & key-moment extraction
│   │   └── visionScanner.ts         # Gemini Vision API client & board calibration helpers
│   ├── workers/ai.worker.ts         # Non-blocking Web Worker for deep search & rollouts
│   ├── hooks/useBackgammonAI.ts     # Async Worker orchestration hook
│   └── components/                  # Board, BoardArrows, EvalBar, EvalGraph, AnalysisPanel,
│                                    # GameReviewPanel, MoveHistory, ImagePositionModal
└── training/
    ├── backgammon_env.py            # Parallel Numba JIT simulator & hard-position generator
    ├── bearoff_dp.py                # 54,264-state exact bear-off solver (NumPy)
    ├── train_td_gammon.py           # PyTorch CUDA trainer (TD-Lambda + 2-Ply Distillation)
    ├── tournament_benchmark.py      # 10,000-game GPU-batched Round-Robin Depth League
    └── benchmark.py                 # Duplicate-Dice (CRN) baseline benchmark
```

---

## 🚀 Getting Started

### 1. Frontend Web App
```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Run unit test suite (Vitest)
npx vitest run

# Build production bundle
npm run build
```

### 2. Configure AI Vision Board Scanner (`.env`)
Copy `.env.example` to `.env` in the project root and add a free API key from [Google AI Studio](https://aistudio.google.com/apikey):
```env
VITE_GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Local GPU Training & Depth League (Optional — NVIDIA CUDA GPU)
```powershell
python -m venv training\.venv
.\training\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r training\requirements.txt

# Run mega-batched GPU self-play training (auto-resumes from latest checkpoint)
python training\train_td_gammon.py --warmup-samples 500000 --selfplay-games 15000000 --num-envs 4096 --hard-ratio 0.30 --resume

# Run 10,000-game parallel Round-Robin league across Depths 1, 2+, 3+, 4+, and 5+
python training\tournament_benchmark.py --games-per-pair 1000
```

---

## 🧪 Testing & Verification
The engine is verified by automated **Vitest** unit test suites (`rules.test.ts`, `bearoffTablebase.test.ts`, `neuralNet.test.ts`, `ai.test.ts`, `visionScanner.test.ts`, `review.test.ts`) covering move legality and combined steps, combinatorial bear-off indexing, 260-feature perspective symmetry, 36-roll combinatorics bounds, vision calibration transforms, and Game Review accuracy math.