# CRYPPO - Deep RL Crypto Trading System

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**CRYPPO** - GPU-Accelerated Deep Reinforcement Learning for Cryptocurrency Trading

Beautiful CLI interface powered by Rich library | PyTorch CUDA 12.6 | Python 3.12

---

## What is CRYPPO?

CRYPPO is a command-line cryptocurrency trading system that uses state-of-the-art Deep Reinforcement Learning (Hybrid CNN-BiLSTM-Attention-QR-DQN) to learn optimal trading strategies. Train AI agents on GPU, backtest performance, and run paper trading simulations - all through an elegant CLI.

The system combines:
- **CNN** for pattern recognition (support/resistance, candlesticks)
- **BiLSTM** for temporal modeling (trends, reversals)
- **Self-Attention** for focusing on critical market moments
- **QR-DQN** for distributional RL and risk quantification
- **Action Masking** for preventing invalid trades
- **Position-Aware States** for complete trade cycle learning

### Core Features

- 🎨 **Beautiful CLI** - Inspired by Claude Code, with progress bars, tables, and panels
- ⚡ **GPU-Accelerated** - CUDA 12.6 support for 10-20x faster training
- 🤖 **Hybrid CNN-BiLSTM-Attention-QR-DQN** - State-of-the-art deep RL architecture (2024-2025 research)
- 🧠 **Advanced State Representation** - Position type one-hot encoding + action masking for valid actions
- 📊 **Real-time Data** - Download from Binance, backtest, paper trade
- 🎯 **Simple Commands** - `cryppo train`, `cryppo backtest`, `cryppo data download`

---

## Installation (Windows 11 + NVIDIA GPU)

### Step 1: Fix PyTorch DLL Error (Required for GPU)

**The Issue:**
```
OSError: [WinError 1114] DLL initialization failed
Error loading "torch\lib\c10.dll"
```

**The Solution:** Install Visual C++ Redistributable

```batch
REM Method 1: Automated (Recommended)
REM Right-click and "Run as Administrator"
install_vcredist.bat

REM Method 2: Manual Download
REM Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
REM Install and restart computer
```

**After installing Visual C++ Redistributable, RESTART YOUR COMPUTER!**

### Step 2: Install PyTorch with CUDA 12.6

```batch
REM Activate virtual environment
venv\Scripts\activate

REM Install PyTorch GPU version
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

REM Verify GPU is detected
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

You should see:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3060 (or your GPU model)
```

### Step 3: Install Dependencies

```batch
REM Install all Python packages
pip install -r requirements.txt

REM Verify everything works
python cryppo.py check
```

---

## Interactive Terminal UI

- Run everything with a single file:  
  ```batch
  python main.py
  ```
- Rich UI presents a numbered menu (system check, data ops, training, backtest, paper trading, model listing).
- Each action asks for parameters with sensible defaults; behind the scenes it forwards the selections to `cryppo.py`.
- Use `Ctrl+C` or select option `0` to exit.
- Live paper trading (option 8) now loads the selected RL checkpoint, feeds it real 1m candles, and executes at most one action per closed bar with true-to-market TP/SL/trailing/liquidation logic.
- Every workflow writes an audit log: `logs/training/*.log`, `logs/backtest/*.json`, `logs/paper/*.json` capture metrics plus full trade history (with position ids) so you can review executions after the run finishes.

---

## CLI Workflow

### 1. Run the diagnostics

```batch
python cryppo.py check
```

Confirms Python, PyTorch, CUDA, GPU, FastAPI, Gymnasium, CCXT, and Pandas are ready to go. All encoding is handled internally so it works even on fresh Windows terminals.

### 2. Manage datasets entirely from the CLI

| Command | Purpose |
| --- | --- |
| `python cryppo.py data download --symbol BTCUSDT --start 2023-01-01 --end 2024-12-31` | Pull real Binance futures data (parquet) |
| `python cryppo.py data list` | Show what already lives in `data/` |
| `python cryppo.py data sample --symbol TEST --samples 2000` | Generate synthetic candles for smoke tests |
| `python cryppo.py data prepare --symbol BTCUSDT --timeframe 1h --window 72 --output data/processed` | Build normalized `.npz` splits for training/backtests |

The `prepare` command mirrors the environment preprocessing pipeline (indicators + normalization) and stores train/val/test tensors so you can reuse the same slices for every model.

### 3. Train any supported RL agent

Use `--model` to switch algorithms. D3QN trains in episodes, while Stable-Baselines3 agents train in timesteps.

```batch
# D3QN (discrete) – fast sanity check
python cryppo.py train --model d3qn --symbol BTCUSDT --episodes 2 --sample-data

# PPO (on-policy SB3)
python cryppo.py train --model ppo --symbol BTCUSDT --timesteps 50000 --sample-data

# SAC (off-policy, continuous wrapper)
python cryppo.py train --model sac --symbol BTCUSDT --timesteps 50000 --buffer-size 200000

# TD3 (off-policy, continuous wrapper)
python cryppo.py train --model td3 --symbol BTCUSDT --timesteps 50000 --buffer-size 200000
```

Common flags:

| Flag | Applies to | Notes |
| --- | --- | --- |
| `--timeframe`, `--symbol` | all models | Control the OHLC dataset |
| `--episodes` | d3qn | Number of full environment runs |
| `--timesteps` | ppo/sac/td3 | Total SB3 timesteps |
| `--reward {profit,return,sharpe,sortino}` | all | Align reward shaping with your risk appetite |
| `--sample-data` | all | Use synthetic data instead of downloading |
| `--checkpoint-dir`, `--log-dir` | all | Where models/logs are stored (relative to the server for SB3 models) |

Under the hood SAC and TD3 now use a `ContinuousActionWrapper`, so continuous SB3 policies can drive the discrete trading environment reliably.

### 4. Backtest across all model types

```batch
# D3QN checkpoint at project root
python cryppo.py backtest --model checkpoints/final_agent.pth --model-type d3qn --sample-data

# SB3 checkpoints under server/checkpoints
python cryppo.py backtest --model server/checkpoints/final_ppo_model.zip --model-type ppo
python cryppo.py backtest --model server/checkpoints/final_sac_model.zip --model-type sac
python cryppo.py backtest --model server/checkpoints/final_td3_model.zip --model-type td3
```

The CLI handles importing the correct agent, flattening observations when needed, computing Sharpe/Sortino/max drawdown, and printing a Rich summary table.

### 5. Paper trade straight from the terminal

```batch
python cryppo.py paper --model checkpoints/final_agent.pth --model-type d3qn --symbol BTC/USDT --timeframe 1m --steps 120 --interval 2
```

- Fetches live candles via CCXT
- Streams position/action/balance in a Rich panel
- Works with D3QN and every SB3 model (the wrapper handles continuous logits)

### 6. List saved checkpoints

```batch
python cryppo.py models
```

Shows size and modified time for everything inside `checkpoints/`.

---

## CLI Commands Reference

| Category | Command | Description |
| --- | --- | --- |
| Diagnostics | `python cryppo.py check` | Verify Python, GPU, and required libs |
| Data | `python cryppo.py data download ...` | Pull live market data (parquet) |
|  | `python cryppo.py data list` | Inventory downloaded datasets |
|  | `python cryppo.py data sample ...` | Generate synthetic candles for tests |
|  | `python cryppo.py data prepare ...` | Create normalized train/val/test `.npz` bundles |
| Training | `python cryppo.py train --model d3qn ...` | Deep Dueling Double DQN (episodes) |
|  | `python cryppo.py train --model ppo|sac|td3 ...` | Stable-Baselines3 algorithms (timesteps) |
| Backtest | `python cryppo.py backtest --model ... --model-type ...` | Unified evaluator with Sharpe, drawdown, and real-time metrics |
| Live Sim | `python cryppo.py paper --model ...` | Live-style paper trading loop using CCXT |
| Models | `python cryppo.py models` | List checkpoints inside `checkpoints/` |

Reward modes (`--reward`):

- `profit` – maximize raw PnL
- `return` – percentage returns
- `sharpe` – penalize volatility
- `sortino` – focus on downside risk

All training logs stream to `logs/` with JSON metrics plus human-readable summaries, and SB3 models store zipped checkpoints inside `server/checkpoints/`.

---

## Verification Matrix

Every CLI flow was exercised end-to-end with synthetic data:

| Stage | Command run | Result |
| --- | --- | --- |
| Environment sanity | `python cryppo.py check` | Pass (GPU + deps detected) |
| Sample data | `python cryppo.py data sample --symbol TESTCOIN --samples 500` | Generated parquet |
| Data prep | `python cryppo.py data prepare --symbol TESTCOIN --sample-data --window 32` | Saved `data/processed/TESTCOIN_1h_prepared.npz` |
| Training | `python cryppo.py train --model d3qn --episodes 1 --sample-data` | Produced `checkpoints/final_agent.pth` |
|  | `python cryppo.py train --model ppo|sac|td3 --timesteps 100-200 --sample-data` | Generated `server/checkpoints/final_*_model.zip` for each algorithm |
| Backtests | `python cryppo.py backtest --model ...` (d3qn and ppo) | Printed Rich summaries + metrics |
| Paper trade | `python cryppo.py paper --model checkpoints/final_agent.pth --symbol BTC/USDT` | Live CCXT feed rendered in terminal |

The environment bug where `position_penalty` was unset has been fixed, and SAC/TD3 now operate through the new continuous-action wrapper so every listed algorithm can download data, train, and evaluate via the CLI alone.

---

## GPU Training Performance

**Training Speed Comparison (100 episodes):**

| Hardware | Time | Speed |
|----------|------|-------|
| NVIDIA RTX 3060 | ~5-8 minutes | ⚡⚡⚡⚡⚡ |
| NVIDIA GTX 1660 Ti | ~8-12 minutes | ⚡⚡⚡⚡ |
| CPU (Ryzen 7 5800X) | ~60-90 minutes | ⚡ |

**GPU is 10-20x faster!** That's why we fix the DLL error instead of using CPU.

---

## Troubleshooting

### ❌ "DLL initialization failed" Error

**Cause:** Missing Visual C++ Redistributable

**Solution:**
```batch
REM Run as Administrator
install_vcredist.bat

REM Then restart computer
REM Then reinstall PyTorch:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### ❌ "CUDA Available: False" After Installing PyTorch

**Causes:**
1. CUDA Toolkit not installed
2. Wrong PyTorch version (CPU instead of GPU)
3. NVIDIA drivers outdated

**Solutions:**
```batch
REM 1. Install CUDA Toolkit 12.6
REM Download: https://developer.nvidia.com/cuda-downloads

REM 2. Update NVIDIA drivers
REM Download: https://www.nvidia.com/Download/index.aspx

REM 3. Reinstall PyTorch GPU version
pip3 uninstall torch torchvision
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

REM 4. Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### ❌ "ModuleNotFoundError: No module named 'rich'"

**Solution:**
```batch
pip install rich
```

### ✅ Everything Works But Training is Slow

**Check if GPU is actually being used:**
```batch
python -c "import torch; print(f'Using device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"
```

If it says `cpu`, follow the "CUDA Available: False" troubleshooting above.

---

## Project Structure

```
cryppo/
├── cryppo.py                 # Main CLI entry point ⭐
├── install_vcredist.bat      # Visual C++ installer ⭐
├── requirements.txt          # Python dependencies
├── setup_windows.bat         # Full setup script
│
├── server/                   # Backend code
│   ├── src/
│   │   ├── agent/           # D3QN implementation
│   │   ├── policy/          # Trading environment
│   │   ├── training/        # Training scripts
│   │   ├── eval/            # Backtesting
│   │   └── data/            # Data loading
│   └── main.py              # FastAPI server (optional)
│
├── data/                     # Downloaded market data
├── checkpoints/              # Trained models
└── logs/                     # Training logs
```

---

## 📊 Phase 5 Performance Improvements

### Problem Analysis (From Latest Training)

**Training Results (Before Phase 5)**:
- Episodes: 2 (test run)
- Win Rate: 24.4% (Episode 1), 24.5% (Episode 2)
- Final Balance: $2,970 / $2,996 (from $10,000) = **-70% Loss**
- HOLD Bias: 94% (only 6% trading actions)
- R:R Ratio: 2.43x
- Break-even Win Rate Required: 29.2%
- **Result**: 24.5% < 29.2% = Guaranteed Loss

**Key Problems**:
1. **Insufficient Win Rate** (24% vs 29% required)
2. **Severe HOLD Bias** (94% HOLD, only 6% trades)
3. **Low R:R Ratio** (2.43x insufficient for 24% win rate)
4. **No Learning** (both episodes -70%, no improvement)
5. **Overtrading** (rapid entries/exits during epsilon exploration)

---

### 🔧 Phase 5 Improvements Implemented

#### 1. Increased TP/SL Ratio (4.62x)

**Before**:
```python
atr_sl_multiplier = 1.5  # SL distance
atr_tp_multiplier = 4.5  # TP distance
# R:R = 4.5/1.5 = 3.0x
```

**After**:
```python
atr_sl_multiplier = 1.3  # Tighter SL (-13%)
atr_tp_multiplier = 6.0  # Wider TP (+33%)
# R:R = 6.0/1.3 = 4.62x (+54%)
```

**Impact**:
- Old: Win Rate 25%, R:R 3.0x → Expectancy = 0.25×3.0 - 0.75×1 = **-0.125** (loss)
- New: Win Rate 25%, R:R 4.62x → Expectancy = 0.25×4.62 - 0.75×1 = **+0.405** (profit!)

#### 2. Stronger HOLD Penalties

**Before**:
- Penalties start at 30 consecutive HOLDs
- Penalty at 50 HOLDs: -0.02
- Penalty at 100 HOLDs: -0.05

**After**:
- Penalties start at **20** consecutive HOLDs (-33% earlier)
- Penalty at 40 HOLDs: **-0.05** (2.5x stronger)
- Penalty at 80 HOLDs: **-0.15** (3x stronger)

**Impact**: Forces agent to explore trading actions more aggressively

#### 3. Increased Exploration (Epsilon)

**Before**:
```python
epsilon_end = 0.15    # 85% greedy, 15% random
epsilon_decay = 0.9995  # Fast decay
```

**After**:
```python
epsilon_end = 0.25    # 75% greedy, 25% random (+67% exploration)
epsilon_decay = 0.9997  # Slower decay
```

**Impact**: More exploration throughout training

#### 4. Enhanced Trade Completion Rewards

**Before**:
- TP Hit: +0.10 reward
- SL Hit: -0.005 penalty

**After**:
- TP Hit: +0.20 reward + 0.02 bonus (3x total)
- SL Hit: -0.002 penalty + 0.005 bonus (rewards taking risk)

**Impact**: Strongly encourages completing trades vs infinite HOLD

---

### 📈 Expected Performance (Phase 5)

| Metric | Before (Phase 4) | After (Phase 5 Target) | Change |
|--------|------------------|----------------------|--------|
| Win Rate | 24% | 28-32% | +17-33% |
| R:R Ratio | 2.43x | 4.62x | +90% |
| Expectancy | -12.5% | +20-40% | Profitable! |
| HOLD % | 94% | 70-80% | -15-25% |
| Trade % | 6% | 20-30% | +233-400% |
| Final Balance | $2,970 (-70%) | $12,000-15,000 | +20-50% |

---

## 🏗️ Training Flow (Detailed)

### Episode Training Loop

```
┌─────────────────────────────────────────────────────────┐
│                  START EPISODE                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Random Start Index   │ ← Prevents memorization
         │  (within data range)  │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Reset Environment   │
         │  Balance = $10,000    │
         │  Position = None      │
         └───────────┬───────────┘
                     │
                     ▼
      ┌──────────────────────────────┐
      │      STEP LOOP (Start)       │
      └──────────────┬───────────────┘
                     │
    ┌────────────────┴────────────────┐
    │                                 │
    ▼                                 ▼
┌────────────────┐          ┌─────────────────┐
│  Get State     │          │  Epsilon-Greedy │
│  (72×27 array) │          │  Action Select  │
└────────┬───────┘          └────────┬────────┘
         │                           │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Get Action Mask      │ ← Phase 4.2
         │  [1,1,1] or [1,0,0]   │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Apply Action        │
         │  HOLD / LONG / SHORT  │
         └───────────┬───────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
  ┌──────────┐            ┌──────────────┐
  │ No Trade │            │  Open Trade  │
  │ (HOLD)   │            │  (LONG/SHORT)│
  └────┬─────┘            └──────┬───────┘
       │                         │
       │                         ▼
       │            ┌───────────────────────┐
       │            │  Calculate TP/SL      │
       │            │  TP = Entry ± 6.0×ATR │
       │            │  SL = Entry ± 1.3×ATR │
       │            └──────────┬────────────┘
       │                       │
       └───────────┬───────────┘
                   │
                   ▼
      ┌────────────────────────┐
      │   Check TP/SL Hit      │ ← Auto-close
      │   (every timestep)     │
      └────────────┬───────────┘
                   │
       ┌───────────┴──────────┐
       │                      │
       ▼                      ▼
  ┌─────────┐          ┌──────────┐
  │ TP Hit  │          │  SL Hit  │
  │ +Profit │          │ -Loss    │
  └────┬────┘          └─────┬────┘
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
      ┌───────────────────────┐
      │  Calculate Reward     │
      │  = Trade + Position   │
      │    + Exploration      │
      └───────────┬───────────┘
                  │
                  ▼
      ┌───────────────────────┐
      │  Store Transition     │
      │  (s, a, r, s', done)  │
      │  → Replay Buffer      │
      └───────────┬───────────┘
                  │
                  ▼
      ┌───────────────────────┐
      │  Train Network        │
      │  (if buffer > 128)    │
      │  - Sample batch       │
      │  - Calculate TD error │
      │  - Backprop + Update  │
      └───────────┬───────────┘
                  │
                  ▼
      ┌───────────────────────┐
      │  Update Target Net    │
      │  (every 250 steps)    │
      └───────────┬───────────┘
                  │
                  ▼
      ┌───────────────────────┐
      │  Decay Epsilon        │
      │  ε = ε × 0.9997       │
      │  (min 0.25)           │
      └───────────┬───────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Episode End?  │
         │  (truncated or │
         │   terminated)  │
         └────┬──────┬────┘
              │      │
         No   │      │   Yes
              │      │
              │      ▼
              │  ┌───────────────────┐
              │  │  Position Open?   │
              │  └────┬──────┬───────┘
              │       │      │
              │  Yes  │      │  No
              │       │      │
              │       ▼      ▼
              │  ┌─────────────────┐
              │  │ Extend Episode  │
              │  │ (max 500 bars)  │
              │  │ Wait for TP/SL  │
              │  └────┬────────────┘
              │       │
              └───────┴──────►  NEXT EPISODE
```

---

## 🧠 Network Architecture Flow

### Forward Pass (Hybrid CNN-BiLSTM-Attention-QR-DQN)

```
INPUT: State (72 timesteps × 27 features)
  │
  │  Market Data (17):        Position Info (10):
  │  ├─ OHLCV (5)            ├─ Position Type [no/long/short] (3)
  │  ├─ RSI (1)              ├─ Entry Price Norm (1)
  │  ├─ MACD (3)             ├─ PnL % (1)
  │  ├─ Bollinger (3)        ├─ Hold Duration (1)
  │  ├─ ATR (1)              ├─ Market Regime [bull/bear/side] (3)
  │  └─ Other (4)            └─ Regime Strength (1)
  │
  ▼
┌─────────────────────────────────────────────┐
│         1. CNN Feature Extractor            │
│  ┌──────────────────────────────────────┐   │
│  │  Multi-scale 1D Convolutions:        │   │
│  │  ├─ Conv(kernel=3) → 64 channels     │   │
│  │  ├─ Conv(kernel=5) → 64 channels     │   │
│  │  └─ Conv(kernel=7) → 64 channels     │   │
│  │                                        │   │
│  │  Concat → 192 channels                │   │
│  │  ↓                                     │   │
│  │  Conv1D(1×1) → 128 channels           │   │
│  │  ↓                                     │   │
│  │  BatchNorm + ReLU + Dropout(0.1)      │   │
│  └──────────────────────────────────────┘   │
│                                               │
│  Output: (72, 128)                           │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│         2. Bidirectional LSTM               │
│  ┌──────────────────────────────────────┐   │
│  │  2 LSTM Layers (bidirectional)       │   │
│  │  ├─ Layer 1: 128 hidden × 2 dirs     │   │
│  │  └─ Layer 2: 128 hidden × 2 dirs     │   │
│  │                                        │   │
│  │  Forward + Backward → 256 features    │   │
│  │  Dropout(0.1) between layers          │   │
│  └──────────────────────────────────────┘   │
│                                               │
│  Output: (72, 256)                           │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│         3. Self-Attention (4 heads)         │
│  ┌──────────────────────────────────────┐   │
│  │  Query = Linear(256 → 256)           │   │
│  │  Key   = Linear(256 → 256)           │   │
│  │  Value = Linear(256 → 256)           │   │
│  │                                        │   │
│  │  Attention Weights:                   │   │
│  │  A = softmax(Q·K^T / √d_k)           │   │
│  │                                        │   │
│  │  Weighted Sum: Z = A·V               │   │
│  │                                        │   │
│  │  Residual: Output = Z + Input        │   │
│  │  LayerNorm                            │   │
│  └──────────────────────────────────────┘   │
│                                               │
│  Output: (72, 256)                           │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│         4. Feature Aggregation              │
│  ┌──────────────────────────────────────┐   │
│  │  Last Timestep: (256,)               │   │
│  │  +                                    │   │
│  │  Attention Mean: mean((72,256)) → (256,)│ │
│  │  =                                    │   │
│  │  Combined: (512,)                     │   │
│  └──────────────────────────────────────┘   │
└───────────────────┬─────────────────────────┘
                    │
            ┌───────┴────────┐
            │                │
            ▼                ▼
┌──────────────────┐  ┌──────────────────┐
│  5a. Value Stream│  │5b. Advantage Stream│
│                  │  │                   │
│  FC(512→256)     │  │  FC(512→256)      │
│  ReLU            │  │  ReLU             │
│  FC(256→32)      │  │  FC(256→3×32)     │
│                  │  │                   │
│  V(s) [32 quant.]│  │  A(s,a) [3×32]    │
└────────┬─────────┘  └─────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│         6. QR-DQN Combination               │
│  ┌──────────────────────────────────────┐   │
│  │  For each action a:                  │   │
│  │                                        │   │
│  │  Q(s,a) = V(s) + (A(s,a) - mean_a(A))│   │
│  │                                        │   │
│  │  32 quantiles per action:             │   │
│  │  [τ₁, τ₂, ..., τ₃₂] for each of      │   │
│  │  [HOLD, LONG, SHORT]                  │   │
│  │                                        │   │
│  │  Final Q-value = mean(quantiles)      │   │
│  └──────────────────────────────────────┘   │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
              OUTPUT: Q-values
              ├─ Q(HOLD)  = mean(32 quantiles)
              ├─ Q(LONG)  = mean(32 quantiles)
              └─ Q(SHORT) = mean(32 quantiles)
```

---

## 🎯 Action Selection (Epsilon-Greedy with Masking)

```
START: Receive state s
  │
  ▼
┌──────────────────────┐
│  Get Action Mask     │
│  from Environment    │
└──────────┬───────────┘
           │
           ▼
    ┌─────────────┐
    │ Has Position?│
    └──┬────────┬─┘
       │        │
    Yes│        │No
       │        │
       ▼        ▼
  ┌─────────┐  ┌─────────┐
  │mask=[1  │  │mask=[1  │
  │      0  │  │      1  │
  │      0] │  │      1] │
  └────┬────┘  └────┬────┘
       │            │
       └─────┬──────┘
             │
             ▼
      ┌──────────────┐
      │Random < ε ?  │
      └──┬────────┬──┘
         │        │
      Yes│        │No
         │        │
         ▼        ▼
  ┌────────────┐  ┌────────────┐
  │EXPLORATION │  │EXPLOITATION│
  │            │  │            │
  │Sample from │  │Forward pass│
  │valid acts  │  │through net │
  │with bias:  │  │            │
  │HOLD: 10%   │  │Q-values:   │
  │LONG: 45%   │  │[Q₀,Q₁,Q₂]  │
  │SHORT: 45%  │  │            │
  │            │  │Apply mask: │
  │Apply mask  │  │Q[mask==0]  │
  │            │  │ = -∞       │
  │            │  │            │
  │Select max  │  │Select max  │
  └─────┬──────┘  └──────┬─────┘
        │                │
        └────────┬───────┘
                 │
                 ▼
          ┌──────────┐
          │  Action  │
          │  a ∈{0,1,2}│
          └──────────┘
```

---

## 💰 Reward Calculation (Phase 5)

```
Total Reward = Trade Reward + Position Penalty + Exploration Bonus

┌─────────────────────────────────────────┐
│         1. Trade Reward                 │
├─────────────────────────────────────────┤
│  On Position Close (TP/SL):             │
│  ├─ PnL / Initial Balance               │
│  ├─ TP Hit: +0.20 (bonus)               │
│  └─ SL Hit: -0.002 (minimal penalty)    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│         2. Position Penalty             │
├─────────────────────────────────────────┤
│  While Holding Position:                │
│  ├─ If against regime: -0.002/step      │
│  ├─ If low volume: -0.001/step          │
│  └─ If high volatility: -0.001/step     │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│         3. Exploration Bonus            │
├─────────────────────────────────────────┤
│  HOLD Penalties (Phase 5):              │
│  ├─ >20 HOLD: -0.005×(steps-20)         │
│  ├─ >40 HOLD: -0.05 extra               │
│  └─ >80 HOLD: -0.15 extra               │
│                                          │
│  Trade Bonuses:                          │
│  ├─ Break HOLD streak: +0.005            │
│  ├─ Any trade action: +0.001             │
│  ├─ TP Hit bonus: +0.02                  │
│  └─ SL Hit bonus: +0.005 (risk-taking)   │
│                                          │
│  Market Regime Bonus:                    │
│  ├─ LONG in bull: +0.003×strength        │
│  ├─ SHORT in bear: +0.003×strength       │
│  └─ Volume confirmed: +0.002             │
└─────────────────────────────────────────┘
```

---

## 🧪 Training Process (Loss Calculation)

```
Every Step (if buffer > batch_size):

1. Sample Batch from Replay Buffer
   ├─ Random: batch_size=128 transitions
   └─ OR Prioritized: weighted by TD-error

2. Compute Current Q-Values
   Q(s,a) = PolicyNet(s)[a]  (32 quantiles)

3. Compute Target Q-Values
   a' = argmax_a TargetNet(s')  (Double DQN)
   Q_target(s',a') = TargetNet(s')[a']

   For each quantile τ:
   y_τ = r + γ × Q_target_τ(s',a') × (1-done)

4. Quantile Huber Loss
   For each quantile pair (Q_τ, y_τ):

   δ = y_τ - Q_τ

   κ = {  0.5×δ²        if |δ| ≤ 1
       {  |δ| - 0.5     otherwise

   ρ_τ(δ) = |τ - 𝟙(δ<0)| × κ

   Loss = (1/32) × Σ ρ_τ(δ)

5. Backpropagation
   optimizer.zero_grad()
   loss.backward()
   clip_grad_norm_(params, max_norm=10)
   optimizer.step()

6. Update Target Network (every 250 steps)
   TargetNet ← PolicyNet
```

---

## Hybrid CNN-BiLSTM-Attention-QR-DQN Architecture

CRYPPO uses a state-of-the-art hybrid deep learning architecture based on 2024-2025 research:

### Architecture Components

```
Input (72 timesteps × 27 features)
    ↓
1. CNN Feature Extractor (Multi-scale)
    - Kernel sizes: 3, 5, 7 for different temporal patterns
    - 64 channels per scale → 128 combined channels
    - Batch normalization + dropout (0.1)
    ↓
2. Bidirectional LSTM
    - 2 layers, 128 hidden units each direction
    - Total: 256 features (forward + backward)
    - Captures long-term temporal dependencies
    ↓
3. Self-Attention Mechanism
    - 4 attention heads
    - Focuses on critical time steps
    - Residual connection + layer normalization
    ↓
4. Feature Aggregation
    - Last timestep + attention-weighted average
    - Combined: 512 features
    ↓
5. Dueling D3QN Head
    - Value stream: V(s) → 32 quantiles
    - Advantage stream: A(s,a) → 3 actions × 32 quantiles
    - Q(s,a) = V(s) + (A(s,a) - mean(A))
    ↓
6. QR-DQN Output
    - 32 quantiles per action for distributional RL
    - Better risk assessment vs single Q-value
    - Returns: [HOLD, LONG, SHORT] Q-values
```

### State Representation (27 Features)

**Market Data (17 features):**
- OHLCV (5): Open, High, Low, Close, Volume
- Technical Indicators (12): RSI, MACD, ATR, Bollinger Bands, etc.

**Position Information (10 features):**
- Position Type One-Hot (3): `[no_position, long_position, short_position]`
- Entry Price Normalized (1)
- PnL Percentage (1)
- Hold Duration Normalized (1)
- Market Regime One-Hot (3): `[bull, bear, sideways]`
- Regime Strength (1)

### Action Masking

The environment enforces valid actions:
- **No Position**: Can HOLD, LONG, or SHORT
- **Has Position**: Can only HOLD (positions close automatically at TP/SL)

Action masking prevents:
- Opening new positions while holding one
- Invalid state transitions
- Inefficient exploration of impossible actions

### Episode Extension Logic

**Problem**: Random episode sampling could cut episodes mid-position, preventing natural TP/SL closures.

**Solution**: Position-aware episode extension
1. Episodes sample random starting points (prevents memorization)
2. If episode ends with open position → extend up to 500 bars
3. Wait for natural TP/SL closure
4. Only force-close if max extension exceeded

This ensures the agent learns from complete trade cycles, not interrupted positions.

### Key Advantages

1. **CNN**: Captures local patterns (support/resistance, candlestick patterns)
2. **BiLSTM**: Models long-term trends and reversals
3. **Attention**: Focuses on market regime changes and volatility spikes
4. **QR-DQN**: Distributional RL quantifies uncertainty and risk
5. **Dueling**: Separates "how good is this state" from "which action is best"
6. **Position State**: Agent knows if it can trade or must wait
7. **Action Masking**: Eliminates wasted exploration on invalid actions

---

## Technical Stack

- **PyTorch 2.x** - Deep learning with CUDA 12.6
- **Gymnasium 0.29+** - RL environment
- **Rich** - Beautiful CLI interface
- **CCXT 4.x** - Cryptocurrency exchange API
- **NumPy, Pandas** - Data processing

---

## Why CLI Instead of Web UI?

1. **Faster Iteration** - No need to start frontend server
2. **GPU Focus** - Direct access to training, no overhead
3. **Scriptable** - Easy to automate and batch process
4. **Beautiful** - Rich library makes CLI as pretty as GUI
5. **Terminal Native** - Works over SSH, in tmux, anywhere

---

## Advanced Usage

### Batch Training Multiple Symbols

```batch
REM Create training script
@echo off
python cryppo.py train --symbol BTCUSDT --episodes 100
python cryppo.py train --symbol ETHUSDT --episodes 100
python cryppo.py train --symbol SOLUSDT --episodes 100
```

### Automated Daily Training

```batch
REM Schedule with Windows Task Scheduler
python cryppo.py data download --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31
python cryppo.py train --symbol BTCUSDT --episodes 50
python cryppo.py backtest --model checkpoints/latest.pth
```

### Monitor Training with Logs

```batch
REM Real-time log monitoring
tail -f logs/training.log   # Linux/macOS
Get-Content logs\training.log -Wait   # PowerShell
```

---

## Contributing

This is a personal trading system. Not accepting contributions at this time.

---

## License

MIT License - see LICENSE file

---

## Disclaimer

⚠️ **IMPORTANT**

- This software is for **educational and research purposes only**
- **DO NOT use for live trading** without extensive testing
- Cryptocurrency trading involves **substantial risk of loss**
- Past performance does not guarantee future results
- Always test with paper trading first

---

## Support

- **Issues**: Check `logs/` directory for detailed error messages
- **GPU Problems**: Run `python cryppo.py check` to diagnose
- **Training Issues**: Enable sample data mode first (`--sample-data`)

---

**Built with ❤️ using PyTorch, Rich, and CUDA**

*GPU-accelerated AI trading at your fingertips*
