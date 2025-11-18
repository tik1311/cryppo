# CRYPPO - Deep RL Crypto Trading System

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**CRYPPO** - GPU-Accelerated Deep Reinforcement Learning for Cryptocurrency Trading

Beautiful CLI interface powered by Rich library | PyTorch CUDA 12.6 | Python 3.12

---

## What is CRYPPO?

CRYPPO is a command-line cryptocurrency trading system that uses Deep Reinforcement Learning (D3QN) to learn optimal trading strategies. Train AI agents on GPU, backtest performance, and run paper trading simulations - all through an elegant CLI.

### Core Features

- 🎨 **Beautiful CLI** - Inspired by Claude Code, with progress bars, tables, and panels
- ⚡ **GPU-Accelerated** - CUDA 12.6 support for 10-20x faster training
- 🤖 **D3QN Agent** - Deep Dueling Double DQN with experience replay
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
