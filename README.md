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

## Quick Start (CLI)

### 1. System Check

```batch
python cryppo.py check
```

Displays:
```
  ██████╗██████╗ ██╗   ██╗██████╗ ██████╗  ██████╗
 ██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗██╔══██╗██╔═══██╗
 ...

🔍 System Check

Component       Version/Status              Status
Python          3.12.9                      ✓
PyTorch         2.x.x+cu126                 ✓
CUDA            Available                   ✓
GPU             NVIDIA GeForce RTX 3060     ✓
...
```

### 2. Download Data

```batch
REM Download Bitcoin 1H data
python cryppo.py data download --symbol BTCUSDT --start 2023-01-01 --end 2024-12-31

REM List available data
python cryppo.py data list
```

### 3. Train Model (GPU Accelerated)

```batch
REM Quick test (2 episodes, ~10-20 seconds on GPU)
python cryppo.py train --symbol BTCUSDT --episodes 2 --sample-data --reward profit

REM Full training (100 episodes, ~5-10 minutes on GPU)
python cryppo.py train --symbol BTCUSDT --episodes 100 --reward profit

REM Risk-adjusted training (Sharpe ratio reward)
python cryppo.py train --symbol BTCUSDT --episodes 100 --reward sharpe

REM Custom parameters
python cryppo.py train --symbol ETHUSDT --episodes 200 --lr 0.0001 --reward sortino
```

**Real-time Training Progress (NEW!):**
```
🚀 Training D3QN Model

Parameter      Value
Symbol         BTCUSDT
Model          D3QN
Episodes       100
Learning Rate  0.00025
Reward Type    PROFIT

Starting training... (Real-time output)
======================================================================

Episode   1 | Reward:   -12.34 | Avg(10):    -12.34 | Loss: 0.0234 | Eps: 0.998 | Steps: 5432 | Trades: 12 | Win%:  50.0 | Balance: $10,234
Episode   2 | Reward:    23.45 | Avg(10):      5.56 | Loss: 0.0198 | Eps: 0.996 | Steps: 5431 | Trades: 15 | Win%:  53.3 | Balance: $10,456
Episode   3 | Reward:    45.67 | Avg(10):     18.93 | Loss: 0.0176 | Eps: 0.994 | Steps: 5430 | Trades: 18 | Win%:  55.6 | Balance: $10,789
...
Episode 100 | Reward:   123.45 | Avg(10):    112.34 | Loss: 0.0045 | Eps: 0.200 | Steps: 5429 | Trades: 24 | Win%:  62.5 | Balance: $12,345

======================================================================
✓ Training completed!
Model saved to: checkpoints/final_agent.pth
Detailed logs saved to: logs/
```

**NEW: Detailed Episode Logging**
- Automatically saves step-by-step logs for the last training episode
- Logs every action, state, reward, balance change
- Available in two formats:
  - `logs/episode_0100_*.jsonl` - Machine-readable JSON Lines
  - `logs/episode_0100_summary.txt` - Human-readable summary

See `TRAINING_GUIDE.md` for detailed documentation on log analysis.

### 4. Backtest Model

```batch
REM Run backtest with sample data
python cryppo.py backtest --model checkpoints/final_agent.pth --sample-data

REM Run backtest with real data
python cryppo.py backtest --model checkpoints/final_agent.pth

REM Results displayed:
```
📈 Running Backtest

[1/5] Loading market data...
[2/5] Preparing environment data...
[3/5] Creating trading environment...
  [DEBUG] Enabling detailed logging for backtest...
[4/5] Loading trained model...
[5/5] Running backtest simulation...

============================================================
BACKTEST RESULTS
============================================================
Initial Balance:  $10,000.00
Final Balance:    $12,450.00
Total Return:     +24.50%
Total Trades:     127
Win Rate:         58.30%
Sharpe Ratio:     1.8500
Max Drawdown:     12.30%
============================================================

[INFO] Detailed step-by-step logs saved to:
  logs/episode_0001_*.jsonl (step-by-step data in JSON Lines format)
  logs/episode_0001_summary.txt (human-readable summary)
============================================================
```

**NEW: Backtest Detailed Logging**
- Every backtest run automatically saves detailed logs
- Review every action, trade, and balance change
- Verify calculations are correct
- Analyze why the agent made specific decisions

### 5. List Models

```batch
python cryppo.py models
```

---

## CLI Commands Reference

### System

```batch
python cryppo.py check              # Check system requirements and GPU
```

### Data Management

```batch
python cryppo.py data download --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01
python cryppo.py data list          # List downloaded data files
```

### Training

```batch
python cryppo.py train --symbol BTCUSDT --episodes 100 [options]

Options:
  --symbol         Trading symbol (default: BTCUSDT)
  --episodes       Number of episodes (default: 100)
  --model          Model type: d3qn (default: d3qn)
  --lr             Learning rate (default: 0.00025)
  --reward         Reward type: profit/return/sharpe/sortino (default: sharpe)
  --sample-data    Use sample data for testing

Reward Types:
  profit     - Direct PnL reward (aggressive, high volatility)
  return     - Percentage-based return (balanced)
  sharpe     - Risk-adjusted return (conservative, smooth equity)
  sortino    - Downside risk-adjusted (allows upside volatility)

See TRAINING_GUIDE.md for detailed explanation of each reward type.
```

### Backtesting

```batch
python cryppo.py backtest --model checkpoints/final_agent.pth [options]

Options:
  --model          Path to trained model (required)
  --sample-data    Use sample data for testing
```

### Models

```batch
python cryppo.py models             # List all trained models
```

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
