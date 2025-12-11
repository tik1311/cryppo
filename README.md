# CRYPPO - Deep RL Crypto Trading System

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**CRYPPO** - Deep Reinforcement Learning for Cryptocurrency Trading

---

## Overview

CRYPPO is a command-line cryptocurrency trading system using Deep Reinforcement Learning (D3QN - Dueling Double Deep Q-Network) with multi-timeframe analysis. Train AI agents, backtest performance, and run paper trading simulations.

### Architecture

- **D3QN Agent** - Dueling Double DQN for stable Q-learning
- **CNN-BiLSTM-Attention Network** - Hybrid architecture for price pattern recognition
- **Multi-Timeframe** - 1h main + 1m detail for precise TP/SL execution
- **ATR-based TP/SL** - Dynamic take-profit and stop-loss based on volatility
- **Trend Signal** - Built-in moving average crossover for direction bias
- **Risk Management** - 10% capital per trade (Kelly-inspired)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the CLI
python main.py
```

Select from the menu:
1. **System Check** - Verify GPU, data, models
2. **Download Data** - Fetch BTCUSDT from Binance
3. **List Data Files** - Show available datasets
4. **Train Model** - Train a new D3QN agent
5. **Backtest** - Test on historical data
6. **Paper Trade (Historical)** - Simulate on test set
7. **Live Paper Trade** - Real-time with Binance API
8. **List Models** - Show trained models
9. **Exit**

---

## Installation

### Requirements

- Python 3.10+
- NVIDIA GPU (recommended) or CPU
- 8GB+ RAM

### Steps

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 2. Install PyTorch (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify
python main.py  # Select "System Check"
```

---

## Project Structure

```
cryppo/
├── main.py                 # Single entry point (CLI)
├── requirements.txt        # Python dependencies
├── data/                   # Market data (parquet files)
├── checkpoints/            # Trained models (.pth)
├── logs/                   # Training/backtest logs
└── server/src/
    ├── agent/
    │   ├── d3qn_agent.py       # D3QN Agent
    │   └── hybrid_network.py   # Neural Network
    ├── data/
    │   ├── loader.py           # Data loading
    │   ├── features.py         # Technical indicators
    │   └── realtime_fetcher.py # Binance API
    └── policy/
        └── trading_env.py      # Gym Environment
```

---

## How It Works

### Training

1. Downloads OHLCV data from Binance (1h + 1m timeframes)
2. Calculates technical indicators (RSI, MACD, BB, ATR, etc.)
3. Agent explores with epsilon-greedy policy
4. Learns Q-values for actions: HOLD, LONG, SHORT
5. Uses Double DQN to reduce overestimation bias
6. Saves model checkpoint after training

### Trading Logic

- **Entry**: On 1h candle close, agent decides LONG/SHORT/HOLD
- **TP/SL**: Set using ATR multipliers (default: TP=2x, SL=1x ATR)
- **Exit**: Only via TP or SL hit (no manual close)
- **Position Size**: 10% of capital per trade
- **Trend Filter**: Uses MA crossover signal in observations

### Reward Structure

| Event | Reward |
|-------|--------|
| Open Position | +0.01 |
| TP Hit | +1.0 |
| SL Hit | -0.5 |
| Hold > 50 bars | -0.005/bar |
| Bankruptcy | -5.0 |

---

## Configuration

### Training Parameters (main.py)

```python
window_size = 36       # Observation window (hours)
tp_mult = 2.0          # Take-profit at 2x ATR
sl_mult = 1.0          # Stop-loss at 1x ATR
min_bars = 1           # Min bars between trades
lr = 0.0003            # Learning rate
gamma = 0.99           # Discount factor
batch_size = 32        # Replay buffer batch
update_freq = 4        # Network update frequency
```

### Agent Parameters (d3qn_agent.py)

```python
epsilon = 1.0          # Initial exploration
epsilon_min = 0.05     # Minimum exploration
epsilon_decay = 0.9995 # Decay rate per update
```

---

## Troubleshooting

### Low Returns Despite Bull Market

If your model only SHORTs in a bull market:

1. **Delete old model**: Remove `checkpoints/d3qn_BTCUSDT_1h.pth`
2. **Retrain**: Run training with more episodes (50+)
3. **Check trend signal**: Model should follow MA crossover
4. **Verify exploration**: Epsilon should decay slowly

### DLL Error (Windows)

```
OSError: [WinError 1114] DLL initialization failed
```

Solution: Install Visual C++ Redistributable
- Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Install and restart computer

### Out of Memory

- Reduce `batch_size` to 16
- Reduce `window_size` to 24
- Use CPU if GPU memory is limited

---

## Research References

- [Deep RL for Trading - Critical Survey](https://www.mdpi.com/2306-5729/6/11/119)
- [DQN Trading Agent Framework](https://github.com/MehranTaghian/DQN-Trading)
- [Dueling DQN Architecture](https://arxiv.org/abs/1511.06581)
- [Double DQN](https://arxiv.org/abs/1509.06461)

---

## License

MIT License - See LICENSE file

---

## Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research before trading.
