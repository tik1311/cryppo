# CRYPPO - Deep RL Crypto Trading System

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**CRYPPO** - Deep Reinforcement Learning for Cryptocurrency Trading

---

## Overview

CRYPPO is a cryptocurrency trading system using **PPO (Proximal Policy Optimization)** - a state-of-the-art deep reinforcement learning algorithm that learns to trade profitably.

### Why PPO?

Based on [2025 research](https://arxiv.org/pdf/2504.02281v3), PPO outperforms DQN for cryptocurrency trading:
- **More stable training** - Clipped objective prevents catastrophic updates
- **Better for volatile markets** - Adapts well to crypto's high volatility
- **Continuous learning** - Can improve incrementally with more data
- **Balanced exploration** - Learns both LONG and SHORT positions

### Architecture

- **PPO Agent** - Actor-Critic with clipped surrogate objective
- **BiLSTM-CNN Network** - Captures temporal patterns and price structures
- **Multi-Timeframe** - 1h main + 1m detail for precise TP/SL execution
- **Trend-Following Rewards** - Bonus for trading with the trend
- **ATR-based TP/SL** - Dynamic risk management based on volatility

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
4. **Train Model** - Train a new PPO agent
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
    │   ├── ppo_agent.py        # PPO Agent (Primary)
    │   ├── d3qn_agent.py       # D3QN Agent (Legacy)
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

### Training Process

1. **Data Loading**: Downloads OHLCV data from Binance (1h + 1m)
2. **Feature Engineering**: Calculates RSI, MACD, BB, ATR, MA crossovers
3. **PPO Training Loop**:
   - Agent observes market state
   - Selects action (HOLD, LONG, SHORT) based on policy
   - Receives reward based on trade outcome and trend alignment
   - Updates policy using PPO's clipped objective
4. **Model Checkpoint**: Saves best model after training

### Reward Structure

| Event | Reward | Description |
|-------|--------|-------------|
| LONG in uptrend | +0.1 | Trading with the trend |
| SHORT in downtrend | +0.1 | Trading with the trend |
| LONG in downtrend | -0.05 | Trading against trend |
| SHORT in uptrend | -0.05 | Trading against trend |
| TP Hit | +1.0 | Profitable trade |
| SL Hit | -0.5 | Losing trade (asymmetric) |
| Hold > 30 bars | -0.01/bar | Encourages action |

### Trading Logic

- **Entry**: On 1h candle close, agent decides LONG/SHORT/HOLD
- **TP/SL**: Set using ATR multipliers (TP=2.5x, SL=1x ATR)
- **Exit**: Only via TP or SL hit (no manual close)
- **Position Size**: 10% of capital per trade (Kelly-inspired)
- **Trend Filter**: Uses MA crossover signal (MA20 vs MA50)

---

## Configuration

### Training Parameters (main.py)

```python
window_size = 36       # Observation window (hours)
tp_mult = 2.5          # Take-profit at 2.5x ATR
sl_mult = 1.0          # Stop-loss at 1x ATR
min_bars = 2           # Min bars between trades
lr = 3e-4              # Learning rate (PPO standard)
gamma = 0.99           # Discount factor
rollout_length = 2048  # Steps before PPO update
n_epochs = 10          # PPO optimization epochs
batch_size = 64        # Minibatch size
```

### PPO Hyperparameters (ppo_agent.py)

```python
clip_epsilon = 0.2     # PPO clipping range
gae_lambda = 0.95      # GAE lambda
value_coef = 0.5       # Value loss coefficient
entropy_coef = 0.01    # Entropy bonus (exploration)
max_grad_norm = 0.5    # Gradient clipping
```

---

## Training Tips

### For Best Results

1. **Train for 50+ episodes** - PPO needs more data than DQN
2. **Watch the entropy** - Should stay > 0.3 for good exploration
3. **Monitor win rate** - Should improve over episodes
4. **Check trend balance** - Model should trade both LONG and SHORT

### Expected Behavior

```
Episode 1:  WR: ~33%, Random trading
Episode 10: WR: ~40%, Learning trend patterns
Episode 30: WR: ~45%, Balanced LONG/SHORT
Episode 50: WR: ~50%+, Trend-following behavior
```

---

## Troubleshooting

### Model Only Trades One Direction

If model only trades LONG or SHORT:
1. Check entropy in logs - should be > 0.3
2. Increase `entropy_coef` to 0.02
3. Train for more episodes

### Low Win Rate

1. Verify trend signal is working
2. Check TP/SL ratio (should be > 2:1)
3. Ensure training data has both bull and bear markets

### Training Not Improving

1. Check learning rate - try 1e-4 or 5e-4
2. Verify reward scaling is not too extreme
3. Check for NaN in logs

---

## Research References

- [PPO vs DQN for Trading (2025)](https://arxiv.org/pdf/2504.02281v3)
- [Deep RL for Trading - Critical Survey](https://www.mdpi.com/2306-5729/6/11/119)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [FinRL Contest 2025](https://arxiv.org/abs/2504.02281)

---

## License

MIT License - See LICENSE file

---

## Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research before trading.
