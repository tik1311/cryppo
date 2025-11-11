# Usage Guide - RL Crypto Trading System

Complete guide for using the system from data download to backtesting.

## Quick Start

### 1. Setup Environment

```bash
# Run setup script
./setup.sh  # Linux/macOS
# or
setup.bat   # Windows
```

### 2. Download Data (1H Timeframe)

```bash
cd server
python download_data.py \
  --symbols BTC/USDT ETH/USDT \
  --start 2019-01-01 \
  --end 2024-12-31 \
  --output ../data \
  --format npz
```

**Output:** `../data/BTCUSDT_1h.npz`, `../data/ETHUSDT_1h.npz`

### 3. Train Agent

```bash
python train_agent.py \
  --data-path ../data/BTCUSDT_1h.npz \
  --output-dir ../checkpoints \
  --episodes 1000 \
  --eval-freq 50 \
  --save-freq 100
```

**Output:**
- `../checkpoints/best_model.pth` - Best model based on validation
- `../checkpoints/checkpoint_ep*.pth` - Periodic checkpoints
- `../checkpoints/final_model.pth` - Final trained model

### 4. Backtest Model

```bash
python backtest_model.py \
  --model-path ../checkpoints/best_model.pth \
  --data-path ../data/BTCUSDT_1h.npz \
  --save-results ../results/backtest_results.json
```

---

## Detailed Usage

### Data Download

#### Basic Usage
```bash
python download_data.py --symbols BTC/USDT --start 2023-01-01 --end 2023-12-31
```

#### Multiple Symbols
```bash
python download_data.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT BNB/USDT \
  --start 2019-01-01 \
  --end 2024-12-31
```

#### Output Formats
```bash
# NPZ (compressed numpy - recommended)
python download_data.py --symbols BTC/USDT --format npz

# CSV
python download_data.py --symbols BTC/USDT --format csv
```

---

### Training

#### Default Training
```bash
python train_agent.py --data-path ../data/BTCUSDT_1h.npz
```

#### Custom Training
```bash
python train_agent.py \
  --data-path ../data/BTCUSDT_1h.npz \
  --output-dir ./my_models \
  --episodes 2000 \
  --eval-freq 100 \
  --save-freq 200
```

#### Configuration
Edit `server/src/configs/config.yaml` to customize:
- Model architecture (hidden layers, activation)
- Hyperparameters (learning rate, gamma, epsilon)
- Trading parameters (fees, leverage, risk management)
- Reward function (sharpe, sortino, profit)

---

### Backtesting

#### Basic Backtest
```bash
python backtest_model.py \
  --model-path ../checkpoints/best_model.pth \
  --data-path ../data/BTCUSDT_1h.npz
```

#### Save Results
```bash
python backtest_model.py \
  --model-path ../checkpoints/best_model.pth \
  --data-path ../data/BTCUSDT_1h.npz \
  --save-results ../results/backtest_2024.json
```

---

## API Server (Optional)

### Start Server
```bash
cd server
python main.py
```

Server runs on `http://localhost:8000`

### API Endpoints

**Data Management:**
```bash
# Download data via API
curl -X POST http://localhost:8000/api/data/download \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["BTCUSDT"], "start_date": "2023-01-01", "end_date": "2023-12-31"}'

# List available data
curl http://localhost:8000/api/data/available
```

**Training:**
```bash
# Start training
curl -X POST http://localhost:8000/api/train/start \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["BTCUSDT"], "model_type": "d3qn", "timesteps": 2000000}'

# Check task status
curl http://localhost:8000/api/tasks/{task_id}

# List models
curl http://localhost:8000/api/models/available
```

**Backtesting:**
```bash
# Run backtest
curl -X POST http://localhost:8000/api/backtest/run \
  -H "Content-Type: application/json" \
  -d '{"model_path": "./checkpoints/best_model.pth", "symbols": ["BTCUSDT"]}'

# Get results
curl http://localhost:8000/api/backtest/results/{task_id}
```

**System:**
```bash
# Health check
curl http://localhost:8000/api/health

# System info
curl http://localhost:8000/api/system/info

# Get config
curl http://localhost:8000/api/config
```

### Interactive API Docs
Visit http://localhost:8000/docs for Swagger UI with try-it-out functionality.

---

## Configuration

### Main Config File
`server/src/configs/config.yaml`

```yaml
# Key settings for 1H trading
data:
  timeframe: "1h"           # Only 1H supported
  window_size: 72           # 3 days of 1H candles
  symbols: ["BTCUSDT", "ETHUSDT"]

trading:
  initial_balance: 10000.0
  max_leverage: 3.0
  fee_maker: 0.0002         # 0.02%
  fee_taker: 0.0004         # 0.04%
  max_drawdown: 0.20        # 20%

model:
  type: "d3qn"
  network:
    hidden_layers: [256, 256, 128]
  training:
    learning_rate: 0.00025
    gamma: 0.995
    epsilon_decay: 0.9995
  reward_type: "sharpe"     # profit, sharpe, sortino
```

---

## Tips & Best Practices

### 1. Data Quality
- Download multiple years of data (2019-2024) for better training
- Use the most recent data for testing
- Check for data gaps or anomalies

### 2. Training
- Start with shorter training runs (500-1000 episodes) to validate setup
- Monitor validation reward to prevent overfitting
- Save checkpoints regularly
- Experiment with different reward functions

### 3. Backtesting
- Always backtest on unseen data
- Compare with buy-and-hold strategy
- Consider transaction costs and slippage
- Look at multiple metrics (Sharpe, max drawdown, win rate)

### 4. Risk Management
- Never use 100% of capital
- Set appropriate stop losses
- Monitor maximum drawdown
- Test with different leverage levels

---

## Troubleshooting

### Out of Memory
- Reduce window_size in config
- Use smaller batch_size
- Train on fewer episodes at once

### Training Too Slow
- Use GPU if available (CUDA)
- Reduce network size (hidden_layers)
- Decrease eval_freq and save_freq

### Poor Performance
- Try different reward functions
- Adjust learning rate
- Increase training episodes
- Check data quality

### Dependencies Issues
```bash
# TA-Lib installation
# Ubuntu/Debian:
sudo apt-get install ta-lib
pip install TA-Lib

# macOS:
brew install ta-lib
pip install TA-Lib

# Windows:
# Download wheel from: https://github.com/mrjbq7/ta-lib#windows
pip install TA_Lib-*.whl
```

---

## Example Workflow

Complete workflow from scratch:

```bash
# 1. Setup
./setup.sh

# 2. Download data
cd server
python download_data.py --symbols BTC/USDT --start 2019-01-01 --end 2024-12-31

# 3. Train
python train_agent.py \
  --data-path ../data/BTCUSDT_1h.npz \
  --episodes 1000

# 4. Backtest
python backtest_model.py \
  --model-path ../checkpoints/best_model.pth \
  --data-path ../data/BTCUSDT_1h.npz \
  --save-results ../results/backtest.json

# 5. Check results
cat ../results/backtest.json
```

---

## Next Steps

- **Phase 2**: Frontend development (Next.js dashboard)
- **Phase 3**: Live trading integration
- **Enhancements**: More RL algorithms (PPO, SAC, A3C)

---

For issues or questions, visit: https://github.com/GodEyeTee/cryppo/issues
