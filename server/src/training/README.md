# D3QN Crypto Futures Trading - Training

Train a Deep Dueling Double DQN (D3QN) agent to trade cryptocurrency futures with leverage.

## Quick Start

### 1. Train with Synthetic Data (Testing)

```bash
python -m server.src.training.train_d3qn --sample-data --episodes 100
```

### 2. Train with Real Market Data

First, download real market data using the API:

```bash
# Start the server
uvicorn server.main:app --reload

# Then use the API to download data
curl -X POST "http://localhost:8000/api/data/download" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "start": "2023-01-01",
    "end": "2024-12-31"
  }'
```

Once data is downloaded, train with:

```bash
python -m server.src.training.train_d3qn \
  --symbol BTCUSDT \
  --timeframe 1h \
  --episodes 1000 \
  --balance 10000 \
  --leverage 3.0
```

## Command Line Arguments

### Data Parameters
- `--symbol`: Trading pair (default: BTCUSDT)
- `--timeframe`: Candle timeframe (default: 1h)
- `--sample-data`: Use synthetic data for testing

### Training Parameters
- `--episodes`: Number of training episodes (default: 1000)
- `--max-steps`: Maximum steps per episode (default: 10000)
- `--eval-freq`: Evaluate every N episodes (default: 50)
- `--save-freq`: Save checkpoint every N episodes (default: 100)

### Trading Parameters
- `--balance`: Initial balance in USD (default: 10000)
- `--leverage`: Maximum leverage (default: 3.0)
- `--window`: Observation window size in candles (default: 72 = 3 days for 1H)
- `--reward`: Reward type - `profit`, `return`, `sharpe`, or `sortino` (default: sharpe)

### Agent Parameters
- `--lr`: Learning rate (default: 0.00025)
- `--gamma`: Discount factor (default: 0.995)
- `--batch-size`: Batch size for training (default: 64)
- `--buffer-size`: Replay buffer size (default: 100000)

### Output Directories
- `--checkpoint-dir`: Directory to save model checkpoints (default: checkpoints)
- `--log-dir`: Directory for training logs (default: logs)

## Examples

### Fast Training (Testing)
```bash
python -m server.src.training.train_d3qn \
  --sample-data \
  --episodes 50 \
  --max-steps 500
```

### Full Training (Real Data)
```bash
python -m server.src.training.train_d3qn \
  --symbol BTCUSDT \
  --timeframe 1h \
  --episodes 2000 \
  --balance 10000 \
  --leverage 3.0 \
  --window 72 \
  --reward sharpe \
  --lr 0.00025 \
  --gamma 0.995
```

### High-Risk Training (Higher Leverage)
```bash
python -m server.src.training.train_d3qn \
  --symbol BTCUSDT \
  --balance 10000 \
  --leverage 10.0 \
  --reward sortino
```

## Output Files

### Checkpoints
- `checkpoints/best_agent.pth` - Best performing model (highest validation reward)
- `checkpoints/final_agent.pth` - Final model after all episodes
- `checkpoints/agent_episode_N.pth` - Checkpoints every N episodes

### Logs
- `logs/training_YYYYMMDD_HHMMSS.log` - Training log file
- `logs/metrics_YYYYMMDD_HHMMSS.json` - Training metrics (rewards, losses, etc.)

## Training Output

During training, you'll see output like:

```
============================================================
D3QN Crypto Futures Trading - Training
============================================================
Symbol: BTCUSDT
Timeframe: 1h
Episodes: 1000
Initial Balance: $10,000.00
Max Leverage: 3.0x
Window Size: 72 candles
Reward Type: sharpe
============================================================

📊 Loading market data...
✓ Loaded 10,000 candles
  Date range: 2023-01-01 to 2024-12-31

🔧 Preparing data...
✓ Data shape: (9902, 17)
✓ Train: 6,931 | Val: 1,485 | Test: 1,486

🤖 Initializing D3QN agent...
✓ State dim: 1512
✓ Action dim: 4 (HOLD, LONG, SHORT, CLOSE)
✓ Device: cuda

🚀 Starting training...
============================================================
Episode   10 | Reward:    -5.23 | Avg(10):    -8.45 | Loss: 0.0234 | Eps: 0.950 | Steps:  100 | Trades:  5 | Win%:  40.0 | Balance: $9,500
Episode   20 | Reward:     2.15 | Avg(10):     0.12 | Loss: 0.0189 | Eps: 0.902 | Steps:  150 | Trades:  8 | Win%:  50.0 | Balance: $10,200
...
```

## Architecture

The training pipeline consists of:

1. **Data Loader** (`server/src/data/loader.py`)
   - Loads market data from parquet/CSV files
   - Adds technical indicators (SMA, RSI, MACD, etc.)
   - Splits into train/val/test sets

2. **Trading Environment** (`server/src/policy/trading_env.py`)
   - Gymnasium-based RL environment
   - Simulates futures trading with leverage
   - Tracks positions, PnL, liquidations
   - Calculates rewards (profit, Sharpe, Sortino)

3. **D3QN Agent** (`server/src/agent/d3qn_agent.py`)
   - Deep Dueling Double DQN architecture
   - Experience replay buffer
   - Epsilon-greedy exploration
   - Target network for stability

4. **Training Script** (`server/src/training/train_d3qn.py`)
   - Main training loop
   - Periodic evaluation
   - Checkpoint saving
   - Metric logging

## Tips for Better Results

1. **Use real market data** - Synthetic data is only for testing
2. **Train longer** - 1000-2000 episodes recommended
3. **Tune reward type** - Try `sharpe` or `sortino` for risk-adjusted returns
4. **Adjust leverage** - Start with 2-3x, increase carefully
5. **Monitor validation** - Watch for overfitting
6. **Experiment with hyperparameters** - Learning rate, gamma, window size

## Next Steps

After training:

1. **Evaluate on test set** - Check generalization
2. **Backtest** - Use `server/src/eval/backtest.py`
3. **Paper trading** - Test with real-time data (no real money)
4. **Live trading** - Only after extensive testing!

## Troubleshooting

### No data files found
- Download data first using the API (see Quick Start)
- Or use `--sample-data` flag for testing

### CUDA out of memory
- Reduce `--batch-size` (try 32 or 16)
- Reduce `--buffer-size` (try 50000)
- Use CPU by setting `CUDA_VISIBLE_DEVICES=""`

### Agent not learning
- Try different reward types (`--reward sharpe` or `--reward sortino`)
- Adjust learning rate (`--lr 0.0001` or `--lr 0.0005`)
- Increase buffer size for more diverse experiences
- Check data quality (sufficient data, no gaps)

### Training too slow
- Reduce `--max-steps` per episode
- Use GPU if available
- Reduce `--window` size (but maintain enough context)
