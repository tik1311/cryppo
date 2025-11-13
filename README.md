# CRYPPO - Deep RL Crypto Trading System

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**CRYPPO** - Advanced Deep Reinforcement Learning Cryptocurrency Trading Platform

Full-Stack Application with FastAPI Backend + Next.js Frontend + D3QN RL Model

---

## What is CRYPPO?

CRYPPO is an AI-powered cryptocurrency trading system that uses Deep Reinforcement Learning to learn optimal trading strategies. Train intelligent agents to trade crypto futures with leverage, backtest their performance, and run paper trading simulations.

### Core Capabilities

- **Data Management** - Download and prepare historical crypto market data
- **Model Training** - Train D3QN (Deep Dueling DQN) agents with custom parameters
- **Backtesting** - Test trained models on historical data with detailed metrics
- **Paper Trading** - Simulate live trading with real-time market updates
- **Web Interface** - Full-featured UI for all operations
- **CLI Tools** - Command-line tools for training and testing

---

## Features

### Trading

- **Futures Trading Simulation** - LONG, SHORT, and CLOSE positions with leverage
- **Realistic Market Conditions** - Fees (0.02-0.04%), slippage, liquidation
- **Risk Management** - Stop loss, take profit, max drawdown limits
- **Multiple Assets** - BTC, ETH, SOL, XRP, ADA support
- **Flexible Training** - Train by episodes or timesteps

### AI & ML

- **D3QN Agent** - Deep Dueling Double DQN with experience replay
- **Custom Rewards** - Sharpe ratio, Sortino ratio, Profit, or Return
- **Technical Indicators** - 20+ indicators (RSI, MACD, Bollinger, etc.)
- **GPU Acceleration** - CUDA support for faster training

### Monitoring & Analysis

- **Real-time Charts** - Candlestick charts with indicators
- **Performance Metrics** - Returns, Sharpe ratio, win rate, drawdown
- **Trade History** - Detailed log of all trades and P/L
- **Live Updates** - WebSocket-based real-time progress monitoring

---

## Installation

### Prerequisites

- **Python 3.10-3.12** (Python 3.12.9 recommended for Windows 11)
- **Node.js 18+** (for frontend)
- **8GB RAM minimum** (16GB recommended)
- **Optional**: NVIDIA GPU with CUDA 12.6 for faster training

### Quick Setup

#### Windows 11

```batch
REM 1. Clone repository
git clone <repository-url>
cd cryppo

REM 2. Run Windows setup script
setup_windows.bat

REM This will:
REM - Remove old virtual environment
REM - Create fresh Python virtual environment
REM - Install PyTorch with CUDA 12.6
REM - Install all dependencies (without conflicts)
REM - Install frontend packages
REM - Create necessary directories
```

#### Linux / macOS

```bash
# 1. Clone repository
git clone <repository-url>
cd cryppo

# 2. Run setup script
chmod +x setup.sh
./setup.sh

# This will:
# - Remove old virtual environment
# - Create fresh Python virtual environment
# - Install PyTorch with CUDA 12.6 (Linux) or CPU (macOS)
# - Install all dependencies
# - Install frontend packages
# - Create necessary directories
```

### Manual Installation (All Platforms)

#### Step 1: Create Clean Virtual Environment

**Windows:**
```batch
REM Remove old environment
rmdir /s /q venv

REM Create fresh environment
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

**Linux/macOS:**
```bash
# Remove old environment
rm -rf venv

# Create fresh environment
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

#### Step 2: Install PyTorch

**Windows with CUDA 12.6:**
```batch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**Linux with CUDA 12.6:**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**CPU Only (any platform):**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Verify Installation

```bash
# Verify critical packages
python -c "import torch; print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import websockets; print(f'✓ websockets {websockets.__version__}')"
python -c "import fastapi; print(f'✓ FastAPI {fastapi.__version__}')"
python -c "import ccxt; print(f'✓ CCXT {ccxt.__version__}')"
python -c "import gymnasium; print(f'✓ Gymnasium {gymnasium.__version__}')"
```

#### Step 5: Install Frontend

```bash
cd client
npm install
cd ..
```

### Troubleshooting Installation

**1. Dependency Conflicts (alpaca-trade-api, finrl)**

If you see errors like:
```
alpaca-trade-api requires websockets<11
finrl requires ccxt<4
```

**Solution:** These packages are NOT needed for CRYPPO and cause conflicts.

```bash
# Clean install (recommended):
# Run setup_windows.bat (Windows) or ./setup.sh (Linux/macOS)

# Or manually:
# 1. Remove old venv and start fresh
# 2. Install PyTorch first (see Step 2 above)
# 3. Install requirements.txt (which excludes conflicting packages)
```

**2. WebSocket 400 Bad Request Error**

**Root Cause:** Missing `websockets` Python package

```bash
# Check if websockets is installed
python -c "import websockets; print(websockets.__version__)"

# If you see ModuleNotFoundError, install it:
pip install websockets>=12.0
```

The server will show this warning on startup if websockets is missing:
```
✗ ERROR: websockets package not installed!
  WebSocket connections will fail with 400 Bad Request
```

**3. PyTorch Installation Issues**

If PyTorch CUDA installation fails, use CPU version:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**4. TA-Lib Installation (Optional)**

TA-Lib is optional and excluded on Windows by default.

**Windows:** Download wheel from https://github.com/cgohlke/talib-build/releases

**Linux:**
```bash
sudo apt-get install ta-lib
pip install TA-Lib
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

---

## Usage

### Option 1: Full UI (Recommended)

**Start Backend Server:**

```bash
cd server
python main.py
```

Or using uvicorn directly:

```bash
# Development (with auto-reload)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Start Frontend:**

```bash
cd client
npm run dev
```

**Access Web UI:**

- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws

### Option 2: CLI Only (For Testing)

**Train Model (CLI):**

```bash
cd server
python -m src.training.train_d3qn \
  --symbol BTCUSDT \
  --episodes 10 \
  --sample-data
```

**Backtest Model (CLI):**

```bash
python test_backtest.py \
  --model checkpoints/final_agent.pth \
  --sample-data \
  --balance 10000 \
  --leverage 3.0
```

---

## How to Use

### 1. Download Market Data

**Via Web UI:**

1. Go to http://localhost:3000/data
2. Enter symbols: BTCUSDT, ETHUSDT, etc.
3. Select date range
4. Click "Download Data"

**Via CLI:**

```python
from server.src.data.loader import download_and_save_data

download_and_save_data(
    symbol="BTCUSDT",
    timeframe="1h",
    start="2023-01-01",
    end="2024-12-31",
    exchange="binance"
)
```

### 2. Train a Model

**Via Web UI (Recommended):**

1. Go to http://localhost:3000/train
2. Select data file
3. Choose training mode:
   - **Episodes**: Complete runs (easier to understand)
     - Quick test: 2-10 episodes
     - Normal: 100 episodes
     - Full training: 1000+ episodes
   - **Timesteps**: Individual actions (granular control)
     - Quick test: 1,000-10,000 timesteps
     - Normal: 100,000 timesteps
     - Full training: 1,000,000+ timesteps
4. Configure parameters:
   - Model: D3QN (only fully implemented)
   - Learning rate: 0.00025 (default)
   - Gamma: 0.995 (discount factor)
   - Reward: Sharpe (recommended)
   - Batch size: 64
   - Buffer size: 100,000
5. Click "Start Training"
6. Monitor progress in real-time

**Via CLI:**

```bash
cd server

# Train with episodes (quick test)
python -m src.training.train_d3qn \
  --symbol BTCUSDT \
  --episodes 10 \
  --sample-data \
  --reward-type sharpe

# Train with timesteps (granular control)
python -m src.training.train_d3qn \
  --symbol BTCUSDT \
  --episodes 100 \
  --learning-rate 0.00025 \
  --gamma 0.995
```

### 3. Backtest a Model

**Via Web UI:**

1. Go to http://localhost:3000/backtest
2. Select trained model
3. Choose test period
4. Click "Run Backtest"
5. View results:
   - Performance metrics
   - Price chart with trades
   - Trade history table

**Via CLI:**

```bash
# Backtest with sample data (quick)
python test_backtest.py \
  --model checkpoints/final_agent.pth \
  --sample-data \
  --balance 10000 \
  --leverage 3.0

# Backtest with real data
python test_backtest.py \
  --model checkpoints/final_agent.pth \
  --symbol BTCUSDT \
  --balance 10000 \
  --leverage 3.0
```

### 4. Paper Trading

**Via Web UI:**

1. Go to http://localhost:3000/paper-trading
2. Select trained model
3. Configure:
   - Initial balance: $10,000
   - Max leverage: 3x
   - Update interval: 5 seconds
4. Click "Start Paper Trading"
5. Monitor live:
   - Real-time price chart
   - Current position & P/L
   - Trade execution log

---

## Configuration

Edit `server/src/configs/config.yaml`:

```yaml
data:
  symbols: ["BTCUSDT", "ETHUSDT"]
  timeframe: "1h"
  window_size: 72  # 3 days of 1H candles

trading:
  initial_balance: 10000.0
  max_leverage: 3.0
  fee_maker: 0.0002  # 0.02%
  fee_taker: 0.0004  # 0.04%
  max_drawdown: 0.20  # 20%

model:
  type: "d3qn"
  network:
    hidden_layers: [256, 256, 128]
  training:
    learning_rate: 0.00025
    gamma: 0.995
```

---

## Project Structure

```
cryppo/
├── server/                 # FastAPI Backend
│   ├── main.py            # API server entry point
│   └── src/
│       ├── agent/         # D3QN agent implementation
│       ├── data/          # Data loading & processing
│       ├── policy/        # Trading environment (Gymnasium)
│       ├── eval/          # Backtesting engine
│       ├── training/      # Training scripts
│       ├── utils/         # Helper functions
│       └── configs/       # Configuration files
│
├── client/                # Next.js Frontend
│   └── src/
│       ├── app/           # Pages (data, train, backtest, etc.)
│       ├── components/    # React components
│       └── lib/           # API & WebSocket clients
│
├── data/                  # Market data storage
├── checkpoints/           # Trained models
├── logs/                  # Log files
│   ├── server.log        # API server logs
│   ├── training.log      # Training logs
│   └── combined.log      # All logs combined
│
├── test_backtest.py      # CLI backtest script
├── setup.sh              # Setup script
└── README.md             # This file
```

---

## Logs & Debugging

CRYPPO uses separate log files for different operations:

### Log Files

- **`logs/server.log`** - API server operations, requests, WebSocket connections, errors
- **`logs/training.log`** - Model training progress, metrics, episode results, checkpoints
- **`logs/combined.log`** - All logs combined (for debugging)

### Viewing Logs

```bash
# View server logs (real-time)
tail -f logs/server.log

# View training logs (real-time)
tail -f logs/training.log

# Search for errors
grep ERROR logs/combined.log

# View recent training metrics
grep "Episode" logs/training.log | tail -20
```

### Log Rotation

- Logs rotate automatically when they reach size limits
- Server logs: 10MB max, keeps 5 backups
- Training logs: 20MB max, keeps 10 backups
- Combined logs: 50MB max, keeps 3 backups

---

## Troubleshooting

### Common Issues

**1. Dependency Conflicts (alpaca-trade-api, finrl, websockets)**

If you see pip warnings like:
```
alpaca-trade-api requires websockets<11, but you have websockets 15.0.1
finrl requires ccxt<4, but you have ccxt 4.5.12
```

**Solution:** Clean install with fresh virtual environment

```bash
# Windows:
setup_windows.bat

# Linux/macOS:
./setup.sh
```

These packages (alpaca-trade-api, finrl) are NOT used in CRYPPO and have been excluded from requirements.txt to prevent conflicts.

**2. Server won't start (Port in use)**

```bash
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/macOS:
lsof -i :8000
kill -9 <PID>

# Or start with different port:
uvicorn main:app --port 8001
```

**3. WebSocket connection failed (400 Bad Request)**

**Root Cause:** Missing `websockets` Python package

The server will show this warning on startup:
```
✗ ERROR: websockets package not installed!
  WebSocket connections will fail with 400 Bad Request
```

**Solution:**
```bash
# Check if websockets is installed
python -c "import websockets; print(websockets.__version__)"

# If ModuleNotFoundError, install it:
pip install websockets>=12.0
```

**Other possible causes:**
- Backend not running on port 8000
- Check `logs/server.log` for connection errors
- Verify frontend accessing `ws://localhost:8000/ws`
- Try restarting both frontend and backend

**4. Training takes too long**

- Use Episodes mode with 2-10 episodes for quick testing
- Or use Timesteps mode with 1,000 minimum (10,000+ recommended)
- Enable GPU if available (10-20x faster)
- Use sample data instead of real data for testing

**5. Model not found error**

- Check `checkpoints/` directory exists
- Verify model file path is correct
- Only D3QN is fully implemented (SAC, PPO, TD3 are placeholders)

**6. IndexError during backtest**

- This was fixed in latest version
- Make sure you pulled latest changes
- Error was due to improper `terminated`/`truncated` handling

**7. Frontend build fails**

```bash
cd client
rm -rf node_modules package-lock.json
npm install
npm run dev
```

---

## API Documentation

FastAPI provides interactive documentation:

**Swagger UI:** http://localhost:8000/docs

**ReDoc:** http://localhost:8000/redoc

### Key Endpoints

```
POST   /api/data/download        Download market data
GET    /api/data/available        List downloaded data files

POST   /api/train/start           Start model training
GET    /api/train/status/{id}     Check training status
GET    /api/models/available      List trained models

POST   /api/backtest/run          Run backtest
GET    /api/backtest/results/{id} Get backtest results

POST   /api/paper-trading/start   Start paper trading
POST   /api/paper-trading/stop    Stop paper trading
GET    /api/paper-trading/status  Get current status

GET    /api/system/info           System information
WS     /ws                        WebSocket connection
```

---

## Technical Stack

### Backend
- **FastAPI 0.121+** - Modern Python web framework
- **Uvicorn** - ASGI server with WebSocket support
- **PyTorch 2.0+** - Deep learning framework
- **Gymnasium 0.29+** - RL environment (OpenAI Gym successor)
- **CCXT 4.0+** - Cryptocurrency exchange API
- **Pandas/NumPy** - Data processing

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript 5+** - Type-safe JavaScript
- **TailwindCSS 3.3+** - Utility-first CSS
- **KLineChart 9.8+** - Candlestick charts
- **Axios 1.6+** - HTTP client
- **Zustand 4.4+** - State management

### AI/ML
- **D3QN Architecture** - Deep Dueling Double DQN
  - Dueling network: Separates value and advantage streams
  - Double DQN: Prevents Q-value overestimation
  - Experience replay: Improves sample efficiency
  - Target network: Stabilizes training

---

## Performance Tips

### For Training

1. **Use GPU** - 10-20x faster than CPU
2. **Start Small** - Test with 2-10 episodes first
3. **Use Sample Data** - For quick iterations during development
4. **Monitor Logs** - Check `logs/training.log` for progress
5. **Adjust Batch Size** - Larger batches (128-256) if you have enough RAM

### For Production

```bash
# Use Gunicorn with Uvicorn workers (recommended)
cd server
gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  -w 4 \
  -b 0.0.0.0:8000 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

### For Development

```bash
# Backend with auto-reload
cd server
uvicorn main:app --reload

# Frontend with hot-reload
cd client
npm run dev
```

---

## Security Notes

⚠️ **IMPORTANT**

- This software is for **educational and research purposes only**
- **DO NOT use for live trading** without extensive testing
- Cryptocurrency trading involves **substantial risk of loss**
- Always test thoroughly with paper trading first
- Keep your API keys secure (never commit to git)

---

## License

MIT License - see LICENSE file for details

---

## Support

- **Issues**: [GitHub Issues](https://github.com/tik1311/cryppo/issues)
- **Logs**: Check `logs/` directory for detailed error messages

---

**Built with ❤️ using PyTorch, FastAPI, and Next.js**
