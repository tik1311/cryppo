# CRYPPO - Deep RL Crypto Trading System

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**CRYPPO** - Advanced Deep Reinforcement Learning Cryptocurrency Trading Platform

Full-Stack Application พร้อม FastAPI Backend, Next.js Frontend และ Multiple RL Models (D3QN, SAC, PPO, TD3)

---

## 🔄 Recent Updates (Latest)

### Bug Fixes & Improvements
- ✅ **Fixed IndexError in backtest**: Properly handle Gymnasium's `terminated` and `truncated` flags
- ✅ **Added Episodes configuration**: Train by episodes (easier) or timesteps (granular control)
- ✅ **Lowered minimum training requirements**: Can now test with as few as 2 episodes or 10 timesteps
- ✅ **Improved boundary checks**: Prevent data access beyond array bounds in trading environment
- ✅ **Updated to latest WebSocket practices**: Following FastAPI/Uvicorn 2025 best practices
- ✅ **Minimal test data support**: Quick testing with 2,000 sample candles
- ✅ **UI clarity**: Clearly marked which models are implemented (D3QN ✅) vs coming soon (SAC, PPO, TD3 🚧)

### What's Working
- ✅ D3QN model training and backtesting
- ✅ Web UI for data download, training, backtesting
- ✅ Trading environment with proper action handling (no conflicts)
- ✅ Sample data generation for quick testing

### Known Limitations
- ⚠️ Only D3QN is fully implemented (SAC, PPO, TD3 are placeholders)
- ⚠️ Real-time training progress in UI is limited (uses polling)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Requirements](#-requirements)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Development Roadmap](#-development-roadmap)
- [License](#-license)

---

## ✨ Features

### Phase 1 (Current - Backend)
- **FastAPI Backend** - RESTful API with WebSocket support
- **Multiple Timeframes** - Flexible timeframe support (1h, 4h, 1d)
- **Data Management** - Download and prepare data from Binance/OKX
- **Model Training** - Deep Dueling DQN (D3QN) with PER
- **Backtesting** - Comprehensive backtesting engine
- **Real-time Updates** - WebSocket for progress monitoring

### Phase 2 (Completed - Frontend)
- **Next.js 14 Frontend** - Modern React-based web interface
- **Live Charts** - Real-time candlestick charts with KLineChart & Lightweight Charts
- **Trading Dashboard** - Monitor training, backtesting, and paper trading
- **System Monitoring** - GPU status, data inventory, model management
- **Paper Trading** - Live paper trading simulation with real-time updates
- **Performance Metrics** - Real-time metrics, trades history, and visualizations

### Trading Features
- **Multi-Asset Support** - BTC, ETH, SOL, XRP, ADA
- **Realistic Simulation** - Fees (0.02-0.04%), slippage (2 bps), liquidation
- **Risk Management** - Stop loss (2%), take profit (4%), max drawdown (20%)
- **Position Limits** - Max 10-day holding, 12-hour cooldown after SL

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              Frontend (Phase 2)                 │
│         Next.js 14 + React + TailwindCSS        │
│              Port 3000                          │
└────────────────────┬────────────────────────────┘
                     │ REST API / WebSocket
┌────────────────────▼────────────────────────────┐
│              Backend (Phase 1)                  │
│         FastAPI + Uvicorn + WebSockets          │
│              Port 8000                          │
├─────────────────────────────────────────────────┤
│  • Data Download & Preparation                  │
│  • Model Training (D3QN with PER)               │
│  • Backtesting Engine                           │
│  • Paper Trading Simulator                      │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│           Core Trading Engine                   │
│      PyTorch + Gymnasium + NumPy/Pandas         │
└─────────────────────────────────────────────────┘
```

---

## 💻 Requirements

### System Requirements
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: CUDA-capable GPU recommended (optional)
- **Storage**: 10GB for data and models

### Dependencies
See `requirements.txt` for full list. Key dependencies:
- `torch>=2.0.0` - Deep learning framework
- `fastapi>=0.104.0` - Web framework
- `ccxt>=4.0.0` - Cryptocurrency exchange connector
- `gymnasium>=0.29.0` - RL environment
- `pandas>=2.0.0` - Data processing

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd cryppo
```

### 2. Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

### 3. Start Backend Server
```bash
cd server
python main.py
```

### 4. Start Frontend (in another terminal)
```bash
cd client
npm install
npm run dev
```

### 5. Access Web UI

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

**Note**: All operations (data download, training, backtesting, paper trading) are performed through the Web UI. No command-line scripts are required.

---

## 📂 Project Structure

```
cryppo/
├── server/                     # Backend (FastAPI)
│   ├── main.py                # Main API server (Web UI only)
│   └── src/                   # Core modules
│       ├── agent/             # RL agents (D3QN)
│       ├── data/              # Data processing & features
│       ├── policy/            # Trading environment
│       ├── eval/              # Backtesting engine
│       ├── execution/         # Paper trading
│       ├── training/          # Training utilities
│       ├── utils/             # Helper functions
│       ├── database/          # Database schema (SQLite)
│       ├── configs/           # Configuration files
│       │   └── config.yaml    # Main config (flexible timeframe)
│       └── legacy/            # Legacy code (for reference)
│
├── client/                    # Frontend (Next.js 14 + TypeScript)
│   ├── package.json           # Node dependencies
│   └── src/
│       ├── app/               # Next.js pages
│       │   ├── page.tsx       # Home dashboard
│       │   ├── data/          # Data management
│       │   ├── train/         # Training interface
│       │   ├── backtest/      # Backtesting interface
│       │   ├── paper-trading/ # Paper trading interface
│       │   └── system/        # System settings
│       ├── components/        # React components
│       │   ├── CandlestickChart.tsx  # KLineChart integration
│       │   ├── TradingChart.tsx      # Lightweight Charts
│       │   ├── MetricsDisplay.tsx    # Live metrics
│       │   └── TradesList.tsx        # Trade history
│       └── lib/               # API & WebSocket clients
│
├── data/                      # Data storage
│   └── npz/                   # Model files & datasets
│
├── requirements.txt           # Python dependencies
├── setup.sh                   # Setup script
└── README.md                  # This file
```

---

## ⚙️ Configuration

Edit `server/src/configs/config.yaml` to customize:

### Data Configuration
```yaml
data:
  symbols: ["BTCUSDT", "ETHUSDT"]
  timeframe: "1h"              # Only 1H supported
  train_start: "2019-01-01"
  train_end: "2024-12-31"
  window_size: 72              # 3 days of 1H candles
```

### Trading Parameters
```yaml
trading:
  initial_balance: 10000.0
  max_leverage: 3.0
  fee_maker: 0.0002           # 0.02%
  fee_taker: 0.0004           # 0.04%
  max_drawdown: 0.20          # 20% termination
```

### Model Configuration
```yaml
model:
  type: "d3qn"                # Dueling Double DQN
  network:
    hidden_layers: [256, 256, 128]
  training:
    learning_rate: 0.00025
    gamma: 0.995
    total_timesteps: 2000000
```

---

## 📖 Web UI Usage

All operations are performed through the Web UI at http://localhost:3000

### 1. Download Data
1. Navigate to **Data** page
2. Enter symbols (e.g., BTCUSDT, ETHUSDT)
3. Select date range (start and end dates)
4. Click **Download Data**
5. View downloaded files in the data table

### 2. Train Model
1. Navigate to **Train** page
2. Select data files to use for training
3. Configure training parameters:
   - **Model type**: D3QN (only fully implemented model)
   - **Training mode**: Choose between Episodes or Timesteps
     - **Episodes**: Complete runs from start to finish (recommended for beginners)
       - Quick test: 2-10 episodes
       - Normal training: 100 episodes
       - Full training: 1000+ episodes
     - **Timesteps**: Individual actions taken (more granular control)
       - Quick test: 1,000-10,000 timesteps
       - Normal training: 100,000 timesteps
       - Full training: 1,000,000+ timesteps
   - **Learning rate**: 0.00025 (default, works well)
   - **Gamma**: 0.995 (discount factor)
   - **Reward function**: Sharpe (recommended), Sortino, Profit, or Return
   - **Batch size**: 64 (default)
   - **Buffer size**: 100,000 (replay buffer capacity)
4. Click **Start Training**
5. Monitor training progress with real-time metrics

**Note**: SAC, PPO, and TD3 models are listed but not yet fully implemented. Use D3QN for now.

### 3. Backtest Model
1. Navigate to **Backtest** page
2. Select trained model from dropdown
3. Choose symbol and date range for testing
4. Click **Run Backtest**
5. View results:
   - Performance metrics (return, Sharpe, win rate)
   - Price chart with indicators
   - Trade history table

**CLI Alternative** (for testing with minimal data):
```bash
python test_backtest.py \
  --model checkpoints/final_agent.pth \
  --sample-data \
  --balance 10000 \
  --leverage 3.0
```

### 4. Paper Trading
1. Navigate to **Paper Trading** page
2. Select trained model
3. Configure:
   - Initial balance
   - Max leverage
   - Fee rates
   - Update interval
4. Click **Start Paper Trading**
5. Monitor live:
   - Real-time price chart
   - Current position and P/L
   - Trade execution history
   - Performance metrics

---

## 🔌 API Documentation

FastAPI provides interactive API documentation at http://localhost:8000/docs

### Key Endpoints:
- `POST /api/data/download` - Download market data
- `POST /api/train/start` - Start training
- `POST /api/backtest/run` - Run backtest
- `POST /api/paper-trading/start` - Start paper trading
- `GET /api/system/info` - System information
- `WS /ws` - WebSocket for real-time updates

---

## 📊 Features Overview

### Data Management
- Download historical data from exchanges (Binance, etc.)
- Support for multiple symbols (BTC, ETH, SOL, XRP, ADA)
- flexible timeframe only
- View and manage downloaded data files

### Training
- D3QN (Dueling Double DQN) agent
- Customizable hyperparameters
- Multiple reward functions (Sharpe, Sortino, Profit)
- Real-time training progress monitoring (Future)
- Automatic checkpoint saving

### Backtesting
- Comprehensive performance metrics
- Visual chart with indicators
- Trade-by-trade analysis
- Export results to CSV/JSON (Future)

### Paper Trading
- Real-time simulation with live market data
- Position tracking (LONG/SHORT/NEUTRAL)
- Unrealized P/L monitoring
- Live performance metrics
- Trade execution history

---

## 🐛 Troubleshooting

### Common Issues and Solutions

**1. IndexError when running backtest**
```
IndexError: index XXXX is out of bounds for axis 0 with size XXXX
```
**Solution**: This has been fixed in the latest version. The issue was due to improper handling of Gymnasium's `terminated` and `truncated` flags. Make sure you're using the updated `test_backtest.py` and `trading_env.py`.

**2. Training takes too long / system hangs**
- Use **Episodes mode** with 2-10 episodes for quick testing
- Or use **Timesteps mode** with minimum 1,000 timesteps (lower values now supported)
- Reduce sample data size to 2,000 candles for testing
- Close other applications to free up RAM/GPU

**3. WebSocket connection issues**
- Ensure backend server is running on port 8000
- Check firewall settings
- Use latest FastAPI/Uvicorn (WebSocket support improved in 2025)

**4. Model training fails with "Model not found"**
- Only D3QN is fully implemented - use D3QN model type
- Check that data files are downloaded first
- Verify checkpoint directory exists

**5. Action conflicts in trading environment**
- **HOLD (0)**: Do nothing, maintain current state
- **OPEN_LONG (1)**: Open long position, or close short and go long
- **OPEN_SHORT (2)**: Open short position, or close long and go short
- **CLOSE_POSITION (3)**: Close any open position
- If already in a position and same action (e.g., OPEN_LONG while long), it acts as HOLD

---

## 🛠️ Technical Details

### Backend (FastAPI)
- **Framework**: FastAPI 0.121+ with Uvicorn
- **WebSocket**: Real-time bidirectional updates (latest best practices 2025)
  - Stateful connection management
  - Proper error handling with WebSocketException
  - For production: Use Gunicorn + Uvicorn workers
- **Database**: SQLite (tracking & history)
- **ML Framework**: PyTorch 2.0+
- **RL Library**: Gymnasium 0.29+ (formerly OpenAI Gym)
  - Proper handling of `terminated` vs `truncated` flags
  - Custom D3QN implementation with Dueling architecture
  - Experience replay buffer with batch sampling

### Frontend (Next.js)
- **Framework**: Next.js 14.0.4 with App Router
- **Language**: TypeScript 5+
- **UI**: TailwindCSS 3.3+
- **Charts**: KLineChart 9.8+ & Lightweight Charts 4.1+
- **State**: Zustand 4.4+
- **HTTP**: Axios 1.6+

### RL Training
- **Episodes vs Timesteps**:
  - Episodes: Full environment runs (easier to understand)
  - Timesteps: Individual actions (more granular)
- **Minimum values**:
  - Episodes: 1+ (2-10 for quick testing)
  - Timesteps: 10+ (1000+ recommended for meaningful training)

---

## 🗺️ Development Roadmap

### Phase 1: Backend (✅ Completed)
- [x] FastAPI server structure
- [x] Configuration system (YAML)
- [x] Data download script (Binance)
- [x] API endpoints structure
- [x] WebSocket support
- [x] **D3QN Agent implementation**
- [x] **Feature engineering & data processing**
- [x] **Trading Environment (Gymnasium)**
- [x] **Training script (train_agent.py)**
- [x] **Backtesting engine**
- [x] **Database schema (SQLite)**
- [x] **Complete documentation (README + USAGE)**

### Phase 2: Frontend (✅ Completed)
- [x] Next.js 14 setup with TypeScript
- [x] Dashboard layout & navigation
- [x] **Data Management UI** - Download & view data files
- [x] **Training Interface** - Start training with configuration
- [x] **Backtest Interface** - Run backtests & view results with charts
- [x] **System Settings** - View configuration & system info
- [x] **Paper Trading Interface** - Real-time paper trading with live updates
- [x] History page (placeholder for future)
- [x] API Client integration (Axios)
- [x] WebSocket Client (real-time updates ready)
- [x] Responsive design with TailwindCSS
- [x] **Real-time chart integration** - KLineChart & Lightweight Charts
- [x] **Trading metrics display** - Live metrics, trades list, and performance stats

### Phase 3: Enhancement (📋 Future)
- [ ] Multiple timeframe support
- [ ] Advanced RL algorithms (PPO, SAC)
- [ ] Live trading integration
- [ ] Multi-exchange support
- [ ] Portfolio management
- [ ] Advanced risk management

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ⚠️ Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. The authors are not responsible for any financial losses incurred while using this software.

**DO NOT USE THIS FOR LIVE TRADING WITHOUT THOROUGH TESTING AND UNDERSTANDING OF THE RISKS INVOLVED.**

---

## 📞 Contact

- **GitHub**: [@GodEyeTee](https://github.com/GodEyeTee)
- **Issues**: [GitHub Issues](https://github.com/GodEyeTee/cryppo/issues)

---

**Built with ❤️ using PyTorch, FastAPI, and Next.js**
