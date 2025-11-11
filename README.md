# CRYPPO - Deep RL Crypto Trading System

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**CRYPPO** - Advanced Deep Reinforcement Learning Cryptocurrency Trading Platform

Full-Stack Application พร้อม FastAPI Backend, Next.js Frontend และ Multiple RL Models (D3QN, SAC, PPO, TD3)

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
3. Configure hyperparameters:
   - Model type (D3QN)
   - Learning rate
   - Training timesteps
   - Reward function (Sharpe, Sortino, etc.)
4. Click **Start Training**
5. Monitor training progress with real-time metrics

### 3. Backtest Model
1. Navigate to **Backtest** page
2. Select trained model from dropdown
3. Choose symbol and date range for testing
4. Click **Run Backtest**
5. View results:
   - Performance metrics (return, Sharpe, win rate)
   - Price chart with indicators
   - Trade history table

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

## 🛠️ Technical Details

### Backend (FastAPI)
- **Framework**: FastAPI 0.121+
- **WebSocket**: Real-time updates
- **Database**: SQLite (tracking & history)
- **ML Framework**: PyTorch 2.0+
- **RL Library**: Custom D3QN implementation

### Frontend (Next.js)
- **Framework**: Next.js 14.0.4
- **Language**: TypeScript 5+
- **UI**: TailwindCSS 3.3+
- **Charts**: KLineChart 9.8+ & Lightweight Charts 4.1+
- **State**: Zustand 4.4+
- **HTTP**: Axios 1.6+

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
