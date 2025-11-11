# RL Crypto Trading System - 1H Timeframe

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Deep Reinforcement Learning Cryptocurrency Trading System** - เน้นไทม์เฟรม 1 ชั่วโมง (1H) เท่านั้น

ระบบเทรดคริปโตด้วย Deep RL ที่ปรับโครงสร้างใหม่ให้เป็น Full-Stack Application พร้อม FastAPI Backend และ Next.js Frontend (Phase 2)

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
- **1H Timeframe Only** - Focused on 1-hour candlestick data
- **Data Management** - Download and prepare data from Binance/OKX
- **Model Training** - Deep Dueling DQN (D3QN) with PER
- **Backtesting** - Comprehensive backtesting engine
- **Real-time Updates** - WebSocket for progress monitoring

### Phase 2 (Planned - Frontend)
- **Next.js 14 Frontend** - Modern React-based web interface
- **Live Charts** - Real-time candlestick charts with KLineChart
- **Trading Dashboard** - Monitor training, backtesting, and paper trading
- **System Monitoring** - GPU status, data inventory, model management

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

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

### 3. Download Data
```bash
cd server
python download_data.py --symbols BTC/USDT ETH/USDT --start 2019-01-01 --end 2024-12-31
```

### 4. Start Backend Server
```bash
python main.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8000/ws

---

## 📂 Project Structure

```
rl-crypto-1h/
├── server/                     # Backend (FastAPI)
│   ├── main.py                # Main API server
│   ├── download_data.py       # Data download script
│   └── src/                   # Core modules
│       ├── agent/             # RL agents
│       ├── data/              # Data processing
│       ├── policy/            # Trading policies
│       ├── eval/              # Backtesting
│       ├── execution/         # Paper trading
│       ├── training/          # Training utilities
│       ├── utils/             # Helper functions
│       ├── database/          # Database schema
│       ├── configs/           # Configuration files
│       │   └── config.yaml    # Main config
│       └── legacy/            # Legacy code (for reference)
│
├── client/                    # Frontend (Phase 2 - Next.js)
│   └── src/
│       ├── app/               # Next.js pages
│       ├── components/        # React components
│       └── lib/               # API clients
│
├── data/                      # Data storage
│   └── npz/                   # Model files
│
├── requirements.txt           # Python dependencies
├── setup.sh                   # Setup script (Linux/macOS)
├── setup.bat                  # Setup script (Windows)
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

## 📖 Usage

### Download Data
```bash
cd server
python download_data.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --start 2019-01-01 \
  --end 2024-12-31 \
  --output ../data \
  --format npz
```

### Start API Server
```bash
cd server
python main.py
```

### API Endpoints

**Data Management:**
- `POST /api/data/download` - Download market data
- `GET /api/data/available` - List available data files

**Training:**
- `POST /api/train/start` - Start training
- `POST /api/train/pause/{task_id}` - Pause training
- `POST /api/train/resume/{task_id}` - Resume training
- `POST /api/train/cancel/{task_id}` - Cancel training
- `GET /api/models/available` - List trained models

**Backtesting:**
- `POST /api/backtest/run` - Run backtest
- `GET /api/backtest/results/{task_id}` - Get results

**System:**
- `GET /api/health` - Health check
- `GET /api/config` - Get configuration
- `GET /api/system/info` - System information

**WebSocket:**
- `ws://localhost:8000/ws` - Real-time updates

---

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Interactive API documentation with try-it-out functionality.

---

## 🗺️ Development Roadmap

### Phase 1: Backend (✅ Current)
- [x] FastAPI server structure
- [x] Configuration system
- [x] Data download script
- [x] API endpoints structure
- [x] WebSocket support
- [ ] Implement training logic
- [ ] Implement backtesting logic
- [ ] Database integration
- [ ] Complete agent implementation

### Phase 2: Frontend (🔄 Planned)
- [ ] Next.js 14 setup
- [ ] Dashboard layout
- [ ] Data management UI
- [ ] Training interface
- [ ] Backtest visualization
- [ ] Paper trading interface
- [ ] Real-time chart integration
- [ ] System monitoring page

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
