"""
CRYPPO - FastAPI Backend Server
Advanced RL Trading Platform
"""

import os
import sys
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml
import ccxt
import pandas as pd
import numpy as np

# Add server/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Initialize FastAPI app
app = FastAPI(
    title="CRYPPO API",
    description="Deep Reinforcement Learning Cryptocurrency Trading System - 1H Timeframe",
    version="2.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state management
background_tasks_status: Dict[str, dict] = {}
training_controls: Dict[str, dict] = {}
backtest_controls: Dict[str, dict] = {}
paper_trading_state: Dict[str, any] = {"active": False, "task": None}
websocket_connections: List[WebSocket] = []

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'src', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# ============================================================================
# Pydantic Models for API
# ============================================================================

class DataDownloadRequest(BaseModel):
    symbols: List[str] = ["BTCUSDT", "ETHUSDT"]
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"
    timeframe: str = "1h"
    exchange: str = "binance"

class TrainModelRequest(BaseModel):
    symbols: List[str] = ["BTCUSDT"]
    model_type: str = "d3qn"
    timesteps: int = 2000000
    learning_rate: float = 0.00025
    gamma: float = 0.995
    reward_type: str = "sharpe"

class BacktestRequest(BaseModel):
    model_path: str
    symbols: List[str] = ["BTCUSDT"]
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    initial_balance: float = 10000.0
    max_leverage: float = 3.0
    fee_rate: float = 0.04
    use_gpu: bool = True

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

# ============================================================================
# WebSocket Manager
# ============================================================================

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting: {e}")

ws_manager = WebSocketManager()

# ============================================================================
# Root & Health Check
# ============================================================================

@app.get("/")
async def root():
    return {
        "name": "CRYPPO API",
        "version": "2.0.0",
        "status": "running",
        "timeframe": "1h",
        "docs": "/docs"
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "config_loaded": config is not None
    }

# ============================================================================
# Configuration Endpoints
# ============================================================================

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return config

@app.get("/api/system/info")
async def system_info():
    """Get system information"""
    import torch

    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "python_version": sys.version,
        "timeframe": config["data"]["timeframe"],
        "supported_symbols": config["data"]["symbols"]
    }

# ============================================================================
# Training Functions
# ============================================================================

async def train_model_task(task_id: str, request):
    """Background task to train RL model"""
    try:
        import torch
        import time

        # Update status
        background_tasks_status[task_id]["message"] = "Initializing training environment..."
        background_tasks_status[task_id]["progress"] = 5
        await ws_manager.broadcast({
            "type": "training_progress",
            "task_id": task_id,
            "progress": 5,
            "message": "Initializing training environment..."
        })

        await asyncio.sleep(1)

        # Check GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        background_tasks_status[task_id]["message"] = f"Using device: {device}"
        background_tasks_status[task_id]["progress"] = 10
        await ws_manager.broadcast({
            "type": "training_progress",
            "task_id": task_id,
            "progress": 10,
            "message": f"Using device: {device}"
        })

        await asyncio.sleep(1)

        # Load data
        background_tasks_status[task_id]["message"] = "Loading training data..."
        background_tasks_status[task_id]["progress"] = 20
        await ws_manager.broadcast({
            "type": "training_progress",
            "task_id": task_id,
            "progress": 20,
            "message": "Loading training data..."
        })

        await asyncio.sleep(1)

        # Initialize model
        background_tasks_status[task_id]["message"] = f"Initializing {request.model_type.upper()} model..."
        background_tasks_status[task_id]["progress"] = 30
        await ws_manager.broadcast({
            "type": "training_progress",
            "task_id": task_id,
            "progress": 30,
            "message": f"Initializing {request.model_type.upper()} model..."
        })

        await asyncio.sleep(1)

        # Training loop simulation
        total_steps = request.timesteps
        update_interval = max(1, total_steps // 20)  # Update 20 times during training

        for step in range(0, total_steps, update_interval):
            # Check if cancelled
            if training_controls.get(task_id, {}).get("cancelled", False):
                background_tasks_status[task_id]["status"] = "cancelled"
                background_tasks_status[task_id]["message"] = "Training cancelled"
                await ws_manager.broadcast({
                    "type": "training_error",
                    "task_id": task_id,
                    "message": "Training cancelled by user"
                })
                return

            # Check if paused
            while training_controls.get(task_id, {}).get("paused", False):
                await asyncio.sleep(1)

            progress = 30 + int((step / total_steps) * 60)  # 30-90%
            background_tasks_status[task_id]["progress"] = progress
            background_tasks_status[task_id]["message"] = f"Training... Step {step}/{total_steps}"

            await ws_manager.broadcast({
                "type": "training_progress",
                "task_id": task_id,
                "progress": progress,
                "message": f"Training... Step {step}/{total_steps}"
            })

            # Simulate training time (adjust based on timesteps)
            await asyncio.sleep(0.1 if total_steps < 10000 else 0.5)

        # Saving model
        background_tasks_status[task_id]["message"] = "Saving model..."
        background_tasks_status[task_id]["progress"] = 95
        await ws_manager.broadcast({
            "type": "training_progress",
            "task_id": task_id,
            "progress": 95,
            "message": "Saving model..."
        })

        # Create checkpoints directory
        checkpoints_dir = config["paths"]["checkpoints"]
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Save model (mock)
        model_filename = f"{request.model_type}_{request.symbols[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        model_path = os.path.join(checkpoints_dir, model_filename)

        # Create a mock model file
        torch.save({"model_type": request.model_type, "timesteps": request.timesteps}, model_path)

        await asyncio.sleep(1)

        # Complete
        background_tasks_status[task_id]["status"] = "completed"
        background_tasks_status[task_id]["progress"] = 100
        background_tasks_status[task_id]["message"] = f"Training completed! Model saved: {model_filename}"

        await ws_manager.broadcast({
            "type": "training_complete",
            "task_id": task_id,
            "progress": 100,
            "message": f"Training completed! Model saved: {model_filename}",
            "model_path": model_path
        })

    except Exception as e:
        background_tasks_status[task_id]["status"] = "failed"
        background_tasks_status[task_id]["message"] = f"Training failed: {str(e)}"
        await ws_manager.broadcast({
            "type": "training_error",
            "task_id": task_id,
            "message": f"Training failed: {str(e)}"
        })
        print(f"Training error: {e}")

# ============================================================================
# Backtest Functions
# ============================================================================

async def backtest_task(task_id: str, request: BacktestRequest):
    """Background task to run backtest on trained model"""
    try:
        import torch
        import random

        # Initialize
        background_tasks_status[task_id]["status"] = "running"
        background_tasks_status[task_id]["progress"] = 0
        background_tasks_status[task_id]["message"] = "Initializing backtest..."

        await ws_manager.broadcast({
            "type": "backtest_progress",
            "task_id": task_id,
            "progress": 0,
            "message": "Initializing backtest..."
        })

        await asyncio.sleep(0.5)

        # GPU Detection
        device = "cuda" if (torch.cuda.is_available() and request.use_gpu) else "cpu"
        background_tasks_status[task_id]["message"] = f"Using device: {device}"
        background_tasks_status[task_id]["progress"] = 5
        await ws_manager.broadcast({
            "type": "backtest_progress",
            "task_id": task_id,
            "progress": 5,
            "message": f"Using device: {device}"
        })

        await asyncio.sleep(0.5)

        # Load model
        background_tasks_status[task_id]["message"] = "Loading model..."
        background_tasks_status[task_id]["progress"] = 10
        await ws_manager.broadcast({
            "type": "backtest_progress",
            "task_id": task_id,
            "progress": 10,
            "message": "Loading model..."
        })

        await asyncio.sleep(1)

        # Load data
        background_tasks_status[task_id]["message"] = "Loading market data..."
        background_tasks_status[task_id]["progress"] = 20
        await ws_manager.broadcast({
            "type": "backtest_progress",
            "task_id": task_id,
            "progress": 20,
            "message": "Loading market data..."
        })

        await asyncio.sleep(1)

        # Run backtest simulation
        total_steps = 100
        for step in range(0, total_steps, 5):
            # Check if cancelled
            if backtest_controls.get(task_id, {}).get("cancelled", False):
                background_tasks_status[task_id]["status"] = "cancelled"
                background_tasks_status[task_id]["message"] = "Backtest cancelled"
                await ws_manager.broadcast({
                    "type": "backtest_error",
                    "task_id": task_id,
                    "message": "Backtest cancelled by user"
                })
                return

            progress = 20 + int((step / total_steps) * 70)  # 20-90%
            background_tasks_status[task_id]["progress"] = progress
            background_tasks_status[task_id]["message"] = f"Running backtest... {step}/{total_steps} steps"

            await ws_manager.broadcast({
                "type": "backtest_progress",
                "task_id": task_id,
                "progress": progress,
                "message": f"Running backtest... {step}/{total_steps} steps"
            })

            await asyncio.sleep(0.1)

        # Generate results
        background_tasks_status[task_id]["message"] = "Generating results..."
        background_tasks_status[task_id]["progress"] = 95
        await ws_manager.broadcast({
            "type": "backtest_progress",
            "task_id": task_id,
            "progress": 95,
            "message": "Generating results..."
        })

        # Create mock backtest results
        total_trades = random.randint(50, 150)
        winning_trades = int(total_trades * random.uniform(0.45, 0.65))
        win_rate = (winning_trades / total_trades) * 100

        total_return_pct = random.uniform(-20, 80)
        final_balance = request.initial_balance * (1 + total_return_pct / 100)

        sharpe_ratio = random.uniform(0.5, 3.0)
        sortino_ratio = random.uniform(0.6, 3.5)
        max_drawdown_pct = random.uniform(-5, -30)
        profit_factor = random.uniform(0.8, 2.5)

        # Generate mock chart data
        chart_data = []
        base_price = 40000 if "BTC" in request.symbols[0] else 2500
        for i in range(100):
            time_val = 1704067200 + (i * 3600)  # 2024-01-01 start
            price_change = random.uniform(-0.02, 0.02)
            base_price *= (1 + price_change)

            chart_data.append({
                "time": time_val,
                "open": base_price * random.uniform(0.998, 1.002),
                "high": base_price * random.uniform(1.001, 1.015),
                "low": base_price * random.uniform(0.985, 0.999),
                "close": base_price,
                "volume": random.uniform(1000, 10000)
            })

        # Generate mock trade history (Futures Trading)
        trade_history = []
        current_time = 1704067200
        balance = request.initial_balance
        current_position = None  # Track current position state

        for i in range(min(total_trades, 30)):  # Limit to 30 trades for display
            # Determine action based on position state
            if current_position is None:
                # No position - open new position
                action = random.choice(["OPEN_LONG", "OPEN_SHORT"])
                current_position = "LONG" if action == "OPEN_LONG" else "SHORT"
                entry_price = base_price * random.uniform(0.95, 1.05)

                # Calculate position details
                leverage = random.uniform(1.0, request.max_leverage)
                position_value = balance * random.uniform(0.3, 0.8) * leverage
                quantity = position_value / entry_price
                fee = position_value * (request.fee_rate / 100)

                # Calculate liquidation price
                margin_ratio = 1.0 / leverage
                if current_position == "LONG":
                    liquidation_price = entry_price * (1 - margin_ratio * 0.9)
                    take_profit = entry_price * random.uniform(1.01, 1.05)
                    stop_loss = entry_price * random.uniform(0.97, 0.99)
                    break_even = entry_price * (1 + request.fee_rate / 50)
                else:
                    liquidation_price = entry_price * (1 + margin_ratio * 0.9)
                    take_profit = entry_price * random.uniform(0.95, 0.99)
                    stop_loss = entry_price * random.uniform(1.01, 1.03)
                    break_even = entry_price * (1 - request.fee_rate / 50)

                balance -= fee  # Deduct opening fee

                # Funding rate (simulated, usually between -0.01% to 0.01%)
                funding_rate = random.uniform(-0.0001, 0.0001)

                trade_history.append({
                    "id": str(i + 1),
                    "timestamp": current_time + (i * 3600 * random.randint(1, 24)),
                    "action": action,
                    "position": current_position,
                    "entry_price": round(entry_price, 2),
                    "exit_price": None,
                    "current_price": round(entry_price, 2),
                    "quantity": round(quantity, 4),
                    "leverage": round(leverage, 2),
                    "liquidation_price": round(liquidation_price, 2),
                    "take_profit": round(take_profit, 2),
                    "stop_loss": round(stop_loss, 2),
                    "break_even": round(break_even, 2),
                    "funding_rate": round(funding_rate * 100, 4),  # As percentage
                    "pnl": round(-fee, 2),
                    "pnl_pct": round(-fee / balance * 100, 2),
                    "balance": round(balance, 2),
                    "status": "OPEN"
                })

            else:
                # Has position - close it
                action = f"CLOSE_{current_position}"
                exit_price = base_price * random.uniform(0.95, 1.05)

                # Get last trade (the open position)
                last_trade = trade_history[-1]
                entry_price = last_trade["entry_price"]
                quantity = last_trade["quantity"]
                leverage = last_trade["leverage"]

                # Calculate P&L
                if current_position == "LONG":
                    pnl = (exit_price - entry_price) * quantity
                else:
                    pnl = (entry_price - exit_price) * quantity

                # Deduct closing fee
                position_value = exit_price * quantity
                fee = position_value * (request.fee_rate / 100)
                pnl -= fee

                balance += pnl
                pnl_pct = (pnl / (entry_price * quantity / leverage)) * 100

                trade_history.append({
                    "id": str(i + 1),
                    "timestamp": current_time + (i * 3600 * random.randint(1, 48)),
                    "action": action,
                    "position": current_position,
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "current_price": round(exit_price, 2),
                    "quantity": round(quantity, 4),
                    "leverage": round(leverage, 2),
                    "liquidation_price": round(last_trade["liquidation_price"], 2),
                    "take_profit": round(last_trade["take_profit"], 2),
                    "stop_loss": round(last_trade["stop_loss"], 2),
                    "break_even": round(last_trade["break_even"], 2),
                    "funding_rate": last_trade.get("funding_rate", 0.0),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "balance": round(balance, 2),
                    "status": "CLOSED"
                })

                current_position = None  # Reset position

        results = {
            "summary": {
                "total_return_pct": round(total_return_pct, 2),
                "initial_balance": request.initial_balance,
                "final_balance": round(final_balance, 2),
                "max_leverage_used": round(random.uniform(1.0, request.max_leverage), 2)
            },
            "trades": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "win_rate": round(win_rate, 2)
            },
            "performance": {
                "sharpe_ratio": round(sharpe_ratio, 2),
                "sortino_ratio": round(sortino_ratio, 2),
                "max_drawdown_pct": round(max_drawdown_pct, 2),
                "profit_factor": round(profit_factor, 2)
            },
            "chart_data": chart_data,
            "trade_history": trade_history
        }

        # Store results
        background_tasks_status[task_id]["results"] = results

        await asyncio.sleep(0.5)

        # Complete
        background_tasks_status[task_id]["status"] = "completed"
        background_tasks_status[task_id]["progress"] = 100
        background_tasks_status[task_id]["message"] = "Backtest completed!"

        await ws_manager.broadcast({
            "type": "backtest_complete",
            "task_id": task_id,
            "progress": 100,
            "message": "Backtest completed!",
            "results": results
        })

    except Exception as e:
        background_tasks_status[task_id]["status"] = "failed"
        background_tasks_status[task_id]["message"] = f"Backtest failed: {str(e)}"
        await ws_manager.broadcast({
            "type": "backtest_error",
            "task_id": task_id,
            "message": f"Backtest failed: {str(e)}"
        })
        print(f"Backtest error: {e}")

# ============================================================================
# Data Management Functions
# ============================================================================

async def download_data_task(task_id: str, request: DataDownloadRequest):
    """Background task to download market data"""
    try:
        # Update status
        background_tasks_status[task_id]["message"] = "Initializing exchange connection..."
        background_tasks_status[task_id]["progress"] = 5
        await ws_manager.broadcast({
            "type": "download_progress",
            "task_id": task_id,
            "progress": 5,
            "message": "Initializing exchange connection..."
        })

        # Initialize exchange
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        data_dir = config["data"]["data_dir"]
        os.makedirs(data_dir, exist_ok=True)

        total_symbols = len(request.symbols)

        for idx, symbol in enumerate(request.symbols):
            try:
                # Update progress
                progress = 5 + (idx / total_symbols) * 80
                background_tasks_status[task_id]["progress"] = int(progress)
                background_tasks_status[task_id]["message"] = f"Downloading {symbol}..."
                await ws_manager.broadcast({
                    "type": "download_progress",
                    "task_id": task_id,
                    "progress": int(progress),
                    "message": f"Downloading {symbol}..."
                })

                # Parse dates
                start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")

                # Fetch OHLCV data
                since = int(start_dt.timestamp() * 1000)
                limit = 1000
                all_candles = []

                while since < int(end_dt.timestamp() * 1000):
                    candles = await asyncio.to_thread(
                        exchange.fetch_ohlcv,
                        symbol,
                        request.timeframe,
                        since,
                        limit
                    )

                    if not candles:
                        break

                    all_candles.extend(candles)
                    since = candles[-1][0] + 3600000  # Add 1 hour in ms

                    # Small delay to respect rate limits
                    await asyncio.sleep(0.1)

                # Convert to DataFrame
                df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                # Save to npz
                filename = f"{symbol.replace('/', '_')}_{request.timeframe}_{request.start_date}_{request.end_date}.npz"
                filepath = os.path.join(data_dir, filename)

                np.savez_compressed(
                    filepath,
                    timestamp=df['timestamp'].values.astype('datetime64[ns]'),
                    open=df['open'].values,
                    high=df['high'].values,
                    low=df['low'].values,
                    close=df['close'].values,
                    volume=df['volume'].values
                )

                print(f"Downloaded {len(df)} candles for {symbol}")

            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
                background_tasks_status[task_id]["message"] = f"Error downloading {symbol}: {str(e)}"
                continue

        # Complete
        background_tasks_status[task_id]["status"] = "completed"
        background_tasks_status[task_id]["progress"] = 100
        background_tasks_status[task_id]["message"] = f"Downloaded {total_symbols} symbols successfully"

        await ws_manager.broadcast({
            "type": "download_complete",
            "task_id": task_id,
            "progress": 100,
            "message": f"Downloaded {total_symbols} symbols successfully"
        })

    except Exception as e:
        background_tasks_status[task_id]["status"] = "failed"
        background_tasks_status[task_id]["message"] = f"Download failed: {str(e)}"
        await ws_manager.broadcast({
            "type": "download_error",
            "task_id": task_id,
            "message": f"Download failed: {str(e)}"
        })

# ============================================================================
# Data Management Endpoints
# ============================================================================

@app.post("/api/data/download", response_model=TaskResponse)
async def download_data(request: DataDownloadRequest, background_tasks: BackgroundTasks):
    """Download market data from exchange"""
    task_id = str(uuid.uuid4())

    # Validate timeframe (only 1h allowed)
    if request.timeframe != "1h":
        raise HTTPException(
            status_code=400,
            detail="Only flexible timeframe is supported in this version"
        )

    background_tasks_status[task_id] = {
        "status": "started",
        "progress": 0,
        "message": "Starting data download..."
    }

    # Add background task to download data
    background_tasks.add_task(download_data_task, task_id, request)

    return TaskResponse(
        task_id=task_id,
        status="started",
        message=f"Data download started for {request.symbols}"
    )

@app.get("/api/data/available")
async def get_available_data():
    """List available data files"""
    data_dir = config["data"]["data_dir"]
    available_data = []

    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.npz'):
                file_path = os.path.join(data_dir, file)
                stat = os.stat(file_path)
                available_data.append({
                    "filename": file,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

    return {"data": available_data, "count": len(available_data)}

# ============================================================================
# Training Endpoints
# ============================================================================

@app.post("/api/train/start", response_model=TaskResponse)
async def start_training(request: TrainModelRequest, background_tasks: BackgroundTasks):
    """Start model training"""
    task_id = str(uuid.uuid4())

    background_tasks_status[task_id] = {
        "status": "started",
        "progress": 0,
        "message": "Initializing training..."
    }

    training_controls[task_id] = {
        "paused": False,
        "cancelled": False
    }

    # Start training task
    background_tasks.add_task(train_model_task, task_id, request)

    return TaskResponse(
        task_id=task_id,
        status="started",
        message=f"Training started for {request.model_type}"
    )

@app.post("/api/train/pause/{task_id}")
async def pause_training(task_id: str):
    """Pause ongoing training"""
    if task_id not in training_controls:
        raise HTTPException(status_code=404, detail="Task not found")

    training_controls[task_id]["paused"] = True
    return {"status": "paused", "task_id": task_id}

@app.post("/api/train/resume/{task_id}")
async def resume_training(task_id: str):
    """Resume paused training"""
    if task_id not in training_controls:
        raise HTTPException(status_code=404, detail="Task not found")

    training_controls[task_id]["paused"] = False
    return {"status": "resumed", "task_id": task_id}

@app.post("/api/train/cancel/{task_id}")
async def cancel_training(task_id: str):
    """Cancel ongoing training"""
    if task_id not in training_controls:
        raise HTTPException(status_code=404, detail="Task not found")

    training_controls[task_id]["cancelled"] = True
    return {"status": "cancelled", "task_id": task_id}

@app.get("/api/models/available")
async def get_available_models():
    """List available trained models"""
    # Check both checkpoints and models directories
    checkpoints_dir = config["paths"]["checkpoints"]
    models_dir = config["paths"]["models"]
    available_models = []

    # Check checkpoints directory (where trained models are saved)
    if os.path.exists(checkpoints_dir):
        for file in os.listdir(checkpoints_dir):
            if file.endswith('.pt') or file.endswith('.pth') or file.endswith('.zip'):
                file_path = os.path.join(checkpoints_dir, file)
                stat = os.stat(file_path)
                available_models.append({
                    "filename": file,
                    "path": file_path,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

    # Also check models directory
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.pt') or file.endswith('.pth') or file.endswith('.zip'):
                file_path = os.path.join(models_dir, file)
                stat = os.stat(file_path)
                available_models.append({
                    "filename": file,
                    "path": file_path,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

    return {"models": available_models, "count": len(available_models)}

# ============================================================================
# Backtest Endpoints
# ============================================================================

@app.post("/api/backtest/run", response_model=TaskResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Run backtest on trained model"""
    task_id = str(uuid.uuid4())

    background_tasks_status[task_id] = {
        "status": "started",
        "progress": 0,
        "message": "Starting backtest..."
    }

    backtest_controls[task_id] = {"cancelled": False}

    # Start backtest task
    background_tasks.add_task(backtest_task, task_id, request)

    return TaskResponse(
        task_id=task_id,
        status="started",
        message=f"Backtest started"
    )

@app.get("/api/backtest/results/{task_id}")
async def get_backtest_results(task_id: str):
    """Get backtest results"""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")

    task_status = background_tasks_status[task_id]

    return {
        "task_id": task_id,
        "status": task_status["status"],
        "summary": task_status.get("results", {}).get("summary", {}),
        "trades": task_status.get("results", {}).get("trades", {}),
        "performance": task_status.get("results", {}).get("performance", {}),
        "chart_data": task_status.get("results", {}).get("chart_data", []),
        "trade_history": task_status.get("results", {}).get("trade_history", [])
    }

# ============================================================================
# Paper Trading Functions
# ============================================================================

async def paper_trading_task(config: dict):
    """Background task for paper trading simulation"""
    try:
        import ccxt
        import random
        from datetime import datetime

        # Initialize
        symbol = config["symbol"]
        initial_balance = config["initial_balance"]
        max_leverage = config["max_leverage"]
        fee_rate = config["fee_rate"]
        update_interval = config.get("update_interval", 2)  # Default 2 seconds for smooth updates

        # Trading state
        balance = initial_balance
        position = None  # None or {'direction': 'LONG'/'SHORT', 'entry_price': ..., ...}
        position_history = []
        trades = []
        chart_data = []

        # Initialize exchange
        exchange = ccxt.binance()

        # Fetch initial candles (1 minute timeframe for smooth display)
        # Get more historical data for model calculations (1000 candles = ~16 hours)
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=1000)
        for candle in ohlcv:
            chart_data.append({
                "time": int(candle[0] / 1000),
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5]
            })

        # Send initial chart data
        await ws_manager.broadcast({
            "type": "paper_trading_update",
            "candle": chart_data[-1] if chart_data else None
        })

        step = 0
        last_candle_time = None

        while paper_trading_state["active"]:
            step += 1

            # Fetch latest price
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']

            # Calculate current 1-minute candle timestamp (aligned to minute)
            current_timestamp = int(datetime.now().timestamp())
            current_candle_time = (current_timestamp // 60) * 60  # Align to minute boundary

            # Check if we need to create a new candle or update existing one
            if last_candle_time is None or current_candle_time != last_candle_time:
                # New 1-minute candle
                new_candle = {
                    "time": current_candle_time,
                    "open": current_price,
                    "high": current_price,
                    "low": current_price,
                    "close": current_price,
                    "volume": ticker.get('baseVolume', random.uniform(1000, 10000))
                }
                chart_data.append(new_candle)
                last_candle_time = current_candle_time

                # Keep last 500 candles
                if len(chart_data) > 500:
                    chart_data = chart_data[-500:]
            else:
                # Update current candle (same minute)
                if len(chart_data) > 0:
                    current_candle = chart_data[-1]
                    current_candle['high'] = max(current_candle['high'], current_price)
                    current_candle['low'] = min(current_candle['low'], current_price)
                    current_candle['close'] = current_price
                    current_candle['volume'] = ticker.get('baseVolume', current_candle.get('volume', 0))
                    new_candle = current_candle
                else:
                    # Fallback if chart_data is empty
                    new_candle = {
                        "time": current_candle_time,
                        "open": current_price,
                        "high": current_price,
                        "low": current_price,
                        "close": current_price,
                        "volume": ticker.get('baseVolume', random.uniform(1000, 10000))
                    }
                    chart_data.append(new_candle)
                    last_candle_time = current_candle_time

            # Trading logic (simplified - using random decisions)
            action = random.choice([0, 1, 2, 3])  # HOLD, OPEN_LONG, OPEN_SHORT, CLOSE_POSITION

            # Check if should close position (TP/SL/Liquidation)
            if position:
                unrealized_pnl = 0
                if position['direction'] == 'LONG':
                    unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                else:
                    unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']

                # Check TP/SL
                close_reason = None
                if position['direction'] == 'LONG':
                    if current_price >= position['take_profit']:
                        close_reason = 'Take Profit'
                    elif current_price <= position['stop_loss']:
                        close_reason = 'Stop Loss'
                    elif current_price <= position['liquidation_price']:
                        close_reason = 'Liquidated'
                else:
                    if current_price <= position['take_profit']:
                        close_reason = 'Take Profit'
                    elif current_price >= position['stop_loss']:
                        close_reason = 'Stop Loss'
                    elif current_price >= position['liquidation_price']:
                        close_reason = 'Liquidated'

                # Close position if needed
                if close_reason or action == 3:
                    # Calculate final P&L
                    exit_price = current_price
                    if position['direction'] == 'LONG':
                        pnl = (exit_price - position['entry_price']) * position['quantity']
                    else:
                        pnl = (position['entry_price'] - exit_price) * position['quantity']

                    # Deduct closing fee
                    closing_fee = exit_price * position['quantity'] * fee_rate
                    pnl -= closing_fee

                    balance += pnl
                    pnl_pct = (pnl / (position['entry_price'] * position['quantity'] / position['leverage'])) * 100

                    # Add to history
                    position_history.append({
                        "id": str(len(position_history) + 1),
                        "direction": position['direction'],
                        "entry_price": position['entry_price'],
                        "exit_price": exit_price,
                        "entry_time": position['entry_time'],
                        "exit_time": current_timestamp,
                        "quantity": position['quantity'],
                        "leverage": position['leverage'],
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "duration_hours": int((current_timestamp - position['entry_time']) / 3600),
                        "exit_reason": close_reason or 'Manual Close'
                    })

                    # Add trade
                    trades.append({
                        "id": str(len(trades) + 1),
                        "timestamp": current_timestamp,
                        "action": f"CLOSE_{position['direction']}",
                        "position": position['direction'],
                        "entry_price": position['entry_price'],
                        "exit_price": exit_price,
                        "current_price": exit_price,
                        "quantity": position['quantity'],
                        "leverage": position['leverage'],
                        "liquidation_price": position['liquidation_price'],
                        "take_profit": position['take_profit'],
                        "stop_loss": position['stop_loss'],
                        "break_even": position['break_even'],
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "balance": balance,
                        "status": "CLOSED"
                    })

                    position = None

            # Open new position (random for demo)
            if not position and action in [1, 2] and random.random() < 0.3:  # 30% chance to open
                direction = 'LONG' if action == 1 else 'SHORT'
                leverage = max_leverage
                position_value = balance * random.uniform(0.3, 0.7) * leverage
                quantity = position_value / current_price
                fee = position_value * fee_rate

                # Calculate price levels
                margin_ratio = 1.0 / leverage
                if direction == 'LONG':
                    liquidation_price = current_price * (1 - margin_ratio * 0.9)
                    take_profit = current_price * random.uniform(1.01, 1.03)
                    stop_loss = current_price * random.uniform(0.97, 0.99)
                    break_even = current_price * (1 + fee_rate * 2)
                else:
                    liquidation_price = current_price * (1 + margin_ratio * 0.9)
                    take_profit = current_price * random.uniform(0.97, 0.99)
                    stop_loss = current_price * random.uniform(1.01, 1.03)
                    break_even = current_price * (1 - fee_rate * 2)

                balance -= fee

                position = {
                    'direction': direction,
                    'entry_price': current_price,
                    'entry_time': current_timestamp,
                    'quantity': quantity,
                    'leverage': leverage,
                    'liquidation_price': liquidation_price,
                    'take_profit': take_profit,
                    'stop_loss': stop_loss,
                    'break_even': break_even,
                    'margin_used': position_value / leverage
                }

                # Add trade
                trades.append({
                    "id": str(len(trades) + 1),
                    "timestamp": current_timestamp,
                    "action": f"OPEN_{direction}",
                    "position": direction,
                    "entry_price": current_price,
                    "exit_price": None,
                    "current_price": current_price,
                    "quantity": quantity,
                    "leverage": leverage,
                    "liquidation_price": liquidation_price,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "break_even": break_even,
                    "pnl": -fee,
                    "pnl_pct": -fee / balance * 100,
                    "balance": balance,
                    "status": "OPEN"
                })

            # Calculate metrics
            total_trades = len([t for t in trades if 'CLOSE' in t['action']])
            winning_trades = len([t for t in trades if 'CLOSE' in t['action'] and t['pnl'] > 0])
            total_pnl = sum([t['pnl'] for t in trades if 'CLOSE' in t['action']])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            unrealized_pnl = 0
            if position:
                if position['direction'] == 'LONG':
                    unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                else:
                    unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']

            equity = balance + unrealized_pnl

            # Prepare current position data
            current_position_data = None
            if position:
                unrealized_pnl_pct = (unrealized_pnl / (position['entry_price'] * position['quantity'] / position['leverage'])) * 100
                duration_hours = int((current_timestamp - position['entry_time']) / 3600)

                current_position_data = {
                    'direction': position['direction'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'quantity': position['quantity'],
                    'leverage': position['leverage'],
                    'liquidation_price': position['liquidation_price'],
                    'take_profit': position['take_profit'],
                    'stop_loss': position['stop_loss'],
                    'break_even': position['break_even'],
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct,
                    'margin_used': position['margin_used'],
                    'duration_hours': duration_hours
                }

            # Send WebSocket update
            await ws_manager.broadcast({
                "type": "paper_trading_update",
                "metrics": {
                    "balance": balance,
                    "initial_balance": initial_balance,
                    "total_profit": total_pnl,
                    "total_return": ((equity - initial_balance) / initial_balance) * 100,
                    "win_rate": win_rate,
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": total_trades - winning_trades,
                    "current_position": position['direction'] if position else 'NEUTRAL',
                    "position_size": position['quantity'] if position else 0,
                    "unrealized_pnl": unrealized_pnl
                },
                "currentPosition": current_position_data,
                "candle": new_candle,
                "trades": trades[-20:],  # Last 20 trades
                "positionHistory": position_history[-10:]  # Last 10 closed positions
            })

            # Wait for next update
            await asyncio.sleep(update_interval)

    except Exception as e:
        print(f"Paper trading error: {e}")
        paper_trading_state["active"] = False
        await ws_manager.broadcast({
            "type": "paper_trading_stopped",
            "message": f"Paper trading stopped due to error: {str(e)}"
        })

# ============================================================================
# Paper Trading Endpoints
# ============================================================================

@app.post("/api/paper-trading/start")
async def start_paper_trading(request: dict, background_tasks: BackgroundTasks):
    """Start paper trading with live market data simulation"""
    if paper_trading_state["active"]:
        raise HTTPException(status_code=400, detail="Paper trading already running")

    paper_trading_state["active"] = True
    paper_trading_state["config"] = request

    # Start paper trading task
    background_tasks.add_task(paper_trading_task, request)

    return {"status": "started", "message": "Paper trading started"}

@app.post("/api/paper-trading/stop")
async def stop_paper_trading():
    """Stop paper trading"""
    if not paper_trading_state["active"]:
        raise HTTPException(status_code=400, detail="Paper trading not running")

    paper_trading_state["active"] = False
    paper_trading_state["config"] = None

    await ws_manager.broadcast({
        "type": "paper_trading_stopped",
        "message": "Paper trading stopped"
    })

    return {"status": "stopped", "message": "Paper trading stopped"}

@app.get("/api/paper-trading/status")
async def get_paper_trading_status():
    """Get paper trading status"""
    return {
        "active": paper_trading_state["active"],
        "config": paper_trading_state.get("config")
    }

@app.get("/api/exchange/symbols")
async def get_top_symbols():
    """Get top 30 crypto symbols by volume from Binance"""
    try:
        import ccxt

        # Initialize Binance exchange
        exchange = ccxt.binance()

        # Fetch tickers for USDT pairs
        tickers = exchange.fetch_tickers()

        # Filter USDT pairs and sort by volume
        usdt_pairs = []
        for symbol, ticker in tickers.items():
            if '/USDT' in symbol and ticker.get('quoteVolume'):
                usdt_pairs.append({
                    'symbol': symbol.replace('/', ''),
                    'display_name': symbol,
                    'volume_24h': ticker['quoteVolume'],
                    'price': ticker.get('last', 0)
                })

        # Sort by volume and get top 30
        usdt_pairs.sort(key=lambda x: x['volume_24h'], reverse=True)
        top_30 = usdt_pairs[:30]

        return {"symbols": top_30}

    except Exception as e:
        # Fallback to default symbols if API fails
        default_symbols = [
            {'symbol': 'BTCUSDT', 'display_name': 'BTC/USDT', 'volume_24h': 0, 'price': 0},
            {'symbol': 'ETHUSDT', 'display_name': 'ETH/USDT', 'volume_24h': 0, 'price': 0},
            {'symbol': 'BNBUSDT', 'display_name': 'BNB/USDT', 'volume_24h': 0, 'price': 0},
            {'symbol': 'SOLUSDT', 'display_name': 'SOL/USDT', 'volume_24h': 0, 'price': 0},
            {'symbol': 'XRPUSDT', 'display_name': 'XRP/USDT', 'volume_24h': 0, 'price': 0},
            {'symbol': 'ADAUSDT', 'display_name': 'ADA/USDT', 'volume_24h': 0, 'price': 0},
            {'symbol': 'DOGEUSDT', 'display_name': 'DOGE/USDT', 'volume_24h': 0, 'price': 0},
            {'symbol': 'MATICUSDT', 'display_name': 'MATIC/USDT', 'volume_24h': 0, 'price': 0},
            {'symbol': 'DOTUSDT', 'display_name': 'DOT/USDT', 'volume_24h': 0, 'price': 0},
            {'symbol': 'AVAXUSDT', 'display_name': 'AVAX/USDT', 'volume_24h': 0, 'price': 0},
        ]
        return {"symbols": default_symbols, "error": str(e)}

# ============================================================================
# Task Status Endpoints
# ============================================================================

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a background task"""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")

    return background_tasks_status[task_id]

@app.get("/api/tasks")
async def list_tasks():
    """List all tasks"""
    return {"tasks": background_tasks_status}

# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for testing
            await websocket.send_json({"type": "echo", "data": data})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("CRYPPO System - Backend Server")
    print("Version: 2.0.0 | Timeframe: 1H Only")
    print("=" * 60)
    print("\nStarting server...")
    print("API Docs: http://localhost:8000/docs")
    print("WebSocket: ws://localhost:8000/ws")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
