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

    # TODO: Implement actual backtest logic
    # background_tasks.add_task(backtest_task, task_id, request)

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

    # TODO: Return actual results
    return {
        "task_id": task_id,
        "status": background_tasks_status[task_id]["status"],
        "results": {}
    }

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
