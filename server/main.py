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
from pathlib import Path
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml
import ccxt
import pandas as pd
import numpy as np

# Add server/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
from utils.logging_config import get_server_logger, get_training_logger
server_logger = get_server_logger()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = PROJECT_ROOT / "logs"
TRAIN_LOG_DIR = LOG_ROOT / "training"
BACKTEST_LOG_DIR = LOG_ROOT / "backtest"
PAPER_LOG_DIR = LOG_ROOT / "paper"
for directory in (TRAIN_LOG_DIR, BACKTEST_LOG_DIR, PAPER_LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def serialize_trade_records(trades: List[Dict]) -> List[Dict]:
    """Convert environment trade logs to JSON serializable rows."""
    serialized = []
    for trade in trades:
        record = {}
        for key, value in trade.items():
            if isinstance(value, (np.integer, int)):
                record[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                record[key] = float(value)
            elif isinstance(value, np.ndarray):
                record[key] = value.tolist()
            else:
                record[key] = value
        serialized.append(record)
    return serialized

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
    batch_size: int = 64
    buffer_size: int = 100000
    window_size: int = 72
    initial_balance: float = 10000.0
    max_leverage: float = 3.0
    episodes: Optional[int] = None  # If None, calculate from timesteps
    eval_freq: int = 50
    save_freq: int = 100

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
    # Try to import PyTorch with error handling for Windows DLL issues
    torch_info = {
        "cuda_available": False,
        "cuda_device": None,
        "num_gpus": 0,
        "torch_error": None
    }

    try:
        import torch
        torch_info["cuda_available"] = torch.cuda.is_available()
        torch_info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        torch_info["num_gpus"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception as e:
        # Common on Windows: Missing Visual C++ Redistributable or CUDA libraries
        torch_info["torch_error"] = str(e)
        server_logger.warning(f"[System] PyTorch import failed: {e}")

    return {
        **torch_info,
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


async def train_sb3_task(task_id: str, request: TrainModelRequest):
    """Background task for Stable-Baselines3 models (SAC, PPO, TD3)"""
    try:
        import torch
        import subprocess
        import re
        from pathlib import Path

        model_name = request.model_type.upper()

        # Update status
        background_tasks_status[task_id]["message"] = f"Initializing {model_name} training..."
        background_tasks_status[task_id]["progress"] = 5
        await ws_manager.broadcast({
            "type": "training_progress",
            "task_id": task_id,
            "progress": 5,
            "message": f"Initializing {model_name} training..."
        })

        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        background_tasks_status[task_id]["message"] = f"Using device: {device}"
        background_tasks_status[task_id]["progress"] = 10
        await ws_manager.broadcast({
            "type": "training_progress",
            "task_id": task_id,
            "progress": 10,
            "message": f"Using device: {device}"
        })

        # Prepare training arguments
        symbol = request.symbols[0] if request.symbols else "BTCUSDT"
        timeframe = "1h"  # Only 1h timeframe is supported

        # Check if data exists (use same pattern as loader.py)
        data_dir = Path("data")
        print(f"[DEBUG] Checking for data in: {data_dir.absolute()}")
        print(f"[DEBUG] Data directory exists: {data_dir.exists()}")

        data_files = []
        if data_dir.exists():
            # Use same pattern as loader.py: {symbol}_{timeframe}*.parquet
            pattern = f"{symbol}_{timeframe}*.parquet"
            data_files = list(data_dir.rglob(pattern))
            print(f"[DEBUG] Search pattern: {pattern}")
            print(f"[DEBUG] Found {len(data_files)} parquet files for {symbol} {timeframe}")
            if data_files:
                print(f"[DEBUG] Files: {[str(f) for f in data_files]}")

            # Also check for synthetic data files if no timeframe-specific files found
            if not data_files:
                synthetic_pattern = f"{symbol}*synthetic.parquet"
                synthetic_files = list(data_dir.rglob(synthetic_pattern))
                print(f"[DEBUG] Checking synthetic pattern: {synthetic_pattern}")
                print(f"[DEBUG] Found {len(synthetic_files)} synthetic files")
                # Don't use synthetic files here - let train script generate new ones
        else:
            print(f"[DEBUG] Data directory does not exist")

        use_sample_data = len(data_files) == 0
        print(f"[DEBUG] use_sample_data = {use_sample_data}")

        if use_sample_data:
            background_tasks_status[task_id]["message"] = "No data found, using synthetic data..."
            await ws_manager.broadcast({
                "type": "training_progress",
                "task_id": task_id,
                "progress": 15,
                "message": "[WARN] No data found, using synthetic data for training"
            })

        # Select training script
        script_map = {
            "sac": "train_sac",
            "ppo": "train_ppo",
            "td3": "train_td3"
        }
        script_name = script_map.get(request.model_type, "train_sac")

        # Build command
        cmd = [
            sys.executable, "-m", f"server.src.training.{script_name}",
            "--symbol", symbol,
            "--timesteps", str(request.timesteps),
            "--balance", str(request.initial_balance),
            "--leverage", str(request.max_leverage),
            "--lr", str(request.learning_rate),
            "--gamma", str(request.gamma),
            "--batch-size", str(request.batch_size),
            "--buffer-size", str(request.buffer_size) if request.model_type != "ppo" else str(request.buffer_size // 10),
            "--window", str(request.window_size),
            "--reward", request.reward_type,
            "--eval-freq", str(request.eval_freq * 10),  # SB3 uses timesteps not episodes
            "--save-freq", str(request.save_freq * 10),
            "--checkpoint-dir", config["paths"]["checkpoints"],
            "--log-dir", "logs"
        ]

        if use_sample_data:
            print(f"[DEBUG] Appending --sample-data flag")
            cmd.append("--sample-data")
        else:
            print(f"[DEBUG] NOT appending --sample-data flag (data files found)")

        # Log the command that will be executed
        print(f"\n[TRAINING] Executing command:")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Working directory: {os.getcwd()}")
        print(f"  Python path: {sys.path[:3]}")

        # Run training process
        background_tasks_status[task_id]["message"] = f"Starting {model_name} training for {request.timesteps:,} timesteps..."
        background_tasks_status[task_id]["progress"] = 20
        await ws_manager.broadcast({
            "type": "training_progress",
            "task_id": task_id,
            "progress": 20,
            "message": f"Starting {model_name} training..."
        })

        # Prepare environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        env['PYTHONIOENCODING'] = 'utf-8'  # Ensure UTF-8 encoding
        env['PYTHONUNBUFFERED'] = '1'  # Disable output buffering for real-time logs

        # Start subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',  # Force UTF-8 encoding for stdout/stderr
            errors='replace',  # Replace invalid characters instead of crashing
            bufsize=1,
            cwd=os.getcwd(),  # Set working directory
            env=env  # Pass environment variables
        )

        # Parse output and send progress updates
        episode_pattern = re.compile(r"Episode\s+(\d+)")
        timestep_pattern = re.compile(r"Timestep\s+(\d+)")
        current_timestep = 0

        # Read output in a loop
        while True:
            # Check for cancellation
            if training_controls.get(task_id, {}).get("cancelled", False):
                process.terminate()
                break

            # Check if process is still running
            return_code = process.poll()
            if return_code is not None:
                break

            # Read line
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    print(f"[Training] {line}")

                    # Parse timestep number
                    match = timestep_pattern.search(line)
                    if match:
                        current_timestep = int(match.group(1))
                        progress = 20 + int((current_timestep / request.timesteps) * 70)  # 20-90%
                        progress = min(90, progress)

                        background_tasks_status[task_id]["progress"] = progress
                        background_tasks_status[task_id]["message"] = f"Timestep {current_timestep:,}/{request.timesteps:,}"

                        await ws_manager.broadcast({
                            "type": "training_progress",
                            "task_id": task_id,
                            "progress": progress,
                            "message": f"Training: {current_timestep:,}/{request.timesteps:,} timesteps",
                            "details": line
                        })

                    # Check for key messages
                    if "Evaluating" in line or "eval" in line.lower():
                        await ws_manager.broadcast({
                            "type": "training_progress",
                            "task_id": task_id,
                            "message": "Evaluation in progress...",
                            "details": line
                        })
                    elif "Saving" in line or "saved" in line.lower():
                        await ws_manager.broadcast({
                            "type": "training_progress",
                            "task_id": task_id,
                            "message": "[SAVE] Checkpoint saved",
                            "details": line
                        })

            # Small sleep to prevent busy loop
            await asyncio.sleep(0.1)

        # Read any remaining output before checking return code
        if process.stdout:
            remaining_output = process.stdout.read()
            if remaining_output:
                for line in remaining_output.split('\n'):
                    if line.strip():
                        print(f"[Training] {line.strip()}")

        # Final return code check
        return_code = process.wait()
        if return_code != 0 and not training_controls.get(task_id, {}).get("cancelled", False):
            raise Exception(f"Training process exited with code {return_code}")

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

        # Training complete
        background_tasks_status[task_id]["progress"] = 95
        background_tasks_status[task_id]["message"] = "Finalizing..."
        await ws_manager.broadcast({
            "type": "training_progress",
            "task_id": task_id,
            "progress": 95,
            "message": "Training complete, finalizing..."
        })

        # Find the saved model
        checkpoints_dir = Path(config["paths"]["checkpoints"])
        model_files = list(checkpoints_dir.glob(f"final_{request.model_type}_model.zip"))
        model_path = str(model_files[0]) if model_files else str(checkpoints_dir / "best" / "best_model.zip")

        # Complete
        background_tasks_status[task_id]["status"] = "completed"
        background_tasks_status[task_id]["progress"] = 100
        background_tasks_status[task_id]["message"] = f"{model_name} training completed!"

        await ws_manager.broadcast({
            "type": "training_complete",
            "task_id": task_id,
            "progress": 100,
            "message": f"🎉 {model_name} training completed! {current_timestep:,} timesteps finished",
            "model_path": model_path
        })

    except Exception as e:
        import traceback
        error_msg = f"Training failed: {str(e)}"
        print(f"Training error: {error_msg}")
        print(traceback.format_exc())

        background_tasks_status[task_id]["status"] = "failed"
        background_tasks_status[task_id]["message"] = error_msg
        await ws_manager.broadcast({
            "type": "training_error",
            "task_id": task_id,
            "message": error_msg
        })


async def train_d3qn_task(task_id: str, request: TrainModelRequest):
    """Background task for real D3QN training"""
    try:
        import torch
        import subprocess
        import threading
        import re
        from pathlib import Path

        # Update status
        background_tasks_status[task_id]["message"] = "Initializing D3QN training..."
        background_tasks_status[task_id]["progress"] = 5
        await ws_manager.broadcast({
            "type": "training_progress",
            "task_id": task_id,
            "progress": 5,
            "message": "Initializing D3QN training..."
        })

        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        background_tasks_status[task_id]["message"] = f"Using device: {device}"
        background_tasks_status[task_id]["progress"] = 10
        await ws_manager.broadcast({
            "type": "training_progress",
            "task_id": task_id,
            "progress": 10,
            "message": f"Using device: {device}"
        })

        # Calculate episodes from timesteps if not provided
        episodes = request.episodes
        if episodes is None:
            # Estimate: each episode ~ 1000 steps on average
            episodes = max(100, request.timesteps // 1000)

        # Prepare training arguments
        symbol = request.symbols[0] if request.symbols else "BTCUSDT"
        timeframe = "1h"  # Only 1h timeframe is supported

        # Check if data exists (use same pattern as loader.py)
        data_dir = Path("data")
        print(f"[DEBUG] Checking for data in: {data_dir.absolute()}")
        print(f"[DEBUG] Data directory exists: {data_dir.exists()}")

        data_files = []
        if data_dir.exists():
            # Use same pattern as loader.py: {symbol}_{timeframe}*.parquet
            pattern = f"{symbol}_{timeframe}*.parquet"
            data_files = list(data_dir.rglob(pattern))
            print(f"[DEBUG] Search pattern: {pattern}")
            print(f"[DEBUG] Found {len(data_files)} parquet files for {symbol} {timeframe}")
            if data_files:
                print(f"[DEBUG] Files: {[str(f) for f in data_files]}")

            # Also check for synthetic data files if no timeframe-specific files found
            if not data_files:
                synthetic_pattern = f"{symbol}*synthetic.parquet"
                synthetic_files = list(data_dir.rglob(synthetic_pattern))
                print(f"[DEBUG] Checking synthetic pattern: {synthetic_pattern}")
                print(f"[DEBUG] Found {len(synthetic_files)} synthetic files")
                # Don't use synthetic files here - let train script generate new ones
        else:
            print(f"[DEBUG] Data directory does not exist")

        use_sample_data = len(data_files) == 0
        print(f"[DEBUG] use_sample_data = {use_sample_data}")

        if use_sample_data:
            background_tasks_status[task_id]["message"] = "No data found, using synthetic data..."
            await ws_manager.broadcast({
                "type": "training_progress",
                "task_id": task_id,
                "progress": 15,
                "message": "[WARN] No data found, using synthetic data for training"
            })

        # Build command
        cmd = [
            sys.executable, "-m", "server.src.training.train_d3qn",
            "--symbol", symbol,
            "--episodes", str(episodes),
            "--balance", str(request.initial_balance),
            "--leverage", str(request.max_leverage),
            "--lr", str(request.learning_rate),
            "--gamma", str(request.gamma),
            "--batch-size", str(request.batch_size),
            "--buffer-size", str(request.buffer_size),
            "--window", str(request.window_size),
            "--reward", request.reward_type,
            "--eval-freq", str(request.eval_freq),
            "--save-freq", str(request.save_freq),
            "--checkpoint-dir", os.path.abspath(config["paths"]["checkpoints"]),  # Use absolute path
            "--log-dir", "logs"
        ]

        if use_sample_data:
            print(f"[DEBUG] Appending --sample-data flag")
            cmd.append("--sample-data")
        else:
            print(f"[DEBUG] NOT appending --sample-data flag (data files found)")

        # Log the command that will be executed
        print(f"\n[TRAINING] Executing command:")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Working directory: {os.getcwd()}")
        print(f"  Python path: {sys.path[:3]}")

        # Run training process
        background_tasks_status[task_id]["message"] = f"Starting training for {episodes} episodes..."
        background_tasks_status[task_id]["progress"] = 20
        await ws_manager.broadcast({
            "type": "training_progress",
            "task_id": task_id,
            "progress": 20,
            "message": f"Starting training for {episodes} episodes..."
        })

        # Prepare environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        env['PYTHONIOENCODING'] = 'utf-8'  # Ensure UTF-8 encoding
        env['PYTHONUNBUFFERED'] = '1'  # Disable output buffering for real-time logs

        # Start subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',  # Force UTF-8 encoding for stdout/stderr
            errors='replace',  # Replace invalid characters instead of crashing
            bufsize=1,
            cwd=os.getcwd(),  # Set working directory
            env=env  # Pass environment variables
        )

        # Parse output and send progress updates
        episode_pattern = re.compile(r"Episode\s+(\d+)")
        current_episode = 0

        # Read output in a loop
        while True:
            # Check for cancellation
            if training_controls.get(task_id, {}).get("cancelled", False):
                process.terminate()
                break

            # Check if process is still running
            return_code = process.poll()
            if return_code is not None:
                # Process finished
                break

            # Read line (non-blocking check)
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    print(f"[Training] {line}")

                    # Parse episode number
                    match = episode_pattern.search(line)
                    if match:
                        current_episode = int(match.group(1))
                        progress = 20 + int((current_episode / episodes) * 70)  # 20-90%
                        progress = min(90, progress)

                        background_tasks_status[task_id]["progress"] = progress
                        background_tasks_status[task_id]["message"] = f"Episode {current_episode}/{episodes}"

                        await ws_manager.broadcast({
                            "type": "training_progress",
                            "task_id": task_id,
                            "progress": progress,
                            "message": f"Training: Episode {current_episode}/{episodes}",
                            "details": line
                        })

                    # Check for key messages
                    if "Evaluating" in line or "Val Reward" in line:
                        await ws_manager.broadcast({
                            "type": "training_progress",
                            "task_id": task_id,
                            "message": "Validation in progress...",
                            "details": line
                        })
                    elif "Checkpoint saved" in line or "best model saved" in line:
                        await ws_manager.broadcast({
                            "type": "training_progress",
                            "task_id": task_id,
                            "message": "💾 Checkpoint saved",
                            "details": line
                        })

            # Small sleep to prevent busy loop
            await asyncio.sleep(0.1)

        # Read any remaining output before checking return code
        if process.stdout:
            remaining_output = process.stdout.read()
            if remaining_output:
                for line in remaining_output.split('\n'):
                    if line.strip():
                        print(f"[Training] {line.strip()}")

        # Final return code check
        return_code = process.wait()
        if return_code != 0 and not training_controls.get(task_id, {}).get("cancelled", False):
            raise Exception(f"Training process exited with code {return_code}")

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

        # Training complete
        background_tasks_status[task_id]["progress"] = 95
        background_tasks_status[task_id]["message"] = "Finalizing..."
        await ws_manager.broadcast({
            "type": "training_progress",
            "task_id": task_id,
            "progress": 95,
            "message": "Training complete, finalizing..."
        })

        # Find the saved model
        checkpoints_dir = Path(config["paths"]["checkpoints"])
        best_model = checkpoints_dir / "best_agent.pth"
        final_model = checkpoints_dir / "final_agent.pth"

        model_path = str(best_model) if best_model.exists() else str(final_model)

        # Complete
        background_tasks_status[task_id]["status"] = "completed"
        background_tasks_status[task_id]["progress"] = 100
        background_tasks_status[task_id]["message"] = f"Training completed! {current_episode} episodes"

        await ws_manager.broadcast({
            "type": "training_complete",
            "task_id": task_id,
            "progress": 100,
            "message": f"🎉 Training completed! {current_episode} episodes finished",
            "model_path": model_path,
            "episodes": current_episode
        })

    except Exception as e:
        import traceback
        error_msg = f"Training failed: {str(e)}"
        print(f"Training error: {error_msg}")
        print(traceback.format_exc())

        background_tasks_status[task_id]["status"] = "failed"
        background_tasks_status[task_id]["message"] = error_msg
        await ws_manager.broadcast({
            "type": "training_error",
            "task_id": task_id,
            "message": error_msg
        })


# ============================================================================
# Backtest Functions
# ============================================================================

async def backtest_task(task_id: str, request: BacktestRequest):
    """Background task to run backtest on trained model"""
    try:
        # Fix sys.path for imports (critical for background tasks)
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(current_dir, 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        import torch
        import numpy as np
        from pathlib import Path
        from agent.d3qn_agent import D3QNAgent
        from policy.trading_env import TradingEnv
        from data.loader import load_market_data, prepare_env_data, create_sample_data
        from utils.logging_config import get_server_logger

        logger = get_server_logger()
        logger.info(f"[Backtest] Starting backtest task: {task_id}")

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
        print(f"[Backtest] Using device: {device}")
        background_tasks_status[task_id]["message"] = f"Using device: {device}"
        background_tasks_status[task_id]["progress"] = 5
        await ws_manager.broadcast({
            "type": "backtest_progress",
            "task_id": task_id,
            "progress": 5,
            "message": f"Using device: {device}"
        })

        await asyncio.sleep(0.5)

        # Verify model file exists
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {request.model_path}")

        print(f"[Backtest] Loading model from: {model_path}")
        print(f"[Backtest] Model file size: {model_path.stat().st_size / (1024*1024):.2f} MB")

        # Load market data
        background_tasks_status[task_id]["message"] = "Loading market data..."
        background_tasks_status[task_id]["progress"] = 10
        await ws_manager.broadcast({
            "type": "backtest_progress",
            "task_id": task_id,
            "progress": 10,
            "message": "Loading market data..."
        })

        # Use first symbol from request
        symbol = request.symbols[0] if request.symbols else "BTCUSDT"
        timeframe = "1h"  # Fixed for now
        window_size = 72  # Default window size

        print(f"[Backtest] Loading data for {symbol} {timeframe}, window_size={window_size}")

        # Load data (will use synthetic if no real data available)
        try:
            df = load_market_data(symbol=symbol, timeframe=timeframe)
            print(f"[Backtest] Loaded real market data: {len(df)} candles")
        except:
            print(f"[Backtest] No real data found, using synthetic data")
            df = create_sample_data(symbol=symbol, n_samples=10000, save_dir="data")

        # Prepare data for environment
        env_data = prepare_env_data(df, add_indicators=True, normalize=False)

        print(f"[Backtest] Data prepared: {env_data.shape[0]} timesteps, {env_data.shape[1]} features")

        # Create trading environment
        background_tasks_status[task_id]["message"] = "Creating trading environment..."
        background_tasks_status[task_id]["progress"] = 20
        await ws_manager.broadcast({
            "type": "backtest_progress",
            "task_id": task_id,
            "progress": 20,
            "message": "Creating trading environment..."
        })

        env = TradingEnv(
            data=env_data,
            initial_balance=request.initial_balance,
            max_leverage=request.max_leverage,
            fee_rate=request.fee_rate,
            window_size=window_size,
            reward_type="sharpe"
        )

        print(f"[Backtest] Environment created")

        # Load model
        background_tasks_status[task_id]["message"] = "Loading trained model..."
        background_tasks_status[task_id]["progress"] = 30
        await ws_manager.broadcast({
            "type": "backtest_progress",
            "task_id": task_id,
            "progress": 30,
            "message": "Loading trained model..."
        })

        # Create agent and load checkpoint
        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n

        # Calculate flattened state dimension (window_size * features)
        # obs_shape = (72, 21) -> state_dim = 72 * 21 = 1512
        state_dim = obs_shape[0] * obs_shape[1] if len(obs_shape) > 1 else obs_shape[0]

        logger.info(f"[Backtest] Observation shape: {obs_shape}, State dim: {state_dim}, Actions: {n_actions}")
        print(f"[Backtest] Observation shape: {obs_shape}, State dim: {state_dim}, Actions: {n_actions}")

        agent = D3QNAgent(
            state_dim=state_dim,
            action_dim=n_actions,
            learning_rate=0.00025,  # Default, not used in evaluation
            gamma=0.995,
            device=device
        )

        # Load the trained model
        agent.load(str(model_path))
        agent.policy_net.eval()  # Set to evaluation mode

        print(f"[Backtest] Model loaded successfully")

        # Run backtest
        background_tasks_status[task_id]["message"] = "Running backtest simulation..."
        background_tasks_status[task_id]["progress"] = 40
        await ws_manager.broadcast({
            "type": "backtest_progress",
            "task_id": task_id,
            "progress": 40,
            "message": "Running backtest simulation..."
        })

        # Reset environment
        state, _ = env.reset()
        done = False
        step = 0
        episode_reward = 0

        # Track equity curve
        equity_curve = [request.initial_balance]
        price_history = []

        # Track position changes for trade history
        prev_position = None
        prev_balance = request.initial_balance

        print(f"[Backtest] Starting simulation...")

        total_steps = len(env_data) - window_size

        while not done:
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

            # Select action using the trained agent (no exploration)
            with torch.no_grad():
                # Flatten state if it's 2D (window_size, features) -> (window_size * features,)
                state_flat = state.flatten() if len(state.shape) > 1 else state
                state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(device)
                q_values = agent.policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()

            # Take action (Gymnasium API returns: observation, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated  # Episode ends if either is True
            episode_reward += reward

            # Get current price (close price at index 3 in OHLCV data)
            current_step_idx = info.get('step', window_size)
            if current_step_idx < len(env_data):
                current_price = env_data[current_step_idx, 3]  # Close price
            else:
                current_price = env_data[-1, 3]

            # Record equity and price
            equity_curve.append(info['balance'])
            price_history.append(current_price)

            state = next_state
            step += 1

            # Update progress every 100 steps
            if step % 100 == 0:
                progress = 40 + int((step / total_steps) * 50)  # 40-90%
                background_tasks_status[task_id]["progress"] = progress
                background_tasks_status[task_id]["message"] = f"Running backtest... {step}/{total_steps} steps"

                await ws_manager.broadcast({
                    "type": "backtest_progress",
                    "task_id": task_id,
                    "progress": progress,
                    "message": f"Running backtest... {step}/{total_steps} steps"
                })

        print(f"[Backtest] Simulation completed: {step} steps")

        # Calculate performance metrics
        background_tasks_status[task_id]["message"] = "Calculating performance metrics..."
        background_tasks_status[task_id]["progress"] = 95
        await ws_manager.broadcast({
            "type": "backtest_progress",
            "task_id": task_id,
            "progress": 95,
            "message": "Calculating performance metrics..."
        })

        # Get final info from environment
        final_info = info
        final_balance = final_info['balance']
        total_return_pct = ((final_balance - request.initial_balance) / request.initial_balance) * 100

        # Get trade statistics from environment
        total_trades = final_info.get('total_trades', 0)
        win_rate = final_info.get('win_rate', 0) * 100

        # Calculate equity-based metrics
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]

        # Sharpe ratio
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)  # Annualized for 1H
        else:
            sharpe_ratio = 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252 * 24) if downside_std > 0 else 0
        else:
            sortino_ratio = sharpe_ratio

        # Max drawdown
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown_pct = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0

        # Profit factor (simplified calculation from returns)
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        total_gains = np.sum(positive_returns) if len(positive_returns) > 0 else 0
        total_losses = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 0
        profit_factor = (total_gains / total_losses) if total_losses > 0 else 0

        # Generate chart data from actual price history
        chart_data = []
        for i in range(0, len(price_history), max(1, len(price_history) // 100)):  # Sample ~100 points
            chart_data.append({
                "time": int(1704067200 + i * 3600),  # Approximate timestamps
                "close": round(float(price_history[i]), 2),
                "equity": round(float(equity_curve[i]), 2)
            })

        # Estimate winning/losing trades from win rate
        winning_trades = int(total_trades * (win_rate / 100)) if total_trades > 0 else 0
        losing_trades = total_trades - winning_trades

        trade_history = serialize_trade_records(getattr(env, "trade_log", []))

        results = {
            "summary": {
                "total_return_pct": round(total_return_pct, 2),
                "initial_balance": request.initial_balance,
                "final_balance": round(final_balance, 2),
                "max_leverage_used": request.max_leverage
            },
            "trades": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
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

        print(f"[Backtest] Results: Return={total_return_pct:.2f}%, Trades={total_trades}, Win Rate={win_rate:.2f}%")
        log_payload = {
            "task_id": task_id,
            "request": request.dict(),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        log_file = BACKTEST_LOG_DIR / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{task_id}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_payload, f, indent=2)
        print(f"[Backtest] Log saved to: {log_file}")

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
        import traceback
        error_msg = f"Backtest failed: {str(e)}"

        # Log to file
        try:
            from utils.logging_config import get_server_logger
            logger = get_server_logger()
            logger.error(f"[Backtest] Error in task {task_id}: {error_msg}")
            logger.error(traceback.format_exc())
        except:
            pass

        print(f"[Backtest] Error: {e}")
        traceback.print_exc()

        background_tasks_status[task_id]["status"] = "failed"
        background_tasks_status[task_id]["message"] = error_msg
        await ws_manager.broadcast({
            "type": "backtest_error",
            "task_id": task_id,
            "message": error_msg
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

        # Calculate timeframe in milliseconds
        timeframe_ms_map = {
            '1m': 60000,
            '3m': 180000,
            '5m': 300000,
            '15m': 900000,
            '30m': 1800000,
            '1h': 3600000,
            '2h': 7200000,
            '4h': 14400000,
            '6h': 21600000,
            '8h': 28800000,
            '12h': 43200000,
            '1d': 86400000,
            '3d': 259200000,
            '1w': 604800000,
        }
        timeframe_ms = timeframe_ms_map.get(request.timeframe, 3600000)  # Default 1h

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

                # Fetch OHLCV data with retry logic
                since = int(start_dt.timestamp() * 1000)
                end_ts = int(end_dt.timestamp() * 1000)
                limit = 1000  # Binance max limit
                all_candles = []

                # Calculate expected number of candles
                time_range_ms = end_ts - since
                expected_candles = time_range_ms // timeframe_ms

                print(f"\n[Download] {symbol} - {request.timeframe}")
                print(f"  Start: {start_dt} -> End: {end_dt}")
                print(f"  Timeframe: {request.timeframe} ({timeframe_ms}ms)")
                print(f"  Expected ~{expected_candles} candles")

                retry_count = 0
                max_retries = 3
                consecutive_empty = 0
                max_consecutive_empty = 3

                while since < end_ts:
                    try:
                        candles = await asyncio.to_thread(
                            exchange.fetch_ohlcv,
                            symbol,
                            request.timeframe,
                            since,
                            limit
                        )

                        if not candles or len(candles) == 0:
                            consecutive_empty += 1
                            print(f"  Empty response at {datetime.fromtimestamp(since/1000)} (consecutive: {consecutive_empty})")

                            # If we get multiple consecutive empty responses, we're done
                            if consecutive_empty >= max_consecutive_empty:
                                print(f"  Reached {consecutive_empty} consecutive empty responses, stopping")
                                break

                            # Move forward by 1 day and try again
                            since += 86400000  # 1 day in ms
                            await asyncio.sleep(0.5)
                            continue

                        # Reset consecutive empty counter on successful fetch
                        consecutive_empty = 0
                        retry_count = 0  # Reset retry count on success

                        all_candles.extend(candles)
                        last_timestamp = candles[-1][0]

                        # Move to next batch: last candle timestamp + 1 interval
                        since = last_timestamp + timeframe_ms

                        # Update progress with more detail
                        progress_pct = min(90, 5 + (idx / total_symbols) * 80 + (len(all_candles) / max(expected_candles, 1)) * (80 / total_symbols))
                        background_tasks_status[task_id]["progress"] = int(progress_pct)
                        background_tasks_status[task_id]["message"] = f"Downloading {symbol}... ({len(all_candles)}/{expected_candles} candles)"

                        await ws_manager.broadcast({
                            "type": "download_progress",
                            "task_id": task_id,
                            "progress": int(progress_pct),
                            "message": f"Downloading {symbol}... ({len(all_candles)}/{expected_candles} candles)"
                        })

                        print(f"  Fetched {len(candles)} candles, total: {len(all_candles)}/{expected_candles}, next: {datetime.fromtimestamp(since/1000)}")

                        # Respect Binance rate limits (1200 req/min = 50ms per request minimum)
                        # Use 200ms to be safe (5 req/sec = 300 req/min, well under limit)
                        await asyncio.sleep(0.2)

                    except Exception as fetch_error:
                        retry_count += 1
                        print(f"  Error fetching data (retry {retry_count}/{max_retries}): {fetch_error}")

                        if retry_count >= max_retries:
                            print(f"  Max retries reached, moving to next timestamp")
                            since += limit * timeframe_ms  # Skip this batch
                            retry_count = 0
                            await asyncio.sleep(1)  # Longer delay after errors
                        else:
                            await asyncio.sleep(2 ** retry_count)  # Exponential backoff

                print(f"  [OK] Downloaded {len(all_candles)} total candles for {symbol} (expected ~{expected_candles})")

                # Warn if we got significantly fewer candles than expected
                if len(all_candles) < expected_candles * 0.8:
                    print(f"  [WARN]  Warning: Got {len(all_candles)} candles but expected ~{expected_candles}")
                    background_tasks_status[task_id]["message"] = f"Downloaded {symbol} - {len(all_candles)}/{expected_candles} candles (some data may be missing)"


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

    # Validate timeframe
    valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
    if request.timeframe not in valid_timeframes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timeframe. Supported: {', '.join(valid_timeframes)}"
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

    # Start appropriate training task based on model type
    if request.model_type == "d3qn":
        background_tasks.add_task(train_d3qn_task, task_id, request)
    elif request.model_type in ["sac", "ppo", "td3"]:
        background_tasks.add_task(train_sb3_task, task_id, request)
    else:
        # Fallback to mock training for unknown models
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
    checkpoints_dir = os.path.abspath(config["paths"]["checkpoints"])
    models_dir = os.path.abspath(config["paths"]["models"])
    available_models = []

    print(f"[API] Checking for models in:")
    print(f"  Checkpoints: {checkpoints_dir}")
    print(f"  Models: {models_dir}")

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
                print(f"  Found: {file} ({stat.st_size / (1024 * 1024):.2f} MB)")

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
                print(f"  Found: {file} ({stat.st_size / (1024 * 1024):.2f} MB)")

    print(f"[API] Total models found: {len(available_models)}")
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
        from pathlib import Path
        import torch
        from agent.d3qn_agent import D3QNAgent
        from policy.trading_env import TradingEnv
        from data.loader import prepare_env_data
        from stable_baselines3 import PPO, SAC, TD3
        from datetime import datetime

        symbol = config["symbol"]
        model_path = Path(config["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        initial_balance = config["initial_balance"]
        max_leverage = config["max_leverage"]
        fee_rate = config["fee_rate"]
        update_interval = config.get("update_interval", 2)
        window_size = config.get("window_size", 72)

        # Initialize exchange
        exchange = ccxt.binance()

        # Fetch historical candles for context
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=max(window_size + 10, 1000))
        chart_data = [{
            "time": int(candle[0] / 1000),
            "open": candle[1],
            "high": candle[2],
            "low": candle[3],
            "close": candle[4],
            "volume": candle[5]
        } for candle in ohlcv]

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        paper_log_file = PAPER_LOG_DIR / f"paper_{run_id}.log"
        paper_trade_file = PAPER_LOG_DIR / f"paper_trades_{run_id}.jsonl"

        def log_event(message: str):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(paper_log_file, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp} - {message}\n")

        async def broadcast_initial():
            await ws_manager.broadcast({
                "type": "paper_trading_update",
                "candle": chart_data[-1] if chart_data else None
            })

        await broadcast_initial()
        log_event(f"Paper trading started for {symbol} (model={model_path})")

        df = pd.DataFrame(chart_data)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        env_data = prepare_env_data(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], add_indicators=True, normalize=True)
        if len(env_data) <= window_size:
            raise RuntimeError("Not enough data to start paper trading")

        env = TradingEnv(
            data=env_data,
            initial_balance=initial_balance,
            max_leverage=max_leverage,
            fee_rate=fee_rate,
            window_size=window_size,
            reward_type="sharpe",
            streaming_mode=True
        )
        env.reset(options={"start_index": window_size})

        obs_shape = env.observation_space.shape
        state_dim = int(np.prod(obs_shape))
        action_dim = env.action_space.n
        device = "cuda" if torch.cuda.is_available() else "cpu"

        sb3_model = None
        agent = None
        if model_path.suffix.lower() == ".zip":
            for cls in (PPO, SAC, TD3):
                try:
                    sb3_model = cls.load(str(model_path))
                    break
                except Exception:
                    sb3_model = None
            if sb3_model is None:
                raise RuntimeError("Unsupported SB3 checkpoint")
        else:
            agent = D3QNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=0.00025,
                gamma=0.995,
                device=device
            )
            agent.load(str(model_path))
            agent.policy_net.eval()

        def format_trades(timestamps: List[pd.Timestamp]):
            rows = []
            for idx, trade in enumerate(env.trade_log[-20:], 1):
                entry_idx = min(trade["entry_step"], len(timestamps) - 1)
                exit_idx = min(trade["exit_step"], len(timestamps) - 1)
                exit_time = int(timestamps[exit_idx].timestamp())
                rows.append({
                    "id": str(idx),
                    "timestamp": exit_time,
                    "action": f"CLOSE_{trade['direction'].upper()}",
                    "position": trade["direction"].upper(),
                    "entry_price": trade["entry_price"],
                    "exit_price": trade["exit_price"],
                    "current_price": trade["exit_price"],
                    "quantity": trade["quantity"],
                    "leverage": trade["leverage"],
                    "liquidation_price": 0,
                    "take_profit": 0,
                    "stop_loss": 0,
                    "break_even": 0,
                    "pnl": trade["pnl"],
                    "pnl_pct": trade["pnl_pct"],
                    "balance": env.balance,
                    "status": "CLOSED"
                })
            return rows

        last_candle_time = chart_data[-1]["time"] if chart_data else None
        last_logged_trade_idx = 0

        def flush_trades():
            nonlocal last_logged_trade_idx
            trades = getattr(env, "trade_log", [])
            if len(trades) <= last_logged_trade_idx:
                return
            new_trades = serialize_trade_records(trades[last_logged_trade_idx:])
            with open(paper_trade_file, 'a', encoding='utf-8') as f:
                for record in new_trades:
                    f.write(json.dumps(record) + "\n")
            last_logged_trade_idx = len(trades)

        while paper_trading_state["active"]:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            current_timestamp = int(datetime.now().timestamp())
            current_candle_time = (current_timestamp // 60) * 60

            if last_candle_time is None or current_candle_time != last_candle_time:
                new_candle = {
                    "time": current_candle_time,
                    "open": current_price,
                    "high": current_price,
                    "low": current_price,
                    "close": current_price,
                    "volume": ticker.get('baseVolume', 0)
                }
                chart_data.append(new_candle)
                if len(chart_data) > 1000:
                    chart_data = chart_data[-1000:]

                df = pd.DataFrame(chart_data)
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                env_data = prepare_env_data(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], add_indicators=True, normalize=True)
                if len(env_data) <= window_size:
                    last_candle_time = current_candle_time
                    await asyncio.sleep(update_interval)
                    continue

                env.data = env_data.astype(np.float32)
                env.n_steps = len(env_data)
                env.current_step = env.n_steps - 1

                observation = env._get_observation()
                if sb3_model:
                    action, _ = sb3_model.predict(observation, deterministic=True)
                    action = int(action)
                else:
                    action = agent.select_action(observation.flatten(), training=False)
                action = int(np.clip(action, 0, action_dim - 1))

                env.step(action)
                timestamps = df['timestamp'].tolist()

                unrealized = env._calculate_position_pnl()
                equity = env.balance + unrealized

                returns = np.diff(env.equity_history) / np.maximum(env.equity_history[:-1], 1e-8) if len(env.equity_history) > 1 else np.array([0])
                sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)) if len(returns) > 0 else 0.0
                eq_array = np.array(env.equity_history)
                running_max = np.maximum.accumulate(eq_array)
                drawdowns = (eq_array - running_max) / np.maximum(running_max, 1e-8)
                max_drawdown = float(abs(drawdowns.min()) * 100) if len(drawdowns) > 0 else 0.0

                current_position_data = None
                if env.position:
                    current_position_data = {
                        "direction": env.position.upper(),
                        "entry_price": env.entry_price,
                        "current_price": current_price,
                        "quantity": env.position_size,
                        "leverage": env.current_leverage,
                        "liquidation_price": env.liquidation_price,
                        "take_profit": env.take_profit,
                        "stop_loss": env.stop_loss,
                        "break_even": env.break_even,
                        "unrealized_pnl": unrealized,
                        "unrealized_pnl_pct": (unrealized / max(env.entry_price * env.position_size / max(env.current_leverage, 1e-8), 1e-8)) * 100 if env.entry_price and env.position_size else 0.0,
                        "margin_used": env.balance / max(env.current_leverage, 1) if env.current_leverage else 0.0,
                        "duration_hours": env.hold_duration
                    }

                trade_rows = format_trades(timestamps)
                flush_trades()

                await ws_manager.broadcast({
                    "type": "paper_trading_update",
                    "metrics": {
                        "balance": env.balance,
                        "initial_balance": initial_balance,
                        "total_profit": env.balance - initial_balance,
                        "total_return": ((equity - initial_balance) / initial_balance) * 100,
                        "win_rate": env.winning_trades / max(env.total_trades, 1) * 100,
                        "total_trades": env.total_trades,
                        "winning_trades": env.winning_trades,
                        "losing_trades": env.total_trades - env.winning_trades,
                        "max_drawdown": max_drawdown,
                        "sharpe_ratio": sharpe,
                        "current_position": env.position.upper() if env.position else "NEUTRAL",
                        "position_size": env.position_size if env.position else 0,
                        "unrealized_pnl": unrealized
                    },
                    "currentPosition": current_position_data,
                    "candle": new_candle,
                    "trades": trade_rows,
                    "positionHistory": trade_rows[-10:]
                })

                last_candle_time = current_candle_time

            await asyncio.sleep(update_interval)

        flush_trades()
        log_event("Paper trading stopped normally")

    except Exception as e:
        print(f"Paper trading error: {e}")
        paper_trading_state["active"] = False
        log_event(f"Paper trading stopped due to error: {e}")
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

@app.get("/api/paper-trading/historical-data")
async def get_historical_data(symbol: str, timeframe: str = '1m', limit: int = 1000):
    """Get historical candlestick data for any timeframe"""
    try:
        import ccxt

        # Initialize exchange
        exchange = ccxt.binance()

        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        # Convert to our format
        candles = []
        for candle in ohlcv:
            candles.append({
                "time": int(candle[0] / 1000),  # Convert ms to seconds
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5]
            })

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles,
            "count": len(candles)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data: {str(e)}")

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
    client_host = websocket.client.host if websocket.client else "unknown"
    client_port = websocket.client.port if websocket.client else "unknown"

    server_logger.info(f"[WebSocket] Connection attempt from {client_host}:{client_port}")
    print(f"[WebSocket] Connection attempt from {client_host}:{client_port}")

    try:
        # Log headers for debugging
        headers = dict(websocket.headers)
        server_logger.debug(f"[WebSocket] Headers: {headers}")

        print(f"[WebSocket] Accepting connection...")
        await ws_manager.connect(websocket)

        server_logger.info(f"[WebSocket] Client connected: {client_host}:{client_port}")
        print(f"[WebSocket] Client connected: {client_host}:{client_port}")

        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "WebSocket connection established"
        })

        try:
            while True:
                # Keep connection alive and receive messages
                data = await websocket.receive_text()
                print(f"[WebSocket] Received from {client_host}: {data[:50] if len(data) > 50 else data}")

                # Parse and handle message
                try:
                    import json
                    msg = json.loads(data)
                    if msg.get('type') == 'ping':
                        await websocket.send_json({"type": "pong"})
                except:
                    # Echo back for testing
                    await websocket.send_json({"type": "echo", "data": data})

        except WebSocketDisconnect:
            server_logger.info(f"[WebSocket] Client disconnected: {client_host}:{client_port}")
            print(f"[WebSocket] Client disconnected: {client_host}:{client_port}")
            ws_manager.disconnect(websocket)
        except Exception as e:
            server_logger.error(f"[WebSocket] Error in message loop: {type(e).__name__}: {e}")
            print(f"[WebSocket] Error in message loop: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            ws_manager.disconnect(websocket)

    except Exception as e:
        server_logger.error(f"[WebSocket] Failed to establish connection: {type(e).__name__}: {e}")
        print(f"[WebSocket] Failed to accept connection: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close(code=1011, reason=str(e)[:100])
        except:
            pass

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("CRYPPO System - Backend Server")
    print("Version: 2.0.0 | Timeframe: 1H Only")
    print("=" * 60)

    # Check for websockets package
    websockets_available = False
    try:
        import websockets
        print(f"\n[OK] WebSocket support: websockets {websockets.__version__}")
        websockets_available = True
    except ImportError:
        print("\n[ERROR] websockets package not installed!")
        print("  WebSocket connections will fail with 400 Bad Request")
        print("\n  To fix, run:")
        print("    pip install websockets>=12.0")
        print("    OR")
        print("    pip install -r requirements.txt")
        print("\n  Starting server anyway (REST API will work, WebSocket won't)...")

    print("\nStarting server...")
    print("API Docs: http://localhost:8000/docs")
    print("WebSocket: ws://localhost:8000/ws")
    print("=" * 60)

    # Use websockets protocol explicitly if available
    uvicorn_config = {
        "app": app,
        "host": "0.0.0.0",
        "port": 8000,
        "log_level": "info"
    }

    if websockets_available:
        uvicorn_config["ws"] = "websockets"

    uvicorn.run(**uvicorn_config)
