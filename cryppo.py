#!/usr/bin/env python3
"""
CRYPPO - Deep RL Crypto Trading System CLI
Beautiful command-line interface inspired by Claude Code
"""

import sys
import os
import io
import time
import argparse
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import json

import numpy as np

if os.name == "nt":
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "buffer"):
            try:
                buffer = stream.buffer
                setattr(sys, stream_name, io.TextIOWrapper(buffer, encoding="utf-8", errors="ignore"))
            except Exception:
                pass

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

console = Console()


@contextmanager
def server_module_path():
    """Temporarily add server/src to sys.path for imports."""
    server_path = str(Path(__file__).parent / "server")
    added = False
    if server_path not in sys.path:
        sys.path.insert(0, server_path)
        added = True
    try:
        yield
    finally:
        if added and server_path in sys.path:
            sys.path.remove(server_path)


LOG_ROOT = Path(__file__).parent / "logs"
TRAIN_LOG_DIR = LOG_ROOT / "training"
BACKTEST_LOG_DIR = LOG_ROOT / "backtest"
PAPER_LOG_DIR = LOG_ROOT / "paper"
for _dir in (TRAIN_LOG_DIR, BACKTEST_LOG_DIR, PAPER_LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

ACTION_NAMES = ["HOLD", "OPEN_LONG", "OPEN_SHORT", "CLOSE"]


MODEL_REGISTRY: Dict[str, Dict] = {
    "d3qn": {
        "label": "D3QN",
        "module": "src.training.train_d3qn",
        "args": [
            ("--symbol", "symbol"),
            ("--timeframe", "timeframe"),
            ("--episodes", "episodes"),
            ("--max-steps", "max_steps"),
            ("--eval-freq", "eval_freq"),
            ("--save-freq", "save_freq"),
            ("--balance", "balance"),
            ("--leverage", "leverage"),
            ("--window", "window"),
            ("--reward", "reward"),
            ("--lr", "lr"),
            ("--gamma", "gamma"),
            ("--batch-size", "batch_size"),
            ("--buffer-size", "buffer_size"),
            ("--checkpoint-dir", "checkpoint_dir"),
            ("--log-dir", "log_dir"),
        ],
        "flags": [("--sample-data", "sample_data")]
    },
    "ppo": {
        "label": "PPO",
        "module": "src.training.train_ppo",
        "args": [
            ("--symbol", "symbol"),
            ("--timeframe", "timeframe"),
            ("--timesteps", "timesteps"),
            ("--balance", "balance"),
            ("--leverage", "leverage"),
            ("--lr", "lr"),
            ("--gamma", "gamma"),
            ("--n-steps", "n_steps"),
            ("--batch-size", "batch_size"),
            ("--n-epochs", "n_epochs"),
            ("--clip-range", "clip_range"),
            ("--window", "window"),
            ("--reward", "reward"),
            ("--checkpoint-dir", "checkpoint_dir"),
            ("--log-dir", "log_dir"),
            ("--eval-freq", "eval_freq"),
            ("--n-eval-episodes", "n_eval_episodes"),
            ("--save-freq", "save_freq"),
            ("--verbose", "verbose"),
        ],
        "flags": [("--sample-data", "sample_data")]
    },
    "sac": {
        "label": "SAC",
        "module": "src.training.train_sac",
        "args": [
            ("--symbol", "symbol"),
            ("--timeframe", "timeframe"),
            ("--timesteps", "timesteps"),
            ("--balance", "balance"),
            ("--leverage", "leverage"),
            ("--lr", "lr"),
            ("--gamma", "gamma"),
            ("--buffer-size", "buffer_size"),
            ("--batch-size", "batch_size"),
            ("--learning-starts", "learning_starts"),
            ("--tau", "tau"),
            ("--window", "window"),
            ("--reward", "reward"),
            ("--checkpoint-dir", "checkpoint_dir"),
            ("--log-dir", "log_dir"),
            ("--eval-freq", "eval_freq"),
            ("--n-eval-episodes", "n_eval_episodes"),
            ("--save-freq", "save_freq"),
            ("--verbose", "verbose"),
        ],
        "flags": [("--sample-data", "sample_data")]
    },
    "td3": {
        "label": "TD3",
        "module": "src.training.train_td3",
        "args": [
            ("--symbol", "symbol"),
            ("--timeframe", "timeframe"),
            ("--timesteps", "timesteps"),
            ("--balance", "balance"),
            ("--leverage", "leverage"),
            ("--lr", "lr"),
            ("--gamma", "gamma"),
            ("--buffer-size", "buffer_size"),
            ("--batch-size", "batch_size"),
            ("--learning-starts", "learning_starts"),
            ("--tau", "tau"),
            ("--policy-delay", "policy_delay"),
            ("--window", "window"),
            ("--reward", "reward"),
            ("--checkpoint-dir", "checkpoint_dir"),
            ("--log-dir", "log_dir"),
            ("--eval-freq", "eval_freq"),
            ("--n-eval-episodes", "n_eval_episodes"),
            ("--save-freq", "save_freq"),
            ("--verbose", "verbose"),
        ],
        "flags": [("--sample-data", "sample_data")]
    },
}

SB3_MODELS = {"ppo": "PPO", "sac": "SAC", "td3": "TD3"}


def unwrap_env(env):
    """Drill through wrappers to get base TradingEnv."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def serialize_trades(trades):
    """Convert trade records to JSON-serializable list."""
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


def save_json_log(directory: Path, prefix: str, payload: dict) -> Path:
    """Persist JSON payload to logs and return path."""
    directory.mkdir(parents=True, exist_ok=True)
    log_file = directory / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)
    return log_file


def action_to_env(model_key: str, action_value):
    """Convert model output to environment-compatible action."""
    if model_key in {"sac", "td3"}:
        return action_value

    array = np.array(action_value).flatten()
    if array.size == 0:
        return 0
    return int(array[0])


def action_to_index(model_key: str, action_value):
    """Get discrete action index for logging/metrics."""
    if model_key in {"sac", "td3"}:
        array = np.array(action_value, dtype=np.float32).flatten()
        if array.size == 0:
            return 0
        return int(np.argmax(array))
    return action_to_env(model_key, action_value)

# ============================================================================
# Banner & Styling
# ============================================================================

BANNER = """
  ██████╗██████╗ ██╗   ██╗██████╗ ██████╗  ██████╗
 ██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗██╔══██╗██╔═══██╗
 ██║     ██████╔╝ ╚████╔╝ ██████╔╝██████╔╝██║   ██║
 ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝ ██╔═══╝ ██║   ██║
 ╚██████╗██║  ██║   ██║   ██║     ██║     ╚██████╔╝
  ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚═╝      ╚═════╝

 Deep RL Crypto Trading System
 GPU-Accelerated | Python 3.12 | CUDA 12.6
"""

def print_banner():
    """Print beautiful startup banner"""
    try:
        console.print(
            Panel(
                Text(BANNER, style="bold cyan", justify="center"),
                border_style="bright_cyan",
                box=box.DOUBLE
            )
        )
    except UnicodeEncodeError:
        # Fallback for terminals that cannot render block characters
        fallback = "CRYPPO - Deep RL Crypto Trading System"
        try:
            print(fallback)
        except UnicodeEncodeError:
            sys.stdout.write(fallback + "\n")

# ============================================================================
# System Check
# ============================================================================

def check_system():
    """Check system requirements"""
    console.print("\n[bold cyan]System Check[/bold cyan]\n")

    checks = []

    # Check Python
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append(("Python", python_version, "OK" if sys.version_info >= (3, 10) else "FAIL"))

    # Check PyTorch
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        checks.append(("PyTorch", torch_version, "OK"))
        checks.append(("CUDA", "Available" if cuda_available else "Not Available", "OK" if cuda_available else "WARN"))
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            checks.append(("GPU", gpu_name, "OK"))
    except Exception as e:
        checks.append(("PyTorch", f"Error: {str(e)}", "FAIL"))

    # Check other packages
    packages = [
        ("FastAPI", "fastapi"),
        ("Gymnasium", "gymnasium"),
        ("CCXT", "ccxt"),
        ("Pandas", "pandas"),
    ]

    for name, module in packages:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "installed")
            checks.append((name, version, "OK"))
        except:
            checks.append((name, "Not installed", "FAIL"))

    # Create table
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Version/Status", style="white")
    table.add_column("Status", justify="center")

    status_styles = {"OK": "green", "WARN": "yellow", "FAIL": "red"}
    for component, version, status in checks:
        style = status_styles.get(status, "white")
        table.add_row(component, version, f"[{style}]{status}[/{style}]")

    console.print(table)
    console.print()

# ============================================================================
# Data Management
# ============================================================================

def download_data(symbol: str, start: str, end: str, timeframe: str = "1h"):
    """Download market data with progress bar"""
    # Import in isolated scope to avoid sys.path issues
    import sys
    server_path = str(Path(__file__).parent / "server")
    if server_path not in sys.path:
        sys.path.insert(0, server_path)

    try:
        from src.data.loader import download_and_save_data
    finally:
        # Clean up sys.path after import
        if server_path in sys.path:
            sys.path.remove(server_path)

    console.print(f"\n[bold cyan]Downloading {symbol} data[/bold cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Downloading {symbol} ({timeframe})...", total=100)

        try:
            # Download data
            df = download_and_save_data(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                exchange="binance"
            )

            progress.update(task, completed=100)

            # Show summary
            console.print(f"\n[green]OK[/green] Downloaded {len(df)} candles")
            console.print(f"[dim]Period: {df.index[0]} to {df.index[-1]}[/dim]")
            console.print(f"[dim]Saved to: data/{symbol}_{timeframe}_{start}_{end}.parquet[/dim]\n")

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]\n")
            sys.exit(1)

def list_data():
    """List available data files"""
    console.print("\n[bold cyan]Available data files[/bold cyan]\n")

    data_dir = Path("data")
    if not data_dir.exists():
        console.print("[yellow]No data directory found[/yellow]\n")
        console.print(f"[dim]Path checked: {data_dir.absolute()}[/dim]\n")
        return

    # Search for both parquet and pickle files
    files = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.pkl")) + list(data_dir.glob("*.csv"))

    console.print(f"[dim]Searching in: {data_dir.absolute()}[/dim]")
    console.print(f"[dim]Files found: {len(files)}[/dim]\n")

    if not files:
        console.print("[yellow]No data files found[/yellow]\n")
        console.print("[dim]Supported formats: .parquet, .pkl, .csv[/dim]\n")
        return

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("#", style="dim")
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Modified", style="dim")

    for i, file in enumerate(sorted(files), 1):
        size = file.stat().st_size
        size_str = f"{size / 1024 / 1024:.2f} MB" if size > 1024*1024 else f"{size / 1024:.2f} KB"
        modified = file.stat().st_mtime
        from datetime import datetime
        mod_str = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
        table.add_row(str(i), file.name, size_str, mod_str)

    console.print(table)
    console.print()


def prepare_data(
    symbol: str,
    timeframe: str,
    window: int,
    train_ratio: float,
    val_ratio: float,
    output_dir: str,
    add_indicators: bool = True,
    normalize: bool = True,
    sample_data: bool = False,
    sample_size: int = 10000
):
    """Prepare numpy datasets for training/backtesting"""
    if train_ratio + val_ratio >= 1.0:
        console.print("[red]Train + validation ratio must be < 1.0[/red]\n")
        return

    console.print(f"\n[bold cyan]Preparing data for {symbol} ({timeframe})[/bold cyan]")

    with server_module_path():
        from src.data.loader import (
            load_market_data,
            prepare_env_data,
            split_train_test,
            create_sample_data
        )

    if sample_data:
        df = create_sample_data(symbol=symbol, n_samples=sample_size, save_dir="data")
    else:
        df = load_market_data(symbol=symbol, timeframe=timeframe)

    env_data = prepare_env_data(df, add_indicators=add_indicators, normalize=normalize)

    if len(env_data) <= window:
        console.print("[red]Not enough rows after preprocessing. Try a smaller window or more data.[/red]\n")
        return

    train_data, val_data, test_data = split_train_test(env_data, train_ratio=train_ratio, val_ratio=val_ratio)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{symbol}_{timeframe}_prepared.npz"
    np.savez_compressed(file_path, train=train_data, val=val_data, test=test_data)

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Split", style="cyan")
    table.add_column("Samples", justify="right")

    table.add_row("Train", f"{len(train_data):,}")
    table.add_row("Validation", f"{len(val_data):,}")
    table.add_row("Test", f"{len(test_data):,}")

    console.print(table)
    console.print(f"[green]OK[/green] Saved processed dataset to: {file_path}")
    console.print()


def generate_sample_dataset(symbol: str, samples: int):
    """Generate synthetic parquet data for quick experiments"""
    with server_module_path():
        from src.data.loader import create_sample_data

    console.print(f"\n[bold cyan]Generating {samples:,} synthetic candles[/bold cyan]")
    df = create_sample_data(symbol=symbol, n_samples=samples, save_dir="data")

    console.print(f"[green]OK[/green] Sample data ready with {len(df):,} rows")
    console.print()

# ============================================================================
# Training
# ============================================================================

def train(args: argparse.Namespace):
    """Dispatch training to the appropriate script"""
    model_key = args.model.lower()
    config = MODEL_REGISTRY.get(model_key)

    if not config:
        console.print(f"[red]Unknown model '{args.model}'. Available: {', '.join(MODEL_REGISTRY)}[/red]\n")
        return

    console.print(f"\n[bold cyan]Training {config['label']} model[/bold cyan]\n")

    iterations = f"{args.episodes} episodes" if model_key == "d3qn" else f"{args.timesteps:,} timesteps"

    config_table = Table(show_header=False, box=box.SIMPLE)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="white")
    config_table.add_row("Symbol", args.symbol)
    config_table.add_row("Timeframe", args.timeframe)
    config_table.add_row("Model", config["label"])
    config_table.add_row("Iterations", iterations)
    config_table.add_row("Learning Rate", str(args.lr))
    config_table.add_row("Reward", args.reward.upper())
    config_table.add_row("Sample Data", "Yes" if args.sample_data else "No")
    config_table.add_row("Checkpoint Dir", args.checkpoint_dir)

    console.print(config_table)
    console.print()

    server_dir = Path(__file__).parent / "server"
    cmd = [sys.executable, "-m", config["module"]]

    for flag, attr in config["args"]:
        value = getattr(args, attr, None)
        if value is None:
            continue
        cmd.extend([flag, str(value)])

    for flag, attr in config.get("flags", []):
        if getattr(args, attr, False):
            cmd.append(flag)

    try:
        console.print("[cyan]Starting training...[/cyan]")
        console.print("[dim]" + "=" * 70 + "[/dim]\n")

        result = subprocess.run(cmd, cwd=server_dir)

        console.print("\n[dim]" + "=" * 70 + "[/dim]")
        if result.returncode == 0:
            console.print("[green]Training completed[/green]\n")
        else:
            console.print(f"[red]Training failed with exit code {result.returncode}[/red]\n")
            sys.exit(result.returncode)
    except Exception as exc:
        console.print(f"\n[red]Training failed: {exc}[/red]\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# Backtesting
# ============================================================================

def backtest(
    model_path: str,
    model_type: str,
    sample_data: bool,
    symbol: str,
    timeframe: str,
    initial_balance: float,
    max_leverage: float,
    reward_type: str,
    window_size: int
):
    """Run backtest directly inside the CLI"""
    model_key = model_type.lower()
    if model_key not in MODEL_REGISTRY:
        console.print(f"[red]Unknown model '{model_type}'[/red]\n")
        return

    path = Path(model_path)
    if not path.exists():
        console.print(f"[red]Model file not found:[/red] {model_path}\n")
        return

    console.print(f"\n[bold cyan]Backtesting {MODEL_REGISTRY[model_key]['label']}[/bold cyan]")
    console.print(f"[dim]Model file: {model_path}[/dim]\n")

    with server_module_path():
        from src.data.loader import load_market_data, prepare_env_data, create_sample_data
        from src.policy.trading_env import TradingEnv
        from src.policy.wrappers import ContinuousActionWrapper
        if model_key == "d3qn":
            from src.agent.d3qn_agent import D3QNAgent
        else:
            from stable_baselines3 import PPO, SAC, TD3

    if sample_data:
        df = create_sample_data(symbol=symbol, n_samples=5000, save_dir="data")
    else:
        df = load_market_data(symbol=symbol, timeframe=timeframe)

    env_data = prepare_env_data(df, add_indicators=True, normalize=False)
    env = TradingEnv(
        data=env_data,
        initial_balance=initial_balance,
        max_leverage=max_leverage,
        window_size=window_size,
        reward_type=reward_type
    )

    if model_key in {"sac", "td3"}:
        env = ContinuousActionWrapper(env)

    base_env = unwrap_env(env)

    if model_key == "d3qn":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = base_env.action_space.n

        agent = D3QNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.00025,
            gamma=0.995
        )
        agent.load(str(path))

        def select_action(observation):
            return agent.select_action(observation.flatten(), training=False)
    else:
        model_classes = {"ppo": PPO, "sac": SAC, "td3": TD3}
        ModelClass = model_classes[model_key]
        model = ModelClass.load(str(path))

        def select_action(observation):
            action, _ = model.predict(observation, deterministic=True)
            return action

    state, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    equity_curve = []
    last_info = None

    while not done:
        action_raw = select_action(state)
        env_action = action_to_env(model_key, action_raw)

        state, reward, terminated, truncated, info = env.step(env_action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        last_info = info
        equity_curve.append(info["equity"])

    if not last_info:
        console.print("[red]Backtest produced no results.[/red]\n")
        return

    equity_array = np.array(equity_curve) if equity_curve else np.array([initial_balance])
    if len(equity_array) > 1:
        returns = np.diff(equity_array) / equity_array[:-1]
    else:
        returns = np.array([])

    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)) if len(returns) else 0.0
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    max_drawdown = float(abs(drawdown.min()) * 100) if len(drawdown) else 0.0

    final_balance = last_info["balance"]
    total_return_pct = (final_balance - initial_balance) / initial_balance * 100

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Initial Balance", f"${initial_balance:,.2f}")
    table.add_row("Final Balance", f"${final_balance:,.2f}")
    table.add_row("Total Return", f"{total_return_pct:+.2f}%")
    table.add_row("Total Reward", f"{total_reward:.2f}")
    table.add_row("Win Rate", f"{last_info['win_rate'] * 100:.2f}%")
    table.add_row("Trades", str(last_info["total_trades"]))
    table.add_row("Sharpe Ratio", f"{sharpe:.3f}")
    table.add_row("Max Drawdown", f"{max_drawdown:.2f}%")
    table.add_row("Steps", str(steps))

    console.print(table)
    console.print()

    log_payload = {
        "meta": {
            "symbol": symbol,
            "timeframe": timeframe,
            "model": model_path,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat()
        },
        "summary": {
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "total_return_pct": total_return_pct,
            "total_reward": total_reward,
            "trades": last_info["total_trades"],
            "win_rate": last_info["win_rate"] * 100,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown
        },
        "trades": serialize_trades(base_env.trade_log)
    }
    log_file = save_json_log(BACKTEST_LOG_DIR, "backtest", log_payload)
    console.print(f"[dim]Backtest log saved to: {log_file}[/dim]")
    console.print()


def paper_trade(
    model_path: str,
    model_type: str,
    symbol: str,
    timeframe: str,
    interval: float,
    steps: int,
    initial_balance: float,
    max_leverage: float,
    reward_type: str,
    window_size: int
):
    """Run a live-style paper trading loop"""
    model_key = model_type.lower()
    if model_key not in MODEL_REGISTRY:
        console.print(f"[red]Unknown model '{model_type}'[/red]\n")
        return

    path = Path(model_path)
    if not path.exists():
        console.print(f"[red]Model file not found:[/red] {model_path}\n")
        return

    console.print(f"\n[bold cyan]Paper trading {MODEL_REGISTRY[model_key]['label']} on {symbol} ({timeframe})[/bold cyan]")
    console.print(f"[yellow]Note: Actions will occur at candle close only (timeframe: {timeframe})[/yellow]\n")

    import pandas as pd
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True})

    # Parse timeframe to seconds
    timeframe_to_seconds = {
        '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
        '1h': 3600, '4h': 14400, '1d': 86400
    }
    candle_duration = timeframe_to_seconds.get(timeframe, 60)

    # Initial data fetch
    limit = max(window_size * 2, 1000)
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Track last candle timestamp
    last_candle_timestamp = df["timestamp"].iloc[-1]

    with server_module_path():
        from src.data.loader import prepare_env_data
        from src.policy.trading_env import TradingEnv
        from src.policy.wrappers import ContinuousActionWrapper
        if model_key == "d3qn":
            from src.agent.d3qn_agent import D3QNAgent
        else:
            from stable_baselines3 import PPO, SAC, TD3

    env_data = prepare_env_data(df, add_indicators=True, normalize=False)
    env = TradingEnv(
        data=env_data,
        initial_balance=initial_balance,
        max_leverage=max_leverage,
        window_size=window_size,
        reward_type=reward_type,
        streaming_mode=True  # Enable streaming mode for live paper trading
    )
    if model_key in {"sac", "td3"}:
        env = ContinuousActionWrapper(env)

    base_env = unwrap_env(env)

    if model_key == "d3qn":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = base_env.action_space.n
        agent = D3QNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.00025,
            gamma=0.995,
            device=device
        )
        agent.load(str(path))

        def select_action(observation):
            return agent.select_action(observation.flatten(), training=False)
    else:
        model_classes = {"ppo": PPO, "sac": SAC, "td3": TD3}
        ModelClass = model_classes[model_key]
        model = ModelClass.load(str(path))

        def select_action(observation):
            action, _ = model.predict(observation, deterministic=True)
            return action

    state, _ = env.reset()
    total_reward = 0.0
    last_info = None
    equity_curve = []

    def render_panel(step_idx: int, price: float, action_name: str, info: dict, reward_value: float):
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_row("Step", str(step_idx))
        table.add_row("Action", action_name)
        table.add_row("Price", f"${price:,.2f}")
        table.add_row("Balance", f"${info['balance']:,.2f}")
        table.add_row("Equity", f"${info['equity']:,.2f}")
        table.add_row("Trades", str(info["total_trades"]))
        table.add_row("Reward", f"{reward_value:+.4f}")
        return Panel(table, border_style="bright_cyan", title="Paper Trading")

    with Live(render_panel(0, float(base_env.data[base_env.current_step, 3]), "HOLD", {"balance": initial_balance, "equity": initial_balance, "total_trades": 0}, 0.0), refresh_per_second=4) as live:
        action_count = 0
        candle_count = 0

        while action_count < steps:
            # Fetch latest candles
            try:
                new_candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                new_df = pd.DataFrame(new_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms")
                current_candle_timestamp = new_df["timestamp"].iloc[-1]

                # Check if new candle has formed
                new_candle_formed = current_candle_timestamp > last_candle_timestamp

                if new_candle_formed:
                    candle_count += 1
                    last_candle_timestamp = current_candle_timestamp

                    # Log new candle detection
                    if action_count == 0:
                        console.print(f"[green]New candle detected at {current_candle_timestamp}[/green]")

                    # Update environment data with new candles
                    env_data = prepare_env_data(new_df, add_indicators=True, normalize=False)

                    # Calculate how many new candles were added
                    old_n_steps = base_env.n_steps
                    new_n_steps = len(env_data)

                    # Update environment data
                    base_env.data = env_data
                    base_env.n_steps = new_n_steps

                    # Adjust current_step to point to the latest candle
                    # Keep it at the end of the data for streaming
                    if base_env.current_step >= old_n_steps - 1:
                        base_env.current_step = new_n_steps - 1

                    # Perform action only on new candle (one action per candle)
                    action_raw = select_action(state)
                    env_action = action_to_env(model_key, action_raw)
                    action_idx = action_to_index(model_key, action_raw)

                    state, reward, terminated, truncated, info = env.step(env_action)
                    last_info = info
                    total_reward += reward
                    equity_curve.append(info["equity"])
                    action_count += 1

                    price_idx = min(base_env.current_step - 1, base_env.data.shape[0] - 1)
                    price = float(base_env.data[price_idx, 3])

                    panel = render_panel(action_count, price, ACTION_NAMES[action_idx], info, reward)
                    live.update(panel)

                    if terminated or truncated:
                        break
                else:
                    # No new candle yet, just update display with current price
                    price_idx = min(base_env.current_step - 1, base_env.data.shape[0] - 1)
                    price = float(new_df["close"].iloc[-1])
                    current_info = last_info or {"balance": initial_balance, "equity": initial_balance, "total_trades": 0}
                    panel = render_panel(action_count, price, "WAITING", current_info, 0.0)
                    live.update(panel)

            except Exception as e:
                console.print(f"\n[red]Error fetching data: {e}[/red]")
                break

            # Sleep for interval before next check
            time.sleep(max(0.1, interval))

    if not last_info:
        console.print("[red]Paper trading session ended without trades.[/red]\n")
        return

    equity_array = np.array(equity_curve) if equity_curve else np.array([initial_balance])
    if len(equity_array) > 1:
        returns = np.diff(equity_array) / equity_array[:-1]
    else:
        returns = np.array([])

    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)) if len(returns) else 0.0
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    max_drawdown = float(abs(drawdown.min()) * 100) if len(drawdown) else 0.0

    final_balance = last_info["balance"]
    total_return_pct = (final_balance - initial_balance) / initial_balance * 100

    console.print("\n[bold green]Session Summary[/bold green]")
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Steps", str(len(equity_curve)))
    table.add_row("Total Reward", f"{total_reward:.4f}")
    table.add_row("Final Balance", f"${final_balance:,.2f}")
    table.add_row("Return", f"{total_return_pct:+.2f}%")
    table.add_row("Trades", str(last_info["total_trades"]))
    table.add_row("Win Rate", f"{last_info['win_rate'] * 100:.2f}%")
    table.add_row("Sharpe", f"{sharpe:.3f}")
    table.add_row("Max Drawdown", f"{max_drawdown:.2f}%")

    console.print(table)
    console.print()

    paper_payload = {
        "meta": {
            "symbol": symbol,
            "timeframe": timeframe,
            "model": model_path,
            "model_type": model_type,
            "steps": len(equity_curve),
            "timestamp": datetime.now().isoformat()
        },
        "summary": {
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "total_return_pct": total_return_pct,
            "total_reward": total_reward,
            "trades": last_info["total_trades"],
            "win_rate": last_info["win_rate"] * 100,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown
        },
        "trades": serialize_trades(base_env.trade_log)
    }
    paper_log_file = save_json_log(PAPER_LOG_DIR, "paper", paper_payload)
    console.print(f"[dim]Paper trading log saved to: {paper_log_file}[/dim]")
    console.print()

# ============================================================================
# Models Management
# ============================================================================

def list_models():
    """List available trained models"""
    console.print("\n[bold cyan]Available models[/bold cyan]\n")

    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        console.print("[yellow]No checkpoints directory found[/yellow]\n")
        return

    models = list(checkpoints_dir.glob("*.pth"))

    if not models:
        console.print("[yellow]No models found[/yellow]\n")
        return

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("#", style="dim")
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Created", style="dim")

    for i, model in enumerate(sorted(models), 1):
        size = model.stat().st_size
        size_str = f"{size / 1024 / 1024:.2f} MB"
        created = model.stat().st_mtime
        from datetime import datetime
        created_str = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M")
        table.add_row(str(i), model.name, size_str, created_str)

    console.print(table)
    console.print()

# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CRYPPO - Deep RL Crypto Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # System check
    subparsers.add_parser("check", help="Check system requirements")

    # Data commands
    data_parser = subparsers.add_parser("data", help="Data management")
    data_subparsers = data_parser.add_subparsers(dest="data_command")

    download_parser = data_subparsers.add_parser("download", help="Download market data")
    download_parser.add_argument("--symbol", required=True, help="Trading symbol (e.g. BTCUSDT)")
    download_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    download_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    download_parser.add_argument("--timeframe", default="1h", help="Timeframe (default: 1h)")

    data_subparsers.add_parser("list", help="List available data files")
    prepare_parser = data_subparsers.add_parser("prepare", help="Prepare numpy datasets")
    prepare_parser.add_argument("--symbol", required=True, help="Trading symbol")
    prepare_parser.add_argument("--timeframe", default="1h", help="Timeframe to load")
    prepare_parser.add_argument("--window", type=int, default=72, help="Window size for sequences")
    prepare_parser.add_argument("--train-ratio", type=float, default=0.7, dest="train_ratio")
    prepare_parser.add_argument("--val-ratio", type=float, default=0.15, dest="val_ratio")
    prepare_parser.add_argument("--output", default="data/processed", help="Directory for npz file")
    prepare_parser.add_argument("--no-indicators", action="store_true", help="Skip indicator engineering")
    prepare_parser.add_argument("--no-normalize", action="store_true", help="Skip normalization")
    prepare_parser.add_argument("--sample-data", action="store_true", help="Use synthetic sample data")
    prepare_parser.add_argument("--samples", type=int, default=10000, help="Number of synthetic samples")

    sample_parser = data_subparsers.add_parser("sample", help="Generate synthetic parquet data")
    sample_parser.add_argument("--symbol", default="BTCUSDT", help="Symbol label for file name")
    sample_parser.add_argument("--samples", type=int, default=2000, help="Number of rows")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train RL model")
    train_parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    train_parser.add_argument("--timeframe", default="1h", help="Historical timeframe")
    train_parser.add_argument("--model", default="d3qn", choices=list(MODEL_REGISTRY.keys()),
                              help="RL algorithm to use")
    train_parser.add_argument("--episodes", type=int, default=100, help="Episodes (D3QN)")
    train_parser.add_argument("--timesteps", type=int, default=50000, help="Timesteps (PPO/SAC/TD3)")
    train_parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=0.00025,
                              help="Learning rate")
    train_parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    train_parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout steps")
    train_parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    train_parser.add_argument("--clip-range", dest="clip_range", type=float, default=0.2,
                              help="PPO clip range")
    train_parser.add_argument("--max-steps", type=int, default=10000, help="Max env steps per episode (D3QN)")
    train_parser.add_argument("--learning-starts", type=int, default=1000,
                              help="SAC/TD3 warmup steps")
    train_parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")
    train_parser.add_argument("--policy-delay", type=int, default=2, help="TD3 policy delay")
    train_parser.add_argument("--reward", default="sharpe",
                              choices=["profit", "return", "sharpe", "sortino"],
                              help="Reward type")
    train_parser.add_argument("--sample-data", action="store_true", help="Use synthetic sample data")
    train_parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    train_parser.add_argument("--leverage", type=float, default=3.0, help="Max leverage")
    train_parser.add_argument("--window", type=int, default=72, help="Observation window")
    train_parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    train_parser.add_argument("--n-eval-episodes", type=int, default=5, help="Eval episodes")
    train_parser.add_argument("--save-freq", type=int, default=10000, help="Checkpoint frequency")
    train_parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    train_parser.add_argument("--log-dir", default="logs", help="Log directory")
    train_parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--model", required=True, help="Path to model file")
    backtest_parser.add_argument("--model-type", default="d3qn", choices=list(MODEL_REGISTRY.keys()),
                                 help="Specify algorithm for the model file")
    backtest_parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to evaluate")
    backtest_parser.add_argument("--timeframe", default="1h", help="Timeframe for evaluation data")
    backtest_parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    backtest_parser.add_argument("--leverage", type=float, default=3.0, help="Max leverage")
    backtest_parser.add_argument("--window", type=int, default=72, help="Observation window")
    backtest_parser.add_argument("--reward", default="sharpe",
                                 choices=["profit", "return", "sharpe", "sortino"],
                                 help="Reward shaping for env")
    backtest_parser.add_argument("--sample-data", action="store_true", help="Use sample data")

    paper_parser = subparsers.add_parser("paper", help="Live-style paper trading")
    paper_parser.add_argument("--model", required=True, help="Path to trained model")
    paper_parser.add_argument("--model-type", default="d3qn", choices=list(MODEL_REGISTRY.keys()),
                              help="Algorithm the model was trained with")
    paper_parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    paper_parser.add_argument("--timeframe", default="1m", help="Live timeframe (default 1m)")
    paper_parser.add_argument("--interval", type=float, default=2.0,
                              help="Seconds between updates")
    paper_parser.add_argument("--steps", type=int, default=120,
                              help="Maximum steps to simulate")
    paper_parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    paper_parser.add_argument("--leverage", type=float, default=3.0, help="Max leverage")
    paper_parser.add_argument("--window", type=int, default=72, help="Observation window")
    paper_parser.add_argument("--reward", default="sharpe",
                              choices=["profit", "return", "sharpe", "sortino"],
                              help="Reward shaping strategy")

    # Models command
    subparsers.add_parser("models", help="List trained models")

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Route commands
    if args.command == "check":
        check_system()

    elif args.command == "data":
        if args.data_command == "download":
            download_data(args.symbol, args.start, args.end, args.timeframe)
        elif args.data_command == "list":
            list_data()
        elif args.data_command == "prepare":
            prepare_data(
                symbol=args.symbol,
                timeframe=args.timeframe,
                window=args.window,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                output_dir=args.output,
                add_indicators=not args.no_indicators,
                normalize=not args.no_normalize,
                sample_data=args.sample_data,
                sample_size=args.samples
            )
        elif args.data_command == "sample":
            generate_sample_dataset(args.symbol, args.samples)
        else:
            console.print("[yellow]Use: cryppo data [download|list|prepare|sample][/yellow]\n")

    elif args.command == "train":
        train(args)

    elif args.command == "backtest":
        backtest(
            model_path=args.model,
            model_type=args.model_type,
            sample_data=args.sample_data,
            symbol=args.symbol,
            timeframe=args.timeframe,
            initial_balance=args.balance,
            max_leverage=args.leverage,
            reward_type=args.reward,
            window_size=args.window
        )

    elif args.command == "paper":
        paper_trade(
            model_path=args.model,
            model_type=args.model_type,
            symbol=args.symbol,
            timeframe=args.timeframe,
            interval=args.interval,
            steps=args.steps,
            initial_balance=args.balance,
            max_leverage=args.leverage,
            reward_type=args.reward,
            window_size=args.window
        )

    elif args.command == "models":
        list_models()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
