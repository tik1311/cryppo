#!/usr/bin/env python3
"""
CRYPPO - Deep RL Crypto Trading System
All-in-One CLI (Single Entry Point)
Run: python main.py
"""

import sys
import os
import torch
import numpy as np
import time
import gc
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

try:
    import questionary
    from questionary import Style
except ImportError:
    print("Installing questionary...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "questionary>=2.0.0"])
    import questionary
    from questionary import Style

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.align import Align
from rich.text import Text

from src.data.loader import load_multi_timeframe_data, prepare_multi_timeframe_env_data, download_and_save_data
from src.data.realtime_fetcher import RealtimeFetcher
from src.policy.trading_env import TradingEnv
from src.agent.d3qn_agent import D3QNAgent

console = Console()

# Custom style for questionary
custom_style = Style([
    ('qmark', 'fg:#673ab7 bold'),
    ('question', 'bold'),
    ('answer', 'fg:#f44336 bold'),
    ('pointer', 'fg:#673ab7 bold'),
    ('highlighted', 'fg:#673ab7 bold'),
    ('selected', 'fg:#cc5454'),
])


def print_logo():
    """Print CRYPPO logo"""
    logo_lines = [
        "  ██████╗██████╗ ██╗   ██╗██████╗ ██████╗  ██████╗ ",
        " ██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗██╔══██╗██╔═══██╗",
        " ██║     ██████╔╝ ╚████╔╝ ██████╔╝██████╔╝██║   ██║",
        " ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝ ██╔═══╝ ██║   ██║",
        " ╚██████╗██║  ██║   ██║   ██║     ██║     ╚██████╔╝",
        "  ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚═╝      ╚═════╝ ",
        "",
        "         Deep RL Crypto Trading System",
    ]

    logo_text = Text()
    for i, line in enumerate(logo_lines):
        if i < 6:
            logo_text.append(line + "\n", style="bold cyan")
        elif i == 6:
            logo_text.append(line + "\n")
        else:
            logo_text.append(line, style="bold white")

    centered_logo = Align.center(logo_text)
    panel = Panel(
        centered_logo,
        border_style="cyan",
        padding=(1, 2),
        title="[bold cyan]CRYPPO v2.0[/bold cyan]",
        subtitle="[dim]D3QN + Multi-Timeframe[/dim]",
        expand=True
    )
    console.print(panel)


def setup_logger(mode: str) -> logging.Logger:
    """Setup logger for train/backtest/live modes"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{mode}.log"

    logger = logging.getLogger(f"cryppo_{mode}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info(f"CRYPPO - {mode.upper()} Mode")
    logger.info(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    return logger


def cleanup_resources():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ==================== SYSTEM CHECK ====================
def system_check():
    """Perform system checks"""
    console.print(Panel("System Check", style="bold cyan"))

    console.print(f"Python: [green]{sys.version.split()[0]}[/green]")
    console.print(f"PyTorch: [green]{torch.__version__}[/green]")

    if torch.cuda.is_available():
        console.print(f"Device: [green]CUDA ({torch.cuda.get_device_name(0)})[/green]")
    else:
        console.print("Device: [yellow]CPU[/yellow]")

    data_dir = Path("data")
    if data_dir.exists():
        files = list(data_dir.glob("*.parquet"))
        console.print(f"Data Directory: [green]Found ({len(files)} files)[/green]")
    else:
        console.print("Data Directory: [yellow]Not Found[/yellow]")

    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        models = list(checkpoints_dir.glob("*.pth"))
        console.print(f"Models: [green]Found ({len(models)} models)[/green]")
    else:
        console.print("Models: [yellow]Not Found[/yellow]")


# ==================== DOWNLOAD DATA ====================
def download_data():
    """Download market data"""
    console.print(Panel("Download Market Data", style="bold yellow"))

    symbol = Prompt.ask("Symbol", default="BTCUSDT")

    default_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    default_end = datetime.now().strftime("%Y-%m-%d")

    start = Prompt.ask("Start Date (YYYY-MM-DD)", default=default_start)
    end = Prompt.ask("End Date (YYYY-MM-DD)", default=default_end)

    console.print(f"\n[cyan]Downloading {symbol} (1h + 1m) from {start} to {end}...[/cyan]")

    # Download 1h
    console.print("\n[dim]Downloading 1h timeframe...[/dim]")
    download_and_save_data(symbol, "1h", start, end)

    # Download 1m
    console.print("\n[dim]Downloading 1m timeframe...[/dim]")
    download_and_save_data(symbol, "1m", start, end)

    console.print("\n[green]Download complete![/green]")


# ==================== LIST DATA ====================
def list_data():
    """List downloaded data files"""
    console.print(Panel("Available Data Files", style="bold cyan"))

    data_dir = Path("data")
    if not data_dir.exists():
        console.print("[yellow]No data directory found[/yellow]")
        return

    files = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.csv"))

    if not files:
        console.print("[yellow]No data files found[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Modified", style="yellow")

    for file in sorted(files):
        size = file.stat().st_size / 1024
        mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        size_str = f"{size:.1f} KB" if size < 1024 else f"{size/1024:.1f} MB"
        table.add_row(file.name, size_str, mtime)

    console.print(table)


# ==================== LIST MODELS ====================
def list_models():
    """List trained models"""
    console.print(Panel("Trained Models", style="bold cyan"))

    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        console.print("[yellow]No checkpoints directory found[/yellow]")
        return

    files = list(checkpoints_dir.glob("*.pth"))

    if not files:
        console.print("[yellow]No model files found[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Created", style="yellow")

    for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
        size = file.stat().st_size / 1024
        mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        size_str = f"{size:.1f} KB" if size < 1024 else f"{size/1024:.1f} MB"
        table.add_row(file.name, size_str, mtime)

    console.print(table)


# ==================== TRAIN MODEL ====================
def train_model():
    """Train a new model"""
    console.print(Panel("Train Model", style="bold blue"))

    symbol = Prompt.ask("Symbol", default="BTCUSDT")
    episodes = int(Prompt.ask("Episodes", default="10"))
    balance = float(Prompt.ask("Initial Balance ($)", default="10000"))

    # Training parameters (optimized for trend-following)
    window_size = 36
    tp_mult = 2.0  # TP at 2x ATR (realistic target)
    sl_mult = 1.0  # SL at 1x ATR (tight stop)
    min_bars = 1   # Allow frequent trading
    lr = 0.0003    # Slightly higher LR for faster learning
    gamma = 0.99   # Higher discount for long-term rewards
    batch_size = 32
    update_freq = 4

    logger = setup_logger("train")

    console.print("\n[cyan]Loading data...[/cyan]")

    try:
        df_main, df_detail = load_multi_timeframe_data(symbol=symbol, main_timeframe="1h", detail_timeframe="1m")
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        console.print("[yellow]Please download data first![/yellow]")
        return

    console.print(f"[green]Loaded {len(df_main)} main candles, {len(df_detail)} detail candles[/green]")

    # Prepare data
    full_data = prepare_multi_timeframe_env_data(df_main, df_detail, main_timeframe="1h", detail_timeframe="1m")

    # Split: Train on all except last 3 months
    total_steps = len(full_data['main'])
    test_size = min(2160, int(total_steps * 0.15))
    train_end = total_steps - test_size

    train_data = {
        'main': full_data['main'][:train_end],
        'detail': full_data['detail'][:train_end]
    }

    console.print(f"[cyan]Training on {len(train_data['main']):,} steps ({len(train_data['main'])/24:.0f} days)[/cyan]")

    # Init Environment
    env = TradingEnv(
        train_data,
        initial_balance=balance,
        enable_bust_protection=False,
        window_size=window_size,
        atr_tp_multiplier=tp_mult,
        atr_sl_multiplier=sl_mult,
        min_bars_between_trades=min_bars,
        reward_type="pnl",
        logger=logger
    )

    # Init Agent
    obs_shape = env.observation_space['main'].shape
    detail_shape = env.observation_space['detail'].shape

    agent = D3QNAgent(
        window_size=obs_shape[0],
        n_main_features=obs_shape[1],
        n_detail_features=detail_shape[2],
        detail_len=detail_shape[1],
        action_dim=env.action_space.n,
        learning_rate=lr,
        gamma=gamma,
        logger=logger
    )
    agent.batch_size = batch_size

    console.print("\n[green]Starting Training...[/green]\n")

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        total_episode_steps = env.n_steps - env.current_step

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Episode {episode+1}/{episodes}", total=total_episode_steps)

            while not done:
                mask = env.get_action_mask()
                action = agent.select_action(state, training=True, action_mask=mask, step=env.current_step)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store_transition(state, action, reward, next_state, done)

                step_count += 1
                if step_count % update_freq == 0:
                    agent.update()

                state = next_state
                total_reward += reward
                progress.update(task, advance=1)

        win_rate = (env.winning_trades / max(1, env.total_trades)) * 100
        total_return = ((env.balance - env.initial_balance) / env.initial_balance) * 100

        console.print(f"Episode {episode+1}/{episodes} | Reward: {total_reward:.2f} | Balance: ${env.balance:.2f} | Trades: {env.total_trades} | WR: {win_rate:.1f}%")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/d3qn_{symbol}_1h.pth"
    agent.save(save_path)

    console.print(f"\n[bold green]Training Complete! Model saved to {save_path}[/bold green]")
    cleanup_resources()


# ==================== BACKTEST ====================
def backtest():
    """Run backtest on test set"""
    console.print(Panel("Backtest Model", style="bold magenta"))

    symbol = Prompt.ask("Symbol", default="BTCUSDT")
    balance = float(Prompt.ask("Initial Balance ($)", default="10000"))

    logger = setup_logger("backtest")

    console.print("\n[cyan]Loading data...[/cyan]")

    try:
        df_main, df_detail = load_multi_timeframe_data(symbol=symbol, main_timeframe="1h", detail_timeframe="1m")
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        return

    full_data = prepare_multi_timeframe_env_data(df_main, df_detail, main_timeframe="1h", detail_timeframe="1m")

    # Use last 3 months for testing
    total_steps = len(full_data['main'])
    test_size = min(2160, int(total_steps * 0.15))
    train_end = total_steps - test_size

    test_data = {
        'main': full_data['main'][train_end:],
        'detail': full_data['detail'][train_end:]
    }

    console.print(f"[cyan]Testing on {len(test_data['main']):,} steps (last 3 months)[/cyan]")

    # Init Environment (same params as training)
    env = TradingEnv(
        test_data,
        initial_balance=balance,
        window_size=36,
        atr_tp_multiplier=2.0,
        atr_sl_multiplier=1.0,
        min_bars_between_trades=1,
        reward_type="pnl",
        logger=logger
    )

    # Load Agent
    obs_shape = env.observation_space['main'].shape
    detail_shape = env.observation_space['detail'].shape

    agent = D3QNAgent(
        window_size=obs_shape[0],
        n_main_features=obs_shape[1],
        n_detail_features=detail_shape[2],
        detail_len=detail_shape[1],
        action_dim=env.action_space.n,
        logger=logger
    )

    model_path = f"checkpoints/d3qn_{symbol}_1h.pth"
    if os.path.exists(model_path):
        agent.load(model_path)
        console.print(f"[green]Loaded model from {model_path}[/green]")
    else:
        console.print(f"[red]Model not found at {model_path}![/red]")
        return

    # Run backtest
    state, _ = env.reset()
    done = False
    agent.epsilon = 0.0  # No exploration

    total_episode_steps = env.n_steps - env.current_step

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Backtesting...", total=total_episode_steps)

        while not done:
            mask = env.get_action_mask()
            action = agent.select_action(state, training=False, action_mask=mask, step=env.current_step)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            progress.update(task, advance=1)

    # Results
    win_rate = (env.winning_trades / max(1, env.total_trades)) * 100
    total_return = ((env.balance - env.initial_balance) / env.initial_balance) * 100

    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Final Balance", f"${env.balance:.2f}")
    table.add_row("Total Return", f"{total_return:.2f}%")
    table.add_row("Total Trades", str(env.total_trades))
    table.add_row("Winning Trades", str(env.winning_trades))
    table.add_row("Win Rate", f"{win_rate:.2f}%")
    table.add_row("Total PnL", f"${env.total_pnl:.2f}")

    console.print(table)
    console.print(f"[dim]Log saved to: logs/backtest.log[/dim]")

    cleanup_resources()


# ==================== PAPER TRADE (Historical) ====================
def paper_trade_historical():
    """Paper trade on historical test data"""
    console.print(Panel("Paper Trade (Historical Simulation)", style="bold red"))

    symbol = Prompt.ask("Symbol", default="BTCUSDT")
    balance = float(Prompt.ask("Initial Balance ($)", default="10000"))

    logger = setup_logger("live")

    console.print("\n[cyan]Loading data...[/cyan]")

    try:
        df_main, df_detail = load_multi_timeframe_data(symbol=symbol, main_timeframe="1h", detail_timeframe="1m")
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        return

    full_data = prepare_multi_timeframe_env_data(df_main, df_detail, main_timeframe="1h", detail_timeframe="1m")

    # Use last 3 months
    total_steps = len(full_data['main'])
    test_size = min(2160, int(total_steps * 0.15))
    train_end = total_steps - test_size

    test_data = {
        'main': full_data['main'][train_end:],
        'detail': full_data['detail'][train_end:]
    }

    console.print(f"[cyan]Simulating on {len(test_data['main']):,} steps[/cyan]")

    env = TradingEnv(
        test_data,
        initial_balance=balance,
        streaming_mode=True,
        window_size=36,
        atr_tp_multiplier=3.0,
        atr_sl_multiplier=1.5,
        min_bars_between_trades=2,
        logger=logger
    )

    obs_shape = env.observation_space['main'].shape
    detail_shape = env.observation_space['detail'].shape

    agent = D3QNAgent(
        window_size=obs_shape[0],
        n_main_features=obs_shape[1],
        n_detail_features=detail_shape[2],
        detail_len=detail_shape[1],
        action_dim=env.action_space.n,
        logger=logger
    )

    model_path = f"checkpoints/d3qn_{symbol}_1h.pth"
    if os.path.exists(model_path):
        agent.load(model_path)
        console.print(f"[green]Loaded model from {model_path}[/green]")
    else:
        console.print(f"[red]Model not found![/red]")
        return

    state, _ = env.reset()
    done = False
    agent.epsilon = 0.0

    console.print("\n[yellow]Press Ctrl+C to stop[/yellow]\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Running simulation...", total=env.n_steps)

            while not done:
                mask = env.get_action_mask()
                action = agent.select_action(state, training=False, action_mask=mask, step=env.current_step)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                progress.update(task, advance=1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Simulation stopped by user.[/yellow]")

    win_rate = (env.winning_trades / max(1, env.total_trades)) * 100
    total_return = ((env.balance - env.initial_balance) / env.initial_balance) * 100

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"Final Balance: ${env.balance:.2f}")
    console.print(f"Total Return: {total_return:.2f}%")
    console.print(f"Total Trades: {env.total_trades}")
    console.print(f"Win Rate: {win_rate:.2f}%")

    cleanup_resources()


# ==================== LIVE PAPER TRADE ====================
def live_paper_trade():
    """Real-time paper trading with live data from Binance"""
    console.print(Panel("Live Paper Trade (Real-Time)", style="bold green"))

    symbol = Prompt.ask("Symbol", default="BTCUSDT")
    balance = float(Prompt.ask("Initial Balance ($)", default="10000"))
    timeframe = Prompt.ask("Timeframe (for testing use 1m)", default="1m")

    logger = setup_logger("live")

    # Timeframe settings
    tf_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}
    tf_seconds = tf_map.get(timeframe, 60)
    fetch_interval = 5  # Fetch every 5 seconds

    console.print(f"\n[cyan]Starting Live Paper Trade...[/cyan]")
    console.print(f"[dim]Timeframe: {timeframe} | Fetch interval: {fetch_interval}s | Action on candle close[/dim]")

    # Load model
    model_path = f"checkpoints/d3qn_{symbol}_1h.pth"
    if not os.path.exists(model_path):
        console.print(f"[red]Model not found at {model_path}![/red]")
        return

    fetcher = RealtimeFetcher(symbol=symbol, main_timeframe=timeframe, detail_timeframe="1m", window_size=36)

    # Initialize with first data fetch
    df_main, df_detail = fetcher.get_latest_data()
    if df_main is None:
        console.print("[red]Failed to fetch initial data![/red]")
        return

    console.print(f"[green]Fetched {len(df_main)} candles[/green]")

    # State tracking
    current_balance = balance
    position = None  # None, 'long', 'short'
    entry_price = 0.0
    position_size = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    total_trades = 0
    winning_trades = 0
    total_pnl = 0.0
    last_candle_time = None

    # Build agent (just for inference)
    agent = D3QNAgent(
        window_size=36,
        n_main_features=26,  # Will be adjusted based on actual data
        n_detail_features=6,
        detail_len=60,
        action_dim=3,
        logger=logger
    )
    agent.load(model_path)
    agent.epsilon = 0.0

    console.print("\n[yellow]Press Ctrl+C to stop[/yellow]\n")

    try:
        while True:
            # Fetch latest data
            df_main, df_detail = fetcher.get_latest_data()

            if df_main is None or len(df_main) < 36:
                console.print("[yellow]Waiting for data...[/yellow]")
                time.sleep(fetch_interval)
                continue

            current_price = float(df_main['close'].iloc[-1])
            current_time = df_main['timestamp'].iloc[-1]

            # Check if new candle closed
            candle_just_closed = (last_candle_time is None or current_time != last_candle_time)

            # Check position status (TP/SL)
            if position:
                # Check TP/SL
                if position == 'long':
                    if current_price >= take_profit:
                        pnl = (take_profit - entry_price) * position_size
                        current_balance += pnl
                        total_pnl += pnl
                        total_trades += 1
                        if pnl > 0:
                            winning_trades += 1
                        console.print(f"[green]TP HIT! PnL: ${pnl:.2f} | Balance: ${current_balance:.2f}[/green]")
                        logger.info(f"TP HIT @ {take_profit:.2f} | PnL: ${pnl:.2f}")
                        position = None
                    elif current_price <= stop_loss:
                        pnl = (stop_loss - entry_price) * position_size
                        current_balance += pnl
                        total_pnl += pnl
                        total_trades += 1
                        console.print(f"[red]SL HIT! PnL: ${pnl:.2f} | Balance: ${current_balance:.2f}[/red]")
                        logger.info(f"SL HIT @ {stop_loss:.2f} | PnL: ${pnl:.2f}")
                        position = None

                elif position == 'short':
                    if current_price <= take_profit:
                        pnl = (entry_price - take_profit) * position_size
                        current_balance += pnl
                        total_pnl += pnl
                        total_trades += 1
                        if pnl > 0:
                            winning_trades += 1
                        console.print(f"[green]TP HIT! PnL: ${pnl:.2f} | Balance: ${current_balance:.2f}[/green]")
                        logger.info(f"TP HIT @ {take_profit:.2f} | PnL: ${pnl:.2f}")
                        position = None
                    elif current_price >= stop_loss:
                        pnl = (entry_price - stop_loss) * position_size
                        current_balance += pnl
                        total_pnl += pnl
                        total_trades += 1
                        console.print(f"[red]SL HIT! PnL: ${pnl:.2f} | Balance: ${current_balance:.2f}[/red]")
                        logger.info(f"SL HIT @ {stop_loss:.2f} | PnL: ${pnl:.2f}")
                        position = None

            # Only take action on candle close
            if candle_just_closed and position is None:
                last_candle_time = current_time

                # Prepare data for agent (simplified - using main data only for live)
                # In production, you would prepare proper multi-timeframe data
                main_data = df_main[['open', 'high', 'low', 'close', 'volume']].tail(36).values

                # Calculate ATR for TP/SL
                highs = df_main['high'].tail(14).values
                lows = df_main['low'].tail(14).values
                atr = np.mean(highs - lows)

                # Simple decision based on recent trend (placeholder for actual agent inference)
                # In production, you would use: action = agent.select_action(state, training=False)

                # Trend-based simple strategy for testing
                sma_fast = df_main['close'].tail(7).mean()
                sma_slow = df_main['close'].tail(25).mean()
                rsi = df_main['rsi'].iloc[-1] if 'rsi' in df_main.columns else 50

                action = 0  # HOLD by default

                if sma_fast > sma_slow and rsi < 70:
                    action = 1  # LONG
                elif sma_fast < sma_slow and rsi > 30:
                    action = 2  # SHORT

                if action == 1:  # LONG
                    entry_price = current_price
                    position_size = current_balance / current_price
                    take_profit = entry_price + (atr * 3.0)
                    stop_loss = entry_price - (atr * 1.5)
                    position = 'long'
                    console.print(f"[cyan]OPEN LONG @ {entry_price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}[/cyan]")
                    logger.info(f"OPEN LONG @ {entry_price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")

                elif action == 2:  # SHORT
                    entry_price = current_price
                    position_size = current_balance / current_price
                    take_profit = entry_price - (atr * 3.0)
                    stop_loss = entry_price + (atr * 1.5)
                    position = 'short'
                    console.print(f"[magenta]OPEN SHORT @ {entry_price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}[/magenta]")
                    logger.info(f"OPEN SHORT @ {entry_price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")

            # Status display
            pos_status = f"{position.upper()} @ {entry_price:.2f}" if position else "NONE"
            win_rate = (winning_trades / max(1, total_trades)) * 100

            status_line = (
                f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                f"Price: ${current_price:.2f} | "
                f"Position: {pos_status} | "
                f"Balance: ${current_balance:.2f} | "
                f"Trades: {total_trades} | "
                f"WR: {win_rate:.1f}%"
            )
            print(status_line, end='', flush=True)

            time.sleep(fetch_interval)

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Live trading stopped by user.[/yellow]")

    # Final results
    win_rate = (winning_trades / max(1, total_trades)) * 100
    total_return = ((current_balance - balance) / balance) * 100

    console.print(f"\n[bold]Final Results:[/bold]")
    console.print(f"Final Balance: ${current_balance:.2f}")
    console.print(f"Total Return: {total_return:.2f}%")
    console.print(f"Total Trades: {total_trades}")
    console.print(f"Win Rate: {win_rate:.2f}%")
    console.print(f"Total PnL: ${total_pnl:.2f}")

    logger.info("=" * 60)
    logger.info("LIVE PAPER TRADE RESULTS")
    logger.info(f"Final Balance: ${current_balance:.2f}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Win Rate: {win_rate:.2f}%")


# ==================== MAIN MENU ====================
def main():
    """Main menu loop"""
    while True:
        console.clear()
        print_logo()
        console.print()

        choices = [
            "1. System Check",
            "2. Download Data",
            "3. List Data Files",
            "4. Train Model",
            "5. Backtest",
            "6. Paper Trade (Historical)",
            "7. Live Paper Trade (Real-Time)",
            "8. List Models",
            "9. Exit"
        ]

        choice = questionary.select(
            "Select an option:",
            choices=choices,
            style=custom_style,
            qmark="?",
            pointer=">"
        ).ask()

        if choice is None:
            console.print("\n[bold]Goodbye![/bold]")
            break

        console.clear()
        print_logo()

        try:
            if "1." in choice:
                system_check()
            elif "2." in choice:
                download_data()
            elif "3." in choice:
                list_data()
            elif "4." in choice:
                train_model()
            elif "5." in choice:
                backtest()
            elif "6." in choice:
                paper_trade_historical()
            elif "7." in choice:
                live_paper_trade()
            elif "8." in choice:
                list_models()
            elif "9." in choice:
                console.print("\n[bold cyan]Thanks for using CRYPPO![/bold cyan]")
                break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()

        Prompt.ask("\n[dim]Press Enter to continue...[/dim]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bold]Interrupted. Goodbye![/bold]")
        sys.exit(0)
