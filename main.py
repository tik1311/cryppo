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
from src.agent.ppo_agent import PPOAgent

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
    """
    Train a new PPO model - FULLY REDESIGNED FOR PRODUCTION

    Key Improvements:
    1. Split data BEFORE normalization (no data leakage)
    2. Validation-based model checkpointing (no overfitting)
    3. Proper OnlineScaler handling
    4. Walk-Forward style validation
    """
    console.print(Panel("Train Model (PPO + PatchTST) - Production Ready", style="bold blue"))

    symbol = Prompt.ask("Symbol", default="BTCUSDT")

    # Minimum 100 episodes for PPO to converge properly
    episodes_input = Prompt.ask("Episodes (min 100 recommended)", default="200")
    episodes = max(100, int(episodes_input))
    if int(episodes_input) < 100:
        console.print(f"[yellow]Warning: {episodes_input} episodes is too few. Using minimum of 100.[/yellow]")

    balance = float(Prompt.ask("Initial Balance ($)", default="10000"))

    # Architecture selection
    arch_choice = questionary.select(
        "Model Architecture?",
        choices=["PatchTST (Recommended)", "LSTM+Attention (Legacy)"],
        style=custom_style
    ).ask()
    architecture = "patchtst" if "PatchTST" in arch_choice else "lstm"

    # Training parameters - OPTIMIZED FOR MAXIMUM PROFIT
    window_size = 36
    tp_mult = 5.0       # TP at 5.0x ATR - let winners run!
    sl_mult = 2.0       # SL at 2.0x ATR - R:R = 2.5:1
    min_bars = 20       # Wait longer between trades (was 10)
    lr = 3e-4           # Standard LR for faster learning
    gamma = 0.99        # Slightly lower for more immediate rewards
    rollout_length = 1024  # Longer rollouts for better learning
    batch_size = 128    # Smaller batches for more updates

    # Risk Management - CONSERVATIVE
    risk_per_trade = 0.01   # 1% risk per trade (was 2%)
    max_trades_day = 1      # Quality over quantity! (was 2)
    max_daily_loss = 0.06

    logger = setup_logger("train")

    console.print("\n[cyan]Loading data...[/cyan]")

    try:
        df_main, df_detail = load_multi_timeframe_data(symbol=symbol, main_timeframe="1h", detail_timeframe="1m")
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        console.print("[yellow]Please download data first![/yellow]")
        return

    console.print(f"[green]Loaded {len(df_main)} main candles, {len(df_detail)} detail candles[/green]")

    # ===== CRITICAL FIX: Split data BEFORE normalization =====
    from src.data.features import add_technical_indicators, add_minimal_indicators, OnlineScaler

    console.print("\n[cyan]Step 1: Adding technical indicators...[/cyan]")
    df_main = add_technical_indicators(df_main, verbose=True)
    df_detail = add_minimal_indicators(df_detail, verbose=True)
    df_main = df_main.dropna()
    df_detail = df_detail.dropna()

    # Calculate split indices (BEFORE normalization)
    total_steps = len(df_main)
    val_size = min(720, int(total_steps * 0.10))   # ~30 days for validation
    test_size = min(2160, int(total_steps * 0.15))  # ~90 days for test
    train_end = total_steps - val_size - test_size
    val_end = total_steps - test_size

    console.print(f"\n[cyan]Step 2: Splitting data (BEFORE normalization)...[/cyan]")
    console.print(f"  - Train: 0 to {train_end} ({train_end} steps, {train_end/24:.0f} days)")
    console.print(f"  - Val:   {train_end} to {val_end} ({val_size} steps, {val_size/24:.0f} days)")
    console.print(f"  - Test:  {val_end} to {total_steps} ({test_size} steps, {test_size/24:.0f} days)")

    # Split DataFrames
    df_main_train = df_main.iloc[:train_end].copy()
    df_main_val = df_main.iloc[train_end:val_end].copy()
    df_detail_train = df_detail[df_detail['timestamp'] <= df_main_train['timestamp'].max()].copy()
    df_detail_val = df_detail[
        (df_detail['timestamp'] > df_main_train['timestamp'].max()) &
        (df_detail['timestamp'] <= df_main_val['timestamp'].max())
    ].copy()

    # ===== CRITICAL FIX: Fit scaler ONLY on training data =====
    console.print("\n[cyan]Step 3: Fitting scaler on TRAINING data only...[/cyan]")

    price_cols = ['open', 'high', 'low', 'close', 'volume']
    norm_cols = ['returns', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
                 'atr', 'stoch_k', 'stoch_d', 'volume_ratio', 'volatility']

    scaler = OnlineScaler(method='minmax')
    scaler.fit(df_main_train, norm_cols)

    # Save scaler for live trading
    os.makedirs("checkpoints", exist_ok=True)
    scaler.save(f"checkpoints/scaler_{symbol}_1h.pkl")
    console.print(f"  - Scaler saved to checkpoints/scaler_{symbol}_1h.pkl")

    # Transform train and val with the SAME scaler
    df_main_train = scaler.transform(df_main_train)
    df_main_val = scaler.transform(df_main_val)
    df_detail_train = scaler.transform(df_detail_train)
    df_detail_val = scaler.transform(df_detail_val)

    # Prepare aligned data
    console.print("\n[cyan]Step 4: Preparing aligned data...[/cyan]")
    train_data = prepare_multi_timeframe_env_data(
        df_main_train, df_detail_train,
        main_timeframe="1h", detail_timeframe="1m",
        add_indicators=False, normalize=False, verbose=True
    )
    val_data = prepare_multi_timeframe_env_data(
        df_main_val, df_detail_val,
        main_timeframe="1h", detail_timeframe="1m",
        add_indicators=False, normalize=False, verbose=True
    )

    console.print(f"  - Train data shape: main={train_data['main'].shape}, detail={train_data['detail'].shape}")
    console.print(f"  - Val data shape: main={val_data['main'].shape}, detail={val_data['detail'].shape}")

    # Environment settings - OPTIMIZED FOR MAXIMUM PROFIT
    env_kwargs = dict(
        initial_balance=balance,
        enable_bust_protection=True,
        window_size=window_size,
        atr_tp_multiplier=tp_mult,
        atr_sl_multiplier=sl_mult,
        min_bars_between_trades=min_bars,
        max_trades_per_day=max_trades_day,
        risk_per_trade=risk_per_trade,
        reward_type="profit_focused",  # Back to profit-focused (DSR was cutting profits)
        enable_trailing_stop=False,    # DISABLED! Let trades hit TP
        trailing_stop_activation=2.0,  # Only trail after 2x ATR profit (if enabled)
        trailing_stop_distance=1.5,    # Wide trailing distance (if enabled)
        enable_breakeven_stop=False,   # DISABLED! Let trades breathe
        breakeven_activation=2.0,      # Higher activation (if enabled)
        max_daily_loss_pct=max_daily_loss,
        enable_position_scaling=True,
        scale_down_after_losses=2,
        position_scale_factor=0.5,
        logger=logger
    )

    # Training Environment
    env = TradingEnv(train_data, **env_kwargs)

    # Validation Environment (for model selection)
    val_env = TradingEnv(val_data, streaming_mode=True, **{k: v for k, v in env_kwargs.items() if k != 'logger'})

    # Print shapes for verification
    obs_shape = env.observation_space['main'].shape
    detail_shape = env.observation_space['detail'].shape
    console.print(f"\n[cyan]Step 5: Initializing PPO Agent...[/cyan]")
    console.print(f"  - Observation shape: main={obs_shape}, detail={detail_shape}")
    console.print(f"  - Action space: {env.action_space.n} actions")

    # Init PPO Agent with balanced hyperparameters
    agent = PPOAgent(
        window_size=obs_shape[0],
        n_main_features=obs_shape[1],
        n_detail_features=detail_shape[2],
        detail_len=detail_shape[1],
        action_dim=env.action_space.n,
        learning_rate=lr,
        gamma=gamma,
        batch_size=batch_size,
        n_epochs=3,
        entropy_coef=0.05,  # Moderate entropy for balanced exploration
        target_kl=0.02,     # Standard KL for stable learning
        architecture=architecture,
        logger=logger
    )

    console.print(f"[green]Starting PPO Training ({architecture.upper()})...[/green]\n")

    best_val_pnl = -float('inf')
    best_train_balance = balance
    no_improvement_count = 0
    episode_returns = []
    val_interval = 5  # Validate every 5 episodes

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
                action, log_prob, value = agent.select_action(state, training=True, action_mask=mask, step=env.current_step)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.store_transition(state, action, reward, done, log_prob, value, mask)
                step_count += 1

                if step_count % rollout_length == 0:
                    with torch.no_grad():
                        _, _, next_value = agent.select_action(next_state, training=False, action_mask=env.get_action_mask(), step=env.current_step)
                    agent.update(next_value)

                state = next_state
                total_reward += reward
                progress.update(task, advance=1)

        if len(agent.buffer) > 0:
            agent.update(0.0)

        train_win_rate = (env.winning_trades / max(1, env.total_trades)) * 100
        train_return = ((env.balance - env.initial_balance) / env.initial_balance) * 100
        episode_returns.append(train_return)

        agent.update_scheduler(train_return)

        # ===== CRITICAL FIX: Validation-based checkpointing =====
        if (episode + 1) % val_interval == 0:
            # Run validation episode
            val_state, _ = val_env.reset()
            val_done = False
            while not val_done:
                val_mask = val_env.get_action_mask()
                val_action, _, _ = agent.select_action(val_state, training=False, action_mask=val_mask, step=val_env.current_step)
                val_state, _, val_terminated, val_truncated, _ = val_env.step(val_action)
                val_done = val_terminated or val_truncated

            val_pnl = val_env.total_pnl
            val_return = ((val_env.balance - val_env.initial_balance) / val_env.initial_balance) * 100
            val_wr = (val_env.winning_trades / max(1, val_env.total_trades)) * 100

            console.print(f"  [VAL] PnL: ${val_pnl:.2f} | Return: {val_return:+.1f}% | WR: {val_wr:.1f}%")

            # Save ONLY if validation improves
            if val_pnl > best_val_pnl:
                best_val_pnl = val_pnl
                agent.save(f"checkpoints/ppo_{symbol}_1h_best.pth")
                console.print(f"  [green]New best validation model saved![/green]")
                no_improvement_count = 0
            else:
                no_improvement_count += 1

        # Track best training balance (for logging only, NOT for model selection)
        if env.balance > best_train_balance:
            best_train_balance = env.balance

        # Color output
        if train_return > 0:
            return_color = "green"
        elif train_return > -5:
            return_color = "yellow"
        else:
            return_color = "red"

        console.print(
            f"Episode {episode+1}/{episodes} | "
            f"Reward: {total_reward:.2f} | "
            f"Balance: ${env.balance:.2f} | "
            f"Return: [{return_color}]{train_return:+.1f}%[/{return_color}] | "
            f"Trades: {env.total_trades} | "
            f"WR: {train_win_rate:.1f}%"
        )

        # Early stopping based on VALIDATION (not training)
        if no_improvement_count >= 8 and episode >= 50:  # 8 * val_interval = 40 episodes with no improvement
            console.print(f"\n[yellow]Early stopping: No validation improvement for {no_improvement_count * val_interval} episodes[/yellow]")
            break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model
    save_path = f"checkpoints/ppo_{symbol}_1h.pth"
    agent.save(save_path)

    # Summary
    avg_return = np.mean(episode_returns[-10:]) if len(episode_returns) >= 10 else np.mean(episode_returns)

    console.print(f"\n[bold green]Training Complete![/bold green]")
    console.print(f"Best Train Balance: ${best_train_balance:.2f}")
    console.print(f"Best Validation PnL: ${best_val_pnl:.2f}")
    console.print(f"Avg Return (last 10 episodes): {avg_return:.1f}%")
    console.print(f"Final model: {save_path}")
    console.print(f"Best validation model: checkpoints/ppo_{symbol}_1h_best.pth")
    console.print(f"Scaler: checkpoints/scaler_{symbol}_1h.pkl")
    cleanup_resources()


# ==================== BACKTEST ====================
def backtest():
    """
    Run backtest on test set - PRODUCTION VERSION

    Uses saved scaler from training to ensure no data leakage.
    """
    console.print(Panel("Backtest Model (PPO) - Production Ready", style="bold magenta"))

    symbol = Prompt.ask("Symbol", default="BTCUSDT")
    balance = float(Prompt.ask("Initial Balance ($)", default="10000"))

    # Ask which model to use
    model_choice = questionary.select(
        "Which model to use?",
        choices=["Best (ppo_*_best.pth) - Recommended", "Latest (ppo_*.pth)"],
        style=custom_style
    ).ask()

    use_best = "best" in model_choice.lower()

    logger = setup_logger("backtest")

    # Load saved scaler (if available)
    scaler_path = f"checkpoints/scaler_{symbol}_1h.pkl"
    scaler = None
    if os.path.exists(scaler_path):
        from src.data.features import OnlineScaler
        scaler = OnlineScaler.load(scaler_path)
        console.print(f"[green]Loaded scaler from {scaler_path}[/green]")
    else:
        console.print(f"[yellow]Warning: No scaler found at {scaler_path}. Using default normalization.[/yellow]")

    console.print("\n[cyan]Loading data...[/cyan]")

    try:
        df_main, df_detail = load_multi_timeframe_data(symbol=symbol, main_timeframe="1h", detail_timeframe="1m")
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        return

    # Add indicators first
    from src.data.features import add_technical_indicators, add_minimal_indicators

    df_main = add_technical_indicators(df_main, verbose=True)
    df_detail = add_minimal_indicators(df_detail, verbose=True)
    df_main = df_main.dropna()
    df_detail = df_detail.dropna()

    # Calculate test split
    total_steps = len(df_main)
    test_size = min(2160, int(total_steps * 0.15))
    train_end = total_steps - test_size

    # Get test data
    df_main_test = df_main.iloc[train_end:].copy()
    df_detail_test = df_detail[df_detail['timestamp'] > df_main.iloc[train_end-1]['timestamp']].copy()

    # Apply scaler if available
    if scaler is not None:
        df_main_test = scaler.transform(df_main_test)
        df_detail_test = scaler.transform(df_detail_test)
        console.print("[green]Applied saved scaler to test data (no leakage)[/green]")

    # Prepare test data
    test_data = prepare_multi_timeframe_env_data(
        df_main_test, df_detail_test,
        main_timeframe="1h", detail_timeframe="1m",
        add_indicators=False, normalize=(scaler is None), verbose=True
    )

    console.print(f"[cyan]Testing on {len(test_data['main']):,} steps ({len(test_data['main'])/24:.0f} days)[/cyan]")

    # Init Environment with SAME settings as training
    env = TradingEnv(
        test_data,
        initial_balance=balance,
        window_size=36,
        atr_tp_multiplier=5.0,       # R:R = 2.5:1
        atr_sl_multiplier=2.0,       # Tight SL
        min_bars_between_trades=20,  # Less trading
        max_trades_per_day=1,        # Quality trades only
        risk_per_trade=0.01,         # 1% risk
        reward_type="profit_focused",
        streaming_mode=True,
        # DISABLED - Let trades hit TP!
        enable_trailing_stop=False,
        trailing_stop_activation=2.0,
        trailing_stop_distance=1.5,
        enable_breakeven_stop=False,
        breakeven_activation=2.0,
        max_daily_loss_pct=0.06,
        enable_position_scaling=True,
        scale_down_after_losses=2,
        position_scale_factor=0.5,
        logger=logger
    )

    # Detect architecture from saved model
    if use_best:
        model_path = f"checkpoints/ppo_{symbol}_1h_best.pth"
    else:
        model_path = f"checkpoints/ppo_{symbol}_1h.pth"

    # Try to detect architecture from checkpoint
    architecture = "lstm"  # Default to LSTM for backward compatibility
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            # First check config
            if 'config' in checkpoint and 'architecture' in checkpoint['config']:
                architecture = checkpoint['config']['architecture']
            else:
                # Detect from state_dict keys
                state_keys = list(checkpoint['policy_state_dict'].keys())
                has_patch = any('patch' in k.lower() for k in state_keys)
                has_lstm = any('lstm' in k for k in state_keys)
                if has_patch and not has_lstm:
                    architecture = "patchtst"
                else:
                    architecture = "lstm"
            console.print(f"[dim]Detected architecture: {architecture.upper()}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not detect architecture: {e}[/yellow]")

    # Load PPO Agent with detected architecture
    obs_shape = env.observation_space['main'].shape
    detail_shape = env.observation_space['detail'].shape

    agent = PPOAgent(
        window_size=obs_shape[0],
        n_main_features=obs_shape[1],
        n_detail_features=detail_shape[2],
        detail_len=detail_shape[1],
        action_dim=env.action_space.n,
        architecture=architecture,  # Use detected architecture
        logger=logger
    )

    if os.path.exists(model_path):
        agent.load(model_path)
        console.print(f"[green]Loaded model from {model_path}[/green]")
    else:
        console.print(f"[red]Model not found at {model_path}![/red]")
        # Try the other model
        alt_path = f"checkpoints/ppo_{symbol}_1h.pth" if use_best else f"checkpoints/ppo_{symbol}_1h_best.pth"
        if os.path.exists(alt_path):
            agent.load(alt_path)
            console.print(f"[yellow]Loaded alternative model from {alt_path}[/yellow]")
        else:
            console.print(f"[red]No model found! Train first.[/red]")
            return

    # Run backtest
    state, _ = env.reset()
    done = False

    total_episode_steps = env.n_steps - env.current_step

    # Track equity curve for analysis
    equity_curve = [balance]
    trade_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Backtesting...", total=total_episode_steps)
        prev_trades = 0

        while not done:
            mask = env.get_action_mask()
            action, _, _ = agent.select_action(state, training=False, action_mask=mask, step=env.current_step)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state

            equity_curve.append(info['equity'])

            # Track individual trades
            if info['total_trades'] > prev_trades:
                if len(env.recent_trades) > 0:
                    trade_results.append(env.recent_trades[-1])
                prev_trades = info['total_trades']

            progress.update(task, advance=1)

    # Calculate detailed metrics
    win_rate = (env.winning_trades / max(1, env.total_trades)) * 100
    total_return = ((env.balance - env.initial_balance) / env.initial_balance) * 100

    # Max Drawdown
    equity_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / peak * 100
    max_drawdown = np.max(drawdown)

    # Sharpe Ratio (simplified)
    returns = np.diff(equity_arr) / equity_arr[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)  # Annualized

    # Profit Factor
    winning_pnl = sum([t for t in trade_results if t > 0])
    losing_pnl = abs(sum([t for t in trade_results if t < 0]))
    profit_factor = winning_pnl / max(losing_pnl, 1)

    # Average trade
    avg_win = np.mean([t for t in trade_results if t > 0]) if winning_pnl > 0 else 0
    avg_loss = np.mean([t for t in trade_results if t < 0]) if losing_pnl > 0 else 0

    # Results table
    table = Table(title="Backtest Results - Profit Analysis")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="magenta", width=20)

    # Return coloring
    return_style = "green" if total_return > 0 else "red"
    pnl_style = "green" if env.total_pnl > 0 else "red"

    table.add_row("Final Balance", f"${env.balance:.2f}")
    table.add_row("Total Return", f"[{return_style}]{total_return:+.2f}%[/{return_style}]")
    table.add_row("Total PnL", f"[{pnl_style}]${env.total_pnl:.2f}[/{pnl_style}]")
    table.add_row("───────────────────────", "─────────────────")
    table.add_row("Total Trades", str(env.total_trades))
    table.add_row("Winning Trades", str(env.winning_trades))
    table.add_row("Win Rate", f"{win_rate:.1f}%")
    table.add_row("───────────────────────", "─────────────────")
    table.add_row("Max Drawdown", f"{max_drawdown:.1f}%")
    table.add_row("Sharpe Ratio", f"{sharpe:.2f}")
    table.add_row("Profit Factor", f"{profit_factor:.2f}")
    table.add_row("───────────────────────", "─────────────────")
    table.add_row("Avg Win", f"${avg_win:.2f}")
    table.add_row("Avg Loss", f"${avg_loss:.2f}")

    console.print(table)

    # Verdict
    if total_return > 5 and win_rate > 35 and profit_factor > 1.2:
        console.print("\n[bold green]VERDICT: PROFITABLE MODEL[/bold green]")
    elif total_return > 0 and profit_factor > 1.0:
        console.print("\n[bold yellow]VERDICT: MARGINALLY PROFITABLE - Needs improvement[/bold yellow]")
    else:
        console.print("\n[bold red]VERDICT: NOT PROFITABLE - Requires retraining[/bold red]")

    console.print(f"\n[dim]Log saved to: logs/backtest.log[/dim]")

    cleanup_resources()


# ==================== LIVE PAPER TRADE ====================
def live_paper_trade():
    """
    Real-time paper trading with live data from Binance using PPO model.

    PRODUCTION VERSION: Uses saved scaler for consistent normalization.
    """
    console.print(Panel("Live Paper Trade (Real-Time) - Production Ready", style="bold green"))

    symbol = Prompt.ask("Symbol", default="BTCUSDT")
    balance = float(Prompt.ask("Initial Balance ($)", default="10000"))
    timeframe = Prompt.ask("Timeframe", default="1h")

    # Ask which model to use
    model_choice = questionary.select(
        "Which model to use?",
        choices=["Best (ppo_*_best.pth) - Recommended", "Latest (ppo_*.pth)"],
        style=custom_style
    ).ask()

    use_best = "best" in model_choice.lower()

    logger = setup_logger("live")

    # Load saved scaler for consistent normalization
    scaler_path = f"checkpoints/scaler_{symbol}_1h.pkl"
    scaler = None
    if os.path.exists(scaler_path):
        from src.data.features import OnlineScaler
        scaler = OnlineScaler.load(scaler_path)
        console.print(f"[green]Loaded scaler from {scaler_path}[/green]")
    else:
        console.print(f"[yellow]Warning: No scaler found. Will use default normalization.[/yellow]")
        console.print(f"[yellow]For best results, train a model first to create a scaler.[/yellow]")

    # Timeframe settings
    tf_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}
    tf_seconds = tf_map.get(timeframe, 3600)
    fetch_interval = 10  # Fetch every 10 seconds

    console.print(f"\n[cyan]Starting Live Paper Trade with PPO Model...[/cyan]")
    console.print(f"[dim]Timeframe: {timeframe} | Fetch interval: {fetch_interval}s | Action on candle close[/dim]")

    # Determine model path based on choice
    if use_best:
        model_path = f"checkpoints/ppo_{symbol}_1h_best.pth"
        alt_path = f"checkpoints/ppo_{symbol}_1h.pth"
    else:
        model_path = f"checkpoints/ppo_{symbol}_1h.pth"
        alt_path = f"checkpoints/ppo_{symbol}_1h_best.pth"

    if not os.path.exists(model_path):
        if os.path.exists(alt_path):
            model_path = alt_path
            console.print(f"[yellow]Requested model not found, using {alt_path}[/yellow]")
        else:
            console.print(f"[red]Model not found! Train first.[/red]")
            return

    # Use 1m for accurate TP/SL checking
    fetcher = RealtimeFetcher(symbol=symbol, main_timeframe=timeframe, detail_timeframe="1m", window_size=36)

    # Initialize with first data fetch - retry if needed
    console.print("[cyan]Fetching initial data from Binance...[/cyan]")
    df_main, df_detail = None, None
    for attempt in range(3):
        df_main, df_detail = fetcher.get_latest_data(raw=True)
        if df_main is not None and len(df_main) >= 100:
            break
        console.print(f"[yellow]Attempt {attempt+1}/3: Waiting for more data... (got {len(df_main) if df_main is not None else 0})[/yellow]")
        time.sleep(5)

    if df_main is None or len(df_main) < 100:
        console.print(f"[red]Failed to fetch enough initial data! Got {len(df_main) if df_main is not None else 0} candles, need 100+.[/red]")
        return

    console.print(f"[green]Fetched {len(df_main)} main candles, {len(df_detail)} detail candles[/green]")

    # Add indicators first
    from src.data.features import add_technical_indicators, add_minimal_indicators

    df_main = add_technical_indicators(df_main, verbose=False)
    df_detail = add_minimal_indicators(df_detail, verbose=False)
    df_main = df_main.dropna()
    df_detail = df_detail.dropna()

    # Apply saved scaler if available
    if scaler is not None:
        df_main = scaler.transform(df_main)
        df_detail = scaler.transform(df_detail)
        console.print("[green]Applied saved scaler to live data[/green]")

    # Prepare initial env data
    full_data = prepare_multi_timeframe_env_data(
        df_main, df_detail,
        main_timeframe=timeframe, detail_timeframe="1m",
        add_indicators=False, normalize=(scaler is None)
    )

    # Verify data is valid
    if full_data['main'].shape[0] < 36:
        console.print(f"[red]Not enough data after processing! Got {full_data['main'].shape[0]} steps.[/red]")
        console.print("[yellow]Tip: This can happen if main and detail timeframes don't align. Try again in a few minutes.[/yellow]")
        return

    # Create a dummy env to get observation shape - MATCH TRAINING SETTINGS
    dummy_env = TradingEnv(
        full_data,
        initial_balance=balance,
        window_size=36,
        streaming_mode=True,
        atr_tp_multiplier=5.0,       # R:R = 2.5:1
        atr_sl_multiplier=2.0,
        min_bars_between_trades=20,
        max_trades_per_day=1,
        risk_per_trade=0.01,
    )

    obs_shape = dummy_env.observation_space['main'].shape
    detail_shape = dummy_env.observation_space['detail'].shape

    # Detect architecture from checkpoint
    architecture = "lstm"  # Default to LSTM for backward compatibility
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        # First check config
        if 'config' in checkpoint and 'architecture' in checkpoint['config']:
            architecture = checkpoint['config']['architecture']
        else:
            # Detect from state_dict keys
            state_keys = list(checkpoint['policy_state_dict'].keys())
            has_patch = any('patch' in k.lower() for k in state_keys)
            has_lstm = any('lstm' in k for k in state_keys)
            if has_patch and not has_lstm:
                architecture = "patchtst"
            else:
                architecture = "lstm"
        console.print(f"[dim]Detected architecture: {architecture.upper()}[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not detect architecture: {e}[/yellow]")

    # Build PPO agent with detected architecture
    agent = PPOAgent(
        window_size=obs_shape[0],
        n_main_features=obs_shape[1],
        n_detail_features=detail_shape[2],
        detail_len=detail_shape[1],
        action_dim=3,
        architecture=architecture,
        logger=logger
    )
    agent.load(model_path)
    console.print(f"[green]Loaded model from {model_path}[/green]")

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
    trades_today = 0
    last_trade_step = -100
    step_count = 0

    console.print("\n[yellow]Press Ctrl+C to stop[/yellow]\n")

    try:
        while True:
            # Fetch latest data (raw=True for proper indicator calculation)
            df_main, df_detail = fetcher.get_latest_data(raw=True)

            if df_main is None or len(df_main) < 100:
                console.print("[yellow]Waiting for data...[/yellow]")
                time.sleep(fetch_interval)
                continue

            current_price = float(df_main['close'].iloc[-1])
            current_time = df_main['timestamp'].iloc[-1]

            # Check if new candle closed
            candle_just_closed = (last_candle_time is None or current_time != last_candle_time)

            if candle_just_closed:
                step_count += 1
                last_candle_time = current_time

                # Reset daily trades counter (every 24 steps for 1h)
                if step_count % 24 == 0:
                    trades_today = 0

            # Check position status (TP/SL)
            if position:
                if position == 'long':
                    if current_price >= take_profit:
                        pnl = (take_profit - entry_price) * position_size
                        current_balance += pnl
                        total_pnl += pnl
                        total_trades += 1
                        if pnl > 0:
                            winning_trades += 1
                        console.print(f"\n[green]TP HIT! PnL: ${pnl:.2f} | Balance: ${current_balance:.2f}[/green]")
                        logger.info(f"TP HIT @ {take_profit:.2f} | PnL: ${pnl:.2f}")
                        position = None
                    elif current_price <= stop_loss:
                        pnl = (stop_loss - entry_price) * position_size
                        current_balance += pnl
                        total_pnl += pnl
                        total_trades += 1
                        console.print(f"\n[red]SL HIT! PnL: ${pnl:.2f} | Balance: ${current_balance:.2f}[/red]")
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
                        console.print(f"\n[green]TP HIT! PnL: ${pnl:.2f} | Balance: ${current_balance:.2f}[/green]")
                        logger.info(f"TP HIT @ {take_profit:.2f} | PnL: ${pnl:.2f}")
                        position = None
                    elif current_price >= stop_loss:
                        pnl = (entry_price - stop_loss) * position_size
                        current_balance += pnl
                        total_pnl += pnl
                        total_trades += 1
                        console.print(f"\n[red]SL HIT! PnL: ${pnl:.2f} | Balance: ${current_balance:.2f}[/red]")
                        logger.info(f"SL HIT @ {stop_loss:.2f} | PnL: ${pnl:.2f}")
                        position = None

            # Only take action on candle close
            if candle_just_closed and position is None:
                # Check trading conditions
                can_trade = (
                    trades_today < 3 and
                    step_count - last_trade_step >= 5
                )

                if can_trade:
                    # Prepare state for PPO agent
                    # Add indicators first
                    df_main_proc = add_technical_indicators(df_main.copy(), verbose=False)
                    df_detail_proc = add_minimal_indicators(df_detail.copy(), verbose=False)
                    df_main_proc = df_main_proc.dropna()
                    df_detail_proc = df_detail_proc.dropna()

                    # Apply saved scaler if available
                    if scaler is not None:
                        df_main_proc = scaler.transform(df_main_proc)
                        df_detail_proc = scaler.transform(df_detail_proc)

                    full_data = prepare_multi_timeframe_env_data(
                        df_main_proc, df_detail_proc,
                        main_timeframe=timeframe, detail_timeframe="1m",
                        add_indicators=False, normalize=(scaler is None), verbose=False
                    )

                    # Skip if data alignment failed
                    if full_data['main'].shape[0] == 0:
                        continue

                    # Update dummy env with latest data
                    dummy_env.data = full_data
                    dummy_env.main_data = full_data['main']
                    dummy_env.detail_data = full_data['detail']
                    dummy_env.n_steps = len(full_data['main'])
                    dummy_env.current_step = len(full_data['main']) - 1
                    dummy_env.balance = current_balance
                    dummy_env.position = None

                    # Get observation
                    state = dummy_env._get_observation()
                    mask = dummy_env.get_action_mask()

                    # Get action from PPO agent
                    action, _, _ = agent.select_action(state, training=False, action_mask=mask, step=step_count)

                    # Calculate ATR for TP/SL
                    highs = df_main['high'].tail(14).values
                    lows = df_main['low'].tail(14).values
                    closes = df_main['close'].tail(14).values
                    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
                    atr = np.mean(tr)

                    # Risk per trade: 1% - CONSERVATIVE
                    risk_amount = current_balance * 0.01
                    sl_distance = atr * 2.0  # 2x ATR SL

                    if action == 1:  # LONG
                        entry_price = current_price
                        position_size = risk_amount / sl_distance
                        take_profit = entry_price + (atr * 5.0)  # 5x ATR TP - R:R = 2.5:1
                        stop_loss = entry_price - (atr * 2.0)    # 2x ATR SL
                        position = 'long'
                        trades_today += 1
                        last_trade_step = step_count
                        console.print(f"\n[cyan]OPEN LONG @ {entry_price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f} | Size: {position_size:.6f}[/cyan]")
                        logger.info(f"OPEN LONG @ {entry_price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")

                    elif action == 2:  # SHORT
                        entry_price = current_price
                        position_size = risk_amount / sl_distance
                        take_profit = entry_price - (atr * 5.0)  # 5x ATR TP - R:R = 2.5:1
                        stop_loss = entry_price + (atr * 2.0)    # 2x ATR SL
                        position = 'short'
                        trades_today += 1
                        last_trade_step = step_count
                        console.print(f"\n[magenta]OPEN SHORT @ {entry_price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f} | Size: {position_size:.6f}[/magenta]")
                        logger.info(f"OPEN SHORT @ {entry_price:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")

            # Status display
            pos_status = f"{position.upper()} @ {entry_price:.2f}" if position else "NONE"
            win_rate = (winning_trades / max(1, total_trades)) * 100
            total_return = ((current_balance - balance) / balance) * 100

            status_line = (
                f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                f"Price: ${current_price:,.2f} | "
                f"Pos: {pos_status} | "
                f"Bal: ${current_balance:,.2f} ({total_return:+.1f}%) | "
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
    console.print(f"Total Return: {total_return:+.2f}%")
    console.print(f"Total Trades: {total_trades}")
    console.print(f"Win Rate: {win_rate:.1f}%")
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
            "5. Backtest (Historical Test)",
            "6. Live Paper Trade (Real-Time)",
            "7. List Models",
            "8. Exit"
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
                live_paper_trade()
            elif "7." in choice:
                list_models()
            elif "8." in choice:
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
