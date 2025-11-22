#!/usr/bin/env python3
"""
CRYPPO - Deep RL Crypto Trading System CLI (Multi-Timeframe Edition)
"""

import sys
import os
import argparse
import torch
import numpy as np
import time
import gc
import logging
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.layout import Layout

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

from src.data.loader import load_multi_timeframe_data, prepare_multi_timeframe_env_data, download_and_save_data
from src.data.realtime_fetcher import RealtimeFetcher
from src.policy.trading_env import TradingEnv
from src.agent.d3qn_agent import D3QNAgent

console = Console()

def setup_logger(mode: str, debug: bool = False) -> logging.Logger:
    """
    Setup logger for train/backtest/live modes
    Creates separate log files: train.log, backtest.log, live.log
    Overwrites previous log (keeps only latest run)

    Args:
        mode: 'train', 'backtest', or 'live'
        debug: If True, enable DEBUG level logging (detailed Q-values, confidence, etc.)
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"{mode}.log"

    # Create logger
    logger = logging.getLogger(f"cryppo_{mode}")
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler (mode='w' to overwrite)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # Log header
    logger.info("="*60)
    logger.info(f"CRYPPO - {mode.upper()} Mode")
    logger.info(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)

    return logger

def check_system():
    """Perform system checks"""
    console.print(Panel("System Check", style="bold cyan"))
    
    # Python
    console.print(f"Python: [green]{sys.version.split()[0]}[/green]")
    
    # PyTorch
    try:
        import torch
        console.print(f"PyTorch: [green]{torch.__version__}[/green]")
        if torch.cuda.is_available():
            console.print(f"Device: [green]CUDA ({torch.cuda.get_device_name(0)})[/green]")
        else:
            console.print("Device: [yellow]CPU[/yellow]")
    except ImportError:
        console.print("PyTorch: [red]Not Found[/red]")

    # Data Directory
    data_dir = Path("data")
    if data_dir.exists():
        files = list(data_dir.glob("*.parquet"))
        console.print(f"Data Directory: [green]Found ({len(files)} files)[/green]")
    else:
        console.print("Data Directory: [yellow]Not Found (will be created)[/yellow]")

def run_backtest(args):
    """Run backtest on test set (last 3 months)"""
    # Setup logger
    logger = setup_logger("backtest")

    console.print(f"[bold cyan]Running Backtest ({args.symbol})[/bold cyan]")
    console.print("[dim]Testing on last 3 months (out-of-sample)[/dim]\n")

    # Log configuration
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Timeframes: {args.main_tf} + {args.detail_tf}")
    logger.info(f"Initial Balance: ${args.balance:.2f}")
    logger.info(f"Testing on last 3 months (out-of-sample)")
    logger.info("")

    # 1. Load FULL Data
    try:
        df_main, df_detail = load_multi_timeframe_data(
            symbol=args.symbol,
            main_timeframe=args.main_tf,
            detail_timeframe=args.detail_tf,
            start_date=args.start if args.start else None,
            end_date=args.end if args.end else None
        )
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        logger.error(f"Error loading data: {e}")
        return

    # 2. Prepare Data
    full_data = prepare_multi_timeframe_env_data(
        df_main, df_detail,
        main_timeframe=args.main_tf,
        detail_timeframe=args.detail_tf
    )

    # 3. Use TEST split (last 3 months)
    total_steps = len(full_data['main'])
    test_size = min(2160, int(total_steps * 0.15))
    train_end = total_steps - test_size

    test_data = {
        'main': full_data['main'][train_end:],
        'detail': full_data['detail'][train_end:]
    }

    console.print(f"[cyan]Using test data:[/cyan]")
    console.print(f"  • Test set: {len(test_data['main']):,} steps (last 3 months)")
    console.print()

    logger.info(f"Test set: {len(test_data['main'])} steps")
    logger.info("")

    # 4. Init Env with TEST data
    env = TradingEnv(test_data, initial_balance=args.balance, logger=logger)
    logger.info(f"Environment initialized: {env.n_steps} steps")
    logger.info("")

    # 4. Load Agent
    obs_shape = env.observation_space['main'].shape
    detail_shape = env.observation_space['detail'].shape

    agent = D3QNAgent(
        window_size=obs_shape[0],
        n_main_features=obs_shape[1],
        n_detail_features=detail_shape[2],
        detail_len=detail_shape[1],
        action_dim=env.action_space.n,
        logger=logger  # Pass logger for detailed action selection logging
    )
    
    model_path = args.model_path
    if not model_path:
        model_path = f"checkpoints/d3qn_{args.symbol}_{args.main_tf}.pth"
        
    if os.path.exists(model_path):
        agent.load(model_path)
        console.print(f"[green]Loaded model from {model_path}[/green]")
    else:
        console.print(f"[red]Model not found at {model_path}. Running with random weights.[/red]")

    # 5. Run Loop
    state, _ = env.reset()
    done = False
    total_reward = 0
    agent.epsilon = 0.0 # No exploration

    # Calculate actual number of steps to execute
    total_steps = env.n_steps - env.current_step

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Backtesting...", total=total_steps)
        
        while not done:
            mask = env.get_action_mask()
            action = agent.select_action(state, training=False, action_mask=mask, step=env.current_step)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            progress.update(task, advance=1)

    # 6. Results
    win_rate = (env.winning_trades / max(1, env.total_trades)) * 100
    total_return = ((env.balance - env.initial_balance) / env.initial_balance) * 100

    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Final Balance", f"${env.balance:.2f}")
    table.add_row("Total Return", f"{total_return:.2f}%")
    table.add_row("Total Trades", str(env.total_trades))
    table.add_row("Win Rate", f"{win_rate:.2f}%")

    console.print(table)

    # Log results
    logger.info("="*60)
    logger.info("BACKTEST RESULTS")
    logger.info(f"Initial Balance: ${env.initial_balance:.2f}")
    logger.info(f"Final Balance: ${env.balance:.2f}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Total Trades: {env.total_trades}")
    logger.info(f"Winning Trades: {env.winning_trades}")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    logger.info(f"Total PnL: ${env.total_pnl:.2f}")
    logger.info("="*60)

    console.print(f"[dim]Log saved to: logs/backtest.log[/dim]")

def run_paper_trade(args):
    """Run paper trading simulation on test set"""
    # Setup logger
    logger = setup_logger("live")

    console.print(Panel(f"Starting Paper Trade Simulation ({args.symbol})", style="bold red"))
    console.print("[dim]Simulating live market execution on last 3 months...[/dim]\n")

    # Log configuration
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Timeframes: {args.main_tf} + {args.detail_tf}")
    logger.info(f"Initial Balance: ${args.balance:.2f}")
    logger.info(f"Using test data (last 3 months)")
    logger.info("")

    # 1. Load FULL Data
    try:
        df_main, df_detail = load_multi_timeframe_data(
            symbol=args.symbol,
            main_timeframe=args.main_tf,
            detail_timeframe=args.detail_tf,
            start_date=args.start if args.start else None,
            end_date=args.end if args.end else None
        )
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        logger.error(f"Error loading data: {e}")
        return

    # 2. Prepare Data
    full_data = prepare_multi_timeframe_env_data(
        df_main, df_detail,
        main_timeframe=args.main_tf,
        detail_timeframe=args.detail_tf
    )

    # 3. Use TEST split (last 3 months)
    total_steps = len(full_data['main'])
    test_size = min(2160, int(total_steps * 0.15))
    train_end = total_steps - test_size

    test_data = {
        'main': full_data['main'][train_end:],
        'detail': full_data['detail'][train_end:]
    }

    console.print(f"[cyan]Paper trading on test data:[/cyan]")
    console.print(f"  • Test set: {len(test_data['main']):,} steps (last 3 months)")
    console.print()

    logger.info(f"Test set: {len(test_data['main'])} steps")
    logger.info("")

    # 4. Init Env with streaming mode
    env = TradingEnv(test_data, initial_balance=args.balance, streaming_mode=True, logger=logger)
    logger.info(f"Environment initialized: {env.n_steps} steps")
    logger.info("")

    # 4. Load Agent
    obs_shape = env.observation_space['main'].shape
    detail_shape = env.observation_space['detail'].shape

    agent = D3QNAgent(
        window_size=obs_shape[0],
        n_main_features=obs_shape[1],
        n_detail_features=detail_shape[2],
        detail_len=detail_shape[1],
        action_dim=env.action_space.n,
        logger=logger  # Pass logger for detailed action selection logging
    )
    
    model_path = args.model_path
    if not model_path:
        model_path = f"checkpoints/d3qn_{args.symbol}_{args.main_tf}.pth"
        
    if os.path.exists(model_path):
        agent.load(model_path)
        console.print(f"[green]Loaded model from {model_path}[/green]")
    else:
        console.print(f"[red]Model not found at {model_path}. Using random weights.[/red]")

    # 5. Run Simulation Loop
    state, _ = env.reset()
    done = False
    total_reward = 0
    agent.epsilon = 0.0
    
    console.print("\n[yellow]Press Ctrl+C to stop simulation[/yellow]\n")
    console.print("[cyan]Note: Each step = 1 hour. Simulation runs at full speed (no delay).[/cyan]\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Running paper trade simulation", total=env.n_steps)

            while not done:
                mask = env.get_action_mask()
                action = agent.select_action(state, training=False, action_mask=mask, step=env.current_step)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward

                # Update progress (no delay - run at full speed!)
                progress.update(task, advance=1)

                # Print trade notifications
                if env.total_trades > 0 and env.current_step % 100 == 0:
                    win_rate = (env.winning_trades / env.total_trades) * 100
                    console.print(f"[dim]Step {env.current_step}/{env.n_steps} | Trades: {env.total_trades} | WR: {win_rate:.1f}%[/dim]") 
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Simulation stopped by user.[/yellow]")
        logger.info("Simulation stopped by user")

    # 6. Results
    win_rate = (env.winning_trades / max(1, env.total_trades)) * 100
    total_return = ((env.balance - env.initial_balance) / env.initial_balance) * 100

    console.print(Panel("Paper Trade Results", style="bold red"))
    console.print(f"Final Balance: ${env.balance:.2f}")
    console.print(f"Total Return: {total_return:.2f}%")

    # Log results
    logger.info("="*60)
    logger.info("PAPER TRADE RESULTS")
    logger.info(f"Initial Balance: ${env.initial_balance:.2f}")
    logger.info(f"Final Balance: ${env.balance:.2f}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Total Trades: {env.total_trades}")
    logger.info(f"Winning Trades: {env.winning_trades}")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    logger.info(f"Total PnL: ${env.total_pnl:.2f}")
    logger.info("="*60)

    console.print(f"[dim]Log saved to: logs/live.log[/dim]")

def run_train(args):
    """Run training"""
    # Setup logger
    logger = setup_logger("train")

    console.print(f"[bold cyan]Training D3QN ({args.main_tf} + {args.detail_tf})[/bold cyan]")
    console.print("\n[green]⚡ Performance Optimizations:[/green]")
    console.print("  • Window size: 36 (was 72) - 50% faster")
    console.print("  • Batch size: 32 (was 64) - 50% less memory")
    console.print("  • Replay buffer: 10k (was 50k) - 80% less memory")
    console.print("  • Network: Simplified (1 LSTM layer, 16 quantiles)")
    console.print("  • Update frequency: Every 4 steps (was every step)")
    console.print("  • Memory cleanup: Enabled after each episode")

    console.print("\n[magenta]💰 Profitability Improvements:[/magenta]")
    console.print("  • TP/SL Ratio: 2.5:1.5 - Balanced for 37.5% breakeven")
    console.print("  • Learning Rate: 0.0003 - Faster learning")
    console.print("  • Epsilon: 1.0 → 0.15 (decay 0.9998) - Balanced exploration")
    console.print("  • Anti-Overtrading: 3 bars minimum between trades")
    console.print("  • NO entry penalty - Learn through TP/SL rewards")
    console.print("  • Reward: TP +1.0, SL -0.5, Hold +0.0001")
    console.print("  • Win streak bonus: >40% +0.3, >50% +0.5")
    console.print("  • [yellow]Bust protection DISABLED[/yellow] - Train on 100% of data")

    console.print("\n[cyan]📊 NEW: Trend & Swing Features (Dow Theory):[/cyan]")
    console.print("  • Trend Direction: SMA 7 vs 25 crossover")
    console.print("  • Trend Strength: EMA distance indicator")
    console.print("  • Swing Points: Automatic high/low detection")
    console.print("  • Support/Resistance: Distance to swing points")
    console.print("  • Dow Theory: Higher Highs/Lows detection")
    console.print("  • Structure Score: Uptrend/Downtrend confirmation")
    console.print("  [yellow]→ Model can now learn quality entry points![/yellow]\n")

    # Load FULL dataset (not auto-limited)
    console.print("[cyan]📊 Data Strategy:[/cyan]")
    console.print("  • Loading FULL dataset (all available data)")
    console.print("  • Train: ALL data except last 3 months")
    console.print("  • Test: Last 3 months (for out-of-sample validation)")
    console.print()

    # Log training configuration
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Timeframes: {args.main_tf} + {args.detail_tf}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Initial Balance: ${args.balance:.2f}")
    logger.info(f"Using FULL dataset - no date limits")
    logger.info("")

    # 1. Load FULL Data (no date limits unless user specifies)
    try:
        df_main, df_detail = load_multi_timeframe_data(
            symbol=args.symbol,
            main_timeframe=args.main_tf,
            detail_timeframe=args.detail_tf,
            start_date=args.start if args.start else None,
            end_date=args.end if args.end else None
        )
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        logger.error(f"Error loading data: {e}")
        return

    console.print(f"[green]✓[/green] Loaded {len(df_main)} main candles ({len(df_detail)} detail candles)")

    # 2. Prepare and Split Data (Train on ALL except last 3 months)
    full_data = prepare_multi_timeframe_env_data(
        df_main, df_detail,
        main_timeframe=args.main_tf,
        detail_timeframe=args.detail_tf
    )

    # Calculate split point: reserve last 3 months (~2160 hours) for testing
    total_steps = len(full_data['main'])
    test_size = min(2160, int(total_steps * 0.15))  # 3 months or 15%, whichever is smaller
    train_end = total_steps - test_size

    # Split data
    train_data = {
        'main': full_data['main'][:train_end],
        'detail': full_data['detail'][:train_end]
    }

    console.print(f"[cyan]Data Split:[/cyan]")
    console.print(f"  • Training:   {len(train_data['main']):,} steps ({len(train_data['main'])/24:.0f} days)")
    console.print(f"  • Test:       {test_size:,} steps (~3 months)")
    console.print(f"  • Total:      {total_steps:,} steps")

    logger.info(f"Data split: {len(train_data['main'])} train / {test_size} test")
    logger.info("")

    # 3. Init Env with TRAINING data only (bust protection disabled for full episode completion)
    env = TradingEnv(train_data, initial_balance=args.balance, enable_bust_protection=False, logger=logger)
    logger.info(f"Environment initialized: {env.n_steps} steps")
    logger.info(f"Bust protection: DISABLED (training will complete all data even if losing)")
    logger.info("")

    # 4. Init Agent
    obs_shape = env.observation_space['main'].shape
    detail_shape = env.observation_space['detail'].shape

    agent = D3QNAgent(
        window_size=obs_shape[0],
        n_main_features=obs_shape[1],
        n_detail_features=detail_shape[2],
        detail_len=detail_shape[1],
        action_dim=env.action_space.n,
        logger=logger  # Pass logger for detailed action selection logging
    )
    
    # 5. Train Loop
    console.print("\n[green]Starting Training Loop...[/green]")

    # Training optimization: Update every N steps instead of every step
    update_frequency = 4  # Update every 4 steps for efficiency

    for episode in range(args.episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        action_counts = {0: 0, 1: 0, 2: 0}  # Track HOLD, LONG, SHORT

        logger.info(f"{'='*60}")
        logger.info(f"EPISODE {episode+1}/{args.episodes}")
        logger.info(f"{'='*60}")

        # Calculate actual number of steps to execute
        total_steps = env.n_steps - env.current_step

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Episode {episode+1}/{args.episodes}", total=total_steps)

            while not done:
                mask = env.get_action_mask()
                action = agent.select_action(state, training=True, action_mask=mask, step=env.current_step)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store_transition(state, action, reward, next_state, done)

                # Track actions
                action_counts[action] += 1

                # Update every N steps for efficiency
                step_count += 1
                loss = None
                if step_count % update_frequency == 0:
                    loss = agent.update()

                # Log progress every 100 steps
                if step_count % 100 == 0:
                    current_wr = (env.winning_trades / max(1, env.total_trades)) * 100
                    logger.info(f"Step {env.current_step}/{env.n_steps} | Balance: ${env.balance:.2f} | Trades: {env.total_trades} | WR: {current_wr:.1f}% | Epsilon: {agent.epsilon:.3f}")

                state = next_state
                total_reward += reward
                progress.update(task, advance=1)

        # Calculate metrics
        win_rate = (env.winning_trades / max(1, env.total_trades)) * 100
        total_return = ((env.balance - env.initial_balance) / env.initial_balance) * 100
        total_actions = sum(action_counts.values())

        console.print(f"Episode {episode+1}/{args.episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f} | Balance: ${env.balance:.2f}")

        # Log episode results
        logger.info(f"")
        logger.info(f"EPISODE {episode+1}/{args.episodes} SUMMARY")
        logger.info(f"-" * 60)
        logger.info(f"Performance:")
        logger.info(f"  Total Reward: {total_reward:.2f}")
        logger.info(f"  Final Balance: ${env.balance:.2f}")
        logger.info(f"  Total Return: {total_return:.2f}%")
        logger.info(f"  Total PnL: ${env.total_pnl:.2f}")
        logger.info(f"")
        logger.info(f"Trading Stats:")
        logger.info(f"  Total Trades: {env.total_trades}")
        logger.info(f"  Winning Trades: {env.winning_trades}")
        logger.info(f"  Losing Trades: {env.total_trades - env.winning_trades}")
        logger.info(f"  Win Rate: {win_rate:.2f}%")
        logger.info(f"")
        logger.info(f"Action Distribution:")
        logger.info(f"  HOLD: {action_counts[0]} ({action_counts[0]/total_actions*100:.1f}%)")
        logger.info(f"  OPEN LONG: {action_counts[1]} ({action_counts[1]/total_actions*100:.1f}%)")
        logger.info(f"  OPEN SHORT: {action_counts[2]} ({action_counts[2]/total_actions*100:.1f}%)")
        logger.info(f"")
        logger.info(f"Agent Stats:")
        logger.info(f"  Epsilon: {agent.epsilon:.4f}")
        logger.info(f"  Steps Completed: {env.current_step}/{env.n_steps}")
        logger.info(f"  Replay Buffer Size: {len(agent.memory)}")
        logger.info("")

        # Memory optimization: Clear cache after each episode
        if episode % 1 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final summary
    logger.info("="*60)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Total Episodes: {args.episodes}")
    logger.info(f"Final Balance: ${env.balance:.2f}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    logger.info("="*60)

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/d3qn_{args.symbol}_{args.main_tf}.pth"
    agent.save(save_path)
    logger.info(f"Model saved to: {save_path}")
    console.print(f"[bold green]Training Complete! Model saved to {save_path}[/bold green]")
    console.print(f"[dim]Log saved to: logs/train.log[/dim]")

def main():
    parser = argparse.ArgumentParser(description="CRYPPO Trading System")
    subparsers = parser.add_subparsers(dest="command")

    # 1. Download Data
    dl_parser = subparsers.add_parser("data", help="Data management")
    dl_sub = dl_parser.add_subparsers(dest="subcommand")
    
    dl_down = dl_sub.add_parser("download", help="Download market data")
    dl_down.add_argument("--symbol", default="BTCUSDT")
    dl_down.add_argument("--timeframe", default="1h")
    dl_down.add_argument("--start", default="2023-01-01")
    dl_down.add_argument("--end", default="2024-12-31")

    # 2. Train
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--symbol", default="BTCUSDT")
    train_parser.add_argument("--main-tf", default="1h")
    train_parser.add_argument("--detail-tf", default="1m")
    train_parser.add_argument("--episodes", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--balance", type=float, default=10000.0)
    train_parser.add_argument("--start", default=None)
    train_parser.add_argument("--end", default=None)

    # 3. Backtest
    bt_parser = subparsers.add_parser("backtest", help="Backtest model")
    bt_parser.add_argument("--symbol", default="BTCUSDT")
    bt_parser.add_argument("--main-tf", default="1h")
    bt_parser.add_argument("--detail-tf", default="1m")
    bt_parser.add_argument("--model-path", default="")
    bt_parser.add_argument("--balance", type=float, default=10000.0)
    bt_parser.add_argument("--start", default=None)
    bt_parser.add_argument("--end", default=None)

    # 4. Paper Trade
    pt_parser = subparsers.add_parser("paper", help="Paper trade simulation")
    pt_parser.add_argument("--symbol", default="BTCUSDT")
    pt_parser.add_argument("--main-tf", default="1h")
    pt_parser.add_argument("--detail-tf", default="1m")
    pt_parser.add_argument("--model-path", default="")
    pt_parser.add_argument("--balance", type=float, default=10000.0)
    pt_parser.add_argument("--start", default=None)
    pt_parser.add_argument("--end", default=None)

    # 5. Check
    subparsers.add_parser("check", help="System check")

    args = parser.parse_args()

    if args.command == "check":
        check_system()
        
    elif args.command == "data" and args.subcommand == "download":
        download_and_save_data(args.symbol, args.timeframe, args.start, args.end)
        
    elif args.command == "train":
        run_train(args)
        
    elif args.command == "backtest":
        run_backtest(args)
        
    elif args.command == "paper":
        run_paper_trade(args)

if __name__ == "__main__":
    main()
