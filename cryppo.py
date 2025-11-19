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
from src.policy.trading_env import TradingEnv
from src.agent.d3qn_agent import D3QNAgent

console = Console()

def setup_logger(mode: str) -> logging.Logger:
    """
    Setup logger for train/backtest/live modes
    Creates separate log files: train.log, backtest.log, live.log
    Overwrites previous log (keeps only latest run)
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"{mode}.log"

    # Create logger
    logger = logging.getLogger(f"cryppo_{mode}")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler (mode='w' to overwrite)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

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
    """Run backtest"""
    # Setup logger
    logger = setup_logger("backtest")

    console.print(f"[bold cyan]Running Backtest ({args.symbol})[/bold cyan]")

    # Log configuration
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Timeframes: {args.main_tf} + {args.detail_tf}")
    logger.info(f"Initial Balance: ${args.balance:.2f}")
    logger.info(f"Date Range: {args.start} to {args.end}")
    logger.info("")

    # 1. Load Data
    try:
        df_main, df_detail = load_multi_timeframe_data(
            symbol=args.symbol,
            main_timeframe=args.main_tf,
            detail_timeframe=args.detail_tf,
            start_date=args.start,
            end_date=args.end
        )
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        logger.error(f"Error loading data: {e}")
        return

    # 2. Prepare Data
    data_dict = prepare_multi_timeframe_env_data(
        df_main, df_detail, 
        main_timeframe=args.main_tf, 
        detail_timeframe=args.detail_tf
    )
    
    # 3. Init Env
    env = TradingEnv(data_dict, initial_balance=args.balance)
    
    # 4. Load Agent
    obs_shape = env.observation_space['main'].shape
    detail_shape = env.observation_space['detail'].shape
    
    agent = D3QNAgent(
        window_size=obs_shape[0],
        n_main_features=obs_shape[1],
        n_detail_features=detail_shape[2],
        detail_len=detail_shape[1],
        action_dim=env.action_space.n
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
    
    with Progress(
        SpinnerColumn(), 
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Backtesting...", total=env.n_steps)
        
        while not done:
            mask = env.get_action_mask()
            action = agent.select_action(state, training=False, action_mask=mask)
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
    """Run paper trading simulation"""
    # Setup logger
    logger = setup_logger("live")

    console.print(Panel(f"Starting Paper Trade Simulation ({args.symbol})", style="bold red"))
    console.print("[dim]Simulating live market execution...[/dim]")

    # Log configuration
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Timeframes: {args.main_tf} + {args.detail_tf}")
    logger.info(f"Initial Balance: ${args.balance:.2f}")
    logger.info(f"Date Range: {args.start} to {args.end}")
    logger.info("")

    # 1. Load Data (Use recent data for simulation)
    try:
        df_main, df_detail = load_multi_timeframe_data(
            symbol=args.symbol,
            main_timeframe=args.main_tf,
            detail_timeframe=args.detail_tf,
            start_date=args.start,
            end_date=args.end
        )
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        logger.error(f"Error loading data: {e}")
        return

    # 2. Prepare Data
    data_dict = prepare_multi_timeframe_env_data(
        df_main, df_detail, 
        main_timeframe=args.main_tf, 
        detail_timeframe=args.detail_tf
    )
    
    # 3. Init Env with streaming mode
    env = TradingEnv(data_dict, initial_balance=args.balance, streaming_mode=True)
    
    # 4. Load Agent
    obs_shape = env.observation_space['main'].shape
    detail_shape = env.observation_space['detail'].shape
    
    agent = D3QNAgent(
        window_size=obs_shape[0],
        n_main_features=obs_shape[1],
        n_detail_features=detail_shape[2],
        detail_len=detail_shape[1],
        action_dim=env.action_space.n
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
    
    try:
        with Live(console=console, refresh_per_second=4) as live:
            while not done:
                mask = env.get_action_mask()
                action = agent.select_action(state, training=False, action_mask=mask)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward
                
                # Update Status
                status_text = f"""
[bold cyan]Status[/bold cyan]
Step: {env.current_step}/{env.n_steps}
Balance: ${env.balance:.2f}
Equity: ${env.equity:.2f}
Position: {env.position if env.position else 'NONE'}
Trades: {env.total_trades} (Wins: {env.winning_trades})
                """
                live.update(Panel(status_text, title="Live Simulation"))
                
                # Simulate delay
                time.sleep(0.1) 
                
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
    console.print("  • TP/SL Ratio: 2.5:1.5 (was 4.5:1.5) - Better win rate")
    console.print("  • Learning Rate: 0.0003 (was 0.0001) - Faster learning")
    console.print("  • Epsilon Min: 0.15 (was 0.10) - More exploration")
    console.print("  • Epsilon Decay: 0.9998 (was 0.9995) - Slower decay")
    console.print("  • Reward Shaping: Bonus for win streaks > 40%")
    console.print("  • Strong TP bonus (+0.5), Strong SL penalty (-0.3)\n")

    # Auto-limit data range if not specified (use last 3 months for efficiency)
    if not args.start and not args.end:
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        console.print(f"[yellow]⚡ Auto-limiting data to last 3 months ({start_date} to {end_date}) for efficiency[/yellow]")
        args.start = start_date
        args.end = end_date

    # Log training configuration
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Timeframes: {args.main_tf} + {args.detail_tf}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Initial Balance: ${args.balance:.2f}")
    logger.info(f"Date Range: {args.start} to {args.end}")
    logger.info("")

    # 1. Load Data
    try:
        df_main, df_detail = load_multi_timeframe_data(
            symbol=args.symbol,
            main_timeframe=args.main_tf,
            detail_timeframe=args.detail_tf,
            start_date=args.start,
            end_date=args.end
        )
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        return

    # 2. Prepare Data
    data_dict = prepare_multi_timeframe_env_data(
        df_main, df_detail, 
        main_timeframe=args.main_tf, 
        detail_timeframe=args.detail_tf
    )
    
    # 3. Init Env
    env = TradingEnv(data_dict, initial_balance=args.balance)
    
    # 4. Init Agent
    obs_shape = env.observation_space['main'].shape
    detail_shape = env.observation_space['detail'].shape
    
    agent = D3QNAgent(
        window_size=obs_shape[0],
        n_main_features=obs_shape[1],
        n_detail_features=detail_shape[2],
        detail_len=detail_shape[1],
        action_dim=env.action_space.n
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

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Episode {episode+1}/{args.episodes}", total=env.n_steps)

            while not done:
                mask = env.get_action_mask()
                action = agent.select_action(state, action_mask=mask)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store_transition(state, action, reward, next_state, done)

                # Update every N steps for efficiency
                step_count += 1
                if step_count % update_frequency == 0:
                    loss = agent.update()

                state = next_state
                total_reward += reward
                progress.update(task, advance=1)

        # Calculate metrics
        win_rate = (env.winning_trades / max(1, env.total_trades)) * 100
        total_return = ((env.balance - env.initial_balance) / env.initial_balance) * 100

        console.print(f"Episode {episode+1}/{args.episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f} | Balance: ${env.balance:.2f}")

        # Log episode results
        logger.info(f"Episode {episode+1}/{args.episodes}")
        logger.info(f"  Total Reward: {total_reward:.2f}")
        logger.info(f"  Final Balance: ${env.balance:.2f}")
        logger.info(f"  Total Return: {total_return:.2f}%")
        logger.info(f"  Total Trades: {env.total_trades}")
        logger.info(f"  Win Rate: {win_rate:.2f}%")
        logger.info(f"  Epsilon: {agent.epsilon:.4f}")
        logger.info(f"  Steps: {env.current_step}/{env.n_steps}")
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
