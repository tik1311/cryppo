"""
Training Script for Multi-Timeframe D3QN Agent with Beautiful CLI
Phase 6: Train with hierarchical multi-timeframe system

Usage:
    python train_multitf.py --primary-tf 5m --episodes 200 --symbol BTCUSDT
"""

import argparse
import numpy as np
from pathlib import Path
import torch
from datetime import datetime
import json
import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import box

from server.src.data.multi_timeframe_loader import MultiTimeframeLoader
from server.src.policy.multi_timeframe_env import make_multitf_env
from server.src.agent.multi_timeframe_agent import MultiTimeframeD3QNAgent


console = Console()


def create_status_table(episode, total_episodes, metrics):
    """Create beautiful status table that updates in place"""
    table = Table(title=f"Episode {episode}/{total_episodes}", box=box.ROUNDED, show_header=True, header_style="bold cyan")

    table.add_column("Metric", style="cyan", justify="right")
    table.add_column("Value", style="green", justify="left")

    if metrics:
        latest = {k: v[-1] if v else 0 for k, v in metrics.items()}

        table.add_row("💰 Balance", f"${latest['final_balance']:.2f}")
        profit_pct = (latest['final_balance'] / 10000 - 1) * 100
        table.add_row("📈 Profit", f"{profit_pct:+.2f}%")
        table.add_row("🎯 Win Rate", f"{latest['win_rate']*100:.1f}%")
        table.add_row("📊 Trades", f"{latest['total_trades']}")
        table.add_row("⏸️  HOLD %", f"{latest['hold_pct']:.1f}%")
        table.add_row("🎲 Epsilon", f"{latest['epsilon']:.4f}")
        table.add_row("🏆 Reward", f"{latest['rewards']:.2f}")

    return table


def train_multitf_agent(args):
    """Train Multi-Timeframe D3QN Agent with Beautiful CLI"""

    # Header
    console.print(Panel.fit(
        "[bold cyan]🚀 Phase 6: Multi-Timeframe D3QN Training[/bold cyan]\n"
        f"[yellow]Primary: {args.primary_tf} | Detail: 1m | Episodes: {args.episodes}[/yellow]",
        border_style="cyan"
    ))

    try:
        # Step 1: Create data loader
        console.print("\n[bold cyan]📦 Step 1:[/bold cyan] Initializing data loader...")
        loader = MultiTimeframeLoader(
            primary_timeframe=args.primary_tf,
            detail_timeframe='1m',
            window_size=args.window_size
        )
        console.print("[green]✓[/green] Data loader ready\n")

        # Step 2: Download data
        console.print(f"[bold cyan]📥 Step 2:[/bold cyan] Downloading {args.symbol} data...")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task(f"Fetching {args.primary_tf} and 1m candles...", total=None)

            try:
                primary_df, detail_df = loader.download_data(
                    symbol=args.symbol,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    data_dir=Path(args.data_dir)
                )
                progress.update(task, completed=True)
            except Exception as e:
                console.print(f"[red]✗ Download failed: {e}[/red]")
                console.print("[yellow]Using existing data if available...[/yellow]")
                # Try to load existing data
                primary_path = Path(args.data_dir) / f"{args.symbol.replace('/', '_')}_{args.primary_tf}.parquet"
                detail_path = Path(args.data_dir) / f"{args.symbol.replace('/', '_')}_1m.parquet"
                if primary_path.exists() and detail_path.exists():
                    import pandas as pd
                    primary_df = pd.read_parquet(primary_path)
                    detail_df = pd.read_parquet(detail_path)
                    console.print("[green]✓[/green] Loaded existing data")
                else:
                    console.print("[red]✗ No existing data found[/red]")
                    return

        console.print(f"[green]✓[/green] Downloaded {len(primary_df)} primary + {len(detail_df)} detail candles\n")

        # Step 3: Prepare data
        console.print("[bold cyan]🔧 Step 3:[/bold cyan] Preparing multi-timeframe data...")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Aligning timeframes & adding indicators...", total=None)
            data = loader.prepare_for_training(primary_df, detail_df, normalize=True)
            progress.update(task, completed=True)

        primary_data = data['primary']
        detail_data = data['detail']

        console.print(f"[green]✓[/green] Primary: {primary_data.shape} | Detail: {detail_data.shape}\n")

        # Step 4: Create environment
        console.print("[bold cyan]🏗️  Step 4:[/bold cyan] Creating environment...")
        env = make_multitf_env(
            primary_data=primary_data,
            detail_data=detail_data,
            primary_timeframe=args.primary_tf,
            initial_balance=args.initial_balance,
            window_size=args.window_size,
            fee_rate=0.0004,
            max_leverage=1.0,
            atr_sl_multiplier=1.3,
            atr_tp_multiplier=6.0
        )
        console.print("[green]✓[/green] Multi-timeframe environment ready\n")

        # Step 5: Create agent
        console.print("[bold cyan]🤖 Step 5:[/bold cyan] Creating D3QN agent...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        console.print(f"[yellow]Using device: {device.upper()}[/yellow]")

        agent = MultiTimeframeD3QNAgent(
            primary_window=args.window_size,
            primary_features=primary_data.shape[1],
            detail_ratio=detail_data.shape[1],
            detail_features=detail_data.shape[2],
            action_dim=3,
            n_quantiles=32,
            learning_rate=args.lr,
            gamma=0.995,
            buffer_size=100000,
            batch_size=128,
            epsilon_start=1.0,
            epsilon_end=0.25,
            epsilon_decay=0.9997,
            target_update_freq=250,
            device=device
        )
        console.print("[green]✓[/green] Agent initialized with ~730k parameters\n")

        # Step 6: Training
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        console.print(Panel.fit(
            "[bold green]🚀 Starting Training[/bold green]\n"
            f"Episodes: {args.episodes} | Save every: {args.save_freq} ep",
            border_style="green"
        ))

        metrics = {
            'episodes': [],
            'rewards': [],
            'final_balance': [],
            'win_rate': [],
            'total_trades': [],
            'hold_pct': [],
            'epsilon': []
        }

        # Training loop with live display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            training_task = progress.add_task(
                "[cyan]Training episodes...",
                total=args.episodes
            )

            for episode in range(1, args.episodes + 1):
                state, info = env.reset()
                episode_reward = 0
                steps = 0
                done = False

                # Episode progress
                progress.update(training_task, description=f"[cyan]Episode {episode}/{args.episodes}")

                while not done:
                    action_mask = env.get_action_mask()
                    action = agent.select_action(
                        state['primary'],
                        state['detail'],
                        action_mask,
                        training=True
                    )

                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    agent.store_experience(
                        state['primary'], state['detail'], action, reward,
                        next_state['primary'], next_state['detail'], done
                    )

                    if len(agent.replay_buffer) >= agent.batch_size:
                        loss = agent.train_step()

                    episode_reward += reward
                    state = next_state
                    steps += 1

                # Log metrics
                final_balance = info['balance']
                win_rate = info['win_rate']
                total_trades = info['total_trades']

                action_counts = env.env.action_counts
                total_actions = sum(action_counts.values())
                hold_pct = action_counts.get(0, 0) / max(total_actions, 1) * 100

                metrics['episodes'].append(episode)
                metrics['rewards'].append(float(episode_reward))
                metrics['final_balance'].append(float(final_balance))
                metrics['win_rate'].append(float(win_rate))
                metrics['total_trades'].append(int(total_trades))
                metrics['hold_pct'].append(float(hold_pct))
                metrics['epsilon'].append(float(agent.epsilon))

                # Update progress
                progress.update(training_task, advance=1)

                # Print status every N episodes
                if episode % 10 == 0 or episode == args.episodes:
                    console.print("\n")
                    console.print(create_status_table(episode, args.episodes, metrics))

                # Save checkpoint
                if episode % args.save_freq == 0:
                    checkpoint_path = log_dir / f"mt_d3qn_ep{episode}.pth"
                    agent.save(checkpoint_path)
                    console.print(f"[dim]💾 Saved checkpoint: {checkpoint_path.name}[/dim]")

        # Save final model
        final_path = log_dir / "mt_d3qn_final.pth"
        agent.save(final_path)

        # Save metrics
        metrics_path = log_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Summary
        console.print("\n")
        console.print(Panel.fit(
            "[bold green]✓ Training Completed![/bold green]\n\n"
            f"[cyan]Final Model:[/cyan] {final_path}\n"
            f"[cyan]Metrics:[/cyan] {metrics_path}\n"
            f"[cyan]Episodes:[/cyan] {args.episodes}\n"
            f"[cyan]Final Balance:[/cyan] ${metrics['final_balance'][-1]:.2f}",
            border_style="green",
            title="🎉 Success"
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Training interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Multi-Timeframe D3QN Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with 5m primary + 1m detail
  python train_multitf.py --primary-tf 5m --episodes 200

  # Train with 15m primary
  python train_multitf.py --primary-tf 15m --episodes 200 --symbol BTC/USDT

  # Short test run
  python train_multitf.py --episodes 10 --save-freq 5
        """
    )

    # Data args
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--primary-tf', type=str, default='5m', help='Primary timeframe (5m, 15m, 1h)')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')

    # Training args
    parser.add_argument('--episodes', type=int, default=200, help='Number of episodes')
    parser.add_argument('--window-size', type=int, default=72, help='Window size')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')

    # Logging args
    parser.add_argument('--log-dir', type=str, default='logs/multitf', help='Log directory')
    parser.add_argument('--save-freq', type=int, default=50, help='Save frequency (episodes)')

    args = parser.parse_args()

    train_multitf_agent(args)
