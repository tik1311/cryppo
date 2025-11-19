"""
Training Script for Multi-Timeframe D3QN Agent with Interactive CLI
Phase 6: Train with hierarchical multi-timeframe system

Usage:
    python train_multitf.py --primary-tf 5m --episodes 200 --symbol BTCUSDT

Controls during training:
    Space - Pause/Resume
    Q - Quit training
    S - Save checkpoint now
"""

import argparse
import numpy as np
from pathlib import Path
import torch
from datetime import datetime
import json
import sys
import threading
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import box
from rich.text import Text
from rich.align import Align
import select

from server.src.data.multi_timeframe_loader import MultiTimeframeLoader
from server.src.policy.multi_timeframe_env import make_multitf_env
from server.src.agent.multi_timeframe_agent import MultiTimeframeD3QNAgent


console = Console()


class TrainingState:
    """Shared state between training and keyboard listener"""
    def __init__(self):
        self.paused = False
        self.should_quit = False
        self.should_save = False
        self.last_key = None

    def reset_flags(self):
        self.should_save = False


def make_header(primary_tf: str, episodes: int, device: str) -> Panel:
    """Create header panel"""
    content = Text()
    content.append("🚀 Multi-Timeframe D3QN Training\n", style="bold cyan")
    content.append(f"Primary: {primary_tf} | Detail: 1m | Episodes: {episodes}\n", style="yellow")
    content.append(f"Device: {device.upper()}", style="green")

    return Panel(
        Align.center(content),
        border_style="cyan",
        box=box.DOUBLE
    )


def make_controls_panel(state: TrainingState) -> Panel:
    """Create controls panel"""
    controls = Table.grid(padding=(0, 2))
    controls.add_column(style="cyan", justify="center")
    controls.add_column(style="yellow", justify="center")
    controls.add_column(style="green", justify="center")

    pause_text = "▶️  RESUME" if state.paused else "⏸️  PAUSE"
    controls.add_row(
        "[Space] " + pause_text,
        "[S] 💾 SAVE",
        "[Q] ❌ QUIT"
    )

    status_text = "⏸  PAUSED" if state.paused else "▶️  RUNNING"
    status_style = "yellow" if state.paused else "green"

    return Panel(
        controls,
        title=f"[{status_style}]{status_text}[/{status_style}]",
        border_style="blue",
        box=box.ROUNDED
    )


def make_progress_panel(episode: int, total: int, step: int, epsilon: float) -> Panel:
    """Create progress panel"""
    progress_pct = (episode / total) * 100
    bar_width = 30
    filled = int(bar_width * episode / total)
    bar = "█" * filled + "░" * (bar_width - filled)

    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", justify="right")
    table.add_column(style="white", justify="left")

    table.add_row("Episode:", f"{episode}/{total}")
    table.add_row("Progress:", f"[cyan]{bar}[/cyan] {progress_pct:.1f}%")
    table.add_row("Steps:", f"{step:,}")
    table.add_row("Epsilon:", f"{epsilon:.4f}")

    return Panel(
        table,
        title="[bold cyan]📊 Training Progress[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED
    )


def make_metrics_panel(metrics: dict) -> Panel:
    """Create metrics panel"""
    if not metrics['episodes']:
        return Panel(
            "[dim]Waiting for first episode...[/dim]",
            title="[bold green]📈 Episode Metrics[/bold green]",
            border_style="green",
            box=box.ROUNDED
        )

    latest = {k: v[-1] if v else 0 for k, v in metrics.items()}

    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", justify="right")
    table.add_column(justify="left")

    balance = latest['final_balance']
    profit_pct = (balance / 10000 - 1) * 100
    profit_color = "green" if profit_pct >= 0 else "red"

    table.add_row("💰 Balance:", f"[white]${balance:,.2f}[/white]")
    table.add_row("📈 Profit:", f"[{profit_color}]{profit_pct:+.2f}%[/{profit_color}]")
    table.add_row("🎯 Win Rate:", f"[white]{latest['win_rate']*100:.1f}%[/white]")
    table.add_row("📊 Trades:", f"[white]{latest['total_trades']}[/white]")
    table.add_row("⏸️  HOLD%:", f"[yellow]{latest['hold_pct']:.1f}%[/yellow]")
    table.add_row("🏆 Reward:", f"[white]{latest['rewards']:.2f}[/white]")

    return Panel(
        table,
        title="[bold green]📈 Episode Metrics[/bold green]",
        border_style="green",
        box=box.ROUNDED
    )


def make_stats_panel(metrics: dict) -> Panel:
    """Create statistics panel"""
    if len(metrics['episodes']) < 2:
        return Panel(
            "[dim]Need more episodes for statistics...[/dim]",
            title="[bold magenta]📊 Statistics[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED
        )

    # Calculate stats
    recent_n = min(20, len(metrics['rewards']))
    recent_rewards = metrics['rewards'][-recent_n:]
    recent_balances = metrics['final_balance'][-recent_n:]
    recent_win_rates = metrics['win_rate'][-recent_n:]

    avg_reward = np.mean(recent_rewards)
    avg_balance = np.mean(recent_balances)
    avg_win_rate = np.mean(recent_win_rates)

    best_balance = max(metrics['final_balance'])
    best_episode = metrics['episodes'][metrics['final_balance'].index(best_balance)]

    table = Table.grid(padding=(0, 2))
    table.add_column(style="magenta", justify="right")
    table.add_column(style="white", justify="left")

    table.add_row(f"Last {recent_n} Avg:", "")
    table.add_row("  Reward:", f"{avg_reward:.2f}")
    table.add_row("  Balance:", f"${avg_balance:,.2f}")
    table.add_row("  Win Rate:", f"{avg_win_rate*100:.1f}%")
    table.add_row("", "")
    table.add_row("Best Balance:", f"${best_balance:,.2f} (ep {best_episode})")

    return Panel(
        table,
        title="[bold magenta]📊 Statistics[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED
    )


def make_layout() -> Layout:
    """Create layout for live display"""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="controls", size=3),
        Layout(name="body"),
        Layout(name="footer", size=1)
    )

    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right")
    )

    layout["left"].split_column(
        Layout(name="progress"),
        Layout(name="metrics")
    )

    layout["right"].split_column(
        Layout(name="stats")
    )

    return layout


def keyboard_listener(state: TrainingState):
    """Listen for keyboard input in background thread"""
    import sys
    import tty
    import termios

    # Only works on Unix systems
    if not hasattr(sys.stdin, 'fileno'):
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        while not state.should_quit:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1).lower()
                state.last_key = ch

                if ch == ' ':
                    state.paused = not state.paused
                elif ch == 'q':
                    state.should_quit = True
                elif ch == 's':
                    state.should_save = True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def train_multitf_agent(args):
    """Train Multi-Timeframe D3QN Agent with Interactive CLI"""

    # Show startup messages
    console.print(Panel.fit(
        "[bold cyan]🚀 Initializing Multi-Timeframe D3QN Training[/bold cyan]",
        border_style="cyan"
    ))

    try:
        # Step 1: Create data loader
        console.print("[cyan]📦 Loading data...[/cyan]", end=" ")
        loader = MultiTimeframeLoader(
            primary_timeframe=args.primary_tf,
            detail_timeframe='1m',
            window_size=args.window_size
        )
        console.print("[green]✓[/green]")

        # Step 2: Download/load data
        try:
            primary_df, detail_df = loader.download_data(
                symbol=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                data_dir=Path(args.data_dir)
            )
        except Exception:
            # Try to load existing data
            primary_path = Path(args.data_dir) / f"{args.symbol.replace('/', '_')}_{args.primary_tf}.parquet"
            detail_path = Path(args.data_dir) / f"{args.symbol.replace('/', '_')}_1m.parquet"
            if primary_path.exists() and detail_path.exists():
                import pandas as pd
                primary_df = pd.read_parquet(primary_path)
                detail_df = pd.read_parquet(detail_path)
            else:
                console.print("[red]✗ No data found[/red]")
                return

        console.print(f"[cyan]🔧 Preparing data...[/cyan] ({len(primary_df)} candles)", end=" ")
        data = loader.prepare_for_training(primary_df, detail_df, normalize=True)
        primary_data = data['primary']
        detail_data = data['detail']
        console.print("[green]✓[/green]")

        # Step 3: Create environment
        console.print("[cyan]🏗️  Creating environment...[/cyan]", end=" ")
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
        console.print("[green]✓[/green]")

        # Step 4: Create agent
        console.print("[cyan]🤖 Creating agent...[/cyan]", end=" ")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        console.print("[green]✓[/green]")

        # Step 5: Setup logging
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[green]✓ Setup complete! Starting training...[/green]\n")
        time.sleep(1)

        # Training state
        state = TrainingState()
        metrics = {
            'episodes': [],
            'rewards': [],
            'final_balance': [],
            'win_rate': [],
            'total_trades': [],
            'hold_pct': [],
            'epsilon': []
        }

        # Start keyboard listener in background thread
        if sys.platform != 'win32':  # Only on Unix
            listener_thread = threading.Thread(target=keyboard_listener, args=(state,), daemon=True)
            listener_thread.start()

        # Training loop with live display
        layout = make_layout()

        with Live(layout, console=console, refresh_per_second=4, screen=False) as live:

            for episode in range(1, args.episodes + 1):
                if state.should_quit:
                    break

                env_state, info = env.reset()
                episode_reward = 0
                steps = 0
                done = False

                while not done:
                    # Check pause state
                    while state.paused and not state.should_quit:
                        time.sleep(0.1)
                        # Update display while paused
                        layout["header"] = make_header(args.primary_tf, args.episodes, device)
                        layout["controls"] = make_controls_panel(state)
                        layout["progress"] = make_progress_panel(episode, args.episodes, steps, agent.epsilon)
                        layout["metrics"] = make_metrics_panel(metrics)
                        layout["stats"] = make_stats_panel(metrics)
                        layout["footer"] = Align.center(
                            Text("⏸  Training paused - Press Space to resume", style="yellow bold")
                        )

                    if state.should_quit:
                        break

                    # Training step
                    action_mask = env.get_action_mask()
                    action = agent.select_action(
                        env_state['primary'],
                        env_state['detail'],
                        action_mask,
                        training=True
                    )

                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    agent.store_experience(
                        env_state['primary'], env_state['detail'], action, reward,
                        next_state['primary'], next_state['detail'], done
                    )

                    if len(agent.replay_buffer) >= agent.batch_size:
                        loss = agent.train_step()

                    episode_reward += reward
                    env_state = next_state
                    steps += 1

                    # Update display every 10 steps
                    if steps % 10 == 0:
                        layout["header"] = make_header(args.primary_tf, args.episodes, device)
                        layout["controls"] = make_controls_panel(state)
                        layout["progress"] = make_progress_panel(episode, args.episodes, steps, agent.epsilon)
                        layout["metrics"] = make_metrics_panel(metrics)
                        layout["stats"] = make_stats_panel(metrics)
                        layout["footer"] = Align.center(
                            Text(f"Episode {episode}/{args.episodes} • Step {steps}", style="dim")
                        )

                if state.should_quit:
                    break

                # Log episode metrics
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

                # Update final display
                layout["header"] = make_header(args.primary_tf, args.episodes, device)
                layout["controls"] = make_controls_panel(state)
                layout["progress"] = make_progress_panel(episode, args.episodes, steps, agent.epsilon)
                layout["metrics"] = make_metrics_panel(metrics)
                layout["stats"] = make_stats_panel(metrics)
                layout["footer"] = Align.center(
                    Text(f"Episode {episode} completed • {steps} steps • Reward: {episode_reward:.2f}", style="green")
                )

                # Save checkpoint
                if state.should_save or episode % args.save_freq == 0:
                    checkpoint_path = log_dir / f"mt_d3qn_ep{episode}.pth"
                    agent.save(checkpoint_path)
                    layout["footer"] = Align.center(
                        Text(f"💾 Saved: {checkpoint_path.name}", style="cyan bold")
                    )
                    time.sleep(0.5)
                    state.reset_flags()

        # Save final model
        final_path = log_dir / "mt_d3qn_final.pth"
        agent.save(final_path)

        # Save metrics
        metrics_path = log_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Summary
        console.print("\n")
        if state.should_quit:
            console.print(Panel.fit(
                "[yellow]⚠ Training stopped by user[/yellow]\n\n"
                f"[cyan]Completed Episodes:[/cyan] {len(metrics['episodes'])}/{args.episodes}\n"
                f"[cyan]Model saved:[/cyan] {final_path}\n"
                f"[cyan]Metrics saved:[/cyan] {metrics_path}",
                border_style="yellow",
                title="⏹️  Stopped"
            ))
        else:
            profit_pct = (metrics['final_balance'][-1] / 10000 - 1) * 100
            profit_color = "green" if profit_pct >= 0 else "red"

            console.print(Panel.fit(
                "[bold green]✓ Training Completed![/bold green]\n\n"
                f"[cyan]Final Model:[/cyan] {final_path}\n"
                f"[cyan]Metrics:[/cyan] {metrics_path}\n"
                f"[cyan]Episodes:[/cyan] {args.episodes}\n"
                f"[cyan]Final Balance:[/cyan] ${metrics['final_balance'][-1]:,.2f}\n"
                f"[cyan]Final Profit:[/cyan] [{profit_color}]{profit_pct:+.2f}%[/{profit_color}]",
                border_style="green",
                title="🎉 Success"
            ))

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Training interrupted (Ctrl+C)[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Multi-Timeframe D3QN Agent with Interactive CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with 5m primary + 1m detail
  python train_multitf.py --primary-tf 5m --episodes 200

  # Train with 15m primary
  python train_multitf.py --primary-tf 15m --episodes 200 --symbol BTC/USDT

  # Short test run
  python train_multitf.py --episodes 10 --save-freq 5

Interactive Controls:
  Space - Pause/Resume training
  S     - Save checkpoint now
  Q     - Quit training gracefully
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
