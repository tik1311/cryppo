#!/usr/bin/env python3
"""
CRYPPO - Interactive Main Entry Point
Beautiful interactive CLI with arrow key navigation

Usage:
    python main.py                    # Interactive mode
    python main.py check              # Direct command mode
    python main.py train --help       # Command-line mode
"""

import sys
import subprocess
from typing import Optional

try:
    import questionary
    from questionary import Style
except ImportError:
    print("Installing questionary for interactive menus...")
    subprocess.run([sys.executable, "-m", "pip", "install", "questionary>=2.0.0"], check=True)
    import questionary
    from questionary import Style

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box

console = Console()

# Custom style for menus
custom_style = Style([
    ('qmark', 'fg:#00ffff bold'),           # Question mark
    ('question', 'bold'),                    # Question text
    ('answer', 'fg:#00ff00 bold'),          # Selected answer
    ('pointer', 'fg:#00ffff bold'),         # Pointer
    ('highlighted', 'fg:#00ffff bold'),     # Highlighted choice
    ('selected', 'fg:#00ff00'),             # Selected choice
    ('separator', 'fg:#666666'),            # Separator
    ('instruction', 'fg:#666666'),          # Instructions
])

BANNER = """
  ██████╗██████╗ ██╗   ██╗██████╗ ██████╗  ██████╗
 ██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗██╔══██╗██╔═══██╗
 ██║     ██████╔╝ ╚████╔╝ ██████╔╝██████╔╝██║   ██║
 ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝ ██╔═══╝ ██║   ██║
 ╚██████╗██║  ██║   ██║   ██║     ██║     ╚██████╔╝
  ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚═╝      ╚═════╝

 Deep RL Crypto Trading System
 GPU-Accelerated | Python 3.12 | CUDA 12.6
 🛡️ Risk Management: ATR-Based TP/SL + Trailing Stop
"""


def print_banner():
    """Print beautiful startup banner"""
    console.print(
        Panel(
            Align.center(Text(BANNER, style="bold cyan")),
            border_style="bright_cyan",
            box=box.DOUBLE,
            padding=(1, 2)
        )
    )


def run_cryppo_command(args: list):
    """Execute cryppo.py with given arguments"""
    cmd = [sys.executable, "cryppo.py"] + args
    console.print(f"\n[dim]Running: {' '.join(cmd)}[/dim]\n")
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        return False


def handle_system_check():
    """Run system check"""
    return run_cryppo_command(["check"])


def handle_download_data():
    """Download market data"""
    console.print("\n[bold cyan]📥 Download Market Data[/bold cyan]\n")

    symbol = questionary.text(
        "Trading symbol:",
        default="BTCUSDT",
        style=custom_style
    ).ask()

    if not symbol:
        return False

    timeframe = questionary.select(
        "Select timeframe:",
        choices=["1h", "4h", "1d", "15m", "5m"],
        default="1h",
        style=custom_style
    ).ask()

    start = questionary.text(
        "Start date (YYYY-MM-DD):",
        default="2023-01-01",
        style=custom_style
    ).ask()

    end = questionary.text(
        "End date (YYYY-MM-DD):",
        default="2024-12-31",
        style=custom_style
    ).ask()

    if not all([symbol, timeframe, start, end]):
        return False

    return run_cryppo_command([
        "data", "download",
        "--symbol", symbol.upper(),
        "--timeframe", timeframe,
        "--start", start,
        "--end", end
    ])


def handle_list_data():
    """List available data files"""
    return run_cryppo_command(["data", "list"])


def handle_train():
    """Train RL model"""
    console.print("\n[bold cyan]🚀 Train RL Model[/bold cyan]\n")

    model = questionary.select(
        "Select model type:",
        choices=["d3qn", "ppo", "sac", "td3"],
        default="d3qn",
        style=custom_style
    ).ask()

    if not model:
        return False

    symbol = questionary.text(
        "Trading symbol:",
        default="BTCUSDT",
        style=custom_style
    ).ask()

    reward = questionary.select(
        "Select reward type:",
        choices=["sharpe", "sortino", "profit", "return"],
        default="sharpe",
        style=custom_style
    ).ask()

    episodes = questionary.text(
        "Number of episodes:",
        default="100",
        validate=lambda x: x.isdigit() and int(x) > 0,
        style=custom_style
    ).ask()

    sample_data = questionary.confirm(
        "Use sample data for testing?",
        default=True,
        style=custom_style
    ).ask()

    if not all([model, symbol, reward, episodes is not None]):
        return False

    args = [
        "train",
        "--model", model,
        "--symbol", symbol.upper(),
        "--reward", reward,
        "--episodes", episodes
    ]

    if sample_data:
        args.append("--sample-data")

    return run_cryppo_command(args)


def handle_backtest():
    """Run backtest on trained model"""
    console.print("\n[bold cyan]📊 Backtest Model[/bold cyan]\n")

    model_path = questionary.text(
        "Model path:",
        default="checkpoints/final_agent.pth",
        style=custom_style
    ).ask()

    if not model_path:
        return False

    sample_data = questionary.confirm(
        "Use sample data for testing?",
        default=True,
        style=custom_style
    ).ask()

    args = ["backtest", "--model", model_path]

    if sample_data:
        args.append("--sample-data")

    return run_cryppo_command(args)


def handle_list_models():
    """List trained models"""
    return run_cryppo_command(["models"])


def show_main_menu():
    """Show main interactive menu"""
    print_banner()

    console.print("\n[bold]Use arrow keys to navigate, Enter to select:[/bold]\n")

    choices = [
        "🔍 System Check - Validate environment and dependencies",
        "📥 Download Data - Fetch market data from exchange",
        "📂 List Data - Show available data files",
        "🚀 Train Model - Train RL agent (D3QN/PPO/SAC/TD3)",
        "📊 Backtest - Evaluate trained model",
        "📁 List Models - Show trained checkpoints",
        "❌ Exit"
    ]

    action = questionary.select(
        "What would you like to do?",
        choices=choices,
        style=custom_style,
        use_shortcuts=True,
        use_arrow_keys=True,
        instruction="\n"
    ).ask()

    if not action or "Exit" in action:
        console.print("\n[bold green]👋 Goodbye![/bold green]\n")
        return False

    # Map choice to handler
    handlers = {
        "System Check": handle_system_check,
        "Download Data": handle_download_data,
        "List Data": handle_list_data,
        "Train Model": handle_train,
        "Backtest": handle_backtest,
        "List Models": handle_list_models,
    }

    for key, handler in handlers.items():
        if key in action:
            success = handler()

            # Ask to continue
            if success:
                continue_choice = questionary.confirm(
                    "\nRun another task?",
                    default=True,
                    style=custom_style
                ).ask()

                if continue_choice:
                    console.clear()
                    return True

            return False

    return False


def main():
    """Main entry point"""
    # If arguments provided, pass directly to cryppo.py
    if len(sys.argv) > 1:
        from cryppo import main as cryppo_main
        cryppo_main()
        return

    # Otherwise, show interactive menu
    try:
        while True:
            if not show_main_menu():
                break
    except KeyboardInterrupt:
        console.print("\n\n[yellow]👋 Interrupted by user[/yellow]\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
