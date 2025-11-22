#!/usr/bin/env python3
"""
Interactive Launcher for CRYPPO (Multi-Timeframe)
"""

import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime, timedelta

try:
    import questionary
    from questionary import Style
except ImportError:
    print("Installing required dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "questionary>=2.0.0"])
    import questionary
    from questionary import Style

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.table import Table

console = Console()

# Custom style for questionary
custom_style = Style([
    ('qmark', 'fg:#673ab7 bold'),
    ('question', 'bold'),
    ('answer', 'fg:#f44336 bold'),
    ('pointer', 'fg:#673ab7 bold'),
    ('highlighted', 'fg:#673ab7 bold'),
    ('selected', 'fg:#cc5454'),
    ('separator', 'fg:#cc5454'),
    ('instruction', ''),
    ('text', ''),
    ('disabled', 'fg:#858585 italic')
])

def print_logo():
    """Print CRYPPO ASCII logo with beautiful border"""
    logo = """
╔══════════════════════════════════════════════════════╗
║         ██████╗██████╗ ██╗   ██╗██████╗ ██████╗      ║
║        ██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗██╔══██╗     ║
║        ██║     ██████╔╝ ╚████╔╝ ██████╔╝██████╔╝     ║
║        ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝ ██╔═══╝      ║
║        ╚██████╗██║  ██║   ██║   ██║     ██║          ║
║         ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚═╝          ║
║                                                      ║
║        Deep RL Crypto Trading System                 ║
║        🛡️ Risk Management: ATR-Based TP/SL            ║
╚══════════════════════════════════════════════════════╝
"""
    console.print(logo, style="bold cyan")

def run_command(args):
    """Run cryppo.py with given arguments"""
    cmd = [sys.executable, "cryppo.py"] + args
    subprocess.run(cmd)

def get_date_range():
    """Helper to get start and end dates"""
    default_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    default_end = datetime.now().strftime("%Y-%m-%d")

    console.print(f"\n[dim]Default Range: {default_start} to {default_end}[/dim]")
    start = Prompt.ask("Start Date (YYYY-MM-DD)", default=default_start)
    end = Prompt.ask("End Date (YYYY-MM-DD)", default=default_end)
    return start, end

def list_data_files():
    """List downloaded data files"""
    console.print("\n[bold cyan]📂 Available Data Files[/bold cyan]\n")

    data_dir = Path("data")
    if not data_dir.exists():
        console.print("[yellow]No data directory found[/yellow]")
        return

    # Look for both .parquet and .csv files
    parquet_files = list(data_dir.glob("*.parquet"))
    csv_files = list(data_dir.glob("*.csv"))
    data_files = parquet_files + csv_files

    if not data_files:
        console.print("[yellow]No data files found[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Modified", style="yellow")

    for file in sorted(data_files):
        size = file.stat().st_size / 1024  # KB
        mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

        if size < 1024:
            size_str = f"{size:.1f} KB"
        else:
            size_str = f"{size/1024:.1f} MB"

        table.add_row(file.name, size_str, mtime)

    console.print(table)

def list_models():
    """List trained models"""
    console.print("\n[bold cyan]📁 Trained Models[/bold cyan]\n")

    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        console.print("[yellow]No checkpoints directory found[/yellow]")
        return

    # Look for .pth, .pt, and .zip model files
    model_files = (list(checkpoints_dir.glob("*.pth")) +
                   list(checkpoints_dir.glob("*.pt")) +
                   list(checkpoints_dir.glob("*.zip")))

    if not model_files:
        console.print("[yellow]No model files found[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Created", style="yellow")

    for file in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True):
        size = file.stat().st_size / 1024  # KB
        mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

        if size < 1024:
            size_str = f"{size:.1f} KB"
        else:
            size_str = f"{size/1024:.1f} MB"

        table.add_row(file.name, size_str, mtime)

    console.print(table)

def system_check():
    """Run system check"""
    console.print("\n[bold green]🔍 Running System Check...[/bold green]\n")
    run_command(["check"])

def download_data():
    """Download market data"""
    console.print("\n[bold yellow]📥 Download Market Data[/bold yellow]")
    symbol = Prompt.ask("Symbol", default="BTCUSDT")
    start, end = get_date_range()

    if Confirm.ask(f"Download {symbol} (1h & 1m) from {start} to {end}?"):
        # Download 1h
        console.print("\n[cyan]Downloading 1h timeframe...[/cyan]")
        run_command(["data", "download", "--symbol", symbol, "--timeframe", "1h", "--start", start, "--end", end])
        # Download 1m
        console.print("\n[cyan]Downloading 1m timeframe...[/cyan]")
        run_command(["data", "download", "--symbol", symbol, "--timeframe", "1m", "--start", start, "--end", end])

def train_model():
    """Train a new model"""
    console.print("\n[bold blue]🚀 Train Model[/bold blue]")
    symbol = Prompt.ask("Symbol", default="BTCUSDT")
    episodes = Prompt.ask("Episodes", default="10")

    use_date_range = Confirm.ask("Filter by date range?", default=False)
    args = ["train", "--symbol", symbol, "--episodes", episodes]

    if use_date_range:
        start, end = get_date_range()
        args.extend(["--start", start, "--end", end])

    run_command(args)

def backtest_model():
    """Backtest a trained model"""
    console.print("\n[bold magenta]📊 Backtest Model[/bold magenta]")
    symbol = Prompt.ask("Symbol", default="BTCUSDT")

    use_date_range = Confirm.ask("Filter by date range?", default=False)
    args = ["backtest", "--symbol", symbol]

    if use_date_range:
        start, end = get_date_range()
        args.extend(["--start", start, "--end", end])

    run_command(args)

def paper_trade():
    """Run paper trade simulation (historical data)"""
    console.print("\n[bold red]📈 Paper Trade (Historical Simulation)[/bold red]")
    console.print("[dim]Note: This uses historical data. For real-time, use: python paper_trade_live.py[/dim]\n")
    symbol = Prompt.ask("Symbol", default="BTCUSDT")

    # Paper trade uses test set (last 3 months) - no date range needed
    args = ["paper", "--symbol", symbol]
    console.print("[cyan]Using test set (last 3 months of data)[/cyan]\n")

    run_command(args)

def auto_tune():
    """Auto-tune hyperparameters for maximum profitability"""
    console.print("\n[bold yellow]🔧 Auto Tune Parameters[/bold yellow]")
    console.print("[dim]This will automatically find the best parameters for profitability[/dim]\n")

    symbol = Prompt.ask("Symbol", default="BTCUSDT")
    trials = Prompt.ask("Number of trials", default="50")
    train_episodes = Prompt.ask("Training episodes per trial", default="3")
    use_full_data = Confirm.ask("Use full dataset (recommended)?", default=True)

    args = [
        sys.executable, "auto_tune.py",
        "--symbol", symbol,
        "--trials", trials,
        "--train-episodes", train_episodes
    ]

    if use_full_data:
        args.append("--full-data")

    console.print("\n[cyan]Starting auto-tuning... This may take a while.[/cyan]")
    subprocess.run(args)

def main():
    """Main menu loop"""
    while True:
        console.clear()
        print_logo()

        console.print()  # Add spacing

        # Menu choices
        choices = [
            "🔍 System Check",
            "📥 Download Data",
            "📂 List Data",
            "🚀 Train Model",
            "🔧 Auto Tune Parameters",
            "📊 Backtest",
            "📈 Paper Trade (Live Simulation)",
            "📁 List Models",
            "❌ Exit"
        ]

        choice = questionary.select(
            "What would you like to do?",
            choices=choices,
            style=custom_style,
            qmark="?",
            pointer="❯"
        ).ask()

        if choice is None:  # User pressed Ctrl+C
            console.print("\n[bold]Goodbye![/bold]")
            break

        console.clear()
        print_logo()

        # Handle menu selection
        if choice == "🔍 System Check":
            system_check()
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")

        elif choice == "📥 Download Data":
            download_data()
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")

        elif choice == "📂 List Data":
            list_data_files()
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")

        elif choice == "🚀 Train Model":
            train_model()
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")

        elif choice == "🔧 Auto Tune Parameters":
            auto_tune()
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")

        elif choice == "📊 Backtest":
            backtest_model()
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")

        elif choice == "📈 Paper Trade (Live Simulation)":
            paper_trade()
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")

        elif choice == "📁 List Models":
            list_models()
            Prompt.ask("\n[dim]Press Enter to continue...[/dim]")

        elif choice == "❌ Exit":
            console.print("\n[bold cyan]Thanks for using CRYPPO![/bold cyan]")
            console.print("[dim]Happy trading! 🚀[/dim]\n")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bold]Interrupted by user. Goodbye![/bold]")
        sys.exit(0)
