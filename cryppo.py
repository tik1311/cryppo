#!/usr/bin/env python3
"""
CRYPPO - Deep RL Crypto Trading System CLI
Beautiful command-line interface inspired by Claude Code
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from typing import Optional

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
    console.print(
        Panel(
            Text(BANNER, style="bold cyan", justify="center"),
            border_style="bright_cyan",
            box=box.DOUBLE
        )
    )

# ============================================================================
# System Check
# ============================================================================

def check_system():
    """Check system requirements"""
    console.print("\n[bold cyan]🔍 System Check[/bold cyan]\n")

    checks = []

    # Check Python
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append(("Python", python_version, "✓" if sys.version_info >= (3, 10) else "✗"))

    # Check PyTorch
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        checks.append(("PyTorch", torch_version, "✓"))
        checks.append(("CUDA", "Available" if cuda_available else "Not Available", "✓" if cuda_available else "⚠"))
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            checks.append(("GPU", gpu_name, "✓"))
    except Exception as e:
        checks.append(("PyTorch", f"Error: {str(e)}", "✗"))

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
            checks.append((name, version, "✓"))
        except:
            checks.append((name, "Not installed", "✗"))

    # Create table
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Version/Status", style="white")
    table.add_column("Status", justify="center")

    for component, version, status in checks:
        style = "green" if status == "✓" else ("yellow" if status == "⚠" else "red")
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

    console.print(f"\n[bold cyan]📥 Downloading {symbol} Data[/bold cyan]\n")

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
            console.print(f"\n[green]✓[/green] Downloaded {len(df)} candles")
            console.print(f"[dim]Period: {df.index[0]} to {df.index[-1]}[/dim]")
            console.print(f"[dim]Saved to: data/{symbol}_{timeframe}.pkl[/dim]\n")

        except Exception as e:
            console.print(f"\n[red]✗ Error: {e}[/red]\n")
            sys.exit(1)

def list_data():
    """List available data files"""
    console.print("\n[bold cyan]📊 Available Data Files[/bold cyan]\n")

    data_dir = Path("data")
    if not data_dir.exists():
        console.print("[yellow]No data directory found[/yellow]\n")
        return

    files = list(data_dir.glob("*.pkl"))

    if not files:
        console.print("[yellow]No data files found[/yellow]\n")
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

# ============================================================================
# Training
# ============================================================================

def train(
    symbol: str,
    episodes: int,
    model: str = "d3qn",
    learning_rate: float = 0.00025,
    sample_data: bool = False
):
    """Train model with beautiful progress tracking"""
    console.print(f"\n[bold cyan]🚀 Training {model.upper()} Model[/bold cyan]\n")

    # Show configuration
    config_table = Table(show_header=False, box=box.SIMPLE)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="white")

    config_table.add_row("Symbol", symbol)
    config_table.add_row("Model", model.upper())
    config_table.add_row("Episodes", str(episodes))
    config_table.add_row("Learning Rate", str(learning_rate))
    config_table.add_row("Sample Data", "Yes" if sample_data else "No")

    console.print(config_table)
    console.print()

    # Build command to run training as subprocess
    # This avoids sys.path issues and PyTorch DLL conflicts
    server_dir = Path(__file__).parent / "server"

    cmd = [
        sys.executable,  # Use current Python interpreter
        "-m", "src.training.train_d3qn",
        "--symbol", symbol,
        "--episodes", str(episodes),
        "--lr", str(learning_rate),  # Changed from --learning-rate
        "--gamma", "0.995",
        "--batch-size", "64",
        "--buffer-size", "100000",
        "--reward", "sharpe",  # Changed from --reward-type
        "--save-freq", "10"
    ]

    if sample_data:
        cmd.append("--sample-data")

    try:
        with console.status("[cyan]Training in progress...", spinner="dots"):
            # Run training as subprocess with clean Python environment
            result = subprocess.run(
                cmd,
                cwd=server_dir,
                capture_output=True,
                text=True
            )

        if result.returncode == 0:
            console.print(f"\n[green]✓ Training completed![/green]")
            console.print(f"[dim]Model saved to: checkpoints/[/dim]\n")

            # Show training output
            if result.stdout:
                console.print("[dim]Training output:[/dim]")
                console.print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        else:
            console.print(f"\n[red]✗ Training failed![/red]\n")
            if result.stderr:
                console.print("[red]Error:[/red]")
                console.print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            sys.exit(1)

    except Exception as e:
        console.print(f"\n[red]✗ Training failed: {e}[/red]\n")
        sys.exit(1)

# ============================================================================
# Backtesting
# ============================================================================

def backtest(model_path: str, sample_data: bool = False):
    """Run backtest with beautiful results display"""
    console.print(f"\n[bold cyan]📈 Running Backtest[/bold cyan]\n")

    console.print(f"[dim]Model: {model_path}[/dim]\n")

    # Build command to run backtest as subprocess
    cmd = [
        sys.executable,
        "test_backtest.py",
        "--model", model_path
    ]

    if sample_data:
        cmd.append("--sample-data")

    try:
        with console.status("[cyan]Running backtest...", spinner="dots"):
            # Run backtest as subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

        if result.returncode == 0:
            # Display results from output
            console.print("\n[bold green]Backtest Results[/bold green]\n")

            # Show backtest output
            if result.stdout:
                console.print(result.stdout)
        else:
            console.print(f"\n[red]✗ Backtest failed![/red]\n")
            if result.stderr:
                console.print("[red]Error:[/red]")
                console.print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            sys.exit(1)

    except Exception as e:
        console.print(f"\n[red]✗ Backtest failed: {e}[/red]\n")
        sys.exit(1)

# ============================================================================
# Models Management
# ============================================================================

def list_models():
    """List available trained models"""
    console.print("\n[bold cyan]🤖 Available Models[/bold cyan]\n")

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

    # Train command
    train_parser = subparsers.add_parser("train", help="Train RL model")
    train_parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    train_parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    train_parser.add_argument("--model", default="d3qn", help="Model type (d3qn)")
    train_parser.add_argument("--lr", type=float, default=0.00025, help="Learning rate")
    train_parser.add_argument("--sample-data", action="store_true", help="Use sample data")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--model", required=True, help="Path to model file")
    backtest_parser.add_argument("--sample-data", action="store_true", help="Use sample data")

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
        else:
            console.print("[yellow]Use: cryppo data download/list[/yellow]\n")

    elif args.command == "train":
        train(args.symbol, args.episodes, args.model, args.lr, args.sample_data)

    elif args.command == "backtest":
        backtest(args.model, args.sample_data)

    elif args.command == "models":
        list_models()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
