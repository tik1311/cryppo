#!/usr/bin/env python3
"""
Interactive launcher for CRYPPO.
Provides a friendly Rich-based UI so everything can be executed by running:
    python main.py
"""

import sys
import subprocess
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich import box

console = Console()


BANNER = """
  ██████╗██████╗ ██╗   ██╗██████╗ ██████╗  ██████╗
 ██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗██╔══██╗██╔═══██╗
 ██║     ██████╔╝ ╚████╔╝ ██████╔╝██████╔╝██║   ██║
 ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝ ██╔═══╝ ██║   ██║
 ╚██████╗██║  ██║   ██║   ██║     ██║     ╚██████╔╝
  ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚═╝      ╚═════╝

        Interactive CLI for CRYPPO
"""


def run_command(args: List[str]) -> None:
    """Execute cryppo.py with the supplied arguments."""
    cmd = [sys.executable, "cryppo.py"] + args
    console.print(
        Panel(
            Text(" ".join(cmd), style="bold white"),
            title="Executing",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        console.print("[yellow]Command interrupted by user[/yellow]")


def show_menu() -> None:
    """Display available actions."""
    table = Table(title="Choose an action", box=box.SIMPLE_HEAVY)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Task", style="bold white")
    table.add_column("Description", style="dim")
    table.add_row("1", "System Check", "Validate environment, GPU, dependencies")
    table.add_row("2", "Download Data", "Fetch market data via CCXT")
    table.add_row("3", "List Data", "Show available files in data/")
    table.add_row("4", "Prepare Data", "Create normalized NPZ splits")
    table.add_row("5", "Train Model", "Train D3QN/PPO/SAC/TD3")
    table.add_row("6", "Backtest Model", "Evaluate a saved checkpoint")
    table.add_row("7", "Paper Trading", "Live-style CCXT simulation")
    table.add_row("8", "List Models", "Show files inside checkpoints/")
    table.add_row("0", "Exit", "Quit the UI")
    console.print(table)


def handle_system_check():
    run_command(["check"])


def handle_download_data():
    symbol = Prompt.ask("Symbol", default="BTCUSDT").upper()
    timeframe = Prompt.ask("Timeframe", default="1h")
    start = Prompt.ask("Start date (YYYY-MM-DD)", default="2023-01-01")
    end = Prompt.ask("End date (YYYY-MM-DD)", default="2024-12-31")
    run_command(
        [
            "data",
            "download",
            "--symbol",
            symbol,
            "--start",
            start,
            "--end",
            end,
            "--timeframe",
            timeframe,
        ]
    )


def handle_list_data():
    run_command(["data", "list"])


def handle_prepare_data():
    symbol = Prompt.ask("Symbol", default="BTCUSDT").upper()
    timeframe = Prompt.ask("Timeframe", default="1h")
    window = Prompt.ask("Window size", default="72")
    output = Prompt.ask("Output directory", default="data/processed")
    cmd = [
        "data",
        "prepare",
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--window",
        window,
        "--output",
        output,
    ]
    run_command(cmd)


def handle_train():
    model = Prompt.ask(
        "Model (d3qn/ppo/sac/td3)",
        default="d3qn",
        choices=["d3qn", "ppo", "sac", "td3"],
    )
    symbol = Prompt.ask("Symbol", default="BTCUSDT").upper()
    timeframe = Prompt.ask("Timeframe", default="1h")
    reward = Prompt.ask(
        "Reward (profit/return/sharpe/sortino)",
        default="sharpe",
        choices=["profit", "return", "sharpe", "sortino"],
    )

    args = [
        "train",
        "--model",
        model,
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--reward",
        reward,
    ]

    if model == "d3qn":
        episodes = Prompt.ask("Episodes", default="10")
        args.extend(["--episodes", episodes])
    else:
        timesteps = Prompt.ask("Timesteps", default="50000")
        args.extend(["--timesteps", timesteps])

    run_command(args)


def handle_backtest():
    model_path = Prompt.ask(
        "Path to model",
        default="checkpoints/final_agent.pth",
    )
    model_type = Prompt.ask(
        "Model type (d3qn/ppo/sac/td3)",
        default="d3qn",
        choices=["d3qn", "ppo", "sac", "td3"],
    )
    symbol = Prompt.ask("Symbol", default="BTCUSDT").upper()
    timeframe = Prompt.ask("Timeframe", default="1h")
    args = [
        "backtest",
        "--model",
        model_path,
        "--model-type",
        model_type,
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
    ]
    run_command(args)


def handle_paper():
    model_path = Prompt.ask(
        "Path to model",
        default="checkpoints/final_agent.pth",
    )
    model_type = Prompt.ask(
        "Model type (d3qn/ppo/sac/td3)",
        default="d3qn",
        choices=["d3qn", "ppo", "sac", "td3"],
    )
    symbol = Prompt.ask("Symbol (e.g. BTC/USDT)", default="BTC/USDT").upper()
    timeframe = Prompt.ask("Timeframe", default="1m")
    steps = Prompt.ask("Steps to simulate", default="120")
    interval = Prompt.ask("Seconds between updates", default="2")
    args = [
        "paper",
        "--model",
        model_path,
        "--model-type",
        model_type,
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--steps",
        steps,
        "--interval",
        interval,
    ]
    run_command(args)


def handle_list_models():
    run_command(["models"])


HANDLERS = {
    "1": handle_system_check,
    "2": handle_download_data,
    "3": handle_list_data,
    "4": handle_prepare_data,
    "5": handle_train,
    "6": handle_backtest,
    "7": handle_paper,
    "8": handle_list_models,
}


def main():
    console.clear()
    console.print(Panel(Text(BANNER, justify="center", style="bold cyan"), border_style="bright_cyan", box=box.DOUBLE))

    while True:
        show_menu()
        choice = Prompt.ask("Select option", default="1")
        if choice == "0":
            console.print("[bold green]Goodbye![/bold green]")
            break
        handler = HANDLERS.get(choice)
        if handler:
            handler()
        else:
            console.print("[red]Invalid choice[/red]")
            continue

        if not Confirm.ask("Run another task?", default=True):
            console.print("[bold green]Goodbye![/bold green]")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Exited by user[/yellow]")
