#!/usr/bin/env python3
"""
Interactive Launcher for CRYPPO (Multi-Timeframe)
"""

import sys
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.align import Align
from rich.text import Text
from datetime import datetime, timedelta

console = Console()

def run_command(args):
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

def main():
    while True:
        console.clear()
        
        title = Text("CRYPPO - Deep RL Trading System", style="bold magenta")
        subtitle = Text("Multi-Timeframe Edition (1h + 1m)", style="dim white")
        
        console.print(Panel(Align.center(title + "\n" + subtitle), border_style="cyan"))
        
        console.print("\n[bold cyan]Main Menu[/bold cyan]")
        console.print("1. [green]System Check[/green]")
        console.print("2. [yellow]Download Data[/yellow] (Main + Detail)")
        console.print("3. [blue]Train Model[/blue]")
        console.print("4. [magenta]Backtest Model[/magenta]")
        console.print("5. [red]Paper Trade (Simulation)[/red]")
        console.print("0. Exit")
        
        choice = Prompt.ask("\nSelect option", default="1")
        
        if choice == "1":
            run_command(["check"])
            Prompt.ask("\nPress Enter to continue...")
            
        elif choice == "2":
            console.print("\n[bold yellow]Download Market Data[/bold yellow]")
            symbol = Prompt.ask("Symbol", default="BTCUSDT")
            start, end = get_date_range()
            
            if Confirm.ask(f"Download {symbol} (1h & 1m) from {start} to {end}?"):
                # Download 1h
                run_command(["data", "download", "--symbol", symbol, "--timeframe", "1h", "--start", start, "--end", end])
                # Download 1m
                run_command(["data", "download", "--symbol", symbol, "--timeframe", "1m", "--start", start, "--end", end])
            Prompt.ask("\nPress Enter to continue...")

        elif choice == "3":
            console.print("\n[bold blue]Train Model[/bold blue]")
            symbol = Prompt.ask("Symbol", default="BTCUSDT")
            episodes = Prompt.ask("Episodes", default="10")
            
            use_date_range = Confirm.ask("Filter by date range?", default=False)
            args = ["train", "--symbol", symbol, "--episodes", episodes]
            
            if use_date_range:
                start, end = get_date_range()
                args.extend(["--start", start, "--end", end])
                
            run_command(args)
            Prompt.ask("\nPress Enter to continue...")

        elif choice == "4":
            console.print("\n[bold magenta]Backtest Model[/bold magenta]")
            symbol = Prompt.ask("Symbol", default="BTCUSDT")
            
            use_date_range = Confirm.ask("Filter by date range?", default=False)
            args = ["backtest", "--symbol", symbol]
            
            if use_date_range:
                start, end = get_date_range()
                args.extend(["--start", start, "--end", end])
                
            run_command(args)
            Prompt.ask("\nPress Enter to continue...")
            
        elif choice == "5":
            console.print("\n[bold red]Paper Trade Simulation[/bold red]")
            console.print("[dim]Simulating live trading using recent data...[/dim]")
            symbol = Prompt.ask("Symbol", default="BTCUSDT")
            
            use_date_range = Confirm.ask("Filter by date range?", default=False)
            args = ["paper", "--symbol", symbol]
            
            if use_date_range:
                start, end = get_date_range()
                args.extend(["--start", start, "--end", end])
            
            run_command(args)
            Prompt.ask("\nPress Enter to continue...")

        elif choice == "0":
            console.print("[bold]Goodbye![/bold]")
            break

if __name__ == "__main__":
    main()
