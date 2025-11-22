#!/usr/bin/env python3
"""
Real-time Paper Trading (Live Simulation)
Uses Binance API to fetch real-time data and simulate live trading
"""

import sys
import os
import torch
import numpy as np
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from datetime import datetime

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

from src.data.realtime_fetcher import RealtimeFetcher
from src.data.loader import prepare_multi_timeframe_env_data
from src.agent.d3qn_agent import D3QNAgent

console = Console()


class LivePaperTrader:
    """
    Live Paper Trading Simulator

    Key principles:
    1. Actions are taken ONLY when 1h candle closes
    2. TP/SL are checked using 1m real-time data
    3. Entry happens at 1h candle close price
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        main_timeframe: str = "1h",
        detail_timeframe: str = "1m",
        initial_balance: float = 10000.0,
        model_path: str = "checkpoints/d3qn_BTCUSDT_1h.pth",
        window_size: int = 36
    ):
        self.symbol = symbol
        self.main_timeframe = main_timeframe
        self.detail_timeframe = detail_timeframe
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.model_path = model_path
        self.window_size = window_size

        # Position tracking
        self.position = None  # None, 'long', 'short'
        self.entry_price = 0.0
        self.position_size = 0.0
        self.take_profit = 0.0
        self.stop_loss = 0.0
        self.entry_time = None

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        # Initialize fetcher
        self.fetcher = RealtimeFetcher(
            symbol=symbol,
            main_timeframe=main_timeframe,
            detail_timeframe=detail_timeframe,
            window_size=window_size
        )

        # Load agent
        self._load_agent()

    def _load_agent(self):
        """Load trained agent"""
        console.print(f"[cyan]Loading model from {self.model_path}...[/cyan]")

        if not os.path.exists(self.model_path):
            console.print(f"[red]Model not found at {self.model_path}![/red]")
            console.print("[yellow]Please train a model first: python cryppo.py train[/yellow]")
            sys.exit(1)

        # Initialize agent (we'll set dimensions after first data fetch)
        self.agent = None
        console.print(f"[green]✓[/green] Model path verified")

    def _initialize_agent_with_data(self, main_features: int, detail_features: int):
        """Initialize agent with correct dimensions"""
        if self.agent is not None:
            return

        self.agent = D3QNAgent(
            window_size=self.window_size,
            n_main_features=main_features,
            n_detail_features=detail_features,
            detail_len=60,  # 60 × 1m = 1h
            action_dim=3
        )

        self.agent.load(self.model_path)
        self.agent.epsilon = 0.0  # No exploration in live trading
        console.print(f"[green]✓[/green] Agent initialized and loaded")

    def _calculate_tp_sl(self, entry_price: float, position_type: str, atr: float):
        """Calculate TP/SL based on ATR"""
        atr_tp_multiplier = 2.5
        atr_sl_multiplier = 1.5

        if position_type == 'long':
            self.take_profit = entry_price + (atr * atr_tp_multiplier)
            self.stop_loss = entry_price - (atr * atr_sl_multiplier)
        else:  # short
            self.take_profit = entry_price - (atr * atr_tp_multiplier)
            self.stop_loss = entry_price + (atr * atr_sl_multiplier)

    def _open_position(self, position_type: str, price: float, atr: float):
        """Open position"""
        self.position = position_type
        self.entry_price = price
        self.entry_time = datetime.utcnow()

        # Calculate position size (using full balance with 1x leverage)
        fee_rate = 0.0004
        self.position_size = self.balance / price
        fee = price * self.position_size * fee_rate
        self.balance -= fee

        # Calculate TP/SL
        self._calculate_tp_sl(price, position_type, atr)

        console.print(f"\n[green]{'='*60}[/green]")
        console.print(f"[green]OPENED {position_type.upper()} POSITION[/green]")
        console.print(f"[green]{'='*60}[/green]")
        console.print(f"Entry Price: ${price:,.2f}")
        console.print(f"Position Size: {self.position_size:.4f} {self.symbol[:3]}")
        console.print(f"Take Profit: ${self.take_profit:,.2f} (+{((self.take_profit/price - 1) * 100):.2f}%)")
        console.print(f"Stop Loss: ${self.stop_loss:,.2f} ({((self.stop_loss/price - 1) * 100):.2f}%)")
        console.print(f"Fee Paid: ${fee:.2f}\n")

    def _close_position(self, price: float, reason: str):
        """Close position"""
        if self.position is None:
            return

        # Calculate PnL
        if self.position == 'long':
            pnl = (price - self.entry_price) * self.position_size
        else:  # short
            pnl = (self.entry_price - price) * self.position_size

        # Fees
        fee_rate = 0.0004
        fee = price * self.position_size * fee_rate
        pnl -= fee

        # Update balance
        self.balance += pnl
        self.total_pnl += pnl

        # Update stats
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        # Calculate holding time
        hold_time = (datetime.utcnow() - self.entry_time).total_seconds() / 3600  # hours

        console.print(f"\n[yellow]{'='*60}[/yellow]")
        console.print(f"[yellow]CLOSED {self.position.upper()} POSITION ({reason.upper()})[/yellow]")
        console.print(f"[yellow]{'='*60}[/yellow]")
        console.print(f"Entry: ${self.entry_price:,.2f} | Exit: ${price:,.2f}")
        console.print(f"PnL: ${pnl:,.2f} ({(pnl/self.balance * 100):.2f}%)")
        console.print(f"Fee Paid: ${fee:.2f}")
        console.print(f"Hold Time: {hold_time:.1f}h")
        console.print(f"Balance: ${self.balance:,.2f}")
        console.print(f"Total Trades: {self.total_trades} | Win Rate: {(self.winning_trades/max(1,self.total_trades)*100):.1f}%\n")

        # Reset position
        self.position = None
        self.entry_price = 0.0
        self.position_size = 0.0

    def _check_tp_sl(self, df_detail):
        """Check if TP/SL was hit using detail timeframe"""
        if self.position is None:
            return False

        # Get high/low from detail data (1m candles)
        detail_high = df_detail['high'].max()
        detail_low = df_detail['low'].min()
        current_price = df_detail['close'].iloc[-1]

        if self.position == 'long':
            if detail_low <= self.stop_loss:
                self._close_position(self.stop_loss, 'sl')
                return True
            elif detail_high >= self.take_profit:
                self._close_position(self.take_profit, 'tp')
                return True
        else:  # short
            if detail_high >= self.stop_loss:
                self._close_position(self.stop_loss, 'sl')
                return True
            elif detail_low <= self.take_profit:
                self._close_position(self.take_profit, 'tp')
                return True

        return False

    def _get_current_state(self, df_main, df_detail):
        """Prepare current state for agent"""
        # Use last window_size candles
        df_main_window = df_main.tail(self.window_size)
        df_detail_window = df_detail.tail(self.window_size * 60)  # 60 × 1m per 1h

        # Extract features
        main_cols = ['open', 'high', 'low', 'close', 'volume',
                     'returns', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
                     'atr', 'stoch_k', 'stoch_d', 'volume_ratio', 'volatility',
                     'sma_7', 'sma_25', 'sma_99', 'ema_7', 'ema_25',
                     'trend_direction', 'trend_strength', 'swing_high', 'swing_low',
                     'distance_to_swing_high', 'distance_to_swing_low', 'dow_structure']

        detail_cols = ['open', 'high', 'low', 'close', 'volume', 'atr']

        main_data = df_main_window[[c for c in main_cols if c in df_main_window.columns]].values
        detail_data = df_detail_window[[c for c in detail_cols if c in df_detail_window.columns]].values

        # Reshape detail data: [window_size, 60, features]
        detail_data = detail_data.reshape(self.window_size, 60, -1)

        # Add position info (simplified)
        has_position = 1.0 if self.position else 0.0
        pos_info = np.array([[0, 0, 0, 0, 0, 0]] * self.window_size, dtype=np.float32)
        pos_info[:, 0] = 1.0 - has_position  # no position
        pos_info[:, 1] = 1.0 if self.position == 'long' else 0.0
        pos_info[:, 2] = 1.0 if self.position == 'short' else 0.0

        main_data = np.concatenate([main_data, pos_info], axis=1).astype(np.float32)

        return {
            'main': main_data,
            'detail': detail_data.astype(np.float32)
        }

    def run(self):
        """Run live paper trading"""
        console.print(Panel(f"🚀 Starting Live Paper Trading - {self.symbol}", style="bold green"))
        console.print(f"[cyan]Initial Balance: ${self.initial_balance:,.2f}[/cyan]")
        console.print(f"[cyan]Main Timeframe: {self.main_timeframe}[/cyan]")
        console.print(f"[cyan]Detail Timeframe: {self.detail_timeframe}[/cyan]")
        console.print(f"\n[yellow]⚠️  Important: Actions are taken ONLY when {self.main_timeframe} candle closes[/yellow]")
        console.print(f"[yellow]⚠️  TP/SL are checked using {self.detail_timeframe} real-time data[/yellow]\n")

        try:
            step = 0
            while True:
                step += 1

                # Wait for 1h candle to close
                console.print(f"\n[dim]Step {step}:[/dim]")
                candle_close_time = self.fetcher.wait_for_candle_close(self.main_timeframe)
                console.print(f"[green]✓[/green] {self.main_timeframe} candle closed at {candle_close_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

                # Fetch latest data
                console.print(f"[cyan]Fetching latest data...[/cyan]")
                df_main, df_detail = self.fetcher.get_latest_data()

                if df_main is None or df_detail is None:
                    console.print("[red]Failed to fetch data. Retrying...[/red]")
                    time.sleep(5)
                    continue

                # Initialize agent on first run
                if self.agent is None:
                    n_main_features = len([c for c in df_main.columns if c not in ['timestamp']])
                    n_detail_features = len([c for c in df_detail.columns if c not in ['timestamp']])
                    self._initialize_agent_with_data(n_main_features + 6, n_detail_features)

                # Check TP/SL (using detail timeframe)
                if self.position is not None:
                    self._check_tp_sl(df_detail)

                # Get current state
                try:
                    state = self._get_current_state(df_main, df_detail)
                except Exception as e:
                    console.print(f"[red]Error preparing state: {e}[/red]")
                    continue

                # Make decision (ONLY if no position or can open new one)
                if self.position is None:
                    action = self.agent.select_action(state, training=False)
                    current_price = df_main['close'].iloc[-1]
                    current_atr = df_main['atr'].iloc[-1]

                    if action == 1:  # OPEN LONG
                        self._open_position('long', current_price, current_atr)
                    elif action == 2:  # OPEN SHORT
                        self._open_position('short', current_price, current_atr)
                    else:  # HOLD
                        console.print(f"[dim]Decision: HOLD | Price: ${current_price:,.2f}[/dim]")

                # Display current status
                win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
                total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100

                table = Table(title="Current Status")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Balance", f"${self.balance:,.2f}")
                table.add_row("Total Return", f"{total_return:.2f}%")
                table.add_row("Position", self.position.upper() if self.position else "NONE")
                if self.position:
                    current_price = df_main['close'].iloc[-1]
                    unrealized_pnl = (current_price - self.entry_price) * self.position_size if self.position == 'long' else (self.entry_price - current_price) * self.position_size
                    table.add_row("Unrealized PnL", f"${unrealized_pnl:.2f}")
                table.add_row("Total Trades", str(self.total_trades))
                table.add_row("Win Rate", f"{win_rate:.2f}%")

                console.print(table)

        except KeyboardInterrupt:
            console.print("\n[yellow]Simulation stopped by user.[/yellow]")

            # Close any open position
            if self.position is not None:
                df_main, df_detail = self.fetcher.get_latest_data()
                if df_main is not None:
                    current_price = df_main['close'].iloc[-1]
                    self._close_position(current_price, 'manual')

            # Final stats
            win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
            total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100

            console.print(f"\n[bold]Final Results:[/bold]")
            console.print(f"Initial Balance: ${self.initial_balance:,.2f}")
            console.print(f"Final Balance: ${self.balance:,.2f}")
            console.print(f"Total Return: {total_return:.2f}%")
            console.print(f"Total Trades: {self.total_trades}")
            console.print(f"Win Rate: {win_rate:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live Paper Trading")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    parser.add_argument("--model", default="checkpoints/d3qn_BTCUSDT_1h.pth", help="Model path")

    args = parser.parse_args()

    trader = LivePaperTrader(
        symbol=args.symbol,
        initial_balance=args.balance,
        model_path=args.model
    )

    trader.run()
