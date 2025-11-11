"""
Backtesting Engine
Evaluates trained agent on historical data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import json


class BacktestEngine:
    """
    Backtest a trading agent on historical data
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.0004,
        max_leverage: float = 3.0
    ):
        """
        Initialize Backtest Engine

        Args:
            initial_balance: Starting capital
            fee_rate: Trading fee rate
            max_leverage: Maximum leverage
        """
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.max_leverage = max_leverage

        self.reset()

    def reset(self):
        """Reset backtest state"""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = None
        self.entry_price = 0.0
        self.position_size = 0.0

        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.timestamps = []

    def execute_trade(self, timestamp, price: float, action: str):
        """
        Execute a trade

        Args:
            timestamp: Trade timestamp
            price: Current price
            action: 'buy', 'sell', or 'hold'
        """
        if action == 'buy':
            if self.position is None:
                self._open_long(timestamp, price)
            elif self.position == 'short':
                self._close_position(timestamp, price)
                self._open_long(timestamp, price)

        elif action == 'sell':
            if self.position == 'long':
                self._close_position(timestamp, price)

        # Update equity
        if self.position:
            self.equity = self.balance + self._calculate_pnl(price)
        else:
            self.equity = self.balance

        self.equity_curve.append(self.equity)
        self.timestamps.append(timestamp)

    def _open_long(self, timestamp, price: float):
        """Open long position"""
        available_capital = self.balance * self.max_leverage
        fee = available_capital * self.fee_rate
        self.position_size = (available_capital - fee) / price

        self.position = 'long'
        self.entry_price = price
        self.balance -= fee

        self.trades.append({
            'timestamp': timestamp,
            'type': 'open',
            'position': 'long',
            'price': price,
            'size': self.position_size,
            'fee': fee,
            'balance': self.balance
        })

    def _close_position(self, timestamp, price: float):
        """Close current position"""
        if not self.position:
            return

        # Calculate PnL
        if self.position == 'long':
            pnl = (price - self.entry_price) * self.position_size
        else:
            pnl = (self.entry_price - price) * self.position_size

        # Deduct fee
        fee = price * self.position_size * self.fee_rate
        pnl -= fee

        # Update balance
        self.balance += pnl

        self.trades.append({
            'timestamp': timestamp,
            'type': 'close',
            'position': self.position,
            'entry_price': self.entry_price,
            'exit_price': price,
            'size': self.position_size,
            'pnl': pnl,
            'pnl_pct': (pnl / (self.entry_price * self.position_size)) * 100,
            'fee': fee,
            'balance': self.balance
        })

        # Reset position
        self.position = None
        self.entry_price = 0.0
        self.position_size = 0.0

    def _calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL"""
        if not self.position:
            return 0.0

        if self.position == 'long':
            return (current_price - self.entry_price) * self.position_size
        else:
            return (self.entry_price - current_price) * self.position_size

    def get_results(self) -> dict:
        """
        Calculate backtest results and metrics

        Returns:
            Dictionary with performance metrics
        """
        # Filter closed trades only
        closed_trades = [t for t in self.trades if t['type'] == 'close']

        if len(closed_trades) == 0:
            return {
                'error': 'No trades executed'
            }

        # Calculate metrics
        total_pnl = sum(t['pnl'] for t in closed_trades)
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]

        # Returns
        total_return_pct = (self.balance - self.initial_balance) / self.initial_balance * 100

        # Win rate
        win_rate = len(winning_trades) / len(closed_trades) * 100

        # Average win/loss
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Max drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown_pct = abs(drawdown.min()) * 100

        # Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252 * 24)  # 1H data
        else:
            sharpe_ratio = 0

        # Sortino ratio
        if len(self.equity_curve) > 1:
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
            sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252 * 24)
        else:
            sortino_ratio = 0

        results = {
            'summary': {
                'initial_balance': self.initial_balance,
                'final_balance': self.balance,
                'final_equity': self.equity,
                'total_pnl': total_pnl,
                'total_return_pct': total_return_pct
            },
            'trades': {
                'total_trades': len(closed_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate
            },
            'performance': {
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown_pct': max_drawdown_pct,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio
            },
            'equity_curve': self.equity_curve,
            'timestamps': [str(t) for t in self.timestamps],
            'trade_history': closed_trades
        }

        return results

    def print_results(self):
        """Print backtest results"""
        results = self.get_results()

        if 'error' in results:
            print(f"Error: {results['error']}")
            return

        print("\n" + "=" * 60)
        print("Backtest Results")
        print("=" * 60)

        print("\n📊 Summary:")
        print(f"  Initial Balance: ${results['summary']['initial_balance']:,.2f}")
        print(f"  Final Balance:   ${results['summary']['final_balance']:,.2f}")
        print(f"  Total PnL:       ${results['summary']['total_pnl']:,.2f}")
        print(f"  Total Return:    {results['summary']['total_return_pct']:.2f}%")

        print("\n📈 Trades:")
        print(f"  Total Trades:    {results['trades']['total_trades']}")
        print(f"  Winning Trades:  {results['trades']['winning_trades']}")
        print(f"  Losing Trades:   {results['trades']['losing_trades']}")
        print(f"  Win Rate:        {results['trades']['win_rate']:.2f}%")

        print("\n💹 Performance:")
        print(f"  Avg Win:         ${results['performance']['avg_win']:.2f}")
        print(f"  Avg Loss:        ${results['performance']['avg_loss']:.2f}")
        print(f"  Profit Factor:   {results['performance']['profit_factor']:.2f}")
        print(f"  Max Drawdown:    {results['performance']['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio:    {results['performance']['sharpe_ratio']:.4f}")
        print(f"  Sortino Ratio:   {results['performance']['sortino_ratio']:.4f}")

        print("\n" + "=" * 60)

    def save_results(self, filepath: str):
        """Save results to JSON file"""
        results = self.get_results()

        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        results = convert_types(results)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {filepath}")
