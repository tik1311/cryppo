#!/usr/bin/env python3
"""
Analyze trading behavior to find why high win rate but negative returns
"""

import numpy as np
import sys
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

from src.policy.trading_env import TradingEnv

def analyze_trading_behavior():
    """Simulate trading and analyze behavior"""
    print("=" * 80)
    print("TRADING BEHAVIOR ANALYSIS")
    print("=" * 80)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 500
    base_price = 92000.0

    prices = []
    for i in range(n_samples):
        # Trending + noise
        trend = i * 10  # Uptrend
        noise = np.random.randn() * base_price * 0.01
        price = base_price + trend + noise
        prices.append(price)

    prices = np.array(prices)

    # Create OHLCV
    data = np.zeros((n_samples, 6))
    for i in range(n_samples):
        open_price = prices[i]
        high_price = open_price * (1 + abs(np.random.randn() * 0.005))
        low_price = open_price * (1 - abs(np.random.randn() * 0.005))
        close_price = np.random.uniform(low_price, high_price)
        volume = 1000.0
        data[i] = [i, open_price, high_price, low_price, close_price, volume]

    # Create environment
    env = TradingEnv(
        data=data,
        initial_balance=10000.0,
        window_size=20,
        streaming_mode=False
    )

    # Track detailed statistics
    wins = []
    losses = []
    close_reasons = {'TP': 0, 'SL': 0, 'TRAIL': 0, 'MANUAL_CLOSE': 0}
    early_closes = []  # Track if closed before TP/SL

    print("\n[INFO] Starting simulation...")
    print(f"  Initial Balance: ${env.initial_balance:,.2f}")
    print(f"  Leverage: {env.max_leverage}x")
    print(f"  TP Multiplier: {env.atr_tp_multiplier}x ATR")
    print(f"  SL Multiplier: {env.atr_sl_multiplier}x ATR")

    state, _ = env.reset()
    total_trades = 0
    positions_opened = 0

    # Simulate 200 steps
    for step in range(200):
        # Simple strategy: alternate between SHORT and CLOSE
        if env.position is None:
            action = 2  # OPEN_SHORT
            positions_opened += 1
        else:
            # Randomly close or hold
            if np.random.random() < 0.3:  # 30% chance to close manually
                action = 3  # CLOSE
            else:
                action = 0  # HOLD

        state, reward, terminated, truncated, info = env.step(action)

        # Track if position closed
        if env.position is None and positions_opened > total_trades:
            total_trades += 1

            # Get last trade info
            if env.total_trades > 0:
                last_pnl = env.total_pnl - sum(wins) - sum(losses)

                if last_pnl > 0:
                    wins.append(last_pnl)
                else:
                    losses.append(abs(last_pnl))

        if terminated or truncated:
            break

    # Calculate statistics
    total_wins = len(wins)
    total_losses = len(losses)
    total = total_wins + total_losses

    if total == 0:
        print("\n[ERROR] No trades executed")
        return

    win_rate = (total_wins / total) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\n[STATISTICS]")
    print(f"  Total Trades: {total}")
    print(f"  Wins: {total_wins} ({win_rate:.1f}%)")
    print(f"  Losses: {total_losses} ({100-win_rate:.1f}%)")
    print(f"  Final Balance: ${env.balance:,.2f}")
    print(f"  Total PnL: ${env.total_pnl:,.2f}")
    print(f"  Return: {(env.balance/env.initial_balance - 1)*100:.2f}%")

    print(f"\n[TRADE QUALITY]")
    print(f"  Average Win: ${avg_win:.2f}")
    print(f"  Average Loss: ${avg_loss:.2f}")
    if avg_loss > 0:
        print(f"  Win/Loss Ratio: {avg_win/avg_loss:.2f}x")
        expected_value = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
        print(f"  Expected Value per Trade: ${expected_value:.2f}")

    # Analyze the problem
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if win_rate > 90 and env.balance < env.initial_balance:
        print("\n🔴 PROBLEM DETECTED: High Win Rate + Negative Returns")
        print("\nThis is the classic 'death by 1000 cuts' pattern:")
        print("  ✓ Many small wins")
        print("  ✗ Few large losses")
        print("  = Net negative despite high win rate")

        if avg_loss > avg_win * 5:
            print(f"\n🔴 CRITICAL: Average loss ({avg_loss:.2f}) is {avg_loss/avg_win:.1f}x larger than average win ({avg_win:.2f})")
            print("\nROOT CAUSE:")
            print("  - Agent closes positions early when profitable (small wins)")
            print("  - Agent doesn't close when losing (waits for SL = big loss)")
            print("  - TP/SL might still be too tight despite ATR fix")

    # Check current TP/SL settings
    print(f"\n[CURRENT SETTINGS]")
    print(f"  Leverage: {env.max_leverage}x")
    print(f"  ATR Period: {env.atr_period}")
    print(f"  TP Multiplier: {env.atr_tp_multiplier}x ATR")
    print(f"  SL Multiplier: {env.atr_sl_multiplier}x ATR")
    print(f"  Trailing Multiplier: {env.trailing_distance_multiplier}x ATR")

    return {
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'final_balance': env.balance,
        'total_pnl': env.total_pnl
    }

if __name__ == "__main__":
    analyze_trading_behavior()
