#!/usr/bin/env python3
"""
Debug script to understand why all trades have same PnL
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "server"))

from src.policy.trading_env import TradingEnv

def debug_position_behavior():
    """Debug position open/close behavior"""
    print("=" * 80)
    print("DEBUG: WHY SAME PNL EVERY TRADE?")
    print("=" * 80)

    # Create realistic BTC price data
    np.random.seed(42)
    n_samples = 100
    base_price = 27500.0

    # Create price data with some volatility
    data = []
    for i in range(n_samples):
        open_price = base_price + np.random.randn() * 300
        high_price = open_price + abs(np.random.randn() * 200)
        low_price = open_price - abs(np.random.randn() * 200)
        close_price = np.random.uniform(low_price, high_price)
        volume = 1000.0
        data.append([i, open_price, high_price, low_price, close_price, volume])

    data = np.array(data)

    # Create environment
    env = TradingEnv(
        data=data,
        initial_balance=10000.0,
        max_leverage=1.5,  # Should be 1.5x!
        window_size=20,
        streaming_mode=False
    )

    print(f"\n[ENVIRONMENT CONFIG]")
    print(f"  Max Leverage: {env.max_leverage}x")
    print(f"  TP Multiplier: {env.atr_tp_multiplier}x ATR")
    print(f"  SL Multiplier: {env.atr_sl_multiplier}x ATR")
    print(f"  Trailing Multiplier: {env.trailing_distance_multiplier}x ATR")

    # Reset and open a SHORT position
    state, _ = env.reset()
    env.current_step = 50  # Move to middle

    # Open SHORT
    print(f"\n" + "=" * 80)
    print(f"OPENING SHORT POSITION")
    print(f"=" * 80)

    state, reward, terminated, truncated, info = env.step(2)  # OPEN_SHORT

    if env.position == 'short':
        entry_price = env.entry_price
        position_size = env.position_size
        leverage = env.current_leverage
        tp = env.take_profit
        sl = env.stop_loss
        trailing_sl = env.trailing_stop_loss
        atr = env._calculate_atr()

        print(f"\n[POSITION OPENED]")
        print(f"  Entry Price: ${entry_price:,.2f}")
        print(f"  Position Size: {position_size:.4f} BTC")
        print(f"  Leverage: {leverage:.1f}x")
        print(f"  Notional: ${entry_price * position_size:,.2f}")
        print(f"  ATR: ${atr:.2f}")
        print(f"  TP: ${tp:,.2f} ({abs((tp-entry_price)/entry_price*100):.2f}%)")
        print(f"  SL: ${sl:,.2f} ({abs((sl-entry_price)/entry_price*100):.2f}%)")
        print(f"  Trailing SL: ${trailing_sl:,.2f}")

        # Calculate expected PnL at TP and SL
        tp_distance = abs(entry_price - tp)
        sl_distance = abs(entry_price - sl)

        tp_pnl = tp_distance * position_size
        sl_pnl = sl_distance * position_size

        print(f"\n[EXPECTED PNL]")
        print(f"  At TP: +${tp_pnl:,.2f} (before fee)")
        print(f"  At SL: -${sl_pnl:,.2f} (before fee)")
        print(f"  Risk:Reward Ratio: 1:{tp_pnl/sl_pnl:.2f}")

        # Simulate next few steps
        print(f"\n" + "=" * 80)
        print(f"SIMULATING NEXT STEPS")
        print(f"=" * 80)

        for i in range(10):
            if env.position is None:
                break

            step_idx = env.current_step
            prices = env.data[step_idx]
            open_p, high_p, low_p, close_p = prices[1:5]

            prev_bars = env.bars_in_position
            prev_trailing = env.trailing_stop_loss

            # Take action (HOLD)
            state, reward, terminated, truncated, info = env.step(0)

            print(f"\n  Step {i+1}:")
            print(f"    Open: ${open_p:,.2f} | High: ${high_p:,.2f} | Low: ${low_p:,.2f} | Close: ${close_p:,.2f}")
            print(f"    Bars in position: {prev_bars} → {env.bars_in_position}")
            print(f"    Trailing SL: ${prev_trailing:,.2f} → ${env.trailing_stop_loss:,.2f}")

            # Check SL/TP conditions manually
            if env.position == 'short':
                sl_hit = high_p >= sl
                trailing_hit = prev_bars > 1 and high_p >= prev_trailing
                tp_hit = low_p <= tp

                print(f"    SL check: high ({high_p:.2f}) >= SL ({sl:.2f}) = {sl_hit}")
                print(f"    Trailing check: bars>{prev_bars} AND high ({high_p:.2f}) >= Trailing ({prev_trailing:.2f}) = {trailing_hit}")
                print(f"    TP check: low ({low_p:.2f}) <= TP ({tp:.2f}) = {tp_hit}")

            if env.position is None:
                print(f"\n    ❌ POSITION CLOSED!")
                print(f"    Final Balance: ${env.balance:,.2f}")
                print(f"    Total PnL: ${env.total_pnl:,.2f}")
                break

            print(f"    Position still open, balance: ${env.balance:,.2f}")

    print(f"\n" + "=" * 80)
    print(f"ANALYSIS")
    print(f"=" * 80)

    if env.max_leverage != 1.5:
        print(f"\n❌ PROBLEM 1: Leverage is {env.max_leverage}x, should be 1.5x!")
    else:
        print(f"\n✅ Leverage is correct: {env.max_leverage}x")

    print(f"\n[OBSERVATIONS]")
    print(f"  - Check if Trailing Stop updates in first bar")
    print(f"  - Check if SL/Trailing distance is too tight")
    print(f"  - Check if bars_in_position counter works")
    print(f"  - Check exit price vs SL/Trailing level")

if __name__ == "__main__":
    debug_position_behavior()
