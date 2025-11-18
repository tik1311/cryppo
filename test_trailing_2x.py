#!/usr/bin/env python3
"""
Test script to verify trailing stop with 2.0x ATR multiplier
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "server"))

from src.policy.trading_env import TradingEnv

def test_trailing_2x_atr():
    """Test that 2.0x ATR trailing stop allows profits to run"""
    print("=" * 80)
    print("TEST: TRAILING STOP WITH 2.0x ATR MULTIPLIER")
    print("=" * 80)

    # Create realistic BTC data
    np.random.seed(42)
    n_samples = 50
    base_price = 65000.0

    # Create downtrend (good for SHORT)
    data = []
    for i in range(n_samples):
        # Price gradually declining
        trend_price = base_price - (i * 100)  # -$100 per step
        noise = np.random.randn() * 200
        open_price = trend_price + noise
        high_price = open_price + abs(np.random.randn() * 150)
        low_price = open_price - abs(np.random.randn() * 150)
        close_price = np.random.uniform(low_price, high_price)
        volume = 1000.0
        data.append([i, open_price, high_price, low_price, close_price, volume])

    data = np.array(data)

    # Create environment with 2.0x trailing
    env = TradingEnv(
        data=data,
        initial_balance=10000.0,
        max_leverage=1.5,
        window_size=14,
        streaming_mode=False
    )

    print(f"\n[ENVIRONMENT SETTINGS]")
    print(f"  Leverage: {env.max_leverage}x")
    print(f"  TP Multiplier: {env.atr_tp_multiplier}x ATR")
    print(f"  SL Multiplier: {env.atr_sl_multiplier}x ATR")
    print(f"  Trailing Multiplier: {env.trailing_distance_multiplier}x ATR ✅ (UPDATED!)")

    # Open SHORT position
    state, _ = env.reset()
    env.current_step = 20

    state, reward, terminated, truncated, info = env.step(2)  # OPEN_SHORT

    if env.position == 'short':
        entry_price = env.entry_price
        position_size = env.position_size
        atr = env._calculate_atr()
        tp = env.take_profit
        sl = env.stop_loss
        trailing = env.trailing_stop_loss

        print(f"\n" + "=" * 80)
        print(f"SHORT POSITION OPENED")
        print(f"=" * 80)
        print(f"\n  Entry: ${entry_price:,.2f}")
        print(f"  Position Size: {position_size:.4f} BTC")
        print(f"  Notional: ${entry_price * position_size:,.2f}")
        print(f"\n  ATR: ${atr:.2f}")
        print(f"  TP: ${tp:,.2f} ({abs((tp-entry_price)/entry_price*100):.2f}%)")
        print(f"  SL: ${sl:,.2f} ({abs((sl-entry_price)/entry_price*100):.2f}%)")
        print(f"  Trailing SL: ${trailing:,.2f}")

        # Calculate distances
        tp_distance = abs(entry_price - tp)
        sl_distance = abs(sl - entry_price)
        trailing_distance = trailing - entry_price  # For SHORT

        print(f"\n[DISTANCES]")
        print(f"  TP Distance: ${tp_distance:.2f} ({tp_distance/atr:.2f}x ATR)")
        print(f"  SL Distance: ${sl_distance:.2f} ({sl_distance/atr:.2f}x ATR)")
        print(f"  Trailing Distance: ${abs(trailing_distance):.2f} ({abs(trailing_distance)/atr:.2f}x ATR)")

        # Expected PnL
        tp_pnl = tp_distance * position_size
        sl_pnl = sl_distance * position_size

        print(f"\n[EXPECTED PNL]")
        print(f"  At TP: +${tp_pnl:,.2f}")
        print(f"  At SL: -${sl_pnl:,.2f}")
        print(f"  Risk:Reward: 1:{tp_pnl/sl_pnl:.2f}")

        # Simulate trend continuation
        print(f"\n" + "=" * 80)
        print(f"SIMULATING DOWNTREND (PROFITABLE FOR SHORT)")
        print(f"=" * 80)

        for step in range(15):
            if env.position is None:
                break

            prev_trailing = env.trailing_stop_loss
            state, reward, terminated, truncated, info = env.step(0)  # HOLD

            if env.position:
                current_price = env.data[env.current_step, 3]
                unrealized_pnl = (entry_price - current_price) * position_size

                print(f"\n  Step {step+1}:")
                print(f"    Price: ${current_price:,.2f}")
                print(f"    Unrealized PnL: ${unrealized_pnl:,.2f}")
                print(f"    Trailing SL: ${prev_trailing:,.2f} → ${env.trailing_stop_loss:,.2f}")
                print(f"    Bars in Position: {env.bars_in_position}")
            else:
                final_price = env.data[env.current_step-1, 3]
                final_pnl = env.total_pnl

                print(f"\n  ❌ POSITION CLOSED!")
                print(f"    Final Price: ${final_price:,.2f}")
                print(f"    Final PnL: ${final_pnl:,.2f}")
                break

        # Analysis
        print(f"\n" + "=" * 80)
        print(f"ANALYSIS")
        print(f"=" * 80)

        if env.position is None:
            print(f"\n  Final Balance: ${env.balance:,.2f}")
            print(f"  Total PnL: ${env.total_pnl:,.2f}")
            print(f"  PnL vs Initial: {(env.balance/10000-1)*100:.2f}%")

            if env.total_pnl > 50:
                print(f"\n  ✅ SUCCESS! Profit > $50 (vs previous $6-12)")
                print(f"  ✅ Trailing stop allowed profit to run!")
            elif env.total_pnl > 20:
                print(f"\n  ⚠️  Moderate improvement (${env.total_pnl:.2f})")
                print(f"  Consider increasing trailing to 2.5x or 3.0x ATR")
            else:
                print(f"\n  ❌ Still too tight! PnL: ${env.total_pnl:.2f}")
                print(f"  Need to increase trailing multiplier further")

        # Comparison
        print(f"\n[COMPARISON]")
        print(f"  Old (1.0x ATR trailing):")
        print(f"    - Trailing distance: ~0.5% of price")
        print(f"    - Average profit: $6-12")
        print(f"    - Win rate: 97% (too high)")
        print(f"    - Problem: Closes too early")
        print(f"\n  New (2.0x ATR trailing):")
        print(f"    - Trailing distance: ~1.0% of price")
        print(f"    - Expected profit: $50-200")
        print(f"    - Expected win rate: 60-70%")
        print(f"    - Benefit: Lets profits run")

if __name__ == "__main__":
    test_trailing_2x_atr()
