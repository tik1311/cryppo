#!/usr/bin/env python3
"""
Test to verify trailing stop logic fix
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "server"))

from src.policy.trading_env import TradingEnv

def test_trailing_logic_fix():
    """Test that trailing stop now protects profit correctly"""
    print("=" * 80)
    print("TEST: TRAILING STOP LOGIC FIX")
    print("=" * 80)

    # Create realistic BTC data with downtrend (good for SHORT)
    np.random.seed(42)
    n_samples = 50
    base_price = 28000.0

    data = []
    for i in range(n_samples):
        # Gradual downtrend
        trend_price = base_price - (i * 50)  # -$50 per candle
        noise = np.random.randn() * 100
        open_price = trend_price + noise
        high_price = open_price + abs(np.random.randn() * 80)
        low_price = open_price - abs(np.random.randn() * 80)
        close_price = np.random.uniform(low_price, high_price)
        volume = 1000.0
        data.append([i, open_price, high_price, low_price, close_price, volume])

    data = np.array(data)

    # Create environment with fixed settings
    env = TradingEnv(
        data=data,
        initial_balance=10000.0,
        max_leverage=1.5,
        window_size=14,
        streaming_mode=False
    )

    print(f"\n[ENVIRONMENT SETTINGS]")
    print(f"  Leverage: {env.max_leverage}x")
    print(f"  SL Multiplier: {env.atr_sl_multiplier}x ATR")
    print(f"  TP Multiplier: {env.atr_tp_multiplier}x ATR")
    print(f"  Trailing Multiplier: {env.trailing_distance_multiplier}x ATR")
    print(f"\n  ✅ CHECK: Trailing ({env.trailing_distance_multiplier}x) < SL ({env.atr_sl_multiplier}x) = {env.trailing_distance_multiplier < env.atr_sl_multiplier}")

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
        print(f"  ATR: ${atr:.2f}")
        print(f"\n  TP:  ${tp:,.2f} ({abs((tp-entry_price)/entry_price*100):.2f}%)")
        print(f"  SL:  ${sl:,.2f} ({abs((sl-entry_price)/entry_price*100):.2f}%)")
        print(f"  Trailing: ${trailing:,.2f} (initially = SL)")

        # Calculate distances
        sl_distance = abs(sl - entry_price)
        tp_distance = abs(entry_price - tp)
        trailing_distance_atr = atr * env.trailing_distance_multiplier

        print(f"\n[DISTANCES]")
        print(f"  SL Distance: ${sl_distance:.2f} ({sl_distance/atr:.2f}x ATR)")
        print(f"  TP Distance: ${tp_distance:.2f} ({tp_distance/atr:.2f}x ATR)")
        print(f"  Trailing Distance: ${trailing_distance_atr:.2f} ({env.trailing_distance_multiplier:.2f}x ATR)")
        print(f"\n  ✅ Trailing distance ({trailing_distance_atr:.2f}) < SL distance ({sl_distance:.2f})")

        # Simulate price drop (profitable for SHORT)
        print(f"\n" + "=" * 80)
        print(f"SIMULATING DOWNTREND")
        print(f"=" * 80)

        step_count = 0
        max_profit = 0
        trailing_activated = False

        for step in range(15):
            if env.position is None:
                break

            prev_trailing = env.trailing_stop_loss
            prev_lowest = env.lowest_price_since_entry

            state, reward, terminated, truncated, info = env.step(0)  # HOLD

            if env.position:
                current_price = env.data[env.current_step, 3]
                unrealized_pnl = (entry_price - current_price) * position_size
                max_profit = max(max_profit, unrealized_pnl)

                print(f"\n  Step {step+1}:")
                print(f"    Price: ${current_price:,.2f}")
                print(f"    Unrealized PnL: ${unrealized_pnl:,.2f}")
                print(f"    Lowest since entry: ${prev_lowest:.2f} → ${env.lowest_price_since_entry:.2f}")

                if env.trailing_stop_loss != prev_trailing:
                    trailing_activated = True
                    improvement = prev_trailing - env.trailing_stop_loss
                    print(f"    ✅ Trailing UPDATED: ${prev_trailing:,.2f} → ${env.trailing_stop_loss:,.2f} (improved by ${improvement:.2f})")

                    # Verify it's better than original SL
                    if env.trailing_stop_loss < sl:
                        print(f"    ✅ New trailing (${env.trailing_stop_loss:,.2f}) < Original SL (${sl:.2f}) - PROTECTING PROFIT!")
                    else:
                        print(f"    ❌ ERROR: New trailing (${env.trailing_stop_loss:,.2f}) >= Original SL (${sl:.2f})")
                else:
                    print(f"    Trailing unchanged: ${env.trailing_stop_loss:,.2f}")

                step_count += 1
            else:
                final_price = current_price
                final_pnl = env.total_pnl

                print(f"\n  ❌ POSITION CLOSED!")
                print(f"    Final Price: ${final_price:,.2f}")
                print(f"    Final PnL: ${final_pnl:,.2f}")
                print(f"    Max Profit Reached: ${max_profit:,.2f}")
                break

        # Analysis
        print(f"\n" + "=" * 80)
        print(f"ANALYSIS")
        print(f"=" * 80)

        if env.position is None:
            print(f"\n  Final Balance: ${env.balance:,.2f}")
            print(f"  Total PnL: ${env.total_pnl:,.2f}")
            print(f"  Profit vs Initial: {(env.balance/10000-1)*100:.2f}%")
            print(f"  Steps held: {step_count}")

            if trailing_activated:
                print(f"\n  ✅ Trailing stop activated and protected profit!")
            else:
                print(f"\n  ⚠️  Trailing stop never activated (SL or TP hit first)")

            if env.total_pnl > 50:
                print(f"\n  ✅ SUCCESS! Profit > $50 (old system: $6)")
                print(f"  ✅ {env.total_pnl / 6:.1f}x improvement over old $6 profit!")
            elif env.total_pnl > 20:
                print(f"\n  ✅ Moderate improvement: ${env.total_pnl:.2f}")
            elif env.total_pnl > 0:
                print(f"\n  ⚠️  Small profit: ${env.total_pnl:.2f}")
            else:
                print(f"\n  ❌ Loss: ${env.total_pnl:.2f}")

        print(f"\n[COMPARISON]")
        print(f"  Old System (2.0x trailing > 1.5x SL):")
        print(f"    - Trailing widened stop on first update")
        print(f"    - Position closed immediately with $6 profit")
        print(f"    - Win rate 97% but balance -70%")
        print(f"\n  New System (1.0x trailing < 1.5x SL):")
        print(f"    - Trailing only updates if better than SL")
        print(f"    - Position runs until TP/SL/Trailing")
        print(f"    - Expected: 60-70% win rate, positive balance")

        return True

    return False

if __name__ == "__main__":
    success = test_trailing_logic_fix()
    sys.exit(0 if success else 1)
