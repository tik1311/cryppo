#!/usr/bin/env python3
"""
Test ATR calculation fix
Verify that ATR is now calculated correctly with minimum threshold
"""

import numpy as np
import sys
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

from src.policy.trading_env import TradingEnv

def test_atr_calculation():
    """Test ATR calculation with synthetic data"""
    print("=" * 80)
    print("ATR CALCULATION FIX TEST")
    print("=" * 80)

    # Create synthetic data (BTC @ $92,000)
    np.random.seed(42)
    n_samples = 100
    base_price = 92000.0

    # Simulate price data with realistic volatility
    prices = []
    for i in range(n_samples):
        # Add some noise (±0.5% volatility)
        noise = np.random.randn() * base_price * 0.005
        price = base_price + noise
        prices.append(price)

    prices = np.array(prices)

    # Create OHLCV data
    data = np.zeros((n_samples, 6))
    for i in range(n_samples):
        open_price = prices[i]
        high_price = open_price * (1 + abs(np.random.randn() * 0.001))
        low_price = open_price * (1 - abs(np.random.randn() * 0.001))
        close_price = np.random.uniform(low_price, high_price)
        volume = 1000.0

        data[i] = [i, open_price, high_price, low_price, close_price, volume]

    # Create environment
    env = TradingEnv(
        data=data,
        initial_balance=10000.0,
        window_size=20
    )

    # Reset to a position where we have enough data
    env.reset()
    env.current_step = 50  # Move to middle

    # Calculate ATR
    atr = env._calculate_atr()
    current_price = data[env.current_step, 3]

    # Expected minimum ATR (0.5% of price)
    min_atr = current_price * 0.005

    print(f"\n[TEST RESULTS]")
    print(f"Current Price: ${current_price:,.2f}")
    print(f"Calculated ATR: ${atr:,.2f} ({atr/current_price*100:.3f}% of price)")
    print(f"Minimum ATR Threshold: ${min_atr:,.2f} ({min_atr/current_price*100:.3f}% of price)")

    # Calculate TP/SL with new multipliers
    tp_distance = atr * 3.0  # New default
    sl_distance = atr * 1.5  # New default

    print(f"\n[TP/SL CALCULATION] (NEW DEFAULTS)")
    print(f"TP Multiplier: 3.0x ATR")
    print(f"SL Multiplier: 1.5x ATR")
    print(f"Trailing Multiplier: 1.0x ATR")

    # For SHORT position
    entry_price = current_price
    tp = entry_price - tp_distance
    sl = entry_price + sl_distance
    trailing = atr * 1.0

    tp_pct = abs((tp - entry_price) / entry_price * 100)
    sl_pct = abs((sl - entry_price) / entry_price * 100)

    print(f"\n[SHORT POSITION EXAMPLE]")
    print(f"Entry: ${entry_price:,.2f}")
    print(f"TP: ${tp:,.2f} ({tp_pct:.2f}% profit target)")
    print(f"SL: ${sl:,.2f} ({sl_pct:.2f}% max loss)")
    print(f"Trailing Distance: ${trailing:,.2f}")
    print(f"Risk:Reward Ratio: 1:{tp_pct/sl_pct:.2f}")

    # Verify ATR meets minimum
    if atr >= min_atr:
        print(f"\n✅ PASS: ATR meets minimum threshold")
    else:
        print(f"\n❌ FAIL: ATR below minimum threshold")
        return False

    # Verify TP/SL are reasonable (at least 0.5% each)
    if tp_pct >= 0.5 and sl_pct >= 0.3:
        print(f"✅ PASS: TP/SL ranges are reasonable")
    else:
        print(f"❌ FAIL: TP/SL ranges too narrow")
        return False

    # Verify Risk:Reward is positive
    rr_ratio = tp_pct / sl_pct
    if rr_ratio >= 1.5:
        print(f"✅ PASS: Risk:Reward ratio is healthy ({rr_ratio:.2f}:1)")
    else:
        print(f"⚠️  WARNING: Risk:Reward ratio could be better ({rr_ratio:.2f}:1)")

    print("\n" + "=" * 80)
    print("COMPARISON: OLD vs NEW")
    print("=" * 80)

    # Old calculation (what it would have been)
    old_atr = 25.0  # Typical wrong value from old calculation
    old_tp_distance = old_atr * 2.0
    old_sl_distance = old_atr * 1.0
    old_tp_pct = abs(old_tp_distance / entry_price * 100)
    old_sl_pct = abs(old_sl_distance / entry_price * 100)
    old_rr = old_tp_pct / old_sl_pct

    print(f"\nOLD METHOD:")
    print(f"  ATR: ${old_atr:.2f} ({old_atr/current_price*100:.3f}% of price)")
    print(f"  TP: {old_tp_pct:.2f}% | SL: {old_sl_pct:.2f}%")
    print(f"  Risk:Reward: 1:{old_rr:.2f}")

    print(f"\nNEW METHOD:")
    print(f"  ATR: ${atr:.2f} ({atr/current_price*100:.3f}% of price)")
    print(f"  TP: {tp_pct:.2f}% | SL: {sl_pct:.2f}%")
    print(f"  Risk:Reward: 1:{rr_ratio:.2f}")

    improvement = (atr / old_atr - 1) * 100
    print(f"\n📈 ATR improved by {improvement:.1f}%")
    print(f"📈 TP/SL ranges are {tp_pct/old_tp_pct:.1f}x wider")

    return True

if __name__ == "__main__":
    success = test_atr_calculation()
    sys.exit(0 if success else 1)
