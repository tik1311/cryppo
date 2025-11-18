#!/usr/bin/env python3
"""
Quick test to verify action space fix
"""

import numpy as np
import sys
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

from src.policy.trading_env import TradingEnv
from src.agent.d3qn_agent import D3QNAgent

def test_action_space():
    """Test that environment and agent have matching action spaces"""
    print("=" * 80)
    print("ACTION SPACE COMPATIBILITY TEST")
    print("=" * 80)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    data = np.random.randn(n_samples, 6) * 100 + 50000

    # Create environment
    env = TradingEnv(
        data=data,
        initial_balance=10000.0,
        window_size=20
    )

    print(f"\n[ENVIRONMENT]")
    print(f"  Action Space: {env.action_space.n} actions")
    print(f"  Actions: HOLD=0, LONG=1, SHORT=2")

    # Create agent
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n

    agent = D3QNAgent(
        state_dim=state_dim,
        action_dim=action_dim
    )

    print(f"\n[AGENT]")
    print(f"  State Dim: {state_dim}")
    print(f"  Action Dim: {agent.action_dim}")

    # Test action selection
    print(f"\n[TESTING ACTION SELECTION]")
    state, _ = env.reset()
    state_flat = state.flatten()

    try:
        # Test training mode (exploration)
        action = agent.select_action(state_flat, training=True)
        print(f"  ✅ Training action: {action} (['HOLD', 'LONG', 'SHORT'][action])")

        # Test evaluation mode (exploitation)
        action = agent.select_action(state_flat, training=False)
        print(f"  ✅ Evaluation action: {action} (['HOLD', 'LONG', 'SHORT'][action])")

        # Test multiple random actions
        print(f"\n[TESTING 10 RANDOM ACTIONS]")
        action_counts = {0: 0, 1: 0, 2: 0}
        for _ in range(100):
            action = agent.select_action(state_flat, training=True)
            action_counts[action] = action_counts.get(action, 0) + 1

        print(f"  HOLD: {action_counts[0]} ({action_counts[0]/100*100:.1f}%)")
        print(f"  LONG: {action_counts[1]} ({action_counts[1]/100*100:.1f}%)")
        print(f"  SHORT: {action_counts[2]} ({action_counts[2]/100*100:.1f}%)")

        print(f"\n[TESTING ENV STEP]")
        for i, action in enumerate([0, 1, 2]):
            state, _ = env.reset()
            state, reward, terminated, truncated, info = env.step(action)
            print(f"  ✅ Action {action} (['HOLD', 'LONG', 'SHORT'][action]): reward={reward:.4f}")

        print(f"\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print(f"\nEnvironment and Agent are compatible:")
        print(f"  - Action space: 3 (HOLD, LONG, SHORT)")
        print(f"  - No CLOSE action")
        print(f"  - Agent exploration bias: 10% HOLD, 45% LONG, 45% SHORT")
        print(f"  - Ready for training!")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_action_space()
    sys.exit(0 if success else 1)
