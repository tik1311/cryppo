"""
Simple backtest script to test trained D3QN model
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent))

from server.src.agent.d3qn_agent import D3QNAgent
from server.src.policy.trading_env import TradingEnv
from server.src.data.loader import load_market_data, prepare_env_data, create_sample_data


def run_backtest(
    model_path: str,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    initial_balance: float = 10000.0,
    max_leverage: float = 3.0,
    fee_rate: float = 0.0004,
    window_size: int = 72,
    use_sample_data: bool = True
):
    """Run backtest on trained model"""

    print("="*60)
    print("D3QN Model Backtest")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Symbol: {symbol}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Max Leverage: {max_leverage}x")
    print("="*60)
    print()

    # Load model
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    print("[1/5] Loading market data...")
    try:
        if use_sample_data:
            df = create_sample_data(symbol=symbol, n_samples=10000, save_dir="data")
            print(f"  Using synthetic data: {len(df)} candles")
        else:
            df = load_market_data(symbol=symbol, timeframe=timeframe)
            print(f"  Loaded real data: {len(df)} candles")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # Prepare data for environment
    print("[2/5] Preparing environment data...")
    env_data = prepare_env_data(df, add_indicators=True, normalize=False)
    print(f"  Data shape: {env_data.shape[0]} timesteps, {env_data.shape[1]} features")

    # Create environment
    print("[3/5] Creating trading environment...")
    env = TradingEnv(
        data=env_data,
        initial_balance=initial_balance,
        max_leverage=max_leverage,
        fee_rate=fee_rate,
        window_size=window_size,
        reward_type="sharpe"
    )
    print(f"  Environment created successfully")

    # Load model
    print("[4/5] Loading trained model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # Calculate flattened state dimension
    state_dim = np.prod(obs_shape)  # For 2D observation, flatten it

    agent = D3QNAgent(
        state_dim=state_dim,
        action_dim=n_actions,
        learning_rate=0.00025,
        gamma=0.995,
        device=device
    )

    agent.load(str(model_path))
    agent.policy_net.eval()
    print(f"  Model loaded successfully")
    print(f"  State dim: {state_dim}, Action dim: {n_actions}")

    # Run backtest
    print("[5/5] Running backtest simulation...")
    state, _ = env.reset()
    done = False
    step = 0
    total_reward = 0

    equity_curve = [initial_balance]

    while not done:
        # Select action using trained model
        with torch.no_grad():
            # Flatten state if it's 2D
            state_flat = state.flatten() if len(state.shape) > 1 else state
            state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(device)
            q_values = agent.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()

        # Take action
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        equity_curve.append(info['balance'])

        state = next_state
        step += 1

        if step % 1000 == 0:
            print(f"  Step {step}/{len(env_data)-window_size} - Balance: ${info['balance']:,.2f}")

    # Print results
    print()
    print("="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    final_balance = info['balance']
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    total_trades = info.get('total_trades', 0)
    win_rate = info.get('win_rate', 0) * 100

    print(f"Initial Balance:  ${initial_balance:,.2f}")
    print(f"Final Balance:    ${final_balance:,.2f}")
    print(f"Total Return:     {total_return:+.2f}%")
    print(f"Total Reward:     {total_reward:.2f}")
    print(f"Total Trades:     {total_trades}")
    print(f"Win Rate:         {win_rate:.2f}%")
    print(f"Total Steps:      {step}")

    # Calculate additional metrics
    equity_array = np.array(equity_curve)
    returns = np.diff(equity_array) / equity_array[:-1]

    if len(returns) > 0 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
    else:
        sharpe_ratio = 0

    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    max_drawdown = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0

    print(f"Sharpe Ratio:     {sharpe_ratio:.4f}")
    print(f"Max Drawdown:     {max_drawdown:.2f}%")

    print("="*60)
    print("[SUCCESS] Backtest completed!")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest D3QN model")
    parser.add_argument("--model", type=str, default="checkpoints/final_agent.pth",
                       help="Path to trained model")
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                       help="Trading symbol")
    parser.add_argument("--balance", type=float, default=10000.0,
                       help="Initial balance")
    parser.add_argument("--leverage", type=float, default=3.0,
                       help="Max leverage")
    parser.add_argument("--sample-data", action="store_true",
                       help="Use synthetic sample data")

    args = parser.parse_args()

    run_backtest(
        model_path=args.model,
        symbol=args.symbol,
        initial_balance=args.balance,
        max_leverage=args.leverage,
        use_sample_data=args.sample_data
    )
