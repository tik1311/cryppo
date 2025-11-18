"""
Simple backtest script to test trained D3QN model
"""

import sys
import torch
import numpy as np
from pathlib import Path
import random

# Add server to path
sys.path.insert(0, str(Path(__file__).parent))

from server.src.agent.d3qn_agent import D3QNAgent
from server.src.policy.trading_env import TradingEnv
from server.src.data.loader import load_market_data, prepare_env_data, create_sample_data
from server.src.utils.detailed_logger import DetailedStepLogger


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
            # Use minimal sample data for quick testing (2000 samples)
            df = create_sample_data(symbol=symbol, n_samples=2000, save_dir="data")
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
    print(f"  [DEBUG] Enabling detailed logging for backtest...")
    env.enable_detailed_logging = True

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

    # Run backtest with detailed logging
    print("[5/5] Running backtest simulation with detailed logging...")

    # Initialize detailed logger
    detailed_logger = DetailedStepLogger(log_dir="logs/backtest/detailed", prefix="backtest")
    print(f"  Detailed logs will be saved to: logs/backtest/detailed/")

    state, _ = env.reset()
    terminated = False
    truncated = False
    step = 0
    total_reward = 0

    equity_curve = [initial_balance]
    action_names = ['HOLD', 'LONG', 'SHORT']
    episode_actions = {'HOLD': 0, 'LONG': 0, 'SHORT': 0}
    episode_trades_start = 0

    # Loop until episode ends (either terminated or truncated)
    while not (terminated or truncated):
        # Get current price
        current_price = env.data[env.current_step, 3]

        # Get Q-values for logging
        state_flat = state.flatten() if len(state.shape) > 1 else state
        q_values = agent.get_q_values(state_flat)

        # Select action using trained model (greedy, no exploration)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(device)
            q_values_tensor = agent.policy_net(state_tensor)
            action = q_values_tensor.argmax(dim=1).item()

        episode_actions[action_names[action]] += 1

        # Calculate ATR
        atr = env._calculate_atr() if hasattr(env, '_calculate_atr') else 0.0

        # Take action (Gymnasium API: observation, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        equity_curve.append(info['balance'])

        # Detailed step logging
        step_data = {
            'episode': 1,
            'step': step,
            'global_step': env.current_step - env.window_size,
            'price': current_price,
            'action': action,
            'action_name': action_names[action],
            'q_values': q_values.tolist(),
            'epsilon': 0.0,  # No exploration in backtest
            'position': env.position,
            'position_size': env.position_size if env.position else 0.0,
            'entry_price': env.entry_price if env.position else 0.0,
            'unrealized_pnl': env._calculate_position_pnl() if env.position else 0.0,
            'unrealized_pnl_pct': (env._calculate_position_pnl() / env.balance * 100) if env.position and env.balance > 0 else 0.0,
            'tp': env.take_profit if env.position else 0.0,
            'sl': env.stop_loss if env.position else 0.0,
            'atr': atr,
            'balance': env.balance,
            'equity': env.equity,
            'trade_reward': 0.0,
            'exploration_bonus': 0.0,
            'position_penalty': 0.0,
            'total_reward': reward,
            'loss': None,  # No training in backtest
            'reason': f"Greedy (Q={q_values[action]:.4f} best)",
            'terminated': terminated,
            'truncated': truncated
        }
        detailed_logger.log_step(step_data)

        state = next_state
        step += 1

        if step % 1000 == 0:
            print(f"  Step {step}/{len(env_data)-window_size} - Balance: ${info['balance']:,.2f}")

    # Log all trades
    if hasattr(env, 'trade_log'):
        for trade in env.trade_log:
            trade_data = {
                'episode': 1,
                'trade_id': trade.get('position_id', 'unknown'),
                'direction': trade['direction'],
                'entry_step': trade['entry_step'],
                'entry_price': trade['entry_price'],
                'exit_step': trade['exit_step'],
                'exit_price': trade['exit_price'],
                'exit_reason': 'tp' if trade['pnl'] > 0 else 'sl',
                'duration_steps': trade['duration_steps'],
                'size': trade['quantity'],
                'leverage': trade['leverage'],
                'atr_at_entry': 0.0,
                'tp_price': 0.0,
                'sl_price': 0.0,
                'gross_pnl': trade['pnl'],
                'fees': 0.0,
                'net_pnl': trade['pnl'],
                'pnl_pct_vs_margin': trade['pnl_pct'],
                'pnl_pct_vs_notional': trade['pnl_pct'] / trade['leverage'] if trade['leverage'] > 0 else 0.0,
                'balance_before': 0.0,
                'balance_after': 0.0
            }
            detailed_logger.log_trade(trade_data)

    # Calculate episode statistics
    final_balance = info['balance']
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    total_trades = info.get('total_trades', 0)
    win_rate = info.get('win_rate', 0)

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

    # Calculate trade statistics
    if hasattr(env, 'trade_log') and env.trade_log:
        winning_trades = sum(1 for t in env.trade_log if t['pnl'] > 0)
        losing_trades = sum(1 for t in env.trade_log if t['pnl'] <= 0)
        avg_win = np.mean([t['pnl'] for t in env.trade_log if t['pnl'] > 0]) if winning_trades > 0 else 0.0
        avg_loss_trade = np.mean([t['pnl'] for t in env.trade_log if t['pnl'] <= 0]) if losing_trades > 0 else 0.0
        rr_ratio = abs(avg_win / avg_loss_trade) if avg_loss_trade != 0 else 0.0
    else:
        winning_trades = 0
        losing_trades = 0
        avg_win = 0.0
        avg_loss_trade = 0.0
        rr_ratio = 0.0

    # Log episode summary
    episode_data = {
        'episode': 1,
        'steps': step,
        'total_reward': total_reward,
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'return_pct': total_return,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss_trade,
        'rr_ratio': rr_ratio,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'action_distribution': episode_actions,
        'avg_loss': 0.0,  # No training loss in backtest
        'epsilon_start': 0.0,
        'epsilon_end': 0.0
    }
    detailed_logger.log_episode_end(episode_data)

    # Close detailed logger
    detailed_logger.close()

    # Print results
    print()
    print("="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    print(f"Initial Balance:  ${initial_balance:,.2f}")
    print(f"Final Balance:    ${final_balance:,.2f}")
    print(f"Total Return:     {total_return:+.2f}%")
    print(f"Total Reward:     {total_reward:.2f}")
    print(f"Total Trades:     {total_trades}")
    print(f"Winning Trades:   {winning_trades}")
    print(f"Losing Trades:    {losing_trades}")
    print(f"Win Rate:         {win_rate*100:.2f}%")
    print(f"Average Win:      ${avg_win:.2f}")
    print(f"Average Loss:     ${avg_loss_trade:.2f}")
    print(f"RR Ratio:         {rr_ratio:.2f}:1")
    print(f"Total Steps:      {step}")
    print(f"Sharpe Ratio:     {sharpe_ratio:.4f}")
    print(f"Max Drawdown:     {max_drawdown:.2f}%")

    print("="*60)
    print("[SUCCESS] Backtest completed!")
    print("="*60)
    print()
    print("[INFO] Detailed logs saved to:")
    print(f"  logs/backtest/detailed/steps_*.jsonl (every step)")
    print(f"  logs/backtest/detailed/trades_*.jsonl (every trade)")
    print(f"  logs/backtest/detailed/episodes_*.jsonl (episode summary)")
    print(f"  logs/backtest/detailed/analysis_*.txt (human-readable)")
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
