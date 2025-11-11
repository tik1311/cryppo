"""
Backtest Script for Trained Models
1H Crypto Trading
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent.d3qn_agent import D3QNAgent
from data.features import add_technical_indicators, normalize_features
from eval.backtest import BacktestEngine


def load_config():
    """Load configuration"""
    config_path = os.path.join(os.path.dirname(__file__), 'src', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from file"""
    if data_path.endswith('.npz'):
        data = np.load(data_path)
        df = pd.DataFrame({
            'timestamp': data['timestamp'],
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'volume': data['volume']
        })
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    return df


def backtest(args):
    """Main backtest function"""
    print("=" * 60)
    print("RL Crypto Trading - Backtest")
    print("D3QN Agent | 1H Timeframe")
    print("=" * 60)

    # Load config
    config = load_config()

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    df = load_data(args.data_path)
    print(f"Loaded {len(df)} candles")

    # Prepare data
    print("Preparing data...")
    df = add_technical_indicators(df)
    df = df.dropna()

    # Normalize
    exclude_cols = ['timestamp'] if 'timestamp' in df.columns else []
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    df_norm = normalize_features(df, feature_cols, method='minmax')

    # Get test data
    test_data = df_norm[feature_cols].values
    timestamps = df['timestamp'].values if 'timestamp' in df.columns else range(len(df))
    prices = df['close'].values

    print(f"Test data shape: {test_data.shape}")

    # Load agent
    print(f"\nLoading model from: {args.model_path}")

    # Determine state and action dimensions
    window_size = config['data']['window_size']
    n_features = test_data.shape[1]
    state_dim = window_size * (n_features + 4)  # +4 for position info
    action_dim = 3  # HOLD, BUY, SELL

    agent = D3QNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    agent.load(args.model_path)
    print(f"Model loaded on device: {agent.device}")

    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        initial_balance=config['trading']['initial_balance'],
        fee_rate=config['trading']['fee_taker'],
        max_leverage=config['trading']['max_leverage']
    )

    # Run backtest
    print("\nRunning backtest...")
    print("=" * 60)

    position_info = np.zeros(4)  # [is_in_position, entry_price_norm, pnl_pct, hold_duration]

    for i in range(window_size, len(test_data)):
        # Get state
        market_window = test_data[i - window_size:i]
        position_info_tiled = np.tile(position_info, (window_size, 1))
        state = np.concatenate([market_window, position_info_tiled], axis=1)
        state_flat = state.reshape(-1)

        # Get action from agent
        action = agent.select_action(state_flat, training=False)

        # Map action to trading signal
        action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
        trade_action = action_map[action]

        # Execute in backtest engine
        current_price = prices[i]
        timestamp = timestamps[i]
        backtest_engine.execute_trade(timestamp, current_price, trade_action)

        # Update position info for next state
        if backtest_engine.position:
            position_info[0] = 1.0
            position_info[1] = (backtest_engine.entry_price / current_price - 1.0)
            pnl = backtest_engine._calculate_pnl(current_price)
            position_info[2] = pnl / backtest_engine.balance
            position_info[3] = min(backtest_engine.trades[-1].get('duration', 0) / 100.0, 1.0) if backtest_engine.trades else 0.0
        else:
            position_info = np.zeros(4)

        # Progress
        if i % 100 == 0:
            print(f"Progress: {i}/{len(test_data)} - Equity: ${backtest_engine.equity:.2f}")

    # Print results
    backtest_engine.print_results()

    # Save results
    if args.save_results:
        backtest_engine.save_results(args.save_results)


def main():
    parser = argparse.ArgumentParser(description='Backtest trained D3QN model')

    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model (.pth)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to test data (.npz or .csv)')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Path to save results JSON')

    args = parser.parse_args()

    backtest(args)


if __name__ == "__main__":
    main()
