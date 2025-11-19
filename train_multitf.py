"""
Training Script for Multi-Timeframe D3QN Agent
Phase 6: Train with hierarchical multi-timeframe system

Usage:
    python train_multitf.py --primary-tf 5m --episodes 200 --symbol BTCUSDT
"""

import argparse
import numpy as np
from pathlib import Path
import torch
from datetime import datetime
import json

from server.src.data.multi_timeframe_loader import MultiTimeframeLoader
from server.src.policy.multi_timeframe_env import make_multitf_env
from server.src.agent.multi_timeframe_agent import MultiTimeframeD3QNAgent


def train_multitf_agent(args):
    """
    Train Multi-Timeframe D3QN Agent

    Args:
        args: Command line arguments
    """
    print("=" * 80)
    print("🚀 Phase 6: Multi-Timeframe D3QN Training")
    print("=" * 80)

    # Create data loader
    loader = MultiTimeframeLoader(
        primary_timeframe=args.primary_tf,
        detail_timeframe='1m',
        window_size=args.window_size
    )

    # Download data
    print(f"\n[Step 1] Downloading data for {args.symbol}")
    primary_df, detail_df = loader.download_data(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=Path(args.data_dir)
    )

    # Prepare for training
    print(f"\n[Step 2] Preparing multi-timeframe data")
    data = loader.prepare_for_training(primary_df, detail_df, normalize=True)

    primary_data = data['primary']
    detail_data = data['detail']

    print(f"  Primary shape: {primary_data.shape}")
    print(f"  Detail shape: {detail_data.shape}")

    # Create environment
    print(f"\n[Step 3] Creating multi-timeframe environment")
    env = make_multitf_env(
        primary_data=primary_data,
        detail_data=detail_data,
        primary_timeframe=args.primary_tf,
        initial_balance=args.initial_balance,
        window_size=args.window_size,
        fee_rate=0.0004,
        max_leverage=1.0,  # Conservative 1x
        atr_sl_multiplier=1.3,  # Phase 5: Tighter SL
        atr_tp_multiplier=6.0   # Phase 5: Wider TP (4.62x R:R)
    )

    # Create agent
    print(f"\n[Step 4] Creating multi-timeframe D3QN agent")
    agent = MultiTimeframeD3QNAgent(
        primary_window=args.window_size,
        primary_features=primary_data.shape[1],
        detail_ratio=detail_data.shape[1],
        detail_features=detail_data.shape[2],
        action_dim=3,  # HOLD, LONG, SHORT
        n_quantiles=32,
        learning_rate=args.lr,
        gamma=0.995,
        buffer_size=100000,
        batch_size=128,
        epsilon_start=1.0,
        epsilon_end=0.25,  # Phase 5: More exploration
        epsilon_decay=0.9997,  # Phase 5: Slower decay
        target_update_freq=250,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Training loop
    print(f"\n[Step 5] Training for {args.episodes} episodes")
    print("=" * 80)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'episodes': [],
        'rewards': [],
        'final_balance': [],
        'win_rate': [],
        'total_trades': [],
        'hold_pct': [],
        'epsilon': []
    }

    for episode in range(1, args.episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            # Get action
            action_mask = env.get_action_mask()
            action = agent.select_action(
                state['primary'],
                state['detail'],
                action_mask,
                training=True
            )

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience
            agent.store_experience(
                state['primary'], state['detail'], action, reward,
                next_state['primary'], next_state['detail'], done
            )

            # Train
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.train_step()

            episode_reward += reward
            state = next_state
            steps += 1

        # Episode summary
        final_balance = info['balance']
        win_rate = info['win_rate']
        total_trades = info['total_trades']

        # Calculate HOLD percentage
        action_counts = env.env.action_counts
        total_actions = sum(action_counts.values())
        hold_pct = action_counts.get(0, 0) / max(total_actions, 1) * 100

        # Log metrics
        metrics['episodes'].append(episode)
        metrics['rewards'].append(float(episode_reward))
        metrics['final_balance'].append(float(final_balance))
        metrics['win_rate'].append(float(win_rate))
        metrics['total_trades'].append(int(total_trades))
        metrics['hold_pct'].append(float(hold_pct))
        metrics['epsilon'].append(float(agent.epsilon))

        # Print summary
        profit_pct = (final_balance / args.initial_balance - 1.0) * 100
        print(f"\nEpisode {episode}/{args.episodes}")
        print(f"  Reward: {episode_reward:.2f} | Steps: {steps}")
        print(f"  Balance: ${final_balance:.2f} ({profit_pct:+.1f}%)")
        print(f"  Trades: {total_trades} | Win Rate: {win_rate*100:.1f}%")
        print(f"  HOLD: {hold_pct:.1f}% | Epsilon: {agent.epsilon:.4f}")

        # Save checkpoint every N episodes
        if episode % args.save_freq == 0:
            checkpoint_path = log_dir / f"mt_d3qn_ep{episode}.pth"
            agent.save(checkpoint_path)

    # Save final model
    final_path = log_dir / "mt_d3qn_final.pth"
    agent.save(final_path)

    # Save metrics
    metrics_path = log_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("✓ Training completed!")
    print(f"  Final model: {final_path}")
    print(f"  Metrics: {metrics_path}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multi-Timeframe D3QN Agent')

    # Data args
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--primary-tf', type=str, default='5m', help='Primary timeframe (5m, 15m, 1h)')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')

    # Training args
    parser.add_argument('--episodes', type=int, default=200, help='Number of episodes')
    parser.add_argument('--window-size', type=int, default=72, help='Window size')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')

    # Logging args
    parser.add_argument('--log-dir', type=str, default='logs/multitf', help='Log directory')
    parser.add_argument('--save-freq', type=int, default=50, help='Save frequency (episodes)')

    args = parser.parse_args()

    train_multitf_agent(args)
