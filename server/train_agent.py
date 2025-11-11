"""
Training Script for D3QN Agent
1H Crypto Trading
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent.d3qn_agent import D3QNAgent
from policy.trading_env import TradingEnv
from data.features import add_technical_indicators, normalize_features


def load_config():
    """Load configuration"""
    config_path = os.path.join(os.path.dirname(__file__), 'src', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from file

    Args:
        data_path: Path to data file (.npz or .csv)

    Returns:
        DataFrame with OHLCV data
    """
    if data_path.endswith('.npz'):
        # Load from compressed numpy
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
        # Load from CSV
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    return df


def prepare_data(df: pd.DataFrame, config: dict) -> dict:
    """
    Prepare data for training

    Args:
        df: DataFrame with OHLCV data
        config: Configuration dictionary

    Returns:
        Dictionary with train/val data
    """
    print("Preparing data...")

    # Add technical indicators
    df = add_technical_indicators(df)

    # Drop NaN values
    df = df.dropna()

    print(f"Data shape after indicators: {df.shape}")

    # Get feature columns
    exclude_cols = ['timestamp'] if 'timestamp' in df.columns else []
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Normalize
    df_norm = normalize_features(df, feature_cols, method='minmax')

    # Split train/val
    train_size = int(len(df_norm) * 0.8)

    train_data = df_norm[:train_size][feature_cols].values
    val_data = df_norm[train_size:][feature_cols].values

    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    print(f"Features: {len(feature_cols)}")

    return {
        'train': train_data,
        'val': val_data,
        'n_features': len(feature_cols),
        'feature_cols': feature_cols
    }


def train_episode(env: TradingEnv, agent: D3QNAgent, training: bool = True) -> dict:
    """
    Train one episode

    Args:
        env: Trading environment
        agent: D3QN agent
        training: Whether to train (vs evaluate)

    Returns:
        Episode statistics
    """
    state, _ = env.reset()
    done = False
    episode_reward = 0.0
    episode_steps = 0
    losses = []

    while not done:
        # Flatten state for agent
        state_flat = state.reshape(-1)

        # Select action
        action = agent.select_action(state_flat, training=training)

        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        if training:
            next_state_flat = next_state.reshape(-1)
            agent.store_transition(state_flat, action, reward, next_state_flat, done)

            # Update agent
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

        episode_reward += reward
        episode_steps += 1
        state = next_state

    return {
        'reward': episode_reward,
        'steps': episode_steps,
        'final_equity': info['equity'],
        'total_trades': info['total_trades'],
        'win_rate': info['win_rate'],
        'avg_loss': np.mean(losses) if losses else 0.0
    }


def train(args):
    """Main training function"""
    print("=" * 60)
    print("RL Crypto Trading - Training")
    print("D3QN Agent | 1H Timeframe")
    print("=" * 60)

    # Load config
    config = load_config()
    model_config = config['model']
    training_config = model_config['training']

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    df = load_data(args.data_path)
    print(f"Loaded {len(df)} candles")

    # Prepare data
    data_dict = prepare_data(df, config)

    # Create environments
    train_env = TradingEnv(
        data=data_dict['train'],
        initial_balance=config['trading']['initial_balance'],
        fee_rate=config['trading']['fee_taker'],
        max_leverage=config['trading']['max_leverage'],
        window_size=config['data']['window_size'],
        reward_type=model_config['reward_type']
    )

    val_env = TradingEnv(
        data=data_dict['val'],
        initial_balance=config['trading']['initial_balance'],
        fee_rate=config['trading']['fee_taker'],
        max_leverage=config['trading']['max_leverage'],
        window_size=config['data']['window_size'],
        reward_type=model_config['reward_type']
    )

    # Get state dimension
    dummy_state, _ = train_env.reset()
    state_dim = dummy_state.reshape(-1).shape[0]
    action_dim = train_env.action_space.n

    print(f"\nState dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Create agent
    agent = D3QNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=training_config['learning_rate'],
        gamma=training_config['gamma'],
        epsilon_start=training_config['epsilon_start'],
        epsilon_end=training_config['epsilon_end'],
        epsilon_decay=training_config['epsilon_decay'],
        buffer_size=training_config['buffer_size'],
        batch_size=training_config['batch_size'],
        target_update_freq=training_config['target_update_freq'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\nDevice: {agent.device}")
    print(f"Network: {model_config['network']['hidden_layers']}")

    # Training loop
    n_episodes = args.episodes
    eval_freq = args.eval_freq
    save_freq = args.save_freq

    print(f"\nStarting training for {n_episodes} episodes...")
    print("=" * 60)

    best_val_reward = float('-inf')

    for episode in range(1, n_episodes + 1):
        # Train episode
        train_stats = train_episode(train_env, agent, training=True)

        # Log training
        if episode % 10 == 0:
            print(f"\nEpisode {episode}/{n_episodes}")
            print(f"  Reward: {train_stats['reward']:.4f}")
            print(f"  Final Equity: ${train_stats['final_equity']:.2f}")
            print(f"  Trades: {train_stats['total_trades']}")
            print(f"  Win Rate: {train_stats['win_rate']:.2%}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Avg Loss: {train_stats['avg_loss']:.6f}")

        # Evaluate
        if episode % eval_freq == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at episode {episode}")
            print(f"{'='*60}")

            val_stats = train_episode(val_env, agent, training=False)

            print(f"Val Reward: {val_stats['reward']:.4f}")
            print(f"Val Equity: ${val_stats['final_equity']:.2f}")
            print(f"Val Trades: {val_stats['total_trades']}")
            print(f"Val Win Rate: {val_stats['win_rate']:.2%}")

            # Save best model
            if val_stats['reward'] > best_val_reward:
                best_val_reward = val_stats['reward']
                model_path = os.path.join(args.output_dir, 'best_model.pth')
                agent.save(model_path)
                print(f"✓ Saved best model (reward: {best_val_reward:.4f})")

        # Save checkpoint
        if episode % save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_ep{episode}.pth')
            agent.save(checkpoint_path)
            print(f"✓ Saved checkpoint at episode {episode}")

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    agent.save(final_path)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation reward: {best_val_reward:.4f}")
    print(f"Models saved to: {args.output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train D3QN agent for crypto trading')

    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to data file (.npz or .csv)')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Output directory for models')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--eval-freq', type=int, default=50,
                        help='Evaluation frequency (episodes)')
    parser.add_argument('--save-freq', type=int, default=100,
                        help='Checkpoint save frequency (episodes)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train
    train(args)


if __name__ == "__main__":
    main()
