"""
Training Script for D3QN Agent on Crypto Futures Trading

This script trains a Deep Dueling Double DQN agent to trade cryptocurrency futures.

Usage:
    python -m server.src.training.train_d3qn --symbol BTCUSDT --episodes 1000

Features:
    - Loads market data with technical indicators
    - Trains D3QN agent with experience replay
    - Saves checkpoints every N episodes
    - Logs training metrics (loss, rewards, epsilon, etc.)
    - Periodic evaluation on validation set
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Debug: Print Python path and imports
print(f"[DEBUG] Python executable: {sys.executable}")
print(f"[DEBUG] Python path: {sys.path[:3]}...")
print(f"[DEBUG] Current working directory: {os.getcwd()}")

try:
    print("[DEBUG] Importing D3QNAgent...")
    from src.agent.d3qn_agent import D3QNAgent
    print("[DEBUG] ✓ D3QNAgent imported successfully")
except Exception as e:
    print(f"[DEBUG] ✗ Failed to import D3QNAgent: {e}")
    raise

try:
    print("[DEBUG] Importing TradingEnv...")
    from src.policy.trading_env import TradingEnv
    print("[DEBUG] ✓ TradingEnv imported successfully")
except Exception as e:
    print(f"[DEBUG] ✗ Failed to import TradingEnv: {e}")
    raise

try:
    print("[DEBUG] Importing data loader functions...")
    from src.data.loader import load_market_data, prepare_env_data, split_train_test, create_sample_data
    print("[DEBUG] ✓ Data loader functions imported successfully")
except Exception as e:
    print(f"[DEBUG] ✗ Failed to import data loader functions: {e}")
    raise

try:
    print("[DEBUG] Importing BacktestEngine...")
    from src.eval.backtest import BacktestEngine
    print("[DEBUG] ✓ BacktestEngine imported successfully")
except Exception as e:
    print(f"[DEBUG] ✗ Failed to import BacktestEngine: {e}")
    raise

print("[DEBUG] All imports successful!")


class TrainingLogger:
    """Simple logger for training metrics"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{timestamp}.log"
        self.metrics_file = self.log_dir / f"metrics_{timestamp}.json"

        self.metrics = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'epsilon': [],
            'steps': [],
            'trades': [],
            'win_rate': [],
            'final_balance': []
        }

    def log(self, message: str):
        """Log message to console and file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    def add_episode(self, episode: int, metrics: Dict):
        """Add episode metrics"""
        self.metrics['episodes'].append(episode)
        for key in ['rewards', 'losses', 'epsilon', 'steps', 'trades', 'win_rate', 'final_balance']:
            if key in metrics:
                self.metrics[key].append(metrics[key])

    def save_metrics(self):
        """Save metrics to JSON"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)


def train_d3qn(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    episodes: int = 1000,
    max_steps: int = 10000,
    eval_freq: int = 50,
    save_freq: int = 100,
    use_sample_data: bool = False,
    initial_balance: float = 10000.0,
    max_leverage: float = 3.0,
    learning_rate: float = 0.00025,
    gamma: float = 0.995,
    batch_size: int = 64,
    buffer_size: int = 100000,
    window_size: int = 72,
    reward_type: str = 'sharpe',
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs"
):
    """
    Train D3QN agent

    Args:
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        eval_freq: Evaluate every N episodes
        save_freq: Save checkpoint every N episodes
        use_sample_data: Use synthetic data for testing
        initial_balance: Starting capital
        max_leverage: Maximum leverage
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        batch_size: Batch size for training
        buffer_size: Replay buffer size
        window_size: Observation window size
        reward_type: Reward calculation method
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for logs
    """
    # Initialize logger
    logger = TrainingLogger(log_dir)
    logger.log("=" * 60)
    logger.log("D3QN Crypto Futures Trading - Training")
    logger.log("=" * 60)
    logger.log(f"Symbol: {symbol}")
    logger.log(f"Timeframe: {timeframe}")
    logger.log(f"Episodes: {episodes}")
    logger.log(f"Initial Balance: ${initial_balance:,.2f}")
    logger.log(f"Max Leverage: {max_leverage}x")
    logger.log(f"Window Size: {window_size} candles")
    logger.log(f"Reward Type: {reward_type}")
    logger.log("=" * 60)

    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    # Load market data
    logger.log("\n📊 Loading market data...")
    try:
        if use_sample_data:
            logger.log("⚠️  Using synthetic data for testing")
            df = create_sample_data(symbol=symbol, n_samples=10000, save_dir="data")
        else:
            df = load_market_data(symbol=symbol, timeframe=timeframe)

        logger.log(f"✓ Loaded {len(df):,} candles")
        logger.log(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    except FileNotFoundError as e:
        logger.log(f"❌ Error: {e}")
        logger.log("\n💡 To download real market data:")
        logger.log("   1. Start the server: uvicorn server.main:app")
        logger.log("   2. Download data via API:")
        logger.log(f"      POST /api/data/download")
        logger.log(f"      {{'symbol': '{symbol}', 'timeframe': '{timeframe}', 'start': '2023-01-01', 'end': '2024-12-31'}}")
        logger.log("\n   Or use --sample-data flag to train with synthetic data")
        return

    # Prepare data
    logger.log("\n🔧 Preparing data...")
    data = prepare_env_data(df, add_indicators=True, normalize=True)
    logger.log(f"✓ Data shape: {data.shape}")
    logger.log(f"  Features: {data.shape[1]}")

    # Split into train/val/test
    train_data, val_data, test_data = split_train_test(data, train_ratio=0.7, val_ratio=0.15)
    logger.log(f"✓ Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")

    # Create environments
    logger.log("\n🌍 Creating environments...")
    train_env = TradingEnv(
        data=train_data,
        initial_balance=initial_balance,
        max_leverage=max_leverage,
        window_size=window_size,
        reward_type=reward_type
    )
    val_env = TradingEnv(
        data=val_data,
        initial_balance=initial_balance,
        max_leverage=max_leverage,
        window_size=window_size,
        reward_type=reward_type
    )
    logger.log(f"✓ Train env: {train_env.n_steps} steps")
    logger.log(f"✓ Val env: {val_env.n_steps} steps")

    # Initialize agent
    logger.log("\n🤖 Initializing D3QN agent...")
    state_dim = train_env.observation_space.shape[0] * train_env.observation_space.shape[1]  # Flatten
    action_dim = train_env.action_space.n

    agent = D3QNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size
    )

    logger.log(f"✓ State dim: {state_dim}")
    logger.log(f"✓ Action dim: {action_dim} (HOLD, LONG, SHORT, CLOSE)")
    logger.log(f"✓ Device: {agent.device}")

    # Training loop
    logger.log("\n🚀 Starting training...")
    logger.log("=" * 60)

    best_val_reward = -float('inf')
    episode_rewards = []

    for episode in range(1, episodes + 1):
        # Reset environment
        state, info = train_env.reset()
        state = state.flatten()  # Flatten state for agent

        episode_reward = 0
        episode_losses = []
        step = 0

        # Episode loop
        done = False
        while not done and step < max_steps:
            # Select action
            action = agent.select_action(state, training=True)

            # Take step
            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            next_state = next_state.flatten()

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Update agent
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

            # Update state
            state = next_state
            episode_reward += reward
            step += 1

        # Episode statistics
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        agent_stats = agent.get_stats()

        # Log episode
        if episode % 10 == 0:
            avg_reward_10 = np.mean(episode_rewards[-10:])
            logger.log(
                f"Episode {episode:4d} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Avg(10): {avg_reward_10:8.2f} | "
                f"Loss: {avg_loss:.4f} | "
                f"ε: {agent_stats['epsilon']:.3f} | "
                f"Steps: {step:4d} | "
                f"Trades: {info['total_trades']:2d} | "
                f"Win%: {info['win_rate']*100:5.1f} | "
                f"Balance: ${info['balance']:,.0f}"
            )

        # Save metrics
        logger.add_episode(episode, {
            'rewards': episode_reward,
            'losses': avg_loss,
            'epsilon': agent_stats['epsilon'],
            'steps': step,
            'trades': info['total_trades'],
            'win_rate': info['win_rate'],
            'final_balance': info['balance']
        })

        # Evaluate on validation set
        if episode % eval_freq == 0:
            logger.log("\n📈 Evaluating on validation set...")
            val_reward, val_info = evaluate_agent(agent, val_env, episodes=5)
            logger.log(
                f"  Val Reward: {val_reward:.2f} | "
                f"Trades: {val_info['trades']} | "
                f"Win%: {val_info['win_rate']*100:.1f} | "
                f"Balance: ${val_info['balance']:,.0f}"
            )

            # Save best model
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                best_path = checkpoint_path / "best_agent.pth"
                agent.save(str(best_path))
                logger.log(f"  💾 New best model saved! (reward: {val_reward:.2f})")

        # Save checkpoint
        if episode % save_freq == 0:
            checkpoint_file = checkpoint_path / f"agent_episode_{episode}.pth"
            agent.save(str(checkpoint_file))
            logger.log(f"\n💾 Checkpoint saved: {checkpoint_file}")

            # Save metrics
            logger.save_metrics()

    # Final evaluation
    logger.log("\n" + "=" * 60)
    logger.log("🎉 Training complete!")
    logger.log("=" * 60)

    # Final test
    logger.log("\n🧪 Final evaluation on test set...")
    test_env = TradingEnv(
        data=test_data,
        initial_balance=initial_balance,
        max_leverage=max_leverage,
        window_size=window_size,
        reward_type=reward_type
    )

    test_reward, test_info = evaluate_agent(agent, test_env, episodes=10)
    logger.log(
        f"Test Reward: {test_reward:.2f} | "
        f"Trades: {test_info['trades']} | "
        f"Win%: {test_info['win_rate']*100:.1f} | "
        f"Balance: ${test_info['balance']:,.0f}"
    )

    # Save final model
    final_path = checkpoint_path / "final_agent.pth"
    agent.save(str(final_path))
    logger.log(f"\n💾 Final model saved: {final_path}")

    # Save metrics
    logger.save_metrics()
    logger.log(f"📊 Metrics saved: {logger.metrics_file}")
    logger.log(f"📝 Logs saved: {logger.log_file}")

    logger.log("\n✅ Done!")


def evaluate_agent(agent: D3QNAgent, env: TradingEnv, episodes: int = 5) -> tuple:
    """
    Evaluate agent on environment

    Args:
        agent: D3QN agent
        env: Trading environment
        episodes: Number of episodes to evaluate

    Returns:
        (average_reward, info_dict)
    """
    total_reward = 0
    total_trades = 0
    total_win_rates = []
    final_balances = []

    for _ in range(episodes):
        state, info = env.reset()
        state = state.flatten()

        episode_reward = 0
        done = False

        while not done:
            # Select greedy action (no exploration)
            action = agent.select_action(state, training=False)

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = next_state.flatten()

            state = next_state
            episode_reward += reward

        total_reward += episode_reward
        total_trades += info['total_trades']
        total_win_rates.append(info['win_rate'])
        final_balances.append(info['balance'])

    avg_reward = total_reward / episodes
    avg_trades = total_trades / episodes
    avg_win_rate = np.mean(total_win_rates) if total_win_rates else 0.0
    avg_balance = np.mean(final_balances)

    return avg_reward, {
        'trades': avg_trades,
        'win_rate': avg_win_rate,
        'balance': avg_balance
    }


def main():
    """Main entry point"""
    try:
        parser = argparse.ArgumentParser(description="Train D3QN agent for crypto futures trading")

        # Data parameters
        parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
        parser.add_argument('--timeframe', type=str, default='1h', help='Candle timeframe')
        parser.add_argument('--sample-data', action='store_true', help='Use synthetic sample data')

        # Training parameters
        parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
        parser.add_argument('--max-steps', type=int, default=10000, help='Max steps per episode')
        parser.add_argument('--eval-freq', type=int, default=50, help='Evaluation frequency')
        parser.add_argument('--save-freq', type=int, default=100, help='Checkpoint save frequency')

        # Trading parameters
        parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
        parser.add_argument('--leverage', type=float, default=3.0, help='Max leverage')
        parser.add_argument('--window', type=int, default=72, help='Observation window size')
        parser.add_argument('--reward', type=str, default='sharpe', choices=['profit', 'return', 'sharpe', 'sortino'], help='Reward type')

        # Agent parameters
        parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')
        parser.add_argument('--gamma', type=float, default=0.995, help='Discount factor')
        parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
        parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')

        # Paths
        parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
        parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')

        args = parser.parse_args()

        # Train
        train_d3qn(
            symbol=args.symbol,
            timeframe=args.timeframe,
            episodes=args.episodes,
            max_steps=args.max_steps,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            use_sample_data=args.sample_data,
            initial_balance=args.balance,
            max_leverage=args.leverage,
            learning_rate=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            window_size=args.window,
            reward_type=args.reward,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir
        )
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: Training failed!")
        print(f"Error message: {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
