"""
Training Script for SAC (Soft Actor-Critic) Agent

This script trains a Soft Actor-Critic agent using Stable-Baselines3
for cryptocurrency futures trading.

SAC is an off-policy algorithm that excels at continuous action spaces
and has shown excellent performance in trading tasks.

Usage:
    python -m server.src.training.train_sac --symbol BTCUSDT --timesteps 100000
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.policy.trading_env import TradingEnv
from src.policy.wrappers import ContinuousActionWrapper
from src.data.loader import load_market_data, prepare_env_data, split_train_test, create_sample_data


class TradingCallback(BaseCallback):
    """
    Custom callback for tracking training metrics and progress
    """

    def __init__(self, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

        # Metrics file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.log_dir / f"sac_metrics_{timestamp}.json"

        self.metrics = {
            "timesteps": [],
            "episode_rewards": [],
            "episode_lengths": [],
            "loss": [],
            "learning_rate": []
        }

    def _on_step(self) -> bool:
        """Called at each step"""
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout"""
        # Log training metrics
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]

            if 'r' in ep_info and 'l' in ep_info:
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                self.episode_count += 1

                # Log to console
                if self.episode_count % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])

                    print(f"Episode {self.episode_count:4d} | "
                          f"Timestep {self.num_timesteps:8d} | "
                          f"Reward: {ep_info['r']:8.2f} | "
                          f"Avg(10): {avg_reward:8.2f} | "
                          f"Length: {int(avg_length):4d}")

                # Save metrics
                self.metrics["timesteps"].append(int(self.num_timesteps))
                self.metrics["episode_rewards"].append(float(ep_info['r']))
                self.metrics["episode_lengths"].append(int(ep_info['l']))

                # Save to file every 50 episodes
                if self.episode_count % 50 == 0:
                    self._save_metrics()

    def _save_metrics(self):
        """Save metrics to JSON file"""
        import numpy as np

        # Convert numpy types to Python native types
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        metrics_native = convert_to_native(self.metrics)

        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_native, f, indent=2)


def train_sac(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    timesteps: int = 100000,
    use_sample_data: bool = False,
    initial_balance: float = 10000.0,
    max_leverage: float = 3.0,
    learning_rate: float = 0.0003,
    gamma: float = 0.995,
    buffer_size: int = 100000,
    learning_starts: int = 1000,
    batch_size: int = 256,
    tau: float = 0.005,
    window_size: int = 72,
    reward_type: str = 'sharpe',
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    save_freq: int = 10000,
    verbose: int = 1
):
    """
    Train SAC agent

    Args:
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        timesteps: Total training timesteps
        use_sample_data: Use synthetic data for testing
        initial_balance: Starting capital
        max_leverage: Maximum leverage
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        buffer_size: Replay buffer size
        learning_starts: Steps before training starts
        batch_size: Batch size for training
        tau: Soft update coefficient
        window_size: Observation window size
        reward_type: Reward calculation method
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for logs
        eval_freq: Evaluate every N timesteps
        n_eval_episodes: Number of eval episodes
        save_freq: Save checkpoint every N timesteps
        verbose: Verbosity level
    """
    print("=" * 60)
    print("SAC Crypto Futures Trading - Training")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Max Leverage: {max_leverage}x")
    print(f"Learning Rate: {learning_rate}")
    print(f"Reward Type: {reward_type}")
    print("=" * 60)

    # Create directories
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)

    # Load market data
    print("\n[DATA] Loading market data...")
    try:
        if use_sample_data:
            print("[WARN]  Using synthetic data for testing")
            df = create_sample_data(symbol=symbol, n_samples=10000, save_dir="data")
        else:
            df = load_market_data(symbol=symbol, timeframe=timeframe)

        print(f"[OK] Loaded {len(df):,} candles")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    except FileNotFoundError as e:
        print(f"[ERROR] Error: {e}")
        print("\n[INFO] Use --sample-data flag to train with synthetic data")
        return

    # Prepare data
    print("\n[PREP] Preparing data...")
    data = prepare_env_data(df, add_indicators=True, normalize=True)
    print(f"[OK] Data shape: {data.shape}")

    # Split into train/val/test
    train_data, val_data, test_data = split_train_test(data, train_ratio=0.7, val_ratio=0.15)
    print(f"[OK] Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")

    # Create environments
    print("\n[ENV] Creating environments...")

    def make_env(data):
        """Create environment"""
        env = TradingEnv(
            data=data,
            initial_balance=initial_balance,
            max_leverage=max_leverage,
            window_size=window_size,
            reward_type=reward_type
        )
        env = ContinuousActionWrapper(env)
        env = Monitor(env)
        return env

    # Training environment
    train_env = DummyVecEnv([lambda: make_env(train_data)])

    # Evaluation environment
    eval_env = DummyVecEnv([lambda: make_env(val_data)])

    print(f"[OK] Environments created")

    # Initialize SAC agent
    print("\n[AGENT] Initializing SAC agent...")

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=1,
        gradient_steps=1,
        verbose=verbose,
        tensorboard_log=None  # Disable tensorboard to avoid dependency issues
    )

    print(f"[OK] SAC agent initialized")
    print(f"  Policy: MlpPolicy")
    print(f"  Buffer size: {buffer_size:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning starts: {learning_starts:,}")

    # Setup callbacks
    print("\n[DATA] Setting up callbacks...")

    callbacks = []

    # Custom trading callback
    trading_callback = TradingCallback(log_dir=log_dir, verbose=verbose)
    callbacks.append(trading_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoint_path),
        name_prefix="sac_model",
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_path / "best"),
        log_path=str(log_path),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)

    print(f"[OK] Callbacks ready")

    # Training
    print("\n[TRAIN] Starting training...")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=False  # Disable to avoid tqdm dependency
        )

        print("\n" + "=" * 60)
        print("[DONE] Training complete!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n[WARN]  Training interrupted by user")

    # Save final model
    final_model_path = checkpoint_path / "final_sac_model.zip"
    model.save(str(final_model_path))
    print(f"\n[SAVE] Final model saved: {final_model_path}")

    # Save training metrics
    trading_callback._save_metrics()
    print(f"[DATA] Metrics saved: {trading_callback.metrics_file}")

    # Evaluate on test set
    print("\n[TEST] Final evaluation on test set...")
    test_env = DummyVecEnv([lambda: make_env(test_data)])

    obs = test_env.reset()
    total_reward = 0
    n_episodes = 0
    episode_rewards = []

    for _ in range(10):  # 10 test episodes
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            episode_reward += reward[0]

            if done[0]:
                episode_rewards.append(episode_reward)
                n_episodes += 1
                total_reward += episode_reward
                obs = test_env.reset()
                break

    avg_reward = total_reward / n_episodes if n_episodes > 0 else 0
    print(f"Test Episodes: {n_episodes}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Std Reward: {np.std(episode_rewards):.2f}")

    print("\n[COMPLETE] Done!")

    return model


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train SAC agent for crypto futures trading")

    # Data parameters
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--timeframe', type=str, default='1h', help='Candle timeframe')
    parser.add_argument('--sample-data', action='store_true', help='Use synthetic sample data')

    # Training parameters
    parser.add_argument('--timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--eval-freq', type=int, default=10000, help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=10000, help='Checkpoint save frequency')
    parser.add_argument('--n-eval-episodes', type=int, default=5, help='Number of eval episodes')

    # Trading parameters
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--leverage', type=float, default=3.0, help='Max leverage')
    parser.add_argument('--window', type=int, default=72, help='Observation window size')
    parser.add_argument('--reward', type=str, default='sharpe',
                        choices=['profit', 'return', 'sharpe', 'sortino'], help='Reward type')

    # SAC parameters
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.995, help='Discount factor')
    parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning-starts', type=int, default=1000, help='Steps before training')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient')

    # Paths
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')

    # Other
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')

    args = parser.parse_args()

    # Train
    train_sac(
        symbol=args.symbol,
        timeframe=args.timeframe,
        timesteps=args.timesteps,
        use_sample_data=args.sample_data,
        initial_balance=args.balance,
        max_leverage=args.leverage,
        learning_rate=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        window_size=args.window,
        reward_type=args.reward,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_freq=args.save_freq,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
