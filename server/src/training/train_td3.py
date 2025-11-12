"""
Training Script for TD3 (Twin Delayed DDPG) Agent

TD3 is an off-policy algorithm that improves upon DDPG.
Excellent for continuous control and risk management.

Usage:
    python -m server.src.training.train_td3 --symbol BTCUSDT --timesteps 100000
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from src.policy.trading_env import TradingEnv
from src.data.loader import load_market_data, prepare_env_data, split_train_test, create_sample_data


class TradingCallback(BaseCallback):
    """Custom callback for tracking training metrics"""

    def __init__(self, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.log_dir / f"td3_metrics_{timestamp}.json"

        self.metrics = {
            "timesteps": [],
            "episode_rewards": [],
            "episode_lengths": []
        }

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]

            if 'r' in ep_info and 'l' in ep_info:
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                self.episode_count += 1

                if self.episode_count % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])

                    print(f"Episode {self.episode_count:4d} | "
                          f"Timestep {self.num_timesteps:8d} | "
                          f"Reward: {ep_info['r']:8.2f} | "
                          f"Avg(10): {avg_reward:8.2f} | "
                          f"Length: {int(avg_length):4d}")

                self.metrics["timesteps"].append(int(self.num_timesteps))
                self.metrics["episode_rewards"].append(float(ep_info['r']))
                self.metrics["episode_lengths"].append(int(ep_info['l']))

                if self.episode_count % 50 == 0:
                    self._save_metrics()

    def _save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)


def train_td3(
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
    policy_delay: int = 2,
    target_policy_noise: float = 0.2,
    target_noise_clip: float = 0.5,
    window_size: int = 72,
    reward_type: str = 'sharpe',
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    save_freq: int = 10000,
    verbose: int = 1
):
    """Train TD3 agent"""
    print("=" * 60)
    print("TD3 Crypto Futures Trading - Training")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Buffer Size: {buffer_size:,}")
    print(f"Policy Delay: {policy_delay}")
    print("=" * 60)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)

    print("\n[DATA] Loading market data...")
    try:
        if use_sample_data:
            print("[WARN]  Using synthetic data")
            df = create_sample_data(symbol=symbol, n_samples=10000, save_dir="data")
        else:
            df = load_market_data(symbol=symbol, timeframe=timeframe)
        print(f"[OK] Loaded {len(df):,} candles")
    except FileNotFoundError as e:
        print(f"[ERROR] Error: {e}")
        return

    print("\n[PREP] Preparing data...")
    data = prepare_env_data(df, add_indicators=True, normalize=True)
    train_data, val_data, test_data = split_train_test(data, train_ratio=0.7, val_ratio=0.15)
    print(f"[OK] Train: {len(train_data):,} | Val: {len(val_data):,}")

    print("\n[ENV] Creating environments...")

    def make_env(data):
        env = TradingEnv(
            data=data,
            initial_balance=initial_balance,
            max_leverage=max_leverage,
            window_size=window_size,
            reward_type=reward_type
        )
        return Monitor(env)

    train_env = DummyVecEnv([lambda: make_env(train_data)])
    eval_env = DummyVecEnv([lambda: make_env(val_data)])
    print(f"[OK] Environments created")

    print("\n[AGENT] Initializing TD3 agent...")

    # Add action noise for exploration
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    model = TD3(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        policy_delay=policy_delay,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        action_noise=action_noise,
        verbose=verbose,
        tensorboard_log=str(log_path / "tensorboard")
    )
    print(f"[OK] TD3 agent initialized")

    print("\n[DATA] Setting up callbacks...")
    callbacks = []
    callbacks.append(TradingCallback(log_dir=log_dir, verbose=verbose))
    callbacks.append(CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoint_path),
        name_prefix="td3_model",
        verbose=1
    ))
    callbacks.append(EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_path / "best"),
        log_path=str(log_path),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        verbose=1
    ))
    print(f"[OK] Callbacks ready")

    print("\n[TRAIN] Starting training...")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True
        )
        print("\n[DONE] Training complete!")
    except KeyboardInterrupt:
        print("\n[WARN]  Training interrupted")

    final_model_path = checkpoint_path / "final_td3_model.zip"
    model.save(str(final_model_path))
    print(f"\n[SAVE] Final model saved: {final_model_path}")

    print("\n[COMPLETE] Done!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train TD3 agent")
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--sample-data', action='store_true')
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--balance', type=float, default=10000.0)
    parser.add_argument('--leverage', type=float, default=3.0)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-starts', type=int, default=1000)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--policy-delay', type=int, default=2)
    parser.add_argument('--window', type=int, default=72)
    parser.add_argument('--reward', type=str, default='sharpe')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--eval-freq', type=int, default=10000)
    parser.add_argument('--n-eval-episodes', type=int, default=5)
    parser.add_argument('--save-freq', type=int, default=10000)
    parser.add_argument('--verbose', type=int, default=1)

    args = parser.parse_args()

    train_td3(
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
        policy_delay=args.policy_delay,
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
