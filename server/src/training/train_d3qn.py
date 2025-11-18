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
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Debug: Print Python path and imports
print(f"[DEBUG] Python executable: {sys.executable}")
print(f"[DEBUG] Python path: {sys.path[:3]}...")
print(f"[DEBUG] Current working directory: {os.getcwd()}")

try:
    print("[DEBUG] Importing D3QNAgent...")
    from src.agent.d3qn_agent import D3QNAgent
    print("[DEBUG] [OK] D3QNAgent imported successfully")
except Exception as e:
    print(f"[DEBUG] [FAIL] Failed to import D3QNAgent: {e}")
    raise

try:
    print("[DEBUG] Importing TradingEnv...")
    from src.policy.trading_env import TradingEnv
    print("[DEBUG] [OK] TradingEnv imported successfully")
except Exception as e:
    print(f"[DEBUG] [FAIL] Failed to import TradingEnv: {e}")
    raise

try:
    print("[DEBUG] Importing data loader functions...")
    from src.data.loader import load_market_data, prepare_env_data, split_train_test, create_sample_data
    print("[DEBUG] [OK] Data loader functions imported successfully")
except Exception as e:
    print(f"[DEBUG] [FAIL] Failed to import data loader functions: {e}")
    raise

try:
    print("[DEBUG] Importing BacktestEngine...")
    from src.eval.backtest import BacktestEngine
    print("[DEBUG] [OK] BacktestEngine imported successfully")
except Exception as e:
    print(f"[DEBUG] [FAIL] Failed to import BacktestEngine: {e}")
    raise

try:
    print("[DEBUG] Importing DetailedStepLogger...")
    from src.utils.detailed_logger import DetailedStepLogger
    print("[DEBUG] [OK] DetailedStepLogger imported successfully")
except Exception as e:
    print(f"[DEBUG] [FAIL] Failed to import DetailedStepLogger: {e}")
    raise

print("[DEBUG] All imports successful!")


class TrainingLogger:
    """Simple logger for training metrics"""

    def __init__(self, log_dir: str | Path = "logs"):
        # Get project root (parent of server/)
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        log_dir_path = Path(log_dir)
        if not log_dir_path.is_absolute():
            log_dir_path = project_root / log_dir_path
        self.log_dir = log_dir_path
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{self.timestamp}.log"
        self.metrics_file = self.log_dir / f"metrics_{self.timestamp}.json"

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
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    def add_episode(self, episode: int, metrics: Dict):
        """Add episode metrics"""
        self.metrics['episodes'].append(episode)
        for key in ['rewards', 'losses', 'epsilon', 'steps', 'trades', 'win_rate', 'final_balance']:
            if key in metrics:
                self.metrics[key].append(metrics[key])

    def save_metrics(self):
        """Save metrics to JSON"""
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

    def save_trade_log(self, trades: List[Dict], label: str = "trades"):
        """Persist trade history to JSON"""
        if not trades:
            return

        import numpy as np

        def convert(value):
            if isinstance(value, (np.integer, int)):
                return int(value)
            if isinstance(value, (np.floating, float)):
                return float(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value

        serialized = [{k: convert(v) for k, v in trade.items()} for trade in trades]
        trade_file = self.log_dir / f"{label}_{self.timestamp}.json"
        with open(trade_file, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2)
        self.log(f"[INFO] Trade log saved: {trade_file}")


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
    log_path = Path(log_dir) / "training"
    logger = TrainingLogger(log_path)
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

    # Create checkpoint directory (relative to project root, not cwd)
    # Get project root (parent of server/)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    checkpoint_path = project_root / checkpoint_dir
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    logger.log(f"[INFO] Checkpoints will be saved to: {checkpoint_path}")

    # Load market data
    logger.log("\n[DATA] Loading market data...")
    try:
        if use_sample_data:
            logger.log("[WARN] Using synthetic data for testing")
            df = create_sample_data(symbol=symbol, n_samples=10000, save_dir="data")
        else:
            df = load_market_data(symbol=symbol, timeframe=timeframe)

        logger.log(f"[OK] Loaded {len(df):,} candles")
        logger.log(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    except FileNotFoundError as e:
        logger.log(f"[ERROR] {e}")
        logger.log("\n[INFO] To download real market data:")
        logger.log("   1. Start the server: uvicorn server.main:app")
        logger.log("   2. Download data via API:")
        logger.log(f"      POST /api/data/download")
        logger.log(f"      {{'symbol': '{symbol}', 'timeframe': '{timeframe}', 'start': '2023-01-01', 'end': '2024-12-31'}}")
        logger.log("\n   Or use --sample-data flag to train with synthetic data")
        return

    # Prepare data
    logger.log("\n[PREP] Preparing data...")
    data = prepare_env_data(df, add_indicators=True, normalize=True)
    logger.log(f"[OK] Data shape: {data.shape}")
    logger.log(f"  Features: {data.shape[1]}")

    # Split into train/val/test
    train_data, val_data, test_data = split_train_test(data, train_ratio=0.7, val_ratio=0.15)
    logger.log(f"[OK] Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")

    # Create environments
    logger.log("\n[ENV] Creating environments...")
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
    logger.log(f"[OK] Train env: {train_env.n_steps} steps")
    logger.log(f"[OK] Val env: {val_env.n_steps} steps")

    # Initialize agent
    logger.log("\n[AGENT] Initializing D3QN agent...")
    window_size = train_env.observation_space.shape[0]
    n_features = train_env.observation_space.shape[1]
    state_dim = window_size * n_features  # Flatten
    action_dim = train_env.action_space.n

    agent = D3QNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        window_size=window_size,
        n_features=n_features,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        use_hybrid_network=True  # Enable hybrid CNN-LSTM-Attention architecture
    )

    logger.log(f"[OK] State dim: {state_dim}")
    logger.log(f"[OK] Window size: {window_size}, Features: {n_features}")
    logger.log(f"[OK] Action dim: {action_dim} (HOLD, LONG, SHORT, CLOSE)")
    logger.log(f"[OK] Using hybrid CNN-BiLSTM-Attention-QR-DQN architecture")
    logger.log(f"[OK] Device: {agent.device}")

    # Training loop
    logger.log("\n[TRAIN] Starting training...")
    logger.log("=" * 60)

    best_val_reward = -float('inf')
    episode_rewards = []

    # Initialize detailed logger (logs EVERY step)
    detailed_logger = DetailedStepLogger(log_dir="logs", mode="train")
    logger.log(f"[INFO] Detailed logs will be saved to: logs/train/")

    for episode in range(1, episodes + 1):
        # Enable detailed environment logging for specific episodes
        # Log first 2 episodes, then every 25th episode, and last 2 episodes
        enable_env_logging = (episode <= 2 or episode >= episodes - 1 or episode % 25 == 0)
        if enable_env_logging:
            logger.log(f"\n[DEBUG] Episode {episode} - Enabling detailed environment logging")
            train_env.enable_detailed_logging = True
        else:
            train_env.enable_detailed_logging = False

        # Reset environment
        state, info = train_env.reset()
        state = state.flatten()  # Flatten state for agent

        episode_reward = 0
        episode_losses = []
        epsilon_start = agent.epsilon
        step = 0

        # Episode tracking for detailed logging
        episode_actions = {'HOLD': 0, 'LONG': 0, 'SHORT': 0}
        episode_trades_start = len(train_env.trade_log) if hasattr(train_env, 'trade_log') else 0

        # Episode loop
        done = False
        while not done and step < max_steps:
            # Get current state info from environment
            current_price = train_env.data[train_env.current_step, 3]  # Close price

            # Get Q-values for logging
            q_values = agent.get_q_values(state)

            # Select action
            action = agent.select_action(state, training=True)
            action_names = ['HOLD', 'LONG', 'SHORT']
            episode_actions[action_names[action]] += 1

            # Determine why this action was chosen
            if random.random() < agent.epsilon:
                reason = f"Exploration (eps={agent.epsilon:.3f})"
            else:
                reason = f"Exploitation (Q={q_values[action]:.4f} best)"

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

            # Calculate ATR for logging
            atr = train_env._calculate_atr() if hasattr(train_env, '_calculate_atr') else 0.0

            # Detailed step logging
            step_data = {
                'episode': episode,
                'step': step,
                'global_step': train_env.current_step - train_env.window_size,
                'price': current_price,
                'action': action,
                'action_name': action_names[action],
                'q_values': q_values.tolist(),
                'epsilon': agent.epsilon,
                'position': train_env.position,
                'position_size': train_env.position_size if train_env.position else 0.0,
                'entry_price': train_env.entry_price if train_env.position else 0.0,
                'unrealized_pnl': train_env._calculate_position_pnl() if train_env.position else 0.0,
                'unrealized_pnl_pct': (train_env._calculate_position_pnl() / train_env.balance * 100) if train_env.position and train_env.balance > 0 else 0.0,
                'tp': train_env.take_profit if train_env.position else 0.0,
                'sl': train_env.stop_loss if train_env.position else 0.0,
                'atr': atr,
                'balance': train_env.balance,
                'equity': train_env.equity,
                'trade_reward': 0.0,  # Will be updated if trade closes
                'exploration_bonus': 0.0,
                'position_penalty': 0.0,
                'total_reward': reward,
                'loss': loss if loss is not None else 0.0,
                'reason': reason,
                'terminated': terminated,
                'truncated': truncated
            }
            detailed_logger.log_step(step_data)

            # Update state
            state = next_state
            episode_reward += reward
            step += 1

        # Log all trades from this episode
        if hasattr(train_env, 'trade_log'):
            episode_trades_end = len(train_env.trade_log)
            for trade in train_env.trade_log[episode_trades_start:episode_trades_end]:
                trade_data = {
                    'episode': episode,
                    'trade_id': trade.get('position_id', 'unknown'),
                    'direction': trade['direction'],
                    'entry_step': trade['entry_step'],
                    'entry_price': trade['entry_price'],
                    'exit_step': trade['exit_step'],
                    'exit_price': trade['exit_price'],
                    'exit_reason': 'tp' if trade['pnl'] > 0 else 'sl',  # Simplified
                    'duration_steps': trade['duration_steps'],
                    'size': trade['quantity'],
                    'leverage': trade['leverage'],
                    'atr_at_entry': 0.0,  # Would need to store this
                    'tp_price': 0.0,  # Would need to store this
                    'sl_price': 0.0,  # Would need to store this
                    'gross_pnl': trade['pnl'],
                    'fees': 0.0,  # Included in pnl
                    'net_pnl': trade['pnl'],
                    'pnl_pct_vs_margin': trade['pnl_pct'],
                    'pnl_pct_vs_notional': trade['pnl_pct'] / trade['leverage'] if trade['leverage'] > 0 else 0.0,
                    'balance_before': 0.0,  # Would need to track
                    'balance_after': 0.0  # Would need to track
                }
                detailed_logger.log_trade(trade_data)

        # Episode statistics
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        agent_stats = agent.get_stats()

        # Calculate episode statistics for detailed logger
        episode_trades = train_env.trade_log[episode_trades_start:] if hasattr(train_env, 'trade_log') else []
        winning_trades = sum(1 for t in episode_trades if t['pnl'] > 0)
        losing_trades = sum(1 for t in episode_trades if t['pnl'] <= 0)
        win_rate = winning_trades / len(episode_trades) if episode_trades else 0.0
        avg_win = np.mean([t['pnl'] for t in episode_trades if t['pnl'] > 0]) if winning_trades > 0 else 0.0
        avg_loss_trade = np.mean([t['pnl'] for t in episode_trades if t['pnl'] <= 0]) if losing_trades > 0 else 0.0
        rr_ratio = abs(avg_win / avg_loss_trade) if avg_loss_trade != 0 else 0.0

        # Log episode summary
        episode_data = {
            'episode': episode,
            'steps': step,
            'total_reward': episode_reward,
            'initial_balance': initial_balance,
            'final_balance': train_env.balance,
            'return_pct': (train_env.balance - initial_balance) / initial_balance * 100,
            'total_trades': len(episode_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss_trade,
            'rr_ratio': rr_ratio,
            'max_drawdown': 0.0,  # Would need to calculate
            'sharpe_ratio': 0.0,  # Would need to calculate
            'action_distribution': episode_actions,
            'avg_loss': avg_loss,
            'epsilon_start': epsilon_start,
            'epsilon_end': agent.epsilon
        }
        detailed_logger.log_episode_end(episode_data)

        # Standard episode logging (terminal output)
        agent_stats = agent.get_stats()

        # Log episode (every episode for better progress tracking)
        avg_reward_10 = np.mean(episode_rewards[-min(10, len(episode_rewards)):])
        logger.log(
            f"Episode {episode:4d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Avg(10): {avg_reward_10:8.2f} | "
            f"Loss: {avg_loss:.4f} | "
            f"Eps: {agent_stats['epsilon']:.3f} | "  # Changed from epsilon to Eps
            f"Steps: {step:4d} | "
            f"Trades: {info['total_trades']:2d} | "
            f"Win%: {info['win_rate']*100:5.1f} | "
            f"Balance: ${info['balance']:,.0f}"
        )
        import sys
        sys.stdout.flush()  # Force flush to ensure real-time output

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
            logger.log("\n[EVAL] Evaluating on validation set...")
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
                logger.log(f"  [SAVE] New best model saved! (reward: {val_reward:.2f})")

        # Save checkpoint
        if episode % save_freq == 0:
            checkpoint_file = checkpoint_path / f"agent_episode_{episode}.pth"
            agent.save(str(checkpoint_file))
            logger.log(f"\n[SAVE] Checkpoint saved: {checkpoint_file}")

            # Save metrics
            logger.save_metrics()

    # Final evaluation
    logger.log("\n" + "=" * 60)
    logger.log("[DONE] Training complete!")
    logger.log("=" * 60)

    # Final test
    logger.log("\n[TEST] Final evaluation on test set...")
    logger.log("[DEBUG] Enabling detailed logging for test episodes...")
    test_env = TradingEnv(
        data=test_data,
        initial_balance=initial_balance,
        max_leverage=max_leverage,
        window_size=window_size,
        reward_type=reward_type
    )
    test_env.enable_detailed_logging = True

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
    logger.log(f"\n[SAVE] Final model saved: {final_path}")

    # Save metrics
    logger.save_metrics()
    logger.save_trade_log(train_env.trade_log, "train_trades")
    logger.save_trade_log(test_env.trade_log, "test_trades")
    logger.log(f"[INFO] Metrics saved: {logger.metrics_file}")
    logger.log(f"[INFO] Logs saved: {logger.log_file}")
    logger.log(f"\n[INFO] Detailed episode logs saved to: {project_root / 'logs'}")
    logger.log(f"  - episode_XXXX_*.jsonl (step-by-step data in JSON Lines format)")
    logger.log(f"  - episode_XXXX_summary.txt (human-readable summary)")

    # Close detailed logger
    detailed_logger.close()
    logger.log(f"\n[INFO] Detailed training logs closed and saved to: logs/training/detailed/")

    logger.log("\n[COMPLETE] Done!")


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
        print(f"\n[ERROR] Training failed!")
        print(f"Error message: {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
