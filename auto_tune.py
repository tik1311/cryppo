#!/usr/bin/env python3
"""
CRYPPO - Auto Parameter Tuning System
Automatically finds optimal hyperparameters for maximum profitability
Uses Optuna for Bayesian optimization
"""

import sys
import os
import optuna
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import warnings
warnings.filterwarnings('ignore')

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

from src.data.loader import load_multi_timeframe_data, prepare_multi_timeframe_env_data
from src.policy.trading_env import TradingEnv
from src.agent.d3qn_agent import D3QNAgent

console = Console()


class ParameterTuner:
    """
    Auto-tune hyperparameters for maximum profitability

    Search Space:
    - window_size: [50, 60, 72, 100, 120]
    - learning_rate: [0.0001, 0.001] (log scale)
    - epsilon_min: [0.05, 0.20]
    - epsilon_decay: [0.995, 0.9999]
    - batch_size: [16, 32, 64]
    - atr_tp_multiplier: [2.0, 5.0]
    - atr_sl_multiplier: [1.0, 2.5]
    - min_bars_between_trades: [2, 5]
    - update_frequency: [1, 2, 4, 8]
    - gamma (discount): [0.95, 0.99]
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        main_tf: str = "1h",
        detail_tf: str = "1m",
        train_episodes: int = 5,  # Short training for tuning
        val_episodes: int = 1,    # Validation episodes
        n_trials: int = 100,      # Number of parameter combinations to try
        use_full_data: bool = True  # Use ALL data (not just 3 months)
    ):
        self.symbol = symbol
        self.main_tf = main_tf
        self.detail_tf = detail_tf
        self.train_episodes = train_episodes
        self.val_episodes = val_episodes
        self.n_trials = n_trials
        self.use_full_data = use_full_data

        # Results storage
        self.best_params = None
        self.best_score = -float('inf')
        self.results_history = []

        # Load data once
        console.print("[cyan]Loading market data...[/cyan]")
        self._load_data()

    def _load_data(self):
        """Load and split data into train/val/test"""
        # Load FULL data (no date limits!)
        df_main, df_detail = load_multi_timeframe_data(
            symbol=self.symbol,
            main_timeframe=self.main_tf,
            detail_timeframe=self.detail_tf,
            start_date=None if self.use_full_data else None,
            end_date=None if self.use_full_data else None
        )

        # Prepare data
        data_dict = prepare_multi_timeframe_env_data(
            df_main, df_detail,
            main_timeframe=self.main_tf,
            detail_timeframe=self.detail_tf
        )

        # Split: 70% train, 15% val, 15% test
        n = len(data_dict['main'])
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        self.train_data = {
            'main': data_dict['main'][:train_end],
            'detail': data_dict['detail'][:train_end]
        }

        self.val_data = {
            'main': data_dict['main'][train_end:val_end],
            'detail': data_dict['detail'][train_end:val_end]
        }

        self.test_data = {
            'main': data_dict['main'][val_end:],
            'detail': data_dict['detail'][val_end:]
        }

        console.print(f"[green]✓[/green] Data loaded:")
        console.print(f"  Train: {len(self.train_data['main'])} steps")
        console.print(f"  Val:   {len(self.val_data['main'])} steps")
        console.print(f"  Test:  {len(self.test_data['main'])} steps")

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function
        Returns: Sharpe ratio * win_rate * total_return (composite score)
        """
        # Sample hyperparameters
        params = {
            # Environment parameters
            'window_size': trial.suggest_categorical('window_size', [50, 60, 72, 100, 120]),
            'atr_tp_multiplier': trial.suggest_float('atr_tp_multiplier', 2.0, 5.0),
            'atr_sl_multiplier': trial.suggest_float('atr_sl_multiplier', 1.0, 2.5),
            'min_bars_between_trades': trial.suggest_int('min_bars_between_trades', 2, 5),

            # Agent parameters
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
            'gamma': trial.suggest_float('gamma', 0.95, 0.99),
            'epsilon_min': trial.suggest_float('epsilon_min', 0.05, 0.20),
            'epsilon_decay': trial.suggest_float('epsilon_decay', 0.995, 0.9999),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'update_frequency': trial.suggest_categorical('update_frequency', [1, 2, 4, 8]),

            # Reward shaping
            'tp_reward': trial.suggest_float('tp_reward', 0.5, 2.0),
            'sl_penalty': trial.suggest_float('sl_penalty', 0.3, 1.0),
            'win_streak_bonus': trial.suggest_float('win_streak_bonus', 0.1, 0.5),
        }

        try:
            # Train on train set
            train_metrics = self._train_and_evaluate(
                params,
                self.train_data,
                episodes=self.train_episodes,
                mode='train'
            )

            # Validate on val set
            val_metrics = self._train_and_evaluate(
                params,
                self.val_data,
                episodes=self.val_episodes,
                mode='val',
                load_weights=True
            )

            # Calculate composite score (maximize profitability + consistency)
            score = self._calculate_score(val_metrics)

            # Store results
            self.results_history.append({
                'trial': trial.number,
                'params': params,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'score': score
            })

            # Print progress
            console.print(f"[yellow]Trial {trial.number}[/yellow] | Score: [cyan]{score:.4f}[/cyan] | "
                         f"Val Return: [green]{val_metrics['total_return']:.2f}%[/green] | "
                         f"WR: {val_metrics['win_rate']:.1f}%")

            return score

        except Exception as e:
            console.print(f"[red]Trial {trial.number} failed: {e}[/red]")
            return -1000.0  # Large penalty for failed trials

    def _train_and_evaluate(
        self,
        params: Dict[str, Any],
        data: Dict,
        episodes: int,
        mode: str = 'train',
        load_weights: bool = False
    ) -> Dict[str, float]:
        """Train agent and return metrics"""

        # Create environment with tuned parameters
        # Disable bust protection for training trials (to allow full episode completion)
        env = TradingEnv(
            data,
            initial_balance=10000.0,
            window_size=params['window_size'],
            atr_tp_multiplier=params['atr_tp_multiplier'],
            atr_sl_multiplier=params['atr_sl_multiplier'],
            enable_bust_protection=(mode == 'val')  # Only enable for validation
        )

        # Temporarily modify min_bars_between_trades
        env.min_bars_between_trades = params['min_bars_between_trades']

        # Create agent with tuned parameters
        obs_shape = env.observation_space['main'].shape
        detail_shape = env.observation_space['detail'].shape

        agent = D3QNAgent(
            window_size=obs_shape[0],
            n_main_features=obs_shape[1],
            n_detail_features=detail_shape[2],
            detail_len=detail_shape[1],
            action_dim=env.action_space.n,
            learning_rate=params['learning_rate'],
            gamma=params['gamma']
        )

        # Override agent parameters
        agent.epsilon_min = params['epsilon_min']
        agent.epsilon_decay = params['epsilon_decay']
        agent.batch_size = params['batch_size']

        if load_weights and os.path.exists('.tune_checkpoint.pth'):
            agent.load('.tune_checkpoint.pth')

        # Training/Validation loop
        all_returns = []
        all_win_rates = []

        for episode in range(episodes):
            state, _ = env.reset()
            done = False

            while not done:
                if mode == 'train':
                    # Training: explore and learn
                    mask = env.get_action_mask()
                    action = agent.select_action(state, action_mask=mask)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    # Modify reward based on tuned parameters
                    if env.position is None and reward > 0:  # TP hit
                        reward = reward * params['tp_reward']
                    elif env.position is None and reward < 0:  # SL hit
                        reward = reward * params['sl_penalty']

                    agent.store_transition(state, action, reward, next_state, done)

                    # Update based on tuned frequency
                    if agent.steps % params['update_frequency'] == 0:
                        agent.update()

                    state = next_state
                else:
                    # Validation: exploit only (no exploration, no learning)
                    agent.epsilon = 0.0
                    mask = env.get_action_mask()
                    action = agent.select_action(state, training=False, action_mask=mask)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    state = next_state

            # Episode metrics
            total_return = ((env.balance - env.initial_balance) / env.initial_balance) * 100
            win_rate = (env.winning_trades / max(1, env.total_trades)) * 100

            all_returns.append(total_return)
            all_win_rates.append(win_rate)

        # Save weights after training
        if mode == 'train':
            agent.save('.tune_checkpoint.pth')

        # Calculate aggregate metrics
        avg_return = np.mean(all_returns)
        avg_win_rate = np.mean(all_win_rates)
        sharpe = np.mean(all_returns) / (np.std(all_returns) + 1e-6)

        final_balance = env.balance
        total_trades = env.total_trades

        return {
            'total_return': avg_return,
            'win_rate': avg_win_rate,
            'sharpe_ratio': sharpe,
            'final_balance': final_balance,
            'total_trades': total_trades
        }

    def _calculate_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate composite score for optimization
        Prioritize: Positive return + High win rate + Consistency
        """
        total_return = metrics['total_return']
        win_rate = metrics['win_rate']
        sharpe = metrics['sharpe_ratio']
        trades = metrics['total_trades']

        # Penalty for no trades
        if trades == 0:
            return -1000.0

        # Penalty for too few trades
        if trades < 10:
            return -500.0

        # Primary: Total return (must be positive)
        if total_return < 0:
            return total_return  # Negative score for losing strategies

        # Composite score (weighted)
        score = (
            total_return * 1.0 +      # Weight: 1.0 (most important)
            win_rate * 0.5 +          # Weight: 0.5
            sharpe * 10.0 +           # Weight: 10.0 (consistency)
            np.log(trades) * 5.0      # Weight: 5.0 (encourage trading activity)
        )

        return score

    def run_optimization(self):
        """Run Optuna optimization"""
        console.print("\n[bold cyan]Starting Parameter Optimization...[/bold cyan]")
        console.print(f"Trials: {self.n_trials} | Train Episodes: {self.train_episodes}")

        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        # Get best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value

        console.print("\n[bold green]✓ Optimization Complete![/bold green]")
        self._print_best_params()

        # Save results
        self._save_results(study)

        return self.best_params

    def _print_best_params(self):
        """Print best parameters in a nice table"""
        table = Table(title="🏆 Best Parameters Found")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")

        for key, value in self.best_params.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.6f}")
            else:
                table.add_row(key, str(value))

        console.print(table)
        console.print(f"\n[green]Best Score: {self.best_score:.4f}[/green]")

    def _save_results(self, study):
        """Save optimization results to file"""
        results_dir = Path("tuning_results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save best parameters
        best_params_file = results_dir / f"best_params_{timestamp}.json"
        with open(best_params_file, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'best_score': self.best_score,
                'n_trials': self.n_trials,
                'symbol': self.symbol,
                'timestamp': timestamp
            }, f, indent=2)

        # Save full history
        history_file = results_dir / f"history_{timestamp}.json"
        with open(history_file, 'w') as f:
            json.dump(self.results_history, f, indent=2)

        # Save Optuna study
        study_file = results_dir / f"study_{timestamp}.db"
        optuna.visualization.plot_optimization_history(study).write_html(
            str(results_dir / f"plot_history_{timestamp}.html")
        )
        optuna.visualization.plot_param_importances(study).write_html(
            str(results_dir / f"plot_importance_{timestamp}.html")
        )

        console.print(f"\n[dim]Results saved to: {results_dir}/[/dim]")

    def test_best_params(self) -> Dict[str, float]:
        """Test best parameters on test set"""
        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimization first.")

        console.print("\n[cyan]Testing best parameters on test set...[/cyan]")

        test_metrics = self._train_and_evaluate(
            self.best_params,
            self.test_data,
            episodes=3,  # Multiple test episodes
            mode='val',
            load_weights=True
        )

        # Print results
        console.print("\n[bold green]Test Results:[/bold green]")
        console.print(f"  Total Return: [cyan]{test_metrics['total_return']:.2f}%[/cyan]")
        console.print(f"  Win Rate: [cyan]{test_metrics['win_rate']:.1f}%[/cyan]")
        console.print(f"  Sharpe Ratio: [cyan]{test_metrics['sharpe_ratio']:.2f}[/cyan]")
        console.print(f"  Total Trades: [cyan]{test_metrics['total_trades']}[/cyan]")
        console.print(f"  Final Balance: [cyan]${test_metrics['final_balance']:.2f}[/cyan]")

        return test_metrics


def main():
    """Main tuning script"""
    import argparse

    parser = argparse.ArgumentParser(description="CRYPPO Auto Parameter Tuning")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--main-tf", default="1h", help="Main timeframe")
    parser.add_argument("--detail-tf", default="1m", help="Detail timeframe")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--train-episodes", type=int, default=5, help="Training episodes per trial")
    parser.add_argument("--full-data", action="store_true", help="Use full data (not just 3 months)")

    args = parser.parse_args()

    # Create tuner
    tuner = ParameterTuner(
        symbol=args.symbol,
        main_tf=args.main_tf,
        detail_tf=args.detail_tf,
        n_trials=args.trials,
        train_episodes=args.train_episodes,
        use_full_data=args.full_data
    )

    # Run optimization
    best_params = tuner.run_optimization()

    # Test on test set
    tuner.test_best_params()

    console.print("\n[bold green]✓ Auto-tuning complete! Check tuning_results/ for details.[/bold green]")


if __name__ == "__main__":
    main()
