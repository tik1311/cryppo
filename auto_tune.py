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
    Auto-tune hyperparameters for maximum profitability (2025 Edition)

    Enhanced Search Space (Based on 2025 Research):

    Environment Parameters:
    - window_size: [36, 50, 72, 100] (shorter for efficiency)
    - atr_tp_multiplier: [2.5, 6.0] (wider range, focus on high R:R)
    - atr_sl_multiplier: [0.8, 1.5] (tighter SL options)
    - min_bars_between_trades: [2, 8] (anti-overtrading)
    - position_size_pct: [0.5, 1.0] (risk per trade)

    Agent Hyperparameters:
    - learning_rate: [0.0001, 0.001] (log scale)
    - gamma: [0.95, 0.99] (discount factor)
    - epsilon_min: [0.05, 0.25] (exploration)
    - epsilon_decay: [0.995, 0.9999] (decay rate)
    - batch_size: [16, 32, 64] (training batch)
    - update_frequency: [1, 2, 4, 8] (update interval)
    - target_update_freq: [1000, 5000, 10000] (target net sync)

    Reward Shaping:
    - reward_type: ['pnl', 'sharpe', 'sortino'] (reward function)
    - tp_reward_multiplier: [1.0, 3.0] (TP bonus)
    - sl_penalty_multiplier: [0.3, 1.5] (SL penalty)
    - hold_penalty_threshold: [20, 60] (consecutive holds)
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
        # Sample hyperparameters (2025 Enhanced Search Space)
        params = {
            # ========== Environment Parameters ==========
            'window_size': trial.suggest_categorical('window_size', [36, 50, 72, 100]),

            # TP/SL Strategy (Focus on High R:R for profitability)
            'atr_tp_multiplier': trial.suggest_float('atr_tp_multiplier', 2.5, 6.0),
            'atr_sl_multiplier': trial.suggest_float('atr_sl_multiplier', 0.8, 1.5),
            # Calculate R:R ratio for logging
            'rr_ratio': None,  # Will be calculated below

            # Risk Management
            'min_bars_between_trades': trial.suggest_int('min_bars_between_trades', 2, 8),
            'position_size_pct': trial.suggest_float('position_size_pct', 0.5, 1.0),

            # ========== Agent Hyperparameters ==========
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
            'gamma': trial.suggest_float('gamma', 0.95, 0.99),
            'epsilon_min': trial.suggest_float('epsilon_min', 0.05, 0.25),
            'epsilon_decay': trial.suggest_float('epsilon_decay', 0.995, 0.9999),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'update_frequency': trial.suggest_categorical('update_frequency', [1, 2, 4, 8]),
            'target_update_freq': trial.suggest_categorical('target_update_freq', [1000, 5000, 10000]),

            # ========== Reward Function (2025 Research) ==========
            'reward_type': trial.suggest_categorical('reward_type', ['pnl', 'sharpe']),
            'tp_reward_multiplier': trial.suggest_float('tp_reward_multiplier', 1.0, 3.0),
            'sl_penalty_multiplier': trial.suggest_float('sl_penalty_multiplier', 0.3, 1.5),
            'hold_penalty_threshold': trial.suggest_int('hold_penalty_threshold', 20, 60),
        }

        # Calculate R:R ratio for analysis
        params['rr_ratio'] = params['atr_tp_multiplier'] / params['atr_sl_multiplier']

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

            # Enhanced progress display with R:R ratio
            console.print(
                f"[yellow]Trial {trial.number}[/yellow] | "
                f"Score: [cyan]{score:.4f}[/cyan] | "
                f"Return: [green]{val_metrics['total_return']:.2f}%[/green] | "
                f"WR: {val_metrics['win_rate']:.1f}% | "
                f"R:R: [magenta]{params['rr_ratio']:.2f}:1[/magenta] | "
                f"Trades: {val_metrics['total_trades']}"
            )

            # Detailed logging to file
            self._log_trial_details(trial.number, params, train_metrics, val_metrics, score)

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

        # Create environment with tuned parameters (2025 Enhanced)
        # Disable bust protection for training trials (to allow full episode completion)
        env = TradingEnv(
            data,
            initial_balance=10000.0,
            window_size=params['window_size'],
            atr_tp_multiplier=params['atr_tp_multiplier'],
            atr_sl_multiplier=params['atr_sl_multiplier'],
            reward_type=params.get('reward_type', 'pnl'),  # Use tuned reward function
            enable_bust_protection=(mode == 'val')  # Only enable for validation
        )

        # Apply risk management parameters
        env.min_bars_between_trades = params['min_bars_between_trades']
        # Note: position_size_pct would need environment modification to support

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

        # Override agent parameters (2025 Enhanced)
        agent.epsilon_min = params['epsilon_min']
        agent.epsilon_decay = params['epsilon_decay']
        agent.batch_size = params['batch_size']
        agent.target_update_freq = params['target_update_freq']

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

                    # Apply reward shaping based on tuned parameters (2025 Enhanced)
                    if env.position is None and reward > 0:  # TP hit
                        reward = reward * params['tp_reward_multiplier']
                    elif env.position is None and reward < 0:  # SL hit
                        reward = reward * params['sl_penalty_multiplier']

                    # Hold penalty based on tuned threshold
                    if env.consecutive_holds > params['hold_penalty_threshold']:
                        hold_penalty = -0.001 * (env.consecutive_holds - params['hold_penalty_threshold'])
                        reward += hold_penalty

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

    def _log_trial_details(self, trial_num: int, params: Dict, train_metrics: Dict, val_metrics: Dict, score: float):
        """
        Log detailed trial information to file
        Saves: parameters, metrics, R:R analysis, profitability breakdown
        """
        log_dir = Path("tuning_results/logs")
        log_dir.mkdir(exist_ok=True, parents=True)

        log_file = log_dir / f"trial_{trial_num:04d}.json"

        # Calculate breakeven win rate for this R:R
        rr_ratio = float(params['rr_ratio'])
        breakeven_wr = float(1.0 / (1.0 + rr_ratio) * 100)

        # Convert metrics to native Python types (avoid numpy types)
        actual_wr = float(val_metrics['win_rate'])
        total_return = float(val_metrics['total_return'])
        total_trades = int(val_metrics['total_trades'])

        log_data = {
            'trial': int(trial_num),
            'score': float(score),
            'timestamp': datetime.now().isoformat(),

            # Parameters (convert all to native types)
            'params': self._convert_to_native_types(params),

            # R:R Analysis
            'rr_analysis': {
                'rr_ratio': rr_ratio,
                'breakeven_wr': breakeven_wr,
                'actual_wr': actual_wr,
                'wr_above_breakeven': bool(actual_wr > breakeven_wr),
                'expectancy': float((actual_wr/100 * rr_ratio) - ((100-actual_wr)/100))
            },

            # Train Metrics
            'train': self._convert_to_native_types(train_metrics),

            # Validation Metrics
            'val': self._convert_to_native_types(val_metrics),

            # Profitability Assessment
            'profitable': bool(total_return > 0),
            'return_per_trade': float(total_return / max(1, total_trades))
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def _convert_to_native_types(self, obj):
        """
        Convert numpy types to native Python types for JSON serialization
        """
        import numpy as np

        if isinstance(obj, dict):
            return {key: self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

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
