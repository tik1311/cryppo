"""
Detailed Logger for Trading Analysis

This logger captures EVERYTHING needed to debug why the model is not profitable:
- Every action taken and why (Q-values for all actions)
- Entry/exit decisions with full context (price, ATR, TP/SL levels)
- Reward breakdown (trade_reward, exploration_bonus, penalties)
- Position tracking (PnL, duration, unrealized profit)
- Episode statistics and performance metrics

Format: Human-readable text + JSON for easy parsing
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np


class DetailedStepLogger:
    """Logs every single step with full context for post-training analysis"""

    def __init__(self, log_dir: str = "logs/detailed", mode: str = "train", session_id: str = None):
        """
        Initialize logger

        Args:
            log_dir: Base log directory (e.g., "logs")
            mode: Logging mode - "train", "backtest", or "paper"
            session_id: Optional session ID (if None, will use timestamp)
        """
        self.log_dir = Path(log_dir)
        self.mode = mode
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create mode-specific directory
        self.mode_dir = self.log_dir / mode
        self.mode_dir.mkdir(parents=True, exist_ok=True)

        # Use consistent filenames (append mode) instead of creating new files each time
        self.step_log_file = self.mode_dir / f"{mode}_steps.jsonl"
        self.trade_log_file = self.mode_dir / f"{mode}_trades.jsonl"
        self.episode_log_file = self.mode_dir / f"{mode}_episodes.jsonl"
        self.analysis_log_file = self.mode_dir / f"{mode}_analysis.txt"

        # Open files in APPEND mode with UTF-8 encoding (fixes Windows cp874 issue)
        self.step_log = open(self.step_log_file, 'a', buffering=1, encoding='utf-8')
        self.trade_log = open(self.trade_log_file, 'a', buffering=1, encoding='utf-8')
        self.episode_log = open(self.episode_log_file, 'a', buffering=1, encoding='utf-8')
        self.analysis_log = open(self.analysis_log_file, 'a', buffering=1, encoding='utf-8')

        # Episode tracking
        self.current_episode = 0
        self.episode_steps = []
        self.episode_trades = []

        # Write session header
        self._write_session_header()

    def _write_session_header(self):
        """Write session header with format explanation"""
        header = f"""

{'='*80}
NEW {self.mode.upper()} SESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {self.session_id}
{'='*80}

LOG FILES:
- {self.step_log_file.name}: Every single step with full state/action/reward
- {self.trade_log_file.name}: Every trade entry/exit with PnL and reasons
- {self.episode_log_file.name}: Episode summaries with statistics
- {self.analysis_log_file.name}: Human-readable analysis (this file)

LOG FORMAT:
- All prices in USD
- All percentages as decimals (0.01 = 1%)
- All times in steps (1 step = 1 hour)
- Q-values: [HOLD, LONG, SHORT]
- NO special characters (epsilon = eps) for Windows compatibility

KEY METRICS TO WATCH:
1. Win Rate: Should be >25% for RR 4.5:1 to be profitable
2. Avg Win vs Avg Loss: Should match RR ratio (~4.5:1)
3. Action Distribution: Too much HOLD means no trading
4. Q-Value Evolution: Should stabilize and differentiate actions
5. Epsilon Decay: Should gradually decrease exploration

{'='*80}

"""
        self.analysis_log.write(header)
        self.analysis_log.flush()

    def log_step(self, step_data: Dict[str, Any]):
        """
        Log a single step with full context

        Args:
            step_data: {
                'episode': int,
                'step': int,
                'global_step': int,
                'price': float,
                'action': int,
                'action_name': str,
                'q_values': [float, float, float],  # [HOLD, LONG, SHORT]
                'epsilon': float,
                'position': Optional[str],  # 'long', 'short', None
                'position_size': float,
                'entry_price': float,
                'unrealized_pnl': float,
                'unrealized_pnl_pct': float,
                'tp': float,
                'sl': float,
                'atr': float,
                'balance': float,
                'equity': float,
                'trade_reward': float,
                'exploration_bonus': float,
                'position_penalty': float,
                'total_reward': float,
                'loss': Optional[float],  # Training loss
                'reason': str  # Why this action was chosen
            }
        """
        # Convert numpy types to Python types
        step_data_clean = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                          for k, v in step_data.items()}

        # Write JSON line
        self.step_log.write(json.dumps(step_data_clean) + '\n')

        # Store for episode summary
        self.episode_steps.append(step_data_clean)

        # Write human-readable summary for first/last 10 steps of each episode
        if len(self.episode_steps) <= 10 or step_data.get('terminated') or step_data.get('truncated'):
            self._write_step_summary(step_data_clean)

    def _write_step_summary(self, step_data: Dict[str, Any]):
        """Write human-readable step summary (NO special characters for Windows)"""
        summary = f"""
Step {step_data['step']} (Episode {step_data['episode']}):
  Price: ${step_data['price']:.2f} | Balance: ${step_data['balance']:,.2f}
  Action: {step_data['action_name']} (eps={step_data['epsilon']:.3f})
  Q-values: HOLD={step_data['q_values'][0]:.4f}, LONG={step_data['q_values'][1]:.4f}, SHORT={step_data['q_values'][2]:.4f}
  Reason: {step_data['reason']}
  Position: {step_data['position'] or 'None'} | Size: {step_data['position_size']:.4f}
"""

        if step_data['position']:
            summary += f"""  Entry: ${step_data['entry_price']:.2f} | TP: ${step_data['tp']:.2f} | SL: ${step_data['sl']:.2f}
  Unrealized PnL: ${step_data['unrealized_pnl']:.2f} ({step_data['unrealized_pnl_pct']:.2f}%)
"""

        summary += f"""  Reward Breakdown: trade={step_data['trade_reward']:.4f}, explore={step_data['exploration_bonus']:.4f}, penalty={step_data['position_penalty']:.4f}, total={step_data['total_reward']:.4f}
"""

        if step_data.get('loss') is not None:
            summary += f"  Training Loss: {step_data['loss']:.6f}\n"

        self.analysis_log.write(summary)
        self.analysis_log.flush()

    def log_trade(self, trade_data: Dict[str, Any]):
        """
        Log a completed trade

        Args:
            trade_data: {
                'episode': int,
                'trade_id': int,
                'direction': str,  # 'long' or 'short'
                'entry_step': int,
                'entry_price': float,
                'exit_step': int,
                'exit_price': float,
                'exit_reason': str,  # 'tp', 'sl', 'liquidation'
                'duration_steps': int,
                'size': float,
                'leverage': float,
                'atr_at_entry': float,
                'tp_price': float,
                'sl_price': float,
                'gross_pnl': float,
                'fees': float,
                'net_pnl': float,
                'pnl_pct_vs_margin': float,
                'pnl_pct_vs_notional': float,
                'balance_before': float,
                'balance_after': float
            }
        """
        # Convert numpy types
        trade_data_clean = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                           for k, v in trade_data.items()}

        # Write JSON line
        self.trade_log.write(json.dumps(trade_data_clean) + '\n')

        # Store for episode summary
        self.episode_trades.append(trade_data_clean)

        # Write human-readable summary
        win_loss = "WIN ✓" if trade_data['net_pnl'] > 0 else "LOSS ✗"
        summary = f"""
{'='*80}
TRADE #{trade_data['trade_id']} - {win_loss}
{'='*80}
Direction: {trade_data['direction'].upper()} | Leverage: {trade_data['leverage']:.1f}x
Entry: ${trade_data['entry_price']:.2f} (step {trade_data['entry_step']})
Exit: ${trade_data['exit_price']:.2f} (step {trade_data['exit_step']}) via {trade_data['exit_reason'].upper()}
Duration: {trade_data['duration_steps']} steps
ATR at Entry: ${trade_data['atr_at_entry']:.2f}
TP Target: ${trade_data['tp_price']:.2f} | SL: ${trade_data['sl_price']:.2f}

PnL Analysis:
  Gross PnL: ${trade_data['gross_pnl']:+.2f}
  Fees: ${trade_data['fees']:.2f}
  Net PnL: ${trade_data['net_pnl']:+.2f}
  PnL vs Margin: {trade_data['pnl_pct_vs_margin']:+.2f}%
  PnL vs Notional: {trade_data['pnl_pct_vs_notional']:+.2f}%

Balance: ${trade_data['balance_before']:,.2f} → ${trade_data['balance_after']:,.2f}
{'='*80}

"""
        self.analysis_log.write(summary)
        self.analysis_log.flush()

    def log_episode_end(self, episode_data: Dict[str, Any]):
        """
        Log episode summary with statistics

        Args:
            episode_data: {
                'episode': int,
                'steps': int,
                'total_reward': float,
                'initial_balance': float,
                'final_balance': float,
                'return_pct': float,
                'total_trades': int,
                'winning_trades': int,
                'losing_trades': int,
                'win_rate': float,
                'avg_win': float,
                'avg_loss': float,
                'rr_ratio': float,  # avg_win / avg_loss
                'max_drawdown': float,
                'sharpe_ratio': float,
                'action_distribution': Dict[str, int],
                'avg_loss': float,  # Training loss
                'epsilon_start': float,
                'epsilon_end': float
            }
        """
        # Convert numpy types
        episode_data_clean = {k: float(v) if isinstance(v, (np.floating, np.integer))
                             else (dict(v) if isinstance(v, dict) else v)
                             for k, v in episode_data.items()}

        # Write JSON line
        self.episode_log.write(json.dumps(episode_data_clean) + '\n')

        # Write detailed human-readable summary
        summary = f"""

{'#'*80}
EPISODE {episode_data['episode']} SUMMARY
{'#'*80}

Duration: {episode_data['steps']} steps

FINANCIAL RESULTS:
  Initial Balance: ${episode_data['initial_balance']:,.2f}
  Final Balance: ${episode_data['final_balance']:,.2f}
  Return: {episode_data['return_pct']:+.2f}%
  Total Reward: {episode_data['total_reward']:.2f}
  Max Drawdown: {episode_data['max_drawdown']:.2f}%
  Sharpe Ratio: {episode_data['sharpe_ratio']:.4f}

TRADING STATISTICS:
  Total Trades: {episode_data['total_trades']}
  Winning Trades: {episode_data['winning_trades']}
  Losing Trades: {episode_data['losing_trades']}
  Win Rate: {episode_data['win_rate']:.2%}

  Average Win: ${episode_data['avg_win']:.2f}
  Average Loss: ${episode_data['avg_loss']:.2f}
  Risk/Reward Ratio: {episode_data['rr_ratio']:.2f}:1

ACTION DISTRIBUTION:
  HOLD: {episode_data['action_distribution'].get('HOLD', 0)}
  LONG: {episode_data['action_distribution'].get('LONG', 0)}
  SHORT: {episode_data['action_distribution'].get('SHORT', 0)}

LEARNING PROGRESS:
  Training Loss: {episode_data['avg_loss']:.6f}
  Epsilon: {episode_data['epsilon_start']:.3f} → {episode_data['epsilon_end']:.3f}

PROFITABILITY ANALYSIS:
  Required Win Rate for RR 4.5:1: 18.18%
  Actual Win Rate: {episode_data['win_rate']:.2%}
  Status: {"PROFITABLE ✓" if episode_data['return_pct'] > 0 else "UNPROFITABLE ✗"}

DIAGNOSIS:
"""

        # Add diagnostic insights
        if episode_data['win_rate'] < 0.18:
            summary += "  ⚠️ Win rate below minimum (18%) - agent not finding good setups\n"
        elif episode_data['win_rate'] < 0.25:
            summary += "  ⚠️ Win rate marginal (18-25%) - barely profitable\n"
        else:
            summary += "  ✓ Win rate above 25% - good setup selection\n"

        if episode_data['rr_ratio'] < 3.0:
            summary += "  ⚠️ Actual RR ratio below expected (4.5:1) - exits too early?\n"
        else:
            summary += "  ✓ Actual RR ratio healthy\n"

        total_actions = sum(episode_data['action_distribution'].values())
        hold_pct = episode_data['action_distribution'].get('HOLD', 0) / max(total_actions, 1)
        if hold_pct > 0.7:
            summary += "  ⚠️ Too much HOLD (>70%) - agent too conservative\n"
        elif hold_pct < 0.3:
            summary += "  ⚠️ Too little HOLD (<30%) - overtrading?\n"
        else:
            summary += "  ✓ Healthy action balance\n"

        summary += f"\n{'#'*80}\n\n"

        self.analysis_log.write(summary)
        self.analysis_log.flush()

        # Reset for next episode
        self.episode_steps = []
        self.episode_trades = []
        self.current_episode += 1

    def close(self):
        """Close all log files"""
        if hasattr(self, 'step_log'):
            self.step_log.close()
        if hasattr(self, 'trade_log'):
            self.trade_log.close()
        if hasattr(self, 'episode_log'):
            self.episode_log.close()
        if hasattr(self, 'analysis_log'):
            # Write final summary
            final = f"""

{'='*80}
LOGGING SESSION ENDED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {self.session_id}
Mode: {self.mode.upper()}
{'='*80}

Total Episodes Logged: {self.current_episode}

Log Files:
- {self.step_log_file}
- {self.trade_log_file}
- {self.episode_log_file}
- {self.analysis_log_file}

To analyze these logs:
1. Check {self.analysis_log_file.name} for human-readable summaries
2. Parse {self.episode_log_file.name} for episode statistics
3. Parse {self.trade_log_file.name} for trade-by-trade analysis
4. Parse {self.step_log_file.name} for detailed step analysis

Look for patterns in:
- When does the agent WIN vs LOSE?
- What Q-values lead to profitable trades?
- Is the agent learning (epsilon decay, loss decrease)?
- Are TP/SL levels appropriate (check actual vs expected RR)?

{'='*80}
"""
            # Check if file is still open before writing
            if not self.analysis_log.closed:
                self.analysis_log.write(final)
                self.analysis_log.close()

    def __del__(self):
        """Cleanup - safe destructor"""
        try:
            self.close()
        except (ValueError, AttributeError):
            # File already closed or attribute doesn't exist - ignore
            pass
