"""
Trading Environment for RL (Multi-Timeframe Support)
Based on Gymnasium (OpenAI Gym)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union
import json
from pathlib import Path

OPEN_IDX = 0
HIGH_IDX = 1
LOW_IDX = 2
CLOSE_IDX = 3
VOLUME_IDX = 4


class TradingEnv(gym.Env):
    """
    Trading Environment for Multi-Timeframe Crypto Trading
    
    Action Space (Discrete 3):
        0: HOLD
        1: OPEN_LONG
        2: OPEN_SHORT
        
        * NO CLOSE ACTION: Positions are closed ONLY by TP/SL.
        * If position exists, OPEN_LONG/OPEN_SHORT are masked (invalid).
        
    Observation Space (Dict):
        - 'main': [window_size, main_features + position_info]
        - 'detail': [window_size, detail_len, detail_features]
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        initial_balance: float = 10000.0,
        fee_rate: float = 0.0004,
        max_leverage: float = 1.0,
        window_size: int = 36,  # Reduced from 72 to 36 for efficiency
        reward_type: str = 'pnl',  # Changed from 'sharpe' to 'pnl' for clearer signal
        min_episode_length: int = 500,  # Reduced from 1000 to 500
        atr_period: int = 14,
        atr_sl_multiplier: float = 1.5,
        atr_tp_multiplier: float = 2.5,  # Reduced from 4.5 to 2.5 (better win rate)
        streaming_mode: bool = False,
        enable_bust_protection: bool = True,  # Disable for training, enable for backtest/paper
        logger = None  # Add logger parameter
    ):
        super().__init__()

        self.data = data
        self.main_data = data['main']
        self.detail_data = data['detail']
        
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.max_leverage = max_leverage
        self.window_size = window_size
        self.reward_type = reward_type
        self.min_episode_length = min_episode_length
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier
        self.streaming_mode = streaming_mode
        self.enable_bust_protection = enable_bust_protection

        self.n_main_features = self.main_data.shape[1]
        self.n_detail_features = self.detail_data.shape[2]
        self.detail_len = self.detail_data.shape[1]
        self.n_steps = len(self.main_data)

        # Action space: HOLD, OPEN_LONG, OPEN_SHORT
        self.action_space = spaces.Discrete(3)

        # Observation space
        # Position info: [no_pos, long_pos, short_pos, entry_price_norm, pnl_pct, hold_duration_norm]
        self.n_pos_features = 6
        
        self.observation_space = spaces.Dict({
            'main': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(window_size, self.n_main_features + self.n_pos_features), 
                dtype=np.float32
            ),
            'detail': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(window_size, self.detail_len, self.n_detail_features),
                dtype=np.float32
            )
        })

        # Logging
        self.logger = logger
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.step_log = []
        self.enable_detailed_logging = False
        self.verbose_trades = True  # Always log trades

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, dict]:
        super().reset(seed=seed)
        
        if hasattr(self, 'step_log') and len(self.step_log) > 0:
            self._save_episode_log()
            
        self.step_log = []
        self.episode_count = getattr(self, 'episode_count', 0) + 1

        # Determine start step
        if options and "start_index" in options:
            self.current_step = int(options["start_index"])
        elif self.streaming_mode:
            self.current_step = self.window_size
        else:
            max_start = self.n_steps - self.window_size - self.min_episode_length
            if max_start > self.window_size:
                self.current_step = np.random.randint(self.window_size, max_start)
            else:
                self.current_step = self.window_size

        # Account state
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = None  # None, 'long', 'short'
        self.entry_price = 0.0
        self.position_size = 0.0
        self.hold_duration = 0
        
        # TP/SL
        self.take_profit = 0.0
        self.stop_loss = 0.0
        
        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.trade_log = []
        self.equity_history = [self.initial_balance]
        
        # Patience / Anti-Overtrading
        self.consecutive_holds = 0
        self.last_trade_step = -100
        self.min_bars_between_trades = 3  # Balanced: not too restrictive, prevents spam

        return self._get_observation(), self._get_info()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        current_idx = min(self.current_step, self.n_steps - 1)
        start_idx = current_idx - self.window_size
        end_idx = current_idx

        # 1. Get Market Data
        main_window = self.main_data[start_idx:end_idx]      # [window, features]
        detail_window = self.detail_data[start_idx:end_idx]  # [window, detail_len, features]

        # 2. Build Position Info
        current_price = self.main_data[current_idx, CLOSE_IDX]
        
        has_no_pos = 1.0 if self.position is None else 0.0
        has_long = 1.0 if self.position == 'long' else 0.0
        has_short = 1.0 if self.position == 'short' else 0.0
        
        if self.position:
            entry_norm = (self.entry_price / current_price - 1.0)
            pnl_pct = self._calculate_pnl(current_price) / self.balance
            hold_norm = self.hold_duration / 100.0
        else:
            entry_norm = 0.0
            pnl_pct = 0.0
            hold_norm = 0.0
            
        pos_info = np.array([has_no_pos, has_long, has_short, entry_norm, pnl_pct, hold_norm])
        
        # Tile position info to match window size
        pos_info_tiled = np.tile(pos_info, (self.window_size, 1))
        
        # Concatenate to main features
        main_obs = np.concatenate([main_window, pos_info_tiled], axis=1)
        
        return {
            'main': main_obs.astype(np.float32),
            'detail': detail_window.astype(np.float32)
        }

    def get_action_mask(self) -> np.ndarray:
        """
        Mask invalid actions.
        If position exists: [1, 0, 0] (Only HOLD allowed)
        If no position: [1, 1, 1] (All allowed)
        """
        mask = np.zeros(3, dtype=np.float32)
        if self.position:
            mask[0] = 1.0  # HOLD
        else:
            mask[0] = 1.0
            mask[1] = 1.0
            mask[2] = 1.0
        return mask

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, dict]:
        terminated = False
        truncated = False
        
        current_price = float(self.main_data[self.current_step, CLOSE_IDX])
        high_price = float(self.main_data[self.current_step, HIGH_IDX])
        low_price = float(self.main_data[self.current_step, LOW_IDX])
        
        reward = 0.0
        trade_reward = 0.0
        
        # 1. Handle Actions
        if self.position is None:
            # Only allow opening if enough time passed since last trade
            if action in [1, 2] and (self.current_step - self.last_trade_step >= self.min_bars_between_trades):
                if action == 1:
                    self._open_position('long', current_price)
                elif action == 2:
                    self._open_position('short', current_price)
                self.last_trade_step = self.current_step
                self.consecutive_holds = 0
                # NO entry penalty - let model learn quality entries through TP/SL rewards
            else:
                # HOLD or Invalid Open
                self.consecutive_holds += 1
                # Tiny reward for patience (don't make it too attractive)
                reward += 0.0001
        else:
            # Already in position - Action is ignored (forced HOLD)
            self.hold_duration += 1
            self.consecutive_holds = 0
            
            # Check TP/SL
            exit_price = None
            exit_reason = None
            
            if self.position == 'long':
                if low_price <= self.stop_loss:
                    exit_price = self.stop_loss
                    exit_reason = 'sl'
                elif high_price >= self.take_profit:
                    exit_price = self.take_profit
                    exit_reason = 'tp'
            elif self.position == 'short':
                if high_price >= self.stop_loss:
                    exit_price = self.stop_loss
                    exit_reason = 'sl'
                elif low_price <= self.take_profit:
                    exit_price = self.take_profit
                    exit_reason = 'tp'
            
            if exit_price:
                trade_reward = self._close_position(exit_price, exit_reason)
                # Normalize PnL to [-1, 1] range for stable learning
                normalized_pnl = np.tanh(trade_reward / 100.0)
                reward += normalized_pnl

                # Reward structure for quality entry points:
                # - Focus on learning WHEN to enter (quality over quantity)
                # - TP = Good entry point → Strong positive reward
                # - SL = Bad entry point → Moderate penalty (SL is risk management, not failure)
                # - Win streak bonus = Consistent quality entries
                if exit_reason == 'tp':
                    reward += 1.0  # Strong reward for hitting TP (good entry point)
                    # Extra bonus for maintaining win streak (quality consistency)
                    if self.winning_trades > 0 and self.total_trades > 0:
                        recent_win_rate = self.winning_trades / self.total_trades
                        if recent_win_rate > 0.4:
                            reward += 0.3  # Bonus for >40% win rate
                        if recent_win_rate > 0.5:
                            reward += 0.2  # Extra bonus for >50% win rate
                elif exit_reason == 'sl':
                    reward -= 0.5  # Penalty for hitting SL (poor entry point)
            
            # Time penalty to discourage holding forever
            if self.hold_duration > 100:
                reward -= 0.001

        # 2. Update State
        self.current_step += 1
        
        # Update Equity
        if self.position:
            pnl = self._calculate_pnl(current_price)
            self.equity = self.balance + pnl
        else:
            self.equity = self.balance
            
        self.equity_history.append(self.equity)
        
        # 3. Check Termination
        if self.current_step >= self.n_steps:
            truncated = True
            if self.position:
                self._close_position(current_price, 'timeout')

        # Bust protection (can be disabled for training to allow full episode completion)
        if self.enable_bust_protection and self.equity <= self.initial_balance * 0.5:
            terminated = True
            reward -= 1.0
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"BUST PROTECTION TRIGGERED at step {self.current_step} (Equity: ${self.equity:.2f})")

        info = self._get_info()
        observation = self._get_observation()
        
        return observation, reward, terminated, truncated, info

    def _open_position(self, pos_type: str, price: float):
        self.position = pos_type
        self.entry_price = price
        self.hold_duration = 0
        
        # Calculate Position Size (100% equity used)
        margin = self.balance
        self.position_size = (margin * self.max_leverage) / price
        
        # Fee
        fee = (price * self.position_size) * self.fee_rate
        self.balance -= fee
        
        # Set TP/SL based on ATR
        atr = self._calculate_atr()
        if pos_type == 'long':
            self.take_profit = price + (atr * self.atr_tp_multiplier)
            self.stop_loss = price - (atr * self.atr_sl_multiplier)
        else:
            self.take_profit = price - (atr * self.atr_tp_multiplier)
            self.stop_loss = price + (atr * self.atr_sl_multiplier)
            
        # Log trade open
        trade_msg = f"[OPEN {pos_type.upper()}] @ {price:.2f} | TP: {self.take_profit:.2f} | SL: {self.stop_loss:.2f} | Size: {self.position_size:.4f}"
        if self.total_trades < 5 or self.verbose_trades:
            print(f"  {trade_msg}")
        if self.logger:
            self.logger.info(f"Step {self.current_step} | {trade_msg}")

    def _close_position(self, price: float, reason: str) -> float:
        if not self.position: return 0.0
        
        # Calculate PnL
        if self.position == 'long':
            pnl = (price - self.entry_price) * self.position_size
        else:
            pnl = (self.entry_price - price) * self.position_size
            
        # Fee
        fee = (price * self.position_size) * self.fee_rate
        pnl -= fee
        
        self.balance += pnl
        self.total_pnl += pnl
        self.total_trades += 1
        if pnl > 0: self.winning_trades += 1
        
        # Calculate win rate
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        pnl_pct = (pnl / (self.entry_price * self.position_size)) * 100

        # Log trade close
        trade_msg = f"[CLOSE {self.position.upper()}] @ {price:.2f} ({reason}) | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | Balance: ${self.balance:.2f} | Trades: {self.total_trades} | Win Rate: {win_rate:.1f}%"
        if self.total_trades <= 10 or self.verbose_trades:
            print(f"  {trade_msg}")
        if self.logger:
            self.logger.info(f"Step {self.current_step} | {trade_msg}")

        self.position = None
        return pnl

    def _calculate_pnl(self, current_price: float) -> float:
        if not self.position: return 0.0
        if self.position == 'long':
            return (current_price - self.entry_price) * self.position_size
        else:
            return (self.entry_price - current_price) * self.position_size

    def _calculate_atr(self) -> float:
        # Simple ATR approximation if not enough data
        if self.current_step < 20:
            return self.main_data[self.current_step, CLOSE_IDX] * 0.01
            
        highs = self.main_data[self.current_step-14:self.current_step, HIGH_IDX]
        lows = self.main_data[self.current_step-14:self.current_step, LOW_IDX]
        return np.mean(highs - lows)

    def _get_info(self) -> dict:
        return {
            'step': self.current_step,
            'balance': self.balance,
            'equity': self.equity,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades)
        }
        
    def _save_episode_log(self):
        pass  # Implement logging if needed
