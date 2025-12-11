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
        atr_sl_multiplier: float = 1.0,  # Tight SL for controlled risk
        atr_tp_multiplier: float = 4.0,  # OPTIMIZED 2025: 2.0 → 4.0 (R:R = 4:1)
        # R:R = 4:1 → Breakeven WR = 20% (model achieves 24-28% → PROFITABLE!)
        # Research: Higher R:R allows lower win rate but bigger profits when right
        streaming_mode: bool = False,
        enable_bust_protection: bool = True,  # Disable for training, enable for backtest/paper
        min_bars_between_trades: int = 3,
        tp_reward_multiplier: float = 1.0,
        sl_penalty_multiplier: float = 1.0,
        hold_penalty_threshold: int = 100,
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
        self.min_bars_between_trades = min_bars_between_trades
        self.tp_reward_multiplier = tp_reward_multiplier
        self.sl_penalty_multiplier = sl_penalty_multiplier
        self.hold_penalty_threshold = hold_penalty_threshold

        self.n_main_features = self.main_data.shape[1]
        self.n_detail_features = self.detail_data.shape[2]
        self.detail_len = self.detail_data.shape[1]
        self.n_steps = len(self.main_data)

        # Action space: HOLD, OPEN_LONG, OPEN_SHORT
        self.action_space = spaces.Discrete(3)

        # Observation space
        # Position info: [no_pos, long_pos, short_pos, entry_price_norm, pnl_pct, hold_duration_norm,
        #                 capital_ratio, can_trade, bankruptcy_risk, trend_signal]
        # NEW (2025): Added capital awareness + trend signal
        self.n_pos_features = 10
        
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
        self.consecutive_holds = 0
        self.last_trade_step = -100
        # self.min_bars_between_trades is set in __init__

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

        # 2025 Enhancement: Capital Awareness Features
        capital_ratio = float(self.balance / self.initial_balance)

        # Check if agent has enough capital to trade
        min_position_cost = self._calculate_min_position_cost(current_price)
        can_trade = 1.0 if self.balance >= min_position_cost else 0.0

        # Bankruptcy risk: 0.0 = safe, 1.0 = about to die
        bankruptcy_risk = max(0.0, 1.0 - capital_ratio)

        # Trend Signal: Compare current price to moving average
        # +1 = bullish (price > MA), -1 = bearish (price < MA), 0 = neutral
        trend_signal = self._calculate_trend_signal(current_idx)

        pos_info = np.array([has_no_pos, has_long, has_short, entry_norm, pnl_pct, hold_norm,
                            capital_ratio, can_trade, bankruptcy_risk, trend_signal])
        
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

        # Entry/Exit price: Use CLOSE price of 1h candle (action happens when 1h candle closes)
        current_price = float(self.main_data[self.current_step, CLOSE_IDX])

        # TP/SL checking: Use detail timeframe (1m) for accurate execution tracking
        # Check all 1m candles within this 1h period to see if TP/SL was hit
        detail_high = float(np.max(self.detail_data[self.current_step, :, HIGH_IDX]))
        detail_low = float(np.min(self.detail_data[self.current_step, :, LOW_IDX]))

        reward = 0.0
        trade_reward = 0.0

        # 1. Handle Actions (only executed on 1h candle close)
        if self.position is None:
            # Only allow opening if enough time passed since last trade
            if action in [1, 2] and (self.current_step - self.last_trade_step >= self.min_bars_between_trades):
                if action == 1:
                    self._open_position('long', current_price)
                elif action == 2:
                    self._open_position('short', current_price)
                self.last_trade_step = self.current_step
                self.consecutive_holds = 0
                # Small reward for taking action (encourages trading)
                reward += 0.01
            else:
                # HOLD when no position
                self.consecutive_holds += 1

                # Log HOLD action periodically
                if self.logger and self.current_step % 100 == 0:
                    self.logger.debug(
                        f"Step {self.current_step} | ACTION: HOLD | "
                        f"Consecutive Holds: {self.consecutive_holds}"
                    )

                # Small penalty for excessive holding (encourages action)
                if self.consecutive_holds > 50:
                    reward -= 0.005
        else:
            # Already in position - Action is ignored (forced HOLD)
            self.hold_duration += 1
            self.consecutive_holds = 0

            # Check TP/SL using detail timeframe (1m) for accuracy
            # This ensures we catch TP/SL hits that happened intrabar (within the 1h candle)
            exit_price = None
            exit_reason = None

            if self.position == 'long':
                # Check if SL was hit during any 1m candle in this 1h period
                if detail_low <= self.stop_loss:
                    exit_price = self.stop_loss
                    exit_reason = 'sl'
                # Check if TP was hit during any 1m candle in this 1h period
                elif detail_high >= self.take_profit:
                    exit_price = self.take_profit
                    exit_reason = 'tp'
            elif self.position == 'short':
                # Check if SL was hit during any 1m candle in this 1h period
                if detail_high >= self.stop_loss:
                    exit_price = self.stop_loss
                    exit_reason = 'sl'
                # Check if TP was hit during any 1m candle in this 1h period
                elif detail_low <= self.take_profit:
                    exit_price = self.take_profit
                    exit_reason = 'tp'
            
            if exit_price:
                trade_reward = self._close_position(exit_price, exit_reason)

                # Simple and clear reward structure:
                # TP hit = Positive reward (good trade)
                # SL hit = Negative reward (bad trade)
                if exit_reason == 'tp':
                    reward += 1.0  # Clear positive signal for TP
                elif exit_reason == 'sl':
                    reward -= 0.5  # Smaller penalty for SL (asymmetric to encourage trading)

                if self.logger:
                    self.logger.info(
                        f"Step {self.current_step} | TRADE CLOSED ({exit_reason.upper()}) | "
                        f"PnL: ${trade_reward:.2f} | Reward: {reward:.3f}"
                    )
            
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

        # Bankruptcy check (if enabled)
        if self.enable_bust_protection and self.balance < self.initial_balance * 0.1:
            terminated = True
            reward -= 5.0  # Penalty for going bankrupt
            if self.logger:
                self.logger.error(f"BANKRUPTCY at step {self.current_step} | Balance: ${self.balance:.2f}")

        info = self._get_info()
        observation = self._get_observation()

        return observation, reward, terminated, truncated, info

    def _open_position(self, pos_type: str, price: float):
        self.position = pos_type
        self.entry_price = price
        self.hold_duration = 0

        # Risk Management: Use only 10% of capital per trade (Kelly-inspired)
        # This allows for multiple losses without bankruptcy
        risk_per_trade = 0.10  # 10% of capital
        margin = self.balance * risk_per_trade
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
        trade_msg = f"[OPEN {pos_type.upper()}] @ {price:.2f} | TP: {self.take_profit:.2f} | SL: {self.stop_loss:.2f} | Size: {self.position_size:.6f}"
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

    def _calculate_min_position_cost(self, current_price: float) -> float:
        """
        Calculate minimum capital required to open a position (2025 Enhancement)

        Based on research: Agent must understand when it can't trade anymore = BANKRUPTCY

        Components:
        - Minimum position value (0.1% of initial balance or $10, whichever is larger)
        - Entry fee
        - Exit fee (worst case: SL hit immediately)
        - Safety margin (10%)

        Returns minimum balance needed to trade
        """
        # Minimum position size (in quote currency)
        min_position_value = max(self.initial_balance * 0.001, 10.0)  # $10 minimum or 0.1% of capital

        # Calculate position size in base currency
        min_position_size = min_position_value / current_price

        # Entry fee
        entry_fee = (current_price * min_position_size) * self.fee_rate

        # Exit fee (assume worst case: immediate SL)
        atr = self._calculate_atr()
        sl_price = current_price - (atr * self.atr_sl_multiplier)  # Long position worst case
        exit_fee = (abs(sl_price) * min_position_size) * self.fee_rate

        # Total cost with 10% safety margin
        total_cost = (min_position_value + entry_fee + exit_fee) * 1.1

        return total_cost

    def _calculate_atr(self) -> float:
        # Simple ATR approximation if not enough data
        step = min(self.current_step, self.n_steps - 1)
        current_price = self.main_data[step, CLOSE_IDX]

        if self.current_step < 20:
            return current_price * 0.01

        start = max(0, step - 14)
        highs = self.main_data[start:step, HIGH_IDX]
        lows = self.main_data[start:step, LOW_IDX]
        return np.mean(highs - lows) if len(highs) > 0 else current_price * 0.01

    def _calculate_trend_signal(self, current_idx: int) -> float:
        """
        Calculate trend signal based on price vs moving averages.
        Returns: +1 (bullish), -1 (bearish), 0 (neutral)
        """
        if current_idx < 50:
            return 0.0

        current_price = self.main_data[current_idx, CLOSE_IDX]

        # Calculate short and long term MAs
        start_short = max(0, current_idx - 20)
        start_long = max(0, current_idx - 50)

        ma_short = np.mean(self.main_data[start_short:current_idx, CLOSE_IDX])
        ma_long = np.mean(self.main_data[start_long:current_idx, CLOSE_IDX])

        # Trend logic: Price > MA_short > MA_long = Strong Bullish
        if current_price > ma_short > ma_long:
            return 1.0  # Bullish
        elif current_price < ma_short < ma_long:
            return -1.0  # Bearish
        elif current_price > ma_long:
            return 0.5  # Weak bullish
        elif current_price < ma_long:
            return -0.5  # Weak bearish
        else:
            return 0.0  # Neutral

    def _calculate_sharpe_reward(self, window: int = 100) -> float:
        """
        Calculate Sharpe ratio-based reward (2025 Research)

        Sharpe Ratio = (Mean Return - Risk-free Rate) / Std Dev of Returns
        For crypto, we assume risk-free rate = 0

        Returns: Sharpe ratio scaled to [-1, 1] range for stable learning
        """
        if len(self.equity_history) < 2:
            return 0.0

        # Use recent window for calculation
        recent_equity = self.equity_history[-window:]

        if len(recent_equity) < 2:
            return 0.0

        # Calculate returns
        returns = np.diff(recent_equity) / recent_equity[:-1]

        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Avoid division by zero
        if std_return < 1e-8:
            return 0.0

        sharpe = mean_return / std_return

        # Scale to [-1, 1] for reward stability
        # Typical Sharpe in crypto: -5 to +5
        scaled_sharpe = np.tanh(sharpe / 2.0)

        return float(scaled_sharpe)

    def _get_reward_by_type(self, base_reward: float) -> float:
        """
        Apply reward type transformation (2025 Enhanced)

        Args:
            base_reward: PnL-based reward

        Returns:
            Transformed reward based on self.reward_type
        """
        if self.reward_type == 'sharpe':
            # Sharpe ratio reward (research-based)
            sharpe_reward = self._calculate_sharpe_reward()
            # Blend with PnL reward (70% Sharpe, 30% PnL)
            return 0.7 * sharpe_reward + 0.3 * base_reward

        elif self.reward_type == 'sortino':
            # Sortino ratio (only penalize downside volatility)
            # TODO: Implement if needed
            return base_reward

        else:  # 'pnl' (default)
            return base_reward

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
