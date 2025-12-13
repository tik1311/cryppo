"""
Trading Environment for RL (Multi-Timeframe Support)
MAJOR OVERHAUL: Focus on PROFITABILITY and REALISTIC TRADING
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

    REDESIGNED FOR PROFITABILITY:
    - Realistic reward based on actual PnL percentage
    - Smart position sizing with proper risk management
    - Market regime awareness (trend/sideways detection)
    - Anti-overtrading mechanism
    - Dynamic TP/SL based on volatility

    Action Space (Discrete 3):
        0: HOLD (wait for better opportunity)
        1: OPEN_LONG
        2: OPEN_SHORT

    Key Improvements:
        1. PnL-based reward (not binary TP/SL)
        2. Trade quality metrics in observation
        3. Volatility-adjusted position sizing
        4. Market regime filter
        5. Patience reward for waiting
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        initial_balance: float = 10000.0,
        fee_rate: float = 0.0004,
        max_leverage: float = 1.0,
        window_size: int = 36,
        reward_type: str = 'profit_focused',
        min_episode_length: int = 500,
        # TP/SL Settings - Optimized for better win rate
        atr_period: int = 14,
        atr_sl_multiplier: float = 1.5,   # Tighter SL for faster exits (was 2.0)
        atr_tp_multiplier: float = 2.0,   # R:R = 1.33:1 but higher WR (was 3.0)
        streaming_mode: bool = False,
        enable_bust_protection: bool = True,
        # Anti-Overtrade Settings
        min_bars_between_trades: int = 5,  # Increased from 3
        max_trades_per_day: int = 3,       # NEW: Limit trades per 24h period
        # Risk Settings
        risk_per_trade: float = 0.02,      # 2% risk per trade (was 10%!)
        # Reward Settings
        win_reward_mult: float = 1.0,
        loss_penalty_mult: float = 1.0,
        hold_reward: float = 0.001,        # Small reward for patience
        # Market Filter
        trend_threshold: float = 0.02,     # 2% trend required
        volatility_threshold: float = 0.005,  # Min volatility to trade
        logger=None
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
        self.max_trades_per_day = max_trades_per_day
        self.risk_per_trade = risk_per_trade
        self.win_reward_mult = win_reward_mult
        self.loss_penalty_mult = loss_penalty_mult
        self.hold_reward = hold_reward
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold

        self.n_main_features = self.main_data.shape[1]
        self.n_detail_features = self.detail_data.shape[2]
        self.detail_len = self.detail_data.shape[1]
        self.n_steps = len(self.main_data)

        # Action space: HOLD, OPEN_LONG, OPEN_SHORT
        self.action_space = spaces.Discrete(3)

        # Position info features (enhanced):
        # [no_pos, long_pos, short_pos, entry_price_norm, unrealized_pnl_pct,
        #  hold_duration_norm, capital_ratio, can_trade, drawdown_pct,
        #  trend_strength, volatility_norm, trade_count_today, win_rate_recent]
        self.n_pos_features = 13

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
        self.verbose_trades = True

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
        self.peak_equity = self.initial_balance
        self.position = None
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
        self.recent_trades = []  # Last 10 trades for win rate calc

        # Anti-Overtrading
        self.consecutive_holds = 0
        self.last_trade_step = -100
        self.trades_today = 0
        self.last_day_step = 0  # Track 24h period (24 bars for 1h)

        # Market Analysis Cache
        self._cached_trend = 0.0
        self._cached_volatility = 0.0
        self._cache_step = -1

        return self._get_observation(), self._get_info()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        current_idx = min(self.current_step, self.n_steps - 1)
        current_idx = max(0, current_idx)  # Ensure non-negative

        # Calculate slice indices
        start_idx = max(0, current_idx - self.window_size + 1)
        end_idx = current_idx + 1  # Include current_idx

        # Get window data
        main_window = self.main_data[start_idx:end_idx]
        detail_window = self.detail_data[start_idx:end_idx]

        # Handle empty data case - create zeros with proper shape
        if len(main_window) == 0 or self.n_steps == 0:
            main_window = np.zeros((1, self.n_main_features), dtype=np.float32)
            detail_window = np.zeros((1, self.detail_len, self.n_detail_features), dtype=np.float32)

        # Pad to window_size if needed
        if len(main_window) < self.window_size:
            pad_size = self.window_size - len(main_window)
            # Save first element for edge padding
            edge_main = main_window[0].copy()
            edge_detail = detail_window[0].copy()
            # Create padded arrays
            padded_main = np.zeros((self.window_size, main_window.shape[1]), dtype=np.float32)
            padded_detail = np.zeros((self.window_size, detail_window.shape[1], detail_window.shape[2]), dtype=np.float32)
            # Fill with edge values first
            for i in range(pad_size):
                padded_main[i] = edge_main
                padded_detail[i] = edge_detail
            # Copy original data to end
            padded_main[pad_size:] = main_window
            padded_detail[pad_size:] = detail_window
            main_window = padded_main
            detail_window = padded_detail

        # Get current price safely
        if self.n_steps > 0:
            current_price = float(self.main_data[current_idx, CLOSE_IDX])
        else:
            current_price = 1.0  # Default fallback

        # Position State
        has_no_pos = 1.0 if self.position is None else 0.0
        has_long = 1.0 if self.position == 'long' else 0.0
        has_short = 1.0 if self.position == 'short' else 0.0

        if self.position:
            entry_norm = (self.entry_price / current_price - 1.0) * 10  # Scale up
            pnl_pct = self._calculate_pnl(current_price) / max(self.balance, 1) * 100
            hold_norm = min(self.hold_duration / 50.0, 2.0)  # Cap at 2.0
        else:
            entry_norm = 0.0
            pnl_pct = 0.0
            hold_norm = 0.0

        # Capital & Risk
        capital_ratio = self.balance / self.initial_balance
        can_trade = 1.0 if self._can_open_trade(current_price) else 0.0

        # Drawdown
        drawdown_pct = (self.peak_equity - self.equity) / max(self.peak_equity, 1) * 100

        # Market Analysis (cached for efficiency)
        if self._cache_step != current_idx:
            self._cached_trend = self._calculate_trend_strength(current_idx)
            self._cached_volatility = self._calculate_volatility(current_idx)
            self._cache_step = current_idx

        trend_strength = self._cached_trend
        volatility_norm = min(self._cached_volatility / 0.05, 2.0)  # Normalize to ~1.0

        # Trading Stats
        trade_count_norm = self.trades_today / max(self.max_trades_per_day, 1)

        # Recent Win Rate
        if len(self.recent_trades) > 0:
            win_rate_recent = sum(1 for t in self.recent_trades if t > 0) / len(self.recent_trades)
        else:
            win_rate_recent = 0.5  # Neutral

        pos_info = np.array([
            has_no_pos, has_long, has_short,
            entry_norm, pnl_pct, hold_norm,
            capital_ratio, can_trade, drawdown_pct,
            trend_strength, volatility_norm, trade_count_norm, win_rate_recent
        ], dtype=np.float32)

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
        Returns action mask - let the model decide freely.
        Only restrict actions that are physically impossible (like opening position when already in one).
        The model should learn when to trade through training, not through hard constraints.
        """
        mask = np.ones(3, dtype=np.float32)  # Allow all actions by default
        current_price = self.main_data[min(self.current_step, self.n_steps - 1), CLOSE_IDX]

        if self.position:
            # If in position, can only HOLD (wait for TP/SL)
            mask = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            # Not in position - check if we have enough capital
            if not self._can_open_trade(current_price):
                # Can't trade due to capital/cooldown, only HOLD
                mask = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            # Otherwise, allow all actions - let the model decide

        return mask

    def _can_open_trade(self, current_price: float) -> bool:
        """Check if we can open a new trade."""
        # Check time since last trade
        if self.current_step - self.last_trade_step < self.min_bars_between_trades:
            return False

        # Check daily trade limit (24 bars = 1 day for 1h timeframe)
        if self.current_step - self.last_day_step >= 24:
            self.trades_today = 0
            self.last_day_step = self.current_step

        if self.trades_today >= self.max_trades_per_day:
            return False

        # Check minimum capital
        min_cost = self._calculate_min_position_cost(current_price)
        if self.balance < min_cost:
            return False

        return True

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, dict]:
        terminated = False
        truncated = False
        reward = 0.0

        current_idx = min(self.current_step, self.n_steps - 1)
        current_price = float(self.main_data[current_idx, CLOSE_IDX])

        # Detail timeframe for TP/SL checking
        detail_high = float(np.max(self.detail_data[current_idx, :, HIGH_IDX]))
        detail_low = float(np.min(self.detail_data[current_idx, :, LOW_IDX]))

        # Get market conditions
        trend = self._calculate_trend_strength(current_idx)
        volatility = self._calculate_volatility(current_idx)

        # 1. Handle Actions - Let the model decide freely
        if self.position is None:
            # No position - consider opening
            if action in [1, 2] and self._can_open_trade(current_price):
                # Execute the action - model decides when to trade
                if action == 1:  # LONG
                    self._open_position('long', current_price, volatility)
                    # Stronger bonus/penalty for trend-following
                    if trend > self.trend_threshold:
                        reward += 0.1  # Good bonus for trend-following (was 0.02)
                    elif trend < -self.trend_threshold:
                        reward -= 0.05  # Penalty for counter-trend

                elif action == 2:  # SHORT
                    self._open_position('short', current_price, volatility)
                    # Stronger bonus/penalty for trend-following
                    if trend < -self.trend_threshold:
                        reward += 0.1  # Good bonus for trend-following (was 0.02)
                    elif trend > self.trend_threshold:
                        reward -= 0.05  # Penalty for counter-trend

                self.last_trade_step = self.current_step
                self.trades_today += 1
                self.consecutive_holds = 0
            else:
                # HOLD when no position or can't trade
                self.consecutive_holds += 1

        else:
            # In position - check TP/SL
            self.hold_duration += 1
            self.consecutive_holds = 0

            exit_price = None
            exit_reason = None

            if self.position == 'long':
                if detail_low <= self.stop_loss:
                    exit_price = self.stop_loss
                    exit_reason = 'sl'
                elif detail_high >= self.take_profit:
                    exit_price = self.take_profit
                    exit_reason = 'tp'
            elif self.position == 'short':
                if detail_high >= self.stop_loss:
                    exit_price = self.stop_loss
                    exit_reason = 'sl'
                elif detail_low <= self.take_profit:
                    exit_price = self.take_profit
                    exit_reason = 'tp'

            if exit_price:
                pnl, pnl_pct = self._close_position(exit_price, exit_reason)

                # PROFIT-FOCUSED REWARD:
                # Directly based on PnL percentage
                if exit_reason == 'tp':
                    # Winning trade: Reward proportional to profit
                    reward += pnl_pct * self.win_reward_mult * 10  # Scale up
                    if self.logger:
                        self.logger.info(f"Step {self.current_step} | ✅ TP HIT | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | Reward: {reward:.3f}")
                else:
                    # Losing trade: Penalty proportional to loss
                    reward += pnl_pct * self.loss_penalty_mult * 10  # pnl_pct is negative
                    if self.logger:
                        self.logger.info(f"Step {self.current_step} | ❌ SL HIT | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | Reward: {reward:.3f}")

                # Bonus/Penalty for trade quality
                if self.hold_duration < 3:
                    reward -= 0.1  # Penalty for very quick exit (likely bad entry)
                elif self.hold_duration > 50:
                    reward -= 0.05  # Small penalty for holding too long

            # Unrealized PnL as small continuous reward
            else:
                unrealized_pnl = self._calculate_pnl(current_price)
                unrealized_pnl_pct = unrealized_pnl / max(self.balance, 1)
                reward += unrealized_pnl_pct * 0.1  # Small continuous signal

        # 2. Update State
        self.current_step += 1

        # Update Equity
        if self.position:
            pnl = self._calculate_pnl(current_price)
            self.equity = self.balance + pnl
        else:
            self.equity = self.balance

        self.equity_history.append(self.equity)
        self.peak_equity = max(self.peak_equity, self.equity)

        # 3. Check Termination
        if self.current_step >= self.n_steps:
            truncated = True
            if self.position:
                self._close_position(current_price, 'timeout')

        # Bankruptcy check
        if self.enable_bust_protection and self.balance < self.initial_balance * 0.5:
            # 50% drawdown = stop
            terminated = True
            reward -= 2.0  # Significant penalty
            if self.logger:
                self.logger.error(f"💀 MAX DRAWDOWN at step {self.current_step} | Balance: ${self.balance:.2f}")

        # Drawdown penalty (continuous)
        current_drawdown = (self.peak_equity - self.equity) / max(self.peak_equity, 1)
        if current_drawdown > 0.1:  # More than 10% drawdown
            reward -= current_drawdown * 0.1  # Penalty proportional to drawdown

        info = self._get_info()
        observation = self._get_observation()

        return observation, float(reward), terminated, truncated, info

    def _open_position(self, pos_type: str, price: float, volatility: float):
        """Open a new position with proper risk management."""
        self.position = pos_type
        self.entry_price = price
        self.hold_duration = 0

        # Position sizing based on risk per trade (2% rule)
        atr = self._calculate_atr()

        # Calculate position size based on risk
        # Risk = Position Size * SL Distance
        # Position Size = Risk Amount / SL Distance
        sl_distance = atr * self.atr_sl_multiplier
        risk_amount = self.balance * self.risk_per_trade  # 2% of balance

        self.position_size = risk_amount / sl_distance

        # Cap position size to prevent over-leverage
        max_position_value = self.balance * self.max_leverage
        max_position_size = max_position_value / price
        self.position_size = min(self.position_size, max_position_size)

        # Entry fee
        fee = (price * self.position_size) * self.fee_rate
        self.balance -= fee

        # Set TP/SL based on ATR
        if pos_type == 'long':
            self.take_profit = price + (atr * self.atr_tp_multiplier)
            self.stop_loss = price - (atr * self.atr_sl_multiplier)
        else:
            self.take_profit = price - (atr * self.atr_tp_multiplier)
            self.stop_loss = price + (atr * self.atr_sl_multiplier)

        # Calculate R:R
        tp_distance = abs(self.take_profit - price)
        sl_distance = abs(self.stop_loss - price)
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0

        # Log
        trade_msg = f"[OPEN {pos_type.upper()}] @ {price:.2f} | TP: {self.take_profit:.2f} | SL: {self.stop_loss:.2f} | Size: {self.position_size:.6f} | R:R: {rr_ratio:.1f}"
        if self.verbose_trades:
            print(f"  {trade_msg}")
        if self.logger:
            self.logger.info(f"Step {self.current_step} | {trade_msg}")

    def _close_position(self, price: float, reason: str) -> Tuple[float, float]:
        """Close position and return PnL."""
        if not self.position:
            return 0.0, 0.0

        # Calculate PnL
        if self.position == 'long':
            pnl = (price - self.entry_price) * self.position_size
        else:
            pnl = (self.entry_price - price) * self.position_size

        # Exit fee
        fee = (price * self.position_size) * self.fee_rate
        pnl -= fee

        # Calculate PnL percentage
        position_value = self.entry_price * self.position_size
        pnl_pct = (pnl / position_value) * 100 if position_value > 0 else 0

        self.balance += pnl
        self.total_pnl += pnl
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1

        # Update recent trades for win rate
        self.recent_trades.append(pnl)
        if len(self.recent_trades) > 10:
            self.recent_trades.pop(0)

        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100

        # Log
        emoji = "✅" if pnl > 0 else "❌"
        trade_msg = f"{emoji} [CLOSE {self.position.upper()}] @ {price:.2f} ({reason}) | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | Balance: ${self.balance:.2f} | Trades: {self.total_trades} | WR: {win_rate:.1f}%"
        if self.verbose_trades:
            print(f"  {trade_msg}")
        if self.logger:
            self.logger.info(f"Step {self.current_step} | {trade_msg}")

        self.position = None
        return pnl, pnl_pct

    def _calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL."""
        if not self.position:
            return 0.0
        if self.position == 'long':
            return (current_price - self.entry_price) * self.position_size
        else:
            return (self.entry_price - current_price) * self.position_size

    def _calculate_min_position_cost(self, current_price: float) -> float:
        """Calculate minimum capital required to open a position."""
        min_position_value = self.initial_balance * 0.01  # 1% minimum
        entry_fee = min_position_value * self.fee_rate
        exit_fee = min_position_value * self.fee_rate
        return (min_position_value + entry_fee + exit_fee) * 1.2  # 20% buffer

    def _calculate_atr(self) -> float:
        """Calculate Average True Range."""
        step = min(self.current_step, self.n_steps - 1)
        current_price = self.main_data[step, CLOSE_IDX]

        if step < 20:
            return current_price * 0.02  # Default 2%

        start = max(0, step - self.atr_period)

        # True Range calculation
        highs = self.main_data[start:step, HIGH_IDX]
        lows = self.main_data[start:step, LOW_IDX]
        closes = self.main_data[start:step, CLOSE_IDX]

        if len(highs) < 2:
            return current_price * 0.02

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )

        return float(np.mean(tr)) if len(tr) > 0 else current_price * 0.02

    def _calculate_trend_strength(self, current_idx: int) -> float:
        """
        Calculate trend strength using multiple indicators.
        Returns: -1 to +1 (bearish to bullish)
        """
        if current_idx < 50:
            return 0.0

        closes = self.main_data[max(0, current_idx - 50):current_idx, CLOSE_IDX]
        if len(closes) < 20:
            return 0.0

        current_price = closes[-1]

        # EMA 20 and 50
        ema_20 = self._ema(closes, 20)
        ema_50 = self._ema(closes, min(50, len(closes)))

        # Price position relative to EMAs
        price_vs_ema20 = (current_price - ema_20) / ema_20 if ema_20 > 0 else 0
        price_vs_ema50 = (current_price - ema_50) / ema_50 if ema_50 > 0 else 0

        # EMA alignment (20 vs 50)
        ema_alignment = (ema_20 - ema_50) / ema_50 if ema_50 > 0 else 0

        # Combine signals
        trend = (price_vs_ema20 * 0.3 + price_vs_ema50 * 0.3 + ema_alignment * 0.4) * 10

        # Clamp to [-1, 1]
        return float(np.clip(trend, -1.0, 1.0))

    def _calculate_volatility(self, current_idx: int) -> float:
        """Calculate recent price volatility."""
        if current_idx < 20:
            return 0.02

        closes = self.main_data[max(0, current_idx - 20):current_idx, CLOSE_IDX]
        if len(closes) < 2:
            return 0.02

        returns = np.diff(closes) / closes[:-1]
        return float(np.std(returns)) if len(returns) > 0 else 0.02

    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return float(np.mean(data)) if len(data) > 0 else 0.0

        multiplier = 2 / (period + 1)
        ema = data[-period]

        for price in data[-period + 1:]:
            ema = (price - ema) * multiplier + ema

        return float(ema)

    def _get_info(self) -> dict:
        return {
            'step': self.current_step,
            'balance': self.balance,
            'equity': self.equity,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'total_pnl': self.total_pnl,
            'drawdown': (self.peak_equity - self.equity) / max(self.peak_equity, 1)
        }

    def _save_episode_log(self):
        pass
