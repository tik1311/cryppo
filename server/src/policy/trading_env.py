"""
Trading Environment for RL
Based on Gymnasium (OpenAI Gym)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import json
from pathlib import Path
from datetime import datetime


OPEN_IDX = 0
HIGH_IDX = 1
LOW_IDX = 2
CLOSE_IDX = 3
VOLUME_IDX = 4


class TradingEnv(gym.Env):
    """
    Trading Environment for 1H Crypto Trading

    Action Space (Futures Trading):
        0: HOLD (do nothing)
        1: OPEN_LONG (or close short and go long)
        2: OPEN_SHORT (or close long and go short)

        NOTE: Positions can ONLY be closed by TP/SL/Trailing Stop.
              No manual CLOSE action to prevent premature exits and ensure
              disciplined trading according to risk management rules.

    State Space:
        - OHLCV data + technical indicators
        - Position information (is_long, entry_price, etc.)
        - Account information (balance, equity, pnl)
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data: np.ndarray,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.0004,
        max_leverage: float = 1.5,  # REDUCED from 3.0 - lower risk, prevent -78% losses
        window_size: int = 72,
        reward_type: str = 'sharpe',
        min_episode_length: int = 1200,  # Minimum steps per episode (20-30% more than avg ~1000)
        atr_period: int = 14,  # ATR calculation period
        atr_sl_multiplier: float = 1.5,  # Stop Loss = Entry ± (ATR × 1.5) [INCREASED from 1.0]
        atr_tp_multiplier: float = 3.0,  # Take Profit = Entry ± (ATR × 3.0) [INCREASED from 2.0]
        trailing_distance_multiplier: float = 1.0,  # Trailing distance = ATR × 1.0 [MUST be < SL (1.5x) to protect profit!]
        streaming_mode: bool = False
    ):
        """
        Initialize Trading Environment

        Args:
            data: Market data array [timesteps, features]
            initial_balance: Starting capital
            fee_rate: Trading fee rate (0.04%)
            max_leverage: Maximum leverage allowed (default 1.5x)
            window_size: Number of historical timesteps in observation
            reward_type: Type of reward ('profit', 'sharpe', 'sortino')
            min_episode_length: Minimum steps per episode (ensures 20-30% longer episodes)
            atr_period: Period for ATR calculation (default 14)
            atr_sl_multiplier: Multiplier for Stop Loss calculation (default 1.5x ATR)
            atr_tp_multiplier: Multiplier for Take Profit calculation (default 3.0x ATR)
            trailing_distance_multiplier: Multiplier for Trailing Stop distance (default 2.0x ATR - industry standard for crypto)
        """
        super().__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.max_leverage = max_leverage
        self.window_size = window_size
        self.reward_type = reward_type
        self.min_episode_length = min_episode_length
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier
        self.trailing_distance_multiplier = trailing_distance_multiplier

        self.n_features = data.shape[1]
        self.n_steps = len(data)

        # Action space: HOLD, OPEN_LONG, OPEN_SHORT
        # REMOVED: CLOSE_POSITION (positions close automatically via TP/SL/Trailing)
        self.action_space = spaces.Discrete(3)

        # Observation space: window of features + position info
        # Position info: [is_in_position, entry_price_normalized, pnl_pct, hold_duration]
        obs_shape = (window_size, self.n_features + 4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

        # Detailed logging setup
        self.episode_count = 0
        self.step_log = []  # Store every step details
        self.enable_detailed_logging = False  # Enable for last episode
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.streaming_mode = streaming_mode
        # Episode state
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Save detailed log of previous episode (if any)
        if hasattr(self, 'step_log') and len(self.step_log) > 0:
            self._save_episode_log()

        # Increment episode counter
        self.episode_count += 1
        self.step_log = []  # Reset step log for new episode

        start_index = None
        if options and "start_index" in options:
            start_index = int(options["start_index"])

        if self.streaming_mode and start_index is None:
            start_index = self.window_size

        if start_index is not None:
            self.current_step = int(np.clip(start_index, self.window_size, self.n_steps - 1))
        else:
            max_start = len(self.data) - self.window_size - self.min_episode_length
            if max_start > self.window_size:
                self.current_step = np.random.randint(self.window_size, max_start)
            else:
                self.current_step = self.window_size

        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = None  # None, 'long', or 'short'
        self.entry_price = 0.0
        self.position_size = 0.0
        self.hold_duration = 0
        self.current_leverage = 0.0
        self.liquidation_price = 0.0
        self.take_profit = 0.0
        self.stop_loss = 0.0
        self.break_even = 0.0
        self.funding_rate = 0.0001  # 0.01% default funding rate
        self.last_funding_time = 0

        # Trailing Stop Loss tracking (NEW!)
        self.trailing_stop_loss = 0.0
        self.highest_price_since_entry = 0.0  # For LONG positions
        self.lowest_price_since_entry = 0.0   # For SHORT positions
        self.bars_in_position = 0  # Number of bars held (prevent immediate trail stop)

        # Position duration tracking (to prevent holding too long or not trading)
        self.steps_in_position = 0  # How long current position held
        self.steps_without_position = 0  # How long without any position
        self.max_position_duration = 72  # 72 hours (3 days) max hold
        self.max_no_position_duration = 72  # 72 hours (3 days) max without position

        # Track history for rewards
        self.equity_history = [self.initial_balance]
        self.returns_history = []

        # Episode stats
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.trade_log: List[Dict[str, Any]] = []
        self.position_id_counter = 0
        self.current_position_id = None
        self.entry_step = 0

        # Action diversity tracking (for exploration bonus)
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.consecutive_holds = 0
        self.last_action = -1

        return self._get_observation(), self._get_info()

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation

        Returns:
            State tensor [window_size, n_features + position_info]
        """
        # Ensure we don't go beyond data bounds
        current_idx = min(self.current_step, self.n_steps - 1)

        # Get market data window
        start_idx = current_idx - self.window_size
        end_idx = current_idx
        market_data = self.data[start_idx:end_idx]

        # Current price for normalization
        current_price = self.data[current_idx, 3]  # Close price (index 3)

        # Position information
        is_in_position = 1.0 if self.position is not None else 0.0
        entry_price_norm = (self.entry_price / current_price - 1.0) if self.position else 0.0
        pnl_pct = self._calculate_position_pnl() / self.balance if self.position else 0.0
        hold_duration_norm = self.hold_duration / 100.0  # Normalize hold duration

        # Position info array repeated for each timestep
        position_info = np.array([is_in_position, entry_price_norm, pnl_pct, hold_duration_norm])
        position_info_tiled = np.tile(position_info, (self.window_size, 1))

        # Concatenate market data with position info
        observation = np.concatenate([market_data, position_info_tiled], axis=1)

        return observation.astype(np.float32)

    def _get_info(self) -> dict:
        """Get additional info"""
        return {
            'step': self.current_step,
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1)
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute a single timestep using OHLC data"""
        terminated = False
        truncated = False

        prices = self.data[self.current_step]
        open_price = float(prices[OPEN_IDX])
        high_price = float(prices[HIGH_IDX])
        low_price = float(prices[LOW_IDX])
        close_price = float(prices[CLOSE_IDX])

        trade_reward = 0.0
        exploration_bonus = 0.0
        position_penalty = 0.0

        self.action_counts[action] = self.action_counts.get(action, 0) + 1

        action_names = ['HOLD', 'OPEN_LONG', 'OPEN_SHORT']
        if self.current_step - self.window_size < 3:
            print(f"  [ACTION] Step {self.current_step}: {action_names[action]} | Pos: {self.position or 'None'}")

        if action == 0:
            self.consecutive_holds += 1
            if self.consecutive_holds > 20:
                hold_penalty = -0.0005 * (self.consecutive_holds - 20)
                exploration_bonus += hold_penalty
        else:
            if self.consecutive_holds > 30:
                exploration_bonus += 0.002
            self.consecutive_holds = 0

        total_actions = sum(self.action_counts.values())
        if total_actions > 100:
            action_freq = self.action_counts[action] / total_actions
            if action_freq < 0.10:
                exploration_bonus += 0.0002

        if action == 1:
            if self.position is None:
                self._open_position('long', close_price)
                exploration_bonus += 0.001
            else:
                position_penalty -= 0.01
                if self.total_trades < 5:
                    print(f"  [BLOCKED] Cannot OPEN_LONG - already have {self.position.upper()} position. Wait for TP/SL!")
        elif action == 2:
            if self.position is None:
                self._open_position('short', close_price)
                exploration_bonus += 0.001
            else:
                position_penalty -= 0.01
                if self.total_trades < 5:
                    print(f"  [BLOCKED] Cannot OPEN_SHORT - already have {self.position.upper()} position. Wait for TP/SL!")

        if self.position and self.current_step % 8 == 0:
            funding_fee = self._apply_funding_rate(close_price)
            self.balance -= funding_fee

        position_penalty = 0.0
        if self.position:
            self.steps_in_position += 1
            self.steps_without_position = 0
            if self.steps_in_position > self.max_position_duration:
                position_penalty = -0.001
        else:
            self.steps_without_position += 1
            self.steps_in_position = 0

        if self.position:
            self.bars_in_position += 1
            atr = self._calculate_atr()
            trailing_distance = atr * self.trailing_distance_multiplier

            if self.position == 'long':
                if high_price > self.highest_price_since_entry:
                    self.highest_price_since_entry = high_price
                    new_trailing_stop = high_price - trailing_distance
                    # Only update if new trailing protects profit:
                    # Must be: (1) better than current trailing, (2) better than SL, (3) above entry to protect profit
                    if (new_trailing_stop > self.trailing_stop_loss and
                        new_trailing_stop > self.stop_loss and
                        new_trailing_stop >= self.entry_price):
                        old_trailing = self.trailing_stop_loss
                        self.trailing_stop_loss = new_trailing_stop
                        if self.total_trades <= 5:
                            print(f"  [TRAIL] Stop updated: ${old_trailing:.2f} -> ${self.trailing_stop_loss:.2f} (price: ${high_price:.2f})")
                if self.break_even and self.highest_price_since_entry >= self.break_even:
                    if self.stop_loss < self.break_even:
                        self.stop_loss = self.break_even
                    if self.trailing_stop_loss < self.break_even:
                        self.trailing_stop_loss = self.break_even
            else:
                if self.lowest_price_since_entry == 0 or low_price < self.lowest_price_since_entry:
                    self.lowest_price_since_entry = low_price
                    new_trailing_stop = low_price + trailing_distance
                    # Only update if new trailing protects profit:
                    # Must be: (1) better than current trailing, (2) better than SL, (3) below entry to protect profit
                    if (new_trailing_stop < self.trailing_stop_loss and
                        new_trailing_stop < self.stop_loss and
                        new_trailing_stop <= self.entry_price):
                        old_trailing = self.trailing_stop_loss
                        self.trailing_stop_loss = new_trailing_stop
                        if self.total_trades <= 5:
                            print(f"  [TRAIL] Stop updated: ${old_trailing:.2f} -> ${self.trailing_stop_loss:.2f} (price: ${low_price:.2f})")
                if self.break_even and self.lowest_price_since_entry <= self.break_even:
                    if self.stop_loss > self.break_even:
                        self.stop_loss = self.break_even
                    if self.trailing_stop_loss > self.break_even:
                        self.trailing_stop_loss = self.break_even

            exit_event = None
            exit_price = close_price
            # Don't check exits on the same bar we opened (bars_in_position == 1)
            # This prevents positions from closing using the same candle's high/low that was used for entry
            if self.bars_in_position > 1:
                if self.position == 'long':
                    stop_candidates = []
                    if self.trailing_stop_loss > 0 and low_price <= self.trailing_stop_loss:
                        stop_candidates.append(('trail', self.trailing_stop_loss))
                    if self.stop_loss > 0 and low_price <= self.stop_loss:
                        stop_candidates.append(('sl', self.stop_loss))
                    tp_hit = self.take_profit > 0 and high_price >= self.take_profit
                    if stop_candidates:
                        reason, level = max(stop_candidates, key=lambda x: x[1])
                        exit_event = reason
                        exit_price = level
                    elif tp_hit:
                        exit_event = 'tp'
                        exit_price = self.take_profit
                else:
                    stop_candidates = []
                    if self.trailing_stop_loss > 0 and high_price >= self.trailing_stop_loss:
                        stop_candidates.append(('trail', self.trailing_stop_loss))
                    if self.stop_loss > 0 and high_price >= self.stop_loss:
                        stop_candidates.append(('sl', self.stop_loss))
                    tp_hit = self.take_profit > 0 and low_price <= self.take_profit
                    if stop_candidates:
                        reason, level = min(stop_candidates, key=lambda x: x[1])
                        exit_event = reason
                        exit_price = level
                    elif tp_hit:
                        exit_event = 'tp'
                        exit_price = self.take_profit

            if exit_event:
                trade_reward = self._close_position(exit_price)
                if exit_event == 'tp':
                    trade_reward += 0.02
                    if self.total_trades <= 5:
                        print(f"  [WIN] Take Profit HIT @ ${exit_price:.2f}")
                elif exit_event == 'trail':
                    trade_reward -= 0.01
                    if self.total_trades <= 5:
                        print(f"  [TRAIL SL] Trailing Stop HIT @ ${exit_price:.2f}")
                else:
                    trade_reward -= 0.02
                    if self.total_trades <= 5:
                        print(f"  [RISK] Stop Loss HIT @ ${exit_price:.2f}")

        if self.position and self._check_liquidation(low_price, high_price):
            trade_reward = -self.balance * 0.9
            self.balance *= 0.1
            self.position = None
            self.entry_price = 0.0
            self.position_size = 0.0
            self.hold_duration = 0
            self.liquidation_price = 0.0
            self.steps_in_position = 0
            terminated = True

        if self.position:
            self.equity = self.balance + self._calculate_position_pnl()
            self.hold_duration += 1
        else:
            self.equity = self.balance

        self.equity_history.append(self.equity)
        reward = self._calculate_reward(trade_reward) + position_penalty + exploration_bonus

        self.current_step += 1
        if not self.streaming_mode and self.current_step >= self.n_steps:
            if self.position:
                final_price = self.data[self.current_step - 1, CLOSE_IDX]
                self._close_position(final_price)
            truncated = True

        if self.equity <= self.initial_balance * 0.3:
            terminated = True

        if truncated or terminated:
            if self.current_step >= self.n_steps:
                observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                observation = self._get_observation()
        else:
            observation = self._get_observation()

        info = self._get_info()

        if self.enable_detailed_logging:
            step_data = {
                'episode': self.episode_count,
                'step': self.current_step - self.window_size,
                'global_step': self.current_step,
                'action': int(action),
                'action_name': ['HOLD', 'LONG', 'SHORT'][action] if action < 3 else 'INVALID',
                'price': float(close_price),
                'position': self.position,
                'position_size': float(self.position_size) if self.position else 0.0,
                'entry_price': float(self.entry_price) if self.position else 0.0,
                'balance': float(self.balance),
                'equity': float(self.equity),
                'unrealized_pnl': float(self._calculate_position_pnl()) if self.position else 0.0,
                'take_profit': float(self.take_profit) if self.position else 0.0,
                'stop_loss': float(self.stop_loss) if self.position else 0.0,
                'trailing_stop_loss': float(self.trailing_stop_loss) if self.position else 0.0,
                'highest_price': float(self.highest_price_since_entry) if self.position == 'long' else 0.0,
                'lowest_price': float(self.lowest_price_since_entry) if self.position == 'short' else 0.0,
                'trade_reward': float(trade_reward),
                'exploration_bonus': float(exploration_bonus),
                'position_penalty': float(position_penalty),
                'total_reward': float(reward),
                'total_trades': int(self.total_trades),
                'win_rate': float(info['win_rate']),
                'consecutive_holds': int(self.consecutive_holds),
                'terminated': terminated,
                'truncated': truncated,
            }
            self.step_log.append(step_data)

        return observation, reward, terminated, truncated, info


    def _calculate_atr(self, period: Optional[int] = None) -> float:
        """
        Calculate Average True Range (ATR) at current step

        ATR measures market volatility using the 14-period average of true ranges.
        True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))

        Args:
            period: ATR period (default: use self.atr_period)

        Returns:
            ATR value in price units
        """
        if period is None:
            period = self.atr_period

        # Need at least period+1 candles for ATR
        if self.current_step < self.window_size + period:
            # Not enough data, use simple volatility as fallback
            recent_data = self.data[max(0, self.current_step - 20):self.current_step + 1]
            if len(recent_data) > 1:
                close_col = 3  # Close price column
                close_prices = recent_data[:, close_col]
                # Use standard deviation as volatility measure
                # Multiply by 2 to approximate ATR-like value
                volatility = np.std(close_prices) * 2.0

                # Ensure minimum ATR (0.5% of current price)
                current_price = self.data[self.current_step, 3]
                min_atr = current_price * 0.005
                return max(volatility, min_atr)
            else:
                # Ultimate fallback: 0.8% of current price
                current_price = self.data[self.current_step, 3]
                return current_price * 0.008

        # Get OHLC data for ATR calculation
        start_idx = self.current_step - period
        end_idx = self.current_step + 1

        high = self.data[start_idx:end_idx, 1]   # High price (column 1)
        low = self.data[start_idx:end_idx, 2]    # Low price (column 2)
        close = self.data[start_idx:end_idx, 3]  # Close price (column 3)

        # Calculate True Range
        tr_list = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],                    # Current high - current low
                abs(high[i] - close[i-1]),           # Current high - previous close
                abs(low[i] - close[i-1])             # Current low - previous close
            )
            tr_list.append(tr)

        # ATR = average of true ranges
        if len(tr_list) > 0:
            atr = np.mean(tr_list)
        else:
            # Fallback
            current_price = close[-1]
            atr = current_price * 0.008

        # IMPORTANT: Ensure minimum ATR threshold (0.5% of current price)
        # This prevents unrealistically tight TP/SL ranges
        current_price = close[-1]
        min_atr = current_price * 0.005
        atr = max(atr, min_atr)

        return float(atr)

    def _open_position(self, position_type: str, price: float):
        """
        Open a futures trading position with leverage
        Uses ATR-based TP/SL and initializes Trailing Stop Loss

        Correct futures trading calculation:
        - Margin = amount of balance used (we use 100% of balance)
        - Position Value (Notional) = Margin × Leverage
        - Position Size (Quantity) = Position Value / Entry Price
        - Fee = Position Value × Fee Rate (NOT Margin × Leverage × Fee Rate)
        """
        self.position = position_type
        self.entry_price = price
        self.hold_duration = 0
        self.entry_step = self.current_step

        # Calculate position size with leverage
        self.current_leverage = self.max_leverage

        # Margin is the balance we're using (use 100% for now)
        margin = self.balance

        # Position value (notional value)
        position_value = margin * self.current_leverage

        # Position size in base currency (e.g., BTC)
        self.position_size = position_value / price

        # Calculate fee based on NOTIONAL VALUE (not margin)
        # Fee = notional_value × fee_rate
        notional_value = price * self.position_size
        fee = notional_value * self.fee_rate

        # Deduct fee from balance
        self.balance -= fee

        # Calculate liquidation price (simplified)
        # Liquidation happens when loss reaches margin (1/leverage of position)
        margin_ratio = 1.0 / self.current_leverage
        if position_type == 'long':
            # Long: liquidated if price drops too much
            self.liquidation_price = price * (1 - margin_ratio * 0.9)  # 90% of margin
        else:
            # Short: liquidated if price rises too much
            self.liquidation_price = price * (1 + margin_ratio * 0.9)

        # ATR-BASED Stop Loss & Take Profit (dynamic, adapts to volatility!)
        atr = self._calculate_atr()

        if position_type == 'long':
            # LONG: TP above, SL below
            self.take_profit = price + (atr * self.atr_tp_multiplier)  # Entry + (ATR × 2.0)
            self.stop_loss = price - (atr * self.atr_sl_multiplier)     # Entry - (ATR × 1.0)
            self.break_even = price * (1 + self.fee_rate * 2)

            # Initialize Trailing Stop Loss
            self.trailing_stop_loss = self.stop_loss  # Start at initial SL
            self.highest_price_since_entry = price
            self.lowest_price_since_entry = 0.0
            self.bars_in_position = 0  # Reset bar counter

        else:  # SHORT
            # SHORT: TP below, SL above
            self.take_profit = price - (atr * self.atr_tp_multiplier)  # Entry - (ATR × 2.0)
            self.stop_loss = price + (atr * self.atr_sl_multiplier)     # Entry + (ATR × 1.0)
            self.break_even = price * (1 - self.fee_rate * 2)

            # Initialize Trailing Stop Loss
            self.trailing_stop_loss = self.stop_loss  # Start at initial SL
            self.highest_price_since_entry = 0.0
            self.lowest_price_since_entry = price
            self.bars_in_position = 0  # Reset bar counter

        # Debug logging (only first 5 trades per episode)
        if self.total_trades < 5:
            tp_pct = abs((self.take_profit - price) / price * 100)
            sl_pct = abs((self.stop_loss - price) / price * 100)
            notional = price * self.position_size
            print(f"  [TRADE] Open {position_type.upper()} @ ${price:.2f}")
            print(f"          Margin: ${margin:.2f} | Leverage: {self.current_leverage:.1f}x")
            print(f"          Position Size: {self.position_size:.4f} | Notional: ${notional:.2f}")
            print(f"          Fee: ${fee:.2f} ({fee/margin*100:.2f}% of margin)")
            print(f"          ATR: ${atr:.2f} | TP: ${self.take_profit:.2f} (+{tp_pct:.2f}%) | SL: ${self.stop_loss:.2f} (-{sl_pct:.2f}%)")
            print(f"          Trailing SL: ${self.trailing_stop_loss:.2f}")

    def _close_position(self, price: float) -> float:
        """Close current position and return PnL"""
        if not self.position:
            return 0.0

        position_type = self.position
        entry_price = self.entry_price
        entry_step = self.entry_step
        position_id = self.current_position_id or f"POS-{self.position_id_counter:06d}"

        # Calculate PnL
        if position_type == 'long':
            pnl = (price - entry_price) * self.position_size
        else:  # short
            pnl = (entry_price - price) * self.position_size

        # Deduct closing fee
        fee = price * self.position_size * self.fee_rate
        pnl -= fee

        # Update balance
        self.balance += pnl

        # Track trade
        self.total_trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1

        # Debug logging (only first 5 trades per episode)
        if self.total_trades <= 5:
            # Calculate PnL percentage based on margin used
            # In futures, PnL% should be calculated against margin, not notional
            margin_used = (entry_price * self.position_size) / self.current_leverage if self.current_leverage > 0 else entry_price * self.position_size
            pnl_pct_vs_margin = (pnl / margin_used) * 100 if margin_used > 0 else 0
            pnl_pct_vs_notional = (pnl / (entry_price * self.position_size)) * 100 if entry_price * self.position_size > 0 else 0

            print(f"  [TRADE] Close {position_type.upper()} @ ${price:.2f}")
            print(f"          Gross PnL: ${pnl + fee:.2f} | Fee: ${fee:.2f} | Net PnL: ${pnl:.2f}")
            print(f"          PnL vs Margin: {pnl_pct_vs_margin:+.2f}% | PnL vs Notional: {pnl_pct_vs_notional:+.2f}%")
            print(f"          Balance: ${self.balance:.2f} | Total PnL: ${self.total_pnl:.2f}")

        trade_record = {
            "position_id": position_id,
            "direction": position_type,
            "entry_price": entry_price,
            "exit_price": price,
            "quantity": self.position_size,
            "leverage": self.current_leverage,
            "pnl": pnl,
            "pnl_pct": (pnl / max(entry_price * self.position_size / max(self.current_leverage, 1e-6), 1e-6)) * 100
                      if entry_price and self.position_size else 0.0,
            "entry_step": entry_step,
            "exit_step": self.current_step,
            "duration_steps": self.current_step - entry_step,
            "episode": self.episode_count
        }
        self.trade_log.append(trade_record)

        # Reset position AND trailing stop loss
        self.position = None
        self.entry_price = 0.0
        self.position_size = 0.0
        self.hold_duration = 0
        self.current_leverage = 0.0
        self.liquidation_price = 0.0
        self.take_profit = 0.0
        self.stop_loss = 0.0
        self.break_even = 0.0
        self.steps_in_position = 0

        # Reset trailing stop tracking
        self.trailing_stop_loss = 0.0
        self.highest_price_since_entry = 0.0
        self.lowest_price_since_entry = 0.0
        self.bars_in_position = 0
        self.current_position_id = None

        return pnl

    def _check_liquidation(self, low_price: float, high_price: float) -> bool:
        """Check if position should be liquidated"""
        if not self.position or self.liquidation_price == 0.0:
            return False

        if self.position == 'long':
            # Long position liquidated if price drops below liquidation price
            return low_price <= self.liquidation_price
        else:
            # Short position liquidated if price rises above liquidation price
            return high_price >= self.liquidation_price

    def _apply_funding_rate(self, current_price: float) -> float:
        """
        Apply funding rate for perpetual futures
        Funding rate is paid every 8 hours
        Longs pay shorts when rate is positive, vice versa when negative
        """
        if not self.position:
            return 0.0

        position_value = current_price * self.position_size
        funding_fee = position_value * self.funding_rate

        # Longs pay when positive rate, receive when negative
        # Shorts receive when positive rate, pay when negative
        if self.position == 'long':
            return funding_fee  # Pay if positive
        else:
            return -funding_fee  # Receive if positive (pay if negative)

    def _calculate_position_pnl(self) -> float:
        """Calculate unrealized PnL of current position"""
        if not self.position:
            return 0.0

        # Use current_step - 1 if we're at or beyond the data boundary
        step_idx = min(self.current_step, self.n_steps - 1)
        current_price = self.data[step_idx, 3]

        if self.position == 'long':
            return (current_price - self.entry_price) * self.position_size
        else:  # short
            return (self.entry_price - current_price) * self.position_size

    def _calculate_reward(self, trade_reward: float = 0.0) -> float:
        """
        Calculate reward based on reward type

        Args:
            trade_reward: Reward from closing a trade (in dollars)

        Returns:
            Reward value (normalized, typically between -1 and 1)

        Note: Reduced amplification to prevent unrealistic profit expectations
        """
        # Base reward from trading activity
        # Normalize by initial balance to get percentage return
        base_reward = 0.0
        if trade_reward != 0:
            # Convert trade PnL to percentage return
            # Reduced from 5.0x to 2.0x to prevent over-optimization
            base_reward = (trade_reward / self.initial_balance) * 2.0

        if self.reward_type == 'profit':
            # Direct profit reward + step reward
            step_reward = (self.equity - self.equity_history[-2]) / self.initial_balance if len(self.equity_history) > 1 else 0.0
            # Reduced amplification from 2.0 to 1.0
            return base_reward + step_reward * 1.0

        elif self.reward_type == 'return':
            # Percentage return
            if len(self.equity_history) < 2:
                return base_reward
            step_return = (self.equity_history[-1] - self.equity_history[-2]) / self.equity_history[-2]
            # Reduced amplification from 5.0 to 2.0
            return base_reward + step_return * 2.0

        elif self.reward_type == 'sharpe':
            # Sharpe ratio (risk-adjusted return) + step reward
            step_reward = 0.0
            if len(self.equity_history) >= 2:
                step_return = (self.equity_history[-1] - self.equity_history[-2]) / self.equity_history[-2]
                # Reduced amplification from 5.0 to 2.0 for more realistic learning
                step_reward = step_return * 2.0

            # Calculate Sharpe only if we have enough data
            sharpe_reward = 0.0
            if len(self.equity_history) >= 10:
                equity_window = np.array(self.equity_history[-30:])
                returns = np.diff(equity_window) / equity_window[:-1]
                if len(returns) > 0 and np.std(returns) > 0:
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
                    sharpe_reward = sharpe * 0.05  # Reduced from 0.1

            return base_reward + step_reward + sharpe_reward

        elif self.reward_type == 'sortino':
            # Sortino ratio (downside risk-adjusted return) + step reward
            step_reward = 0.0
            if len(self.equity_history) >= 2:
                step_return = (self.equity_history[-1] - self.equity_history[-2]) / self.equity_history[-2]
                # Reduced amplification from 10 to 3.0
                step_reward = step_return * 3.0

            sortino_reward = 0.0
            if len(self.equity_history) >= 10:
                equity_window = np.array(self.equity_history[-30:])
                returns = np.diff(equity_window) / equity_window[:-1]
                downside_returns = returns[returns < 0]

                if len(returns) > 0 and len(downside_returns) > 0:
                    sortino = np.mean(returns) / (np.std(downside_returns) + 1e-6)
                    sortino_reward = sortino * 0.05  # Reduced from 0.1
                elif len(returns) > 0:
                    sortino_reward = np.mean(returns) * 0.05

            return base_reward + step_reward + sortino_reward

        else:
            return base_reward

    def render(self):
        """Render environment state"""
        print(f"Step: {self.current_step}/{self.n_steps}")

    def _save_episode_log(self):
        """Save detailed episode log to file"""
        if not self.step_log:
            return

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"episode_{self.episode_count:04d}_{timestamp}.jsonl"

        # Save as JSON Lines (one JSON object per line)
        with open(filename, 'w') as f:
            for step_data in self.step_log:
                f.write(json.dumps(step_data) + '\n')

        # Also create summary
        summary_file = self.log_dir / f"episode_{self.episode_count:04d}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Episode {self.episode_count} Summary\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total Steps: {len(self.step_log)}\n")
            f.write(f"Total Trades: {self.total_trades}\n")
            f.write(f"Winning Trades: {self.winning_trades}\n")
            f.write(f"Win Rate: {self.winning_trades / max(self.total_trades, 1) * 100:.2f}%\n")
            f.write(f"Initial Balance: ${self.initial_balance:,.2f}\n")
            f.write(f"Final Balance: ${self.balance:,.2f}\n")
            f.write(f"Final Equity: ${self.equity:,.2f}\n")
            f.write(f"Total PnL: ${self.balance - self.initial_balance:,.2f}\n")
            f.write(f"Return: {(self.balance - self.initial_balance) / self.initial_balance * 100:+.2f}%\n")
            f.write("\n" + "=" * 70 + "\n\n")
            f.write("First 10 Steps:\n")
            f.write("-" * 70 + "\n")
            for i, step in enumerate(self.step_log[:10]):
                f.write(f"Step {step['step']:4d} | {step['action_name']:5s} | ")
                f.write(f"Price: ${step['price']:8.2f} | Balance: ${step['balance']:10.2f} | ")
                f.write(f"Reward: {step['total_reward']:+8.4f}\n")

            f.write("\n" + "-" * 70 + "\n")
            f.write("Last 10 Steps:\n")
            f.write("-" * 70 + "\n")
            for i, step in enumerate(self.step_log[-10:]):
                f.write(f"Step {step['step']:4d} | {step['action_name']:5s} | ")
                f.write(f"Price: ${step['price']:8.2f} | Balance: ${step['balance']:10.2f} | ")
                f.write(f"Reward: {step['total_reward']:+8.4f}\n")

        print(f"  [LOG] Episode {self.episode_count} detailed log saved:")
        print(f"        - {filename}")
        print(f"        - {summary_file}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Equity: ${self.equity:.2f}")
        print(f"Position: {self.position}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Win Rate: {self.winning_trades / max(self.total_trades, 1):.2%}")
        print(f"Total PnL: ${self.total_pnl:.2f}")
        print("-" * 50)

