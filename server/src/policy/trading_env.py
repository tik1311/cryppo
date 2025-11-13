"""
Trading Environment for RL
Based on Gymnasium (OpenAI Gym)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional


class TradingEnv(gym.Env):
    """
    Trading Environment for 1H Crypto Trading

    Action Space (Futures Trading):
        0: HOLD (do nothing)
        1: OPEN_LONG (or close short and go long)
        2: OPEN_SHORT (or close long and go short)
        3: CLOSE_POSITION (close any open position)

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
        max_leverage: float = 3.0,
        window_size: int = 72,
        reward_type: str = 'sharpe'
    ):
        """
        Initialize Trading Environment

        Args:
            data: Market data array [timesteps, features]
            initial_balance: Starting capital
            fee_rate: Trading fee rate (0.04%)
            max_leverage: Maximum leverage allowed
            window_size: Number of historical timesteps in observation
            reward_type: Type of reward ('profit', 'sharpe', 'sortino')
        """
        super().__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.max_leverage = max_leverage
        self.window_size = window_size
        self.reward_type = reward_type

        self.n_features = data.shape[1]
        self.n_steps = len(data)

        # Action space: HOLD, OPEN_LONG, OPEN_SHORT, CLOSE_POSITION
        self.action_space = spaces.Discrete(4)

        # Observation space: window of features + position info
        # Position info: [is_in_position, entry_price_normalized, pnl_pct, hold_duration]
        obs_shape = (window_size, self.n_features + 4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

        # Episode state
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

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
        """
        Execute one time step (Futures Trading)

        Args:
            action: 0=HOLD, 1=OPEN_LONG, 2=OPEN_SHORT, 3=CLOSE_POSITION

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Initialize episode status flags
        terminated = False
        truncated = False

        # Get current price
        current_price = self.data[self.current_step, 3]  # Close price

        # Execute action
        trade_reward = 0.0
        exploration_bonus = 0.0

        if action == 1:  # OPEN_LONG
            if self.position is None:
                self._open_position('long', current_price)
                exploration_bonus = 0.0001  # Small reward for taking action
            elif self.position == 'short':
                trade_reward = self._close_position(current_price)
                self._open_position('long', current_price)

        elif action == 2:  # OPEN_SHORT
            if self.position is None:
                self._open_position('short', current_price)
                exploration_bonus = 0.0001  # Small reward for taking action
            elif self.position == 'long':
                trade_reward = self._close_position(current_price)
                self._open_position('short', current_price)

        elif action == 3:  # CLOSE_POSITION
            if self.position is not None:
                trade_reward = self._close_position(current_price)

        # Apply funding rate every 8 hours (simulated)
        if self.position and self.current_step % 8 == 0:
            funding_fee = self._apply_funding_rate(current_price)
            self.balance -= funding_fee

        # Track position duration
        position_penalty = 0.0
        if self.position:
            self.steps_in_position += 1
            self.steps_without_position = 0

            # FIXED penalty for holding position too long (> 3 days)
            # Constant penalty, not growing with time
            if self.steps_in_position > self.max_position_duration:
                position_penalty = -0.001  # Small fixed penalty per step
        else:
            self.steps_without_position += 1
            self.steps_in_position = 0

            # NO PENALTY for not trading - let agent learn naturally
            # Penalty causes agent to never learn proper trading timing
            # if self.steps_without_position > self.max_no_position_duration:
            #     position_penalty = -0.0001

        # Auto-close if position held too long
        if self.position and self.steps_in_position >= self.max_position_duration * 1.2:  # 20% grace period
            trade_reward = self._close_position(current_price)
            trade_reward -= 0.1  # Extra penalty for forced close

        # Check liquidation BEFORE updating equity
        if self.position and self._check_liquidation(current_price):
            # Liquidated! Force close with total loss
            trade_reward = -self.balance * 0.9  # Lose 90% of balance
            self.balance *= 0.1  # Keep only 10%
            self.position = None
            self.entry_price = 0.0
            self.position_size = 0.0
            self.hold_duration = 0
            self.liquidation_price = 0.0
            self.steps_in_position = 0
            terminated = True

        # Update equity
        if self.position:
            self.equity = self.balance + self._calculate_position_pnl()
            self.hold_duration += 1
        else:
            self.equity = self.balance

        self.equity_history.append(self.equity)

        # Calculate reward with position penalty and exploration bonus
        reward = self._calculate_reward(trade_reward) + position_penalty + exploration_bonus

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        if self.current_step >= self.n_steps:
            # End of data - close position if open
            if self.position:
                final_price = self.data[self.current_step - 1, 3]  # Use last valid price
                self._close_position(final_price)
            truncated = True

        if self.equity <= self.initial_balance * 0.3:
            # Exceeded max drawdown (70%)
            terminated = True

        # Get new observation (only if not done)
        if truncated or terminated:
            # Return zeros if we've gone past the data
            if self.current_step >= self.n_steps:
                observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                observation = self._get_observation()
        else:
            observation = self._get_observation()

        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _open_position(self, position_type: str, price: float):
        """Open a futures trading position with leverage"""
        self.position = position_type
        self.entry_price = price
        self.hold_duration = 0

        # Calculate position size (use leverage)
        self.current_leverage = self.max_leverage
        available_capital = self.balance * self.current_leverage
        fee = available_capital * self.fee_rate
        self.position_size = (available_capital - fee) / price

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

        # Set TP/SL/BE (example: 2% TP, 1% SL)
        if position_type == 'long':
            self.take_profit = price * 1.02  # 2% profit
            self.stop_loss = price * 0.99    # 1% loss
            self.break_even = price * (1 + self.fee_rate * 2)  # 2x fee to break even
        else:
            self.take_profit = price * 0.98
            self.stop_loss = price * 1.01
            self.break_even = price * (1 - self.fee_rate * 2)

    def _close_position(self, price: float) -> float:
        """Close current position and return PnL"""
        if not self.position:
            return 0.0

        # Calculate PnL
        if self.position == 'long':
            pnl = (price - self.entry_price) * self.position_size
        else:  # short
            pnl = (self.entry_price - price) * self.position_size

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

        # Reset position
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

        return pnl

    def _check_liquidation(self, current_price: float) -> bool:
        """Check if position should be liquidated"""
        if not self.position or self.liquidation_price == 0.0:
            return False

        if self.position == 'long':
            # Long position liquidated if price drops below liquidation price
            return current_price <= self.liquidation_price
        else:
            # Short position liquidated if price rises above liquidation price
            return current_price >= self.liquidation_price

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
            trade_reward: Reward from closing a trade

        Returns:
            Reward value
        """
        # Base reward from trading activity
        base_reward = 0.0
        if trade_reward != 0:
            # Give normalized reward for trade execution (positive or negative)
            base_reward = trade_reward / self.initial_balance

        if self.reward_type == 'profit':
            # Direct profit reward + small step reward
            step_reward = (self.equity - self.equity_history[-2]) / self.initial_balance if len(self.equity_history) > 1 else 0.0
            return base_reward + step_reward

        elif self.reward_type == 'return':
            # Percentage return
            if len(self.equity_history) < 2:
                return base_reward
            step_return = (self.equity_history[-1] - self.equity_history[-2]) / self.equity_history[-2]
            return base_reward + step_return

        elif self.reward_type == 'sharpe':
            # Sharpe ratio (risk-adjusted return) + step reward
            step_reward = 0.0
            if len(self.equity_history) >= 2:
                step_return = (self.equity_history[-1] - self.equity_history[-2]) / self.equity_history[-2]
                step_reward = step_return * 10  # Scale up for learning

            # Calculate Sharpe only if we have enough data
            sharpe_reward = 0.0
            if len(self.equity_history) >= 10:
                equity_window = np.array(self.equity_history[-30:])
                returns = np.diff(equity_window) / equity_window[:-1]
                if len(returns) > 0 and np.std(returns) > 0:
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
                    sharpe_reward = sharpe * 0.1  # Small sharpe component

            return base_reward + step_reward + sharpe_reward

        elif self.reward_type == 'sortino':
            # Sortino ratio (downside risk-adjusted return) + step reward
            step_reward = 0.0
            if len(self.equity_history) >= 2:
                step_return = (self.equity_history[-1] - self.equity_history[-2]) / self.equity_history[-2]
                step_reward = step_return * 10  # Scale up for learning

            sortino_reward = 0.0
            if len(self.equity_history) >= 10:
                equity_window = np.array(self.equity_history[-30:])
                returns = np.diff(equity_window) / equity_window[:-1]
                downside_returns = returns[returns < 0]

                if len(returns) > 0 and len(downside_returns) > 0:
                    sortino = np.mean(returns) / (np.std(downside_returns) + 1e-6)
                    sortino_reward = sortino * 0.1
                elif len(returns) > 0:
                    sortino_reward = np.mean(returns) * 0.1

            return base_reward + step_reward + sortino_reward

        else:
            return base_reward

    def render(self):
        """Render environment state"""
        print(f"Step: {self.current_step}/{self.n_steps}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Equity: ${self.equity:.2f}")
        print(f"Position: {self.position}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Win Rate: {self.winning_trades / max(self.total_trades, 1):.2%}")
        print(f"Total PnL: ${self.total_pnl:.2f}")
        print("-" * 50)
