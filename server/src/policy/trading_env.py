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

    Action Space:
        0: HOLD (do nothing)
        1: BUY (go long)
        2: SELL (close position or go short)

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

        # Action space: HOLD, BUY, SELL
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
        # Get market data window
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        market_data = self.data[start_idx:end_idx]

        # Current price for normalization
        current_price = self.data[self.current_step, 3]  # Close price (index 3)

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
        Execute one time step

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Get current price
        current_price = self.data[self.current_step, 3]  # Close price

        # Execute action
        trade_reward = 0.0

        if action == 1:  # BUY (go long)
            if self.position is None:
                self._open_position('long', current_price)
            elif self.position == 'short':
                trade_reward = self._close_position(current_price)
                self._open_position('long', current_price)

        elif action == 2:  # SELL (close or go short)
            if self.position == 'long':
                trade_reward = self._close_position(current_price)
            elif self.position is None:
                # Could implement short selling here
                pass

        # Update equity
        if self.position:
            self.equity = self.balance + self._calculate_position_pnl()
            self.hold_duration += 1
        else:
            self.equity = self.balance

        self.equity_history.append(self.equity)

        # Calculate reward
        reward = self._calculate_reward(trade_reward)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = False
        truncated = False

        if self.current_step >= self.n_steps - 1:
            # End of data
            if self.position:
                self._close_position(current_price)
            truncated = True

        if self.equity <= self.initial_balance * 0.5:
            # Exceeded max drawdown (50%)
            terminated = True

        # Get new observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _open_position(self, position_type: str, price: float):
        """Open a trading position"""
        self.position = position_type
        self.entry_price = price
        self.hold_duration = 0

        # Calculate position size (use leverage)
        available_capital = self.balance * self.max_leverage
        fee = available_capital * self.fee_rate
        self.position_size = (available_capital - fee) / price

        # Deduct fee from balance
        self.balance -= fee

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

        return pnl

    def _calculate_position_pnl(self) -> float:
        """Calculate unrealized PnL of current position"""
        if not self.position:
            return 0.0

        current_price = self.data[self.current_step, 3]

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
        if self.reward_type == 'profit':
            # Direct profit reward
            if trade_reward != 0:
                return trade_reward / self.initial_balance
            return 0.0

        elif self.reward_type == 'return':
            # Percentage return
            if len(self.equity_history) < 2:
                return 0.0
            return (self.equity_history[-1] - self.equity_history[-2]) / self.equity_history[-2]

        elif self.reward_type == 'sharpe':
            # Sharpe ratio (risk-adjusted return)
            if len(self.equity_history) < 10:
                return 0.0

            returns = np.diff(self.equity_history[-30:]) / self.equity_history[-31:-1]
            if len(returns) == 0 or np.std(returns) == 0:
                return 0.0

            sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
            return sharpe

        elif self.reward_type == 'sortino':
            # Sortino ratio (downside risk-adjusted return)
            if len(self.equity_history) < 10:
                return 0.0

            returns = np.diff(self.equity_history[-30:]) / self.equity_history[-31:-1]
            downside_returns = returns[returns < 0]

            if len(downside_returns) == 0:
                return np.mean(returns)

            sortino = np.mean(returns) / (np.std(downside_returns) + 1e-6)
            return sortino

        else:
            return 0.0

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
