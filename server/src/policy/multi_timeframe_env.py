"""
Multi-Timeframe Trading Environment Wrapper
Phase 6: Wraps TradingEnv to provide multi-timeframe states

State Structure:
    primary_state: [window_size, n_features + 10] - Primary timeframe (5m/15m/1h)
    detail_state: [detail_ratio, detail_features] - 1m detail candles within current primary

Example:
    - Primary: 5m timeframe, window=72 → [72, 27] last 72x 5m candles
    - Detail: Current 5m candle = 5x 1m candles → [5, 8] microstructure
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from .trading_env import TradingEnv


class MultiTimeframeEnv(gym.Wrapper):
    """
    Wrapper for Multi-Timeframe Trading

    Combines:
    - Primary timeframe (5m/15m/1h) for trend/regime
    - Detail timeframe (1m) for timing/microstructure
    """

    def __init__(
        self,
        primary_data: np.ndarray,
        detail_data: np.ndarray,
        primary_timeframe: str = '5m',
        detail_ratio: int = 5,
        **env_kwargs
    ):
        """
        Initialize Multi-Timeframe Environment

        Args:
            primary_data: [N, features] - Primary timeframe data
            detail_data: [N, detail_ratio, detail_features] - Detail candles within primary
            primary_timeframe: Primary timeframe string (5m, 15m, 1h)
            detail_ratio: Number of detail candles per primary (5, 15, 60)
            **env_kwargs: Arguments for base TradingEnv
        """
        # Create base environment with primary data
        base_env = TradingEnv(data=primary_data, **env_kwargs)
        super().__init__(base_env)

        self.primary_data = primary_data
        self.detail_data = detail_data
        self.primary_timeframe = primary_timeframe
        self.detail_ratio = detail_ratio
        self.detail_features = detail_data.shape[2]

        # Override observation space to include both timeframes
        # We return a dict with 'primary' and 'detail' keys
        self.observation_space = spaces.Dict({
            'primary': self.env.observation_space,  # [window_size, features]
            'detail': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(detail_ratio, self.detail_features),
                dtype=np.float32
            )
        })

        print(f"[MultiTF Env] Primary: {primary_timeframe}, Detail: 1m (x{detail_ratio})")
        print(f"[MultiTF Env] Primary data: {primary_data.shape}")
        print(f"[MultiTF Env] Detail data: {detail_data.shape}")

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], dict]:
        """Reset environment and return multi-timeframe state"""
        primary_state, info = self.env.reset(**kwargs)

        # Get detail state for current step
        detail_state = self._get_detail_state()

        # Combine into dict
        state = {
            'primary': primary_state,
            'detail': detail_state
        }

        return state, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        """Execute action and return multi-timeframe state"""
        primary_state, reward, terminated, truncated, info = self.env.step(action)

        # Get detail state for new step
        detail_state = self._get_detail_state()

        # Combine into dict
        state = {
            'primary': primary_state,
            'detail': detail_state
        }

        return state, reward, terminated, truncated, info

    def _get_detail_state(self) -> np.ndarray:
        """
        Get detail candles for current primary candle

        Returns:
            detail_state: [detail_ratio, detail_features] - 1m candles within current primary
        """
        current_idx = min(self.env.current_step, len(self.detail_data) - 1)

        # Get detail candles corresponding to current primary candle
        detail_state = self.detail_data[current_idx]  # [detail_ratio, detail_features]

        return detail_state.astype(np.float32)

    def get_action_mask(self) -> np.ndarray:
        """Forward action mask from base environment"""
        return self.env.get_action_mask()


class MultiTimeframeVectorEnv:
    """
    Vectorized Multi-Timeframe Environment for parallel training
    Useful for PPO/A2C algorithms that benefit from parallel experience collection
    """

    def __init__(
        self,
        n_envs: int,
        primary_data: np.ndarray,
        detail_data: np.ndarray,
        **env_kwargs
    ):
        """
        Initialize vectorized multi-timeframe environments

        Args:
            n_envs: Number of parallel environments
            primary_data: Primary timeframe data
            detail_data: Detail timeframe data
            **env_kwargs: Arguments for MultiTimeframeEnv
        """
        self.n_envs = n_envs
        self.envs = [
            MultiTimeframeEnv(primary_data, detail_data, **env_kwargs)
            for _ in range(n_envs)
        ]

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        """Reset all environments"""
        states = []
        infos = []

        for env in self.envs:
            state, info = env.reset()
            states.append(state)
            infos.append(info)

        # Stack states
        stacked_states = {
            'primary': np.stack([s['primary'] for s in states]),
            'detail': np.stack([s['detail'] for s in states])
        }

        return stacked_states, infos

    def step(self, actions):
        """Step all environments"""
        states = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for env, action in zip(self.envs, actions):
            state, reward, terminated, truncated, info = env.step(action)
            states.append(state)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

        # Stack
        stacked_states = {
            'primary': np.stack([s['primary'] for s in states]),
            'detail': np.stack([s['detail'] for s in states])
        }

        return (
            stacked_states,
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos
        )

    def get_action_masks(self):
        """Get action masks for all environments"""
        return np.array([env.get_action_mask() for env in self.envs])


# Helper function to create multi-timeframe environment from data
def make_multitf_env(
    primary_data: np.ndarray,
    detail_data: np.ndarray,
    primary_timeframe: str = '5m',
    **kwargs
) -> MultiTimeframeEnv:
    """
    Factory function to create multi-timeframe environment

    Args:
        primary_data: Primary timeframe data [N, features]
        detail_data: Detail data [N, detail_ratio, detail_features]
        primary_timeframe: Primary timeframe string
        **kwargs: Additional arguments for TradingEnv

    Returns:
        MultiTimeframeEnv instance
    """
    # Infer detail ratio from data shape
    detail_ratio = detail_data.shape[1]

    env = MultiTimeframeEnv(
        primary_data=primary_data,
        detail_data=detail_data,
        primary_timeframe=primary_timeframe,
        detail_ratio=detail_ratio,
        **kwargs
    )

    return env


if __name__ == '__main__':
    # Test multi-timeframe environment
    print("Testing Multi-Timeframe Environment\n")

    # Simulate data
    n_steps = 1000
    window_size = 72
    primary_features = 27
    detail_ratio = 5
    detail_features = 8

    # Random data for testing
    primary_data = np.random.randn(n_steps + window_size, primary_features)
    detail_data = np.random.randn(n_steps + window_size, detail_ratio, detail_features)

    # Create environment
    env = make_multitf_env(
        primary_data=primary_data,
        detail_data=detail_data,
        primary_timeframe='5m',
        initial_balance=10000,
        window_size=window_size
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test reset
    state, info = env.reset()
    print(f"\nReset state:")
    print(f"  Primary: {state['primary'].shape}")
    print(f"  Detail: {state['detail'].shape}")

    # Test step
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)

    print(f"\nStep result:")
    print(f"  Action: {action}")
    print(f"  Reward: {reward:.4f}")
    print(f"  Primary: {next_state['primary'].shape}")
    print(f"  Detail: {next_state['detail'].shape}")

    # Test action mask
    mask = env.get_action_mask()
    print(f"\nAction mask: {mask}")

    print("\n✓ Multi-Timeframe Environment test passed!")
