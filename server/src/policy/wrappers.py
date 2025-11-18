"""
Policy wrappers used by CLI and training scripts.

Currently exposes a ContinuousActionWrapper that allows continuous-control
algorithms (SAC/TD3) from Stable-Baselines3 to interact with the discrete
TradingEnv by mapping the continuous Box actions to discrete indices.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ContinuousActionWrapper(gym.ActionWrapper):
    """
    Convert continuous Box actions to discrete actions expected by TradingEnv.

    The wrapped environment still receives discrete actions (0..N-1) while the
    agent operates on a continuous Box space where each dimension represents the
    preference score for a discrete action. We simply take the argmax.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError("ContinuousActionWrapper requires a discrete env")

        action_dim = env.action_space.n
        # Allow algorithms like SAC/TD3 that expect bounded continuous spaces.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )

    def action(self, action):
        """
        Map continuous action vector to discrete index by selecting argmax.
        """
        action_array = np.array(action, dtype=np.float32).flatten()
        if action_array.size == 0:
            return 0

        index = int(np.argmax(action_array))
        return index

    def reverse_action(self, action):
        """
        Map discrete action back to one-hot vector for logging/debugging.
        """
        arr = np.zeros(self.env.action_space.n, dtype=np.float32)
        arr[int(action)] = 1.0
        return arr
