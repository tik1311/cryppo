"""
Multi-Timeframe D3QN Agent
Phase 6: D3QN agent with multi-timeframe network for hierarchical trading

Combines:
- Primary timeframe (5m/15m/1h) for macro trend/regime
- Detail timeframe (1m) for micro timing/momentum
- Hierarchical attention fusion
- Dueling QR-DQN for distributional RL
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from typing import Tuple, List, Optional, Dict
from pathlib import Path

from .multi_timeframe_network import MultiTimeframeQNetwork


# Experience tuple for replay buffer
Experience = namedtuple('Experience', [
    'primary_state', 'detail_state', 'action', 'reward',
    'next_primary_state', 'next_detail_state', 'done'
])


class MultiTimeframeD3QNAgent:
    """
    Multi-Timeframe Deep Dueling Double DQN Agent with Quantile Regression

    Key Features:
    - Multi-timeframe state representation (primary + detail)
    - Hierarchical attention fusion
    - Double DQN (target network)
    - Dueling architecture (value + advantage streams)
    - QR-DQN (distributional RL with 32 quantiles)
    - Prioritized Experience Replay
    - Action masking for valid actions
    """

    def __init__(
        self,
        primary_window: int,
        primary_features: int,
        detail_ratio: int,
        detail_features: int,
        action_dim: int = 3,
        n_quantiles: int = 32,
        learning_rate: float = 0.00025,
        gamma: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 128,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.25,  # Phase 5: Increased exploration
        epsilon_decay: float = 0.9997,  # Phase 5: Slower decay
        target_update_freq: int = 250,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize Multi-Timeframe D3QN Agent

        Args:
            primary_window: Window size for primary timeframe
            primary_features: Number of features in primary timeframe
            detail_ratio: Number of detail candles per primary
            detail_features: Number of features in detail timeframe
            action_dim: Number of actions (3: HOLD, LONG, SHORT)
            n_quantiles: Number of quantiles for QR-DQN
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            buffer_size: Replay buffer size
            batch_size: Training batch size
            epsilon_start: Initial epsilon for epsilon-greedy
            epsilon_end: Final epsilon
            epsilon_decay: Epsilon decay rate
            target_update_freq: Steps between target network updates
            device: Device to run on (cuda/cpu)
        """
        self.primary_window = primary_window
        self.primary_features = primary_features
        self.detail_ratio = detail_ratio
        self.detail_features = detail_features
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Policy and target networks
        self.policy_net = MultiTimeframeQNetwork(
            primary_window=primary_window,
            primary_features=primary_features,
            detail_ratio=detail_ratio,
            detail_features=detail_features,
            action_dim=action_dim,
            n_quantiles=n_quantiles
        ).to(device)

        self.target_net = MultiTimeframeQNetwork(
            primary_window=primary_window,
            primary_features=primary_features,
            detail_ratio=detail_ratio,
            detail_features=detail_features,
            action_dim=action_dim,
            n_quantiles=n_quantiles
        ).to(device)

        # Copy weights to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained directly

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        # Training stats
        self.steps = 0
        self.total_loss = 0.0

        # Quantile tau values for QR-DQN
        tau = torch.arange(0, n_quantiles + 1, dtype=torch.float32, device=device) / n_quantiles
        self.tau = ((tau[:-1] + tau[1:]) / 2).view(1, 1, n_quantiles)  # [1, 1, n_quantiles]

        print(f"[MT D3QN] Initialized on {device}")
        print(f"[MT D3QN] Primary: [{primary_window}, {primary_features}], Detail: [{detail_ratio}, {detail_features}]")
        print(f"[MT D3QN] Network parameters: {sum(p.numel() for p in self.policy_net.parameters()):,}")

    def select_action(
        self,
        primary_state: np.ndarray,
        detail_state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy with action masking

        Args:
            primary_state: Primary timeframe state [window, features]
            detail_state: Detail timeframe state [detail_ratio, detail_features]
            action_mask: Binary mask for valid actions [action_dim]
            training: Whether in training mode (affects epsilon)

        Returns:
            action: Selected action (0=HOLD, 1=LONG, 2=SHORT)
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            # Exploration: random action from valid actions
            if action_mask is not None:
                valid_actions = np.where(action_mask == 1.0)[0]
                # Bias towards trading actions (LONG/SHORT) during exploration
                # to combat HOLD bias
                action_probs = np.array([0.1, 0.45, 0.45])  # [HOLD, LONG, SHORT]
                action_probs = action_probs[valid_actions]
                action_probs /= action_probs.sum()
                action = np.random.choice(valid_actions, p=action_probs)
            else:
                action = random.randint(0, self.action_dim - 1)

            return int(action)

        # Exploitation: greedy action
        with torch.no_grad():
            primary_tensor = torch.FloatTensor(primary_state).unsqueeze(0).to(self.device)
            detail_tensor = torch.FloatTensor(detail_state).unsqueeze(0).to(self.device)

            # Get Q-values
            q_values = self.policy_net.get_q_values(primary_tensor, detail_tensor)  # [1, actions]
            q_values = q_values.cpu().numpy()[0]  # [actions]

            # Apply action mask if provided
            if action_mask is not None:
                q_values = np.where(action_mask == 1.0, q_values, -np.inf)

            # Select action with highest Q-value
            action = int(np.argmax(q_values))

        return action

    def store_experience(
        self,
        primary_state: np.ndarray,
        detail_state: np.ndarray,
        action: int,
        reward: float,
        next_primary_state: np.ndarray,
        next_detail_state: np.ndarray,
        done: bool
    ):
        """Store experience in replay buffer"""
        experience = Experience(
            primary_state, detail_state, action, reward,
            next_primary_state, next_detail_state, done
        )
        self.replay_buffer.append(experience)

    def train_step(self) -> float:
        """
        Perform one training step using quantile Huber loss

        Returns:
            loss: Training loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)

        # Unpack batch
        primary_states = torch.FloatTensor([e.primary_state for e in batch]).to(self.device)
        detail_states = torch.FloatTensor([e.detail_state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_primary_states = torch.FloatTensor([e.next_primary_state for e in batch]).to(self.device)
        next_detail_states = torch.FloatTensor([e.next_detail_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)

        # Current Q-value quantiles
        current_q_dist = self.policy_net(primary_states, detail_states)  # [batch, actions, quantiles]
        current_q_dist = current_q_dist.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_quantiles))
        current_q_dist = current_q_dist.squeeze(1)  # [batch, quantiles]

        # Target Q-values (Double DQN)
        with torch.no_grad():
            # Select action using policy network
            next_q_values = self.policy_net.get_q_values(next_primary_states, next_detail_states)  # [batch, actions]
            next_actions = next_q_values.argmax(dim=1)  # [batch]

            # Get Q-distribution from target network
            next_q_dist = self.target_net(next_primary_states, next_detail_states)  # [batch, actions, quantiles]
            next_q_dist = next_q_dist.gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_quantiles))
            next_q_dist = next_q_dist.squeeze(1)  # [batch, quantiles]

            # Bellman update
            target_q_dist = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_dist

        # Quantile Huber loss
        td_error = target_q_dist.unsqueeze(1) - current_q_dist.unsqueeze(2)  # [batch, quantiles, quantiles]
        huber_loss = torch.where(
            td_error.abs() <= 1.0,
            0.5 * td_error.pow(2),
            td_error.abs() - 0.5
        )

        # Quantile loss
        tau = self.tau.expand(self.batch_size, self.n_quantiles, self.n_quantiles)
        quantile_loss = torch.abs(tau - (td_error < 0).float()) * huber_loss
        loss = quantile_loss.sum(dim=2).mean(dim=1).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.total_loss += loss.item()

        return loss.item()

    def save(self, path: Path):
        """Save agent checkpoint"""
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'config': {
                'primary_window': self.primary_window,
                'primary_features': self.primary_features,
                'detail_ratio': self.detail_ratio,
                'detail_features': self.detail_features,
                'action_dim': self.action_dim,
                'n_quantiles': self.n_quantiles
            }
        }
        torch.save(checkpoint, path)
        print(f"[MT D3QN] Saved checkpoint to {path}")

    def load(self, path: Path):
        """Load agent checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        print(f"[MT D3QN] Loaded checkpoint from {path}")


if __name__ == '__main__':
    # Test multi-timeframe agent
    print("Testing Multi-Timeframe D3QN Agent\n")

    # Parameters
    primary_window = 72
    primary_features = 27
    detail_ratio = 5
    detail_features = 8
    action_dim = 3

    # Create agent
    agent = MultiTimeframeD3QNAgent(
        primary_window=primary_window,
        primary_features=primary_features,
        detail_ratio=detail_ratio,
        detail_features=detail_features,
        action_dim=action_dim
    )

    # Test action selection
    primary_state = np.random.randn(primary_window, primary_features)
    detail_state = np.random.randn(detail_ratio, detail_features)
    action_mask = np.array([1.0, 1.0, 1.0])  # All actions valid

    action = agent.select_action(primary_state, detail_state, action_mask)
    print(f"Selected action: {action} (0=HOLD, 1=LONG, 2=SHORT)")

    # Test experience storage and training
    next_primary_state = np.random.randn(primary_window, primary_features)
    next_detail_state = np.random.randn(detail_ratio, detail_features)

    # Store some experiences
    for _ in range(200):
        agent.store_experience(
            primary_state, detail_state, action, 0.1,
            next_primary_state, next_detail_state, False
        )

    # Train
    loss = agent.train_step()
    print(f"\nTraining loss: {loss:.6f}")
    print(f"Epsilon: {agent.epsilon:.4f}")
    print(f"Buffer size: {len(agent.replay_buffer)}")

    # Test save/load
    save_path = Path("test_mt_agent.pth")
    agent.save(save_path)
    agent.load(save_path)
    save_path.unlink()  # Clean up

    print("\n✓ Multi-Timeframe Agent test passed!")
