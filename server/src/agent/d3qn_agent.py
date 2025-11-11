"""
Deep Dueling Double DQN (D3QN) Agent
Combines:
- Deep Q-Network (DQN)
- Double DQN (prevents overestimation)
- Dueling DQN (separate value and advantage streams)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from typing import Tuple, List, Optional


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network Architecture
    Separates value function and advantage function
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int] = [256, 256, 128]):
        super(DuelingQNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)

        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            Q-values [batch_size, action_dim]
        """
        features = self.features(state)

        # Value function
        value = self.value_stream(features)

        # Advantage function
        advantage = self.advantage_stream(features)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class ReplayBuffer:
    """
    Experience Replay Buffer
    Stores transitions and samples random batches for training
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch from buffer"""
        experiences = random.sample(self.buffer, batch_size)

        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class D3QNAgent:
    """
    Deep Dueling Double DQN Agent
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.00025,
        gamma: float = 0.995,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize D3QN Agent

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting epsilon for exploration
            epsilon_end: Minimum epsilon
            epsilon_decay: Epsilon decay rate
            buffer_size: Replay buffer capacity
            batch_size: Batch size for training
            target_update_freq: Frequency to update target network
            device: Device to run on (cuda/cpu)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Networks
        self.policy_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Training stats
        self.steps = 0
        self.episodes = 0
        self.losses = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state
            training: Whether in training mode (uses exploration)

        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            return random.randrange(self.action_dim)

        # Greedy action (exploitation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """
        Perform one gradient descent step on a batch from replay buffer

        Returns:
            Loss value if update performed, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use policy network to select action, target network to evaluate
        with torch.no_grad():
            # Select best actions using policy network
            next_actions = self.policy_net(next_states).argmax(dim=1)
            # Evaluate actions using target network
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Compute target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss (Huber loss is more robust than MSE)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Store loss
        loss_value = loss.item()
        self.losses.append(loss_value)

        return loss_value

    def save(self, path: str):
        """Save agent to file"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
        }, path)

    def load(self, path: str):
        """Load agent from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']

    def get_stats(self) -> dict:
        """Get training statistics"""
        return {
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'buffer_size': len(self.memory)
        }
