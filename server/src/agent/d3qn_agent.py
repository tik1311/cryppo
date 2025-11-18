"""
Deep Dueling Double DQN (D3QN) Agent with Hybrid CNN-LSTM-Attention Architecture
Combines:
- Deep Q-Network (DQN)
- Double DQN (prevents overestimation)
- Dueling DQN (separate value and advantage streams)
- CNN-BiLSTM-Attention for temporal pattern recognition
- QR-DQN distributional RL for risk assessment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from typing import Tuple, List, Optional

from .hybrid_network import HybridQNetwork


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network Architecture
    Separates value function and advantage function
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int] = [512, 512, 256, 128]):
        super(DuelingQNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature layers (DEEPER network for complex pattern recognition)
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(0.15))  # Reduced dropout for better learning
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)

        # Value stream V(s) - LARGER capacity
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # Advantage stream A(s,a) - LARGER capacity
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_dim)
        )

        # Initialize advantage layer to favor trading actions over HOLD
        self._init_advantage_bias()

    def _init_advantage_bias(self):
        """
        Initialize advantage output layer with bias towards trading actions
        Research shows: initializing Q-values to favor exploration helps!
        """
        with torch.no_grad():
            # Get the final linear layer of advantage stream
            final_layer = self.advantage_stream[-1]

            # Set bias: HOLD=0.0, LONG=+0.5, SHORT=+0.5, CLOSE=+0.3
            # This gives trading actions initial advantage
            if hasattr(final_layer, 'bias') and final_layer.bias is not None:
                final_layer.bias[0] = 0.0   # HOLD - neutral
                if final_layer.bias.shape[0] > 1:
                    final_layer.bias[1] = 0.5   # OPEN_LONG - favored
                if final_layer.bias.shape[0] > 2:
                    final_layer.bias[2] = 0.5   # OPEN_SHORT - favored
                if final_layer.bias.shape[0] > 3:
                    final_layer.bias[3] = 0.3   # CLOSE - slightly favored

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


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer (2025 Best Practice)
    Samples important experiences more frequently
    - TD error = priority (bigger mistakes → learn more)
    - α = prioritization exponent (0=uniform, 1=full priority)
    - β = importance sampling weight (corrects bias from prioritization)
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta_start: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0=none, 1=full)
        self.beta = beta_start  # Importance sampling weight (increases over time)
        self.beta_increment = beta_increment
        self.max_priority = 1.0  # Initialize with high priority

        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer with maximum priority"""
        experience = Experience(state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(self.max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = self.max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch based on priorities"""
        if len(self.buffer) == 0:
            return None

        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        # Increase beta over time (reduces bias correction as training progresses)
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]

        # Convert to tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor(np.array([e.action for e in experiences]))
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences]))
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor(np.array([e.done for e in experiences]))
        weights = torch.FloatTensor(weights)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6  # Small constant to avoid zero priority
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class ReplayBuffer:
    """
    Standard Experience Replay Buffer (kept for backward compatibility)
    Use PrioritizedReplayBuffer for better performance
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch from buffer"""
        experiences = random.sample(self.buffer, batch_size)

        # Convert to numpy arrays first for much faster tensor creation
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor(np.array([e.action for e in experiences]))
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences]))
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor(np.array([e.done for e in experiences]))

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
        window_size: int = 72,
        n_features: int = 25,
        learning_rate: float = 0.0003,
        gamma: float = 0.97,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.15,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 100000,
        batch_size: int = 128,
        target_update_freq: int = 250,
        use_prioritized_replay: bool = True,
        use_hybrid_network: bool = True,  # NEW: Use hybrid CNN-LSTM-Attention architecture
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize D3QN Agent with Hybrid Network

        Args:
            state_dim: Dimension of state space (window_size * n_features)
            action_dim: Number of actions
            window_size: Time window length
            n_features: Number of features per timestep
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting epsilon for exploration
            epsilon_end: Minimum epsilon
            epsilon_decay: Epsilon decay rate
            buffer_size: Replay buffer capacity
            batch_size: Batch size for training
            target_update_freq: Frequency to update target network
            use_prioritized_replay: Use PER (2025 best practice)
            use_hybrid_network: Use hybrid CNN-LSTM-Attention network
            device: Device to run on (cuda/cpu)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.window_size = window_size
        self.n_features = n_features
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_hybrid_network = use_hybrid_network
        self.device = torch.device(device)

        # Networks - Choose architecture
        if use_hybrid_network:
            # Use new Hybrid CNN-LSTM-Attention architecture
            self.policy_net = HybridQNetwork(
                window_size=window_size,
                n_features=n_features,
                action_dim=action_dim,
                cnn_channels=64,
                lstm_hidden=128,
                num_quantiles=32,
                num_attention_heads=4
            ).to(self.device)
            self.target_net = HybridQNetwork(
                window_size=window_size,
                n_features=n_features,
                action_dim=action_dim,
                cnn_channels=64,
                lstm_hidden=128,
                num_quantiles=32,
                num_attention_heads=4
            ).to(self.device)
        else:
            # Fallback to original MLP architecture
            self.policy_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
            self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer (Prioritized for better learning!)
        self.use_prioritized_replay = use_prioritized_replay
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size)
        else:
            self.memory = ReplayBuffer(buffer_size)

        # Training stats
        self.steps = 0
        self.episodes = 0
        self.losses = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy with exploration bias

        Args:
            state: Current state
            training: Whether in training mode (uses exploration)

        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Random action with STRONG bias towards trading (not holding)
            # Action 0 = HOLD, 1 = LONG, 2 = SHORT
            # Bias: 10% HOLD, 45% LONG, 45% SHORT
            # Force agent to explore trading actions
            # NOTE: No CLOSE action - positions close via TP/SL/Trailing only
            action_probs = [0.10, 0.45, 0.45]
            return random.choices(range(self.action_dim), weights=action_probs)[0]

        # Greedy action (exploitation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions (for logging/debugging)

        Args:
            state: Current state

        Returns:
            Q-values for all actions [HOLD, LONG, SHORT]
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy()[0]

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

        # Sample batch (different for prioritized vs standard replay)
        if self.use_prioritized_replay:
            batch = self.memory.sample(self.batch_size)
            if batch is None:
                return None
            states, actions, rewards, next_states, dones, weights, indices = batch
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = None
            indices = None

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

        # Calculate TD errors for prioritized replay
        td_errors = (current_q_values - target_q_values).detach().cpu().numpy()

        # Compute loss (Huber loss is more robust than MSE)
        if self.use_prioritized_replay and weights is not None:
            # Weighted loss for prioritized replay (corrects sampling bias)
            elementwise_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
            loss = (elementwise_loss * weights).mean()
        else:
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities for prioritized replay
        if self.use_prioritized_replay and indices is not None:
            self.memory.update_priorities(indices, td_errors)

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
