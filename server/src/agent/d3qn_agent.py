"""
Deep Dueling Double DQN (D3QN) Agent (Multi-Timeframe)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from typing import Tuple, List, Optional, Dict

from .hybrid_network import HybridQNetwork

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):  # Reduced from 50000 to 10000 for efficiency
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        experiences = random.sample(self.buffer, batch_size)
        
        # Helper to stack dict states
        def stack_states(states_list):
            return {
                'main': torch.FloatTensor(np.array([s['main'] for s in states_list])),
                'detail': torch.FloatTensor(np.array([s['detail'] for s in states_list]))
            }

        states = stack_states([e.state for e in experiences])
        actions = torch.LongTensor(np.array([e.action for e in experiences]))
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences]))
        next_states = stack_states([e.next_state for e in experiences])
        dones = torch.FloatTensor(np.array([e.done for e in experiences]))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class D3QNAgent:
    def __init__(
        self,
        window_size: int,
        n_main_features: int,
        n_detail_features: int,
        detail_len: int,
        action_dim: int,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        confidence_threshold: float = 0.01,
        logger = None
    ):
        self.window_size = window_size
        self.n_main_features = n_main_features
        self.n_detail_features = n_detail_features
        self.detail_len = detail_len
        self.action_dim = action_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Lower minimum for better exploitation
        self.epsilon_decay = 0.9995  # Slower decay for more exploration
        self.logger = logger

        # Adaptive confidence-based trading
        self.base_confidence_threshold = confidence_threshold
        self.confidence_threshold = confidence_threshold
        
        self.policy_net = HybridQNetwork(
            window_size, n_main_features, n_detail_features, detail_len, action_dim
        ).to(self.device)
        
        self.target_net = HybridQNetwork(
            window_size, n_main_features, n_detail_features, detail_len, action_dim
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(10000)
        self.steps = 0
        self.batch_size = 32
        self.target_update_freq = 5000

    def select_action(self, state: Dict[str, np.ndarray], training: bool = True, action_mask: Optional[np.ndarray] = None, step: int = 0) -> int:
        """
        Select action using Q-values from the network.

        Training mode: Uses epsilon-greedy exploration
        Inference mode: Uses pure exploitation (best Q-value action)
        """
        with torch.no_grad():
            state_tensor = {
                'main': torch.FloatTensor(state['main']).unsqueeze(0).to(self.device),
                'detail': torch.FloatTensor(state['detail']).unsqueeze(0).to(self.device)
            }
            q_values = self.policy_net(state_tensor)
            q_values_np = q_values.cpu().numpy()[0]

            # Apply action mask (if in position, can only HOLD)
            if action_mask is not None:
                q_values_np[action_mask == 0] = -np.inf

            # Get Q-values for each action
            hold_q = q_values_np[0]
            long_q = q_values_np[1] if action_mask is None or action_mask[1] > 0 else -np.inf
            short_q = q_values_np[2] if action_mask is None or action_mask[2] > 0 else -np.inf

            # Best action based on Q-values
            best_action = int(np.argmax(q_values_np))

            if training:
                # Epsilon-greedy exploration during training
                if random.random() < self.epsilon:
                    # Random action (including HOLD)
                    if action_mask is not None:
                        valid_actions = np.where(action_mask > 0)[0]
                        action = np.random.choice(valid_actions)
                    else:
                        action = random.randint(0, 2)

                    if self.logger and step % 200 == 0:
                        action_names = ["HOLD", "LONG", "SHORT"]
                        self.logger.debug(
                            f"Step {step} | EXPLORE → {action_names[action]} | "
                            f"Q[H]={hold_q:.3f} Q[L]={long_q:.3f} Q[S]={short_q:.3f} | "
                            f"Epsilon={self.epsilon:.4f}"
                        )
                    return action
                else:
                    # Exploitation
                    if self.logger and step % 200 == 0:
                        action_names = ["HOLD", "LONG", "SHORT"]
                        self.logger.debug(
                            f"Step {step} | EXPLOIT → {action_names[best_action]} | "
                            f"Q[H]={hold_q:.3f} Q[L]={long_q:.3f} Q[S]={short_q:.3f}"
                        )
                    return best_action
            else:
                # Pure exploitation during inference/backtest
                if self.logger and step % 200 == 0:
                    action_names = ["HOLD", "LONG", "SHORT"]
                    self.logger.debug(
                        f"Step {step} | INFERENCE → {action_names[best_action]} | "
                        f"Q[H]={hold_q:.3f} Q[L]={long_q:.3f} Q[S]={short_q:.3f}"
                    )
                return best_action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move to device
        states['main'] = states['main'].to(self.device)
        states['detail'] = states['detail'].to(self.device)
        next_states['main'] = next_states['main'].to(self.device)
        next_states['detail'] = next_states['detail'].to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        # Q(s,a)
        curr_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN target
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        loss = nn.functional.smooth_l1_loss(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Gradient clipping for stability
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
