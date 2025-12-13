"""
PPO Agent for Cryptocurrency Trading
REDESIGNED FOR PROFITABILITY: Better exploration, stable learning, and market-aware decisions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import os


class ActorCritic(nn.Module):
    """
    Actor-Critic Network for PPO
    IMPROVED ARCHITECTURE:
    - Separate feature extractors for trend/volatility awareness
    - Attention mechanism for important features
    - Better initialization for trading
    """

    def __init__(
        self,
        window_size: int,
        n_main_features: int,
        n_detail_features: int,
        detail_len: int,
        action_dim: int = 3,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.window_size = window_size
        self.n_main_features = n_main_features
        self.n_detail_features = n_detail_features
        self.detail_len = detail_len
        self.action_dim = action_dim

        # Main timeframe encoder (1h) with residual connection
        self.main_encoder = nn.Sequential(
            nn.Linear(n_main_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),  # GELU works better than ReLU for trading
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )

        # Bidirectional LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )

        # Self-attention for important time steps
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Detail timeframe encoder (1m) - Convolutional
        self.detail_conv = nn.Sequential(
            nn.Conv1d(n_detail_features, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Combined feature size
        # LSTM bidirectional: hidden_dim * 2
        # Detail conv: 32
        combined_size = hidden_dim * 2 + 32

        # Shared layers with residual
        self.shared_fc1 = nn.Linear(combined_size, 256)
        self.shared_ln1 = nn.LayerNorm(256)
        self.shared_fc2 = nn.Linear(256, 128)
        self.shared_ln2 = nn.LayerNorm(128)

        # Actor head (policy) - outputs action logits
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, action_dim)
        )

        # Critic head (value) - outputs state value
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with careful scaling for trading."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'actor' in name:
                    # Small init for actor to start with balanced actions
                    nn.init.orthogonal_(m.weight, gain=0.01)
                elif 'critic' in name:
                    # Normal init for critic
                    nn.init.orthogonal_(m.weight, gain=1.0)
                else:
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: Dict[str, torch.Tensor], action_mask: Optional[torch.Tensor] = None):
        main = state['main']  # [batch, window, features]
        detail = state['detail']  # [batch, window, detail_len, features]

        batch_size = main.shape[0]

        # Encode main timeframe
        main_encoded = self.main_encoder(main)  # [batch, window, 128]

        # LSTM
        lstm_out, _ = self.lstm(main_encoded)  # [batch, window, hidden*2]

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # [batch, window, hidden*2]

        # Take last timestep with attention
        lstm_features = attn_out[:, -1, :]  # [batch, hidden*2]

        # Encode detail timeframe (use last window only)
        detail_last = detail[:, -1, :, :]  # [batch, detail_len, features]
        detail_permuted = detail_last.permute(0, 2, 1)  # [batch, features, detail_len]
        detail_features = self.detail_conv(detail_permuted).squeeze(-1)  # [batch, 32]

        # Combine features
        combined = torch.cat([lstm_features, detail_features], dim=-1)

        # Shared layers with residual
        x = F.gelu(self.shared_ln1(self.shared_fc1(combined)))
        x = F.gelu(self.shared_ln2(self.shared_fc2(x)))

        # Actor output (action logits)
        action_logits = self.actor(x)

        # Apply action mask (mask invalid actions)
        if action_mask is not None:
            # Set masked actions to very negative value
            action_logits = action_logits.masked_fill(action_mask == 0, -1e9)

        # Critic output (state value)
        value = self.critic(x)

        return action_logits, value


class RolloutBuffer:
    """Buffer for storing rollout data with better memory management."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.clear()

    def add(self, state, action, reward, done, log_prob, value, action_mask):
        self.states_main.append(state['main'])
        self.states_detail.append(state['detail'])
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.action_masks.append(action_mask)

        # Trim if too large
        if len(self.actions) > self.max_size:
            self._trim()

    def _trim(self):
        """Remove oldest entries."""
        trim_count = len(self.actions) - self.max_size
        self.states_main = self.states_main[trim_count:]
        self.states_detail = self.states_detail[trim_count:]
        self.actions = self.actions[trim_count:]
        self.rewards = self.rewards[trim_count:]
        self.dones = self.dones[trim_count:]
        self.log_probs = self.log_probs[trim_count:]
        self.values = self.values[trim_count:]
        self.action_masks = self.action_masks[trim_count:]

    def clear(self):
        self.states_main = []
        self.states_detail = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.action_masks = []

    def get(self, device):
        return {
            'states_main': torch.FloatTensor(np.array(self.states_main)).to(device),
            'states_detail': torch.FloatTensor(np.array(self.states_detail)).to(device),
            'actions': torch.LongTensor(np.array(self.actions)).to(device),
            'rewards': torch.FloatTensor(np.array(self.rewards)).to(device),
            'dones': torch.FloatTensor(np.array(self.dones)).to(device),
            'log_probs': torch.FloatTensor(np.array(self.log_probs)).to(device),
            'values': torch.FloatTensor(np.array(self.values)).to(device),
            'action_masks': torch.FloatTensor(np.array(self.action_masks)).to(device),
        }

    def __len__(self):
        return len(self.actions)


class PPOAgent:
    """
    Proximal Policy Optimization Agent
    REDESIGNED FOR PROFITABILITY:
    - Better exploration with entropy bonus
    - Adaptive learning rate
    - Reward normalization
    - Early stopping on poor performance
    """

    def __init__(
        self,
        window_size: int,
        n_main_features: int,
        n_detail_features: int,
        detail_len: int,
        action_dim: int = 3,
        learning_rate: float = 1e-4,  # Lower LR for stability
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.05,  # Higher entropy for more exploration (was 0.02)
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,  # Fewer epochs to prevent overfitting (was 10)
        batch_size: int = 128,  # Larger batch for stability (was 64)
        target_kl: float = 0.02,  # Slightly higher KL threshold (was 0.01)
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger=None
    ):
        self.window_size = window_size
        self.n_main_features = n_main_features
        self.n_detail_features = n_detail_features
        self.detail_len = detail_len
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl
        self.device = torch.device(device)
        self.logger = logger

        # Actor-Critic Network
        self.policy = ActorCritic(
            window_size=window_size,
            n_main_features=n_main_features,
            n_detail_features=n_detail_features,
            detail_len=detail_len,
            action_dim=action_dim
        ).to(self.device)

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        # Learning rate scheduler - reduce on plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize reward
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(max_size=8192)

        # Training stats
        self.training_step = 0
        self.total_updates = 0

        # Reward normalization (running mean/std)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

    def select_action(
        self,
        state: Dict[str, np.ndarray],
        training: bool = True,
        action_mask: Optional[np.ndarray] = None,
        step: int = 0
    ) -> Tuple[int, float, float]:
        """
        Select action using the policy network.
        Returns: (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = {
                'main': torch.FloatTensor(state['main']).unsqueeze(0).to(self.device),
                'detail': torch.FloatTensor(state['detail']).unsqueeze(0).to(self.device)
            }

            mask_tensor = None
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

            action_logits, value = self.policy(state_tensor, mask_tensor)

            # Apply softmax with temperature for better exploration during training
            if training:
                # Higher temperature early, lower later (but never too low)
                temperature = max(0.7, 1.2 - self.total_updates * 0.0005)  # Slower annealing
                probs = F.softmax(action_logits / temperature, dim=-1)
            else:
                # During inference: slightly higher temperature to encourage trading
                probs = F.softmax(action_logits / 0.8, dim=-1)

            dist = Categorical(probs)

            if training:
                action = dist.sample()
            else:
                # During inference: sample to allow trading based on learned policy
                # Higher probability actions are more likely to be selected
                action = dist.sample()

            log_prob = dist.log_prob(action)

            action_int = action.item()
            log_prob_float = log_prob.item()
            value_float = value.item()

            # Periodic logging
            if self.logger and step % 500 == 0:
                probs_np = probs.cpu().numpy()[0]
                action_names = ["HOLD", "LONG", "SHORT"]
                self.logger.debug(
                    f"Step {step} | Action: {action_names[action_int]} | "
                    f"Probs: H={probs_np[0]:.3f} L={probs_np[1]:.3f} S={probs_np[2]:.3f} | "
                    f"Value: {value_float:.3f}"
                )

            return action_int, log_prob_float, value_float

    def store_transition(
        self,
        state: Dict[str, np.ndarray],
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        action_mask: np.ndarray
    ):
        """Store transition in rollout buffer with reward normalization."""
        # Update running reward statistics
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_std = np.sqrt(
            ((self.reward_count - 1) * self.reward_std ** 2 + delta * (reward - self.reward_mean)) /
            max(self.reward_count, 1)
        )

        # Normalize reward (but don't modify original)
        # normalized_reward = (reward - self.reward_mean) / max(self.reward_std, 1e-8)

        self.buffer.add(state, action, reward, done, log_prob, value, action_mask)

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)

        return advantages, returns

    def update(self, next_value: float = 0.0):
        """
        Update policy using PPO objective with early stopping.
        """
        if len(self.buffer) < self.batch_size:
            return None

        # Get data from buffer
        data = self.buffer.get(self.device)

        # Compute GAE
        advantages, returns = self.compute_gae(
            data['rewards'].cpu().numpy(),
            data['values'].cpu().numpy(),
            data['dones'].cpu().numpy(),
            next_value
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs with early stopping
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0

        n_samples = len(self.buffer)
        indices = np.arange(n_samples)
        n_batches = 0
        early_stop = False

        for epoch in range(self.n_epochs):
            if early_stop:
                break

            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get batch data
                batch_states = {
                    'main': data['states_main'][batch_indices],
                    'detail': data['states_detail'][batch_indices]
                }
                batch_actions = data['actions'][batch_indices]
                batch_old_log_probs = data['log_probs'][batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_masks = data['action_masks'][batch_indices]

                # Forward pass
                action_logits, values = self.policy(batch_states, batch_masks)

                # Calculate new log probs
                probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track KL divergence for early stopping
                with torch.no_grad():
                    kl = (batch_old_log_probs - new_log_probs).mean().item()
                    total_kl += kl

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_batches += 1

            # Check KL divergence for early stopping
            avg_kl = total_kl / max(n_batches, 1)
            if avg_kl > self.target_kl * 1.5:
                if self.logger:
                    self.logger.info(f"Early stopping at epoch {epoch + 1} due to KL divergence: {avg_kl:.4f}")
                early_stop = True
                break

        # Clear buffer after update
        self.buffer.clear()

        self.total_updates += 1

        # Calculate averages
        if n_batches > 0:
            avg_loss = total_loss / n_batches
            avg_policy_loss = total_policy_loss / n_batches
            avg_value_loss = total_value_loss / n_batches
            avg_entropy = total_entropy / n_batches
            avg_kl = total_kl / n_batches

            if self.logger:
                self.logger.info(
                    f"PPO Update #{self.total_updates} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Policy: {avg_policy_loss:.4f} | "
                    f"Value: {avg_value_loss:.4f} | "
                    f"Entropy: {avg_entropy:.4f} | "
                    f"KL: {avg_kl:.4f}"
                )

            return {
                'loss': avg_loss,
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'entropy': avg_entropy,
                'kl': avg_kl
            }

        return None

    def update_scheduler(self, metric: float):
        """Update learning rate based on performance."""
        self.scheduler.step(metric)

    def save(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_updates': self.total_updates,
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'reward_count': self.reward_count,
            'config': {
                'window_size': self.window_size,
                'n_main_features': self.n_main_features,
                'n_detail_features': self.n_detail_features,
                'detail_len': self.detail_len,
                'action_dim': self.action_dim,
            }
        }, path)

        if self.logger:
            self.logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        # weights_only=False for compatibility with numpy scalars in checkpoint
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'total_updates' in checkpoint:
            self.total_updates = checkpoint['total_updates']
        if 'reward_mean' in checkpoint:
            self.reward_mean = checkpoint['reward_mean']
            self.reward_std = checkpoint['reward_std']
            self.reward_count = checkpoint['reward_count']

        if self.logger:
            self.logger.info(f"Model loaded from {path}")
