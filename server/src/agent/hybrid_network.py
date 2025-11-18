"""
Hybrid CNN-BiLSTM-Attention-QR-DQN Network for Crypto Trading
Based on 2024-2025 research best practices

Architecture:
1. Input (window_size, features) - temporal market data
2. 1D-CNN - extract local patterns and trends
3. Bi-LSTM - capture long-term temporal dependencies
4. Self-Attention - focus on critical moments
5. Dueling architecture - separate value and advantage
6. QR-DQN head - distributional RL for risk assessment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class CNNFeatureExtractor(nn.Module):
    """
    1D-CNN for extracting local patterns from time series
    Uses multiple kernel sizes to capture different temporal scales
    """
    def __init__(self, input_channels: int, hidden_channels: int = 64):
        super(CNNFeatureExtractor, self).__init__()

        # Multi-scale convolutions (different kernel sizes)
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3)

        # Combine multi-scale features
        self.combine = nn.Conv1d(hidden_channels * 3, hidden_channels * 2, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(hidden_channels * 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features]
        Returns:
            [batch, seq_len, hidden_channels*2]
        """
        # Transpose for Conv1d: [batch, features, seq_len]
        x = x.transpose(1, 2)

        # Multi-scale convolutions
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(x))
        c3 = F.relu(self.conv3(x))

        # Concatenate multi-scale features
        combined = torch.cat([c1, c2, c3], dim=1)

        # Combine and normalize
        out = self.combine(combined)
        out = self.bn1(out)
        out = self.dropout(F.relu(out))

        # Transpose back: [batch, seq_len, hidden_channels*2]
        return out.transpose(1, 2)


class SelfAttention(nn.Module):
    """
    Self-Attention mechanism to focus on important time steps
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super(SelfAttention, self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Multi-head attention components
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            mask: Optional [batch, seq_len] attention mask
        Returns:
            output: [batch, seq_len, hidden_dim]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections in batch
        Q = self.query(x)  # [batch, seq_len, hidden_dim]
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [batch, num_heads, seq_len, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        # [batch, num_heads, seq_len, seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # [batch, num_heads, seq_len, head_dim]

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Final linear projection
        output = self.out(context)

        return output, attention_weights


class HybridQNetwork(nn.Module):
    """
    Hybrid CNN-BiLSTM-Attention Network with Dueling Architecture
    and QR-DQN distributional head
    """
    def __init__(
        self,
        window_size: int,
        n_features: int,
        action_dim: int,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        num_quantiles: int = 32,
        num_attention_heads: int = 4
    ):
        super(HybridQNetwork, self).__init__()

        self.window_size = window_size
        self.n_features = n_features
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles

        # 1. CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(n_features, cnn_channels)
        cnn_output_dim = cnn_channels * 2

        # 2. Bi-LSTM Layer
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        lstm_output_dim = lstm_hidden * 2  # Bidirectional

        # 3. Self-Attention Layer
        self.attention = SelfAttention(lstm_output_dim, num_attention_heads)

        # Layer normalization after attention
        self.ln = nn.LayerNorm(lstm_output_dim)

        # 4. Aggregate temporal information
        # We'll use the last timestep + attention-weighted average
        aggregated_dim = lstm_output_dim * 2

        # 5. Dueling Architecture
        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(aggregated_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_quantiles)  # Output: num_quantiles for distributional RL
        )

        # Advantage stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(aggregated_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * num_quantiles)  # Output: action_dim * num_quantiles
        )

        # Initialize with trading bias
        self._init_advantage_bias()

    def _init_advantage_bias(self):
        """Initialize advantage to favor trading actions"""
        with torch.no_grad():
            final_layer = self.advantage_stream[-1]
            if hasattr(final_layer, 'bias') and final_layer.bias is not None:
                # Reshape to [action_dim, num_quantiles]
                bias = final_layer.bias.view(self.action_dim, self.num_quantiles)
                # HOLD = 0.0, LONG/SHORT = +0.3 across all quantiles
                bias[0, :] = 0.0  # HOLD
                if bias.shape[0] > 1:
                    bias[1, :] = 0.3  # LONG
                if bias.shape[0] > 2:
                    bias[2, :] = 0.3  # SHORT
                final_layer.bias.copy_(bias.view(-1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with CNN-BiLSTM-Attention

        Args:
            state: [batch, window_size, n_features] or [batch, window_size * n_features]

        Returns:
            Q-values: [batch, action_dim] (mean across quantiles for compatibility)
        """
        batch_size = state.shape[0]

        # Reshape if flattened
        if len(state.shape) == 2:
            state = state.view(batch_size, self.window_size, self.n_features)

        # 1. CNN feature extraction
        cnn_features = self.cnn(state)  # [batch, window_size, cnn_output_dim]

        # 2. Bi-LSTM temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        # lstm_out: [batch, window_size, lstm_hidden*2]

        # 3. Self-Attention
        attn_out, attn_weights = self.attention(lstm_out)
        attn_out = self.ln(attn_out + lstm_out)  # Residual connection

        # 4. Aggregate temporal information
        # Last timestep
        last_hidden = attn_out[:, -1, :]  # [batch, lstm_hidden*2]

        # Attention-weighted average
        attn_avg = torch.mean(attn_out, dim=1)  # [batch, lstm_hidden*2]

        # Concatenate
        features = torch.cat([last_hidden, attn_avg], dim=1)  # [batch, lstm_hidden*4]

        # 5. Dueling streams with QR-DQN
        value = self.value_stream(features)  # [batch, num_quantiles]
        advantage = self.advantage_stream(features)  # [batch, action_dim * num_quantiles]

        # Reshape advantage
        advantage = advantage.view(batch_size, self.action_dim, self.num_quantiles)

        # Expand value for broadcasting
        value = value.unsqueeze(1).expand_as(advantage)

        # Dueling formula: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_quantiles = value + (advantage - advantage_mean)
        # [batch, action_dim, num_quantiles]

        # For standard Q-learning compatibility, return mean across quantiles
        q_values = q_quantiles.mean(dim=2)  # [batch, action_dim]

        return q_values

    def forward_quantiles(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning full quantile distribution
        Used for QR-DQN training

        Returns:
            [batch, action_dim, num_quantiles]
        """
        batch_size = state.shape[0]

        # Reshape if flattened
        if len(state.shape) == 2:
            state = state.view(batch_size, self.window_size, self.n_features)

        # Same as forward but return full quantiles
        cnn_features = self.cnn(state)
        lstm_out, _ = self.lstm(cnn_features)
        attn_out, _ = self.attention(lstm_out)
        attn_out = self.ln(attn_out + lstm_out)

        last_hidden = attn_out[:, -1, :]
        attn_avg = torch.mean(attn_out, dim=1)
        features = torch.cat([last_hidden, attn_avg], dim=1)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        advantage = advantage.view(batch_size, self.action_dim, self.num_quantiles)
        value = value.unsqueeze(1).expand_as(advantage)

        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_quantiles = value + (advantage - advantage_mean)

        return q_quantiles
