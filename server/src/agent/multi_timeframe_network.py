"""
Multi-Timeframe Hybrid Network for Hierarchical RL Trading
Phase 6: Combines primary timeframe (macro) with 1m detail (micro) for better decisions

Architecture:
    Primary TF (5m/15m/1h) → CNN-BiLSTM-Attention → Macro features (trend, regime)
    Detail TF (1m) → Multi-Head CNN → Micro features (timing, momentum, spread)
    → Hierarchical Attention Fusion → Dueling QR-DQN Head → Q-values

Based on research:
- Multi-level deep Q-networks (Nature 2024)
- Hierarchical attention for multi-timeframe trading
- Soft attention weighting based on market conditions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DetailMicrostructureCNN(nn.Module):
    """
    Processes 1m detail candles for microstructure features
    Focuses on: timing, momentum, volume profile, spread
    """

    def __init__(self, detail_ratio: int, detail_features: int = 8):
        """
        Args:
            detail_ratio: Number of detail candles per primary (5, 15, 60)
            detail_features: Features per detail candle (OHLCV + indicators)
        """
        super().__init__()

        self.detail_ratio = detail_ratio
        self.detail_features = detail_features

        # Multi-scale 1D CNN for detail patterns
        # Captures patterns across detail candles (e.g., 5x 1m within 5m)
        self.conv1 = nn.Conv1d(detail_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(detail_features, 32, kernel_size=5, padding=2)

        # Combine multi-scale features
        self.combine = nn.Conv1d(64, 64, kernel_size=1)

        # Batch norm and dropout
        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.1)

        # Output projection
        self.fc = nn.Linear(64 * detail_ratio, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, detail_ratio, detail_features] - Detail candles
        Returns:
            micro_features: [batch, 128] - Microstructure features
        """
        # Transpose for Conv1d: [batch, features, time]
        x = x.transpose(1, 2)  # [batch, detail_features, detail_ratio]

        # Multi-scale convolutions
        c1 = F.relu(self.conv1(x))  # [batch, 32, detail_ratio]
        c2 = F.relu(self.conv2(x))  # [batch, 32, detail_ratio]

        # Concatenate
        combined = torch.cat([c1, c2], dim=1)  # [batch, 64, detail_ratio]

        # Combine and normalize
        combined = F.relu(self.combine(combined))
        combined = self.bn(combined)
        combined = self.dropout(combined)

        # Flatten
        flattened = combined.reshape(combined.size(0), -1)  # [batch, 64*detail_ratio]

        # Project to feature space
        micro_features = F.relu(self.fc(flattened))  # [batch, 128]

        return micro_features


class PrimaryMacroCNN(nn.Module):
    """
    Processes primary timeframe for macro trend and regime
    Uses CNN-BiLSTM-Attention (same as HybridQNetwork)
    """

    def __init__(self, window_size: int, n_features: int):
        super().__init__()

        # Multi-scale CNN for primary timeframe
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_features, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(n_features, 64, kernel_size=7, padding=3)

        # Combine multi-scale features
        self.combine = nn.Conv1d(192, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.1)

        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=256,  # BiLSTM output is 256 (128*2)
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(256)

        # Output: 256 (last timestep) + 256 (attention mean) = 512
        self.macro_feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, window_size, n_features] - Primary timeframe data
        Returns:
            macro_features: [batch, 512] - Macro trend and regime features
        """
        # Transpose for Conv1d
        x_conv = x.transpose(1, 2)  # [batch, features, time]

        # Multi-scale CNN
        c1 = F.relu(self.conv1(x_conv))
        c2 = F.relu(self.conv2(x_conv))
        c3 = F.relu(self.conv3(x_conv))

        # Concatenate and combine
        combined = torch.cat([c1, c2, c3], dim=1)  # [batch, 192, time]
        combined = F.relu(self.combine(combined))
        combined = self.bn1(combined)
        combined = self.dropout1(combined)

        # Transpose back for LSTM
        combined = combined.transpose(1, 2)  # [batch, time, 128]

        # BiLSTM
        lstm_out, _ = self.lstm(combined)  # [batch, time, 256]

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)  # Residual

        # Feature aggregation
        last_timestep = attn_out[:, -1, :]  # [batch, 256]
        attention_mean = attn_out.mean(dim=1)  # [batch, 256]

        # Concatenate
        macro_features = torch.cat([last_timestep, attention_mean], dim=1)  # [batch, 512]

        return macro_features


class HierarchicalAttentionFusion(nn.Module):
    """
    Fuses macro (primary) and micro (detail) features using soft attention
    Dynamically weights contributions based on market conditions
    """

    def __init__(self, macro_dim: int = 512, micro_dim: int = 128):
        super().__init__()

        # Project both to same dimension
        self.macro_proj = nn.Linear(macro_dim, 256)
        self.micro_proj = nn.Linear(micro_dim, 256)

        # Attention weights (learnable)
        self.attention_fc = nn.Sequential(
            nn.Linear(512, 128),  # Concat macro+micro
            nn.ReLU(),
            nn.Linear(128, 2),  # 2 weights: macro, micro
            nn.Softmax(dim=1)
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1)
        )

    def forward(self, macro_features: torch.Tensor, micro_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            macro_features: [batch, 512] - Primary timeframe features
            micro_features: [batch, 128] - Detail timeframe features
        Returns:
            fused_features: [batch, 512] - Hierarchically fused features
        """
        # Project to same dimension
        macro_proj = F.relu(self.macro_proj(macro_features))  # [batch, 256]
        micro_proj = F.relu(self.micro_proj(micro_features))  # [batch, 256]

        # Compute attention weights
        concat = torch.cat([macro_proj, micro_proj], dim=1)  # [batch, 512]
        weights = self.attention_fc(concat)  # [batch, 2] - softmax weights

        # Weighted combination
        macro_weight = weights[:, 0:1]  # [batch, 1]
        micro_weight = weights[:, 1:2]  # [batch, 1]

        weighted_macro = macro_proj * macro_weight  # [batch, 256]
        weighted_micro = micro_proj * micro_weight  # [batch, 256]

        # Concatenate weighted features
        combined = torch.cat([weighted_macro, weighted_micro], dim=1)  # [batch, 512]

        # Final fusion
        fused = self.fusion(combined)  # [batch, 512]

        return fused


class MultiTimeframeQNetwork(nn.Module):
    """
    Complete Multi-Timeframe Network for RL Trading

    Input:
        - Primary TF: [batch, window_size, n_features]
        - Detail TF: [batch, detail_ratio, detail_features]

    Output:
        - Q-values: [batch, n_actions, n_quantiles]
    """

    def __init__(
        self,
        primary_window: int,
        primary_features: int,
        detail_ratio: int,
        detail_features: int,
        action_dim: int = 3,
        n_quantiles: int = 32
    ):
        super().__init__()

        self.primary_window = primary_window
        self.primary_features = primary_features
        self.detail_ratio = detail_ratio
        self.detail_features = detail_features
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles

        # Primary timeframe processor (macro)
        self.primary_net = PrimaryMacroCNN(primary_window, primary_features)

        # Detail timeframe processor (micro)
        self.detail_net = DetailMicrostructureCNN(detail_ratio, detail_features)

        # Hierarchical fusion
        self.fusion = HierarchicalAttentionFusion(macro_dim=512, micro_dim=128)

        # Dueling QR-DQN head
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_quantiles)  # V(s) [batch, n_quantiles]
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * n_quantiles)  # A(s,a) [batch, actions*quantiles]
        )

    def forward(
        self,
        primary_state: torch.Tensor,
        detail_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            primary_state: [batch, window_size, n_features] - Primary TF
            detail_state: [batch, detail_ratio, detail_features] - Detail TF

        Returns:
            q_values: [batch, action_dim, n_quantiles] - Distributional Q-values
        """
        batch_size = primary_state.size(0)

        # Extract macro features from primary timeframe
        macro_features = self.primary_net(primary_state)  # [batch, 512]

        # Extract micro features from detail timeframe
        micro_features = self.detail_net(detail_state)  # [batch, 128]

        # Hierarchical fusion
        fused_features = self.fusion(macro_features, micro_features)  # [batch, 512]

        # Dueling QR-DQN
        value = self.value_stream(fused_features)  # [batch, n_quantiles]
        advantage = self.advantage_stream(fused_features)  # [batch, actions*quantiles]

        # Reshape advantage
        advantage = advantage.view(batch_size, self.action_dim, self.n_quantiles)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        value = value.unsqueeze(1)  # [batch, 1, n_quantiles]
        advantage_mean = advantage.mean(dim=1, keepdim=True)  # [batch, 1, n_quantiles]

        q_values = value + (advantage - advantage_mean)  # [batch, actions, quantiles]

        return q_values

    def get_q_values(self, primary_state: torch.Tensor, detail_state: torch.Tensor) -> torch.Tensor:
        """
        Get mean Q-values (for action selection)

        Returns:
            mean_q: [batch, action_dim] - Mean Q-value per action
        """
        q_dist = self.forward(primary_state, detail_state)  # [batch, actions, quantiles]
        mean_q = q_dist.mean(dim=2)  # [batch, actions]
        return mean_q


if __name__ == '__main__':
    # Test multi-timeframe network
    print("Testing Multi-Timeframe Q-Network\n")

    # Parameters
    batch_size = 8
    primary_window = 72
    primary_features = 27  # Market data + indicators
    detail_ratio = 5  # 5x 1m candles in 1x 5m candle
    detail_features = 8  # OHLCV + micro indicators
    action_dim = 3  # HOLD, LONG, SHORT
    n_quantiles = 32

    # Create network
    net = MultiTimeframeQNetwork(
        primary_window=primary_window,
        primary_features=primary_features,
        detail_ratio=detail_ratio,
        detail_features=detail_features,
        action_dim=action_dim,
        n_quantiles=n_quantiles
    )

    # Random inputs
    primary_state = torch.randn(batch_size, primary_window, primary_features)
    detail_state = torch.randn(batch_size, detail_ratio, detail_features)

    print(f"Primary state: {primary_state.shape}")
    print(f"Detail state: {detail_state.shape}")

    # Forward pass
    q_dist = net(primary_state, detail_state)
    mean_q = net.get_q_values(primary_state, detail_state)

    print(f"\nQ-distribution: {q_dist.shape} [batch, actions, quantiles]")
    print(f"Mean Q-values: {mean_q.shape} [batch, actions]")
    print(f"\nSample Q-values:")
    print(f"  HOLD:  {mean_q[0, 0].item():.4f}")
    print(f"  LONG:  {mean_q[0, 1].item():.4f}")
    print(f"  SHORT: {mean_q[0, 2].item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n✓ Multi-Timeframe Network test passed!")
