"""
Hybrid CNN-BiLSTM-Attention Network (Dual Input)
Supports Main Timeframe + Detail Timeframe inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

class DetailEncoder(nn.Module):
    """
    Encoder for Detail Timeframe Data (e.g., 1m candles)
    Processes a sequence of detail candles for each main timestep
    Optimized: Simplified architecture for better performance
    """
    def __init__(self, n_features: int, hidden_dim: int = 32):
        super().__init__()
        # Simplified: Single conv layer + pooling for efficiency
        self.conv1 = nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool to single vector per sequence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch * window, detail_len, features]
        x = x.transpose(1, 2)  # -> [B, features, len]
        x = F.relu(self.conv1(x))
        x = self.pool(x).squeeze(2)  # -> [B, hidden_dim]
        return x

class HybridQNetwork(nn.Module):
    def __init__(
        self,
        window_size: int,
        n_main_features: int,
        n_detail_features: int,
        detail_len: int,
        action_dim: int,
        cnn_channels: int = 48,  # Reduced from 64 to 48
        lstm_hidden: int = 64,  # Reduced from 128 to 64
        num_quantiles: int = 16  # Reduced from 32 to 16 for efficiency
    ):
        super().__init__()

        self.window_size = window_size
        self.num_quantiles = num_quantiles
        self.action_dim = action_dim

        # 1. Detail Branch
        self.detail_encoder = DetailEncoder(n_detail_features, hidden_dim=32)

        # 2. Main Branch (CNN) - Simplified to single layer
        # Input dim = Main Features + Detail Embedding (32)
        combined_features = n_main_features + 32

        self.main_cnn = nn.Sequential(
            nn.Conv1d(combined_features, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 3. LSTM Backbone - Reduced to 1 layer
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,  # Reduced from 2 to 1
            batch_first=True,
            bidirectional=True
        )

        # 4. Attention
        self.attention = nn.MultiheadAttention(lstm_hidden*2, num_heads=4, batch_first=True)

        # 5. Dueling Heads (QR-DQN) - Simplified
        self.fc_val = nn.Sequential(
            nn.Linear(lstm_hidden*2, 128),  # Reduced from 256
            nn.ReLU(),
            nn.Linear(128, num_quantiles)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(lstm_hidden*2, 128),  # Reduced from 256
            nn.ReLU(),
            nn.Linear(128, action_dim * num_quantiles)
        )

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        state['main']: [batch, window, main_features]
        state['detail']: [batch, window, detail_len, detail_features]
        """
        main = state['main']
        detail = state['detail']
        
        batch_size = main.shape[0]
        
        # Process Detail: Flatten batch and window
        # [batch * window, detail_len, features]
        detail_flat = detail.view(-1, detail.shape[2], detail.shape[3])
        detail_emb = self.detail_encoder(detail_flat)
        
        # Reshape back: [batch, window, 32]
        detail_emb = detail_emb.view(batch_size, self.window_size, -1)
        
        # Combine Main + Detail
        combined = torch.cat([main, detail_emb], dim=2) # [batch, window, combined_features]
        
        # CNN
        combined = combined.transpose(1, 2) # [batch, features, window]
        cnn_out = self.main_cnn(combined)
        cnn_out = cnn_out.transpose(1, 2) # [batch, window, channels]
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_out) # [batch, window, hidden*2]
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Aggregate (Last step)
        context = attn_out[:, -1, :] # [batch, hidden*2]
        
        # Dueling Heads
        value = self.fc_val(context).view(batch_size, 1, self.num_quantiles)
        adv = self.fc_adv(context).view(batch_size, self.action_dim, self.num_quantiles)
        
        q_dist = value + (adv - adv.mean(dim=1, keepdim=True))
        q_values = q_dist.mean(dim=2) # [batch, action_dim]
        
        return q_values
