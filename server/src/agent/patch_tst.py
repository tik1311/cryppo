"""
PatchTST (Patch Time Series Transformer) for Trading
Based on: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"

Key Innovations:
1. Patching: Segment time series into patches for efficient processing
2. Channel Independence: Process each feature channel independently
3. Instance Normalization: Better handling of distribution shifts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for patches"""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Patch Embedding Layer
    Converts time series into patches and projects them
    """

    def __init__(
        self,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 128,
        n_features: int = 1,
        padding: bool = True
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_features = n_features
        self.padding = padding

        # Linear projection for each patch
        # Input: [batch, n_patches, patch_len * n_features]
        # Output: [batch, n_patches, d_model]
        self.projection = nn.Linear(patch_len * n_features, d_model)

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: [batch, seq_len, n_features]
        Returns:
            patches: [batch, n_patches, d_model]
            n_patches: number of patches
        """
        batch_size, seq_len, n_features = x.shape

        # Padding if needed
        if self.padding:
            pad_len = (self.patch_len - seq_len % self.patch_len) % self.patch_len
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
                seq_len = seq_len + pad_len

        # Calculate number of patches
        n_patches = (seq_len - self.patch_len) // self.stride + 1

        # Create patches using unfold
        # [batch, seq_len, features] -> [batch, n_patches, patch_len, features]
        patches = x.unfold(1, self.patch_len, self.stride)  # [batch, n_patches, features, patch_len]
        patches = patches.permute(0, 1, 3, 2)  # [batch, n_patches, patch_len, features]
        patches = patches.reshape(batch_size, n_patches, -1)  # [batch, n_patches, patch_len * features]

        # Project patches
        patches = self.projection(patches)  # [batch, n_patches, d_model]
        patches = self.norm(patches)

        return patches, n_patches


class TransformerEncoderBlock(nn.Module):
    """Single Transformer Encoder Block with Pre-LayerNorm"""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization (Pre-LN for better training stability)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN Self-Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=src_mask)
        x = x + self.dropout(attn_out)

        # Pre-LN Feed-Forward
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        return x


class PatchTSTEncoder(nn.Module):
    """
    PatchTST Encoder
    Processes time series data using patch-based transformer
    """

    def __init__(
        self,
        seq_len: int = 36,
        n_features: int = 20,
        patch_len: int = 6,
        stride: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        channel_independence: bool = False  # Set False for trading (features interact)
    ):
        super().__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.channel_independence = channel_independence

        # Instance normalization for distribution shift handling
        self.instance_norm = nn.InstanceNorm1d(n_features, affine=True)

        # Patch embedding
        if channel_independence:
            # Process each feature independently
            self.patch_embed = PatchEmbedding(
                patch_len=patch_len,
                stride=stride,
                d_model=d_model,
                n_features=1
            )
        else:
            # Process all features together
            self.patch_embed = PatchEmbedding(
                patch_len=patch_len,
                stride=stride,
                d_model=d_model,
                n_features=n_features
            )

        # Calculate number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.n_patches + 10, dropout=dropout)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, n_features]
        Returns:
            features: [batch, d_model]
        """
        batch_size = x.shape[0]

        # Instance normalization (handle distribution shift)
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        x = self.instance_norm(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, features]

        if self.channel_independence:
            # Process each channel independently
            # [batch, seq_len, features] -> [batch * features, seq_len, 1]
            x = x.permute(0, 2, 1).reshape(-1, self.seq_len, 1)
            patches, _ = self.patch_embed(x)
            # Aggregate back: [batch * features, n_patches, d_model] -> [batch, features, n_patches, d_model]
            patches = patches.reshape(batch_size, self.n_features, -1, self.d_model)
            # Mean over features
            patches = patches.mean(dim=1)  # [batch, n_patches, d_model]
        else:
            # Process all features together
            patches, _ = self.patch_embed(x)  # [batch, n_patches, d_model]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patches = torch.cat([cls_tokens, patches], dim=1)  # [batch, n_patches + 1, d_model]

        # Add positional encoding
        patches = self.pos_encoding(patches)

        # Transformer encoder
        for layer in self.encoder_layers:
            patches = layer(patches)

        # Final normalization
        patches = self.final_norm(patches)

        # Use CLS token as representation
        features = patches[:, 0, :]  # [batch, d_model]

        return features


class DetailEncoder(nn.Module):
    """
    Encoder for detail (1m) timeframe data
    Uses Conv1D for efficient processing of high-frequency data
    """

    def __init__(
        self,
        detail_len: int = 60,
        n_features: int = 5,
        hidden_dim: int = 64,
        output_dim: int = 64
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, detail_len, n_features]
        Returns:
            features: [batch, output_dim]
        """
        x = x.permute(0, 2, 1)  # [batch, n_features, detail_len]
        x = self.conv(x)  # [batch, hidden_dim, 1]
        x = x.squeeze(-1)  # [batch, hidden_dim]
        x = self.fc(x)  # [batch, output_dim]
        return x


class PatchTSTActorCritic(nn.Module):
    """
    PatchTST-based Actor-Critic Network for PPO Trading

    Architecture:
    - PatchTST encoder for main timeframe (1h)
    - Conv encoder for detail timeframe (1m)
    - Shared layers for feature fusion
    - Separate Actor and Critic heads
    """

    def __init__(
        self,
        window_size: int = 36,
        n_main_features: int = 33,  # Including position info
        n_detail_features: int = 5,
        detail_len: int = 60,
        action_dim: int = 3,
        # PatchTST params
        patch_len: int = 6,
        stride: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.window_size = window_size
        self.n_main_features = n_main_features
        self.n_detail_features = n_detail_features
        self.detail_len = detail_len
        self.action_dim = action_dim

        # Main timeframe encoder (PatchTST)
        self.main_encoder = PatchTSTEncoder(
            seq_len=window_size,
            n_features=n_main_features,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            channel_independence=False  # Features should interact
        )

        # Detail timeframe encoder (Conv)
        self.detail_encoder = DetailEncoder(
            detail_len=detail_len,
            n_features=n_detail_features,
            hidden_dim=64,
            output_dim=64
        )

        # Feature fusion
        combined_dim = d_model + 64

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, action_dim)
        )

        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'actor' in name:
                    # Small init for actor (start with balanced actions)
                    nn.init.orthogonal_(module.weight, gain=0.01)
                elif 'critic' in name:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                else:
                    nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        state: dict,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: dict with 'main' [batch, window, features] and 'detail' [batch, window, detail_len, features]
            action_mask: [batch, action_dim] - 1 for valid actions, 0 for invalid
        Returns:
            action_logits: [batch, action_dim]
            value: [batch, 1]
        """
        main = state['main']  # [batch, window, features]
        detail = state['detail']  # [batch, window, detail_len, features]

        # Encode main timeframe with PatchTST
        main_features = self.main_encoder(main)  # [batch, d_model]

        # Encode detail timeframe (last window only)
        detail_last = detail[:, -1, :, :]  # [batch, detail_len, features]
        detail_features = self.detail_encoder(detail_last)  # [batch, 64]

        # Fuse features
        combined = torch.cat([main_features, detail_features], dim=-1)
        fused = self.fusion(combined)  # [batch, 128]

        # Actor output
        action_logits = self.actor(fused)

        # Apply action mask
        if action_mask is not None:
            action_logits = action_logits.masked_fill(action_mask == 0, -1e9)

        # Critic output
        value = self.critic(fused)

        return action_logits, value


# Testing
if __name__ == "__main__":
    # Test PatchTST
    batch_size = 4
    window_size = 36
    n_main_features = 33
    n_detail_features = 5
    detail_len = 60

    model = PatchTSTActorCritic(
        window_size=window_size,
        n_main_features=n_main_features,
        n_detail_features=n_detail_features,
        detail_len=detail_len
    )

    # Dummy input
    state = {
        'main': torch.randn(batch_size, window_size, n_main_features),
        'detail': torch.randn(batch_size, window_size, detail_len, n_detail_features)
    }
    action_mask = torch.ones(batch_size, 3)

    action_logits, value = model(state, action_mask)

    print(f"Action logits shape: {action_logits.shape}")  # [4, 3]
    print(f"Value shape: {value.shape}")  # [4, 1]
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
