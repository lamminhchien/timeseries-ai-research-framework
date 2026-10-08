"""
Hybrid Temporal Feature Extractor Architecture.
Combines Dilated Causal Convolutional Networks (TCN) with a Multi-Head Transformer Encoder.
"""

from typing import List, Tuple
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence order preservation."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq_Len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ExtractorTransformer(nn.Module):
    """
    Dual-Stream Neural Network:
    1. Temporal Stream: Dilated causal Conv1d layers + Transformer Encoder.
    2. Context Stream: Dense projection of portfolio state vectors.
    3. Fused Output: Concatenated latent embedding ready for Actor-Critic policy heads.
    """

    def __init__(
        self,
        market_feature_dim: int = 8,
        seq_length: int = 60,
        account_dim: int = 4,
        tcn_channels: List[int] = None,
        tcn_kernel_size: int = 3,
        tfm_nhead: int = 4,
        tfm_layers: int = 2,
        tfm_dim_ff: int = 128,
        account_embed_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [32, 64]
        
        tcn_out_dim = tcn_channels[-1]
        self.kernel_pad = tcn_kernel_size - 1

        # Stream 1: Causal Temporal Convolution
        conv_layers = []
        in_c = market_feature_dim
        for out_c in tcn_channels:
            conv_layers.extend([
                nn.Conv1d(in_c, out_c, kernel_size=tcn_kernel_size, padding=self.kernel_pad),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_c = out_c
        self.tcn = nn.Sequential(*conv_layers)

        # Stream 1: Multi-Head Transformer Encoder
        self.pos_encoder = PositionalEncoding(tcn_out_dim, dropout=dropout, max_len=seq_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tcn_out_dim,
            nhead=tfm_nhead,
            dim_feedforward=tfm_dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=tfm_layers)

        # Stream 2: Account Vector Processor
        self.account_mlp = nn.Sequential(
            nn.Linear(account_dim, 64),
            nn.ReLU(),
            nn.Linear(64, account_embed_dim),
            nn.Tanh(),
        )

        self.features_dim = tcn_out_dim + account_embed_dim

    def forward(self, market_data: torch.Tensor, account_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        market_data:  [Batch, Seq_Len, Features]
        account_data: [Batch, Account_Dim]
        Returns:      [Batch, Features_Dim]
        """
        batch_size, seq_len, _ = market_data.shape

        # Causal Conv1d requires [Batch, Channels, Seq_Len]
        x = market_data.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)  # [Batch, Padded_Seq_Len, Channels]
        
        # Enforce exact causal horizon matching input sequence length
        x = x[:, :seq_len, :]

        # Positional Encoding + Attention Context
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        temporal_summary = x.mean(dim=1)  # Mean pooling across sequence

        # Context State Processing
        account_summary = self.account_mlp(account_data)

        # Joint Latent Fusion
        return torch.cat([temporal_summary, account_summary], dim=-1)


if __name__ == "__main__":
    # Smoke verification
    model = ExtractorTransformer(market_feature_dim=8, seq_length=60, account_dim=4)
    dummy_mkt = torch.randn(16, 60, 8)
    dummy_acc = torch.randn(16, 4)
    out = model(dummy_mkt, dummy_acc)
    print(f"ExtractorTransformer verification success: Output shape={out.shape}")
    assert out.shape == (16, 64 + 32)
