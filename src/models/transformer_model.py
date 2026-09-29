"""
Transformer Classifier for Temporal NetFlow Benchmark v3.
Processes real temporal flow sequences of shape (batch_size, seq_len=10, input_size=input_dim).
Uses multi-head self-attention with sinusoidal positional encoding across flow history.
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence timesteps."""
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    """
    Compact Transformer Encoder for real temporal NetFlow sequences.
    2 layers, 4 heads, parameter count comparable to KAN (~40k-50k parameters).
    """
    def __init__(self, input_dim: int, num_classes: int, d_model: int = 48,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 96,
                 dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Project flow feature vector to d_model embedding space
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=64)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, seq_len=10, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        encoded = self.norm(encoded)
        # Representation of current flow at the final timestep
        last_step = encoded[:, -1, :]
        last_step = self.dropout(last_step)
        return self.fc(last_step)
