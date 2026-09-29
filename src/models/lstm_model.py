"""
LSTM Classifier for Temporal NetFlow Benchmark v3.
Processes real temporal flow sequences of shape (batch_size, seq_len=10, input_size=input_dim).
Predicts the label of the current flow (last flow in the window) from its sequential history.
"""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM sequence classifier using real temporal flow windows.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: (batch, seq_len=10, input_dim)
        out, (hn, cn) = self.lstm(x)
        # Take representation at the final timestep (current flow state)
        last_step = out[:, -1, :]
        last_step = self.dropout(last_step)
        return self.fc(last_step)
