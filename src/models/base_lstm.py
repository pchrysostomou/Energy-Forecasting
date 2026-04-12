"""
Shared LSTM backbone used by all three models.

All models inherit from this base class to ensure consistent architecture.
The only difference between models is what sits on TOP of this backbone.

Interface Contract:
    Input:  (batch_size, seq_len, num_features) — from DataLoader
    Output: (batch_size, hidden_size) — final hidden state, ready for prediction heads
"""

import torch
import torch.nn as nn


class BaseLSTM(nn.Module):
    """
    Shared LSTM encoder.

    Args:
        input_size (int): Number of input features per timestep
        hidden_size (int): LSTM hidden dimension
        num_layers (int): Number of stacked LSTM layers
        dropout (float): Dropout rate between LSTM layers (0.0 = no dropout)
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,           # Expect (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0,  # Dropout only between layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            h_final: (batch_size, hidden_size) — last timestep's hidden state
        """
        # lstm_out shape: (batch, seq_len, hidden_size)
        # We only need the final timestep's output
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_final = lstm_out[:, -1, :]  # (batch, hidden_size)
        return h_final
