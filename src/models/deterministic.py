"""
Model A: Deterministic LSTM (Baseline)

Owner: Abu (Team B)

This is the simplest model — standard point prediction.
Outputs a single value per timestep. Trained with MSE loss.
This serves as the CONTROL — we compare Models B and C against this.

Interface Contract:
    Input:  (batch_size, seq_len, num_features)
    Output: dict with key "prediction" → (batch_size, 1)
    Loss:   MSE between prediction and target

TODO (Abu):
    1. Inherit from BaseLSTM or use it as an encoder
    2. Add a linear head that maps hidden_size → 1
    3. Return output as {"prediction": tensor}
    4. Implement training loop or use shared train.py
    5. Target: get a working RMSE number on the energy dataset
"""

import torch
import torch.nn as nn
from .base_lstm import BaseLSTM


class DeterministicLSTM(nn.Module):
    """
    Deterministic LSTM — outputs a single point prediction.

    Args:
        input_size (int): Number of input features per timestep
        hidden_size (int): LSTM hidden dimension
        num_layers (int): Number of stacked LSTM layers
        dropout (float): Dropout rate applied between LSTM layers AND
            before the linear head. Default 0.0 (no dropout).
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.encoder = BaseLSTM(input_size, hidden_size, num_layers, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            dict: {"prediction": (batch_size, 1)}
        """
        h = self.encoder(x)             # (batch, hidden_size)
        h = self.drop(h)                # dropout before head
        prediction = self.head(h)        # (batch, 1)
        return {"prediction": prediction}


def deterministic_loss(output: dict, target: torch.Tensor) -> torch.Tensor:
    """
    Standard MSE loss.

    Args:
        output: dict from model forward pass
        target: (batch_size, 1) true values
    Returns:
        scalar loss
    """
    return nn.functional.mse_loss(output["prediction"], target)
