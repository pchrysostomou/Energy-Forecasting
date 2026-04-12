"""
Model C: MC Dropout LSTM (Epistemic Uncertainty)

Owner: Sherry (Team B)

Same architecture as the Deterministic LSTM, but with dropout layers
that STAY ON at test time. Running T forward passes gives T different
predictions — the spread approximates epistemic (model) uncertainty.

This captures EPISTEMIC uncertainty: what the model doesn't know.
E.g., a once-in-a-decade cold snap the model hasn't seen before.

Interface Contract:
    Input:  (batch_size, seq_len, num_features)
    Output (training): dict with key "prediction" → (batch_size, 1)
    Output (inference): dict with keys:
        "prediction" → (batch_size, 1)  — mean of T forward passes
        "uncertainty" → (batch_size, 1) — std of T forward passes
    Loss:   MSE (same as deterministic — dropout is the only difference)

Key implementation detail:
    At test time, call model.train() to keep dropout active, or manually
    set dropout layers to training mode while keeping batchnorm (if any) in eval.

TODO (Sherry):
    1. Build on BaseLSTM but with dropout in the encoder AND between encoder/head
    2. Implement predict_with_uncertainty() method that runs T forward passes
    3. Return mean and std of predictions
    4. T=50 is standard — this is a hyperparameter worth testing
"""

import torch
import torch.nn as nn
from .base_lstm import BaseLSTM


class MCDropoutLSTM(nn.Module):
    """
    MC Dropout LSTM — deterministic at train time, stochastic at test time.

    Args:
        input_size (int): Number of input features per timestep
        hidden_size (int): LSTM hidden dimension
        num_layers (int): Number of stacked LSTM layers
        dropout_rate (float): Dropout probability (applied between LSTM layers AND before head)
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        self.encoder = BaseLSTM(input_size, hidden_size, num_layers, dropout=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)  # Additional dropout before head
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Single forward pass. Dropout is active in train mode,
        and can be forced on in eval mode for MC sampling.

        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            dict: {"prediction": (batch_size, 1)}
        """
        h = self.encoder(x)              # (batch, hidden_size)
        h = self.dropout(h)              # Dropout before prediction head
        prediction = self.head(h)         # (batch, 1)
        return {"prediction": prediction}

    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 50) -> dict:
        """
        MC Dropout inference: run n_samples forward passes with dropout ON,
        then compute mean and std of predictions.

        Args:
            x: (batch_size, seq_len, input_size)
            n_samples: number of stochastic forward passes (default 50)
        Returns:
            dict: {
                "prediction": (batch_size, 1) — mean prediction,
                "uncertainty": (batch_size, 1) — std across samples (epistemic uncertainty)
            }
        """
        # Enable dropout during inference
        self.train()  # Keeps dropout active

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(x)
                predictions.append(output["prediction"])

        # Stack: (n_samples, batch_size, 1)
        predictions = torch.stack(predictions, dim=0)

        return {
            "prediction": predictions.mean(dim=0),    # (batch, 1)
            "uncertainty": predictions.std(dim=0),     # (batch, 1)
        }


def mc_dropout_loss(output: dict, target: torch.Tensor) -> torch.Tensor:
    """
    Standard MSE loss — same as deterministic model.
    The uncertainty comes from dropout at TEST time, not from the loss.
    """
    return nn.functional.mse_loss(output["prediction"], target)
