"""
Model B: Gaussian LSTM (Aleatoric Uncertainty)

Owner: Sufian (Lead / Team B)

Core contribution — outputs a probability distribution N(μ, σ²) instead of
a point prediction. Trained with Negative Log-Likelihood loss.

This captures ALEATORIC uncertainty: irreducible noise in the data.
E.g., Friday evening demand is inherently more variable than Tuesday 3am.

Interface Contract:
    Input:  (batch_size, seq_len, num_features)
    Output: dict with keys "mu" → (batch_size, 1), "sigma" → (batch_size, 1)
    Loss:   Gaussian NLL = 0.5 * log(σ²) + (y - μ)² / (2σ²)

Key implementation detail:
    σ must be positive. We output raw values and apply softplus: σ = log(1 + exp(raw))
    Do NOT use exp() — it's numerically unstable for large values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_lstm import BaseLSTM


class GaussianLSTM(nn.Module):
    """
    Gaussian LSTM — outputs mean (mu) and standard deviation (sigma).

    Args:
        input_size (int): Number of input features per timestep
        hidden_size (int): LSTM hidden dimension
        num_layers (int): Number of stacked LSTM layers
        dropout (float): Dropout rate applied between LSTM layers AND
            before the prediction heads. Default 0.0 (no dropout).
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.encoder = BaseLSTM(input_size, hidden_size, num_layers, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.mu_head = nn.Linear(hidden_size, 1)
        self.sigma_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            dict: {"mu": (batch_size, 1), "sigma": (batch_size, 1)}
                  sigma is guaranteed positive via softplus
        """
        h = self.encoder(x)                          # (batch, hidden_size)
        h = self.drop(h)                             # dropout before heads
        mu = self.mu_head(h)                          # (batch, 1)
        sigma = F.softplus(self.sigma_head(h)) + 1e-6 # (batch, 1), positive + stability
        return {"mu": mu, "sigma": sigma}


def gaussian_nll_loss(output: dict, target: torch.Tensor) -> torch.Tensor:
    """
    Gaussian Negative Log-Likelihood loss.

    NLL = 0.5 * log(σ²) + (y - μ)² / (2σ²)

    The tug-of-war:
        - (y - μ)² / (2σ²) : penalises wrong predictions, softened by large σ
        - 0.5 * log(σ²)    : penalises large σ (can't just say "I'm uncertain about everything")

    Args:
        output: dict with "mu" and "sigma" from model forward pass
        target: (batch_size, 1) true values
    Returns:
        scalar loss (mean over batch)
    """
    mu = output["mu"]
    sigma = output["sigma"]
    variance = sigma ** 2

    nll = 0.5 * torch.log(variance) + (target - mu) ** 2 / (2 * variance)
    return nll.mean()
