"""
Main Training Script -- Deterministic LSTM Baseline

# GenAI statement: This script was assisted by Claude (Anthropic).
# All technical decisions and model implementations were verified by the team.

Trains the DeterministicLSTM model on UK energy demand data and saves:
    - Best model weights   -> saved_models/deterministic_lstm.pt
    - Training history     -> saved_models/training_history.json
    - Scaler parameters    -> saved_models/scaler_params.json

Usage:
    python train_deterministic.py
"""

import argparse
import json
import math
import os
import time

import torch

from src.data_loader import data_preprocess, get_dataloaders
from src.models.deterministic import DeterministicLSTM, deterministic_loss

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
MODEL_PATH = os.path.join(SAVE_DIR, "deterministic_lstm.pt")
HISTORY_PATH = os.path.join(SAVE_DIR, "training_history.json")
SCALER_PATH = os.path.join(SAVE_DIR, "scaler_params.json")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimiser, criterion, device):
    """Run one training epoch. Returns average loss over all batches."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation. Returns average loss over all batches."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Deterministic LSTM")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--hidden-size", type=int, default=64)   # C8 config
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--no-temperature", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    include_temp = not args.no_temperature
    print(f"Loading data (include_temperature={include_temp}) ...")
    train_df, valid_df, test_df, scaler_params = data_preprocess(
        include_temperature=include_temp,
        train_start=2015,   # C8 best config
        train_end=2017,     # C8 best config
    )
    train_loader, valid_loader, test_loader = get_dataloaders(
        train_df, valid_df, test_df,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    x_sample, y_sample = next(iter(train_loader))
    num_features = x_sample.shape[2]
    print(f"num_features: {num_features}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(valid_loader)}")
    print(f"Scaler: mean={scaler_params['mean']:.2f}, std={scaler_params['std']:.2f}")

    # --- Model ---
    model = DeterministicLSTM(
        input_size=num_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=0.2,         
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # --- Optimiser & Loss ---
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = deterministic_loss

    # --- Training loop ---
    os.makedirs(SAVE_DIR, exist_ok=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"\n{'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>12}  {'Val RMSE':>10}  {'Status'}")
    print("-" * 60)

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        val_loss = validate(model, valid_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        val_rmse = math.sqrt(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            status = "<- saved"
        else:
            epochs_no_improve += 1
            status = f"no improve {epochs_no_improve}/{args.patience}"

        print(f"{epoch:>5}  {train_loss:>12.6f}  {val_loss:>12.6f}  {val_rmse:>10.6f}  {status}")

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed:.1f}s")

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    with open(SCALER_PATH, "w") as f:
        json.dump(scaler_params, f, indent=2)

    best_val_rmse_mw = math.sqrt(best_val_loss) * scaler_params["std"]

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Best val RMSE (MW): {best_val_rmse_mw:.2f}")
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()