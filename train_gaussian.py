"""
Gaussian LSTM Training & Evaluation

Trains the GaussianLSTM model that outputs N(mu, sigma^2) predictions,
then runs full probabilistic evaluation on the test set.

Uses the winning config from the baseline sweep:
    - Train: 2015-2017, Val: 2018-2022, Test: 2023-2025
    - hidden=64, layers=2, dropout=0.2
    - Adam lr=0.001, weight_decay=1e-4
    - ReduceLROnPlateau(patience=5, factor=0.5)
    - include_temperature=True

Usage:
    python train_gaussian.py              # Full training + evaluation
    python train_gaussian.py --epochs 5   # Smoke test
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

from src.data_loader import data_preprocess, get_dataloaders
from src.models.gaussian import GaussianLSTM, gaussian_nll_loss
from src.evaluate import (
    compute_rmse, compute_nll, compute_crps, compute_calibration,
    plot_training_curves, plot_predictions, plot_calibration, plot_sigma_analysis,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
MODEL_PATH = os.path.join(SAVE_DIR, "gaussian_lstm.pt")
HISTORY_PATH = os.path.join(SAVE_DIR, "gaussian_training_history.json")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        total_loss += loss.item()
        n += 1
    return total_loss / n


@torch.no_grad()
def validate_rmse_mw(model, loader, device, scaler):
    """Compute RMSE in MW using mu as point prediction."""
    model.eval()
    all_mu, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        all_mu.append(out["mu"].cpu().numpy())
        all_y.append(y.numpy())
    mu = np.concatenate(all_mu).squeeze()
    yt = np.concatenate(all_y).squeeze()
    # De-normalise
    mu_mw = mu * scaler["std"] + scaler["mean"]
    yt_mw = yt * scaler["std"] + scaler["mean"]
    return compute_rmse(yt_mw, mu_mw)


@torch.no_grad()
def predict_all(model, loader, device):
    """Run inference, return (mu, sigma, targets) all as numpy arrays."""
    model.eval()
    all_mu, all_sigma, all_y = [], [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        all_mu.append(out["mu"].cpu().numpy())
        all_sigma.append(out["sigma"].cpu().numpy())
        all_y.append(y.numpy())
    return (np.concatenate(all_mu).squeeze(),
            np.concatenate(all_sigma).squeeze(),
            np.concatenate(all_y).squeeze())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Gaussian LSTM")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train-start", type=int, default=2015)
    parser.add_argument("--train-end", type=int, default=2017)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data (same as baseline winner) ---
    print("Loading data ...")
    train_df, valid_df, test_df, scaler = data_preprocess(
        train_start=args.train_start, train_end=args.train_end,
        include_temperature=True,
    )
    train_loader, valid_loader, test_loader = get_dataloaders(
        train_df, valid_df, test_df,
        seq_len=args.seq_len, batch_size=args.batch_size,
    )
    num_features = next(iter(train_loader))[0].shape[2]
    print(f"num_features: {num_features}, "
          f"train batches: {len(train_loader)}, val batches: {len(valid_loader)}")

    # --- Model ---
    model = GaussianLSTM(
        input_size=num_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"GaussianLSTM params: {total_params:,}")

    # --- Optimiser + Scheduler ---
    optimiser = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=5, factor=0.5,
    )

    # --- Training loop ---
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    criterion = gaussian_nll_loss
    best_val_nll = float("inf")
    best_state = None
    no_improve = 0
    history = {"train_nll": [], "val_nll": [], "val_rmse_mw": []}

    print(f"\n{'Ep':>3}  {'Train NLL':>11}  {'Val NLL':>11}  "
          f"{'Val RMSE MW':>12}  {'LR':>9}  Status")
    print("-" * 70)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_nll = train_one_epoch(model, train_loader, optimiser, criterion, device)
        val_nll = validate(model, valid_loader, criterion, device)
        val_rmse = validate_rmse_mw(model, valid_loader, device, scaler)

        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)
        history["val_rmse_mw"].append(float(val_rmse))

        scheduler.step(val_nll)
        lr_now = optimiser.param_groups[0]["lr"]

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            flag = "<- best"
        else:
            no_improve += 1
            flag = f"({no_improve}/{args.patience})"

        print(f"{epoch:>3}  {train_nll:>11.6f}  {val_nll:>11.6f}  "
              f"{val_rmse:>10.0f} MW  {lr_now:>9.1e}  {flag}")

        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    elapsed = time.time() - t0
    epochs_trained = len(history["train_nll"])
    print(f"Training complete in {elapsed:.0f}s ({epochs_trained} epochs)")

    # --- Save model + history ---
    torch.save(best_state, MODEL_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved model  -> {MODEL_PATH}")
    print(f"Saved history -> {HISTORY_PATH}")

    # ===================================================================
    # EVALUATION
    # ===================================================================
    print("\n" + "=" * 60)
    print("  EVALUATION ON TEST SET")
    print("=" * 60)

    # Restore best model
    model.load_state_dict(best_state)
    model.to(device)

    # --- Inference ---
    mu_norm, sigma_norm, targets_norm = predict_all(model, test_loader, device)

    # De-normalise
    mu_mw = mu_norm * scaler["std"] + scaler["mean"]
    sigma_mw = sigma_norm * scaler["std"]  # std scales linearly
    targets_mw = targets_norm * scaler["std"] + scaler["mean"]

    # --- Metrics ---
    test_rmse = compute_rmse(targets_mw, mu_mw)
    test_nll = compute_nll(targets_norm, mu_norm, sigma_norm)
    test_crps_norm = compute_crps(targets_norm, mu_norm, sigma_norm)
    test_crps_mw = test_crps_norm * scaler["std"]  # scale to MW

    cal = compute_calibration(targets_norm, mu_norm, sigma_norm)
    cal_error = float(np.mean(np.abs(
        np.array(cal["expected"]) - np.array(cal["observed"])
    )))

    # 90% PI width (z=1.645)
    pi90_width_mw = float(np.mean(2 * 1.645 * sigma_mw))

    # --- Summary table ---
    print(f"""
Gaussian LSTM Results
+------------------------+-------------+
| Metric                 | Value       |
+------------------------+-------------+
| Test RMSE (MW)         | {test_rmse:>9.0f}   |  (baseline: 5,014)
| Test NLL               | {test_nll:>9.4f}   |
| Test CRPS (MW)         | {test_crps_mw:>9.0f}   |
| Calibration Error      | {cal_error:>9.4f}   |
| Avg 90% PI Width (MW)  | {pi90_width_mw:>9.0f}   |
| Epochs                 | {epochs_trained:>9}   |
| Training Time          | {elapsed:>8.0f}s   |
+------------------------+-------------+
""")
    print("Calibration detail:")
    for exp, obs in zip(cal["expected"], cal["observed"]):
        diff = obs - exp
        print(f"  {exp:.0%} interval -> {obs:.1%} observed  "
              f"({'overconfident' if diff < -0.02 else 'underconfident' if diff > 0.02 else 'good'})")

    # ===================================================================
    # FIGURES
    # ===================================================================

    # Slice indices (same position as baseline for fair comparison)
    week = 336
    start = len(mu_mw) // 3

    # Figure 1: Training curve
    plot_training_curves(
        history["train_nll"], history["val_nll"],
        save_path=os.path.join(FIG_DIR, "gaussian_training_curve.png"),
        title="Gaussian LSTM -- Training Curve",
        ylabel="Loss (NLL)",
    )
    print("Saved: figures/gaussian_training_curve.png")

    # Figure 2: Predictions with uncertainty bands
    plot_predictions(
        targets_mw[start:start + week],
        mu_mw[start:start + week],
        sigma=sigma_mw[start:start + week],
        save_path=os.path.join(FIG_DIR, "gaussian_predictions_with_uncertainty.png"),
        title=f"Gaussian LSTM -- 1-Week Test (RMSE: {test_rmse:.0f} MW)",
    )
    print("Saved: figures/gaussian_predictions_with_uncertainty.png")

    # Figure 3: Calibration plot
    plot_calibration(
        cal["expected"], cal["observed"],
        save_path=os.path.join(FIG_DIR, "gaussian_calibration.png"),
        title="Gaussian LSTM -- Calibration",
    )
    print("Saved: figures/gaussian_calibration.png")

    # Figure 4: Sigma analysis
    plot_sigma_analysis(
        targets_mw[start:start + week],
        sigma_mw[start:start + week],
        save_path=os.path.join(FIG_DIR, "gaussian_sigma_analysis.png"),
        title="Gaussian LSTM -- Predicted Uncertainty vs Actual Demand",
    )
    print("Saved: figures/gaussian_sigma_analysis.png")


if __name__ == "__main__":
    main()
