"""
MC Dropout LSTM Training & Evaluation

Same architecture as the deterministic baseline — the only difference is
dropout stays ON at test time with T forward passes to estimate epistemic
uncertainty.

Usage:
    python train_mc_dropout.py              # Full training + evaluation
    python train_mc_dropout.py --epochs 5   # Smoke test
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import data_preprocess, get_dataloaders
from src.models.mc_dropout import MCDropoutLSTM, mc_dropout_loss
from src.evaluate import (
    compute_rmse, compute_nll, compute_crps, compute_calibration,
    find_temperature, plot_training_curves, plot_predictions,
    plot_calibration,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
MODEL_PATH = os.path.join(SAVE_DIR, "mc_dropout_lstm.pt")
HISTORY_PATH = os.path.join(SAVE_DIR, "mc_dropout_training_history.json")


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


def mc_predict_all(model, loader, device, n_samples=50):
    """Run MC Dropout inference on entire dataset.

    Returns (mu, sigma, targets) as numpy arrays, all shape (N,).
    mu = mean of T forward passes, sigma = std of T forward passes.
    """
    all_mu, all_sigma, all_y = [], [], []
    for x, y in loader:
        x = x.to(device)
        out = model.predict_with_uncertainty(x, n_samples=n_samples)
        all_mu.append(out["prediction"].cpu().numpy())
        all_sigma.append(out["uncertainty"].cpu().numpy())
        all_y.append(y.numpy())
    return (np.concatenate(all_mu).squeeze(),
            np.concatenate(all_sigma).squeeze(),
            np.concatenate(all_y).squeeze())


def eval_metrics(y_true, mu, sigma, scaler):
    """Compute all probabilistic metrics."""
    mu_mw = mu * scaler["std"] + scaler["mean"]
    sigma_mw = sigma * scaler["std"]
    targets_mw = y_true * scaler["std"] + scaler["mean"]

    rmse = compute_rmse(targets_mw, mu_mw)
    nll = compute_nll(y_true, mu, sigma)
    crps_mw = compute_crps(y_true, mu, sigma) * scaler["std"]
    cal = compute_calibration(y_true, mu, sigma)
    cal_err = float(np.mean(np.abs(
        np.array(cal["expected"]) - np.array(cal["observed"])
    )))
    pi90_mw = float(np.mean(2 * 1.645 * sigma_mw))

    return {
        "rmse_mw": rmse, "nll": nll, "crps_mw": crps_mw,
        "cal_error": cal_err, "pi90_width_mw": pi90_mw, "calibration": cal,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train MC Dropout LSTM")
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
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
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
    model = MCDropoutLSTM(
        input_size=num_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MCDropoutLSTM params: {total_params:,}")

    # --- Optimiser + Scheduler ---
    optimiser = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=5, factor=0.5,
    )

    # ===================================================================
    # TRAINING
    # ===================================================================
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    criterion = mc_dropout_loss
    best_val = float("inf")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"\n{'Ep':>3}  {'Train MSE':>11}  {'Val MSE':>11}  "
          f"{'Val RMSE MW':>12}  {'LR':>9}  Status")
    print("-" * 70)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        val_loss = validate(model, valid_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        val_rmse_mw = math.sqrt(val_loss) * scaler["std"]
        scheduler.step(val_loss)
        lr_now = optimiser.param_groups[0]["lr"]

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            flag = "<- best"
        else:
            no_improve += 1
            flag = f"({no_improve}/{args.patience})"

        print(f"{epoch:>3}  {train_loss:>11.6f}  {val_loss:>11.6f}  "
              f"{val_rmse_mw:>10.0f} MW  {lr_now:>9.1e}  {flag}")

        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    train_time = time.time() - t0
    epochs_trained = len(history["train_loss"])
    print(f"Training complete in {train_time:.0f}s ({epochs_trained} epochs)")

    # Save
    torch.save(best_state, MODEL_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved model   -> {MODEL_PATH}")
    print(f"Saved history -> {HISTORY_PATH}")

    # Restore best
    model.load_state_dict(best_state)
    model.to(device)

    # ===================================================================
    # FIGURE 1: Training curve
    # ===================================================================
    plot_training_curves(
        history["train_loss"], history["val_loss"],
        save_path=os.path.join(FIG_DIR, "mc_dropout_training_curve.png"),
        title="MC Dropout LSTM -- Training Curve",
        ylabel="Loss (MSE)",
    )
    print("Saved: figures/mc_dropout_training_curve.png")

    # ===================================================================
    # T SWEEP — How many forward passes do you need?
    # ===================================================================
    print(f"\n{'='*60}")
    print("  T SWEEP: Forward pass count vs uncertainty stability")
    print(f"{'='*60}")

    T_values = [5, 10, 25, 50, 100]
    sweep_rmse = []
    sweep_mean_sigma = []
    sweep_time = []

    for T in T_values:
        t_start = time.time()
        mu_n, sig_n, tgt_n = mc_predict_all(model, test_loader, device, n_samples=T)
        t_elapsed = time.time() - t_start

        mu_mw = mu_n * scaler["std"] + scaler["mean"]
        sig_mw = sig_n * scaler["std"]
        tgt_mw = tgt_n * scaler["std"] + scaler["mean"]

        rmse = compute_rmse(tgt_mw, mu_mw)
        mean_sig = float(np.mean(sig_mw))
        sweep_rmse.append(rmse)
        sweep_mean_sigma.append(mean_sig)
        sweep_time.append(t_elapsed)

        print(f"  T={T:>3}  RMSE={rmse:>7.0f} MW  "
              f"Mean sigma={mean_sig:>7.0f} MW  Time={t_elapsed:>5.1f}s")

    # Keep T=50 results for main evaluation
    # (they were computed during the sweep)
    idx_50 = T_values.index(50)

    # --- FIGURE 4: T sweep plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(T_values, sweep_rmse, "o-", color="#1f77b4", linewidth=1.5, markersize=6)
    ax1.set_xlabel("Number of forward passes (T)", fontsize=11)
    ax1.set_ylabel("Test RMSE (MW)", fontsize=11)
    ax1.set_title("RMSE vs T", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(T_values, sweep_mean_sigma, "s-", color="#d62728", linewidth=1.5, markersize=6)
    ax2.set_xlabel("Number of forward passes (T)", fontsize=11)
    ax2.set_ylabel("Mean predicted sigma (MW)", fontsize=11)
    ax2.set_title("Uncertainty stability vs T", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("MC Dropout -- Forward Pass Sweep", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mc_dropout_T_sweep.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/mc_dropout_T_sweep.png")

    # T sweep table
    print(f"\nT Sweep Results")
    print(f"+-----+-----------+------------+----------------+")
    print(f"|  T  | RMSE (MW) | Mean s (MW)| Inference Time |")
    print(f"+-----+-----------+------------+----------------+")
    for T, r, s, t in zip(T_values, sweep_rmse, sweep_mean_sigma, sweep_time):
        print(f"| {T:>3} | {r:>7.0f}   | {s:>8.0f}   | {t:>11.1f}s   |")
    print(f"+-----+-----------+------------+----------------+")

    # ===================================================================
    # MAIN EVALUATION (T=50)
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"  EVALUATION ON TEST SET (T={args.n_samples})")
    print(f"{'='*60}")

    print("Running MC Dropout inference (T=50) ...")
    test_mu, test_sigma_raw, test_targets = mc_predict_all(
        model, test_loader, device, n_samples=args.n_samples,
    )

    before = eval_metrics(test_targets, test_mu, test_sigma_raw, scaler)

    # --- Temperature scaling recalibration ---
    print("Running validation inference for temperature scaling ...")
    val_mu, val_sigma, val_targets = mc_predict_all(
        model, valid_loader, device, n_samples=args.n_samples,
    )

    print("Finding optimal temperature T ...")
    T_opt, T_grid, nll_grid = find_temperature(val_targets, val_mu, val_sigma)
    print(f"Optimal T = {T_opt:.4f}")

    with open(os.path.join(SAVE_DIR, "mc_dropout_temperature_T.json"), "w") as f:
        json.dump({"T": T_opt}, f, indent=2)

    test_sigma_cal = test_sigma_raw * T_opt
    after = eval_metrics(test_targets, test_mu, test_sigma_cal, scaler)

    # --- Summary table ---
    print(f"""
MC Dropout LSTM Results (T={args.n_samples} forward passes)
+-------------------+----------------------+---------------------+
| Metric            | Raw                  | Calibrated (T={T_opt:.2f}) |
+-------------------+----------------------+---------------------+
| Test RMSE (MW)    | {before['rmse_mw']:>8.0f}               | {after['rmse_mw']:>8.0f}              |
| Test NLL          | {before['nll']:>8.4f}               | {after['nll']:>8.4f}              |
| Test CRPS (MW)    | {before['crps_mw']:>8.0f}               | {after['crps_mw']:>8.0f}              |
| Calibration Error | {before['cal_error']:>8.4f}               | {after['cal_error']:>8.4f}              |
| 90% PI Width (MW) | {before['pi90_width_mw']:>8.0f}               | {after['pi90_width_mw']:>8.0f}              |
+-------------------+----------------------+---------------------+
""")

    print("Calibration detail (after recalibration):")
    for exp, obs in zip(after["calibration"]["expected"],
                        after["calibration"]["observed"]):
        diff = obs - exp
        label = ("overconfident" if diff < -0.02
                 else "underconfident" if diff > 0.02 else "good")
        print(f"  {exp:.0%} interval -> {obs:.1%} observed  ({label})")

    # ===================================================================
    # FIGURES
    # ===================================================================
    mu_mw = test_mu * scaler["std"] + scaler["mean"]
    sigma_raw_mw = test_sigma_raw * scaler["std"]
    sigma_cal_mw = test_sigma_cal * scaler["std"]
    targets_mw = test_targets * scaler["std"] + scaler["mean"]

    week = 336
    start = len(mu_mw) // 3

    # --- FIGURE 2: Predictions with uncertainty (raw) ---
    plot_predictions(
        targets_mw[start:start + week],
        mu_mw[start:start + week],
        sigma=sigma_cal_mw[start:start + week],
        save_path=os.path.join(FIG_DIR,
                               "mc_dropout_predictions_with_uncertainty.png"),
        title=f"MC Dropout LSTM (T={args.n_samples}, recal.) -- "
              f"RMSE: {after['rmse_mw']:.0f} MW",
    )
    print("Saved: figures/mc_dropout_predictions_with_uncertainty.png")

    # --- FIGURE 3: Calibration comparison (before & after) ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.plot(before["calibration"]["expected"],
            before["calibration"]["observed"],
            "o-", color="#d62728", linewidth=1.5, markersize=5,
            label=f"Before (err={before['cal_error']:.3f})")
    ax.plot(after["calibration"]["expected"],
            after["calibration"]["observed"],
            "s-", color="#2ca02c", linewidth=1.5, markersize=5,
            label=f"After T={T_opt:.2f} (err={after['cal_error']:.3f})")
    ax.set_xlabel("Expected coverage", fontsize=12)
    ax.set_ylabel("Observed coverage", fontsize=12)
    ax.set_title("MC Dropout -- Calibration Before & After", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mc_dropout_calibration.png"), dpi=200)
    plt.close(fig)
    print("Saved: figures/mc_dropout_calibration.png")

    # --- FIGURE 5: High vs Low uncertainty analysis ---
    n_examples = 100

    # Indices of highest and lowest uncertainty
    high_idx = np.argsort(sigma_raw_mw)[-n_examples:]
    low_idx = np.argsort(sigma_raw_mw)[:n_examples]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Plot: overlay high-uncertainty samples on the full test demand
    ax1.plot(targets_mw, color="#1f77b4", linewidth=0.3, alpha=0.5,
             label="Test demand")
    ax1.scatter(high_idx, targets_mw[high_idx], color="#d62728", s=8,
                zorder=3, label=f"Top {n_examples} high-uncertainty")
    ax1.set_ylabel("Demand (MW)", fontsize=11)
    ax1.set_title("High Epistemic Uncertainty Samples", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(targets_mw, color="#1f77b4", linewidth=0.3, alpha=0.5,
             label="Test demand")
    ax2.scatter(low_idx, targets_mw[low_idx], color="#2ca02c", s=8,
                zorder=3, label=f"Top {n_examples} low-uncertainty")
    ax2.set_xlabel("Test sample index", fontsize=11)
    ax2.set_ylabel("Demand (MW)", fontsize=11)
    ax2.set_title("Low Epistemic Uncertainty Samples", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("MC Dropout -- Epistemic Uncertainty Analysis", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,
                             "mc_dropout_high_vs_low_uncertainty.png"), dpi=200)
    plt.close(fig)
    print("Saved: figures/mc_dropout_high_vs_low_uncertainty.png")

    # Print high/low uncertainty stats
    print(f"\nEpistemic uncertainty analysis:")
    print(f"  High-uncertainty samples: mean demand = {targets_mw[high_idx].mean():.0f} MW, "
          f"mean sigma = {sigma_raw_mw[high_idx].mean():.0f} MW")
    print(f"  Low-uncertainty samples:  mean demand = {targets_mw[low_idx].mean():.0f} MW, "
          f"mean sigma = {sigma_raw_mw[low_idx].mean():.0f} MW")


if __name__ == "__main__":
    main()
