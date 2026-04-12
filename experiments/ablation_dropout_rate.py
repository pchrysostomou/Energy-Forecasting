"""
Ablation Study 3: MC Dropout Rate Sweep

Tests how dropout rate affects MC Dropout's uncertainty quality.
Compares dropout_rate = [0.1, 0.2, 0.3, 0.5].

Usage:
    python experiments/ablation_dropout_rate.py

Results saved to:
    - saved_models/ablation_dropout_rate_results.json
    - figures/ablation_dropout_rate.png
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import data_preprocess, get_dataloaders
from src.models.mc_dropout import MCDropoutLSTM, mc_dropout_loss
from src.evaluate import (
    compute_rmse, compute_nll, compute_crps, compute_calibration,
    find_temperature,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEQ_LEN = 336
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LR = 0.001
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10
TRAIN_START = 2015
TRAIN_END = 2017
MC_SAMPLES = 50

DROPOUT_RATES = [0.1, 0.2, 0.3, 0.5]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
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
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        total_loss += loss.item()
        n += 1
    return total_loss / n


def mc_predict_all(model, loader, device, n_samples=50):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load data once
    print("Loading data ...")
    train_df, valid_df, test_df, scaler = data_preprocess(
        train_start=TRAIN_START, train_end=TRAIN_END,
        include_temperature=True,
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, valid_df, test_df, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
    )
    num_features = next(iter(train_loader))[0].shape[2]

    results = []

    for dr in DROPOUT_RATES:
        print(f"\n{'='*60}")
        print(f"  MC DROPOUT RATE: {dr}")
        print(f"{'='*60}")

        # Create model with this dropout rate
        model = MCDropoutLSTM(
            input_size=num_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout_rate=dr,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

        # Training
        optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", patience=5, factor=0.5,
        )

        best_val = float("inf")
        best_state = None
        no_improve = 0

        t0 = time.time()
        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimiser, mc_dropout_loss, device)
            val_loss = validate(model, val_loader, mc_dropout_loss, device)
            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 5 == 0 or no_improve == 0:
                print(f"  Ep {epoch:>3}  Train={train_loss:.6f}  Val={val_loss:.6f}  "
                      f"{'<- best' if no_improve == 0 else f'({no_improve}/{PATIENCE})'}")

            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

        train_time = time.time() - t0

        # Load best model
        model.load_state_dict(best_state)
        model.to(device)

        # MC Dropout inference on test set
        print("  Running MC Dropout inference (T=50) ...")
        test_mu, test_sigma_raw, test_targets = mc_predict_all(
            model, test_loader, device, n_samples=MC_SAMPLES,
        )

        # De-normalise
        mu_mw = test_mu * scaler["std"] + scaler["mean"]
        sigma_raw_mw = test_sigma_raw * scaler["std"]
        targets_mw = test_targets * scaler["std"] + scaler["mean"]

        rmse = compute_rmse(targets_mw, mu_mw)
        raw_mean_sigma = float(np.mean(sigma_raw_mw))

        # Temperature scaling on validation set
        print("  Finding optimal temperature T ...")
        val_mu, val_sigma, val_targets = mc_predict_all(
            model, val_loader, device, n_samples=MC_SAMPLES,
        )
        T_opt, _, _ = find_temperature(val_targets, val_mu, val_sigma)

        # Apply T to test set
        test_sigma_cal = test_sigma_raw * T_opt
        sigma_cal_mw = test_sigma_cal * scaler["std"]

        nll_raw = compute_nll(test_targets, test_mu, test_sigma_raw)
        nll_cal = compute_nll(test_targets, test_mu, test_sigma_cal)
        crps_cal = compute_crps(test_targets, test_mu, test_sigma_cal) * scaler["std"]

        cal_raw = compute_calibration(test_targets, test_mu, test_sigma_raw)
        cal_err_raw = float(np.mean(np.abs(
            np.array(cal_raw["expected"]) - np.array(cal_raw["observed"])
        )))

        cal_after = compute_calibration(test_targets, test_mu, test_sigma_cal)
        cal_err_after = float(np.mean(np.abs(
            np.array(cal_after["expected"]) - np.array(cal_after["observed"])
        )))

        pi90_mw = float(np.mean(2 * 1.645 * sigma_cal_mw))

        print(f"\n  Dropout={dr} Results:")
        print(f"    RMSE:         {rmse:.0f} MW")
        print(f"    Raw mean σ:   {raw_mean_sigma:.0f} MW")
        print(f"    T optimal:    {T_opt:.2f}")
        print(f"    Cal err (raw):  {cal_err_raw:.3f}")
        print(f"    Cal err (T):    {cal_err_after:.3f}")
        print(f"    NLL (cal):    {nll_cal:.4f}")
        print(f"    CRPS (cal):   {crps_cal:.0f} MW")
        print(f"    90% PI width: {pi90_mw:.0f} MW")

        results.append({
            "dropout_rate": dr,
            "rmse_mw": round(rmse),
            "raw_mean_sigma_mw": round(raw_mean_sigma),
            "T_optimal": round(T_opt, 2),
            "cal_error_raw": round(cal_err_raw, 3),
            "cal_error_calibrated": round(cal_err_after, 3),
            "nll_calibrated": round(nll_cal, 4),
            "crps_mw": round(crps_cal),
            "pi90_width_mw": round(pi90_mw),
            "train_time_s": round(train_time),
        })

    # --- Save results ---
    results_path = os.path.join(SAVE_DIR, "ablation_dropout_rate_results.json")
   
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return super().default(obj)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\nSaved results -> {results_path}")

    # --- Generate figure ---
    rates = [r["dropout_rate"] for r in results]
    rmses = [r["rmse_mw"] for r in results]
    raw_sigmas = [r["raw_mean_sigma_mw"] for r in results]
    t_opts = [r["T_optimal"] for r in results]
    cal_errs = [r["cal_error_calibrated"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # RMSE vs dropout rate
    axes[0, 0].plot(rates, rmses, "o-", color="#1f77b4", linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("Dropout Rate")
    axes[0, 0].set_ylabel("Test RMSE (MW)")
    axes[0, 0].set_title("Point accuracy vs dropout rate")
    axes[0, 0].set_xticks(rates)
    axes[0, 0].grid(True, alpha=0.3)
    for x, y in zip(rates, rmses):
        axes[0, 0].annotate(f"{y:,}", (x, y), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=9)

    # Raw sigma vs dropout rate
    axes[0, 1].plot(rates, raw_sigmas, "s-", color="#d62728", linewidth=2, markersize=8)
    axes[0, 1].set_xlabel("Dropout Rate")
    axes[0, 1].set_ylabel("Mean Raw σ (MW)")
    axes[0, 1].set_title("Raw uncertainty vs dropout rate")
    axes[0, 1].set_xticks(rates)
    axes[0, 1].grid(True, alpha=0.3)
    for x, y in zip(rates, raw_sigmas):
        axes[0, 1].annotate(f"{y:,}", (x, y), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=9)

    # T needed vs dropout rate
    axes[1, 0].plot(rates, t_opts, "^-", color="#2ca02c", linewidth=2, markersize=8)
    axes[1, 0].set_xlabel("Dropout Rate")
    axes[1, 0].set_ylabel("Optimal T")
    axes[1, 0].set_title("Temperature T needed vs dropout rate")
    axes[1, 0].set_xticks(rates)
    axes[1, 0].grid(True, alpha=0.3)
    for x, y in zip(rates, t_opts):
        axes[1, 0].annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=9)

    # Calibration error vs dropout rate
    axes[1, 1].plot(rates, cal_errs, "D-", color="#9467bd", linewidth=2, markersize=8)
    axes[1, 1].set_xlabel("Dropout Rate")
    axes[1, 1].set_ylabel("Calibration Error (after T)")
    axes[1, 1].set_title("Calibration quality vs dropout rate")
    axes[1, 1].set_xticks(rates)
    axes[1, 1].grid(True, alpha=0.3)
    for x, y in zip(rates, cal_errs):
        axes[1, 1].annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=9)

    fig.suptitle("Ablation Study 3: MC Dropout Rate Sweep", fontsize=13, y=1.02)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "ablation_dropout_rate.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {fig_path}")

    # --- Print summary table ---
    print(f"\n{'='*80}")
    print(f"  MC DROPOUT RATE ABLATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Rate':>6} {'RMSE(MW)':>10} {'Raw σ(MW)':>10} {'T opt':>8} {'Cal.Err(raw)':>12} {'Cal.Err(T)':>12} {'CRPS(MW)':>10}")
    print(f"{'-'*68}")
    for r in results:
        marker = " <-- current" if r["dropout_rate"] == 0.2 else ""
        print(f"{r['dropout_rate']:>6.1f} {r['rmse_mw']:>8,} {r['raw_mean_sigma_mw']:>8,} "
              f"{r['T_optimal']:>8.2f} {r['cal_error_raw']:>12.3f} {r['cal_error_calibrated']:>12.3f} "
              f"{r['crps_mw']:>8,}{marker}")


if __name__ == "__main__":
    main()
