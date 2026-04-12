"""
Ablation Study 5: Explicit Lag Demand Features

Tests how adding explicit lagged demand values affects forecast quality.
Four configurations are compared across all three models:

    A: No lags       — baseline (temporal features + temperature only)
    B: t-48          — yesterday same half-hour
    C: t-48, t-336   — yesterday + last week same half-hour
    D: t-48, t-336, t-1440, t-17520 — + last month + last year

Motivation: Energy demand is highly periodic. The LSTM sees 336 steps of
history but must learn to attend to the right timesteps implicitly. Explicit
lag features give the model direct access to the strongest autocorrelation
signals and may tighten uncertainty estimates.

Usage:
    python experiments/ablation_lag_features.py

Results saved to:
    - saved_models/ablation_lag_features_results.json
    - figures/ablation_lag_features.png
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import data_preprocess, get_dataloaders
from src.models.deterministic import DeterministicLSTM, deterministic_loss
from src.models.gaussian import GaussianLSTM, gaussian_nll_loss
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
LAG_CONFIGS = [
    {"label": "A: No lags",              "lag_periods": None},
    {"label": "B: + Yesterday",          "lag_periods": [48]},
    {"label": "C: + Last week",          "lag_periods": [48, 336]},
    {"label": "D: + Month & Year",       "lag_periods": [48, 336, 1440, 17520]},
]

SEQ_LEN      = 336
BATCH_SIZE   = 64
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.2
LR           = 0.001
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 100
PATIENCE     = 10
MC_SAMPLES   = 50
TRAIN_START  = 2015
TRAIN_END    = 2017


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
def validate_loss(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        total_loss += loss.item()
        n += 1
    return total_loss / n


def train_model(model, train_loader, val_loader, criterion, device, label=""):
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=5, factor=0.5
    )
    best_val = float("inf")
    best_state = None
    no_improve = 0

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        val_loss = validate_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            flag = "<- best"
        else:
            no_improve += 1
            flag = f"({no_improve}/{PATIENCE})"

        if epoch % 5 == 0 or epoch == 1 or no_improve == 0:
            print(f"  [{label}] Ep {epoch:>3}  Train={train_loss:.6f}  Val={val_loss:.6f}  {flag}")

        if no_improve >= PATIENCE:
            print(f"  [{label}] Early stopping at epoch {epoch}")
            break

    return best_state, time.time() - t0


@torch.no_grad()
def predict_deterministic(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        preds.append(out["prediction"].cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds).squeeze(), np.concatenate(targets).squeeze()


@torch.no_grad()
def predict_gaussian(model, loader, device):
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


def predict_mc_dropout(model, loader, device, n_samples=50):
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

    results = []

    for cfg in LAG_CONFIGS:
        label       = cfg["label"]
        lag_periods = cfg["lag_periods"]

        print(f"\n{'='*60}")
        print(f"  LAG CONFIG: {label}")
        print(f"  lag_periods: {lag_periods}")
        print(f"{'='*60}")

        print("Loading data ...")
        train_df, valid_df, test_df, scaler = data_preprocess(
            train_start=TRAIN_START, train_end=TRAIN_END,
            include_temperature=True,
            lag_periods=lag_periods,
        )
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, valid_df, test_df, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        )
        num_features = next(iter(train_loader))[0].shape[2]
        print(f"num_features: {num_features}")

        # ----------------------------------------------------------------
        # Model A: Deterministic
        # ----------------------------------------------------------------
        print(f"\n--- Deterministic LSTM [{label}] ---")
        model_a = DeterministicLSTM(
            input_size=num_features, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, dropout=DROPOUT,
        ).to(device)

        state_a, time_a = train_model(
            model_a, train_loader, val_loader, deterministic_loss, device,
            label=f"Det-{label}"
        )
        model_a.load_state_dict(state_a)
        model_a.to(device)

        preds_a, targets_a = predict_deterministic(model_a, test_loader, device)
        rmse_a = compute_rmse(
            targets_a * scaler["std"] + scaler["mean"],
            preds_a   * scaler["std"] + scaler["mean"],
        )
        print(f"  Deterministic RMSE: {rmse_a:.0f} MW  ({time_a:.0f}s)")

        # ----------------------------------------------------------------
        # Model B: Gaussian
        # ----------------------------------------------------------------
        print(f"\n--- Gaussian LSTM [{label}] ---")
        model_b = GaussianLSTM(
            input_size=num_features, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, dropout=DROPOUT,
        ).to(device)

        state_b, time_b = train_model(
            model_b, train_loader, val_loader, gaussian_nll_loss, device,
            label=f"Gauss-{label}"
        )
        model_b.load_state_dict(state_b)
        model_b.to(device)

        mu_b, sigma_b, tgt_b = predict_gaussian(model_b, test_loader, device)
        rmse_b = compute_rmse(tgt_b * scaler["std"] + scaler["mean"],
                              mu_b  * scaler["std"] + scaler["mean"])
        nll_b  = compute_nll(tgt_b, mu_b, sigma_b)
        crps_b = compute_crps(tgt_b, mu_b, sigma_b) * scaler["std"]

        val_mu_b, val_sig_b, val_tgt_b = predict_gaussian(model_b, val_loader, device)
        T_b, _, _ = find_temperature(val_tgt_b, val_mu_b, val_sig_b)
        sigma_b_cal = sigma_b * T_b
        cal_b = compute_calibration(tgt_b, mu_b, sigma_b_cal)
        cal_err_b = float(np.mean(np.abs(
            np.array(cal_b["expected"]) - np.array(cal_b["observed"])
        )))
        print(f"  Gaussian RMSE={rmse_b:.0f} MW  NLL={nll_b:.4f}  "
              f"CRPS={crps_b:.0f} MW  CalErr={cal_err_b:.3f}  T={T_b:.2f}  ({time_b:.0f}s)")

        # ----------------------------------------------------------------
        # Model C: MC Dropout
        # ----------------------------------------------------------------
        print(f"\n--- MC Dropout LSTM [{label}] ---")
        model_c = MCDropoutLSTM(
            input_size=num_features, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, dropout_rate=DROPOUT,
        ).to(device)

        state_c, time_c = train_model(
            model_c, train_loader, val_loader, mc_dropout_loss, device,
            label=f"MC-{label}"
        )
        model_c.load_state_dict(state_c)
        model_c.to(device)

        mu_c, sigma_c, tgt_c = predict_mc_dropout(
            model_c, test_loader, device, n_samples=MC_SAMPLES
        )
        rmse_c = compute_rmse(tgt_c * scaler["std"] + scaler["mean"],
                              mu_c  * scaler["std"] + scaler["mean"])
        nll_c  = compute_nll(tgt_c, mu_c, sigma_c)

        val_mu_c, val_sig_c, val_tgt_c = predict_mc_dropout(
            model_c, val_loader, device, n_samples=MC_SAMPLES
        )
        T_c, _, _ = find_temperature(val_tgt_c, val_mu_c, val_sig_c)
        sigma_c_cal = sigma_c * T_c
        cal_c = compute_calibration(tgt_c, mu_c, sigma_c_cal)
        cal_err_c = float(np.mean(np.abs(
            np.array(cal_c["expected"]) - np.array(cal_c["observed"])
        )))
        crps_c = compute_crps(tgt_c, mu_c, sigma_c_cal) * scaler["std"]
        print(f"  MC Dropout RMSE={rmse_c:.0f} MW  NLL={nll_c:.4f}  "
              f"CRPS={crps_c:.0f} MW  CalErr={cal_err_c:.3f}  T={T_c:.2f}  ({time_c:.0f}s)")

        results.append({
            "lag_config":       label,
            "lag_periods":      lag_periods,
            "num_features":     num_features,
            "det_rmse_mw":      round(rmse_a),
            "det_time_s":       round(time_a),
            "gauss_rmse_mw":    round(rmse_b),
            "gauss_nll":        round(nll_b, 4),
            "gauss_crps_mw":    round(crps_b),
            "gauss_cal_error":  round(cal_err_b, 3),
            "gauss_T":          round(T_b, 2),
            "gauss_time_s":     round(time_b),
            "mc_rmse_mw":       round(rmse_c),
            "mc_nll":           round(nll_c, 4),
            "mc_crps_mw":       round(crps_c),
            "mc_cal_error":     round(cal_err_c, 3),
            "mc_T":             round(T_c, 2),
            "mc_time_s":        round(time_c),
        })

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    results_path = os.path.join(SAVE_DIR, "ablation_lag_features_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
    print(f"\nSaved results -> {results_path}")

    # ----------------------------------------------------------------
    # Generate figure (2 rows x 3 cols)
    # ----------------------------------------------------------------
    config_labels = [r["lag_config"] for r in results]
    x = np.arange(len(config_labels))

    det_rmse  = [r["det_rmse_mw"]    for r in results]
    gauss_rmse = [r["gauss_rmse_mw"] for r in results]
    mc_rmse   = [r["mc_rmse_mw"]     for r in results]

    gauss_crps = [r["gauss_crps_mw"] for r in results]
    mc_crps    = [r["mc_crps_mw"]    for r in results]

    gauss_cal  = [r["gauss_cal_error"] for r in results]
    mc_cal     = [r["mc_cal_error"]    for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    short_labels = ["A\nNo lags", "B\n+Yesterday", "C\n+Week", "D\n+Month\n+Year"]

    def line_plot(ax, yvals, ylabel, title, color, marker="o"):
        ax.plot(x, yvals, marker=marker, color=color, linewidth=2, markersize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.grid(True, alpha=0.3)
        for xi, yi in zip(x, yvals):
            ax.annotate(f"{yi:,}", (xi, yi), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)

    line_plot(axes[0, 0], det_rmse,   "RMSE (MW)", "Deterministic RMSE vs Lag Config",  "#1f77b4")
    line_plot(axes[0, 1], gauss_rmse, "RMSE (MW)", "Gaussian RMSE vs Lag Config",        "#2ca02c")
    line_plot(axes[0, 2], mc_rmse,    "RMSE (MW)", "MC Dropout RMSE vs Lag Config",      "#d62728")

    line_plot(axes[1, 0], gauss_crps, "CRPS (MW)", "Gaussian CRPS vs Lag Config",        "#2ca02c", marker="s")
    line_plot(axes[1, 1], mc_crps,    "CRPS (MW)", "MC Dropout CRPS vs Lag Config",      "#d62728", marker="s")
    line_plot(axes[1, 2], mc_cal,     "Cal Error", "MC Dropout Calibration vs Lag Config","#9467bd", marker="^")

    fig.suptitle("Ablation Study 5: Explicit Lag Demand Features", fontsize=13)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "ablation_lag_features.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {fig_path}")

    # ----------------------------------------------------------------
    # Print summary table
    # ----------------------------------------------------------------
    print(f"\n{'='*85}")
    print(f"  LAG FEATURES ABLATION SUMMARY")
    print(f"{'='*85}")
    print(f"{'Config':<22} {'Feats':>5} {'Det RMSE':>10} {'Gauss RMSE':>12} "
          f"{'Gauss CRPS':>12} {'MC RMSE':>10} {'MC CRPS':>10} {'MC Cal':>8}")
    print(f"{'-'*85}")
    for r in results:
        print(f"{r['lag_config']:<22} {r['num_features']:>5} {r['det_rmse_mw']:>10,} "
              f"{r['gauss_rmse_mw']:>12,} {r['gauss_crps_mw']:>12,} "
              f"{r['mc_rmse_mw']:>10,} {r['mc_crps_mw']:>10,} {r['mc_cal_error']:>8.3f}")


if __name__ == "__main__":
    main()
