"""
Ablation Study 4: UK Bank Holiday Indicator

Tests whether adding a binary is_holiday feature improves forecast quality.
Trains all three models (Deterministic, Gaussian, MC Dropout) with and without
the holiday flag. Temperature is always included to isolate the effect of
the holiday feature.

Motivation: The model already has is_weekend but no holiday flag. UK bank
holidays (Easter, Christmas, Boxing Day, etc.) cause demand profiles that look
like Sundays but fall on weekdays. Without is_holiday, the model treats these
as regular weekdays, inflating uncertainty on known patterns.

Usage:
    python experiments/ablation_holiday.py

Results saved to:
    - saved_models/ablation_holiday_results.json
    - figures/ablation_holiday.png
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
CONDITIONS = [
    {"include_holidays": False, "label": "Without Holiday"},
    {"include_holidays": True,  "label": "With Holiday"},
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

    for cond in CONDITIONS:
        label = cond["label"]
        use_holidays = cond["include_holidays"]

        print(f"\n{'='*60}")
        print(f"  CONDITION: {label}")
        print(f"{'='*60}")

        # Load data — temperature always on, holiday flag varies
        print(f"Loading data (include_holidays={use_holidays}) ...")
        train_df, valid_df, test_df, scaler = data_preprocess(
            train_start=TRAIN_START, train_end=TRAIN_END,
            include_temperature=True,
            include_holidays=use_holidays,
        )
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, valid_df, test_df, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        )
        num_features = next(iter(train_loader))[0].shape[2]
        print(f"num_features: {num_features}")

        entry = {
            "condition": label,
            "include_holidays": use_holidays,
            "num_features": num_features,
        }

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
            preds_a  * scaler["std"] + scaler["mean"],
        )
        print(f"  Deterministic RMSE: {rmse_a:.0f} MW  ({time_a:.0f}s)")
        entry["det_rmse_mw"] = round(rmse_a)
        entry["det_time_s"]  = round(time_a)

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
        rmse_b  = compute_rmse(tgt_b * scaler["std"] + scaler["mean"],
                               mu_b  * scaler["std"] + scaler["mean"])
        nll_b   = compute_nll(tgt_b, mu_b, sigma_b)
        crps_b  = compute_crps(tgt_b, mu_b, sigma_b) * scaler["std"]

        val_mu_b, val_sig_b, val_tgt_b = predict_gaussian(model_b, val_loader, device)
        T_b, _, _ = find_temperature(val_tgt_b, val_mu_b, val_sig_b)
        sigma_b_cal = sigma_b * T_b
        cal_b = compute_calibration(tgt_b, mu_b, sigma_b_cal)
        cal_err_b = float(np.mean(np.abs(
            np.array(cal_b["expected"]) - np.array(cal_b["observed"])
        )))

        print(f"  Gaussian RMSE={rmse_b:.0f} MW  NLL={nll_b:.4f}  "
              f"CRPS={crps_b:.0f} MW  CalErr={cal_err_b:.3f}  T={T_b:.2f}  ({time_b:.0f}s)")
        entry["gauss_rmse_mw"]   = round(rmse_b)
        entry["gauss_nll"]       = round(nll_b, 4)
        entry["gauss_crps_mw"]   = round(crps_b)
        entry["gauss_cal_error"] = round(cal_err_b, 3)
        entry["gauss_T"]         = round(T_b, 2)
        entry["gauss_time_s"]    = round(time_b)

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
        entry["mc_rmse_mw"]   = round(rmse_c)
        entry["mc_nll"]       = round(nll_c, 4)
        entry["mc_crps_mw"]   = round(crps_c)
        entry["mc_cal_error"] = round(cal_err_c, 3)
        entry["mc_T"]         = round(T_c, 2)
        entry["mc_time_s"]    = round(time_c)

        results.append(entry)

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    results_path = os.path.join(SAVE_DIR, "ablation_holiday_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
    print(f"\nSaved results -> {results_path}")

    # ----------------------------------------------------------------
    # Generate figure (2 rows x 3 cols)
    # ----------------------------------------------------------------
    without = results[0]
    with_h  = results[1]

    models_all  = ["Deterministic", "Gaussian", "MC Dropout"]
    models_prob = ["Gaussian", "MC Dropout"]

    rmse_without = [without["det_rmse_mw"], without["gauss_rmse_mw"], without["mc_rmse_mw"]]
    rmse_with    = [with_h["det_rmse_mw"],  with_h["gauss_rmse_mw"],  with_h["mc_rmse_mw"]]
    crps_without = [without["gauss_crps_mw"], without["mc_crps_mw"]]
    crps_with    = [with_h["gauss_crps_mw"],  with_h["mc_crps_mw"]]
    cal_without  = [without["gauss_cal_error"], without["mc_cal_error"]]
    cal_with     = [with_h["gauss_cal_error"],  with_h["mc_cal_error"]]
    nll_without  = [without["gauss_nll"], without["mc_nll"]]
    nll_with     = [with_h["gauss_nll"],  with_h["mc_nll"]]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    x3 = np.arange(len(models_all))
    x2 = np.arange(len(models_prob))
    w  = 0.35
    c_without = "#ff7f0e"
    c_with    = "#1f77b4"

    def bar_pair(ax, xpos, v_no, v_yes, ylabel, title, xlabels):
        ax.bar(xpos - w/2, v_no,  w, label="Without Holiday", color=c_without)
        ax.bar(xpos + w/2, v_yes, w, label="With Holiday",    color=c_with)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    bar_pair(axes[0, 0], x3, rmse_without, rmse_with,
             "Test RMSE (MW)", "RMSE — All Models", models_all)
    bar_pair(axes[0, 1], x2, nll_without, nll_with,
             "Test NLL", "NLL — Probabilistic Models", models_prob)
    bar_pair(axes[0, 2], x2, crps_without, crps_with,
             "CRPS (MW)", "CRPS — Probabilistic Models", models_prob)
    bar_pair(axes[1, 0], x2, cal_without, cal_with,
             "Calibration Error", "Calibration Error — Probabilistic Models", models_prob)

    # Summary delta table
    axes[1, 1].axis("off")
    summary_lines = [
        "Holiday Feature — Delta Summary",
        "(positive = worse with holiday, negative = better)",
        "",
        f"{'Model':<14} {'Metric':<12} {'Delta':>10}",
        "-" * 38,
        f"{'Deterministic':<14} {'RMSE (MW)':<12} {with_h['det_rmse_mw'] - without['det_rmse_mw']:>+10,}",
        f"{'Gaussian':<14} {'RMSE (MW)':<12} {with_h['gauss_rmse_mw'] - without['gauss_rmse_mw']:>+10,}",
        f"{'Gaussian':<14} {'CRPS (MW)':<12} {with_h['gauss_crps_mw'] - without['gauss_crps_mw']:>+10,}",
        f"{'Gaussian':<14} {'Cal Error':<12} {with_h['gauss_cal_error'] - without['gauss_cal_error']:>+10.3f}",
        f"{'MC Dropout':<14} {'RMSE (MW)':<12} {with_h['mc_rmse_mw'] - without['mc_rmse_mw']:>+10,}",
        f"{'MC Dropout':<14} {'CRPS (MW)':<12} {with_h['mc_crps_mw'] - without['mc_crps_mw']:>+10,}",
        f"{'MC Dropout':<14} {'Cal Error':<12} {with_h['mc_cal_error'] - without['mc_cal_error']:>+10.3f}",
    ]
    axes[1, 1].text(0.05, 0.95, "\n".join(summary_lines),
                    transform=axes[1, 1].transAxes,
                    fontsize=8, verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    axes[1, 2].axis("off")

    fig.suptitle("Ablation Study 4: UK Bank Holiday Indicator Effect", fontsize=13)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "ablation_holiday.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {fig_path}")

    # ----------------------------------------------------------------
    # Print summary table
    # ----------------------------------------------------------------
    print(f"\n{'='*75}")
    print(f"  HOLIDAY INDICATOR ABLATION SUMMARY")
    print(f"{'='*75}")
    print(f"{'Model':<14} {'Metric':<12} {'Without':>10} {'With':>10} {'Delta':>10}")
    print(f"{'-'*56}")
    rows = [
        ("Deterministic", "RMSE (MW)",  without["det_rmse_mw"],     with_h["det_rmse_mw"],     "d"),
        ("Gaussian",      "RMSE (MW)",  without["gauss_rmse_mw"],   with_h["gauss_rmse_mw"],   "d"),
        ("Gaussian",      "NLL",        without["gauss_nll"],       with_h["gauss_nll"],       "f4"),
        ("Gaussian",      "CRPS (MW)",  without["gauss_crps_mw"],   with_h["gauss_crps_mw"],   "d"),
        ("Gaussian",      "Cal Error",  without["gauss_cal_error"], with_h["gauss_cal_error"], "f3"),
        ("MC Dropout",    "RMSE (MW)",  without["mc_rmse_mw"],      with_h["mc_rmse_mw"],      "d"),
        ("MC Dropout",    "NLL",        without["mc_nll"],          with_h["mc_nll"],          "f4"),
        ("MC Dropout",    "CRPS (MW)",  without["mc_crps_mw"],      with_h["mc_crps_mw"],      "d"),
        ("MC Dropout",    "Cal Error",  without["mc_cal_error"],    with_h["mc_cal_error"],    "f3"),
    ]
    for model, metric, v_no, v_yes, kind in rows:
        delta = v_yes - v_no
        if kind == "d":
            print(f"{model:<14} {metric:<12} {v_no:>10,} {v_yes:>10,} {delta:>+10,}")
        elif kind == "f4":
            print(f"{model:<14} {metric:<12} {v_no:>10.4f} {v_yes:>10.4f} {delta:>+10.4f}")
        else:
            print(f"{model:<14} {metric:<12} {v_no:>10.3f} {v_yes:>10.3f} {delta:>+10.3f}")


if __name__ == "__main__":
    main()
