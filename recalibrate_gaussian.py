"""
Gaussian LSTM Temperature Scaling Recalibration

Finds optimal temperature T on the validation set, applies it to the test set,
and generates before/after comparison figures.

Temperature scaling: sigma_calibrated = sigma_raw * T
- T is a single scalar learned on the validation set (no data leakage)
- T > 1 widens prediction intervals (fixes overconfidence)
- Does NOT affect mu (point predictions / RMSE unchanged)

Usage:
    python recalibrate_gaussian.py
"""

import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import data_preprocess, get_dataloaders
from src.models.gaussian import GaussianLSTM
from src.evaluate import (
    compute_rmse, compute_nll, compute_crps, compute_calibration,
    find_temperature, plot_predictions, plot_calibration,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
MODEL_PATH = os.path.join(SAVE_DIR, "gaussian_lstm.pt")


@torch.no_grad()
def predict_all(model, loader, device):
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


def eval_metrics(y_true_norm, mu_norm, sigma_norm, scaler):
    """Compute all metrics for a given (mu, sigma) pair."""
    mu_mw = mu_norm * scaler["std"] + scaler["mean"]
    sigma_mw = sigma_norm * scaler["std"]
    targets_mw = y_true_norm * scaler["std"] + scaler["mean"]

    rmse = compute_rmse(targets_mw, mu_mw)
    nll = compute_nll(y_true_norm, mu_norm, sigma_norm)
    crps_mw = compute_crps(y_true_norm, mu_norm, sigma_norm) * scaler["std"]
    cal = compute_calibration(y_true_norm, mu_norm, sigma_norm)
    cal_err = float(np.mean(np.abs(
        np.array(cal["expected"]) - np.array(cal["observed"])
    )))
    pi90_mw = float(np.mean(2 * 1.645 * sigma_mw))

    return {
        "rmse_mw": rmse,
        "nll": nll,
        "crps_mw": crps_mw,
        "cal_error": cal_err,
        "pi90_width_mw": pi90_mw,
        "calibration": cal,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    print("Loading data ...")
    train_df, valid_df, test_df, scaler = data_preprocess(
        train_start=2015, train_end=2017, include_temperature=True,
    )
    _, valid_loader, test_loader = get_dataloaders(
        train_df, valid_df, test_df, seq_len=336, batch_size=64,
    )
    num_features = next(iter(valid_loader))[0].shape[2]

    # --- Load model ---
    model = GaussianLSTM(
        input_size=num_features, hidden_size=64, num_layers=2, dropout=0.2,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    print(f"Loaded: {MODEL_PATH}")

    # --- Validation inference (for finding T) ---
    print("Running validation inference ...")
    val_mu, val_sigma, val_targets = predict_all(model, valid_loader, device)

    # --- Find optimal T ---
    print("Finding optimal temperature T ...")
    T_opt, T_grid, nll_grid = find_temperature(val_targets, val_mu, val_sigma)
    print(f"Optimal T = {T_opt:.4f}")

    # Save T
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, "gaussian_temperature_T.json"), "w") as f:
        json.dump({"T": T_opt}, f, indent=2)

    # --- Test inference ---
    print("Running test inference ...")
    test_mu, test_sigma_raw, test_targets = predict_all(model, test_loader, device)
    test_sigma_cal = test_sigma_raw * T_opt

    # --- Metrics: before and after ---
    print("Computing metrics ...")
    before = eval_metrics(test_targets, test_mu, test_sigma_raw, scaler)
    after = eval_metrics(test_targets, test_mu, test_sigma_cal, scaler)

    # --- Comparison table ---
    print(f"""
Recalibration Results (Temperature Scaling)
Optimal T: {T_opt:.4f}
+------------------------+------------+--------------+
| Metric                 | Before T   | After T      |
+------------------------+------------+--------------+
| Test RMSE (MW)         | {before['rmse_mw']:>8.0f}   | {after['rmse_mw']:>8.0f}       |
| Test NLL               | {before['nll']:>8.4f}   | {after['nll']:>8.4f}       |
| Test CRPS (MW)         | {before['crps_mw']:>8.0f}   | {after['crps_mw']:>8.0f}       |
| Calibration Error      | {before['cal_error']:>8.4f}   | {after['cal_error']:>8.4f}       |
| Avg 90% PI Width (MW)  | {before['pi90_width_mw']:>8.0f}   | {after['pi90_width_mw']:>8.0f}       |
+------------------------+------------+--------------+
""")

    # Calibration detail after recalibration
    print("Calibration detail (after recalibration):")
    for exp, obs in zip(after["calibration"]["expected"], after["calibration"]["observed"]):
        diff = obs - exp
        label = "overconfident" if diff < -0.02 else "underconfident" if diff > 0.02 else "good"
        print(f"  {exp:.0%} interval -> {obs:.1%} observed  ({label})")

    # ===================================================================
    # FIGURES
    # ===================================================================
    os.makedirs(FIG_DIR, exist_ok=True)

    # De-normalise for plots
    mu_mw = test_mu * scaler["std"] + scaler["mean"]
    sigma_raw_mw = test_sigma_raw * scaler["std"]
    sigma_cal_mw = test_sigma_cal * scaler["std"]
    targets_mw = test_targets * scaler["std"] + scaler["mean"]

    week = 336
    start = len(mu_mw) // 3

    # --- Figure 1: Calibration comparison (before vs after) ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.plot(before["calibration"]["expected"], before["calibration"]["observed"],
            "o-", color="#d62728", linewidth=1.5, markersize=5,
            label=f"Before (err={before['cal_error']:.3f})")
    ax.plot(after["calibration"]["expected"], after["calibration"]["observed"],
            "s-", color="#2ca02c", linewidth=1.5, markersize=5,
            label=f"After T={T_opt:.2f} (err={after['cal_error']:.3f})")

    ax.set_xlabel("Expected coverage", fontsize=12)
    ax.set_ylabel("Observed coverage", fontsize=12)
    ax.set_title("Gaussian LSTM -- Calibration Before & After Recalibration", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "gaussian_calibration_comparison.png"), dpi=200)
    plt.close(fig)
    print("Saved: figures/gaussian_calibration_comparison.png")

    # --- Figure 2: Recalibrated predictions ---
    plot_predictions(
        targets_mw[start:start + week],
        mu_mw[start:start + week],
        sigma=sigma_cal_mw[start:start + week],
        save_path=os.path.join(FIG_DIR, "gaussian_predictions_recalibrated.png"),
        title=f"Gaussian LSTM (Recalibrated T={T_opt:.2f}) -- "
              f"RMSE: {after['rmse_mw']:.0f} MW",
    )
    print("Saved: figures/gaussian_predictions_recalibrated.png")

    # --- Figure 3: T search landscape ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_grid, nll_grid, color="#1f77b4", linewidth=1.5)
    ax.axvline(T_opt, color="#d62728", linestyle="--", alpha=0.7,
               label=f"Optimal T = {T_opt:.2f}")
    ax.axvline(1.0, color="grey", linestyle=":", alpha=0.5, label="T = 1 (no scaling)")

    ax.set_xlabel("Temperature T", fontsize=12)
    ax.set_ylabel("Validation NLL", fontsize=12)
    ax.set_title("Temperature Scaling -- Validation NLL vs T", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "recalibration_T_search.png"), dpi=200)
    plt.close(fig)
    print("Saved: figures/recalibration_T_search.png")


if __name__ == "__main__":
    main()
