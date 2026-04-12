"""
Unified Evaluation Script — All Models

# GenAI statement: This script structure was assisted by Claude (Anthropic).
# All technical decisions and model implementations were verified by the team.

This script:
    1. Loads saved model weights from saved_models/
    2. Runs inference on the test set for all three models
    3. Computes test metrics (RMSE, NLL, CRPS, Calibration) in original MW units
    4. Generates presentation-quality figures to figures/
    5. Prints a full comparison table

Usage:
    python test.py
"""

import json
import os

import numpy as np
import torch

from src.data_loader import data_preprocess, get_dataloaders
from src.models.deterministic import DeterministicLSTM
from src.models.gaussian import GaussianLSTM
from src.models.mc_dropout import MCDropoutLSTM
from src.evaluate import (
    compute_rmse, compute_nll, compute_crps, compute_calibration,
    plot_predictions, plot_calibration, plot_training_curves,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")

# Model paths
DET_MODEL_PATH    = os.path.join(SAVE_DIR, "deterministic_lstm.pt")
GAUSS_MODEL_PATH  = os.path.join(SAVE_DIR, "gaussian_lstm.pt")
MC_MODEL_PATH     = os.path.join(SAVE_DIR, "mc_dropout_lstm.pt")

# History paths
DET_HISTORY_PATH   = os.path.join(SAVE_DIR, "training_history.json")
GAUSS_HISTORY_PATH = os.path.join(SAVE_DIR, "gaussian_training_history.json")
MC_HISTORY_PATH    = os.path.join(SAVE_DIR, "mc_dropout_training_history.json")

# Scaler
SCALER_PATH = os.path.join(SAVE_DIR, "scaler_params_best.json")

# Config
SEQ_LEN          = 336
BATCH_SIZE       = 64
DET_HIDDEN_SIZE  = 64    # deterministic trained with C8 config hidden_size=64
PROB_HIDDEN_SIZE = 64    # gaussian and mc dropout trained with C8 config hidden_size=64
NUM_LAYERS       = 2
MC_SAMPLES       = 50
WEEK_LEN         = 336


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_deterministic(model, test_loader, device):
    """Run deterministic model inference. Returns (preds, targets) as numpy."""
    model.eval()
    preds, targets = [], []
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        preds.append(out["prediction"].cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds).squeeze(), np.concatenate(targets).squeeze()


@torch.no_grad()
def run_gaussian(model, test_loader, device):
    """Run Gaussian model inference. Returns (mu, sigma, targets) as numpy."""
    model.eval()
    mus, sigmas, targets = [], [], []
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        mus.append(out["mu"].cpu().numpy())
        sigmas.append(out["sigma"].cpu().numpy())
        targets.append(y.numpy())
    return (np.concatenate(mus).squeeze(),
            np.concatenate(sigmas).squeeze(),
            np.concatenate(targets).squeeze())


def run_mc_dropout(model, test_loader, device, n_samples=50):
    """Run MC Dropout inference. Returns (mu, sigma, targets) as numpy."""
    mus, sigmas, targets = [], [], []
    for x, y in test_loader:
        x = x.to(device)
        out = model.predict_with_uncertainty(x, n_samples=n_samples)
        mus.append(out["prediction"].cpu().numpy())
        sigmas.append(out["uncertainty"].cpu().numpy())
        targets.append(y.numpy())
    return (np.concatenate(mus).squeeze(),
            np.concatenate(sigmas).squeeze(),
            np.concatenate(targets).squeeze())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(FIG_DIR, exist_ok=True)

    # --- Load scaler ---
    with open(SCALER_PATH) as f:
        scaler = json.load(f)
    nd_mean, nd_std = scaler["mean"], scaler["std"]
    print(f"Scaler: mean={nd_mean:.2f}, std={nd_std:.2f}")

    # --- Load data ---
    print("Loading data ...")
    train_df, valid_df, test_df, _ = data_preprocess( include_temperature=True, train_start=2015, train_end=2017,)
    _, _, test_loader = get_dataloaders(
        train_df, valid_df, test_df,
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
    )
    x_sample, _ = next(iter(test_loader))
    num_features = x_sample.shape[2]
    print(f"Test batches: {len(test_loader)}, num_features: {num_features}")

    # --- Load temperature T values ---
    with open(os.path.join(SAVE_DIR, "gaussian_temperature_T.json")) as f:
        gauss_T = json.load(f)["T"]
    with open(os.path.join(SAVE_DIR, "mc_dropout_temperature_T.json")) as f:
        mc_T = json.load(f)["T"]

    results = {}

    # -----------------------------------------------------------------------
    # 1. Deterministic LSTM
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("Evaluating: Deterministic LSTM")
    print("="*60)

    det_model = DeterministicLSTM(
        input_size=num_features,
        hidden_size=DET_HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ).to(device)
    det_model.load_state_dict(torch.load(DET_MODEL_PATH, map_location=device, weights_only=True))

    det_preds, det_targets = run_deterministic(det_model, test_loader, device)
    det_preds_mw   = det_preds   * nd_std + nd_mean
    det_targets_mw = det_targets * nd_std + nd_mean

    det_rmse = compute_rmse(det_targets_mw, det_preds_mw)
    print(f"  RMSE: {det_rmse:.0f} MW")

    results["deterministic"] = {"rmse_mw": det_rmse}

    # Figures
    start = len(det_preds_mw) // 3
    plot_predictions(
        det_targets_mw[start:start+WEEK_LEN],
        det_preds_mw[start:start+WEEK_LEN],
        save_path=os.path.join(FIG_DIR, "best_baseline_predictions.png"),
        title=f"Deterministic LSTM -- 1-Week Test (RMSE: {det_rmse:.0f} MW)",
    )

    with open(DET_HISTORY_PATH) as f:
        det_hist = json.load(f)
    plot_training_curves(
        det_hist["train_loss"], det_hist["val_loss"],
        save_path=os.path.join(FIG_DIR, "best_baseline_training_curve.png"),
        title="Deterministic LSTM -- Training Curve",
    )

    # -----------------------------------------------------------------------
    # 2. Gaussian LSTM
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("Evaluating: Gaussian LSTM")
    print("="*60)

    gauss_model = GaussianLSTM(
        input_size=num_features,
        hidden_size=PROB_HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ).to(device)
    gauss_model.load_state_dict(torch.load(GAUSS_MODEL_PATH, map_location=device, weights_only=True))

    gauss_mu, gauss_sigma_raw, gauss_targets = run_gaussian(gauss_model, test_loader, device)
    gauss_sigma_cal = gauss_sigma_raw * gauss_T

    gauss_mu_mw       = gauss_mu       * nd_std + nd_mean
    gauss_sigma_cal_mw = gauss_sigma_cal * nd_std
    gauss_targets_mw  = gauss_targets  * nd_std + nd_mean

    gauss_rmse = compute_rmse(gauss_targets_mw, gauss_mu_mw)
    gauss_nll  = compute_nll(gauss_targets, gauss_mu, gauss_sigma_cal)
    gauss_crps = compute_crps(gauss_targets, gauss_mu, gauss_sigma_cal) * nd_std

    gauss_cal_raw  = compute_calibration(gauss_targets, gauss_mu, gauss_sigma_raw)
    gauss_cal_after = compute_calibration(gauss_targets, gauss_mu, gauss_sigma_cal)
    gauss_cal_err_raw   = float(np.mean(np.abs(np.array(gauss_cal_raw["expected"])  - np.array(gauss_cal_raw["observed"]))))
    gauss_cal_err_after = float(np.mean(np.abs(np.array(gauss_cal_after["expected"]) - np.array(gauss_cal_after["observed"]))))

    print(f"  RMSE:             {gauss_rmse:.0f} MW")
    print(f"  NLL (calibrated): {gauss_nll:.4f}")
    print(f"  CRPS:             {gauss_crps:.0f} MW")
    print(f"  Cal. Error (raw): {gauss_cal_err_raw:.3f}")
    print(f"  Cal. Error (T={gauss_T:.2f}): {gauss_cal_err_after:.3f}")

    results["gaussian"] = {
        "rmse_mw": gauss_rmse, "nll": gauss_nll,
        "crps_mw": gauss_crps, "cal_error_raw": gauss_cal_err_raw,
        "cal_error_calibrated": gauss_cal_err_after, "T": gauss_T,
    }

    # Figures
    plot_predictions(
        gauss_targets_mw[start:start+WEEK_LEN],
        gauss_mu_mw[start:start+WEEK_LEN],
        sigma=gauss_sigma_cal_mw[start:start+WEEK_LEN],
        save_path=os.path.join(FIG_DIR, "gaussian_predictions_with_uncertainty.png"),
        title=f"Gaussian LSTM -- 1-Week Test (RMSE: {gauss_rmse:.0f} MW)",
    )
    plot_calibration(
        gauss_cal_after["expected"], gauss_cal_after["observed"],
        save_path=os.path.join(FIG_DIR, "gaussian_calibration.png"),
        title=f"Gaussian LSTM -- Calibration (T={gauss_T:.2f})",
    )

    # -----------------------------------------------------------------------
    # 3. MC Dropout LSTM
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("Evaluating: MC Dropout LSTM")
    print("="*60)

    mc_model = MCDropoutLSTM(
        input_size=num_features,
        hidden_size=PROB_HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_rate=0.2,
    ).to(device)
    mc_model.load_state_dict(torch.load(MC_MODEL_PATH, map_location=device, weights_only=True))

    mc_mu, mc_sigma_raw, mc_targets = run_mc_dropout(mc_model, test_loader, device, n_samples=MC_SAMPLES)
    mc_sigma_cal = mc_sigma_raw * mc_T

    mc_mu_mw       = mc_mu       * nd_std + nd_mean
    mc_sigma_cal_mw = mc_sigma_cal * nd_std
    mc_targets_mw  = mc_targets  * nd_std + nd_mean

    mc_rmse = compute_rmse(mc_targets_mw, mc_mu_mw)
    mc_nll  = compute_nll(mc_targets, mc_mu, mc_sigma_cal)
    mc_crps = compute_crps(mc_targets, mc_mu, mc_sigma_cal) * nd_std

    mc_cal_raw   = compute_calibration(mc_targets, mc_mu, mc_sigma_raw)
    mc_cal_after = compute_calibration(mc_targets, mc_mu, mc_sigma_cal)
    mc_cal_err_raw   = float(np.mean(np.abs(np.array(mc_cal_raw["expected"])  - np.array(mc_cal_raw["observed"]))))
    mc_cal_err_after = float(np.mean(np.abs(np.array(mc_cal_after["expected"]) - np.array(mc_cal_after["observed"]))))

    print(f"  RMSE:             {mc_rmse:.0f} MW")
    print(f"  NLL (calibrated): {mc_nll:.4f}")
    print(f"  CRPS:             {mc_crps:.0f} MW")
    print(f"  Cal. Error (raw): {mc_cal_err_raw:.3f}")
    print(f"  Cal. Error (T={mc_T:.2f}): {mc_cal_err_after:.3f}")

    results["mc_dropout"] = {
        "rmse_mw": mc_rmse, "nll": mc_nll,
        "crps_mw": mc_crps, "cal_error_raw": mc_cal_err_raw,
        "cal_error_calibrated": mc_cal_err_after, "T": mc_T,
    }

    # Figures
    plot_predictions(
        mc_targets_mw[start:start+WEEK_LEN],
        mc_mu_mw[start:start+WEEK_LEN],
        sigma=mc_sigma_cal_mw[start:start+WEEK_LEN],
        save_path=os.path.join(FIG_DIR, "mc_dropout_predictions_with_uncertainty.png"),
        title=f"MC Dropout LSTM (T={MC_SAMPLES}, recal.) -- RMSE: {mc_rmse:.0f} MW",
    )
    plot_calibration(
        mc_cal_after["expected"], mc_cal_after["observed"],
        save_path=os.path.join(FIG_DIR, "mc_dropout_calibration.png"),
        title=f"MC Dropout -- Calibration (T={mc_T:.2f})",
    )

    # -----------------------------------------------------------------------
    # Final summary table
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'RMSE (MW)':>10} {'NLL':>10} {'CRPS (MW)':>10} {'Cal.Err':>10}")
    print(f"{'-'*70}")
    print(f"{'Deterministic LSTM':<25} {det_rmse:>10.0f} {'-':>10} {'-':>10} {'-':>10}")
    print(f"{'Gaussian LSTM':<25} {gauss_rmse:>10.0f} {gauss_nll:>10.4f} {gauss_crps:>10.0f} {gauss_cal_err_after:>10.3f}")
    print(f"{'MC Dropout LSTM':<25} {mc_rmse:>10.0f} {mc_nll:>10.4f} {mc_crps:>10.0f} {mc_cal_err_after:>10.3f}")
    print(f"{'='*70}")

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()