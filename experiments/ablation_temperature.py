"""
Ablation Study 1: Temperature Feature Ablation

Compares all three models WITH vs WITHOUT the temperature feature.
Purpose: Does exogenous weather data actually help prediction?

Usage:
    python experiments/ablation_temperature.py

Results saved to:
    - saved_models/ablation_temperature_results.json
    - figures/ablation_temperature.png
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

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
# Shared training config (from best baseline: C8)
# ---------------------------------------------------------------------------
SEQ_LEN = 336
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 0.001
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10
TRAIN_START = 2015
TRAIN_END = 2017
MC_SAMPLES = 50


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


def train_model(model, train_loader, val_loader, criterion, device, label=""):
    """Train with early stopping. Returns best model state and training time."""
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
        val_loss = validate(model, val_loader, criterion, device)
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

    elapsed = time.time() - t0
    return best_state, elapsed


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
    # --- 第一步：强制环境设置 ---
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 这一行是杀手锏：彻底让程序找不到显卡，强制走 CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
    
    # --- 第二步：定义设备 ---
    device = torch.device("cpu") 
    print(f"--- 模式确认：{device} (5070 Ti 暂时休眠) ---")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    results = []

    for use_temp in [True, False]:
        temp_label = "with_temp" if use_temp else "without_temp"
        print(f"\n{'='*60}")
        print(f"  TEMPERATURE ABLATION: {temp_label}")
        print(f"{'='*60}")

        # --- Load data ---
        print(f"Loading data (include_temperature={use_temp}) ...")
        train_df, valid_df, test_df, scaler = data_preprocess(
            train_start=TRAIN_START, train_end=TRAIN_END,
            include_temperature=use_temp,
        )
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, valid_df, test_df, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        )
        num_features = next(iter(train_loader))[0].shape[2]
        print(f"num_features: {num_features}")

        entry = {"config": temp_label, "num_features": num_features}

        # ---- Model A: Deterministic ----
        print(f"\n--- Deterministic LSTM ({temp_label}) ---")
        model_a = DeterministicLSTM(
            input_size=num_features, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, dropout=DROPOUT,
        ).to(device)

        state_a, time_a = train_model(
            model_a, train_loader, val_loader, deterministic_loss, device,
            label=f"Det-{temp_label}"
        )
        model_a.load_state_dict(state_a)
        model_a.to(device)

        preds, targets = predict_deterministic(model_a, test_loader, device)
        rmse_a = compute_rmse(
            targets * scaler["std"] + scaler["mean"],
            preds * scaler["std"] + scaler["mean"]
        )
        print(f"  Deterministic RMSE: {rmse_a:.0f} MW ({time_a:.0f}s)")
        entry["det_rmse_mw"] = round(rmse_a)
        entry["det_time_s"] = round(time_a)

        # ---- Model B: Gaussian ----
        print(f"\n--- Gaussian LSTM ({temp_label}) ---")
        model_b = GaussianLSTM(
            input_size=num_features, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, dropout=DROPOUT,
        ).to(device)

        state_b, time_b = train_model(
            model_b, train_loader, val_loader, gaussian_nll_loss, device,
            label=f"Gauss-{temp_label}"
        )
        model_b.load_state_dict(state_b)
        model_b.to(device)

        mu_norm, sigma_norm, targets_norm = predict_gaussian(model_b, test_loader, device)
        mu_mw = mu_norm * scaler["std"] + scaler["mean"]
        sigma_mw = sigma_norm * scaler["std"]
        targets_mw = targets_norm * scaler["std"] + scaler["mean"]

        rmse_b = compute_rmse(targets_mw, mu_mw)
        nll_b = compute_nll(targets_norm, mu_norm, sigma_norm)
        crps_b = compute_crps(targets_norm, mu_norm, sigma_norm) * scaler["std"]

        # Temperature scaling on validation set
        val_mu, val_sigma, val_targets = predict_gaussian(model_b, val_loader, device)
        T_opt_b, _, _ = find_temperature(val_targets, val_mu, val_sigma)
        sigma_cal = sigma_norm * T_opt_b
        cal_b = compute_calibration(targets_norm, mu_norm, sigma_cal)
        cal_err_b = float(np.mean(np.abs(
            np.array(cal_b["expected"]) - np.array(cal_b["observed"])
        )))

        print(f"  Gaussian: RMSE={rmse_b:.0f} MW, NLL={nll_b:.4f}, "
              f"CRPS={crps_b:.0f} MW, CalErr={cal_err_b:.3f} (T={T_opt_b:.2f})")

        entry["gauss_rmse_mw"] = round(rmse_b)
        entry["gauss_nll"] = round(nll_b, 4)
        entry["gauss_crps_mw"] = round(crps_b)
        entry["gauss_cal_error"] = round(cal_err_b, 3)
        entry["gauss_T"] = round(T_opt_b, 2)
        entry["gauss_time_s"] = round(time_b)

        # ---- Model C: MC Dropout ----
        print(f"\n--- MC Dropout LSTM ({temp_label}) ---")
        model_c = MCDropoutLSTM(
            input_size=num_features, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, dropout_rate=DROPOUT,
        ).to(device)

        state_c, time_c = train_model(
            model_c, train_loader, val_loader, mc_dropout_loss, device,
            label=f"MC-{temp_label}"
        )
        model_c.load_state_dict(state_c)
        model_c.to(device)

        mc_mu, mc_sigma, mc_targets = predict_mc_dropout(
            model_c, test_loader, device, n_samples=MC_SAMPLES
        )
        mc_mu_mw = mc_mu * scaler["std"] + scaler["mean"]
        mc_sigma_mw = mc_sigma * scaler["std"]
        mc_targets_mw = mc_targets * scaler["std"] + scaler["mean"]

        rmse_c = compute_rmse(mc_targets_mw, mc_mu_mw)
        nll_c = compute_nll(mc_targets, mc_mu, mc_sigma)

        # Temperature scaling
        val_mc_mu, val_mc_sigma, val_mc_targets = predict_mc_dropout(
            model_c, val_loader, device, n_samples=MC_SAMPLES
        )
        T_opt_c, _, _ = find_temperature(val_mc_targets, val_mc_mu, val_mc_sigma)
        mc_sigma_cal = mc_sigma * T_opt_c
        cal_c = compute_calibration(mc_targets, mc_mu, mc_sigma_cal)
        cal_err_c = float(np.mean(np.abs(
            np.array(cal_c["expected"]) - np.array(cal_c["observed"])
        )))
        crps_c = compute_crps(mc_targets, mc_mu, mc_sigma_cal) * scaler["std"]

        print(f"  MC Dropout: RMSE={rmse_c:.0f} MW, NLL={nll_c:.4f}, "
              f"CalErr={cal_err_c:.3f} (T={T_opt_c:.2f})")

        entry["mc_rmse_mw"] = round(rmse_c)
        entry["mc_nll"] = round(nll_c, 4)
        entry["mc_crps_mw"] = round(crps_c)
        entry["mc_cal_error"] = round(cal_err_c, 3)
        entry["mc_T"] = round(T_opt_c, 2)
        entry["mc_time_s"] = round(time_c)

        results.append(entry)

    # --- Save results ---
    results_path = os.path.join(SAVE_DIR, "ablation_temperature_results.json")
    results = json.loads(json.dumps(results, default=lambda x: float(x) if hasattr(x, 'item') else x))
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results -> {results_path}")

    # --- Generate figure ---
    with_temp = results[0]
    without_temp = results[1]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # RMSE comparison
    models = ["Deterministic", "Gaussian", "MC Dropout"]
    rmse_with = [with_temp["det_rmse_mw"], with_temp["gauss_rmse_mw"], with_temp["mc_rmse_mw"]]
    rmse_without = [without_temp["det_rmse_mw"], without_temp["gauss_rmse_mw"], without_temp["mc_rmse_mw"]]

    x = np.arange(len(models))
    w = 0.35
    axes[0].bar(x - w/2, rmse_with, w, label="With temperature", color="#1f77b4")
    axes[0].bar(x + w/2, rmse_without, w, label="Without temperature", color="#ff7f0e")
    axes[0].set_ylabel("Test RMSE (MW)")
    axes[0].set_title("RMSE: Temperature Feature Ablation")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=9)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    for i, (v1, v2) in enumerate(zip(rmse_with, rmse_without)):
        axes[0].text(i - w/2, v1 + 50, f"{v1:,}", ha="center", fontsize=8)
        axes[0].text(i + w/2, v2 + 50, f"{v2:,}", ha="center", fontsize=8)

    # NLL comparison (Gaussian and MC Dropout only)
    prob_models = ["Gaussian", "MC Dropout"]
    nll_with = [with_temp["gauss_nll"], with_temp["mc_nll"]]
    nll_without = [without_temp["gauss_nll"], without_temp["mc_nll"]]

    x2 = np.arange(len(prob_models))
    axes[1].bar(x2 - w/2, nll_with, w, label="With temperature", color="#1f77b4")
    axes[1].bar(x2 + w/2, nll_without, w, label="Without temperature", color="#ff7f0e")
    axes[1].set_ylabel("Test NLL")
    axes[1].set_title("NLL: Temperature Feature Ablation")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(prob_models, fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Calibration error comparison
    cal_with = [with_temp["gauss_cal_error"], with_temp["mc_cal_error"]]
    cal_without = [without_temp["gauss_cal_error"], without_temp["mc_cal_error"]]

    axes[2].bar(x2 - w/2, cal_with, w, label="With temperature", color="#1f77b4")
    axes[2].bar(x2 + w/2, cal_without, w, label="Without temperature", color="#ff7f0e")
    axes[2].set_ylabel("Calibration Error")
    axes[2].set_title("Calibration: Temperature Feature Ablation")
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(prob_models, fontsize=9)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Ablation Study 1: Effect of Temperature Feature", fontsize=13, y=1.02)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "ablation_temperature.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {fig_path}")

    # --- Print summary table ---
    print(f"\n{'='*70}")
    print(f"  TEMPERATURE ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Metric':<12} {'With Temp':>10} {'No Temp':>10} {'Delta':>10}")
    print(f"{'-'*57}")
    print(f"{'Deterministic':<15} {'RMSE (MW)':<12} {with_temp['det_rmse_mw']:>10,} {without_temp['det_rmse_mw']:>10,} {without_temp['det_rmse_mw'] - with_temp['det_rmse_mw']:>+10,}")
    print(f"{'Gaussian':<15} {'RMSE (MW)':<12} {with_temp['gauss_rmse_mw']:>10,} {without_temp['gauss_rmse_mw']:>10,} {without_temp['gauss_rmse_mw'] - with_temp['gauss_rmse_mw']:>+10,}")
    print(f"{'Gaussian':<15} {'NLL':<12} {with_temp['gauss_nll']:>10.4f} {without_temp['gauss_nll']:>10.4f} {without_temp['gauss_nll'] - with_temp['gauss_nll']:>+10.4f}")
    print(f"{'MC Dropout':<15} {'RMSE (MW)':<12} {with_temp['mc_rmse_mw']:>10,} {without_temp['mc_rmse_mw']:>10,} {without_temp['mc_rmse_mw'] - with_temp['mc_rmse_mw']:>+10,}")
    print(f"{'MC Dropout':<15} {'Cal. Error':<12} {with_temp['mc_cal_error']:>10.3f} {without_temp['mc_cal_error']:>10.3f} {without_temp['mc_cal_error'] - with_temp['mc_cal_error']:>+10.3f}")


if __name__ == "__main__":
    main()
