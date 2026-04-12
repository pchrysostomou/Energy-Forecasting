"""
Baseline LSTM Hyperparameter Sweep

Systematically tests regularisation, training data recency, and architecture
changes to reduce overfitting in the deterministic LSTM baseline.

Groups:
    A: Regularisation (dropout, LR scheduler, weight decay)
    B: Training data recency (fewer, more recent years)
    C: Architecture (hidden size, depth)

Usage:
    python -m experiments.baseline_sweep
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import data_preprocess, get_dataloaders
from src.models.deterministic import DeterministicLSTM, deterministic_loss
from src.evaluate import compute_rmse, plot_training_curves, plot_predictions

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")

SEQ_LEN = 336
BATCH_SIZE = 64
PATIENCE = 10
MAX_EPOCHS = 100


# ---------------------------------------------------------------------------
# Training engine
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
def evaluate(model, loader, criterion, device):
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
def predict_all(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        preds.append(model(x)["prediction"].cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds).squeeze(), np.concatenate(targets).squeeze()


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------
def run_experiment(cfg, device):
    """Run one experiment defined by *cfg* dict. Returns results dict."""
    label = cfg["label"]
    print(f"\n{'='*60}")
    print(f"  Experiment: {label}")
    print(f"{'='*60}")

    # --- Data ---
    train_df, valid_df, test_df, scaler = data_preprocess(
        train_start=cfg["train_start"], train_end=cfg["train_end"],
        valid_start=2018, valid_end=2022,
        test_start=2023, test_end=2025,
        include_temperature=True,
    )
    train_loader, valid_loader, test_loader = get_dataloaders(
        train_df, valid_df, test_df, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
    )
    num_features = next(iter(train_loader))[0].shape[2]
    print(f"  Train size: {len(train_df)}, features: {num_features}")

    # --- Model ---
    model = DeterministicLSTM(
        input_size=num_features,
        hidden_size=cfg["hidden"],
        num_layers=cfg["layers"],
        dropout=cfg["dropout"],
    ).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Optimiser ---
    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )

    # --- LR Scheduler (optional) ---
    scheduler = None
    if cfg["scheduler"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", patience=5, factor=0.5,
        )

    # --- Training loop ---
    criterion = deterministic_loss
    best_val = float("inf")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        t_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        v_loss = evaluate(model, valid_loader, criterion, device)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if scheduler is not None:
            scheduler.step(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            flag = "<- best"
        else:
            no_improve += 1
            flag = f"({no_improve}/{PATIENCE})"

        if epoch <= 5 or epoch % 5 == 0 or no_improve >= PATIENCE:
            lr_now = optimiser.param_groups[0]["lr"]
            print(f"    Epoch {epoch:>3}  train={t_loss:.6f}  val={v_loss:.6f}  "
                  f"lr={lr_now:.1e}  {flag}")

        if no_improve >= PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0

    # --- Restore best & test ---
    model.load_state_dict(best_state)
    model.to(device)

    preds_norm, targets_norm = predict_all(model, test_loader, device)
    preds_mw = preds_norm * scaler["std"] + scaler["mean"]
    targets_mw = targets_norm * scaler["std"] + scaler["mean"]

    val_rmse_mw = math.sqrt(best_val) * scaler["std"]
    test_rmse_mw = compute_rmse(targets_mw, preds_mw)

    print(f"    Val RMSE: {val_rmse_mw:.0f} MW | Test RMSE: {test_rmse_mw:.0f} MW "
          f"| {len(history['train_loss'])} epochs | {elapsed:.0f}s")

    return {
        "label": label,
        "cfg": cfg,
        "best_val_mse": best_val,
        "val_rmse_mw": val_rmse_mw,
        "test_rmse_mw": test_rmse_mw,
        "epochs": len(history["train_loss"]),
        "time_s": elapsed,
        "history": history,
        "state_dict": best_state,
        "scaler": scaler,
        "preds_mw": preds_mw,
        "targets_mw": targets_mw,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_bar_chart(results, save_path):
    """Bar chart comparing test RMSE across experiments."""
    labels = [r["label"] for r in results]
    test_rmses = [r["test_rmse_mw"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 5))
    colours = ["#d62728" if r == max(test_rmses) else
               "#2ca02c" if r == min(test_rmses) else "#1f77b4"
               for r in test_rmses]
    bars = ax.bar(range(len(labels)), test_rmses, color=colours)

    for bar, val in zip(bars, test_rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Test RMSE (MW)", fontsize=11)
    ax.set_title("Baseline Sweep -- Test RMSE Comparison", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    all_results = []

    # -----------------------------------------------------------------------
    # GROUP A: Regularisation (train 2001-2017)
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# GROUP A: Regularisation")
    print("#" * 60)

    group_a_configs = [
        {
            "label": "A1: baseline",
            "train_start": 2001, "train_end": 2017,
            "hidden": 128, "layers": 2, "dropout": 0.0,
            "lr": 0.001, "weight_decay": 0.0, "scheduler": False,
        },
        {
            "label": "A2: +dropout",
            "train_start": 2001, "train_end": 2017,
            "hidden": 128, "layers": 2, "dropout": 0.2,
            "lr": 0.001, "weight_decay": 0.0, "scheduler": False,
        },
        {
            "label": "A3: +drop+sched",
            "train_start": 2001, "train_end": 2017,
            "hidden": 128, "layers": 2, "dropout": 0.2,
            "lr": 0.001, "weight_decay": 0.0, "scheduler": True,
        },
        {
            "label": "A4: +drop+sched+wd",
            "train_start": 2001, "train_end": 2017,
            "hidden": 128, "layers": 2, "dropout": 0.2,
            "lr": 0.001, "weight_decay": 1e-4, "scheduler": True,
        },
    ]

    for cfg in group_a_configs:
        all_results.append(run_experiment(cfg, device))

    # Pick best Group A config by val RMSE
    best_a = min(all_results, key=lambda r: r["val_rmse_mw"])
    best_a_cfg = best_a["cfg"]
    print(f"\n>>> Best Group A: {best_a['label']} "
          f"(val={best_a['val_rmse_mw']:.0f}, test={best_a['test_rmse_mw']:.0f} MW)")

    # -----------------------------------------------------------------------
    # GROUP B: Training data recency (use best regularisation from A)
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# GROUP B: Training Data Recency")
    print("#" * 60)

    for start_yr, lbl in [(2009, "B5: 2009-2017"), (2012, "B6: 2012-2017"),
                           (2015, "B7: 2015-2017")]:
        cfg = {**best_a_cfg, "label": lbl, "train_start": start_yr, "train_end": 2017}
        all_results.append(run_experiment(cfg, device))

    # Best from A + B combined
    best_ab = min(all_results, key=lambda r: r["val_rmse_mw"])
    best_ab_cfg = best_ab["cfg"]
    print(f"\n>>> Best Group A+B: {best_ab['label']} "
          f"(val={best_ab['val_rmse_mw']:.0f}, test={best_ab['test_rmse_mw']:.0f} MW)")

    # -----------------------------------------------------------------------
    # GROUP C: Architecture (use best regularisation + years from A+B)
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# GROUP C: Architecture")
    print("#" * 60)

    for hidden, layers, lbl in [(64, 2, "C8: hidden=64"),
                                 (128, 3, "C9: 3 layers")]:
        cfg = {**best_ab_cfg, "label": lbl, "hidden": hidden, "layers": layers}
        all_results.append(run_experiment(cfg, device))

    # -----------------------------------------------------------------------
    # Overall best
    # -----------------------------------------------------------------------
    best = min(all_results, key=lambda r: r["val_rmse_mw"])
    print(f"\n>>> OVERALL BEST: {best['label']} "
          f"(val={best['val_rmse_mw']:.0f}, test={best['test_rmse_mw']:.0f} MW)")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'='*95}")
    print(f"{'#':>3} | {'Config':<22} | {'Train Years':<12} | "
          f"{'Val RMSE':>10} | {'Test RMSE':>10} | {'Epochs':>6} | {'Time':>6}")
    print(f"{'-'*95}")
    for i, r in enumerate(all_results, 1):
        yrs = f"{r['cfg']['train_start']}-{r['cfg']['train_end']}"
        print(f"{i:>3} | {r['label']:<22} | {yrs:<12} | "
              f"{r['val_rmse_mw']:>8.0f} MW | {r['test_rmse_mw']:>8.0f} MW | "
              f"{r['epochs']:>6} | {r['time_s']:>5.0f}s")
    print(f"{'='*95}")

    # -----------------------------------------------------------------------
    # Save best model + figures
    # -----------------------------------------------------------------------
    best_model_path = os.path.join(SAVE_DIR, "deterministic_lstm_best.pt")
    torch.save(best["state_dict"], best_model_path)
    print(f"\nSaved best model -> {best_model_path}")

    # Save best scaler
    best_scaler_path = os.path.join(SAVE_DIR, "scaler_params_best.json")
    with open(best_scaler_path, "w") as f:
        json.dump(best["scaler"], f, indent=2)

    # Save sweep results summary
    summary = []
    for r in all_results:
        summary.append({
            "label": r["label"],
            "train_years": f"{r['cfg']['train_start']}-{r['cfg']['train_end']}",
            "val_rmse_mw": round(float(r["val_rmse_mw"]), 1),
            "test_rmse_mw": round(float(r["test_rmse_mw"]), 1),
            "epochs": int(r["epochs"]),
            "time_s": round(float(r["time_s"]), 1),
            "config": {k: (float(v) if isinstance(v, (np.floating,)) else v)
                       for k, v in r["cfg"].items() if k != "label"},
        })
    with open(os.path.join(SAVE_DIR, "sweep_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Figure 1: Bar chart comparison
    bar_path = os.path.join(FIG_DIR, "baseline_sweep_comparison.png")
    plot_bar_chart(all_results, bar_path)
    print(f"Saved: {bar_path}")

    # Figure 2: Best training curve
    curve_path = os.path.join(FIG_DIR, "best_baseline_training_curve.png")
    plot_training_curves(
        best["history"]["train_loss"], best["history"]["val_loss"],
        save_path=curve_path,
        title=f"Best Config ({best['label']}) -- Training Curve",
    )
    print(f"Saved: {curve_path}")

    # Figure 3: Best 1-week prediction
    week = 336
    start = len(best["preds_mw"]) // 3
    pred_path = os.path.join(FIG_DIR, "best_baseline_predictions.png")
    plot_predictions(
        best["targets_mw"][start:start + week],
        best["preds_mw"][start:start + week],
        save_path=pred_path,
        title=f"Best Config ({best['label']}) -- Test RMSE: {best['test_rmse_mw']:.0f} MW",
    )
    print(f"Saved: {pred_path}")


if __name__ == "__main__":
    main()
