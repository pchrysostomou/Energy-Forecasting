"""
Ablation Study 2: Sequence Length Sweep

Tests how much historical context the LSTM needs.
Compares seq_len = [96, 168, 336, 672] (2 days, 3.5 days, 1 week, 2 weeks).

Usage:
    python experiments/ablation_seq_length.py

Results saved to:
    - saved_models/ablation_seq_length_results.json
    - figures/ablation_seq_length.png
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
from src.models.deterministic import DeterministicLSTM, deterministic_loss
from src.evaluate import compute_rmse

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
# Config (same as best baseline C8, only seq_len varies)
# ---------------------------------------------------------------------------
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

SEQ_LENGTHS = [96, 168, 336, 672]
SEQ_LABELS = ["2 days", "3.5 days", "1 week", "2 weeks"]


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


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        preds.append(out["prediction"].cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds).squeeze(), np.concatenate(targets).squeeze()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load data once (with temperature, the winning config)
    print("Loading data ...")
    train_df, valid_df, test_df, scaler = data_preprocess(
        train_start=TRAIN_START, train_end=TRAIN_END,
        include_temperature=True,
    )

    results = []

    for seq_len, label in zip(SEQ_LENGTHS, SEQ_LABELS):
        print(f"\n{'='*60}")
        print(f"  SEQUENCE LENGTH: {seq_len} ({label})")
        print(f"{'='*60}")

        # Create dataloaders with this seq_len
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, valid_df, test_df,
            seq_len=seq_len,
            batch_size=BATCH_SIZE,
        )
        num_features = next(iter(train_loader))[0].shape[2]
        print(f"  Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

        # Create model
        model = DeterministicLSTM(
            input_size=num_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(device)

        # Training
        optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", patience=5, factor=0.5,
        )

        best_val = float("inf")
        best_state = None
        no_improve = 0
        epochs_trained = 0

        t0 = time.time()
        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimiser, deterministic_loss, device)
            val_loss = validate(model, val_loader, deterministic_loss, device)
            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            epochs_trained = epoch

            if epoch % 5 == 0 or no_improve == 0:
                val_rmse_mw = math.sqrt(val_loss) * scaler["std"]
                print(f"  Ep {epoch:>3}  Train={train_loss:.6f}  Val={val_loss:.6f}  "
                      f"ValRMSE={val_rmse_mw:.0f} MW  {'<- best' if no_improve == 0 else f'({no_improve}/{PATIENCE})'}")

            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - t0

        # Evaluate on test set
        model.load_state_dict(best_state)
        model.to(device)
        preds_norm, targets_norm = predict_all(model, test_loader, device)

        preds_mw = preds_norm * scaler["std"] + scaler["mean"]
        targets_mw = targets_norm * scaler["std"] + scaler["mean"]
        test_rmse = compute_rmse(targets_mw, preds_mw)

        val_rmse_mw = math.sqrt(best_val) * scaler["std"]

        print(f"\n  seq_len={seq_len} ({label})")
        print(f"  Val RMSE:  {val_rmse_mw:.0f} MW")
        print(f"  Test RMSE: {test_rmse:.0f} MW")
        print(f"  Epochs:    {epochs_trained}")
        print(f"  Time:      {elapsed:.0f}s")

        results.append({
            "seq_len": seq_len,
            "label": label,
            "val_rmse_mw": round(val_rmse_mw),
            "test_rmse_mw": round(test_rmse),
            "epochs": epochs_trained,
            "time_s": round(elapsed),
        })

    # --- Save results ---
    results_path = os.path.join(SAVE_DIR, "ablation_seq_length_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results -> {results_path}")

    # --- Generate figure ---
    seq_lens = [r["seq_len"] for r in results]
    test_rmses = [r["test_rmse_mw"] for r in results]
    times = [r["time_s"] for r in results]
    labels = [r["label"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # RMSE vs seq_len
    ax1.plot(seq_lens, test_rmses, "o-", color="#1f77b4", linewidth=2, markersize=8)
    ax1.set_xlabel("Sequence Length (half-hourly steps)", fontsize=11)
    ax1.set_ylabel("Test RMSE (MW)", fontsize=11)
    ax1.set_title("RMSE vs Sequence Length", fontsize=12)
    ax1.set_xticks(seq_lens)
    ax1.set_xticklabels([f"{s}\n({l})" for s, l in zip(seq_lens, labels)], fontsize=9)
    ax1.grid(True, alpha=0.3)
    for s, r in zip(seq_lens, test_rmses):
        ax1.annotate(f"{r:,}", (s, r), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9)

    # Highlight current default
    default_idx = seq_lens.index(336)
    ax1.scatter([336], [test_rmses[default_idx]], s=150, facecolors="none",
                edgecolors="#d62728", linewidths=2, zorder=5, label="Current default")
    ax1.legend(fontsize=9)

    # Training time vs seq_len
    ax2.bar(range(len(seq_lens)), times, color="#2ca02c", alpha=0.8)
    ax2.set_xlabel("Sequence Length", fontsize=11)
    ax2.set_ylabel("Training Time (seconds)", fontsize=11)
    ax2.set_title("Training Time vs Sequence Length", fontsize=12)
    ax2.set_xticks(range(len(seq_lens)))
    ax2.set_xticklabels([f"{s}\n({l})" for s, l in zip(seq_lens, labels)], fontsize=9)
    ax2.grid(True, alpha=0.3)
    for i, t in enumerate(times):
        ax2.text(i, t + 2, f"{t}s", ha="center", fontsize=9)

    fig.suptitle("Ablation Study 2: Sequence Length Sweep", fontsize=13, y=1.02)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "ablation_seq_length.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {fig_path}")

    # --- Print summary table ---
    print(f"\n{'='*60}")
    print(f"  SEQUENCE LENGTH ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'seq_len':>8} {'Context':>10} {'Val RMSE':>10} {'Test RMSE':>10} {'Epochs':>8} {'Time':>8}")
    print(f"{'-'*54}")
    for r in results:
        marker = " <-- default" if r["seq_len"] == 336 else ""
        print(f"{r['seq_len']:>8} {r['label']:>10} {r['val_rmse_mw']:>8,} MW {r['test_rmse_mw']:>8,} MW {r['epochs']:>8} {r['time_s']:>6}s{marker}")


if __name__ == "__main__":
    main()
