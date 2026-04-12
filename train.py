"""
Unified Training Script - All Models


Trains all three models sequentially:
    1. Deterministic LSTM (baseline)
    2. Gaussian LSTM
    3. MC Dropout LSTM

Usage:
    python train.py

Saved models:
    saved_models/deterministic_lstm.pt
    saved_models/gaussian_lstm.pt
    saved_models/mc_dropout_lstm.pt
"""

import subprocess
import sys
import time

scripts = [
    ("Deterministic LSTM (Baseline)", "train_deterministic.py"),
    ("Gaussian LSTM",                 "train_gaussian.py"),
    ("MC Dropout LSTM",               "train_mc_dropout.py"),
]

print("=" * 60)
print("COMP0197 — Training all models")
print("=" * 60)

for name, script in scripts:
    print(f"\n>>> Training: {name}")
    print(f"    Running: python {script}")
    print("-" * 60)

    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed with return code {result.returncode}.")
        print("Stopping training pipeline.")
        sys.exit(result.returncode)

    print(f"\n    Done in {elapsed:.1f}s")

print("\n" + "=" * 60)
print("All models trained successfully.")
print("Weights saved in saved_models/")
print("Run 'python test.py' to evaluate and generate figures.")
print("=" * 60)