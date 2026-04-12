"""
Extended Analysis: Richer visualisations from ablation JSON results.

Produces four new figures from saved JSON results (no retraining needed):

    figures/ken_lag_full_metrics.png        -- NRMSE%, CRPS, NLL, Cal, T_opt
    figures/ken_lag_improvement.png         -- % improvement over config A (no lags)
    figures/ken_holiday_delta.png           -- Holiday delta as horizontal bar chart
    figures/ken_comprehensive_summary.png   -- Heatmap of all metrics, both ablations

Usage:
    python experiments/ablation_extended_analysis.py
"""

import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
SAVE_DIR  = os.path.join(PROJECT_ROOT, "saved_models")
FIG_DIR   = os.path.join(PROJECT_ROOT, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(os.path.join(SAVE_DIR, "ablation_lag_features_results.json")) as f:
    lag_results = json.load(f)

with open(os.path.join(SAVE_DIR, "ablation_holiday_results.json")) as f:
    hol_results = json.load(f)

with open(os.path.join(SAVE_DIR, "scaler_params_best.json")) as f:
    scaler = json.load(f)

MEAN_DEMAND = scaler["mean"]   # 31,323 MW

# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------
for r in lag_results:
    r["det_nrmse_pct"]   = r["det_rmse_mw"]   / MEAN_DEMAND * 100
    r["gauss_nrmse_pct"] = r["gauss_rmse_mw"] / MEAN_DEMAND * 100
    r["mc_nrmse_pct"]    = r["mc_rmse_mw"]    / MEAN_DEMAND * 100

for r in hol_results:
    r["det_nrmse_pct"]   = r["det_rmse_mw"]   / MEAN_DEMAND * 100
    r["gauss_nrmse_pct"] = r["gauss_rmse_mw"] / MEAN_DEMAND * 100
    r["mc_nrmse_pct"]    = r["mc_rmse_mw"]    / MEAN_DEMAND * 100

# Baseline (config A) for % improvement
base = lag_results[0]

def pct_improvement(baseline_val, val):
    """Returns % improvement (positive = better, negative = worse)."""
    return (baseline_val - val) / baseline_val * 100

# Short config labels
short_labels = ["A\nNo lags", "B\n+Yesterday", "C\n+Week", "D\n+Month\n+Year"]
x = np.arange(len(lag_results))


# ===========================================================================
# Figure 1: Lag ablation — full metrics panel (6 subplots)
# ===========================================================================
fig1, axes = plt.subplots(2, 3, figsize=(16, 10))

colors = {"det": "#1f77b4", "gauss": "#2ca02c", "mc": "#d62728"}
markers = {"det": "o", "gauss": "s", "mc": "^"}

def line(ax, y, label, model, annotate_fmt="{:.1f}"):
    ax.plot(x, y, marker=markers[model], color=colors[model],
            label=label, linewidth=2, markersize=8)
    for xi, yi in zip(x, y):
        ax.annotate(annotate_fmt.format(yi), (xi, yi),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color=colors[model])

# --- Panel 1: NRMSE% ---
ax = axes[0, 0]
line(ax, [r["det_nrmse_pct"]   for r in lag_results], "Deterministic", "det", "{:.1f}%")
line(ax, [r["gauss_nrmse_pct"] for r in lag_results], "Gaussian",      "gauss", "{:.1f}%")
line(ax, [r["mc_nrmse_pct"]    for r in lag_results], "MC Dropout",    "mc", "{:.1f}%")
ax.set_title("Normalised RMSE (% of mean demand)")
ax.set_ylabel("NRMSE (%)")
ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=8)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))

# --- Panel 2: CRPS (MW) ---
ax = axes[0, 1]
line(ax, [r["gauss_crps_mw"] for r in lag_results], "Gaussian",   "gauss", "{:,.0f}")
line(ax, [r["mc_crps_mw"]    for r in lag_results], "MC Dropout", "mc", "{:,.0f}")
ax.set_title("CRPS (MW) — Probabilistic Quality")
ax.set_ylabel("CRPS (MW)")
ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=8)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# --- Panel 3: NLL ---
ax = axes[0, 2]
line(ax, [r["gauss_nll"] for r in lag_results], "Gaussian",   "gauss", "{:.2f}")
ax2 = ax.twinx()
ax2.plot(x, [r["mc_nll"] for r in lag_results],
         marker=markers["mc"], color=colors["mc"], linewidth=2,
         markersize=8, linestyle="--", label="MC Dropout (right)")
ax2.set_ylabel("MC Dropout NLL", color=colors["mc"])
ax2.tick_params(axis="y", labelcolor=colors["mc"])
ax.set_title("NLL (lower = better)\nDual axis: Gaussian left, MC right")
ax.set_ylabel("Gaussian NLL", color=colors["gauss"])
ax.tick_params(axis="y", labelcolor=colors["gauss"])
ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=8)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 4: Calibration Error ---
ax = axes[1, 0]
line(ax, [r["gauss_cal_error"] for r in lag_results], "Gaussian",   "gauss", "{:.3f}")
line(ax, [r["mc_cal_error"]    for r in lag_results], "MC Dropout", "mc", "{:.3f}")
ax.set_title("Calibration Error (lower = better)")
ax.set_ylabel("Calibration Error")
ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=8)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# --- Panel 5: T_opt (calibration temperature) ---
ax = axes[1, 1]
line(ax, [r["gauss_T"] for r in lag_results], "Gaussian",   "gauss", "{:.2f}")
ax2 = ax.twinx()
ax2.plot(x, [r["mc_T"] for r in lag_results],
         marker=markers["mc"], color=colors["mc"], linewidth=2, markersize=8,
         linestyle="--", label="MC Dropout (right)")
ax2.set_ylabel("MC Dropout T_opt", color=colors["mc"])
ax2.tick_params(axis="y", labelcolor=colors["mc"])
ax.set_title("T_opt — Temperature Scaling Needed\n(closer to 1.0 = better calibrated raw uncertainty)")
ax.set_ylabel("Gaussian T_opt", color=colors["gauss"])
ax.tick_params(axis="y", labelcolor=colors["gauss"])
ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=8)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(1.0, color=colors["gauss"], linestyle=":", alpha=0.5, label="T=1 (perfect)")
ax2.axhline(1.0, color=colors["mc"],    linestyle=":", alpha=0.5)

# --- Panel 6: Sharpness proxy: CRPS / RMSE ratio ---
# Lower ratio = uncertainty quality improving faster than point accuracy
ax = axes[1, 2]
gauss_ratio = [r["gauss_crps_mw"] / r["gauss_rmse_mw"] for r in lag_results]
mc_ratio    = [r["mc_crps_mw"]    / r["mc_rmse_mw"]    for r in lag_results]
line(ax, gauss_ratio, "Gaussian",   "gauss", "{:.3f}")
line(ax, mc_ratio,    "MC Dropout", "mc", "{:.3f}")
ax.set_title("CRPS / RMSE Ratio\n(lower = sharper uncertainty relative to point error)")
ax.set_ylabel("CRPS / RMSE")
ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=8)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig1.suptitle("Ablation Study 5: Lag Features — Full Metrics Analysis", fontsize=13)
fig1.tight_layout()
p1 = os.path.join(FIG_DIR, "ken_lag_full_metrics.png")
fig1.savefig(p1, dpi=200, bbox_inches="tight")
plt.close(fig1)
print(f"Saved -> {p1}")


# ===========================================================================
# Figure 2: % improvement over config A (no lags)
# ===========================================================================
fig2, axes = plt.subplots(1, 3, figsize=(15, 6))

metrics = [
    ("RMSE",     [r["det_rmse_mw"]    for r in lag_results],
                 [r["gauss_rmse_mw"]  for r in lag_results],
                 [r["mc_rmse_mw"]     for r in lag_results]),
    ("CRPS",     [None]*4,
                 [r["gauss_crps_mw"]  for r in lag_results],
                 [r["mc_crps_mw"]     for r in lag_results]),
    ("Cal Error",[None]*4,
                 [r["gauss_cal_error"]for r in lag_results],
                 [r["mc_cal_error"]   for r in lag_results]),
]

for ax, (metric, det_vals, gauss_vals, mc_vals) in zip(axes, metrics):
    w = 0.25
    for i, (vals, model, label) in enumerate([
        (det_vals,   "det",   "Deterministic"),
        (gauss_vals, "gauss", "Gaussian"),
        (mc_vals,    "mc",    "MC Dropout"),
    ]):
        if vals[0] is None:
            continue
        impr = [0.0] + [pct_improvement(vals[0], v) for v in vals[1:]]
        offset = (i - 1) * w
        bars = ax.bar(x + offset, impr, w, label=label, color=colors[model])
        for bar, val in zip(bars, impr):
            if abs(val) > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (1 if val >= 0 else -3),
                        f"{val:+.1f}%", ha="center", fontsize=7,
                        color=colors[model])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{metric} — % Improvement vs Config A")
    ax.set_ylabel("% Improvement (positive = better)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))

fig2.suptitle("Ablation Study 5: % Improvement Over Baseline (A: No Lags)", fontsize=13)
fig2.tight_layout()
p2 = os.path.join(FIG_DIR, "ken_lag_improvement.png")
fig2.savefig(p2, dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"Saved -> {p2}")


# ===========================================================================
# Figure 3: Holiday ablation — horizontal delta bar chart
# ===========================================================================
fig3, axes = plt.subplots(1, 2, figsize=(14, 7))

without = hol_results[0]
with_h  = hol_results[1]

deltas = [
    ("Deterministic\nRMSE (MW)",  with_h["det_rmse_mw"]     - without["det_rmse_mw"],     "rmse"),
    ("Gaussian\nRMSE (MW)",       with_h["gauss_rmse_mw"]   - without["gauss_rmse_mw"],   "rmse"),
    ("Gaussian\nNLL",             with_h["gauss_nll"]        - without["gauss_nll"],        "prob"),
    ("Gaussian\nCRPS (MW)",       with_h["gauss_crps_mw"]   - without["gauss_crps_mw"],   "rmse"),
    ("Gaussian\nCal Error",       with_h["gauss_cal_error"]  - without["gauss_cal_error"],  "prob"),
    ("MC Dropout\nRMSE (MW)",     with_h["mc_rmse_mw"]      - without["mc_rmse_mw"],       "rmse"),
    ("MC Dropout\nNLL",           with_h["mc_nll"]           - without["mc_nll"],           "prob"),
    ("MC Dropout\nCRPS (MW)",     with_h["mc_crps_mw"]      - without["mc_crps_mw"],       "rmse"),
    ("MC Dropout\nCal Error",     with_h["mc_cal_error"]     - without["mc_cal_error"],     "prob"),
]

labels_d  = [d[0] for d in deltas]
values_d  = [d[1] for d in deltas]
bar_colors = ["#2ca02c" if v < 0 else "#d62728" for v in values_d]

ax = axes[0]
bars = ax.barh(labels_d, values_d, color=bar_colors)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("Delta (With Holiday − Without Holiday)")
ax.set_title("Holiday Feature: Delta per Metric\nGreen = improved, Red = degraded")
for bar, val in zip(bars, values_d):
    pad = max(abs(v) for v in values_d) * 0.02
    ax.text(val + (pad if val >= 0 else -pad),
            bar.get_y() + bar.get_height()/2,
            f"{val:+.3f}" if abs(val) < 10 else f"{val:+.0f}",
            va="center", ha="left" if val >= 0 else "right", fontsize=8)
ax.grid(True, alpha=0.3, axis="x")

# Panel 2: NRMSE% comparison
ax2 = axes[1]
model_labels = ["Deterministic", "Gaussian", "MC Dropout"]
nrmse_without = [without["det_nrmse_pct"], without["gauss_nrmse_pct"], without["mc_nrmse_pct"]]
nrmse_with    = [with_h["det_nrmse_pct"],  with_h["gauss_nrmse_pct"],  with_h["mc_nrmse_pct"]]

xpos = np.arange(len(model_labels))
w = 0.35
b1 = ax2.bar(xpos - w/2, nrmse_without, w, label="Without Holiday", color="#ff7f0e")
b2 = ax2.bar(xpos + w/2, nrmse_with,    w, label="With Holiday",    color="#1f77b4")
for b, v in list(zip(b1, nrmse_without)) + list(zip(b2, nrmse_with)):
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
             f"{v:.1f}%", ha="center", fontsize=9)
ax2.set_title("NRMSE (% of mean demand 31,324 MW)")
ax2.set_ylabel("NRMSE (%)")
ax2.set_xticks(xpos)
ax2.set_xticklabels(model_labels)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))

fig3.suptitle("Ablation Study 4: Holiday Indicator — Detailed Impact", fontsize=13)
fig3.tight_layout()
p3 = os.path.join(FIG_DIR, "ken_holiday_delta.png")
fig3.savefig(p3, dpi=200, bbox_inches="tight")
plt.close(fig3)
print(f"Saved -> {p3}")


# ===========================================================================
# Figure 4: Comprehensive summary heatmap — both ablations
# ===========================================================================
fig4, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Left: Lag ablation heatmap ---
lag_row_labels = [r["lag_config"] for r in lag_results]
lag_col_labels = [
    "Det\nNRMSE%", "Det\nRMSE↓",
    "Gauss\nNRMSE%", "Gauss\nCRPS↓", "Gauss\nCal↓", "Gauss\nT_opt",
    "MC\nNRMSE%",  "MC\nCRPS↓",   "MC\nCal↓",   "MC\nT_opt",
]
lag_data = np.array([
    [r["det_nrmse_pct"],  r["det_rmse_mw"],
     r["gauss_nrmse_pct"],r["gauss_crps_mw"], r["gauss_cal_error"], r["gauss_T"],
     r["mc_nrmse_pct"],   r["mc_crps_mw"],    r["mc_cal_error"],    r["mc_T"]]
    for r in lag_results
])

# Normalise each column 0-1 (0=best, 1=worst) for colour
lag_norm = np.zeros_like(lag_data)
for col in range(lag_data.shape[1]):
    mn, mx = lag_data[:, col].min(), lag_data[:, col].max()
    if mx > mn:
        lag_norm[:, col] = (lag_data[:, col] - mn) / (mx - mn)
    else:
        lag_norm[:, col] = 0.5

ax = axes[0]
im = ax.imshow(lag_norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(lag_col_labels)))
ax.set_xticklabels(lag_col_labels, fontsize=8)
ax.set_yticks(range(len(lag_row_labels)))
ax.set_yticklabels(lag_row_labels, fontsize=9)
ax.set_title("Lag Features Ablation\n(green = best, red = worst per column)", fontsize=10)

for i in range(lag_data.shape[0]):
    for j in range(lag_data.shape[1]):
        val = lag_data[i, j]
        txt = f"{val:.1f}%" if j in [0, 2, 6] else (
              f"{val:.2f}"  if j in [5, 9] else (
              f"{val:.3f}"  if j in [4, 8] else f"{val:,.0f}"))
        ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                color="black" if 0.3 < lag_norm[i, j] < 0.8 else "white"
                if lag_norm[i, j] >= 0.8 else "black")

# --- Right: Holiday ablation heatmap ---
hol_row_labels = [r["condition"] for r in hol_results]
hol_col_labels = [
    "Det\nNRMSE%", "Det\nRMSE↓",
    "Gauss\nNRMSE%","Gauss\nNLL↓",  "Gauss\nCRPS↓","Gauss\nCal↓","Gauss\nT_opt",
    "MC\nNRMSE%",   "MC\nNLL↓",     "MC\nCRPS↓",   "MC\nCal↓",   "MC\nT_opt",
]
hol_data = np.array([
    [r["det_nrmse_pct"],   r["det_rmse_mw"],
     r["gauss_nrmse_pct"], r["gauss_nll"],    r["gauss_crps_mw"], r["gauss_cal_error"], r["gauss_T"],
     r["mc_nrmse_pct"],    r["mc_nll"],       r["mc_crps_mw"],    r["mc_cal_error"],    r["mc_T"]]
    for r in hol_results
])

hol_norm = np.zeros_like(hol_data)
for col in range(hol_data.shape[1]):
    mn, mx = hol_data[:, col].min(), hol_data[:, col].max()
    if mx > mn:
        hol_norm[:, col] = (hol_data[:, col] - mn) / (mx - mn)
    else:
        hol_norm[:, col] = 0.5

ax2 = axes[1]
ax2.imshow(hol_norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
ax2.set_xticks(range(len(hol_col_labels)))
ax2.set_xticklabels(hol_col_labels, fontsize=8)
ax2.set_yticks(range(len(hol_row_labels)))
ax2.set_yticklabels(hol_row_labels, fontsize=9)
ax2.set_title("Holiday Indicator Ablation\n(green = best, red = worst per column)", fontsize=10)

for i in range(hol_data.shape[0]):
    for j in range(hol_data.shape[1]):
        val = hol_data[i, j]
        txt = f"{val:.1f}%" if j in [0, 2, 7] else (
              f"{val:.2f}"  if j in [3, 6, 8, 11] else (
              f"{val:.3f}"  if j in [5, 10] else f"{val:,.0f}"))
        ax2.text(j, i, txt, ha="center", va="center", fontsize=7,
                 color="black" if 0.3 < hol_norm[i, j] < 0.8 else "white"
                 if hol_norm[i, j] >= 0.8 else "black")

fig4.suptitle("Comprehensive Metrics Summary — Both Ablation Studies", fontsize=13)
fig4.tight_layout()
p4 = os.path.join(FIG_DIR, "ken_comprehensive_summary.png")
fig4.savefig(p4, dpi=200, bbox_inches="tight")
plt.close(fig4)
print(f"Saved -> {p4}")

# ===========================================================================
# Print derived metrics table
# ===========================================================================
print(f"\n{'='*70}")
print("  DERIVED METRICS — LAG FEATURES")
print(f"{'='*70}")
print(f"{'Config':<22} {'Det NRMSE%':>11} {'Gauss NRMSE%':>13} {'MC NRMSE%':>11} "
      f"{'Gauss T_opt':>12} {'MC T_opt':>9}")
print("-"*70)
for r in lag_results:
    print(f"{r['lag_config']:<22} {r['det_nrmse_pct']:>10.1f}% "
          f"{r['gauss_nrmse_pct']:>12.1f}% {r['mc_nrmse_pct']:>10.1f}% "
          f"{r['gauss_T']:>12.2f} {r['mc_T']:>9.2f}")

print(f"\n{'='*70}")
print("  % IMPROVEMENT OVER CONFIG A")
print(f"{'='*70}")
print(f"{'Config':<22} {'Det RMSE':>10} {'Gauss RMSE':>12} {'Gauss CRPS':>12} "
      f"{'MC RMSE':>9} {'MC CRPS':>9} {'MC Cal':>8}")
print("-"*70)
for r in lag_results:
    print(f"{r['lag_config']:<22} "
          f"{pct_improvement(base['det_rmse_mw'],   r['det_rmse_mw']):>+9.1f}% "
          f"{pct_improvement(base['gauss_rmse_mw'], r['gauss_rmse_mw']):>+11.1f}% "
          f"{pct_improvement(base['gauss_crps_mw'], r['gauss_crps_mw']):>+11.1f}% "
          f"{pct_improvement(base['mc_rmse_mw'],    r['mc_rmse_mw']):>+8.1f}% "
          f"{pct_improvement(base['mc_crps_mw'],    r['mc_crps_mw']):>+8.1f}% "
          f"{pct_improvement(base['mc_cal_error'],  r['mc_cal_error']):>+7.1f}%")

print(f"\nAll figures saved to: {FIG_DIR}")
