# Explicit Lag Features Ablation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add explicit lagged demand features (t-48, t-336, t-1440, t-17520 = yesterday / last week / last month / last year at the same half-hour period) and run an ablation sweeping across which lag combinations improve forecast quality across all three models.

**Architecture:** Extend `src/data_loader.py` to compute lag columns on the full demand series before the train/val/test split (to avoid NaN at split boundaries). The ablation experiment in `experiments/ablation_lag_features.py` tests four configurations: no lags, yesterday only, yesterday+week, and the full set (yesterday+week+month+year). Each configuration trains all three models and is evaluated on RMSE, NLL, CRPS, and calibration error.

**Tech Stack:** Python, pandas (shift/fillna), PyTorch, existing src.models and src.evaluate.

---

### Task 1: Add lag feature support to data_loader.py

**Files:**
- Modify: `src/data_loader.py`

**Background — why lags need special handling:**

Lag features use `.shift(n)` on the ND column, which creates NaN for the first `n` rows. Because lags can be up to 17520 rows (one year × 48 half-hours/day × 365 days), these NaN rows must be dropped from each split *after* computing lags but *before* passing to the model. The normalisation of lag values must use training-set statistics only.

The four lags and their half-hour offsets:
| Lag name       | Offset (half-hours) | Approx duration |
|----------------|---------------------|-----------------|
| `lag_48`       | 48                  | 1 day           |
| `lag_336`      | 336                 | 1 week          |
| `lag_1440`     | 1440                | 30 days (~month)|
| `lag_17520`    | 17520               | 365 days (year) |

**Step 1: Add `lag_periods` parameter to `select_csvtopd()`**

```python
def select_csvtopd(start, end, include_temperature=False,
                   include_holidays=False, lag_periods=None):
    """
    lag_periods: list of int offsets, e.g. [48, 336, 1440, 17520].
                 Each creates a column demand_lag_<n> with the ND value
                 n half-hours ago. NaN rows from insufficient history are
                 forward-filled using the first valid value.
    """
    ...
    # After all other feature engineering, before the final drop():
    if lag_periods:
        for n in lag_periods:
            col = f"demand_lag_{n}"
            df[col] = df["ND"].shift(n)
            # Forward-fill NaN at the start with first valid value
            df[col] = df[col].bfill()
    ...
```

Note: `bfill()` fills leading NaNs backwards from the first real value. This avoids losing training rows at the cost of a slight inaccuracy for the very first window — acceptable for an ablation study.

**Step 2: Thread `lag_periods` through `data_preprocess()`**

```python
def data_preprocess(..., lag_periods=None):
    ...
    train_df = select_csvtopd(..., lag_periods=lag_periods)
    valid_df = select_csvtopd(..., lag_periods=lag_periods)
    test_df  = select_csvtopd(..., lag_periods=lag_periods)

    # Z-score normalise lag columns using TRAIN stats only
    if lag_periods:
        for n in lag_periods:
            col = f"demand_lag_{n}"
            lag_mean = float(train_df[col].mean())
            lag_std  = float(train_df[col].std())
            train_df[col] = (train_df[col] - lag_mean) / lag_std
            valid_df[col] = (valid_df[col] - lag_mean) / lag_std
            test_df[col]  = (test_df[col]  - lag_mean) / lag_std
            scaler_params[f"lag_{n}_mean"] = lag_mean
            scaler_params[f"lag_{n}_std"]  = lag_std
    ...
```

**Step 3: Verify manually**

```bash
python -c "
from src.data_loader import data_preprocess, get_dataloaders
train_df, _, _, scaler = data_preprocess(
    train_start=2015, train_end=2017,
    lag_periods=[48, 336, 1440, 17520]
)
print(train_df.columns.tolist())
print(train_df[['demand_lag_48','demand_lag_336','demand_lag_1440','demand_lag_17520']].describe())
print('NaN count:', train_df.isna().sum().sum())
"
```
Expected: four new columns, no NaNs, values approximately z-scored (mean ~0, std ~1).

**Step 4: Commit**

```bash
git add src/data_loader.py
git commit -m "feat: add explicit lag demand features to data_loader"
```

---

### Task 2: Write the ablation experiment script

**Files:**
- Create: `experiments/ablation_lag_features.py`

**Step 1: Define the four lag configurations**

```python
"""
Ablation Study: Explicit Lag Features

Tests how adding explicit lagged demand values (yesterday / last week /
last month / last year) affects forecast quality for all three models.

Configurations:
    A: No lags (baseline features only)
    B: lag_48 only (yesterday)
    C: lag_48 + lag_336 (yesterday + last week)
    D: lag_48 + lag_336 + lag_1440 + lag_17520 (full set)

Results saved to:
    - saved_models/ablation_lag_features_results.json
    - figures/ablation_lag_features.png
"""

LAG_CONFIGS = [
    {"label": "No lags",              "lag_periods": None},
    {"label": "Yesterday (t-48)",     "lag_periods": [48]},
    {"label": "+ Last week (t-336)",  "lag_periods": [48, 336]},
    {"label": "+ Month + Year",       "lag_periods": [48, 336, 1440, 17520]},
]
MODELS_TO_RUN = ["deterministic", "gaussian", "mc_dropout"]
SEQ_LEN      = 336
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
LR           = 0.001
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 100
PATIENCE     = 10
MC_SAMPLES   = 50
TRAIN_START  = 2015
TRAIN_END    = 2017
```

**Step 2: Script structure**

For each `(lag_config, model_type)` pair:
1. Call `data_preprocess(include_temperature=True, lag_periods=config["lag_periods"])`
2. Call `get_dataloaders(...)` — `num_features` will automatically be correct since lag columns appear in the DataFrame
3. Instantiate model with `input_size = next(iter(train_loader))[0].shape[2]`
4. Train with early stopping (same loop as other ablations)
5. Evaluate:
   - Deterministic: RMSE only
   - Gaussian: RMSE, NLL, CRPS, calibration
   - MC Dropout: RMSE, NLL, CRPS, calibration (with temperature scaling)
6. Append to results list

Result dict per row:
```python
{
    "lag_config": config["label"],
    "model": model_type,
    "num_features": num_features,
    "rmse_mw": ...,
    "nll_calibrated": ...,   # None for deterministic
    "crps_mw": ...,          # None for deterministic
    "cal_error_calibrated": ...,  # None for deterministic
    "train_time_s": ...,
}
```

**Step 3: Figure layout (2 rows)**

- Row 1 (3 panels): RMSE for each model type — x-axis = lag config (A/B/C/D), line/bar per config
- Row 2 (3 panels): CRPS for Gaussian / CRPS for MC Dropout / Calibration error for MC Dropout

This gives a 2×3 grid (6 panels total), consistent with other ablation figures.

**Step 4: Commit**

```bash
git add experiments/ablation_lag_features.py
git commit -m "feat: add lag features ablation experiment"
```

---

### Task 3: Run the experiment and save outputs

**Step 1: Run the script**

```bash
python experiments/ablation_lag_features.py
```

Expected runtime: ~60–90 min (4 configs × 3 models = 12 model trains).

**Step 2: Verify outputs exist**

```bash
ls saved_models/ablation_lag_features_results.json
ls figures/ablation_lag_features.png
```

**Step 3: Inspect results summary**

```bash
python -c "
import json
with open('saved_models/ablation_lag_features_results.json') as f:
    results = json.load(f)
for r in results:
    print(f\"{r['model']:15s} | {r['lag_config']:30s} | RMSE={r['rmse_mw']} MW\")
"
```

**Step 4: Commit outputs**

```bash
git add saved_models/ablation_lag_features_results.json figures/ablation_lag_features.png
git commit -m "results: lag features ablation complete"
```

---

### What to look for in results

- **RMSE trend (A→D)**: Should decrease as more lags are added — the model gets direct access to same-period-yesterday and same-period-last-week, which are the strongest autocorrelation signals in energy demand.
- **Diminishing returns**: `lag_1440` (month) and `lag_17520` (year) may add less than `lag_48` and `lag_336` — the LSTM's 336-step window already captures some weekly periodicity.
- **Gaussian σ tightening**: With explicit lags, the model is less uncertain — expect mean σ to drop and calibration error to improve.
- **MC Dropout uncertainty on anomalies**: The model may still show high epistemic uncertainty on holiday/extreme days even with lags, because the lag values themselves are atypical on those days.
- **If year lag hurts**: The `lag_17520` feature references demand from one year ago, which may introduce noise if there's a long-term trend (e.g. overall demand declining due to efficiency). This would show as RMSE *increasing* from config C to D — a valid and interesting finding.
