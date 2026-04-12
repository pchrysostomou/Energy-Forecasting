# UK Bank Holiday Indicator Ablation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a binary `is_holiday` feature for UK bank holidays and run an ablation comparing model performance with and without it across all three models (Deterministic, Gaussian, MC Dropout).

**Architecture:** Extend `src/data_loader.py` to generate an `is_holiday` column using the `holidays` Python package. Then write a new experiment script `experiments/ablation_holiday.py` that mirrors the style of `experiments/ablation_temperature.py` — training each of the three models twice (without and with `is_holiday`) and evaluating on RMSE, NLL, CRPS, and calibration error.

**Tech Stack:** Python, PyTorch, `holidays` package (pip install holidays), existing src.models (deterministic, gaussian, mc_dropout), existing src.evaluate metrics.

---

### Task 1: Add `is_holiday` feature to data_loader.py

**Files:**
- Modify: `src/data_loader.py` — inside `select_csvtopd()`

**Step 1: Install the holidays package**

```bash
pip install holidays
```

**Step 2: Add the feature inside `select_csvtopd()`**

In `src/data_loader.py`, add a new parameter `include_holidays=False` to `select_csvtopd()` and `data_preprocess()`.

Inside `select_csvtopd()`, after the cyclical encodings block (around line 258), add:

```python
if include_holidays:
    import holidays as hols
    uk_holidays = hols.country_holidays('GB', subdiv='ENG')
    df['is_holiday'] = df['SETTLEMENT_DATE'].dt.date.apply(
        lambda d: 1 if d in uk_holidays else 0
    ).astype(int)
```

Also thread the `include_holidays` parameter through `data_preprocess()`:

```python
def data_preprocess(..., include_holidays=False):
    ...
    train_df = select_csvtopd(train_start, train_end, include_temperature, include_holidays)
    valid_df = select_csvtopd(valid_start, valid_end, include_temperature, include_holidays)
    test_df  = select_csvtopd(test_start,  test_end,  include_temperature, include_holidays)
```

**Step 3: Verify manually**

Run from repo root:
```bash
python -c "
from src.data_loader import data_preprocess
train_df, _, _, _ = data_preprocess(train_start=2015, train_end=2017, include_holidays=True)
print(train_df.columns.tolist())
print('Holiday rows:', train_df['is_holiday'].sum())
print(train_df[train_df['is_holiday'] == 1][['is_holiday']].head())
"
```
Expected: `is_holiday` in columns, non-zero holiday count (UK has ~8 bank holidays/year → ~8 × 3 years × 48 = ~1152 rows should be 1).

**Step 4: Commit**

```bash
git add src/data_loader.py
git commit -m "feat: add is_holiday feature to data_loader"
```

---

### Task 2: Write the ablation experiment script

**Files:**
- Create: `experiments/ablation_holiday.py`

**Step 1: Create the script**

Use `experiments/ablation_temperature.py` as a structural template. The script should:

- Define `CONDITIONS = [False, True]` (without/with `is_holiday`)
- For each condition, train all three models (Deterministic, Gaussian, MC Dropout) with `SEQ_LEN=336`, `HIDDEN_SIZE=64`, `NUM_LAYERS=2`, `EPOCHS=100`, `PATIENCE=10`
- Always include temperature (`include_temperature=True`) so the only variable is the holiday flag
- Evaluate each model: RMSE, NLL (where applicable), CRPS, calibration error
- Save results to `saved_models/ablation_holiday_results.json`
- Save figure to `figures/ablation_holiday.png`

**Step 2: Script structure outline**

```python
"""
Ablation Study: UK Bank Holiday Indicator

Tests whether adding a binary is_holiday feature improves forecast quality.
Trains all three models (Deterministic, Gaussian, MC Dropout) with and without
the holiday flag. Temperature is always included to isolate the effect.

Results saved to:
    - saved_models/ablation_holiday_results.json
    - figures/ablation_holiday.png
"""

CONDITIONS = [
    {"include_holidays": False, "label": "Without Holiday Feature"},
    {"include_holidays": True,  "label": "With Holiday Feature"},
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

For each (condition, model) pair:
1. Call `data_preprocess(include_temperature=True, include_holidays=cond["include_holidays"])`
2. Instantiate the relevant model class
3. Train with early stopping
4. Evaluate: RMSE for all; additionally NLL + CRPS + calibration for Gaussian and MC Dropout
5. Append result dict to `results`

**Step 3: Figure layout (2×3 grid)**

- Row 1: RMSE for Deterministic / Gaussian / MC Dropout (bar chart: without vs with holiday)
- Row 2: CRPS for Gaussian / CRPS for MC Dropout / Calibration error for MC Dropout

**Step 4: Commit**

```bash
git add experiments/ablation_holiday.py
git commit -m "feat: add holiday indicator ablation experiment"
```

---

### Task 3: Run the experiment and save outputs

**Step 1: Run the script**

```bash
python experiments/ablation_holiday.py
```

Expected runtime: ~30–60 min (6 model trains total).

**Step 2: Verify outputs exist**

```bash
ls saved_models/ablation_holiday_results.json
ls figures/ablation_holiday.png
```

**Step 3: Inspect results**

```bash
python -c "
import json
with open('saved_models/ablation_holiday_results.json') as f:
    results = json.load(f)
for r in results:
    print(r)
"
```

**Step 4: Commit outputs**

```bash
git add saved_models/ablation_holiday_results.json figures/ablation_holiday.png
git commit -m "results: holiday indicator ablation complete"
```

---

### What to look for in results

- **RMSE**: Expect a small but consistent drop across all three models when `is_holiday=True`.
- **Calibration error (MC Dropout)**: Should improve noticeably — holiday dates were previously high-uncertainty (model didn't know it was a holiday), now the model can condition on it.
- **σ mean (Gaussian LSTM)**: Should decrease on holiday periods when the feature is present — aleatoric uncertainty tightens because the model now knows the regime.
- **If no change**: Still interesting — it means the LSTM's sequence memory already implicitly captures holiday patterns from surrounding demand drops, which is a valid finding.
