# Probabilistic Energy Demand Forecasting

This project develops a deep learning system for UK electricity demand forecasting with uncertainty quantification. The task is formulated as probabilistic forecasting, allowing the models to represent both aleatoric uncertainty (data noise) and epistemic uncertainty (model uncertainty).

Three LSTM-based approaches are compared:

- Deterministic LSTM baseline  
- Gaussian LSTM (aleatoric uncertainty)  
- MC Dropout LSTM (epistemic uncertainty)  

## Model Comparison

| Model | Output | Uncertainty Type | Loss |
|---|---|---|---|
| Deterministic LSTM | Point prediction | None | MSE |
| Gaussian LSTM | Mean and standard deviation | Aleatoric | NLL |
| MC Dropout LSTM | Stochastic predictions | Epistemic | MSE with dropout at test time |

## Setup

The project runs in the course's `comp0197-pt` micromamba environment. The
environment ships with Python 3.12, PyTorch 2.10 (CPU build), torchvision and
NumPy. Three additional packages are required (the maximum allowed by the
brief):

```bash
micromamba activate comp0197-pt
pip install pandas matplotlib scipy
```

No internet access is required at run time — the demand CSVs, the cached
London temperature series, and the final trained model weights are all
shipped inside the submission folder.

## Main Commands

```bash
# Train all three models from scratch (deterministic -> gaussian -> mc dropout)
python train.py

# Load the saved weights and produce all reported metrics + figures
python test.py

# Optional: rerun any of the ablation studies
python experiments/ablation_temperature.py
python experiments/ablation_seq_length.py
python experiments/ablation_dropout_rate.py
python experiments/ablation_holiday.py
python experiments/ablation_lag_features.py
```

## Project Structure

```text
src/
├── data_loader.py        # demand CSV loading, feature engineering, sliding windows
├── preprocessing.py
├── models/
│   ├── base_lstm.py      # shared LSTM encoder
│   ├── deterministic.py  # baseline (MSE)
│   ├── gaussian.py       # aleatoric (NLL)
│   └── mc_dropout.py     # epistemic (MC dropout at test time)
└── evaluate.py           # RMSE, NLL, CRPS, calibration, plots

configs/default.yaml
train.py / train_deterministic.py / train_gaussian.py / train_mc_dropout.py
test.py
experiments/               # ablation scripts (temperature, seq len, dropout, holiday, lag)
saved_models/              # trained .pt weights + JSON histories, scalers, T values
figures/                   # all reported figures
data/processed/data/       # half-hourly UK ESO demand CSVs (2001-2025)
data/raw/temperature_london.csv  # cached Open-Meteo London 2 m temperature
instruction.pdf            # reproduction steps + package list
```

## Notes

- Demand CSVs and the cached temperature file are checked into the repo, so
  `train.py` and `test.py` run fully offline. `train.py` retains a stdlib
  `urllib`-based fetch path for the temperature data in case the cache is
  ever deleted.
- The three additional packages (`pandas`, `matplotlib`, `scipy`) are the
  only `pip install`s required on top of `comp0197-pt`. Earlier drafts used
  `properscoring`, `holidays`, and `requests`; CRPS is now computed in
  closed form via `scipy.stats.norm`, UK bank holidays are hardcoded in
  `src/data_loader.py`, and the temperature fetch uses stdlib `urllib`.
- Full reproduction details are provided in `instruction.pdf`.
