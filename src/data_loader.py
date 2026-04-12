"""
Data Loading Pipeline

Owners: Yoyo & Rohan (Team A)

This module handles:
    1. Loading the raw UK National Grid energy demand data
    2. Feature engineering (temporal encodings, weather features)
    3. Creating PyTorch Dataset/DataLoader with sliding windows

=== CRITICAL INTERFACE CONTRACT ===

All models expect the DataLoader to produce batches of:
    x: (batch_size, seq_len, num_features)  — input sequence
    y: (batch_size, 1)                       — target value (next timestep demand)

Where:
    seq_len = 336 (one week of half-hourly data) — configurable via config
    num_features = 7 base features (8 with temperature) per timestep

Features per timestep (x):
    - is_weekend: binary        (1 for Sat/Sun, 0 otherwise)
    - hour_sin: float           (sin(2π * settlement_period / 48))
    - hour_cos: float           (cos(2π * settlement_period / 48))
    - day_of_week_sin: float    (sin(2π * day_of_week / 7))
    - day_of_week_cos: float    (cos(2π * day_of_week / 7))
    - day_of_year_sin: float    (sin(2π * day_of_year / 365.25))
    - day_of_year_cos: float    (cos(2π * day_of_year / 365.25))
    - temperature: float        (OPTIONAL — z-score normalised London 2m temp, via Open-Meteo)

Target (y):
    - ND: float                 (National Demand, z-score normalised using train-set stats)

=== TRAIN/VAL/TEST SPLIT — CHRONOLOGICAL ONLY ===

This is TIME SERIES data. You MUST NOT shuffle before splitting.
    - Train: first 70% of data (chronologically)
    - Val:   next 15%
    - Test:  final 15%

Shuffling within the train set for batching is fine AFTER splitting.
Shuffling before splitting = DATA LEAKAGE = fail.

=== TODO (Yoyo & Rohan) ===

Yoyo:
    1. Download UK Grid ESO data (search: "National Grid ESO historic demand data")
    2. Load into pandas, handle missing values (interpolation or forward-fill)
    3. Merge with weather data if available (Met Office or Open-Meteo API)
    4. Save cleaned CSV to data/processed/

Rohan:
    1. Implement feature engineering (cyclical encodings, normalisation)
    2. Implement EnergyDataset(torch.utils.data.Dataset) class
    3. Implement get_dataloaders() function that returns train/val/test loaders
    4. Store normalisation parameters (mean, std) so test.py can de-normalise predictions

=== USAGE EXAMPLE (how other teams will use this) ===

    from src.data_loader import data_preprocess, get_dataloaders

    # Without temperature (7 features):
    train_df, valid_df, test_df, scaler = data_preprocess()

    # With temperature (8 features — fetches from Open-Meteo, cached locally):
    train_df, valid_df, test_df, scaler = data_preprocess(include_temperature=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, valid_df, test_df, seq_len=336, batch_size=64,
    )

    for x_batch, y_batch in train_loader:
        # x_batch: (64, 336, 7 or 8)  — input features
        # y_batch: (64, 1)             — normalised ND
        output = model(x_batch)  # returns dict, e.g. {"prediction": tensor}
"""

import datetime as _dt
import json as _json
import os
import time as _time
from urllib.parse import urlencode as _urlencode
from urllib.request import urlopen as _urlopen

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# --- Constants for Open-Meteo temperature fetching ---
_LONDON_LAT = 51.5074
_LONDON_LON = -0.1278
_TEMP_CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'temperature_london.csv')


# --- UK bank holidays (England), 2001-2025 ---
# Hardcoded so we don't burn a package slot on `holidays`. Generated once with:
#   holidays.country_holidays('GB', subdiv='ENG', years=range(2001, 2026))
# (holidays==0.94). 226 dates including substitute days and one-off royal /
# state events (Royal Wedding 2011, Diamond Jubilee 2012, Platinum Jubilee
# 2022, Queen's funeral 2022, Coronation 2023).
_UK_HOLIDAYS_2001_2025 = frozenset(_dt.date.fromisoformat(d) for d in (
    "2001-01-01", "2001-04-13", "2001-04-16", "2001-05-07", "2001-05-28",
    "2001-08-27", "2001-12-25", "2001-12-26",
    "2002-01-01", "2002-03-29", "2002-04-01", "2002-05-06", "2002-06-03",
    "2002-06-04", "2002-08-26", "2002-12-25", "2002-12-26",
    "2003-01-01", "2003-04-18", "2003-04-21", "2003-05-05", "2003-05-26",
    "2003-08-25", "2003-12-25", "2003-12-26",
    "2004-01-01", "2004-04-09", "2004-04-12", "2004-05-03", "2004-05-31",
    "2004-08-30", "2004-12-25", "2004-12-26", "2004-12-27", "2004-12-28",
    "2005-01-01", "2005-01-03", "2005-03-25", "2005-03-28", "2005-05-02",
    "2005-05-30", "2005-08-29", "2005-12-25", "2005-12-26", "2005-12-27",
    "2006-01-01", "2006-01-02", "2006-04-14", "2006-04-17", "2006-05-01",
    "2006-05-29", "2006-08-28", "2006-12-25", "2006-12-26",
    "2007-01-01", "2007-04-06", "2007-04-09", "2007-05-07", "2007-05-28",
    "2007-08-27", "2007-12-25", "2007-12-26",
    "2008-01-01", "2008-03-21", "2008-03-24", "2008-05-05", "2008-05-26",
    "2008-08-25", "2008-12-25", "2008-12-26",
    "2009-01-01", "2009-04-10", "2009-04-13", "2009-05-04", "2009-05-25",
    "2009-08-31", "2009-12-25", "2009-12-26", "2009-12-28",
    "2010-01-01", "2010-04-02", "2010-04-05", "2010-05-03", "2010-05-31",
    "2010-08-30", "2010-12-25", "2010-12-26", "2010-12-27", "2010-12-28",
    "2011-01-01", "2011-01-03", "2011-04-22", "2011-04-25", "2011-04-29",
    "2011-05-02", "2011-05-30", "2011-08-29", "2011-12-25", "2011-12-26",
    "2011-12-27",
    "2012-01-01", "2012-01-02", "2012-04-06", "2012-04-09", "2012-05-07",
    "2012-06-04", "2012-06-05", "2012-08-27", "2012-12-25", "2012-12-26",
    "2013-01-01", "2013-03-29", "2013-04-01", "2013-05-06", "2013-05-27",
    "2013-08-26", "2013-12-25", "2013-12-26",
    "2014-01-01", "2014-04-18", "2014-04-21", "2014-05-05", "2014-05-26",
    "2014-08-25", "2014-12-25", "2014-12-26",
    "2015-01-01", "2015-04-03", "2015-04-06", "2015-05-04", "2015-05-25",
    "2015-08-31", "2015-12-25", "2015-12-26", "2015-12-28",
    "2016-01-01", "2016-03-25", "2016-03-28", "2016-05-02", "2016-05-30",
    "2016-08-29", "2016-12-25", "2016-12-26", "2016-12-27",
    "2017-01-01", "2017-01-02", "2017-04-14", "2017-04-17", "2017-05-01",
    "2017-05-29", "2017-08-28", "2017-12-25", "2017-12-26",
    "2018-01-01", "2018-03-30", "2018-04-02", "2018-05-07", "2018-05-28",
    "2018-08-27", "2018-12-25", "2018-12-26",
    "2019-01-01", "2019-04-19", "2019-04-22", "2019-05-06", "2019-05-27",
    "2019-08-26", "2019-12-25", "2019-12-26",
    "2020-01-01", "2020-04-10", "2020-04-13", "2020-05-08", "2020-05-25",
    "2020-08-31", "2020-12-25", "2020-12-26", "2020-12-28",
    "2021-01-01", "2021-04-02", "2021-04-05", "2021-05-03", "2021-05-31",
    "2021-08-30", "2021-12-25", "2021-12-26", "2021-12-27", "2021-12-28",
    "2022-01-01", "2022-01-03", "2022-04-15", "2022-04-18", "2022-05-02",
    "2022-06-02", "2022-06-03", "2022-08-29", "2022-09-19", "2022-12-25",
    "2022-12-26", "2022-12-27",
    "2023-01-01", "2023-01-02", "2023-04-07", "2023-04-10", "2023-05-01",
    "2023-05-08", "2023-05-29", "2023-08-28", "2023-12-25", "2023-12-26",
    "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06", "2024-05-27",
    "2024-08-26", "2024-12-25", "2024-12-26",
    "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-05", "2025-05-26",
    "2025-08-25", "2025-12-25", "2025-12-26",
))


def fetch_temperature_data(start_date, end_date):
    """Fetch hourly 2 m temperature from the Open-Meteo Historical Weather API.

    Requests London (51.5074°N, 0.1278°W) data in yearly chunks with a 0.5 s
    delay between requests to respect rate limits.  No API key is needed.
    Uses stdlib ``urllib.request`` so we don't need the ``requests`` package
    (keeps the additional-package count to three).

    Args:
        start_date: First date as 'YYYY-MM-DD'.
        end_date:   Last date as 'YYYY-MM-DD'.

    Returns:
        DataFrame with columns [datetime (tz-naive, Europe/London local),
        temperature (°C)].

    Raises:
        urllib.error.HTTPError: If any API call returns a non-2xx status.
    """
    all_times = []
    all_temps = []

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    for year in range(start_year, end_year + 1):
        chunk_start = start_date if year == start_year else f"{year}-01-01"
        chunk_end = end_date if year == end_year else f"{year}-12-31"

        print(f"  Fetching temperature {chunk_start} -> {chunk_end} ...")
        params = _urlencode({
            "latitude": _LONDON_LAT,
            "longitude": _LONDON_LON,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "hourly": "temperature_2m",
            "timezone": "Europe/London",
        })
        url = f"https://archive-api.open-meteo.com/v1/archive?{params}"
        with _urlopen(url, timeout=30) as resp:
            data = _json.loads(resp.read().decode("utf-8"))

        all_times.extend(data["hourly"]["time"])
        all_temps.extend(data["hourly"]["temperature_2m"])

        if year < end_year:
            _time.sleep(0.5)

    return pd.DataFrame({
        "datetime": pd.to_datetime(all_times),
        "temperature": all_temps,
    })


def _load_or_fetch_temperature(start_year, end_year):
    """Return temperature DataFrame, using a local CSV cache when possible.

    Cache location: ``data/raw/temperature_london.csv``.  If the cache exists
    and covers the requested year range it is loaded directly; otherwise the
    data is fetched via :func:`fetch_temperature_data` and written to disk so
    subsequent calls are instant.

    Args:
        start_year: First year required (inclusive).
        end_year:   Last year required (inclusive).

    Returns:
        DataFrame with columns [datetime, temperature] covering the requested
        range.
    """
    if os.path.exists(_TEMP_CACHE_PATH):
        cached = pd.read_csv(_TEMP_CACHE_PATH, parse_dates=["datetime"])
        cached_min = cached["datetime"].min().year
        cached_max = cached["datetime"].max().year
        if cached_min <= start_year and cached_max >= end_year:
            mask = (
                (cached["datetime"].dt.year >= start_year)
                & (cached["datetime"].dt.year <= end_year)
            )
            return cached[mask].reset_index(drop=True)
        # Cache exists but doesn't cover our range -- widen to cover both
        fetch_start = min(start_year, cached_min)
        fetch_end = max(end_year, cached_max)
    else:
        fetch_start = start_year
        fetch_end = end_year

    print(f"Fetching temperature data for {fetch_start}-{fetch_end} ...")
    df = fetch_temperature_data(f"{fetch_start}-01-01", f"{fetch_end}-12-31")

    os.makedirs(os.path.dirname(_TEMP_CACHE_PATH), exist_ok=True)
    df.to_csv(_TEMP_CACHE_PATH, index=False)
    print(f"Cached temperature data -> {_TEMP_CACHE_PATH}")

    # Filter to requested range
    mask = (
        (df["datetime"].dt.year >= start_year)
        & (df["datetime"].dt.year <= end_year)
    )
    return df[mask].reset_index(drop=True)


def select_csvtopd(start, end, include_temperature=False, include_holidays=False, lag_periods=None):
    """Load UK National Grid ESO demand CSVs and engineer temporal features.

    Loads yearly CSV files from data/processed/data/, extracts the target
    variable (ND = National Demand) and settlement period, then creates:
        - is_weekend: binary flag (1 for Saturday/Sunday, 0 otherwise)
        - hour_sin/cos: cyclical encoding of half-hourly settlement period
        - day_of_week_sin/cos: cyclical encoding of day of week (0=Mon, 6=Sun)
        - day_of_year_sin/cos: cyclical encoding of day of year (seasonal)
        - temperature (optional): hourly London 2 m temp, forward-filled to
          half-hourly resolution so both settlement periods within the same
          clock hour share the same temperature value.

    Raw integer columns (day_of_week, day_of_year) are dropped after encoding
    to avoid implying a false ordinal relationship to the model.

    Args:
        start: First year to load (inclusive).
        end: Last year to load (inclusive).
        include_temperature: If True, fetch / load cached London temperature
            from Open-Meteo and merge into the DataFrame.

    Returns:
        DataFrame with columns: ND, is_weekend, hour_sin, hour_cos,
        day_of_week_sin, day_of_week_cos, day_of_year_sin, day_of_year_cos,
        [temperature].  Temperature column is present only when requested.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '../data/processed/data')
    df = pd.read_csv(os.path.join(data_dir, f"demanddata_{start}.csv"))
    df = df[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"]]
    df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'])

    if start != end:
        for i in range(start+1, end+1):
            df1 = pd.read_csv(os.path.join(data_dir, f"demanddata_{i}.csv"))
            df1 = df1[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"]]
            df1['SETTLEMENT_DATE'] = pd.to_datetime(df1['SETTLEMENT_DATE'])
            df = pd.concat([df, df1], ignore_index=True)

    # --- Temporal indicators (raw integers, used to derive features) ---
    df['day_of_week'] = df['SETTLEMENT_DATE'].dt.dayofweek
    df['day_of_year'] = df['SETTLEMENT_DATE'].dt.dayofyear

    # --- Binary weekend flag (before we drop raw day_of_week) ---
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # --- Cyclical encodings ---
    df['hour_sin'] = np.sin(2 * np.pi * df['SETTLEMENT_PERIOD'] / 48)
    df['hour_cos'] = np.cos(2 * np.pi * df['SETTLEMENT_PERIOD'] / 48)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    # --- Optional: merge hourly temperature (forward-filled to half-hourly) ---
    if include_temperature:
        # Map each settlement period to its clock hour for merging.
        # Period 1-2 -> hour 0, 3-4 -> hour 1, ... , 47-48 -> hour 23.
        df['_hour'] = (df['SETTLEMENT_PERIOD'] - 1) // 2
        df['_datetime_hour'] = (
            df['SETTLEMENT_DATE'] + pd.to_timedelta(df['_hour'], unit='h')
        )

        temp_df = _load_or_fetch_temperature(start, end)
        # Floor to hour in case of sub-hour offsets, then deduplicate
        temp_df['_datetime_hour'] = temp_df['datetime'].dt.floor('h')
        temp_df = temp_df[['_datetime_hour', 'temperature']].drop_duplicates(
            subset='_datetime_hour'
        )

        df = df.merge(temp_df, on='_datetime_hour', how='left')

        # Fill any remaining gaps (DST transitions, edge-of-range mismatches)
        df['temperature'] = df['temperature'].ffill().bfill()

        df = df.drop(columns=['_hour', '_datetime_hour'])

    # --- Optional: explicit lag demand features ---
    if lag_periods:
        for n in lag_periods:
            col = f"demand_lag_{n}"
            df[col] = df["ND"].shift(n)
            df[col] = df[col].bfill()

    # --- Optional: UK bank holiday indicator ---
    if include_holidays:
        df['is_holiday'] = df['SETTLEMENT_DATE'].dt.date.isin(
            _UK_HOLIDAYS_2001_2025
        ).astype(int)

    # Drop raw date/period columns AND raw integer indicators (day_of_week, day_of_year)
    # to avoid leaking ordinal relationships into the model.
    return df.drop(columns=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "day_of_week", "day_of_year"])


def data_preprocess(train_start=2001, train_end=2017, valid_start=2018, valid_end=2022,
                    test_start=2023, test_end=2025, include_temperature=False,
                    include_holidays=False, lag_periods=None):
    """Load, engineer features, and normalise train/validation/test splits.

    Normalisation strategy (no data leakage):
        - Z-score parameters (mean, std) are computed from the TRAINING set only.
        - The same parameters are applied to validation and test sets.
        - Cyclical sin/cos features are already in [-1, 1] and need no scaling.
        - is_weekend is binary {0, 1} and needs no scaling.
        - Temperature (when included) is z-score normalised using train stats.

    Args:
        train_start: First year of training data (inclusive).
        train_end:   Last year of training data (inclusive).
        valid_start: First year of validation data (inclusive).
        valid_end:   Last year of validation data (inclusive).
        test_start:  First year of test data (inclusive).
        test_end:    Last year of test data (inclusive).
        include_temperature: If True, fetch/load London temperature from
            Open-Meteo and include as an additional normalised feature.

    Returns:
        (train_df, valid_df, test_df, scaler_params) where scaler_params is
        {"mean": float, "std": float} computed from the training set's ND
        column.  When temperature is included, also contains "temp_mean" and
        "temp_std".
    """
    # Pre-fetch temperature for the full year range so all three splits share
    # one API call / cache hit instead of three separate fetches.
    if include_temperature:
        _load_or_fetch_temperature(train_start, test_end)

    train_df = select_csvtopd(train_start, train_end, include_temperature, include_holidays, lag_periods)
    valid_df = select_csvtopd(valid_start, valid_end, include_temperature, include_holidays, lag_periods)
    test_df  = select_csvtopd(test_start,  test_end,  include_temperature, include_holidays, lag_periods)

    # --- Z-score normalisation of ND (target), fit on train only ---
    nd_mean = float(train_df["ND"].mean())
    nd_std = float(train_df["ND"].std())

    train_df["ND"] = (train_df["ND"] - nd_mean) / nd_std
    valid_df["ND"] = (valid_df["ND"] - nd_mean) / nd_std
    test_df["ND"]  = (test_df["ND"] - nd_mean) / nd_std

    scaler_params = {"mean": nd_mean, "std": nd_std}

    # --- Z-score normalisation of temperature, fit on train only ---
    if include_temperature:
        temp_mean = float(train_df["temperature"].mean())
        temp_std = float(train_df["temperature"].std())

        train_df["temperature"] = (train_df["temperature"] - temp_mean) / temp_std
        valid_df["temperature"] = (valid_df["temperature"] - temp_mean) / temp_std
        test_df["temperature"]  = (test_df["temperature"] - temp_mean) / temp_std

        scaler_params["temp_mean"] = temp_mean
        scaler_params["temp_std"] = temp_std

    # --- Z-score normalisation of lag columns, fit on train only ---
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

    # --- Verify no NaN values after all transformations ---
    assert not train_df.isna().any().any(), f"NaN found in train_df: {train_df.isna().sum()}"
    assert not valid_df.isna().any().any(), f"NaN found in valid_df: {valid_df.isna().sum()}"
    assert not test_df.isna().any().any(),  f"NaN found in test_df: {test_df.isna().sum()}"

    return train_df, valid_df, test_df, scaler_params

    


    

class EnergyDataset(Dataset):
    """Sliding-window dataset for energy demand forecasting.

    Takes a preprocessed and normalised DataFrame, separates features from the
    target column (ND), and produces sliding windows for time-series prediction.

    For window starting at index i:
        x = features at timesteps [i : i + seq_len]   -> shape (seq_len, num_features)
        y = normalised ND at timestep [i + seq_len]    -> shape (1,)

    Args:
        df: DataFrame with columns [ND, is_weekend, hour_sin, hour_cos, ...].
            Must already be normalised (ND z-scored, no raw integer columns).
        seq_len: Number of past timesteps per input window.
            336 = one week of half-hourly data (48 periods/day × 7 days).
    """

    def __init__(self, df: pd.DataFrame, seq_len: int = 336):
        feature_cols = [c for c in df.columns if c != "ND"]
        self.features = torch.tensor(df[feature_cols].to_numpy(), dtype=torch.float32)
        self.targets = torch.tensor(df["ND"].to_numpy(), dtype=torch.float32)
        self.seq_len = seq_len
        self.num_features = len(feature_cols)

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]          # (seq_len, num_features)
        y = self.targets[idx + self.seq_len].unsqueeze(0)     # (1,)
        return x, y

def get_dataloaders(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame,
                    seq_len: int = 336, batch_size: int = 64):
    """Wrap preprocessed DataFrames in sliding-window DataLoaders.

    Each DataFrame must already be normalised (call data_preprocess() first).
    Shuffling is enabled for the training loader only — validation and test
    loaders preserve chronological order for evaluation and plotting.

    Args:
        train_df: Training split DataFrame (normalised).
        valid_df: Validation split DataFrame (normalised).
        test_df:  Test split DataFrame (normalised).
        seq_len:  Sliding window length (default 336 = 1 week of half-hourly data).
        batch_size: Batch size for all loaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = EnergyDataset(train_df, seq_len)
    valid_dataset = EnergyDataset(valid_df, seq_len)
    test_dataset  = EnergyDataset(test_df,  seq_len)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader



if __name__ == "__main__":
    import sys

    use_temp = "--temperature" in sys.argv

    # 1. Preprocess: load CSVs, engineer features, normalise
    print(f"=== Loading data (include_temperature={use_temp}) ===")
    train_df, valid_df, test_df, scaler_params = data_preprocess(
        include_temperature=use_temp
    )
    print("Columns:", list(train_df.columns))
    print(f"Split sizes: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    print(f"Scaler — ND: mean={scaler_params['mean']:.2f}, std={scaler_params['std']:.2f}")
    if use_temp:
        print(f"Scaler — temp: mean={scaler_params['temp_mean']:.2f}, "
              f"std={scaler_params['temp_std']:.2f}")

    # 2. Create DataLoaders with sliding windows
    train_loader, valid_loader, test_loader = get_dataloaders(train_df, valid_df, test_df)
    x, y = next(iter(train_loader))

    num_features = x.shape[2]
    print(f"\nBatch shapes: x={x.shape}, y={y.shape}")
    print(f"num_features (LSTM input_size): {num_features}")


