"""
data/data_pipeline.py
======================
Data acquisition and preparation layer.

IMPORTANT — DATA PROVENANCE NOTE FOR THE RESEARCHER
----------------------------------------------------
This execution environment has no network access to financial data
vendors (Borsa Istanbul Veri Yayin, Yahoo/Investing.com, Matriks, Finnet,
Refinitiv, etc.), so real BIST tick data cannot be downloaded here.

Two data paths are therefore provided:

1. `load_real_universe(csv_dir, tickers)`  — the path you will use for the
   actual SCI submission. Point it at a directory containing one CSV per
   ticker (columns: Date, Open, High, Low, Close, Volume — the standard
   export format from yfinance / Matriks / Borsa Istanbul Veri Yayin).

2. `generate_synthetic_universe(...)` — a statistically-calibrated
   surrogate (single-factor GBM with regime-switching volatility and a
   realistic cross-asset correlation structure) used ONLY so that the
   rest of the pipeline (features -> LSTM -> optimizers -> portfolio ->
   backtest) is runnable and testable end-to-end without real data. It is
   NOT a substitute for real BIST prices in the final study — replace it
   with (1) before reporting results.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Real-data loader
# --------------------------------------------------------------------------
def load_real_universe(csv_dir: str, tickers: List[str],
                        start: Optional[str] = None, end: Optional[str] = None
                        ) -> Dict[str, pd.DataFrame]:
    """Load one CSV per ticker from `csv_dir/{ticker}.csv`.
    Expected columns (case-insensitive): Date, Open, High, Low, Close, Volume.
    """
    universe = {}
    for tkr in tickers:
        path = os.path.join(csv_dir, f"{tkr}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected real price data at {path}. "
                f"Export daily OHLCV for {tkr} (e.g. via yfinance '{tkr}.IS', "
                f"Borsa Istanbul Veri Yayin, or Matriks) and place it there."
            )
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.rename(columns={"adj close": "close"})
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        if start:
            df = df.loc[df.index >= pd.Timestamp(start)]
        if end:
            df = df.loc[df.index <= pd.Timestamp(end)]
        universe[tkr] = df
    return universe


def load_real_index(csv_dir: str, index_ticker: str = "XU100",
                    start: Optional[str] = None, end: Optional[str] = None) -> pd.Series:
    """Load a benchmark index level series from `csv_dir/{index_ticker}.csv`."""
    path = os.path.join(csv_dir, f"{index_ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected index data at {path}.")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    series = df["close"].astype(float)
    if start:
        series = series.loc[series.index >= pd.Timestamp(start)]
    if end:
        series = series.loc[series.index <= pd.Timestamp(end)]
    return series


# --------------------------------------------------------------------------
# 2. Calibrated synthetic universe (development / CI surrogate)
# --------------------------------------------------------------------------
def _regime_volatility_path(n_days: int, rng: np.random.Generator,
                             calm_vol: float = 0.012, volatile_vol: float = 0.032,
                             p_calm_to_volatile: float = 0.004,
                             p_volatile_to_calm: float = 0.02) -> np.ndarray:
    """Two-state Markov regime switch (calm vs. volatile daily vol), used to
    mimic BIST100's historically observed volatility clustering, including
    stress regimes such as 2025-2026."""
    state = 0  # 0 = calm, 1 = volatile
    vols = np.empty(n_days)
    for t in range(n_days):
        vols[t] = calm_vol if state == 0 else volatile_vol
        if state == 0 and rng.random() < p_calm_to_volatile:
            state = 1
        elif state == 1 and rng.random() < p_volatile_to_calm:
            state = 0
    return vols


def generate_synthetic_universe(tickers: List[str], start: str, end: str,
                                 seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Single-factor GBM: r_i,t = beta_i * market_t + idiosyncratic_i,t.
    This yields a non-trivial, positive-semi-definite covariance structure
    (essential for Markowitz / Dandelion portfolio optimization to produce
    meaningful diversification), plus per-ticker regime-switching volatility
    so recent-period backtests exhibit realistic stress episodes.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)
    n_assets = len(tickers)

    market_vol = _regime_volatility_path(n_days, rng, calm_vol=0.010, volatile_vol=0.026)
    market_drift = 0.0003
    market_return = rng.normal(market_drift, 1.0, n_days) * market_vol

    betas = rng.uniform(0.5, 1.6, n_assets)
    idio_vol_scale = rng.uniform(0.7, 1.4, n_assets)
    annual_drift = rng.uniform(0.05, 0.30, n_assets) / TRADING_DAYS if (TRADING_DAYS := 252) else None
    start_prices = rng.uniform(8.0, 220.0, n_assets)          # plausible BIST TRY price range
    base_volume = rng.uniform(3e6, 6e7, n_assets)              # plausible daily share volume

    universe = {}
    for i, tkr in enumerate(tickers):
        idio_vol_path = _regime_volatility_path(n_days, rng, calm_vol=0.009, volatile_vol=0.022) * idio_vol_scale[i]
        idio = rng.normal(0.0, 1.0, n_days) * idio_vol_path
        daily_returns = annual_drift[i] + betas[i] * market_return + idio
        # occasional idiosyncratic jump risk (earnings surprises, corporate actions)
        jump_mask = rng.random(n_days) < 0.006
        daily_returns[jump_mask] += rng.normal(0.0, 0.05, jump_mask.sum())

        close = start_prices[i] * np.cumprod(1.0 + daily_returns)
        daily_range = np.abs(rng.normal(0.0, 1.0, n_days)) * idio_vol_path + 0.003
        open_ = close / (1.0 + rng.normal(0.0, 1.0, n_days) * idio_vol_path * 0.4)
        high = np.maximum(open_, close) * (1.0 + daily_range * rng.uniform(0.2, 1.0, n_days))
        low = np.minimum(open_, close) * (1.0 - daily_range * rng.uniform(0.2, 1.0, n_days))
        volume = base_volume[i] * (1.0 + 4.0 * np.abs(daily_returns) / (idio_vol_path + 1e-6)) \
            * np.exp(rng.normal(0.0, 0.25, n_days))

        df = pd.DataFrame({
            "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        }, index=dates)
        df.index.name = "date"
        universe[tkr] = df

    return universe


def generate_synthetic_index(universe: Dict[str, pd.DataFrame], seed: int = 42) -> pd.Series:
    """Builds a capitalization-agnostic equal-weight proxy for the BIST100
    buy-and-hold benchmark from the synthetic universe (simple average of
    constituent daily returns, re-based to 100)."""
    rets = pd.DataFrame({t: df["close"].pct_change() for t, df in universe.items()}).dropna()
    index_ret = rets.mean(axis=1)
    index_level = 100.0 * (1.0 + index_ret).cumprod()
    return index_level


# --------------------------------------------------------------------------
# 3. Supervised sequence-dataset construction (for the LSTM/GRU forecasters)
# --------------------------------------------------------------------------
def build_sequences(feature_df: pd.DataFrame, feature_cols: List[str], target_col: str,
                     seq_len: int, horizon: int):
    """Sliding-window transform: X[i] = feature_cols over [i, i+seq_len),
    y[i] = target_col over [i+seq_len, i+seq_len+horizon) (multi-step ahead).
    Returns (X, y, index_of_forecast_origin) as numpy arrays.
    """
    values = feature_df[feature_cols].values.astype(np.float32)
    target = feature_df[target_col].values.astype(np.float32)
    n = len(feature_df)
    X, y, origin_idx = [], [], []
    last_start = n - seq_len - horizon + 1
    for i in range(max(last_start, 0)):
        X.append(values[i: i + seq_len])
        y.append(target[i + seq_len: i + seq_len + horizon])
        origin_idx.append(i + seq_len - 1)
    return np.asarray(X), np.asarray(y), np.asarray(origin_idx)


def chronological_split(n_samples: int, train_frac: float, val_frac: float):
    """Index ranges for a strictly chronological train/val/test split
    (no shuffling — mandatory for time-series to avoid look-ahead bias)."""
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_samples)
    return train_idx, val_idx, test_idx
