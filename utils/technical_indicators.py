"""
utils/technical_indicators.py
==============================
Self-contained (no TA-Lib dependency, for portability) implementations of
the technical indicators requested in the study design: RSI, MACD, and
Bollinger Bands, plus min-max normalization utilities.

All functions operate on a pandas Series/DataFrame indexed by trading date
and are safe to apply per-ticker via groupby.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing == EMA with alpha = 1/window
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)  # neutral RSI during warm-up


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Returns (mid, upper, lower, %B, bandwidth)."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    pctb = (close - lower) / (upper - lower).replace(0.0, np.nan)
    bandwidth = (upper - lower) / mid.replace(0.0, np.nan)
    return mid, upper, lower, pctb.fillna(0.5), bandwidth.fillna(0.0)


def min_max_normalize(series: pd.Series, feature_range=(0.0, 1.0)):
    """Returns (normalized_series, min_value, max_value) so the transform
    can be exactly inverted later (needed to turn LSTM outputs back into
    price levels)."""
    lo, hi = feature_range
    smin, smax = series.min(), series.max()
    denom = (smax - smin) if (smax - smin) > 1e-12 else 1.0
    norm = lo + (series - smin) * (hi - lo) / denom
    return norm, smin, smax


def inverse_min_max(norm_values: np.ndarray, smin: float, smax: float, feature_range=(0.0, 1.0)):
    lo, hi = feature_range
    return smin + (np.asarray(norm_values, dtype=float) - lo) * (smax - smin) / (hi - lo)


def build_feature_frame(ohlcv: pd.DataFrame, cfg) -> pd.DataFrame:
    """Given a single ticker's OHLCV frame (columns: open, high, low, close,
    volume; DatetimeIndex), engineer the full technical-indicator feature
    set described in the study design and min-max normalize each column to
    [0, 1]. Returns a DataFrame ready for sequence windowing.

    `cfg` is the config module (or any object exposing RSI_WINDOW,
    MACD_FAST/SLOW/SIGNAL, BB_WINDOW/NUM_STD).
    """
    df = ohlcv.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["rsi"] = rsi(df["close"], cfg.RSI_WINDOW)
    macd_line, macd_signal, macd_hist = macd(df["close"], cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd_line, macd_signal, macd_hist
    _, _, _, pctb, bandwidth = bollinger_bands(df["close"], cfg.BB_WINDOW, cfg.BB_NUM_STD)
    df["bb_pctb"], df["bb_bandwidth"] = pctb, bandwidth

    # Price & volume normalized separately; oscillators (RSI, %B) are
    # already bounded and only lightly rescaled; MACD/bandwidth are
    # min-max scaled since they are unbounded.
    df["close_norm"], close_min, close_max = min_max_normalize(df["close"])
    df["volume_norm"], _, _ = min_max_normalize(df["volume"].astype(float))
    df["rsi"] = df["rsi"] / 100.0
    df["macd"], _, _ = min_max_normalize(df["macd"].fillna(0.0))
    df["macd_signal"], _, _ = min_max_normalize(df["macd_signal"].fillna(0.0))
    df["macd_hist"], _, _ = min_max_normalize(df["macd_hist"].fillna(0.0))
    df["bb_bandwidth"], _, _ = min_max_normalize(df["bb_bandwidth"].fillna(0.0))
    df["log_return"], _, _ = min_max_normalize(df["log_return"].fillna(0.0))

    df.attrs["close_min"] = close_min
    df.attrs["close_max"] = close_max
    df = df.dropna().copy()
    return df
