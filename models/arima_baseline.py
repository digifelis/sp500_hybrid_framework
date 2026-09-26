"""
models/arima_baseline.py
=========================
Classical ARIMA benchmark (statsmodels), used as the "traditional model"
comparator against the LSTM/GRU family in the forecasting benchmark table.

A lightweight AIC-based grid search selects (p, d, q) from a small default
grid -- deliberately compact so that the rolling-window backtest (which
refits ARIMA at every roll, for every ticker) remains tractable. For a
production SCI run, widen ARIMA_ORDER_GRID and/or swap in pmdarima's
auto_arima if desired.
"""

from __future__ import annotations
import warnings
from typing import Optional, Tuple
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

ARIMA_ORDER_GRID = [(p, d, q) for p in range(0, 4) for d in (0, 1) for q in range(0, 3)]


def select_best_order(series: np.ndarray, order_grid=ARIMA_ORDER_GRID) -> Tuple[int, int, int]:
    best_aic, best_order = np.inf, (1, 1, 0)
    for order in order_grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(series, order=order).fit()
            if fit.aic < best_aic:
                best_aic, best_order = fit.aic, order
        except Exception:
            continue
    return best_order


def fit_predict_arima(train_series: np.ndarray, horizon: int,
                       order: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """Fits ARIMA on `train_series` (1-D array of raw or normalized close
    prices) and returns an `horizon`-step-ahead point forecast."""
    if order is None:
        order = select_best_order(train_series)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = ARIMA(train_series, order=order).fit()
            forecast = fit.forecast(steps=horizon)
        except Exception:
            # Robust fallback: naive last-value persistence if ARIMA fails
            # to converge on a pathological window (e.g. near-constant
            # series during warm-up), which keeps the rolling backtest alive.
            forecast = np.full(horizon, train_series[-1])
    return np.asarray(forecast, dtype=float)
