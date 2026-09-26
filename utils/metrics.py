"""
utils/metrics.py
=================
All evaluation metrics used across the study:

Forecasting accuracy : RMSE, MAPE, MAE, directional accuracy
Portfolio / trading   : cumulative return, annualized return & volatility,
                         Sharpe ratio, Sortino ratio, Maximum Drawdown (MDD),
                         Calmar ratio, portfolio variance

Kept dependency-free (numpy/pandas only) so it can be imported by the
optimizers' fitness functions without pulling in heavy modules.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-8


# --------------------------------------------------------------------------
# Forecasting-accuracy metrics (Phase 1)
# --------------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, expressed in percent. Points where
    |y_true| < EPS are excluded to avoid division blow-ups (irrelevant for
    normalized price levels, which are always > 0)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > EPS
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of steps where the predicted trend direction (up/down)
    matches the realized direction. Useful complement to RMSE/MAPE for
    a *trend*-forecasting study."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float("nan")
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir) * 100.0)


# --------------------------------------------------------------------------
# Return-series helpers (Phase 2)
# --------------------------------------------------------------------------
def portfolio_daily_returns(asset_returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """asset_returns: (T, N) simple daily returns. weights: (N,) summing to 1."""
    weights = np.asarray(weights, dtype=float)
    return (asset_returns.values @ weights)


def cumulative_return(daily_returns: np.ndarray) -> float:
    daily_returns = np.asarray(daily_returns, dtype=float)
    return float(np.prod(1.0 + daily_returns) - 1.0)


def annualized_return(daily_returns: np.ndarray, periods_per_year: int = 252) -> float:
    daily_returns = np.asarray(daily_returns, dtype=float)
    n = len(daily_returns)
    if n == 0:
        return 0.0
    growth = np.prod(1.0 + daily_returns)
    return float(growth ** (periods_per_year / n) - 1.0)


def annualized_volatility(daily_returns: np.ndarray, periods_per_year: int = 252) -> float:
    daily_returns = np.asarray(daily_returns, dtype=float)
    return float(np.std(daily_returns, ddof=1) * np.sqrt(periods_per_year)) if len(daily_returns) > 1 else 0.0


def sharpe_ratio(daily_returns: np.ndarray, risk_free_annual: float = 0.0,
                  periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio using arithmetic daily excess returns."""
    daily_returns = np.asarray(daily_returns, dtype=float)
    if len(daily_returns) < 2:
        return 0.0
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    excess = daily_returns - rf_daily
    sigma = np.std(excess, ddof=1)
    if sigma < EPS:
        return 0.0
    return float(np.mean(excess) / sigma * np.sqrt(periods_per_year))


def sortino_ratio(daily_returns: np.ndarray, risk_free_annual: float = 0.0,
                   periods_per_year: int = 252, target: float = 0.0) -> float:
    """Annualized Sortino ratio: excess return over the *downside deviation*
    (only returns falling below `target`, default 0, penalize risk)."""
    daily_returns = np.asarray(daily_returns, dtype=float)
    if len(daily_returns) < 2:
        return 0.0
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    excess = daily_returns - rf_daily
    downside = np.minimum(daily_returns - target, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    if downside_dev < EPS:
        return 0.0
    return float(np.mean(excess) / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(daily_returns: np.ndarray) -> float:
    """Maximum Drawdown as a POSITIVE fraction (e.g. 0.23 == -23% peak-to-trough)."""
    daily_returns = np.asarray(daily_returns, dtype=float)
    if len(daily_returns) == 0:
        return 0.0
    wealth = np.cumprod(1.0 + daily_returns)
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth - running_max) / running_max
    return float(-np.min(drawdown))


def calmar_ratio(daily_returns: np.ndarray, periods_per_year: int = 252) -> float:
    mdd = max_drawdown(daily_returns)
    if mdd < EPS:
        return 0.0
    return float(annualized_return(daily_returns, periods_per_year) / mdd)


def portfolio_variance(cov_matrix: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(weights @ cov_matrix @ weights)


def summarize_performance(daily_returns: np.ndarray, risk_free_annual: float = 0.0,
                           periods_per_year: int = 252) -> dict:
    """One-call summary table used throughout the benchmarking sections."""
    daily_returns = np.asarray(daily_returns, dtype=float)
    return {
        "CumulativeReturn": cumulative_return(daily_returns),
        "AnnualizedReturn": annualized_return(daily_returns, periods_per_year),
        "AnnualizedVolatility": annualized_volatility(daily_returns, periods_per_year),
        "SharpeRatio": sharpe_ratio(daily_returns, risk_free_annual, periods_per_year),
        "SortinoRatio": sortino_ratio(daily_returns, risk_free_annual, periods_per_year),
        "MaxDrawdown": max_drawdown(daily_returns),
        "CalmarRatio": calmar_ratio(daily_returns, periods_per_year),
    }
