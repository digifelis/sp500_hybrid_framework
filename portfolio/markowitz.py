"""
portfolio/markowitz.py
=======================
Classical (textbook) Markowitz mean-variance portfolio optimization,
solved with SLSQP (scipy.optimize) subject to the budget constraint
(sum of weights = 1) and no-short-selling constraint (0 <= w <= 1).

Serves as the "standard Markowitz Mean-Variance model" benchmark against
the Dandelion-Optimizer-driven multi-objective portfolio of
`portfolio/dandelion_portfolio.py`.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def markowitz_max_sharpe(mean_daily_returns: np.ndarray, cov_matrix: np.ndarray,
                          risk_free_annual: float = 0.0, periods_per_year: int = 252) -> np.ndarray:
    """Tangency portfolio: maximize the (annualized) Sharpe ratio subject to
    sum(w)=1, 0<=w<=1."""
    n = len(mean_daily_returns)

    def neg_sharpe(w):
        port_return = float(np.dot(w, mean_daily_returns) * periods_per_year)
        port_vol = float(np.sqrt(max(w @ cov_matrix @ w, 0.0)) * np.sqrt(periods_per_year))
        if port_vol < 1e-10:
            return 1e6
        return -(port_return - risk_free_annual) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.full(n, 1.0 / n)
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                    constraints=constraints, options={"maxiter": 1000, "ftol": 1e-10})
    w = np.clip(res.x, 0.0, 1.0)
    w = w / w.sum() if w.sum() > 1e-10 else np.full(n, 1.0 / n)
    return w


def markowitz_min_variance(cov_matrix: np.ndarray,
                            target_return: float = None,
                            mean_daily_returns: np.ndarray = None) -> np.ndarray:
    """Global minimum-variance portfolio (optionally on the efficient
    frontier at a specified target return)."""
    n = cov_matrix.shape[0]

    def variance(w):
        return float(w @ cov_matrix @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        if mean_daily_returns is None:
            raise ValueError("mean_daily_returns required when target_return is set")
        constraints.append({"type": "eq", "fun": lambda w: np.dot(w, mean_daily_returns) - target_return})

    bounds = [(0.0, 1.0)] * n
    w0 = np.full(n, 1.0 / n)
    res = minimize(variance, w0, method="SLSQP", bounds=bounds,
                    constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, 1.0)
    w = w / w.sum() if w.sum() > 1e-10 else np.full(n, 1.0 / n)
    return w


def equal_weight_portfolio(n_assets: int) -> np.ndarray:
    """The 1/N naive-diversification benchmark."""
    return np.full(n_assets, 1.0 / n_assets)
