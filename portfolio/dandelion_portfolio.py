"""
portfolio/dandelion_portfolio.py
=================================
Phase 2 core: uses the Dandelion Optimizer (optimizers/dandelion.py) to
solve a constrained multi-objective extension of the Markowitz problem:

    maximize   w_sharpe * Sharpe(w) + w_sortino * Sortino(w)
    minimize   w_mdd * MDD(w) + w_var * AnnualizedVariance(w)
    subject to sum(w) = 1,  0 <= w_i <= 1  (no short-selling)

Because DO is an unconstrained box-optimizer, the budget constraint is
enforced via a standard "repair" (projection) operator applied at fitness
-evaluation time: negative weights are clipped to zero and the vector is
renormalized to sum to 1. This is the conventional approach for applying
population-based metaheuristics to the simplex-constrained portfolio
problem (weights are optimized directly, not returns/covariance inputs).
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

from optimizers.dandelion import dandelion_optimizer
from optimizers.base import OptimizationResult
from utils.metrics import sharpe_ratio, sortino_ratio, max_drawdown


def repair_weights(w: np.ndarray) -> np.ndarray:
    """Projects an arbitrary real vector onto the simplex {w : sum(w)=1, w>=0}."""
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total < 1e-10:
        return np.full_like(w, 1.0 / len(w))
    return w / total


def make_portfolio_fitness(asset_returns: pd.DataFrame, risk_free_annual: float,
                            objective_weights: dict, periods_per_year: int = 252):
    """Returns a `fitness(w) -> float` (minimization) closure over a fixed
    historical return window. AnnualizedVariance is used (rather than raw
    daily variance) purely so all four objective terms live on comparable
    O(0.01-1) scales before weighted summation -- the *reported* metric
    tables still show raw MDD, Sharpe, Sortino, and daily-variance exactly.
    """
    cov_matrix = asset_returns.cov().values * periods_per_year   # annualized covariance

    def fitness(raw_w: np.ndarray) -> float:
        w = repair_weights(raw_w)
        daily_rets = asset_returns.values @ w
        sharpe = sharpe_ratio(daily_rets, risk_free_annual, periods_per_year)
        sortino = sortino_ratio(daily_rets, risk_free_annual, periods_per_year)
        mdd = max_drawdown(daily_rets)
        ann_var = float(w @ cov_matrix @ w)

        reward = objective_weights["sharpe"] * sharpe + objective_weights["sortino"] * sortino
        penalty = objective_weights["mdd"] * mdd + objective_weights["variance"] * ann_var
        return -reward + penalty     # minimize: -(...) makes maximizing Sharpe/Sortino equivalent to minimizing

    return fitness


def optimize_portfolio_dandelion(asset_returns: pd.DataFrame, risk_free_annual: float,
                                  objective_weights: dict, n_pop: int = 40, max_iter: int = 150,
                                  seed: Optional[int] = 42, periods_per_year: int = 252,
                                  verbose: bool = False) -> tuple[np.ndarray, OptimizationResult]:
    """Runs the Dandelion Optimizer over the simplex-constrained weight
    space and returns (final_repaired_weights, raw_optimizer_result)."""
    n_assets = asset_returns.shape[1]
    fitness_fn = make_portfolio_fitness(asset_returns, risk_free_annual, objective_weights, periods_per_year)

    result = dandelion_optimizer(fitness_fn, dim=n_assets, lb=0.0, ub=1.0,
                                  n_pop=n_pop, max_iter=max_iter, seed=seed, verbose=verbose)
    final_weights = repair_weights(result.best_solution)
    return final_weights, result
