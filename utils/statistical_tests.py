"""
utils/statistical_tests.py
============================
Two hypothesis tests needed for Section 4.5 of the manuscript:

1. Diebold-Mariano (1995) test, with the Harvey-Leybourne-Newey (1997)
   small-sample correction, for comparing the forecast accuracy of two
   models from their loss-differential series.

2. Stationary bootstrap (Politis & Romano, 1994) confidence interval and
   p-value for the difference in Sharpe ratio between two daily return
   series -- avoids the i.i.d. assumption a naive bootstrap would make,
   important since daily financial returns are autocorrelated/heteroskedastic.

Both are implemented from their defining equations (no specialized
forecasting-evaluation package dependency) so they can be audited alongside
the rest of the custom-coded methodology.
"""

from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from scipy import stats

from utils.metrics import sharpe_ratio


# --------------------------------------------------------------------------
# 1. Diebold-Mariano test
# --------------------------------------------------------------------------
def diebold_mariano_test(errors_a: np.ndarray, errors_b: np.ndarray,
                          h: int = 1, loss: str = "squared", power: int = 2) -> dict:
    """
    errors_a, errors_b: aligned 1-D arrays of forecast errors (y_true - y_pred),
        one value per forecast origin, for model A and model B respectively
        (e.g., per-origin mean error over the 5-step horizon).
    h: forecast horizon used when the errors were generated (for the
        Newey-West-style lag truncation in the long-run variance estimator).
    loss: "squared" -> L(e) = e^2 ;  "absolute" -> L(e) = |e|.

    Returns a dict with the DM statistic (HLN small-sample corrected),
    two-sided p-value, and which model the test favors.
    Null hypothesis: the two models have equal predictive accuracy.
    Negative DM statistic favors model A (lower average loss); positive favors B.
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    assert len(errors_a) == len(errors_b), "error series must be aligned (same forecast origins)"
    n = len(errors_a)

    if loss == "squared":
        loss_a, loss_b = errors_a ** 2, errors_b ** 2
    elif loss == "absolute":
        loss_a, loss_b = np.abs(errors_a), np.abs(errors_b)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    d = loss_a - loss_b                      # loss differential series
    d_bar = np.mean(d)

    # Newey-West long-run variance of the mean, truncated at h-1 lags
    # (standard choice for an h-step-ahead forecast's error autocorrelation)
    gamma0 = np.var(d, ddof=0)
    lrv = gamma0
    for lag in range(1, h):
        w = 1.0 - lag / h  # Bartlett kernel weight
        cov = np.cov(d[lag:], d[:-lag])[0, 1] if n > lag else 0.0
        lrv += 2 * w * cov
    lrv = max(lrv, 1e-12)

    dm_stat = d_bar / np.sqrt(lrv / n)

    # Harvey-Leybourne-Newey (1997) small-sample correction
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * hln_factor

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=n - 1))
    favors = "A" if dm_stat_corrected < 0 else "B"

    return {
        "n_obs": n, "mean_loss_a": float(np.mean(loss_a)), "mean_loss_b": float(np.mean(loss_b)),
        "dm_statistic": float(dm_stat_corrected), "p_value": float(p_value),
        "favors": favors, "significant_5pct": bool(p_value < 0.05),
    }


# --------------------------------------------------------------------------
# 2. Stationary bootstrap Sharpe-ratio-difference test
# --------------------------------------------------------------------------
def _stationary_bootstrap_indices(n: int, mean_block_len: float, rng: np.random.Generator) -> np.ndarray:
    """One resample of indices [0..n) via Politis & Romano's (1994) stationary
    bootstrap: geometric block lengths, wrapping circularly."""
    p = 1.0 / mean_block_len
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)          # start a new block
        else:
            idx[t] = (idx[t - 1] + 1) % n          # continue the current block
    return idx


def bootstrap_sharpe_difference(returns_a: np.ndarray, returns_b: np.ndarray,
                                 risk_free_annual: float = 0.0, periods_per_year: int = 252,
                                 n_boot: int = 1000, mean_block_len: float = 10.0,
                                 seed: Optional[int] = 42) -> dict:
    """
    returns_a, returns_b: aligned daily return series for strategy A and B
        (same dates -- e.g., DandelionOptimizer vs. Markowitz).
    Returns the observed Sharpe(A) - Sharpe(B), a stationary-bootstrap 95% CI
    for that difference, and a two-sided bootstrap p-value for the null that
    the true difference is zero.
    """
    returns_a = np.asarray(returns_a, dtype=float)
    returns_b = np.asarray(returns_b, dtype=float)
    assert len(returns_a) == len(returns_b), "return series must be aligned (same dates)"
    n = len(returns_a)
    rng = np.random.default_rng(seed)

    observed_diff = (sharpe_ratio(returns_a, risk_free_annual, periods_per_year)
                      - sharpe_ratio(returns_b, risk_free_annual, periods_per_year))

    boot_diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(n, mean_block_len, rng)
        sa = sharpe_ratio(returns_a[idx], risk_free_annual, periods_per_year)
        sb = sharpe_ratio(returns_b[idx], risk_free_annual, periods_per_year)
        boot_diffs[b] = sa - sb

    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])
    # Standard percentile-method two-sided bootstrap p-value: the empirical
    # bootstrap distribution IS the (approximate) sampling distribution of
    # the estimator, so the p-value is twice the smaller tail probability of
    # zero within that distribution. This is guaranteed consistent with the
    # 95% CI test above (CI excludes 0  <=>  p < 0.05), unlike a naive
    # "re-center on the bootstrap mean" construction.
    p_lower = np.mean(boot_diffs <= 0.0)
    p_upper = np.mean(boot_diffs >= 0.0)
    p_value = float(2 * min(p_lower, p_upper))
    p_value = min(p_value, 1.0)

    return {
        "n_obs": n, "n_boot": n_boot,
        "sharpe_a": float(sharpe_ratio(returns_a, risk_free_annual, periods_per_year)),
        "sharpe_b": float(sharpe_ratio(returns_b, risk_free_annual, periods_per_year)),
        "observed_diff": float(observed_diff),
        "ci_95_lower": float(ci_lower), "ci_95_upper": float(ci_upper),
        "p_value": p_value, "significant_5pct": bool(p_value < 0.05),
    }
