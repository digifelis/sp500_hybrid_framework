"""
optimizers/rime.py
===================
RIME: A physics-based optimization algorithm.

Reference
---------
Su, H., Zhao, D., Heidari, A.A., Liu, L., Zhang, X., Mafarja, M., & Chen, H.
(2023). RIME: A physics-based optimization. Neurocomputing, 532, 183-214.
https://doi.org/10.1016/j.neucom.2023.02.010

This is a vectorized NumPy re-implementation, ported directly from the
authors' official MATLAB source code (RIME.m, publicly released at
https://github.com/aliasgharheidaricom/RIME-A-physics-based-optimization),
preserving the exact update equations:

  Soft-rime search (Eq. 3-5)   : exploration via a cosine-modulated
                                  "RimeFactor" perturbation around the
                                  current best agent.
  Hard-rime puncture (Eq. 6-7) : exploitation by probabilistically
                                  overwriting dimensions of an agent with
                                  the best agent's value, with probability
                                  proportional to that agent's *normalized*
                                  (L2-normalized) fitness -- i.e. worse
                                  agents are pulled toward the best agent
                                  more aggressively.
  Positive greedy selection    : an agent (and the global best) is only
                                  updated if the candidate strictly
                                  improves its fitness.

Minimization convention (lower fitness = better), matching the RMSE/MAPE
fitness function used for Phase 1 hyperparameter tuning.
"""

from __future__ import annotations
from typing import Callable, Optional
import numpy as np

from .base import OptimizationResult, initialize_population, clip_to_bounds


def _matlab_round(x: float) -> float:
    """MATLAB's round() rounds half away from zero; NumPy/Python round to
    even. Reproduced exactly for fidelity to the reference implementation."""
    return np.floor(x + 0.5) if x >= 0 else np.ceil(x - 0.5)


def rime_optimizer(fobj: Callable[[np.ndarray], float], dim: int,
                    lb: np.ndarray, ub: np.ndarray,
                    n_pop: int = 30, max_iter: int = 100,
                    seed: Optional[int] = None,
                    callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
                    verbose: bool = False) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    lb_arr = np.broadcast_to(np.asarray(lb, dtype=float), (dim,)).copy()
    ub_arr = np.broadcast_to(np.asarray(ub, dtype=float), (dim,)).copy()

    pop = initialize_population(n_pop, dim, lb_arr, ub_arr, rng)
    fitness = np.array([fobj(ind) for ind in pop], dtype=float)
    n_fevals = n_pop

    best_idx = int(np.argmin(fitness))
    best = pop[best_idx].copy()
    best_fit = float(fitness[best_idx])

    W = 5.0  # soft-rime control parameter (paper Sec. 4.3.1 default)
    curve = np.zeros(max_iter)

    for it in range(1, max_iter + 1):
        rime_factor = (rng.random() - 0.5) * 2.0 * np.cos(np.pi * it / (max_iter / 10.0)) \
            * (1.0 - _matlab_round(it * W / max_iter) / W)
        E = np.sqrt(it / max_iter)

        new_pop = pop.copy()

        # --- Soft-rime search strategy (exploration) ---
        r1 = rng.random((n_pop, dim))
        soft_mask = r1 < E
        rand_uv = rng.random((n_pop, dim))
        soft_candidate = best[None, :] + rime_factor * (
            (ub_arr - lb_arr)[None, :] * rand_uv + lb_arr[None, :]
        )
        new_pop = np.where(soft_mask, soft_candidate, new_pop)

        # --- Hard-rime puncture mechanism (exploitation) ---
        fit_norm = np.linalg.norm(fitness)
        normalized_rates = fitness / fit_norm if fit_norm > 1e-12 else np.zeros_like(fitness)
        r2 = rng.random((n_pop, dim))
        hard_mask = r2 < normalized_rates[:, None]
        new_pop = np.where(hard_mask, best[None, :], new_pop)

        # --- Boundary absorption + positive greedy selection ---
        new_pop = clip_to_bounds(new_pop, lb_arr, ub_arr)
        new_fitness = np.array([fobj(ind) for ind in new_pop], dtype=float)
        n_fevals += n_pop

        improve = new_fitness < fitness
        pop[improve] = new_pop[improve]
        fitness[improve] = new_fitness[improve]

        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_fit:
            best_fit = float(fitness[gen_best_idx])
            best = pop[gen_best_idx].copy()

        curve[it - 1] = best_fit
        if callback is not None:
            callback(it, best, best_fit)
        if verbose and (it % max(1, max_iter // 10) == 0):
            print(f"[RIME] iter {it:4d}/{max_iter} | best_fitness = {best_fit:.6f}")

    return OptimizationResult(best_solution=best, best_fitness=best_fit,
                               convergence_curve=curve, n_fevals=n_fevals, algorithm="RIME")
