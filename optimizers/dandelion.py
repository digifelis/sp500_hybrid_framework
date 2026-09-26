"""
optimizers/dandelion.py
========================
Dandelion Optimizer (DO).

Reference
---------
Zhao, S., Zhang, T., Ma, S., & Chen, M. (2022). Dandelion Optimizer: A
nature-inspired metaheuristic algorithm for engineering applications.
Engineering Applications of Artificial Intelligence, 114, 105075.
https://doi.org/10.1016/j.engappai.2022.105075

Implements the three-stage seed-dispersal metaphor exactly as specified by
the paper's equations (numbering follows the widely-cited reproduction in
Kiziloluk et al., 2024, PeerJ Computer Science, which restates Zhao et
al.'s original Eq. numbers 1-18):

  Rising stage     (Eq. 2-12) : clear-day spiral ascent (exploration) vs.
                                 rainy-day local drift (exploitation),
                                 chosen stochastically each step.
  Descending stage (Eq. 13-14): Brownian-motion descent toward the
                                 population's mean position (exploration).
  Landing stage    (Eq. 15-18): Levy-flight walk toward the elite seed
                                 (exploitation) -- Mantegna's algorithm is
                                 used for the Levy step, since Eq. 17's
                                 sigma expression is exactly the standard
                                 Mantegna sigma_u formula.

All three stages are applied sequentially to the whole population within
every iteration (rise -> descend -> land), after which fitness is
evaluated once per agent and a greedy (elitist) replacement is applied --
matching the algorithm's documented four-part loop (init -> evaluate ->
three-stage position update -> re-evaluate/rank).

Minimization convention (lower fitness = better). This module is used both
for Phase 2 (Markowitz-style portfolio optimization) and is reusable for
Phase 1 hyperparameter tuning if desired.
"""

from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from scipy.special import gamma

from .base import OptimizationResult, initialize_population, clip_to_bounds


def _mantegna_levy(dim: int, rng: np.random.Generator, beta: float = 1.5, s: float = 0.01) -> np.ndarray:
    """Levy(lambda) step via Mantegna's algorithm (Eq. 16-17)."""
    num = gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
    den = gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0)
    sigma_u = (num / den) ** (1.0 / beta)
    u = rng.normal(0.0, sigma_u, dim)
    v = rng.normal(0.0, 1.0, dim)
    return s * u / (np.abs(v) ** (1.0 / beta))


def _lognormal_density_at_standard_normal(rng: np.random.Generator) -> float:
    """Eq. 4: lnY = lognormal PDF (sigma^2=1) evaluated at a standard-normal
    draw y; defined as 0 for y <= 0. A small epsilon guards the y -> 0+
    singularity of the density."""
    y = rng.normal(0.0, 1.0)
    if y <= 1e-6:
        return 0.0
    return float((1.0 / (y * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (np.log(y)) ** 2))


def dandelion_optimizer(fobj: Callable[[np.ndarray], float], dim: int,
                         lb: np.ndarray, ub: np.ndarray,
                         n_pop: int = 40, max_iter: int = 150,
                         seed: Optional[int] = None,
                         callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
                         verbose: bool = False) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    lb_arr = np.broadcast_to(np.asarray(lb, dtype=float), (dim,)).copy()
    ub_arr = np.broadcast_to(np.asarray(ub, dtype=float), (dim,)).copy()
    T = float(max_iter)

    pop = initialize_population(n_pop, dim, lb_arr, ub_arr, rng)     # Eq. 1
    fitness = np.array([fobj(ind) for ind in pop], dtype=float)
    n_fevals = n_pop

    elite_idx = int(np.argmin(fitness))
    elite = pop[elite_idx].copy()
    elite_fit = float(fitness[elite_idx])

    curve = np.zeros(max_iter)
    D_const = max((T - 1.0) ** 2, 1e-6)  # denominator in Eq. 11, guarded for T=1

    for t in range(1, max_iter + 1):
        # ---------------- Rising stage (Eq. 2-12) ----------------
        risen = np.empty_like(pop)
        for i in range(n_pop):
            rndn = rng.normal(0.0, 1.0)
            if rndn < 1.5:  # clear-day sub-stage: spiral, global exploration
                theta = rng.uniform(-np.pi, np.pi)
                r = 1.0 / np.exp(theta)
                vx, vy = r * np.cos(theta), r * np.sin(theta)
                lnY = _lognormal_density_at_standard_normal(rng)
                alpha = rng.random() * ((1.0 / T ** 2) * t ** 2 - (2.0 / T) * t + 1.0)
                Ss = rng.random(dim) * (ub_arr - lb_arr) + lb_arr
                risen[i] = pop[i] + alpha * vx * vy * lnY * (Ss - pop[i])
            else:           # rainy-day sub-stage: local drift, exploitation
                q = (t ** 2) / D_const - (2.0 * t) / D_const + 1.0 + 1.0 / D_const
                k = 1.0 - rng.random() * q
                risen[i] = pop[i] * k
        risen = clip_to_bounds(risen, lb_arr, ub_arr)

        # ---------------- Descending stage (Eq. 13-14) ----------------
        S_mean = risen.mean(axis=0)
        descended = np.empty_like(pop)
        for i in range(n_pop):
            beta_t = rng.normal(0.0, 1.0, dim)  # Brownian-motion vector
            alpha = rng.random() * ((1.0 / T ** 2) * t ** 2 - (2.0 / T) * t + 1.0)
            descended[i] = risen[i] - alpha * beta_t * (S_mean - alpha * beta_t * risen[i])
        descended = clip_to_bounds(descended, lb_arr, ub_arr)

        # ---------------- Landing stage (Eq. 15-18) ----------------
        delta = 2.0 * t / T
        landed = np.empty_like(pop)
        for i in range(n_pop):
            levy_step = _mantegna_levy(dim, rng, beta=1.5)
            alpha = rng.random() * ((1.0 / T ** 2) * t ** 2 - (2.0 / T) * t + 1.0)
            landed[i] = elite + levy_step * alpha * (elite - descended[i] * delta)
        landed = clip_to_bounds(landed, lb_arr, ub_arr)

        # ---------------- Evaluation + greedy (elitist) selection ----------------
        landed_fit = np.array([fobj(ind) for ind in landed], dtype=float)
        n_fevals += n_pop
        improve = landed_fit < fitness
        pop[improve] = landed[improve]
        fitness[improve] = landed_fit[improve]

        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < elite_fit:
            elite_fit = float(fitness[gen_best_idx])
            elite = pop[gen_best_idx].copy()

        curve[t - 1] = elite_fit
        if callback is not None:
            callback(t, elite, elite_fit)
        if verbose and (t % max(1, max_iter // 10) == 0):
            print(f"[DO] iter {t:4d}/{max_iter} | best_fitness = {elite_fit:.6f}")

    return OptimizationResult(best_solution=elite, best_fitness=elite_fit,
                               convergence_curve=curve, n_fevals=n_fevals, algorithm="Dandelion Optimizer")
