"""
optimizers/mgo.py
==================
Mountain Gazelle Optimizer (MGO).

Reference
---------
Abdollahzadeh, B., Gharehchopogh, F.S., Khodadadi, N., & Mirjalili, S.
(2022). Mountain Gazelle Optimizer: A new Nature-inspired Metaheuristic
Algorithm for Global Optimization Problems. Advances in Engineering
Software, 174, 103282. https://doi.org/10.1016/j.advengsoft.2022.103282

Implements the four behavioral mechanisms of the original paper (Eq. 1-9):
  TSM  Territorial Solitary Males      (Eq. 1)   -- local exploitation
  BH   young-male-herd coefficient     (Eq. 2)   -- used inside TSM/MH/BMH
  F    dynamic step-size factor        (Eq. 3)
  Cofi random coefficient vector       (Eq. 4, one of four sub-formulas)
  a    linearly-decreasing factor      (Eq. 5)
  MH   Maternity Herds                 (Eq. 6)   -- balanced explore/exploit
  BMH  Bachelor Male Herds             (Eq. 7-8) -- global exploration
  MSF  Migration in Search of Food     (Eq. 9)   -- pure random exploration

Each iteration, every gazelle is regenerated via ONE of {TSM, MH, BMH, MSF}
selected at random (uniform), consistent with the paper's description that
"each trial member of the population can be produced among the subgroups
[...] during the iterative process" -- i.e. one mechanism per agent per
iteration, followed by greedy (elitist) replacement.

Implementation notes / documented engineering choices where the published
equations are ambiguous in secondary transcriptions:
  * Eq. 4's fourth coefficient sub-formula involves a `cos(...)` term
    reported inconsistently as either a divisor or multiplier across
    sources; it is implemented here as a *multiplier* to avoid an
    unstable near-zero denominator, matching common re-implementations.
  * Eq. 8's coefficient `(2*r - 1)` (r ~ U(0,1)) is used, the standard
    "signed unit interval" pattern used throughout the paper's other
    equations (e.g. Eq. 9's use of `ub-lb` scaling).

Minimization convention (lower fitness = better).
"""

from __future__ import annotations
from typing import Callable, Optional
import numpy as np

from .base import OptimizationResult, initialize_population, clip_to_bounds


def _random_coefficient_vector(dim: int, a: float, rng: np.random.Generator) -> np.ndarray:
    """One random draw from the 4-branch Cofi definition (Eq. 4)."""
    branch = rng.integers(0, 4)
    if branch == 0:
        r3 = rng.random()
        return np.full(dim, (a + 1.0) + r3)
    elif branch == 1:
        N2 = rng.normal(0.0, 1.0, dim)
        return a * N2
    elif branch == 2:
        r4 = rng.random(dim)
        return r4
    else:
        N3 = rng.normal(0.0, 1.0, dim)
        N4 = rng.normal(0.0, 1.0, dim)
        r4 = rng.random()
        return (N3 * N4 ** 2) * np.cos((2.0 * r4) * N3) / 2.0


def _r12(rng: np.random.Generator) -> int:
    """Random integer in {1, 2}, as used for r_i1, r_i2, ... throughout MGO."""
    return int(rng.integers(1, 3))


def mountain_gazelle_optimizer(fobj: Callable[[np.ndarray], float], dim: int,
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
    male_gazelle = pop[best_idx].copy()   # "adult male gazelle" = best-so-far
    best_fit = float(fitness[best_idx])

    curve = np.zeros(max_iter)

    for it in range(1, max_iter + 1):
        a = -1.0 + it * (-1.0 / max_iter)             # Eq. 5: -1 -> -2
        F = rng.normal(0.0, 1.0, dim) * (2.0 - it * (2.0 / max_iter))  # Eq. 3

        for i in range(n_pop):
            X = pop[i]
            Mpr = pop.mean(axis=0)                      # mean position vector
            ra_lo = max(int(round(n_pop / 3.0)), 1)
            ra_idx = rng.integers(min(ra_lo, n_pop - 1), n_pop)
            X_ra = pop[ra_idx]
            r1, r2 = rng.random(), rng.random()
            BH = X_ra * r1 + Mpr * r2                    # Eq. 2

            mechanism = rng.integers(0, 4)
            if mechanism == 0:                            # Territorial Solitary Male
                Cofr = _random_coefficient_vector(dim, a, rng)
                candidate = male_gazelle - np.abs(_r12(rng) * BH - _r12(rng) * X) * F * Cofr
            elif mechanism == 1:                          # Maternity Herd
                Cof1r = _random_coefficient_vector(dim, a, rng)
                X_rand = pop[rng.integers(0, n_pop)]
                candidate = (BH + Cof1r) + (_r12(rng) * male_gazelle - _r12(rng) * X_rand) * Cof1r
            elif mechanism == 2:                          # Bachelor Male Herd
                r6 = rng.random(dim)
                D = (np.abs(X) + np.abs(male_gazelle)) * (2.0 * r6 - 1.0)
                Cofr = _random_coefficient_vector(dim, a, rng)
                candidate = (X - D) + (_r12(rng) * male_gazelle - _r12(rng) * BH) * Cofr
            else:                                         # Migration in Search of Food
                r7 = rng.random(dim)
                candidate = (ub_arr - lb_arr) * r7 + lb_arr

            candidate = clip_to_bounds(candidate, lb_arr, ub_arr)
            cand_fit = fobj(candidate)
            n_fevals += 1

            if cand_fit < fitness[i]:
                pop[i] = candidate
                fitness[i] = cand_fit
                if cand_fit < best_fit:
                    best_fit = float(cand_fit)
                    male_gazelle = candidate.copy()

        curve[it - 1] = best_fit
        if callback is not None:
            callback(it, male_gazelle, best_fit)
        if verbose and (it % max(1, max_iter // 10) == 0):
            print(f"[MGO] iter {it:4d}/{max_iter} | best_fitness = {best_fit:.6f}")

    return OptimizationResult(best_solution=male_gazelle, best_fitness=best_fit,
                               convergence_curve=curve, n_fevals=n_fevals, algorithm="MGO")
