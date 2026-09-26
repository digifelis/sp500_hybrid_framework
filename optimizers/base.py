"""
optimizers/base.py
===================
Shared infrastructure for the three custom metaheuristics (RIME, MGO,
Dandelion Optimizer): a common result container, a vectorized random
initializer, and a small suite of classic benchmark functions (Sphere,
Rastrigin, Rosenbrock, Ackley) used purely to validate that each optimizer
converges correctly BEFORE it is trusted with the expensive LSTM /
portfolio fitness functions -- standard due diligence for an SCI-style
methods section (see `validate_optimizers.py`-style usage in main.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List
import numpy as np


@dataclass
class OptimizationResult:
    best_solution: np.ndarray
    best_fitness: float
    convergence_curve: np.ndarray   # best-so-far fitness per iteration
    history: List[np.ndarray] = field(default_factory=list)  # optional population snapshots
    n_fevals: int = 0
    algorithm: str = ""


def initialize_population(n: int, dim: int, lb: np.ndarray, ub: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    lb = np.broadcast_to(lb, (dim,)).astype(float)
    ub = np.broadcast_to(ub, (dim,)).astype(float)
    return lb + rng.random((n, dim)) * (ub - lb)


def clip_to_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lb), ub)


# --------------------------------------------------------------------------
# Classic CEC-style unimodal / multimodal benchmark functions (minimization)
# global minimum = 0 at x* for all four, used only for algorithm validation
# --------------------------------------------------------------------------
def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    return float(A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


def ackley(x: np.ndarray) -> float:
    d = len(x)
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / d)
    return float(term1 + term2 + 20.0 + np.e)


BENCHMARK_SUITE = {
    "sphere": (sphere, -100.0, 100.0),
    "rastrigin": (rastrigin, -5.12, 5.12),
    "rosenbrock": (rosenbrock, -30.0, 30.0),
    "ackley": (ackley, -32.0, 32.0),
}
