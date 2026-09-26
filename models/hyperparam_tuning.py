"""
models/hyperparam_tuning.py
============================
Couples the metaheuristic optimizers (RIME / MGO) to the LSTM forecaster:
defines the fitness function (weighted RMSE + MAPE on the validation
split, as specified in the study design) and decodes a real-valued
hyperparameter vector -> {learning_rate, hidden_units, dropout_rate,
batch_size, epochs} -> a trained model's validation performance.
"""

from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
import torch

from config import HyperparamBounds, FITNESS_WEIGHTS, GLOBAL_SEED, release_torch_memory
from models.sequence_models import SequenceModelConfig, build_model, train_forecaster, predict
from utils.metrics import rmse, mape
from optimizers.base import OptimizationResult
from optimizers.rime import rime_optimizer
from optimizers.mgo import mountain_gazelle_optimizer


def decode_hyperparams(x: np.ndarray, bounds: HyperparamBounds,
                        input_size: int, horizon: int) -> SequenceModelConfig:
    x = np.clip(x, bounds.lb, bounds.ub)
    lr, hidden_units, dropout, batch_size, epochs = x
    return SequenceModelConfig(
        input_size=input_size,
        hidden_size=int(round(hidden_units)),
        num_layers=2,
        dropout=float(dropout),
        horizon=horizon,
        learning_rate=float(lr),
        batch_size=int(round(batch_size)),
        epochs=int(round(epochs)),
    )


def make_fitness_function(X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           bounds: HyperparamBounds, input_size: int, horizon: int,
                           model_type: str = "lstm",
                           weights: dict = FITNESS_WEIGHTS,
                           fixed_seed: int = GLOBAL_SEED) -> Callable[[np.ndarray], float]:
    """Fitness = weighted_sum(RMSE, MAPE/100) on the held-out validation
    split, computed on the *normalized* [0,1] price scale (as specified:
    "minimizing RMSE and MAPE on the validation set"). MAPE is divided by
    100 so both terms live on a comparable O(0.1) scale before weighting.
    A fixed seed is used for every candidate's training run so that
    fitness differences reflect the hyperparameters rather than training
    stochasticity, which the metaheuristic search relies on.
    """
    def fitness(x: np.ndarray) -> float:
        torch.manual_seed(fixed_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fixed_seed)
        cfg = decode_hyperparams(x, bounds, input_size, horizon)
        model = build_model(model_type, cfg)
        try:
            train_forecaster(model, X_train, y_train, X_val, y_val, cfg,
                              verbose=False, early_stopping_patience=5)
            val_pred = predict(model, X_val)
            val_rmse = rmse(y_val, val_pred)
            val_mape = mape(y_val, val_pred)
            return weights["rmse"] * val_rmse + weights["mape"] * (val_mape / 100.0)
        finally:
            del model
            release_torch_memory()
    return fitness


def run_hyperparameter_search(optimizer_name: str,
                               X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               bounds: HyperparamBounds, input_size: int, horizon: int,
                               n_pop: int, max_iter: int, seed: int = GLOBAL_SEED,
                               model_type: str = "lstm", verbose: bool = True
                               ) -> Tuple[OptimizationResult, SequenceModelConfig]:
    """Runs RIME or MGO to search the 5-D hyperparameter space and returns
    both the raw optimizer result (with convergence curve, for the paper's
    convergence-analysis figure) and the decoded best SequenceModelConfig,
    ready for final retraining."""
    fitness_fn = make_fitness_function(X_train, y_train, X_val, y_val, bounds,
                                        input_size, horizon, model_type=model_type)
    optimizer_name = optimizer_name.upper()
    if optimizer_name == "RIME":
        result = rime_optimizer(fitness_fn, bounds.dim, bounds.lb, bounds.ub,
                                 n_pop=n_pop, max_iter=max_iter, seed=seed, verbose=verbose)
    elif optimizer_name == "MGO":
        result = mountain_gazelle_optimizer(fitness_fn, bounds.dim, bounds.lb, bounds.ub,
                                             n_pop=n_pop, max_iter=max_iter, seed=seed, verbose=verbose)
    else:
        raise ValueError("optimizer_name must be 'RIME' or 'MGO'")

    best_cfg = decode_hyperparams(result.best_solution, bounds, input_size, horizon)
    return result, best_cfg
