"""
config.py
=========
Central configuration for the RIME/MGO-LSTM + Dandelion-Portfolio framework.

All "knobs" a reviewer would ask about (universe, dates, bounds, seeds) live
here so the study is reproducible from a single file.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import os
import numpy as np

# --------------------------------------------------------------------------
# 1. Reproducibility & PyTorch device (GPU / CPU)
# --------------------------------------------------------------------------
GLOBAL_SEED = 42

# "cuda" (default), "cpu", or "cuda:0" / "cuda:1" — override via TORCH_DEVICE env var
TORCH_DEVICE_PREFERENCE = os.environ.get("TORCH_DEVICE", "cuda")

_torch_device = None


def get_torch_device():
    """Return the active PyTorch device. Prefers CUDA when available."""
    global _torch_device
    if _torch_device is not None:
        return _torch_device
    try:
        import torch
    except ImportError:
        return None
    pref = TORCH_DEVICE_PREFERENCE.lower().strip()
    if pref.startswith("cuda") and torch.cuda.is_available():
        idx = pref if ":" in pref else "cuda:0"
        _torch_device = torch.device(idx)
    else:
        _torch_device = torch.device("cpu")
    return _torch_device


def configure_torch_runtime(seed: int = GLOBAL_SEED, verbose: bool = True):
    """Enable cuDNN autotune / TF32 and print the selected compute device."""
    global _torch_device
    _torch_device = None
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required. For GPU: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu124"
        ) from exc

    device = get_torch_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        torch.backends.cudnn.deterministic = True
        if verbose:
            name = torch.cuda.get_device_name(device)
            mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            print(f"[GPU] {device} — {name} ({mem_gb:.1f} GB)")
    elif verbose:
        print("[CPU] CUDA not available; training will run on CPU.")
        print("      Install CUDA PyTorch: pip install torch --index-url "
              "https://download.pytorch.org/whl/cu124")
    set_global_seed(seed)
    return device


def release_torch_memory() -> None:
    """Free GPU cache between successive model trainings (hyperparameter search)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# 2. Investment universe: 25 high-liquidity BIST 100 constituents
#    (large-cap, high free-float, cross-sector — banking, industrials,
#    energy, retail, telecom, defense, aviation — chosen for liquidity
#    and to avoid single-sector concentration bias in the portfolio study)
# --------------------------------------------------------------------------
BIST_TICKERS: List[str] = [
    "MSFT", "AAPL", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "JPM", "BAC", "V",
    "JNJ", "UNH", "LLY", "XOM", "CVX",
    "PG", "WMT", "KO", "PEP", "COST",
    "HD", "MCD", "GE", "CAT", "HON"
]

BENCHMARK_INDEX = "SPY"  # BIST 100 index ticker (buy & hold baseline)

# --------------------------------------------------------------------------
# 3. Study horizon
# --------------------------------------------------------------------------
HISTORY_START = "2021-01-04"     # ~5-year in-sample / model-development window
HISTORY_END = "2026-07-10"       # FIXED: was "2025-12-31", which silently truncated
                                  # the backtest ~6 months short of OOS_BACKTEST_END below
                                  # (real BIST data is available through 2026-07-10)
OOS_BACKTEST_START = "2025-01-01"  # rolling-window out-of-sample test region
OOS_BACKTEST_END = "2026-06-30"    # deliberately spans the volatile 2025-2026 window
TRADING_DAYS_PER_YEAR = 252

# --------------------------------------------------------------------------
# 4. Technical-indicator engineering
# --------------------------------------------------------------------------
RSI_WINDOW = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_WINDOW, BB_NUM_STD = 20, 2.0

FEATURE_COLUMNS: List[str] = [
    "close_norm", "volume_norm", "rsi", "macd", "macd_signal", "macd_hist",
    "bb_pctb", "bb_bandwidth", "log_return",
]

# --------------------------------------------------------------------------
# 5. Sequence-model (Phase 1) settings
# --------------------------------------------------------------------------
SEQUENCE_LENGTH = 42          # look-back window (trading days) 30
FORECAST_HORIZON = 5          # multi-step-ahead forecast (trading days)
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15           # remaining 0.15 is held out as the test split


@dataclass
class HyperparamBounds:
    """Search space for RIME / MGO tuning of the LSTM/GRU forecaster.
    Continuous surrogates are used for the two integer-valued genes
    (hidden_units, batch_size) and rounded at evaluation time -- this is
    standard practice for continuous metaheuristics applied to mixed
    integer/continuous hyperparameter spaces.
    """
    learning_rate: Tuple[float, float] = (5e-4, 5e-3)
    hidden_units: Tuple[float, float] = (32, 128)
    dropout_rate: Tuple[float, float] = (0.1, 0.5)
    batch_size: Tuple[float, float] = (32, 128)
    epochs: Tuple[float, float] = (30, 80)

    @property
    def lb(self) -> np.ndarray:
        return np.array([b[0] for b in
                          (self.learning_rate, self.hidden_units, self.dropout_rate,
                           self.batch_size, self.epochs)])

    @property
    def ub(self) -> np.ndarray:
        return np.array([b[1] for b in
                          (self.learning_rate, self.hidden_units, self.dropout_rate,
                           self.batch_size, self.epochs)])

    @property
    def dim(self) -> int:
        return 5

    param_names = ["learning_rate", "hidden_units", "dropout_rate", "batch_size", "epochs"]


HP_BOUNDS = HyperparamBounds()

# --------------------------------------------------------------------------
# 6. Metaheuristic optimizer settings (Phase 1 - hyperparameter search)
# --------------------------------------------------------------------------
HP_OPTIMIZER_POPULATION = 30   # keep modest by default: each candidate = 1 LSTM training run
HP_OPTIMIZER_ITERATIONS = 50   # scale up (e.g. 30/50) for a full production run
FITNESS_WEIGHTS = {"rmse": 0.6, "mape": 0.4}  # weighted-sum scalarization of the fitness function

# --------------------------------------------------------------------------
# 7. Metaheuristic optimizer settings (Phase 2 - portfolio optimization)
# --------------------------------------------------------------------------
PORTFOLIO_OPTIMIZER_POPULATION = 40
PORTFOLIO_OPTIMIZER_ITERATIONS = 150
RISK_FREE_RATE_ANNUAL = 0.038   # approx. CBRT policy-rate proxy for TRY risk-free asset (user-editable)

PORTFOLIO_OBJECTIVE_WEIGHTS = {
    "sharpe": 0.25,
    "sortino": 0.25,
    "mdd": 0.25,
    "variance": 0.25,
}

# --------------------------------------------------------------------------
# 8. Rolling-window backtest settings
# --------------------------------------------------------------------------
BACKTEST_TRAIN_WINDOW = 252    # trading days of history used to fit models/weights each roll
BACKTEST_REBALANCE_FREQ = 21   # re-optimize portfolio ~monthly
BACKTEST_STEP = 21

# --------------------------------------------------------------------------
# 9. Paths
# --------------------------------------------------------------------------
DATA_DIR = "data_cache"
RESULTS_DIR = "results"
OUTPUTS_DIR = "outputs"
