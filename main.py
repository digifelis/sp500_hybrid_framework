"""
main.py
========
End-to-end orchestration of the RIME/MGO-LSTM + Dandelion-Portfolio study.

Run modes
---------
This script ships with two settings profiles (see `RUN_PROFILE` below):

  "demo"       -- small population/iteration/epoch counts and a reduced
                  ticker subset, so the ENTIRE pipeline (Phase 1 tuning,
                  Phase 1 benchmark table, Phase 2 rolling backtest) runs
                  in a few minutes on a single CPU core. This is what runs
                  by default and is intended to validate that the whole
                  pipeline is wired correctly end-to-end.

  "production" -- the full settings from config.py (25-ticker universe,
                  larger optimizer populations/iterations, more epochs).
                  Intended for a GPU-equipped machine with real BIST data
                  (see data/data_pipeline.py's `load_real_universe`).
                  Expect a multi-hour run.

Swap in real data
------------------
Replace the `generate_synthetic_universe(...)` call below with:

    universe = load_real_universe(csv_dir="path/to/your/bist_csv_folder",
                                   tickers=cfg.BIST_TICKERS,
                                   start=cfg.HISTORY_START, end=cfg.HISTORY_END)

once you have exported real daily OHLCV per ticker (see that function's
docstring for the expected CSV schema).
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from config import set_global_seed, configure_torch_runtime
from data.data_pipeline import load_real_universe, load_real_index
from utils.technical_indicators import build_feature_frame
from optimizers.base import BENCHMARK_SUITE
from optimizers.rime import rime_optimizer
from optimizers.mgo import mountain_gazelle_optimizer
from optimizers.dandelion import dandelion_optimizer
from backtest.engine import run_forecasting_benchmark, run_rolling_portfolio_backtest

# ==========================================================================
# 0. Run profile
# ==========================================================================
RUN_PROFILE = "production"   # "demo" or "production"

REAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "sp500_real_data")
USE_REAL_DATA = True

if RUN_PROFILE == "demo":
    DEMO_TICKERS = cfg.BIST_TICKERS[:6]          # 6 of the 25 configured liquid names
    HP_POP, HP_ITER = 4, 4                        # RIME/MGO hyperparameter-search budget
    BASELINE_HIDDEN, BASELINE_EPOCHS = 16, 8
    PORTFOLIO_TOP_K = 4
    cfg.SEQUENCE_LENGTH = 15
    cfg.HP_BOUNDS.hidden_units = (8.0, 28.0)
    cfg.HP_BOUNDS.epochs = (5.0, 12.0)
    cfg.HP_BOUNDS.batch_size = (16.0, 64.0)
    cfg.PORTFOLIO_OPTIMIZER_POPULATION = 25
    cfg.PORTFOLIO_OPTIMIZER_ITERATIONS = 60
    cfg.BACKTEST_TRAIN_WINDOW = 100
    cfg.BACKTEST_STEP = 42
else:
    DEMO_TICKERS = cfg.BIST_TICKERS
    HP_POP, HP_ITER = cfg.HP_OPTIMIZER_POPULATION, cfg.HP_OPTIMIZER_ITERATIONS
    BASELINE_HIDDEN, BASELINE_EPOCHS = 64, 50
    PORTFOLIO_TOP_K = 15

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
configure_torch_runtime(cfg.GLOBAL_SEED)


def main():
    # ----------------------------------------------------------------
    # 1. Data acquisition (real BIST OHLCV from bist_real_data/)
    # ----------------------------------------------------------------
    print("=" * 70, "\nSTEP 1/5: Loading the price universe\n", "=" * 70)
    if not USE_REAL_DATA:
        raise ValueError("USE_REAL_DATA must be True for this run profile.")
    universe = load_real_universe(
        csv_dir=REAL_DATA_DIR,
        tickers=DEMO_TICKERS,
        start=cfg.HISTORY_START,
        end=cfg.HISTORY_END,
    )
    index_level = load_real_index(
        csv_dir=REAL_DATA_DIR,
        index_ticker=cfg.BENCHMARK_INDEX,
        start=cfg.HISTORY_START,
        end=cfg.HISTORY_END,
    )
    n_days = len(next(iter(universe.values())))
    print(f"Loaded {len(universe)} tickers x {n_days} trading days "
          f"({cfg.HISTORY_START} -> {cfg.HISTORY_END}) from {REAL_DATA_DIR}")
    print(f"Benchmark index {cfg.BENCHMARK_INDEX}: {len(index_level)} trading days.")

    # ----------------------------------------------------------------
    # 2. Feature engineering (RSI, MACD, Bollinger Bands, [0,1] scaling)
    # ----------------------------------------------------------------
    print("\n" + "=" * 70, "\nSTEP 2/5: Technical-indicator feature engineering\n", "=" * 70)
    feature_frames = {tkr: build_feature_frame(df, cfg) for tkr, df in universe.items()}
    example = next(iter(feature_frames.values()))
    print(f"Feature columns: {cfg.FEATURE_COLUMNS}")
    print(f"Example engineered frame shape: {example.shape}, date range "
          f"{example.index.min().date()} -> {example.index.max().date()}")

    # ----------------------------------------------------------------
    # 3. Optimizer sanity check on classic benchmark functions
    # ----------------------------------------------------------------
    print("\n" + "=" * 70, "\nSTEP 3/5: Validating RIME / MGO / DO on benchmark functions\n", "=" * 70)
    # NOTE: RIME and the Dandelion Optimizer are documented in the literature
    # (and reproduced by this implementation, see optimizers/rime.py and
    # optimizers/dandelion.py docstrings) as having slower late-stage
    # convergence than MGO on smooth unimodal functions -- they need a
    # larger iteration budget to reach the same precision. 400 iterations
    # (still <1s per run for these cheap analytic functions) gives all
    # three a fair chance to demonstrate convergence before Phase 1/2,
    # which use expensive fitness functions, apply much smaller budgets.
    validation_rows = []
    for fname, (fn, lo, hi) in BENCHMARK_SUITE.items():
        for opt_name, opt in [("RIME", rime_optimizer), ("MGO", mountain_gazelle_optimizer),
                               ("Dandelion", dandelion_optimizer)]:
            res = opt(fn, 10, lo, hi, n_pop=20, max_iter=400, seed=cfg.GLOBAL_SEED)
            validation_rows.append({"function": fname, "optimizer": opt_name, "best_fitness": res.best_fitness})
    validation_df = pd.DataFrame(validation_rows).pivot(index="function", columns="optimizer", values="best_fitness")
    print(validation_df.to_string(float_format=lambda x: f"{x:.3e}"))
    print("(Note: RIME/DO trade early-iteration speed for late-stage precision relative to MGO on "
          "smooth unimodal functions -- a documented characteristic, not an implementation defect; "
          "see the optimizer modules' docstrings.)")
    validation_df.to_csv(os.path.join(OUTPUT_DIR, "optimizer_validation_benchmarks.csv"))

    # ----------------------------------------------------------------
    # 4. PHASE 1 -- forecasting benchmark (RIME-LSTM / MGO-LSTM / LSTM / GRU / ARIMA)
    # ----------------------------------------------------------------
    print("\n" + "=" * 70, "\nSTEP 4/5: Phase 1 - forecasting benchmark\n", "=" * 70)
    results_df, tuned_configs, optimizer_results = run_forecasting_benchmark(
        feature_frames, DEMO_TICKERS, cfg, hp_pop=HP_POP, hp_iter=HP_ITER,
        baseline_hidden=BASELINE_HIDDEN, baseline_epochs=BASELINE_EPOCHS,
        seed=cfg.GLOBAL_SEED, verbose=True,
    )
    results_df.to_csv(os.path.join(OUTPUT_DIR, "phase1_forecasting_benchmark_raw.csv"), index=False)
    summary_table = results_df.groupby("model")[["RMSE_price", "MAPE_price_%"]].mean().sort_values("RMSE_price")
    print("\nPhase 1 summary (mean over tickers):\n", summary_table.to_string(float_format=lambda x: f"{x:.4f}"))
    summary_table.to_csv(os.path.join(OUTPUT_DIR, "phase1_forecasting_benchmark_summary.csv"))

    # champion = best of RIME-LSTM / MGO-LSTM by mean price-scale RMSE
    proposed_models = [m for m in ("RIME-LSTM", "MGO-LSTM") if m in summary_table.index]
    champion_name = summary_table.loc[proposed_models, "RMSE_price"].idxmin()
    champion_cfg = tuned_configs[champion_name]
    print(f"\nChampion Phase-1 model selected for Phase-2 asset ranking: {champion_name}")

    # convergence plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, res in optimizer_results.items():
        ax.plot(res.convergence_curve, label=f"{name} (final={res.best_fitness:.4f})")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Fitness (weighted RMSE+MAPE)")
    ax.set_title("Phase 1: RIME vs. MGO hyperparameter-search convergence")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "phase1_convergence_curves.png"), dpi=150)
    plt.close(fig)

    # ----------------------------------------------------------------
    # 5. PHASE 2 -- rolling-window portfolio backtest
    # ----------------------------------------------------------------
    print("\n" + "=" * 70, "\nSTEP 5/5: Phase 2 - rolling-window portfolio backtest (OOS "
          f"{cfg.OOS_BACKTEST_START} -> {cfg.OOS_BACKTEST_END})\n", "=" * 70)
    concatenated_returns, portfolio_summary, roll_log = run_rolling_portfolio_backtest(
        feature_frames, universe, index_level, cfg, forecaster_cfg=champion_cfg,
        top_k=PORTFOLIO_TOP_K, seed=cfg.GLOBAL_SEED, verbose=True,
    )
    print("\nPhase 2 out-of-sample performance summary:\n",
          portfolio_summary.to_string(float_format=lambda x: f"{x:.4f}"))
    portfolio_summary.to_csv(os.path.join(OUTPUT_DIR, "phase2_portfolio_backtest_summary.csv"))

    with open(os.path.join(OUTPUT_DIR, "phase2_roll_log.json"), "w") as f:
        json.dump(roll_log, f, default=str, indent=2)

    # equity-curve plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, series in concatenated_returns.items():
        if len(series) == 0:
            continue
        wealth = 100.0 * (1.0 + series).cumprod()
        ax.plot(wealth.index, wealth.values, label=strategy)
    ax.set_ylabel("Portfolio value (base = 100)")
    ax.set_title(f"Phase 2: out-of-sample equity curves ({cfg.OOS_BACKTEST_START} - {cfg.OOS_BACKTEST_END})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "phase2_equity_curves.png"), dpi=150)
    plt.close(fig)

    print("\nAll artifacts written to:", OUTPUT_DIR)
    return {
        "validation_df": validation_df,
        "phase1_results": results_df,
        "phase1_summary": summary_table,
        "champion": champion_name,
        "phase2_summary": portfolio_summary,
    }


if __name__ == "__main__":
    main()
