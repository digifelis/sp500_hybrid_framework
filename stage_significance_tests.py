"""
stage_significance_tests.py
=============================
Run this LOCALLY (same machine/repo as your locked-in production run) to
produce Section 4.5's statistical tests. It reuses your FINAL, already-
decided hyperparameters below -- it does NOT re-run RIME/MGO search, so
it's much faster than main.py (no metaheuristic search, ~5x fewer model
trainings, no re-optimizing DO/Markowitz search budgets).

IMPORTANT: the constants below (RIME_CFG, MGO_CFG, BASELINE_HIDDEN/EPOCHS,
TOP_K) must match the run whose numbers are already in the manuscript.
They are currently set to match your locked run's console log exactly.
If you changed anything in main.py since then, update these to match.

Usage:
    python stage_significance_tests.py

Output (written to ./significance_results/):
    dm_test_results.csv        -- pairwise Diebold-Mariano test vs. champion
    bootstrap_sharpe_results.csv -- pairwise Sharpe-difference bootstrap vs. DO
"""

import os
import numpy as np
import pandas as pd

import config as cfg
from config import set_global_seed
set_global_seed(cfg.GLOBAL_SEED)

from data.data_pipeline import load_real_universe, load_real_index, build_sequences, chronological_split
from utils.technical_indicators import build_feature_frame, inverse_min_max
from utils.statistical_tests import diebold_mariano_test, bootstrap_sharpe_difference
from models.sequence_models import SequenceModelConfig, build_model, train_forecaster, predict
from models.arima_baseline import fit_predict_arima, select_best_order
from backtest.engine import run_rolling_portfolio_backtest

# ============================================================
# LOCKED-IN settings -- must match the run already in the manuscript
# ============================================================
REAL_DATA_DIR = "sp500_real_data"
TICKERS = cfg.BIST_TICKERS
TOP_K = 15

RIME_CFG_PARAMS = dict(hidden_size=104, learning_rate=0.00500, dropout=0.164, batch_size=35, epochs=62)
MGO_CFG_PARAMS = dict(hidden_size=128, learning_rate=0.00488, dropout=0.124, batch_size=47, epochs=72)
BASELINE_HIDDEN, BASELINE_EPOCHS = 64, 50   # must match main.py's production-profile baseline LSTM/GRU

OUT_DIR = "significance_results"
os.makedirs(OUT_DIR, exist_ok=True)
n_features = len(cfg.FEATURE_COLUMNS)


def build_configs():
    return {
        "RIME-LSTM": SequenceModelConfig(input_size=n_features, horizon=cfg.FORECAST_HORIZON, num_layers=2, **RIME_CFG_PARAMS),
        "MGO-LSTM": SequenceModelConfig(input_size=n_features, horizon=cfg.FORECAST_HORIZON, num_layers=2, **MGO_CFG_PARAMS),
        "LSTM": SequenceModelConfig(input_size=n_features, hidden_size=BASELINE_HIDDEN, num_layers=1, dropout=0.2,
                                     horizon=cfg.FORECAST_HORIZON, learning_rate=1e-3, batch_size=32, epochs=BASELINE_EPOCHS),
        "GRU": SequenceModelConfig(input_size=n_features, hidden_size=BASELINE_HIDDEN, num_layers=1, dropout=0.2,
                                    horizon=cfg.FORECAST_HORIZON, learning_rate=1e-3, batch_size=32, epochs=BASELINE_EPOCHS),
    }


def collect_per_origin_errors():
    """For every ticker and every model, compute ONE scalar error per test-set
    forecast origin (mean error across the 5-day horizon, price scale), then
    pool across tickers into one long-format DataFrame: columns
    [ticker, origin_idx, model, error]. This is what the DM test needs
    (aggregate RMSE alone isn't enough -- DM needs the paired per-origin series)."""
    configs = build_configs()
    universe = load_real_universe(REAL_DATA_DIR, TICKERS, start=cfg.HISTORY_START, end=cfg.HISTORY_END)
    rows = []

    for tkr in TICKERS:
        ff = build_feature_frame(universe[tkr], cfg)
        X, y, origin_idx = build_sequences(ff, cfg.FEATURE_COLUMNS, "close_norm", cfg.SEQUENCE_LENGTH, cfg.FORECAST_HORIZON)
        tr, va, te = chronological_split(len(X), cfg.TRAIN_FRACTION, cfg.VAL_FRACTION)
        close_min, close_max = ff.attrs["close_min"], ff.attrs["close_max"]

        for model_name, model_cfg in configs.items():
            model_type = "gru" if model_name == "GRU" else "lstm"
            model = build_model(model_type, model_cfg)
            train_forecaster(model, X[tr], y[tr], X[va], y[va], model_cfg, early_stopping_patience=5)
            y_pred_norm = predict(model, X[te])
            y_true_price = inverse_min_max(y[te], close_min, close_max)
            y_pred_price = inverse_min_max(y_pred_norm, close_min, close_max)
            # RMS error over the 5-day horizon per origin (NOT mean-then-square,
            # which lets same-origin errors of opposite sign cancel and would
            # test a different, non-comparable quantity than the RMSE already
            # reported in Table 3). Squaring this inside diebold_mariano_test
            # recovers the standard per-origin mean-squared-error.
            per_origin_error = np.sqrt(((y_true_price - y_pred_price) ** 2).mean(axis=1))
            # Per-origin MAPE (%) over the same 5-day horizon, so Table 4 (RMSE
            # + MAPE) and Table 6 (DM test on RMS error) come from the SAME
            # trained-model run -- no run-to-run training variance between the
            # descriptive table and the significance test.
            per_origin_mape = (np.abs((y_true_price - y_pred_price) / np.clip(np.abs(y_true_price), 1e-6, None))).mean(axis=1) * 100.0
            for oi, err, mape_v in zip(origin_idx[te], per_origin_error, per_origin_mape):
                rows.append({"ticker": tkr, "origin_idx": oi, "model": model_name, "error": err, "mape_pct": mape_v})

        # ARIMA: same per-origin walk-forward as backtest/engine.py, same step
        close_series = ff["close"].values
        n_train_val = len(tr) + len(va) + cfg.SEQUENCE_LENGTH
        order = select_best_order(close_series[:n_train_val])
        test_origin = origin_idx[te]
        for oi in test_origin:
            history = close_series[: oi + 1]
            fc = fit_predict_arima(history, cfg.FORECAST_HORIZON, order=order)
            actual = close_series[oi + 1: oi + 1 + cfg.FORECAST_HORIZON]
            if len(actual) == cfg.FORECAST_HORIZON:
                arima_rmse = float(np.sqrt(np.mean((actual - fc) ** 2)))
                arima_mape = float(np.mean(np.abs((actual - fc) / np.clip(np.abs(actual), 1e-6, None))) * 100.0)
                rows.append({"ticker": tkr, "origin_idx": oi, "model": "ARIMA",
                             "error": arima_rmse, "mape_pct": arima_mape})

        print(f"  [{tkr}] per-origin errors + MAPE collected for all 5 models")

    return pd.DataFrame(rows)


def run_dm_tests(long_df: pd.DataFrame, champion: str = "MGO-LSTM"):
    """Pools per-origin errors across the whole 25-ticker panel and runs DM
    test of champion vs. each other model."""
    results = []
    champ_errors = long_df[long_df.model == champion].sort_values(["ticker", "origin_idx"])["error"].values
    for model_name in ["ARIMA", "GRU", "LSTM", "RIME-LSTM"]:
        if model_name == champion:
            continue
        other = long_df[long_df.model == model_name].sort_values(["ticker", "origin_idx"])["error"].values
        n = min(len(champ_errors), len(other))
        res = diebold_mariano_test(champ_errors[:n], other[:n], h=cfg.FORECAST_HORIZON, loss="squared")
        res["champion"] = champion
        res["benchmark"] = model_name
        res["interpretation"] = (f"{champion} significantly better" if res["significant_5pct"] and res["favors"] == "A"
                                  else f"{model_name} significantly better" if res["significant_5pct"] else "no significant difference")
        results.append(res)
        print(f"  DM {champion} vs {model_name}: stat={res['dm_statistic']:.3f} p={res['p_value']:.4f} -> {res['interpretation']}")
    return pd.DataFrame(results)


def run_portfolio_bootstrap():
    universe = load_real_universe(REAL_DATA_DIR, TICKERS, start=cfg.HISTORY_START, end=cfg.HISTORY_END)
    index_level = load_real_index(REAL_DATA_DIR, cfg.BENCHMARK_INDEX, start=cfg.HISTORY_START, end=cfg.HISTORY_END)
    feature_frames = {tkr: build_feature_frame(df, cfg) for tkr, df in universe.items()}
    champion_cfg = SequenceModelConfig(input_size=n_features, horizon=cfg.FORECAST_HORIZON, num_layers=2, **MGO_CFG_PARAMS)

    concatenated, summary, roll_log = run_rolling_portfolio_backtest(
        feature_frames, universe, index_level, cfg, forecaster_cfg=champion_cfg,
        top_k=TOP_K, seed=cfg.GLOBAL_SEED, verbose=True,
    )
    for name, series in concatenated.items():
        series.to_csv(os.path.join(OUT_DIR, f"daily_returns_{name}.csv"))

    do_returns = concatenated["DandelionOptimizer"]
    results = []
    for bench in ["Markowitz", "EqualWeight", "BuyAndHoldIndex"]:
        bench_returns = concatenated[bench].reindex(do_returns.index).dropna()
        aligned_do = do_returns.reindex(bench_returns.index)
        res = bootstrap_sharpe_difference(aligned_do.values, bench_returns.values,
                                           cfg.RISK_FREE_RATE_ANNUAL, n_boot=2000, seed=cfg.GLOBAL_SEED)
        res["strategy_a"] = "DandelionOptimizer"
        res["strategy_b"] = bench
        res["interpretation"] = ("DO significantly better" if res["significant_5pct"] and res["observed_diff"] > 0
                                  else f"{bench} significantly better" if res["significant_5pct"] else "no significant difference")
        results.append(res)
        print(f"  Bootstrap DO vs {bench}: diff={res['observed_diff']:.3f} "
              f"95%CI=[{res['ci_95_lower']:.3f},{res['ci_95_upper']:.3f}] p={res['p_value']:.4f} -> {res['interpretation']}")
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 70, "\nCollecting per-origin forecast errors + MAPE (25 tickers x 5 models)\n", "=" * 70)
    long_df = collect_per_origin_errors()
    long_df.to_csv(os.path.join(OUT_DIR, "per_origin_errors_raw.csv"), index=False)

    print("\n" + "=" * 70, "\nDense Table 4 (RMSE + MAPE), all 5 models, every forecast origin\n", "=" * 70)
    dense_table4 = (long_df.groupby(["ticker", "model"])
                     .apply(lambda g: pd.Series({"RMSE_price": np.sqrt(np.mean(g["error"] ** 2)),
                                                  "MAPE_price_%": g["mape_pct"].mean()}))
                     .reset_index())
    dense_table4.to_csv(os.path.join(OUT_DIR, "phase1_forecasting_benchmark_DENSE_raw.csv"), index=False)
    dense_summary = dense_table4.groupby("model")[["RMSE_price", "MAPE_price_%"]].mean().sort_values("RMSE_price")
    dense_summary.to_csv(os.path.join(OUT_DIR, "phase1_forecasting_benchmark_DENSE_summary.csv"))
    print(dense_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 70, "\nDiebold-Mariano tests (champion = MGO-LSTM)\n", "=" * 70)
    dm_df = run_dm_tests(long_df, champion="MGO-LSTM")
    dm_df.to_csv(os.path.join(OUT_DIR, "dm_test_results.csv"), index=False)

    print("\n" + "=" * 70, "\nStationary bootstrap Sharpe-ratio-difference tests (DO vs. benchmarks)\n", "=" * 70)
    boot_df = run_portfolio_bootstrap()
    boot_df.to_csv(os.path.join(OUT_DIR, "bootstrap_sharpe_results.csv"), index=False)

    print("\nAll significance-test artifacts written to:", OUT_DIR)
