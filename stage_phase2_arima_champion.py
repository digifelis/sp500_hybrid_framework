"""
stage_phase2_arima_champion.py
================================
Re-runs Phase 2 (rolling-window portfolio backtest) using ARIMA -- not
RIME-LSTM -- as the Phase-1 asset-ranking signal, holding every other part
of the pipeline (Dandelion Optimizer, Markowitz, equal-weight, rolling-
window design, evaluation) exactly fixed. This directly answers Reviewers
1 and 3's shared objection: Table 4 shows ARIMA has the lowest forecasting
error, yet the original submission used RIME-LSTM (the better of the two
*tuned* models, but not the overall best forecaster) to rank assets. This
script produces the missing ARIMA-ranked comparison so the paper can state
empirically whether Phase 2 results are driven by which forecaster ranks
assets, by the DO/Markowitz weighting stage, or by their interaction.

This run also switches on full-weight-vector logging (engine.py's roll_log
now records do_weights/mkw_weights/eq_weights, not just the single top
weight), which this script uses to additionally compute the Herfindahl-
Hirschman Index (HHI) and effective number of holdings for every roll and
every strategy -- Reviewers 1 and 3's second shared request -- so ONE rerun
answers both concerns rather than two separate ones.

Usage:
    python stage_phase2_arima_champion.py

Output (./phase2_arima_champion/):
    phase2_portfolio_backtest_summary.csv   -- ARIMA-ranked Table 5 variant
    phase2_equity_curves.png
    phase2_roll_log.json                     -- now includes full weight vectors
    phase2_concentration_summary.csv         -- HHI / effective-N per roll, per strategy
    daily_returns_*.csv
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from config import set_global_seed
set_global_seed(cfg.GLOBAL_SEED)

from data.data_pipeline import load_real_universe, load_real_index
from utils.technical_indicators import build_feature_frame
from models.sequence_models import SequenceModelConfig
from backtest.engine import run_rolling_portfolio_backtest

# ============================================================
# Settings -- adjust REAL_DATA_DIR / TICKERS / TOP_K if your config.py
# does not already define these for the S&P 500 run (they should, if this
# is dropped into the same project folder used for the original submission).
# ============================================================
REAL_DATA_DIR = "sp500_real_data"
TICKERS = cfg.BIST_TICKERS          # NOTE: reused variable name; holds the 25 S&P 500 tickers in this project's config.py
TOP_K = 15

OUT_DIR = os.environ.get("ARIMA_OUT_DIR", "phase2_arima_champion")
os.makedirs(OUT_DIR, exist_ok=True)
n_features = len(cfg.FEATURE_COLUMNS)


def herfindahl_summary(roll_log: list) -> pd.DataFrame:
    """HHI = sum(w_i^2) in [1/N, 1]; effective number of holdings = 1/HHI.
    Computed per roll for DO, Markowitz, and equal-weight from the full
    weight vectors now recorded in roll_log."""
    rows = []
    for r in roll_log:
        for strat, key in [("DandelionOptimizer", "do_weights"), ("Markowitz", "mkw_weights"),
                            ("EqualWeight", "eq_weights")]:
            w = np.asarray(r[key])
            hhi = float(np.sum(w ** 2))
            rows.append({"roll": r["roll"], "date": r["date"], "strategy": strat,
                         "n_assets": len(w), "max_weight": float(np.max(w)),
                         "HHI": hhi, "effective_N": 1.0 / hhi})
    return pd.DataFrame(rows)


def main():
    print("=" * 70, "\nPhase 2 rerun with ARIMA as the asset-ranking signal\n", "=" * 70)
    universe = load_real_universe(REAL_DATA_DIR, TICKERS, start=cfg.HISTORY_START, end=cfg.HISTORY_END)
    index_level = load_real_index(REAL_DATA_DIR, cfg.BENCHMARK_INDEX, start=cfg.HISTORY_START, end=cfg.HISTORY_END)
    feature_frames = {tkr: build_feature_frame(df, cfg) for tkr, df in universe.items()}

    # forecaster_cfg is unused when ranking_method="arima" but the function
    # signature still requires one; pass a harmless placeholder.
    placeholder_cfg = SequenceModelConfig(input_size=n_features, horizon=cfg.FORECAST_HORIZON,
                                           num_layers=1, hidden_size=16, learning_rate=1e-3,
                                           dropout=0.1, batch_size=32, epochs=5)

    concatenated, summary, roll_log = run_rolling_portfolio_backtest(
        feature_frames, universe, index_level, cfg, forecaster_cfg=placeholder_cfg,
        top_k=TOP_K, seed=cfg.GLOBAL_SEED, verbose=True,
        ranking_method="arima",
    )
    print("\nPhase 2 (ARIMA-ranked) out-of-sample performance summary:\n",
          summary.to_string(float_format=lambda x: f"{x:.4f}"))
    summary.to_csv(os.path.join(OUT_DIR, "phase2_portfolio_backtest_summary.csv"))
    with open(os.path.join(OUT_DIR, "phase2_roll_log.json"), "w") as f:
        json.dump(roll_log, f, default=str, indent=2)
    for name, series in concatenated.items():
        series.to_csv(os.path.join(OUT_DIR, f"daily_returns_{name}.csv"))

    conc = herfindahl_summary(roll_log)
    conc.to_csv(os.path.join(OUT_DIR, "phase2_concentration_summary.csv"), index=False)
    print("\nConcentration summary (mean across 18 rolls):\n",
          conc.groupby("strategy")[["max_weight", "HHI", "effective_N"]].mean().to_string(float_format=lambda x: f"{x:.4f}"))

    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, series in concatenated.items():
        if len(series) == 0:
            continue
        wealth = 100.0 * (1.0 + series).cumprod()
        ax.plot(wealth.index, wealth.values, label=strategy)
    ax.set_ylabel("Portfolio value (base = 100)")
    ax.set_title(f"Phase 2 (ARIMA-ranked): out-of-sample equity curves ({cfg.OOS_BACKTEST_START} - {cfg.OOS_BACKTEST_END})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "phase2_equity_curves.png"), dpi=150)
    plt.close(fig)

    print("\nAll artifacts written to:", OUT_DIR)
    print("Send back everything in this folder, plus a re-run of phase2_rime_champion "
          "(the RIME-ranked run) using the same updated engine.py, since its roll_log "
          "now also carries full weight vectors needed for the HHI comparison table.")


if __name__ == "__main__":
    main()
