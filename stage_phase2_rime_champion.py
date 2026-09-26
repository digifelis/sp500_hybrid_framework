"""
stage_phase2_rime_champion.py
===============================
Re-runs ONLY Phase 2 (rolling-window portfolio backtest), swapping the
forward-return signal from MGO-LSTM to RIME-LSTM, per the corrected
champion-selection rule: the Diebold-Mariano test (Table 6) is the
scientifically appropriate comparison (paired, formal significance test)
rather than the raw mean-RMSE ranking originally specified in Section 3.6,
and it found RIME-LSTM significantly more accurate than MGO-LSTM
(p ~ 3.7e-12). This does NOT re-run RIME/MGO hyperparameter search (already
locked) or Phase 1's forecasting comparison (Table 4/6 -- see
stage_significance_tests.py) -- only Phase 2's asset-ranking-and-portfolio
loop, using RIME-LSTM's already-tuned configuration (Table 3).

Usage:
    python stage_phase2_rime_champion.py

Output (./phase2_rime_champion/):
    phase2_portfolio_backtest_summary.csv   -- Table 5 replacement
    phase2_equity_curves.png                -- Figure 3 replacement
    phase2_roll_log.json
    daily_returns_*.csv                      -- for a matching bootstrap rerun
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
# LOCKED-IN settings -- must match the run already in the manuscript
# ============================================================
REAL_DATA_DIR = "sp500_real_data"
TICKERS = cfg.BIST_TICKERS
TOP_K = 15

# RIME-LSTM's locked hyperparameters (Table 3) -- now the Phase-2 champion
RIME_CFG_PARAMS = dict(hidden_size=104, learning_rate=0.00500, dropout=0.164, batch_size=35, epochs=62)

OUT_DIR = os.environ.get("RIME_OUT_DIR", "phase2_rime_champion")
os.makedirs(OUT_DIR, exist_ok=True)
n_features = len(cfg.FEATURE_COLUMNS)


def herfindahl_summary(roll_log: list) -> pd.DataFrame:
    """Summarize concentration from the full logged weight vectors."""
    rows = []
    for r in roll_log:
        for strat, key in [("DandelionOptimizer", "do_weights"),
                           ("Markowitz", "mkw_weights"),
                           ("EqualWeight", "eq_weights")]:
            w = np.asarray(r[key], dtype=float)
            hhi = float(np.sum(w ** 2))
            rows.append({"roll": r["roll"], "date": r["date"], "strategy": strat,
                         "n_assets": len(w), "max_weight": float(np.max(w)),
                         "HHI": hhi, "effective_N": 1.0 / hhi})
    return pd.DataFrame(rows)


def main():
    print("=" * 70, "\nPhase 2 rerun with RIME-LSTM as champion (corrected per DM test)\n", "=" * 70)
    universe = load_real_universe(REAL_DATA_DIR, TICKERS, start=cfg.HISTORY_START, end=cfg.HISTORY_END)
    index_level = load_real_index(REAL_DATA_DIR, cfg.BENCHMARK_INDEX, start=cfg.HISTORY_START, end=cfg.HISTORY_END)
    feature_frames = {tkr: build_feature_frame(df, cfg) for tkr, df in universe.items()}

    champion_cfg = SequenceModelConfig(input_size=n_features, horizon=cfg.FORECAST_HORIZON, num_layers=2, **RIME_CFG_PARAMS)

    concatenated, summary, roll_log = run_rolling_portfolio_backtest(
        feature_frames, universe, index_level, cfg, forecaster_cfg=champion_cfg,
        top_k=TOP_K, seed=cfg.GLOBAL_SEED, verbose=True,
    )
    print("\nPhase 2 (RIME-LSTM champion) out-of-sample performance summary:\n",
          summary.to_string(float_format=lambda x: f"{x:.4f}"))
    summary.to_csv(os.path.join(OUT_DIR, "phase2_portfolio_backtest_summary.csv"))
    with open(os.path.join(OUT_DIR, "phase2_roll_log.json"), "w") as f:
        json.dump(roll_log, f, default=str, indent=2)
    for name, series in concatenated.items():
        series.to_csv(os.path.join(OUT_DIR, f"daily_returns_{name}.csv"))

    conc = herfindahl_summary(roll_log)
    conc.to_csv(os.path.join(OUT_DIR, "phase2_concentration_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, series in concatenated.items():
        if len(series) == 0:
            continue
        wealth = 100.0 * (1.0 + series).cumprod()
        ax.plot(wealth.index, wealth.values, label=strategy)
    ax.set_ylabel("Portfolio value (base = 100)")
    ax.set_title(f"Phase 2 (RIME-LSTM champion): out-of-sample equity curves ({cfg.OOS_BACKTEST_START} - {cfg.OOS_BACKTEST_END})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "phase2_equity_curves.png"), dpi=150)
    plt.close(fig)

    print("\nAll artifacts written to:", OUT_DIR)
    print("\nSend back: phase2_portfolio_backtest_summary.csv, phase2_equity_curves.png, "
          "phase2_roll_log.json, and the 4 daily_returns_*.csv files.")


if __name__ == "__main__":
    main()
