"""
per_ticker_dm_test.py
=======================
Reviewers 1 and 3 both flagged that the paper's pooled Diebold-Mariano test
(Table 6) combines forecast errors across 25 stocks without discussing
cross-sectional/serial dependence. This script complements -- not replaces
-- the pooled test by running an independent DM test PER TICKER, so the
paper can report how many of the 25 individual stocks show a significant
result in each direction, rather than relying on the pooled statistic
alone. Run this from the same folder as the framework, with
per_origin_errors_raw.csv (produced by stage_significance_tests.py)
present.

Usage:
    python per_ticker_dm_test.py

Output:
    per_ticker_dm_results.csv   -- one row per (ticker, comparison)
    per_ticker_dm_summary.csv   -- count of significant/insignificant results per comparison
"""

import pandas as pd
import numpy as np
from utils.statistical_tests import diebold_mariano_test

CHAMPION = "RIME-LSTM"   # the paper's actual Phase-1 champion; compare it against each alternative
BENCHMARKS = ["ARIMA", "GRU", "LSTM", "MGO-LSTM"]

df = pd.read_csv("per_origin_errors_raw.csv")
tickers = sorted(df["ticker"].unique())

rows = []
for tkr in tickers:
    champ_err = df[(df.ticker == tkr) & (df.model == CHAMPION)].sort_values("origin_idx")["error"].values
    for bench in BENCHMARKS:
        bench_err = df[(df.ticker == tkr) & (df.model == bench)].sort_values("origin_idx")["error"].values
        n = min(len(champ_err), len(bench_err))
        if n < 10:
            continue
        res = diebold_mariano_test(champ_err[:n], bench_err[:n], h=5, loss="squared")
        res["ticker"] = tkr
        res["benchmark"] = bench
        rows.append(res)

results = pd.DataFrame(rows)
results.to_csv("per_ticker_dm_results.csv", index=False)

summary_rows = []
for bench in BENCHMARKS:
    sub = results[results.benchmark == bench]
    n_champion_better_sig = int(((sub.favors == "A") & (sub.significant_5pct)).sum())
    n_benchmark_better_sig = int(((sub.favors == "B") & (sub.significant_5pct)).sum())
    n_not_sig = int((~sub.significant_5pct).sum())
    summary_rows.append({
        "benchmark": bench, "n_tickers": len(sub),
        f"{CHAMPION}_significantly_better": n_champion_better_sig,
        f"{bench}_significantly_better": n_benchmark_better_sig,
        "not_significant": n_not_sig,
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv("per_ticker_dm_summary.csv", index=False)
print(summary.to_string(index=False))
print("\nFull per-ticker results written to per_ticker_dm_results.csv")
