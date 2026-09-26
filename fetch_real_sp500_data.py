"""
fetch_real_bist_data.py
========================
Run this on a machine with normal internet access (NOT inside the sandboxed
environment used to build the framework/manuscript -- that sandbox's network
is locked to package repositories and cannot reach Yahoo Finance).

Usage:
    pip install yfinance pandas --upgrade
    python fetch_real_bist_data.py

Output:
    ./bist_real_data/{TICKER}.csv   -- one file per constituent, e.g. THYAO.csv
    ./bist_real_data/XU100.csv      -- the BIST 100 index itself

Each CSV has columns Date, Open, High, Low, Close, Volume -- exactly the
schema expected by the framework's data/data_pipeline.py:load_real_universe().
Once you have this folder, either:
  (a) upload the whole `bist_real_data` folder (or a zip of it) back to this
      chat, or
  (b) run main.py yourself with:
          from data.data_pipeline import load_real_universe
          universe = load_real_universe(csv_dir="bist_real_data",
                                         tickers=cfg.BIST_TICKERS,
                                         start=cfg.HISTORY_START, end=cfg.HISTORY_END)
      in place of generate_synthetic_universe(...), and send back the
      resulting outputs/ folder (CSVs + PNGs) for me to fold into the article.
Option (a) is simplest -- I'll do the run and the article update myself.
"""

import time
import pandas as pd
import yfinance as yf

# Same 25-ticker universe as config.py -- keep in sync if you edit config.py
SP500_TICKERS = [
    "MSFT", "AAPL", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "JPM", "BAC", "V",
    "JNJ", "UNH", "LLY", "XOM", "CVX",
    "PG", "WMT", "KO", "PEP", "COST",
    "HD", "MCD", "GE", "CAT", "HON"
]
INDEX_TICKER_YF = "SPY"          # SP500 index on Yahoo Finance
START_DATE = "2021-01-04"             # matches config.HISTORY_START
END_DATE = "2026-07-11"               # today; adjust to match config.HISTORY_END when you re-run

OUT_DIR = "sp500_real_data"


def fetch_one(yf_ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(yf_ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {yf_ticker} -- check the ticker is still listed on Yahoo Finance.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Adj Close": "AdjClose"})
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    return df


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    failures = []
    for tkr in SP500_TICKERS:
        yf_ticker = f"{tkr}"
        try:
            df = fetch_one(yf_ticker, START_DATE, END_DATE)
            df.to_csv(os.path.join(OUT_DIR, f"{tkr}.csv"))
            print(f"  OK  {tkr:8s} -> {len(df):5d} rows  ({df.index.min().date()} to {df.index.max().date()})")
        except Exception as e:
            print(f"  FAIL {tkr:8s} -> {e}")
            failures.append(tkr)
        time.sleep(0.5)  # be polite to the API

    try:
        idx = fetch_one(INDEX_TICKER_YF, START_DATE, END_DATE)
        idx.to_csv(os.path.join(OUT_DIR, "SPY.csv"))
        print(f"  OK  XU100    -> {len(idx):5d} rows  ({idx.index.min().date()} to {idx.index.max().date()})")
    except Exception as e:
        print(f"  FAIL SPY -> {e}")
        failures.append("SPY")

    print("\nDone.", f"{len(SP500_TICKERS) + 1 - len(failures)}/{len(SP500_TICKERS) + 1} series fetched successfully.")
    if failures:
        print("Failed tickers (check manually on finance.yahoo.com, symbol may have changed):", failures)
    print(f"CSVs written to ./{OUT_DIR}/  -- zip this folder and send it back, or point load_real_universe() at it.")


if __name__ == "__main__":
    main()
