# S&P 500 Hybrid Forecasting and Portfolio Framework

## Scope

This repository contains the code and archived artifacts for the study comparing LSTM-based price forecasting with ARIMA, RIME/MGO hyperparameter optimization, and Dandelion-Optimizer-based portfolio weighting on 25 large-cap S&P 500 constituents. Forecast signals are used to select assets in a rolling-window out-of-sample portfolio experiment; Dandelion Optimizer, classical Markowitz weighting, equal weighting, and the SPY buy-and-hold benchmark are evaluated. The repository corresponds to the submitted Financial Innovation manuscript and its reviewer-response reruns. The included raw snapshot contains 25 constituent CSV files plus `SPY.csv`.

## Repository layout

```text
config.py                         central settings and seeds
main.py                           main forecasting and portfolio pipeline
data/                             data loading and sequence construction
utils/                            indicators, metrics, and statistical tests
models/                           LSTM/GRU, ARIMA, and hyperparameter tuning
optimizers/                       RIME, MGO, and Dandelion implementations
portfolio/                        Markowitz and Dandelion portfolio weighting
backtest/                         forecasting and rolling portfolio engine
stage_significance_tests.py       dense forecast errors and statistical tests
stage_phase2_rime_champion.py     RIME-ranked Phase 2 rerun
stage_phase2_arima_champion.py    ARIMA-ranked Phase 2 rerun
per_ticker_dm_test.py             ticker-level DM tests
sp500_real_data/                  source-compatible raw-data location used by scripts
data_raw/sp500_real_data/         archived raw-data snapshot
results/                          organized reproduction artifacts
```

`data_raw/SHA256SUMS.txt` records SHA-256 hashes for the archived raw CSV snapshot. The scripts currently read `sp500_real_data/` relative to the project root; the identical archival copy is also stored under `data_raw/sp500_real_data/`.

## Environment

The verified RIME and ARIMA outputs in this package were generated with:

```text
Python executable: C:\Users\bes-t\anaconda3\python.exe
Python: 3.11.5
PyTorch: 2.6.0+cu124
CUDA reported by PyTorch: 12.4
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
CUDA available: True
```

The exact package snapshot is in [requirements_frozen.txt](requirements_frozen.txt). Reproduction in a fresh isolated environment should use:

```powershell
python -m venv sp500_repro_env
.\sp500_repro_env\Scripts\Activate.ps1
python -m pip install -r requirements_frozen.txt
```

On Windows, use a Python build compatible with the frozen wheels. Do not use a shared or unrelated environment: package-version drift was observed during this project and can change numerical results or even change whether a run uses CPU or CUDA. The later `sp500_env` created during troubleshooting uses Python 3.12 and CPU-only `torch==2.7.0`; it was not used to generate the verified artifacts in `results/`.

## Configuration and fixed seeds

- `config.py` defines `GLOBAL_SEED = 42`.
- The active universe is defined by `config.BIST_TICKERS`; in the current configuration these are 25 S&P 500 tickers despite the legacy variable name.
- The benchmark is `SPY`.
- The history window is `2021-01-04` through `2026-07-10`.
- The rolling out-of-sample window is `2025-01-01` through `2026-06-30`.
- `config.configure_torch_runtime()` disables cuDNN benchmarking and enables `torch.backends.cudnn.deterministic = True`; it also sets the global Python/NumPy/PyTorch seed.

## How to reproduce

Run commands from the repository root. The scripts expect `sp500_real_data/` to exist and contain the 25 constituent files plus `SPY.csv`.

### 1. Main pipeline

```powershell
python .\main.py
```

`main.py` uses the production profile currently present in the source and writes to `outputs/`:

- `optimizer_validation_benchmarks.csv`
- `phase1_forecasting_benchmark_raw.csv`
- `phase1_forecasting_benchmark_summary.csv`
- `phase1_convergence_curves.png`
- `phase2_portfolio_backtest_summary.csv`
- `phase2_roll_log.json`
- `phase2_equity_curves.png`

These artifacts are archived under `results/main_run/`. The source documentation describes a production run as potentially multi-hour; an exact wall-clock duration was not recorded in the archived output.

### 2. Dense significance tests

```powershell
python .\stage_significance_tests.py
```

This writes to `significance_results/`:

- `per_origin_errors_raw.csv`
- `phase1_forecasting_benchmark_DENSE_raw.csv`
- `phase1_forecasting_benchmark_DENSE_summary.csv`
- `dm_test_results.csv`
- `SUPERSEDED_bootstrap_sharpe_results.csv` — not used; this is a historical intermediate calculation and does not match Table 7.
- `daily_returns_BuyAndHoldIndex.csv`
- `daily_returns_DandelionOptimizer.csv`
- `daily_returns_EqualWeight.csv`
- `daily_returns_Markowitz.csv`

These are archived under `results/significance_tests/`. An exact wall-clock duration was not recorded.

### 3. Per-ticker DM test

`per_ticker_dm_test.py` requires `per_origin_errors_raw.csv` in the current working directory. After running the previous step, use:

```powershell
Copy-Item .\significance_results\per_origin_errors_raw.csv .\per_origin_errors_raw.csv -Force
python .\per_ticker_dm_test.py
```

It writes `per_ticker_dm_results.csv` and `per_ticker_dm_summary.csv`; archived copies are under `results/per_ticker_dm/`.

### 4. RIME-ranked Phase 2

```powershell
python .\stage_phase2_rime_champion.py
```

The default output directory is `phase2_rime_champion/`. The script writes `phase2_portfolio_backtest_summary.csv`, `phase2_concentration_summary.csv`, `phase2_roll_log.json`, `phase2_equity_curves.png`, and the four `daily_returns_*.csv` files. The concentration file contains HHI and effective number of holdings calculated from the full logged weight vectors. The archived verified output is under `results/phase2_rime_verified/`.

### 5. ARIMA-ranked Phase 2

```powershell
python .\stage_phase2_arima_champion.py
```

The default output directory is `phase2_arima_champion/`. It writes the same output set as the RIME rerun, including `phase2_concentration_summary.csv`. The archived verified output is under `results/phase2_arima_verified/`.

For a two-run determinism check without overwriting the default directories, the scripts support:

```powershell
$env:RIME_OUT_DIR = "rime_run1"
python .\stage_phase2_rime_champion.py
$env:RIME_OUT_DIR = "rime_run2"
python .\stage_phase2_rime_champion.py
$env:ARIMA_OUT_DIR = "arima_run1"
python .\stage_phase2_arima_champion.py
$env:ARIMA_OUT_DIR = "arima_run2"
python .\stage_phase2_arima_champion.py
```

The verified runs produced 18 rolling periods each. The archived Run 1 files are retained in `results/`; the Run 1/Run 2 comparisons are documented in the two `DETERMINISM_CHECK.md` files. Every listed CSV, JSON, and PNG file compared byte-for-byte identical in the corresponding pair.

## Determinism and reproducibility notes

Two reproducibility issues occurred during the study workflow:

1. GPU training was not initially deterministic. The current `config.py` addresses this in `configure_torch_runtime()` by setting `cudnn.benchmark = False`, `cudnn.deterministic = True`, and fixed seeds for Python, NumPy, and PyTorch. GPU/driver/library changes can still affect numerical behavior, so the exact environment and hardware should be retained.
2. Package-version drift occurred in a shared environment. For that reason, use `requirements_frozen.txt` in a fresh isolated environment. The verified artifacts were generated in the GPU-enabled environment listed above.

The exact byte-level comparison results are recorded in `results/phase2_rime_verified/DETERMINISM_CHECK.md` and `results/phase2_arima_verified/DETERMINISM_CHECK.md`.

## Mapping to the manuscript

| Manuscript item | Reproduction artifact |
|---|---|
| Tables 1–4 and initial Phase 2 result | `results/main_run/phase1_forecasting_benchmark_summary.csv`, `phase1_forecasting_benchmark_raw.csv`, `phase2_portfolio_backtest_summary.csv`, `phase2_roll_log.json` |
| Phase 1 convergence figure | `results/main_run/phase1_convergence_curves.png` |
| Dense Table 4 / MAPE | `results/significance_tests/phase1_forecasting_benchmark_DENSE_raw.csv`, `phase1_forecasting_benchmark_DENSE_summary.csv` |
| Table 6 pooled DM test | `results/significance_tests/dm_test_results.csv` |
| Table 7 (RIME-ranked) stationary bootstrap | `results/phase2_rime_verified/daily_returns_*.csv` |
| Table 7 (ARIMA-ranked) stationary bootstrap | `results/phase2_arima_verified/daily_returns_*.csv` |
| Table 6b ticker-level DM test | `results/per_ticker_dm/per_ticker_dm_results.csv`, `per_ticker_dm_summary.csv` |
| Table 5 / Figure 4a, RIME-ranked Phase 2 | `results/phase2_rime_verified/phase2_portfolio_backtest_summary.csv`, `phase2_equity_curves.png`, `phase2_roll_log.json` |
| Table 5b / Figure 4b, ARIMA-ranked Phase 2 | `results/phase2_arima_verified/phase2_portfolio_backtest_summary.csv`, `phase2_equity_curves.png`, `phase2_roll_log.json` |
| Table 5c, concentration / HHI | `results/phase2_rime_verified/phase2_concentration_summary.csv` and `results/phase2_arima_verified/phase2_concentration_summary.csv` |
| Per-roll asset selections and weights | `selected_assets`, `do_weights`, `mkw_weights`, and `eq_weights` fields in each verified `phase2_roll_log.json` |

The manuscript numbering above follows the requested reviewer-response mapping. If the final accepted manuscript uses different figure/table numbers, update this table before submission.

## Data acquisition

`fetch_real_sp500_data.py` is included as the acquisition script. The repository also contains the frozen snapshot under `data_raw/sp500_real_data/` and the source-compatible `sp500_real_data/` directory used by the current scripts. Do not overwrite the snapshot when reproducing the reported results; use a separate directory for newly acquired data.

## Author contributions

Assoc. Prof. Dr. Mansur BEŞTAŞ
Bitlis Eren University, Faculty of Economics and Administrative Sciences, Business Administration
mbestas@beu.edu.tr
0000-0002-8192-2044

## Known limitations and unresolved items

`results/significance_tests/SUPERSEDED_bootstrap_sharpe_results.csv` is not used for any reported result. It is a historical intermediate calculation and does not match Table 7. Table 7 must be recomputed from the verified raw daily-return files using the stationary bootstrap procedure.

- The exact wall-clock runtime for the archived `main.py` and significance-test runs was not recorded.
- The repository contains older historical output directories (`phase2_rime_champion_1/`, the original `phase2_*_champion/` directories, and the separate verification-run directories). The organized `results/` tree is the curated package; those historical directories were left untouched in the workspace.
- The current source retains some legacy names such as `BIST_TICKERS` and comments inherited from the earlier BIST-oriented framework; the active configured tickers and raw snapshot are S&P 500 assets.
