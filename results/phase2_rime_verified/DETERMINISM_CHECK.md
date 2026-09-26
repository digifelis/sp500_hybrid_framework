# RIME Phase 2 determinism check

The script was run twice consecutively from the same project and Python environment:

- Run 1: `rime_run1/`
- Run 2: `rime_run2/`

The following files were compared with Windows `fc /b`; no byte differences were reported:

- `phase2_portfolio_backtest_summary.csv`
- `phase2_concentration_summary.csv`
- `phase2_roll_log.json`
- `phase2_equity_curves.png`
- `daily_returns_BuyAndHoldIndex.csv`
- `daily_returns_DandelionOptimizer.csv`
- `daily_returns_EqualWeight.csv`
- `daily_returns_Markowitz.csv`

The verified package retains Run 1. Run 2 remains in the workspace as the independent comparison source.
