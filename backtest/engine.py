"""
backtest/engine.py
===================
Two-part evaluation engine:

  PART A -- Forecasting benchmark (Phase 1)
    Head-to-head comparison of RIME-LSTM, MGO-LSTM, plain LSTM, plain GRU,
    and ARIMA on a chronological train/val/test split, reporting RMSE and
    MAPE (both on the normalized [0,1] scale and the inverse-transformed
    price scale).

  PART B -- Rolling-window portfolio backtest (Phase 2)
    True walk-forward evaluation: at every roll, (i) a fresh forecaster is
    trained per ticker on that roll's trailing window and used to rank
    assets by predicted forward return (Phase-1-informed asset selection),
    then (ii) the Dandelion Optimizer, classical Markowitz (max-Sharpe),
    and 1/N are each used to weight the SAME selected assets, and the
    resulting portfolios are held (unrebalanced) over the next out-of-
    sample period. The passive BIST100 buy-and-hold index is tracked in
    parallel. All strategies' realized daily returns are concatenated
    across rolls into one continuous out-of-sample equity curve.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from utils.technical_indicators import build_feature_frame, inverse_min_max
from utils.metrics import rmse, mape, summarize_performance
from data.data_pipeline import build_sequences, chronological_split
from models.sequence_models import SequenceModelConfig, build_model, train_forecaster, predict
from models.hyperparam_tuning import run_hyperparameter_search, decode_hyperparams
from models.arima_baseline import fit_predict_arima, select_best_order
from config import release_torch_memory
from portfolio.markowitz import markowitz_max_sharpe, equal_weight_portfolio
from portfolio.dandelion_portfolio import optimize_portfolio_dandelion


# ==========================================================================
# PART A: Forecasting benchmark
# ==========================================================================
def _make_sequences_for_ticker(feature_df: pd.DataFrame, cfg):
    X, y, origin_idx = build_sequences(feature_df, cfg.FEATURE_COLUMNS, "close_norm",
                                        cfg.SEQUENCE_LENGTH, cfg.FORECAST_HORIZON)
    train_idx, val_idx, test_idx = chronological_split(len(X), cfg.TRAIN_FRACTION, cfg.VAL_FRACTION)
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx]), origin_idx[test_idx]


def run_forecasting_benchmark(feature_frames: Dict[str, pd.DataFrame], tickers: List[str], cfg,
                               tune_ticker: Optional[str] = None,
                               hp_pop: int = 6, hp_iter: int = 5,
                               baseline_hidden: int = 32, baseline_epochs: int = 20,
                               seed: int = 42, verbose: bool = True
                               ) -> Tuple[pd.DataFrame, Dict[str, SequenceModelConfig], Dict[str, object]]:
    """Returns (results_df, tuned_configs, optimizer_results)."""
    tune_ticker = tune_ticker or tickers[0]
    n_features = len(cfg.FEATURE_COLUMNS)

    (Xtr, ytr), (Xval, yval), (Xte, yte), _ = _make_sequences_for_ticker(feature_frames[tune_ticker], cfg)

    if verbose:
        print(f"[Phase 1] Tuning hyperparameters on '{tune_ticker}' "
              f"({len(Xtr)} train / {len(Xval)} val / {len(Xte)} test sequences)")

    optimizer_results = {}
    tuned_configs = {}
    for opt_name in ("RIME", "MGO"):
        result, best_cfg = run_hyperparameter_search(
            opt_name, Xtr, ytr, Xval, yval, cfg.HP_BOUNDS, n_features, cfg.FORECAST_HORIZON,
            n_pop=hp_pop, max_iter=hp_iter, seed=seed, verbose=verbose,
        )
        optimizer_results[opt_name] = result
        tuned_configs[f"{opt_name}-LSTM"] = best_cfg
        if verbose:
            print(f"  -> {opt_name} best fitness={result.best_fitness:.5f} | "
                  f"cfg: hidden={best_cfg.hidden_size} lr={best_cfg.learning_rate:.5f} "
                  f"dropout={best_cfg.dropout:.3f} batch={best_cfg.batch_size} epochs={best_cfg.epochs}")

    tuned_configs["LSTM"] = SequenceModelConfig(input_size=n_features, hidden_size=baseline_hidden,
                                                 num_layers=1, dropout=0.2, horizon=cfg.FORECAST_HORIZON,
                                                 learning_rate=1e-3, batch_size=32, epochs=baseline_epochs)
    tuned_configs["GRU"] = SequenceModelConfig(input_size=n_features, hidden_size=baseline_hidden,
                                                num_layers=1, dropout=0.2, horizon=cfg.FORECAST_HORIZON,
                                                learning_rate=1e-3, batch_size=32, epochs=baseline_epochs)

    rows = []
    for tkr in tickers:
        (Xtr_t, ytr_t), (Xval_t, yval_t), (Xte_t, yte_t), test_origin = _make_sequences_for_ticker(feature_frames[tkr], cfg)
        close_min = feature_frames[tkr].attrs["close_min"]
        close_max = feature_frames[tkr].attrs["close_max"]

        for model_name, model_cfg in tuned_configs.items():
            model_type = "gru" if model_name == "GRU" else "lstm"
            model = build_model(model_type, model_cfg)
            train_forecaster(model, Xtr_t, ytr_t, Xval_t, yval_t, model_cfg, early_stopping_patience=5)
            y_pred_norm = predict(model, Xte_t)
            del model
            release_torch_memory()
            rmse_n, mape_n = rmse(yte_t, y_pred_norm), mape(yte_t, y_pred_norm)
            y_true_price = inverse_min_max(yte_t, close_min, close_max)
            y_pred_price = inverse_min_max(y_pred_norm, close_min, close_max)
            rmse_p, mape_p = rmse(y_true_price, y_pred_price), mape(y_true_price, y_pred_price)
            rows.append({"ticker": tkr, "model": model_name, "RMSE_norm": rmse_n, "MAPE_norm_%": mape_n,
                         "RMSE_price": rmse_p, "MAPE_price_%": mape_p})

        # ARIMA baseline: fit order once on train+val close prices, then walk the test
        # origins (every `horizon` steps to bound runtime) forecasting `horizon` steps ahead.
        close_series = feature_frames[tkr]["close"].values
        n_train_val = len(Xtr_t) + len(Xval_t) + cfg.SEQUENCE_LENGTH
        order = select_best_order(close_series[:n_train_val])
        arima_true, arima_pred = [], []
        step = max(1, cfg.FORECAST_HORIZON)
        for k in range(0, len(test_origin), step):
            origin = test_origin[k]
            history = close_series[: origin + 1]
            fc = fit_predict_arima(history, cfg.FORECAST_HORIZON, order=order)
            actual = close_series[origin + 1: origin + 1 + cfg.FORECAST_HORIZON]
            if len(actual) == cfg.FORECAST_HORIZON:
                arima_true.append(actual)
                arima_pred.append(fc)
        if arima_true:
            arima_true = np.concatenate(arima_true)
            arima_pred = np.concatenate(arima_pred)
            rows.append({"ticker": tkr, "model": "ARIMA",
                         "RMSE_norm": np.nan, "MAPE_norm_%": np.nan,
                         "RMSE_price": rmse(arima_true, arima_pred), "MAPE_price_%": mape(arima_true, arima_pred)})
        if verbose:
            print(f"  [{tkr}] benchmarked all 5 models.")

    results_df = pd.DataFrame(rows)
    return results_df, tuned_configs, optimizer_results


# ==========================================================================
# PART B: Rolling-window portfolio backtest
# ==========================================================================
def _predict_forward_return(feature_df: pd.DataFrame, cfg, model_cfg: SequenceModelConfig,
                             window_end_pos: int, seed: int) -> Optional[float]:
    """Trains a fresh small forecaster on the trailing window
    feature_df.iloc[:window_end_pos] and returns the model's implied
    horizon-cumulative expected return (price scale) from the most recent
    sequence -- i.e. the Phase-1 signal used for asset ranking at this roll.
    Returns None if there is not enough history yet.
    """
    trail = feature_df.iloc[:window_end_pos]
    if len(trail) < cfg.SEQUENCE_LENGTH + cfg.FORECAST_HORIZON + 10:
        return None
    X, y, _ = build_sequences(trail, cfg.FEATURE_COLUMNS, "close_norm", cfg.SEQUENCE_LENGTH, cfg.FORECAST_HORIZON)
    if len(X) < 5:
        return None
    n_val = max(1, int(len(X) * 0.15))
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    model = build_model("lstm", model_cfg)
    train_forecaster(model, X_train, y_train, X_val, y_val, model_cfg, early_stopping_patience=4)

    last_seq = trail[cfg.FEATURE_COLUMNS].values[-cfg.SEQUENCE_LENGTH:].astype(np.float32)[None, ...]
    pred_norm = predict(model, last_seq)[0]                      # (horizon,) normalized close forecast
    del model
    release_torch_memory()
    close_min, close_max = feature_df.attrs["close_min"], feature_df.attrs["close_max"]
    pred_price = inverse_min_max(pred_norm, close_min, close_max)
    current_price = inverse_min_max(np.array([trail["close_norm"].values[-1]]), close_min, close_max)[0]
    if current_price <= 0:
        return None
    return float((pred_price[-1] - current_price) / current_price)   # expected horizon return


def _predict_forward_return_arima(raw_df: pd.DataFrame, cfg, window_end_pos: int,
                                   order: Optional[tuple] = None) -> Optional[float]:
    """ARIMA counterpart to _predict_forward_return: fits ARIMA(p,d,q) on the
    trailing raw close-price window ending at window_end_pos and returns the
    implied horizon-cumulative expected return from a fresh forecast, so
    that Phase-2 asset ranking can be driven by the same forecaster
    Section 4.3 shows is most accurate on this dataset (ARIMA), as a direct
    test of whether Phase-1 asset-selection accuracy -- rather than the
    Dandelion/Markowitz weighting stage -- is what drives Phase-2 portfolio
    performance. `order` may be pre-selected once per ticker (recommended,
    for tractability across many rolls) and passed in; otherwise it is
    re-selected by AIC search on every call.
    """
    close_series = raw_df["close"].values[:window_end_pos]
    if len(close_series) < cfg.SEQUENCE_LENGTH + cfg.FORECAST_HORIZON + 10:
        return None
    use_order = order if order is not None else select_best_order(close_series)
    forecast = fit_predict_arima(close_series, cfg.FORECAST_HORIZON, order=use_order)
    current_price = float(close_series[-1])
    if current_price <= 0:
        return None
    return float((forecast[-1] - current_price) / current_price)


def run_rolling_portfolio_backtest(feature_frames: Dict[str, pd.DataFrame],
                                    raw_universe: Dict[str, pd.DataFrame],
                                    index_level: pd.Series, cfg,
                                    forecaster_cfg: SequenceModelConfig,
                                    top_k: int = 10, seed: int = 42, verbose: bool = True,
                                    ranking_method: str = "lstm",
                                    arima_orders: Optional[Dict[str, tuple]] = None
                                    ) -> Tuple[Dict[str, pd.Series], pd.DataFrame, List[dict]]:
    """ranking_method: "lstm" (default, per Section 3.6/3.9) or "arima" -- an
    alternative Phase-1 signal that ranks assets by ARIMA's own forward-
    return forecast instead of the champion LSTM's, holding every other
    part of the pipeline (Dandelion/Markowitz/equal-weight weighting,
    rolling-window design, evaluation) fixed. arima_orders optionally
    supplies a pre-selected (p,d,q) per ticker (recommended -- selecting
    fresh on every one of the ~450 (ticker, roll) calls is needlessly slow
    and each ticker's order is unlikely to change materially roll-to-roll).
    """
    tickers = list(feature_frames.keys())
    returns_matrix = pd.DataFrame({t: raw_universe[t]["close"].pct_change() for t in tickers}).dropna()
    index_returns = index_level.pct_change().reindex(returns_matrix.index).fillna(0.0)

    all_dates = returns_matrix.index
    oos_mask = (all_dates >= pd.Timestamp(cfg.OOS_BACKTEST_START)) & (all_dates <= pd.Timestamp(cfg.OOS_BACKTEST_END))
    oos_dates = all_dates[oos_mask]
    if len(oos_dates) == 0:
        raise ValueError("No out-of-sample dates found in the generated/loaded price history; "
                          "check OOS_BACKTEST_START/END against the data range.")
    roll_starts = oos_dates[::cfg.BACKTEST_STEP]

    if ranking_method == "arima" and arima_orders is None:
        arima_orders = {}
        for tkr in tickers:
            hist = raw_universe[tkr]["close"].values
            n_use = min(len(hist), cfg.BACKTEST_TRAIN_WINDOW + 60)
            arima_orders[tkr] = select_best_order(hist[:n_use])
        if verbose:
            print(f"[Phase 2, ARIMA ranking] pre-selected orders: {arima_orders}")

    strategy_returns = {"DandelionOptimizer": [], "Markowitz": [], "EqualWeight": [], "BuyAndHoldIndex": []}
    roll_log = []

    for roll_i, roll_start in enumerate(roll_starts):
        start_pos = all_dates.get_loc(roll_start)
        trail_start_pos = max(0, start_pos - cfg.BACKTEST_TRAIN_WINDOW)
        hold_end_pos = min(len(all_dates), start_pos + cfg.BACKTEST_STEP)
        if start_pos >= hold_end_pos:
            continue
        trailing_returns = returns_matrix.iloc[trail_start_pos:start_pos]
        holding_returns = returns_matrix.iloc[start_pos:hold_end_pos]
        if len(trailing_returns) < 30 or len(holding_returns) == 0:
            continue

        # ---- Phase-1-informed asset selection ----
        scores = {}
        for tkr in tickers:
            if ranking_method == "arima":
                raw_df = raw_universe[tkr]
                pos = raw_df.index.get_indexer([roll_start], method="pad")[0]
                if pos <= 0:
                    continue
                score = _predict_forward_return_arima(raw_df, cfg, pos, order=arima_orders.get(tkr))
            else:
                fdf = feature_frames[tkr]
                pos = fdf.index.get_indexer([roll_start], method="pad")[0]
                if pos <= 0:
                    continue
                score = _predict_forward_return(fdf, cfg, forecaster_cfg, pos, seed=seed + roll_i)
            if score is not None:
                scores[tkr] = score
        if len(scores) < max(3, top_k // 2):
            continue
        selected = sorted(scores, key=scores.get, reverse=True)[:top_k]

        trail_sel = trailing_returns[selected]
        hold_sel = holding_returns[selected]

        # ---- Phase-2 portfolio weighting on the SAME selected assets ----
        w_do, _ = optimize_portfolio_dandelion(trail_sel, cfg.RISK_FREE_RATE_ANNUAL,
                                                cfg.PORTFOLIO_OBJECTIVE_WEIGHTS,
                                                n_pop=cfg.PORTFOLIO_OPTIMIZER_POPULATION,
                                                max_iter=cfg.PORTFOLIO_OPTIMIZER_ITERATIONS,
                                                seed=seed + roll_i)
        w_mkw = markowitz_max_sharpe(trail_sel.mean().values, trail_sel.cov().values, cfg.RISK_FREE_RATE_ANNUAL)
        w_eq = equal_weight_portfolio(len(selected))

        strategy_returns["DandelionOptimizer"].append(pd.Series(hold_sel.values @ w_do, index=hold_sel.index))
        strategy_returns["Markowitz"].append(pd.Series(hold_sel.values @ w_mkw, index=hold_sel.index))
        strategy_returns["EqualWeight"].append(pd.Series(hold_sel.values @ w_eq, index=hold_sel.index))
        strategy_returns["BuyAndHoldIndex"].append(index_returns.iloc[start_pos:hold_end_pos])

        # Full weight vectors (not just the top weight) are logged so that
        # concentration can be summarized post hoc by any measure -- e.g.
        # the Herfindahl-Hirschman Index HHI = sum(w_i^2) and the effective
        # number of holdings 1/HHI -- rather than only the single max weight
        # recorded previously.
        roll_log.append({"roll": roll_i, "date": roll_start, "selected_assets": selected,
                          "do_top_weight": float(np.max(w_do)), "mkw_top_weight": float(np.max(w_mkw)),
                          "do_weights": [float(x) for x in w_do], "mkw_weights": [float(x) for x in w_mkw],
                          "eq_weights": [float(x) for x in w_eq]})
        if verbose:
            print(f"[Phase 2] roll {roll_i:2d} @ {roll_start.date()} | "
                  f"{len(selected)} assets selected | DO top-w={np.max(w_do):.2f} | "
                  f"Markowitz top-w={np.max(w_mkw):.2f}")

    concatenated = {k: (pd.concat(v).sort_index() if v else pd.Series(dtype=float)) for k, v in strategy_returns.items()}
    summary_rows = []
    for strategy, series in concatenated.items():
        if len(series) == 0:
            continue
        perf = summarize_performance(series.values, cfg.RISK_FREE_RATE_ANNUAL)
        perf["Strategy"] = strategy
        summary_rows.append(perf)
    summary_df = pd.DataFrame(summary_rows).set_index("Strategy")[
        ["CumulativeReturn", "AnnualizedReturn", "AnnualizedVolatility", "SharpeRatio",
         "SortinoRatio", "MaxDrawdown", "CalmarRatio"]
    ]
    return concatenated, summary_df, roll_log
