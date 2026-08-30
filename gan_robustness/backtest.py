"""
backtest.py — Simplified S180 backtest on synthetic return paths.

Methodology (user-specified):
  For each synthetic path, replicate the portfolio P&L by:
    1. Deriving implied USD exposures from the real S180 backtest:
         exposure[t, i] = per_inst_pnl[t, i] / hist_price_diff[t, i]
       where hist_price_diff is the historical Close.diff() for each instrument.
       If |hist_price_diff| < EXPOSURE_MIN_RET, exposure[t, i] = 0.

    2. For each synthetic path:
         synthetic_pnl[t] = sum_i(exposure[t, i] × synthetic_price_diff[t, i])
       This gives the P&L the portfolio would have earned if price moves were
       those of the synthetic path but positions (expressed in USD/price-point)
       were identical to the real backtest.

    3. Compute SR, CAGR, MDD, Calmar from the resulting equity curve.

Important caveats (logged in output JSON):
  - per_inst_pnl.sum(axis=1) ≠ portfolio pnl.  The gap (commissions + cash
    yield, ~$2.5B cumulative) means the synthetic backtest is GROSS (pre-cost).
    Real baseline SR=0.93 includes costs; synthetic SRs are not directly comparable.
  - Exposure anchoring: for each 504-step synthetic path, exposures are drawn
    from the last 504 dates of the real backtest (most recent regime).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Exposure derivation
# ---------------------------------------------------------------------------

def compute_exposures(
    per_inst_pnl: pd.DataFrame,     # (T, 62) USD PnL per instrument per day
    hist_returns: pd.DataFrame,     # (T, 62) historical price-diffs, same index
    min_ret_threshold: float = 1e-6,
    clip_value: float = 1e10,
) -> pd.DataFrame:
    """
    Derive implied USD exposures from per-instrument P&L and price movements.

    exposure[t, i] = per_inst_pnl[t, i] / hist_returns[t, i]
                     (if |hist_returns[t, i]| > min_ret_threshold)
                   = 0.0  otherwise

    The exposure represents contracts × pointsize × FX-rate (USD per price-point).

    Args:
        per_inst_pnl       : (T, 62) daily instrument P&L in USD
        hist_returns       : (T, 62) historical price differences (Close.diff())
        min_ret_threshold  : |hist_price_diff| below this -> exposure = 0
        clip_value         : absolute cap on derived exposures (safeguard)

    Returns:
        pd.DataFrame of same shape as per_inst_pnl, indexed by dates.
    """
    # Align on shared index/columns
    shared_index = per_inst_pnl.index.intersection(hist_returns.index)
    shared_cols  = per_inst_pnl.columns.intersection(hist_returns.columns)

    pnl  = per_inst_pnl.loc[shared_index, shared_cols].to_numpy(dtype=np.float64)
    hist = hist_returns.loc[shared_index, shared_cols].to_numpy(dtype=np.float64)

    # Mask near-zero price moves
    valid = np.abs(hist) > min_ret_threshold
    exposure = np.where(valid, pnl / np.where(valid, hist, 1.0), 0.0)

    # Absolute clip safeguard
    exposure = np.clip(exposure, -clip_value, clip_value)

    result = pd.DataFrame(exposure, index=shared_index, columns=shared_cols)

    # Warn if many zero exposures (could indicate systematic data issue)
    zero_frac = (exposure == 0).mean()
    if zero_frac > 0.5:
        warnings.warn(
            f"compute_exposures: {zero_frac:.1%} of exposure cells are zero. "
            "Check min_ret_threshold or hist_returns alignment.",
            stacklevel=2,
        )

    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    daily_ret: np.ndarray,    # (T,) daily arithmetic returns (pnl / equity_{t-1})
    equity: np.ndarray,       # (T+1,) equity curve starting at initial_equity
) -> dict:
    """
    Annualise performance metrics from a daily return series.

    Returns dict with keys: sr, cagr, max_dd, calmar, ann_vol.
    """
    T = len(daily_ret)
    if T < 2:
        return {"sr": 0.0, "cagr": 0.0, "max_dd": 0.0, "calmar": 0.0, "ann_vol": 0.0}

    mu_daily   = np.mean(daily_ret)
    std_daily  = np.std(daily_ret, ddof=1)
    ann_vol    = std_daily * np.sqrt(252)
    sr         = (mu_daily * 252) / ann_vol if ann_vol > 0 else 0.0

    # CAGR
    final_equity   = equity[-1]
    initial_equity = equity[0]
    if initial_equity > 0 and final_equity > 0:
        cagr = (final_equity / initial_equity) ** (252.0 / T) - 1.0
    else:
        cagr = 0.0

    # Maximum drawdown
    running_peak = np.maximum.accumulate(equity)
    drawdowns    = (running_peak - equity) / np.where(running_peak > 0, running_peak, 1.0)
    max_dd       = float(np.max(drawdowns))

    calmar = cagr / max_dd if max_dd > 1e-9 else (np.inf if cagr > 0 else 0.0)

    return {
        "sr":      float(sr),
        "cagr":    float(cagr),
        "max_dd":  float(max_dd),
        "calmar":  float(calmar),
        "ann_vol": float(ann_vol),
    }


# ---------------------------------------------------------------------------
# Single synthetic backtest
# ---------------------------------------------------------------------------

def run_synthetic_backtest(
    exposure_df: pd.DataFrame,           # (T_hist, 62) derived from real backtest
    synthetic_path_df: pd.DataFrame,     # (path_length, 62) synthetic price-diffs
    backtest_dates: pd.DatetimeIndex,    # which rows to pull from exposure_df
    initial_equity: float = 1e8,
) -> dict:
    """
    Compute the synthetic portfolio P&L and performance metrics.

    For step t in [0, path_length):
      exposure_t         = exposure_df.loc[backtest_dates[t]]    # (62,) USD/price-point
      synthetic_return_t = synthetic_path_df.iloc[t]             # (62,) price-diffs
      pnl_t              = (exposure_t × synthetic_return_t).sum()
      equity[t+1]        = equity[t] + pnl_t

    The backtest_dates must be a subset of exposure_df.index.
    synthetic_path_df columns must match exposure_df.columns (62 instruments).

    Returns:
        dict with equity (list), daily_ret (list), and all metrics from compute_metrics.
    """
    path_length = len(synthetic_path_df)

    # Align columns: use only instruments present in both
    common_cols = exposure_df.columns.intersection(synthetic_path_df.columns)
    if len(common_cols) < len(exposure_df.columns):
        warnings.warn(
            f"run_synthetic_backtest: only {len(common_cols)}/{len(exposure_df.columns)} "
            "instruments overlap between exposure_df and synthetic_path_df.",
            stacklevel=2,
        )

    exp_matrix   = exposure_df.loc[backtest_dates, common_cols].to_numpy(dtype=np.float64)
    synth_matrix = synthetic_path_df[common_cols].to_numpy(dtype=np.float64)

    equity = np.empty(path_length + 1, dtype=np.float64)
    equity[0] = initial_equity
    daily_ret  = np.empty(path_length, dtype=np.float64)

    for t in range(path_length):
        pnl_t     = np.dot(exp_matrix[t], synth_matrix[t])
        equity[t + 1] = equity[t] + pnl_t
        daily_ret[t]  = pnl_t / equity[t] if equity[t] > 0 else 0.0

    metrics = compute_metrics(daily_ret, equity)
    metrics["equity"]    = equity.tolist()
    metrics["daily_ret"] = daily_ret.tolist()
    return metrics


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all_backtests(
    exposure_df: pd.DataFrame,
    all_paths: list[pd.DataFrame],   # 500 × (path_length, 62)
    config: object,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run run_synthetic_backtest for all synthetic paths.

    Anchors each backtest on the last PATH_LENGTH dates of the real backtest.

    Returns:
        pd.DataFrame of shape (N_PATHS, 5) with columns [sr, cagr, max_dd, calmar, ann_vol].
    """
    backtest_dates = exposure_df.index[-config.PATH_LENGTH:]

    if len(backtest_dates) < config.PATH_LENGTH:
        raise ValueError(
            f"exposure_df has only {len(exposure_df)} rows, "
            f"need at least {config.PATH_LENGTH} for the anchor window."
        )

    records = []
    for i, path_df in enumerate(all_paths):
        result = run_synthetic_backtest(
            exposure_df=exposure_df,
            synthetic_path_df=path_df,
            backtest_dates=backtest_dates,
            initial_equity=config.INITIAL_EQUITY,
        )
        records.append({k: result[k] for k in ["sr", "cagr", "max_dd", "calmar", "ann_vol"]})

        if verbose and (i + 1) % 100 == 0:
            print(f"  [backtest] {i + 1}/{len(all_paths)} paths completed")

    return pd.DataFrame(records)
