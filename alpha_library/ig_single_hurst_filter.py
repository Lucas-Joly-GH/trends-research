"""
Single Alpha (transformation): Hurst-Gated EWMAC Ensemble
==========================================================
Mandelbrot/Wallis (1969); Peters (1991). The Hurst exponent H of a
price series characterises persistence:
    H > 0.5  -> trending (positive autocorrelation in increments)
    H = 0.5  -> random walk
    H < 0.5  -> mean-reverting

Idea
----
Run trend signals only when the market is actually in a trending regime.
Apply EWMAC_Ensemble * gate, where:
    gate = 1            if H > 0.55
    gate = 0            if H < 0.45
    gate = linear ramp  in between

H is estimated via the rescaled-range (R/S) method on a 252-day rolling
window of log returns. NOTE: this is a *transformation* of EWMAC_Ensemble
-- compare against EWMAC_Ensemble baseline (SR_Full = 0.574).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from ig_shared_config import (
    FORECAST_CAP, FORECAST_TARGET, VOL_TARGET,
    get_project_paths, load_mapping, load_fx_rates, load_instrument_data,
    align_fx, blended_volatility, compute_ewmac_raw,
    cap_forecast, run_compounded_portfolio, rolling_forecast_scalar,
)

STRATEGY_NAME = "Single_Alpha_Hurst_Filter"
OOS_START     = 1280
HURST_WINDOW  = 252
HURST_LOW     = 0.45
HURST_HIGH    = 0.55

TREND_SPEEDS = [
    {"fast": 16,  "slow": 64},
    {"fast": 64,  "slow": 256},
    {"fast": 128, "slow": 512},
]
TREND_FDM = 1.10


def _hurst_rs(x: np.ndarray) -> float:
    """
    Rescaled-range Hurst estimate for a 1D array of log returns.
    Uses log-log regression of R/S over chunk sizes [16, 32, 64, 128].
    Returns a scalar H or NaN.
    """
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 64:
        return np.nan
    chunks = [c for c in (16, 32, 64, 128) if c <= n]
    if len(chunks) < 2:
        return np.nan
    rs_vals = []
    for c in chunks:
        # split into floor(n/c) non-overlapping chunks
        k = n // c
        if k < 1:
            continue
        rs_chunk = []
        for i in range(k):
            seg = x[i*c:(i+1)*c]
            mu = seg.mean()
            dev = np.cumsum(seg - mu)
            R = dev.max() - dev.min()
            S = seg.std(ddof=0)
            if S > 0 and R > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_vals.append((c, np.mean(rs_chunk)))
    if len(rs_vals) < 2:
        return np.nan
    cs = np.array([r[0] for r in rs_vals], dtype=float)
    rs = np.array([r[1] for r in rs_vals], dtype=float)
    # log(R/S) = H * log(c) + const
    log_c, log_rs = np.log(cs), np.log(rs)
    slope, _ = np.polyfit(log_c, log_rs, 1)
    return float(slope)


def _rolling_hurst(log_ret: pd.Series, window: int = HURST_WINDOW) -> pd.Series:
    arr = log_ret.values
    out = np.full(len(arr), np.nan)
    # Compute every 10 days, ffill in between (R/S is expensive)
    step = 10
    for t in range(window, len(arr), step):
        out[t] = _hurst_rs(arr[t - window:t])
    s = pd.Series(out, index=log_ret.index).ffill()
    return s


def _hurst_gate(h: pd.Series) -> pd.Series:
    """Linear ramp: 0 below 0.45, 1 above 0.55, linear in [0.45, 0.55]."""
    g = (h - HURST_LOW) / (HURST_HIGH - HURST_LOW)
    return g.clip(0.0, 1.0).fillna(0.0)


def run_strategy():
    paths    = get_project_paths(_HERE)
    mapping  = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])

    inst_signals = {}
    for instrument in tqdm(mapping.index, desc=STRATEGY_NAME):
        data = load_instrument_data(instrument, paths["stats"], paths["panama"])
        if data is None:
            continue
        dates = data["daily_dates"]
        if len(dates) < OOS_START + 5:
            continue
        ccy = mapping.loc[instrument, "currency"]
        fx  = align_fx(ccy, dates, fx_daily)
        if fx is None:
            continue

        close         = data["close"]
        price_changes = data["price_changes"]
        vol           = blended_volatility(price_changes)
        idx           = pd.DatetimeIndex(dates)

        # -- EWMAC Ensemble baseline --
        components = []
        for sp in TREND_SPEEDS:
            raw = compute_ewmac_raw(close, vol, sp["fast"], sp["slow"])
            scaled = cap_forecast(raw.fillna(0.0) * rolling_forecast_scalar(raw.fillna(0.0)))
            components.append(scaled * (1.0 / 3.0))
        ewmac_blend = (components[0] + components[1] + components[2]) * TREND_FDM
        ewmac_blend = cap_forecast(ewmac_blend.fillna(0.0))

        # -- Hurst gate --
        log_p   = np.log(close.replace(0, np.nan))
        log_ret = log_p.diff().fillna(0.0)
        log_ret.index = idx
        h    = _rolling_hurst(log_ret, HURST_WINDOW)
        gate = _hurst_gate(h).reindex(idx).fillna(0.0)

        gated = (ewmac_blend.values * gate.values)
        forecast_scaled = pd.Series(gated, index=idx).clip(-FORECAST_CAP, FORECAST_CAP)

        final_fc = forecast_scaled.reindex(idx).fillna(0.0)
        final_fc.iloc[:OOS_START] = 0.0

        inst_signals[instrument] = {
            "oos_date_set": set(dates[OOS_START:]),
            "forecast":     final_fc,
            "vol":          vol.reindex(idx),
            "fx":           fx.reindex(idx),
            "close":        close.reindex(idx),
            "open":         data["open"].reindex(idx),
            "pointsize":    float(mapping.loc[instrument, "pointsize"]),
            "cost_rt":      float(mapping.loc[instrument, "total_avg_cost_rt"]),
        }

    run_compounded_portfolio(inst_signals, STRATEGY_NAME, paths)


if __name__ == "__main__":
    run_strategy()
