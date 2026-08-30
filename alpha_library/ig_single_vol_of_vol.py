"""
Single Alpha: Volatility-of-Volatility -- Vol regime timing
=============================================================
High vol-of-vol = unstable regime = fade (reduce exposure / contrarian).
Low vol-of-vol = stable regime = ride the trend.

Signal: -rolling_std(daily_vol, 63) * sign(63d return)
When vol-of-vol is high, we scale down conviction; when low, scale up.
Inverted so that stable regimes get stronger signals.

Orthogonal to trend because it's about regime stability, not direction.
Orthogonal to skew/kurtosis because it's about vol dynamics, not return shape.

Reference: Baltussen et al. (2018) "Unknown Unknowns: Uncertainty About
Risk and Stock Returns"; Huang et al. (2019) "Volatility-of-Volatility Risk".
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
    align_fx, blended_volatility, cap_forecast, run_compounded_portfolio,
    rolling_forecast_scalar,
)

STRATEGY_NAME = "Single_Alpha_Vol_of_Vol"
OOS_START = 1280
VOV_WINDOW = 63


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
        if len(dates) < OOS_START + 2:
            continue
        ccy = mapping.loc[instrument, "currency"]
        fx  = align_fx(ccy, dates, fx_daily)
        if fx is None:
            continue

        close         = data["close"]
        price_changes = data["price_changes"]
        vol           = blended_volatility(price_changes)
        idx           = pd.DatetimeIndex(dates)

        # -- SIGNAL: Inverted vol-of-vol * trend direction --
        daily_vol = price_changes.rolling(21, min_periods=10).std()
        vov = daily_vol.rolling(VOV_WINDOW, min_periods=21).std()
        vov_avg = vov.rolling(252, min_periods=63).mean()

        # Normalised: negative when vov is high (unstable), positive when low (stable)
        vov_norm = -(vov / vov_avg - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Direction from 63-day return
        direction = np.sign(close.pct_change(63).fillna(0.0))

        raw = (direction * vov_norm).fillna(0.0)
        forecast_scaled = cap_forecast(raw * rolling_forecast_scalar(raw))

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
