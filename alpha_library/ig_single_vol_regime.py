"""
Single Alpha: Volatility Regime -- Mean-reversion of realised vol
==================================================================
When realised vol is far above its long-run average, it tends to
mean-revert down (contraction phase). When far below, it expands.

Signal: -(vol_short / vol_long - 1) * sign(63d return)
High current vol relative to average = expect contraction = fade.
Low current vol = expect expansion = ride existing trend harder.

Different from Vol_of_Vol (which measures instability) because this
measures vol level relative to its mean.

Reference: Engle & Rangel (2008) "The Spline-GARCH Model for Low-
Frequency Volatility"; Carver (2019) "Leveraged Trading".
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

STRATEGY_NAME = "Single_Alpha_Vol_Regime"
OOS_START = 1280


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

        # -- SIGNAL: Vol regime (short vol vs long vol average) --
        vol_short = price_changes.ewm(span=32, min_periods=16).std()
        vol_long  = price_changes.rolling(512, min_periods=128).std()

        # Negative when vol is elevated (fade), positive when vol is low (trend)
        vol_ratio = -(vol_short / vol_long - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Direction from 63-day return
        direction = np.sign(close.pct_change(63).fillna(0.0))

        raw = (direction * vol_ratio).fillna(0.0)
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
