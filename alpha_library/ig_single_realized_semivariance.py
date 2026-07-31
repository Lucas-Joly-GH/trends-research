"""
Single Alpha: Realized Semivariance -- Upside vs downside vol asymmetry
========================================================================
Decompose realised variance into positive semivariance (from positive
returns) and negative semivariance (from negative returns).

Signal: (RS+ - RS-) / (RS+ + RS-), rolling 63-day window.
Positive = more upside variance = bullish.
Negative = more downside variance = bearish.

Different from Skew (3rd moment) because this decomposes 2nd moment
by sign, capturing directional risk more directly.

Reference: Liu, Lu, Li & Wang (2023) "Time series momentum and reversal:
Intraday information from realized semivariance" (JEF).
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

STRATEGY_NAME = "Single_Alpha_Realized_Semivariance"
OOS_START = 1280
SV_WINDOW = 63


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

        price_changes = data["price_changes"]
        vol           = blended_volatility(price_changes)
        idx           = pd.DatetimeIndex(dates)

        # Normalise returns by vol first
        norm_ret = (price_changes / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # -- SIGNAL: Semivariance ratio --
        pos_sq = (norm_ret.clip(lower=0.0) ** 2)
        neg_sq = (norm_ret.clip(upper=0.0) ** 2)

        rs_plus  = pos_sq.rolling(SV_WINDOW, min_periods=21).mean()
        rs_minus = neg_sq.rolling(SV_WINDOW, min_periods=21).mean()

        total = (rs_plus + rs_minus).replace(0.0, np.nan)
        raw = ((rs_plus - rs_minus) / total).fillna(0.0)

        forecast_scaled = cap_forecast(raw * rolling_forecast_scalar(raw))

        final_fc = forecast_scaled.reindex(idx).fillna(0.0)
        final_fc.iloc[:OOS_START] = 0.0

        inst_signals[instrument] = {
            "oos_date_set": set(dates[OOS_START:]),
            "forecast":     final_fc,
            "vol":          vol.reindex(idx),
            "fx":           fx.reindex(idx),
            "close":        data["close"].reindex(idx),
            "open":         data["open"].reindex(idx),
            "pointsize":    float(mapping.loc[instrument, "pointsize"]),
            "cost_rt":      float(mapping.loc[instrument, "total_avg_cost_rt"]),
        }

    run_compounded_portfolio(inst_signals, STRATEGY_NAME, paths)


if __name__ == "__main__":
    run_strategy()
