"""
Single Alpha: Open Interest Momentum
======================================
Rising OI + rising price = strong trend (new money entering).
Rising OI + falling price = strong downtrend.
Falling OI = liquidation / position unwinding.

Signal: sign(63d price return) * normalised OI change.
This captures positioning conviction orthogonal to price trend alone.

Reference: Bessembinder & Seguin (1993) "Price Volatility, Trading Volume,
and Market Depth"; Hong & Yogo (2012) "What Does Futures Market Interest
Tell Us about the Macroeconomy and Asset Prices?"
"""
import sys
import os
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

STRATEGY_NAME = "Single_Alpha_OI_Momentum"
OOS_START = 1280


def _load_oi(instrument, panama_folder, dates):
    """Load open interest from panama continuous file."""
    panama_file = os.path.join(panama_folder, f"{instrument}_continuous.csv")
    if not os.path.exists(panama_file):
        return None
    df = pd.read_csv(panama_file, usecols=["Date", "OpenInterest"], parse_dates=["Date"])
    df = df.dropna(subset=["OpenInterest"]).sort_values("Date").set_index("Date")
    return df["OpenInterest"].reindex(dates).ffill().fillna(0.0)


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

        oi = _load_oi(instrument, paths["panama"], idx)
        if oi is None or oi.sum() == 0:
            inst_signals[instrument] = {
                "oos_date_set": set(dates[OOS_START:]),
                "forecast":     pd.Series(0.0, index=idx),
                "vol":          vol.reindex(idx),
                "fx":           fx.reindex(idx),
                "close":        close.reindex(idx),
                "open":         data["open"].reindex(idx),
                "pointsize":    float(mapping.loc[instrument, "pointsize"]),
                "cost_rt":      float(mapping.loc[instrument, "total_avg_cost_rt"]),
            }
            continue

        # -- SIGNAL: OI momentum * price direction --
        oi_avg = oi.rolling(63, min_periods=21).mean()
        oi_change = ((oi / oi_avg) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        oi_smooth = oi_change.ewm(span=21).mean()

        # Direction from 63-day return
        direction = np.sign(close.pct_change(63).fillna(0.0))

        raw = (direction * oi_smooth).fillna(0.0)
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
