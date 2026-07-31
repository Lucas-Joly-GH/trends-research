"""
Single Alpha: Volume-Price Divergence -- Smart money detection
================================================================
When price goes up but volume goes down, the move lacks conviction
and is more likely to reverse. Conversely, price up + volume up
confirms the move.

Signal: -sign(63d_ret) * (vol_trend diverges from price_trend)
Specifically: correlation(price_change, volume_change, rolling 63d)
Positive corr = healthy trend. Negative corr = divergence = reversal.

Orthogonal to pure trend and volume momentum individually.

Reference: Llorente et al. (2002) "Dynamic Volume-Return Relation";
Campbell, Grossman & Wang (1993) "Trading Volume and Serial
Correlation in Stock Returns".
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

STRATEGY_NAME = "Single_Alpha_Vol_Price_Divergence"
OOS_START = 1280
CORR_WINDOW = 63


def _load_volume(instrument, panama_folder, dates):
    """Load volume from panama continuous file."""
    panama_file = os.path.join(panama_folder, f"{instrument}_continuous.csv")
    if not os.path.exists(panama_file):
        return None
    df = pd.read_csv(panama_file, usecols=["Date", "Volume"], parse_dates=["Date"])
    df = df.dropna(subset=["Volume"]).sort_values("Date").set_index("Date")
    return df["Volume"].reindex(dates).ffill().fillna(0.0)


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

        volume = _load_volume(instrument, paths["panama"], idx)
        if volume is None or volume.sum() == 0:
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

        # -- SIGNAL: Rolling correlation between price changes and volume changes --
        vol_changes = volume.diff().fillna(0.0)
        rolling_corr = price_changes.rolling(CORR_WINDOW, min_periods=21).corr(vol_changes)
        raw = rolling_corr.fillna(0.0)

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
