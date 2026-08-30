"""
Single Alpha: Range Compression -- ATR contraction predicting expansion
========================================================================
When the N-day ATR compresses relative to its long-run average,
a volatility expansion (breakout) is imminent.

Signal: sign(63d return) * (1 - ATR_short / ATR_long)
When range is compressed (ATR_short < ATR_long), signal is amplified
in the direction of the prevailing trend.

Orthogonal to EWMAC because it captures volatility regime, not
price level. Related to Bollinger Squeeze / TTM Squeeze.

Reference: Mandelbrot (1963) "The Variation of Certain Speculative
Prices"; Chou (2005) "Forecasting Financial Volatilities with
Extreme Values".
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

STRATEGY_NAME = "Single_Alpha_Range_Compression"
OOS_START = 1280
ATR_SHORT = 21
ATR_LONG = 252


def _load_hl(instrument, panama_folder, dates):
    """Load High, Low from panama continuous file."""
    panama_file = os.path.join(panama_folder, f"{instrument}_continuous.csv")
    if not os.path.exists(panama_file):
        return None, None
    df = pd.read_csv(panama_file, usecols=["Date", "High", "Low"],
                     parse_dates=["Date"])
    df = df.dropna(subset=["High", "Low"]).sort_values("Date").set_index("Date")
    h = df["High"].reindex(dates).ffill()
    l = df["Low"].reindex(dates).ffill()
    return h, l


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

        high, low = _load_hl(instrument, paths["panama"], idx)
        if high is None:
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

        # -- SIGNAL: Range compression → expansion forecast --
        true_range = (high - low).clip(lower=0.0)
        atr_short = true_range.rolling(ATR_SHORT, min_periods=10).mean()
        atr_long  = true_range.rolling(ATR_LONG, min_periods=63).mean()

        # Compression ratio: how compressed is current vol vs long-run
        compression = (1.0 - atr_short / atr_long).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Direction from 63-day return
        direction = np.sign(close.pct_change(63).fillna(0.0))

        raw = (direction * compression).fillna(0.0)
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
