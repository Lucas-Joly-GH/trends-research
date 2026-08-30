"""
Single Alpha: Overnight Momentum -- Close-to-Open return persistence
=====================================================================
Decompose daily returns into overnight (close-to-open) and intraday
(open-to-close). Almost all momentum alpha concentrates in the overnight
component. Signal: EWM(21) of overnight returns.

Reference: Lou, Polk & Skouras (2019) "A Tug of War: Overnight vs.
Intraday Returns" (Journal of Financial Economics).
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

STRATEGY_NAME = "Single_Alpha_Overnight_Momentum"
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
        open_         = data["open"]
        price_changes = data["price_changes"]
        vol           = blended_volatility(price_changes)
        idx           = pd.DatetimeIndex(dates)

        # -- SIGNAL: Overnight return = (Open_t - Close_{t-1}) / Close_{t-1} --
        overnight_ret = ((open_ - close.shift(1)) / close.shift(1).abs().clip(lower=1e-8))
        overnight_ret = overnight_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Smooth with EWM(21) for persistence
        raw = overnight_ret.ewm(span=21).mean()
        forecast_scaled = cap_forecast(raw * rolling_forecast_scalar(raw))

        final_fc = forecast_scaled.reindex(idx).fillna(0.0)
        final_fc.iloc[:OOS_START] = 0.0

        inst_signals[instrument] = {
            "oos_date_set": set(dates[OOS_START:]),
            "forecast":     final_fc,
            "vol":          vol.reindex(idx),
            "fx":           fx.reindex(idx),
            "close":        close.reindex(idx),
            "open":         open_.reindex(idx),
            "pointsize":    float(mapping.loc[instrument, "pointsize"]),
            "cost_rt":      float(mapping.loc[instrument, "total_avg_cost_rt"]),
        }

    run_compounded_portfolio(inst_signals, STRATEGY_NAME, paths)


if __name__ == "__main__":
    run_strategy()
