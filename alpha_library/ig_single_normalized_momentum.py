"""
Single Alpha: Normalized Momentum -- Vol-adjusted price then EWMAC
===================================================================
Instead of applying vol-scaling after the EWMA crossover, first normalize
prices by dividing cumulative returns by rolling volatility, THEN apply
the EWMA crossover. This makes the momentum signal vol-invariant before
the crossover computation.

Different from standard EWMAC because normalising before vs after the
crossover changes which signals fire, especially during vol regime transitions.

Reference: Robert Carver, pysystemtrade `normmom` variants;
Blog: qoppac.blogspot.com (2017).
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

STRATEGY_NAME = "Single_Alpha_Normalized_Momentum"
OOS_START = 1280
SPEEDS = [(16, 64), (32, 128), (64, 256)]


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

        # -- SIGNAL: Normalize prices first, then apply EWMAC crossover --
        # Step 1: vol-normalised returns
        vol_norm_ret = (price_changes / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Step 2: cumulative normalised price
        norm_price = vol_norm_ret.cumsum()

        # Step 3: EWMAC on normalised price (no further vol scaling needed)
        fc_list = []
        for fast, slow in SPEEDS:
            ema_fast = norm_price.ewm(span=fast).mean()
            ema_slow = norm_price.ewm(span=slow).mean()
            raw = ema_fast - ema_slow
            fc_list.append(raw)

        # Equal-weight ensemble
        raw_ensemble = pd.concat(fc_list, axis=1).mean(axis=1).fillna(0.0)
        forecast_scaled = cap_forecast(raw_ensemble * rolling_forecast_scalar(raw_ensemble))

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
