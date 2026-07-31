"""
Single Alpha: Anchored Momentum -- Multi-horizon geometric mean
================================================================
Instead of a single lookback, take the geometric average of returns
at 5 horizons: 21d, 42d, 63d, 126d, 252d. This gives a more robust
momentum signal that's less sensitive to any single lookback choice.

Signal: geometric_mean(ret_21d, ret_42d, ret_63d, ret_126d, ret_252d)
normalised by volatility.

Different from EWMAC (which uses MA crossovers) because this is a
pure return-based momentum using geometric averaging across horizons.

Reference: Baltas & Kosowski (2020) "Demystifying Time-Series
Momentum Strategies"; Baz et al. (2015) "Dissecting Investment
Strategies in the Cross Section and Time Series" (AQR).
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

STRATEGY_NAME = "Single_Alpha_Anchored_Momentum"
OOS_START = 1280
HORIZONS = [21, 42, 63, 126, 252]


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

        # -- SIGNAL: Geometric mean of vol-normalised returns at multiple horizons --
        ret_list = []
        for h in HORIZONS:
            r = close.pct_change(h).fillna(0.0)
            # Annualise: divide by sqrt(h/252) to get vol-normalised
            r_norm = r / np.sqrt(h / 252.0)
            ret_list.append(r_norm)

        # Stack and compute sign-preserving geometric mean
        ret_stack = pd.DataFrame(ret_list).T
        # Use mean of signed returns (more robust than geometric mean for mixed signs)
        raw = ret_stack.mean(axis=1).fillna(0.0)

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
