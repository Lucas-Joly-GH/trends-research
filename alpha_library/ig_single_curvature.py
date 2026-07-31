"""
Single Alpha: Term-Structure Curvature (3-point butterfly)
=============================================================
Szymanowska, de Roon, Nijman, van den Goorbergh (2014), RFS 27(1):123
"An Anatomy of Commodity Futures Risk Premia"

Slope-carry (S169's Carry leg) is a 2-point estimator (c1-c2). The
3-point curvature adds the c3 point:

    curv_t  =  (c1 - 2*c2 + c3) / (Δt * sigma_price)

Positive curv = convex curve (near and far both above the mid),
consistent with a storage/inventory state that carry alone conflates
with simple level. Negative curv = concave (squeezed mid), signalling
expected disruption. Szymanowska et al. find curvature explains
risk premia cross-sectionally after controlling for slope, with
independent R^2.

This signal is mechanically orthogonal to 2-point carry because the
(c2) weight is -2 vs +1/-1, so a parallel shift of the whole curve
yields zero curvature.
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
sys.path.insert(0, str(_HERE))
from ig_shared_config import (
    FORECAST_CAP, FORECAST_TARGET, VOL_TARGET,
    get_project_paths, load_mapping, load_fx_rates, load_instrument_data,
    align_fx, blended_volatility,
    cap_forecast, run_compounded_portfolio, rolling_forecast_scalar,
)
from _termstruct_helpers import load_term_structure

STRATEGY_NAME = "Single_Alpha_Curvature"
OOS_START = 1280
SMOOTH_SPAN = 5


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

        ts = load_term_structure(instrument, paths["panama"], paths["contracts"], idx, depth=3)
        if ts is None or ts.get("c3") is None:
            forecast_scaled = pd.Series(0.0, index=idx)
        else:
            c1 = ts["c1"]
            c2 = ts["c2"]
            c3 = ts["c3"]
            y13 = ts["yrs_13"]

            vol_s = vol.reindex(idx)
            curv = (c1 - 2.0 * c2 + c3) / (y13 * vol_s)
            curv = curv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            # Light smoothing, matching Carver carry convention
            curv = curv.ewm(span=SMOOTH_SPAN, min_periods=max(1, SMOOTH_SPAN // 2)).mean()

            forecast_scaled = cap_forecast(
                curv * rolling_forecast_scalar(curv)
            )

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
