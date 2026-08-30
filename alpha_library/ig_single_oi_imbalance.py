"""
Single Alpha: Aggregate Open-Interest Imbalance
=================================================
Proxy for CFTC COT speculator-flow signal from Hong & Yogo (2012),
"What does futures market interest tell us about the macroeconomy
and asset prices?", Journal of Financial Economics 105(3):473.

Mechanism
---------
Hong-Yogo show that when aggregate open interest grows *faster* than
normal, this predicts positive futures returns across commodities
(and bond/FX futures to a weaker extent). Intuition: rising OI
represents new hedging demand being absorbed by speculators, who
require a risk premium. The existing Single_Alpha library includes
an "OI_Momentum" alpha built on nearby-only OI which was strongly
NEGATIVE (SR_full = -0.31). That failure is a measurement artefact:
nearby OI is dominated by roll-off mechanics as expiry approaches,
not by genuine positioning.

This alpha fixes that by summing OI across ALL live contracts for
each instrument on every date, then computing the z-scored growth
rate over a 252-day lookback.

    total_oi_t   = sum over all contracts of OI on date t
    gr_t         = log(total_oi_t) - log(total_oi_{t-126})
    z_t          = (gr_t - mean_252(gr)) / std_252(gr)

Positive z -> OI growing faster than 1-year trend -> long signal.
Orthogonal to price-based alphas by construction because it uses
only position data.
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
from _termstruct_helpers import load_total_oi

STRATEGY_NAME = "Single_Alpha_OI_Imbalance"
OOS_START     = 1280
GROWTH_LOOK   = 126        # 6-month growth (matches Hong-Yogo semi-annual horizon)
Z_WINDOW      = 252        # 1-year z-scoring window
MIN_WINDOW    = 63


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
        if len(dates) < OOS_START + Z_WINDOW + GROWTH_LOOK + 5:
            continue
        ccy = mapping.loc[instrument, "currency"]
        fx  = align_fx(ccy, dates, fx_daily)
        if fx is None:
            continue

        close         = data["close"]
        price_changes = data["price_changes"]
        vol           = blended_volatility(price_changes)
        idx           = pd.DatetimeIndex(dates)

        total_oi = load_total_oi(instrument, paths["contracts"], idx)
        if total_oi is None or total_oi.dropna().sum() == 0:
            forecast_scaled = pd.Series(0.0, index=idx)
        else:
            # Smooth raw OI slightly to suppress weekly reporting noise
            total_oi = total_oi.rolling(5, min_periods=1).mean()
            log_oi   = np.log(total_oi.replace(0, np.nan))
            growth   = log_oi.diff(GROWTH_LOOK)
            mu       = growth.rolling(Z_WINDOW, min_periods=MIN_WINDOW).mean()
            sd       = growth.rolling(Z_WINDOW, min_periods=MIN_WINDOW).std()
            z        = ((growth - mu) / sd.replace(0, np.nan))
            z        = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            forecast_scaled = cap_forecast(
                z * rolling_forecast_scalar(z)
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
