"""
Single Alpha: Same-Calendar-Month Seasonality
================================================
Keloharju, Linnainmaa, Nyberg (2016), "Return Seasonalities",
JF 71(4):1557 -- strong and robust monthly-seasonal effect in
equities AND commodities. Also Gorton, Hayashi, Rouwenhorst
(2013), "The Fundamentals of Commodity Futures Returns",
RFS 26(10):2611, for the commodity-specific harvest cycle channel.

Mechanism
---------
Commodities exhibit persistent month-of-year effects driven by
inventory cycles (harvests, heating/cooling seasons, planting).
Currencies and rates show month-end rebalancing + fiscal-year
effects. The signal takes the *trailing 10-year mean* of the
return earned in the CURRENT calendar month and uses it as the
forecast.

    m_t       = month of date t
    past_m    = monthly returns from past years in month m
    seasonal  = trailing mean over the last 10 occurrences
    raw       = seasonal / std(past_m)   # t-stat-like

Orthogonal to all momentum/carry/skew/vov signals by construction
because it depends only on the *calendar month* — a feature none
of the other 30 alphas use.
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
    align_fx, blended_volatility,
    cap_forecast, run_compounded_portfolio, rolling_forecast_scalar,
)

STRATEGY_NAME = "Single_Alpha_Seasonality"
OOS_START     = 1280
MIN_YEARS     = 5       # need at least 5 prior occurrences of the same month
MAX_YEARS     = 10      # trailing 10-year window


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
        if len(dates) < OOS_START + 5:
            continue
        ccy = mapping.loc[instrument, "currency"]
        fx  = align_fx(ccy, dates, fx_daily)
        if fx is None:
            continue

        close         = data["close"]
        price_changes = data["price_changes"]
        vol           = blended_volatility(price_changes)
        idx           = pd.DatetimeIndex(dates)

        # Build monthly log returns from close
        log_p   = np.log(close.replace(0, np.nan))
        monthly = log_p.resample("ME").last().diff().dropna()

        # For each month in the daily index, find the trailing 10 prior years
        # of the *same* calendar month and use the rolling mean as the forecast.
        month_of = monthly.index.month
        raw_monthly = pd.Series(0.0, index=monthly.index)
        for i, ts_i in enumerate(monthly.index):
            m = ts_i.month
            prior_mask = (month_of == m) & (np.arange(len(monthly)) < i)
            prior_vals = monthly.values[prior_mask]
            if len(prior_vals) >= MIN_YEARS:
                prior_vals = prior_vals[-MAX_YEARS:]
                mu = prior_vals.mean()
                sd = prior_vals.std(ddof=0)
                if sd > 0:
                    raw_monthly.iloc[i] = mu / sd   # t-stat-style seasonal score
        # Forward-fill to daily
        raw_daily = raw_monthly.reindex(idx, method="ffill").fillna(0.0)

        forecast_scaled = cap_forecast(
            raw_daily * rolling_forecast_scalar(raw_daily)
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
