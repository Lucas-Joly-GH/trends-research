"""
Single Alpha: End-of-Month Drift (Turn-of-the-Month effect)
============================================================
Ogden (1990), "Turn-of-Month Evaluations of Liquid Profits and Stock
Returns", JF; Etula, Rinne, Suominen, Vaittinen (2020), "Dash for Cash:
Monthly Market Impact of Institutional Liquidity Needs", RFS.

Mechanism
---------
Concentration of dividend reinvestment, payroll flows, and pension
rebalancing produces a positive drift in the last few trading days of a
month and the first few of the next. Documented across equities, bonds,
and many futures markets.

Signal
------
Pure calendar:
    +1 in the last 4 trading days of the month and the first 3 of the next
     0 elsewhere
After scaling by rolling_forecast_scalar this becomes a long/flat
signal targeting FORECAST_TARGET on active days.

Orthogonal to momentum/carry/skew/vov by construction (calendar-only).
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

STRATEGY_NAME = "Single_Alpha_EOM_Drift"
OOS_START     = 1280
LAST_N_DAYS   = 4    # last N trading days of the month
FIRST_N_DAYS  = 3    # first N trading days of the next month


def _build_eom_signal(idx: pd.DatetimeIndex) -> pd.Series:
    """Return a Series of +1 / 0 over idx for the turn-of-month window."""
    s = pd.Series(0.0, index=idx)

    # Group by year-month, find the last N trading days of each
    months = idx.to_period("M")
    df = pd.DataFrame({"i": np.arange(len(idx)), "m": months})

    for _, grp in df.groupby("m", sort=False):
        rows = grp["i"].values
        # last N of this month
        for j in rows[-LAST_N_DAYS:]:
            s.iloc[j] = 1.0

    # first N trading days of each month (which is "post turn-of-month")
    for _, grp in df.groupby("m", sort=False):
        rows = grp["i"].values
        for j in rows[:FIRST_N_DAYS]:
            s.iloc[j] = 1.0

    return s


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

        # -- SIGNAL: turn-of-month dummy --
        raw = _build_eom_signal(idx)
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
