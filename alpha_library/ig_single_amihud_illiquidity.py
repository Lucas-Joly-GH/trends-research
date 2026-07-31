"""
Single Alpha: Amihud Illiquidity -- Illiquidity premium factor
===============================================================
Compute the Amihud (2002) illiquidity measure: |return| / volume.
Cross-sectionally rank: go long the most illiquid instruments
(they earn a liquidity premium), short the most liquid.

Reference: Amihud (2002) "Illiquidity and Stock Returns" (JFM);
Pedersen "Liquidity and Asset Prices" (NYU Stern).
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

STRATEGY_NAME = "Single_Alpha_Amihud_Illiquidity"
OOS_START = 1280
AMIHUD_WINDOW = 21


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

    # ── PASS 1: compute Amihud illiquidity for all instruments ──
    print("[PASS 1] Computing Amihud illiquidity...")
    illiq_dict = {}
    inst_data_cache = {}

    for instrument in tqdm(mapping.index, desc="Amihud"):
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
            illiq_dict[instrument] = pd.Series(0.0, index=idx)
        else:
            abs_ret = close.pct_change().abs()
            daily_illiq = (abs_ret / volume.clip(lower=1.0)).replace([np.inf, -np.inf], np.nan)
            amihud = daily_illiq.rolling(AMIHUD_WINDOW, min_periods=10).mean().fillna(0.0)
            illiq_dict[instrument] = amihud.reindex(idx).fillna(0.0)

        inst_data_cache[instrument] = (data, vol, fx)

    # ── PASS 2: cross-sectional z-score of illiquidity ──
    print("[PASS 2] Computing cross-sectional z-scores...")
    illiq_panel = pd.DataFrame(illiq_dict)
    xs_mean = illiq_panel.mean(axis=1)
    xs_std  = illiq_panel.std(axis=1).replace(0, np.nan)

    # ── PASS 3: build signals ──
    print("[PASS 3] Building Illiquidity signals...")
    inst_signals = {}
    for instrument in tqdm(list(inst_data_cache.keys()), desc=STRATEGY_NAME):
        data, vol, fx = inst_data_cache[instrument]
        dates = data["daily_dates"]
        idx   = pd.DatetimeIndex(dates)

        illiq = illiq_dict[instrument].reindex(idx).fillna(0.0)
        xs_m = xs_mean.reindex(idx).fillna(0.0)
        xs_s = xs_std.reindex(idx).fillna(1.0)

        # Z-score: positive for illiquid, negative for liquid
        rel_illiq = ((illiq - xs_m) / xs_s).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        forecast_scaled = cap_forecast(rel_illiq * rolling_forecast_scalar(rel_illiq))

        final_fc = forecast_scaled.reindex(idx).fillna(0.0)
        final_fc.iloc[:OOS_START] = 0.0

        inst_signals[instrument] = {
            "oos_date_set": set(dates[OOS_START:]),
            "forecast":     final_fc,
            "vol":          vol.reindex(idx),
            "fx":           fx.reindex(idx),
            "close":        data["close"].reindex(idx),
            "open":         data["open"].reindex(idx),
            "pointsize":    float(mapping.loc[instrument, "pointsize"]),
            "cost_rt":      float(mapping.loc[instrument, "total_avg_cost_rt"]),
        }

    run_compounded_portfolio(inst_signals, STRATEGY_NAME, paths)


if __name__ == "__main__":
    run_strategy()
