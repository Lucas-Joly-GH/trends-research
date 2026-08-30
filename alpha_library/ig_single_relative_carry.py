"""
Single Alpha: Relative Carry -- Cross-sectional carry z-score
==============================================================
Instead of using each instrument's carry in isolation, compute the
cross-sectional z-score of carry across the full universe. Go long
instruments with the highest relative carry, short the lowest.

Different from absolute Carry because it normalises for structural
carry differences across asset classes.

Reference: Robert Carver, "Advanced Futures Trading Strategies" (2023),
pysystemtrade `relcarry`.
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
    compute_carry, cap_forecast, run_compounded_portfolio,
    rolling_forecast_scalar,
)

STRATEGY_NAME = "Single_Alpha_Relative_Carry"
OOS_START = 1280


def run_strategy():
    paths    = get_project_paths(_HERE)
    mapping  = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])

    # ── PASS 1: compute raw carry for all instruments ──
    print("[PASS 1] Computing raw carry for all instruments...")
    carry_dict = {}
    inst_data_cache = {}

    for instrument in tqdm(mapping.index, desc="Raw carry"):
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

        price_changes = data["price_changes"]
        vol = blended_volatility(price_changes)
        idx = pd.DatetimeIndex(dates)

        raw_carry = compute_carry(instrument, paths["panama"], paths["contracts"], dates, vol)
        if raw_carry is not None and raw_carry.dropna().any():
            carry_dict[instrument] = raw_carry.reindex(idx).fillna(0.0)
        else:
            carry_dict[instrument] = pd.Series(0.0, index=idx)

        inst_data_cache[instrument] = (data, vol, fx)

    # ── PASS 2: cross-sectional z-score of carry ──
    print("[PASS 2] Computing cross-sectional carry z-scores...")
    # Build panel of carry values
    carry_panel = pd.DataFrame(carry_dict)

    # Rolling XS z-score (expanding window for stability)
    xs_mean = carry_panel.mean(axis=1)
    xs_std  = carry_panel.std(axis=1).replace(0, np.nan)

    # ── PASS 3: build signals ──
    print("[PASS 3] Building Relative Carry signals...")
    inst_signals = {}
    for instrument in tqdm(list(inst_data_cache.keys()), desc=STRATEGY_NAME):
        data, vol, fx = inst_data_cache[instrument]
        dates = data["daily_dates"]
        idx   = pd.DatetimeIndex(dates)

        raw_carry = carry_dict[instrument].reindex(idx).fillna(0.0)
        xs_m = xs_mean.reindex(idx).fillna(0.0)
        xs_s = xs_std.reindex(idx).fillna(1.0)

        # Z-score: (carry_i - mean) / std
        rel_carry = ((raw_carry - xs_m) / xs_s).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        forecast_scaled = cap_forecast(rel_carry * rolling_forecast_scalar(rel_carry))

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
