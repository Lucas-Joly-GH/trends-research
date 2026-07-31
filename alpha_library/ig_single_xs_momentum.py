"""Single Alpha: XS_Momentum -- Cross-sectional relative momentum (252-day vol-normalised return)"""
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
    align_fx, blended_volatility, compute_ewmac_raw,
    compute_carry, cap_forecast, run_compounded_portfolio,
    rolling_forecast_scalar,
)

STRATEGY_NAME = "Single_Alpha_XS_Momentum"
OOS_START = 1280
XS_LOOKBACK  = 252
XS_MIN_INSTS = 3


def run_strategy():
    paths    = get_project_paths(_HERE)
    mapping  = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])

    # ── PASS 1: compute raw XS for all instruments ──
    print(f"[PASS 1] Computing raw XS momentum for all instruments...")
    raw_cache = {}
    for instrument in tqdm(mapping.index, desc="XS raw"):
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

        # Vol-normalised cumulative return, then 252-day diff
        norm_price = (price_changes / vol.replace(0.0, np.nan)).fillna(0.0).cumsum()
        xs_raw     = norm_price.diff(XS_LOOKBACK).reindex(idx)

        raw_cache[instrument] = {
            "idx":   idx,
            "dates": dates,
            "xs_raw": xs_raw,
            "vol":   vol.reindex(idx),
            "fx":    fx.reindex(idx),
            "close": close.reindex(idx),
            "open":  data["open"].reindex(idx),
            "pointsize": float(mapping.loc[instrument, "pointsize"]),
            "cost_rt":   float(mapping.loc[instrument, "total_avg_cost_rt"]),
        }

    all_instruments = list(raw_cache.keys())

    # ── PASS 2: cross-sectional z-score ──
    print(f"[PASS 2] Cross-sectional z-scoring across {len(all_instruments)} instruments...")
    xs_raw_df = pd.DataFrame({inst: raw_cache[inst]["xs_raw"] for inst in all_instruments})
    n_valid   = xs_raw_df.notna().sum(axis=1)
    cs_mean   = xs_raw_df.mean(axis=1)
    cs_std    = xs_raw_df.std(axis=1).replace(0.0, np.nan)
    xs_zscored = xs_raw_df.subtract(cs_mean, axis=0).divide(cs_std, axis=0)
    xs_zscored[n_valid < XS_MIN_INSTS] = np.nan

    # ── PASS 3: scale per instrument and build inst_signals ──
    print(f"[PASS 3] Scaling and building signals...")
    inst_signals = {}
    for instrument in tqdm(all_instruments, desc=STRATEGY_NAME):
        cache = raw_cache[instrument]
        idx   = cache["idx"]
        dates = cache["dates"]

        xs_z = xs_zscored[instrument].reindex(idx).fillna(0.0)
        forecast_scaled = cap_forecast(xs_z * rolling_forecast_scalar(xs_z))

        final_fc = forecast_scaled.reindex(idx).fillna(0.0)
        final_fc.iloc[:OOS_START] = 0.0

        inst_signals[instrument] = {
            "oos_date_set": set(dates[OOS_START:]),
            "forecast":     final_fc,
            "vol":          cache["vol"],
            "fx":           cache["fx"],
            "close":        cache["close"],
            "open":         cache["open"],
            "pointsize":    cache["pointsize"],
            "cost_rt":      cache["cost_rt"],
        }

    run_compounded_portfolio(inst_signals, STRATEGY_NAME, paths)


if __name__ == "__main__":
    run_strategy()
