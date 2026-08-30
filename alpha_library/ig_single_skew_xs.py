"""Single Alpha: Skew_XS -- Cross-sectional relative skewness"""
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

STRATEGY_NAME = "Single_Alpha_Skew_XS"
OOS_START = 1280
SKEW_WINDOW = 256


def run_strategy():
    paths    = get_project_paths(_HERE)
    mapping  = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])

    # ── PASS 1: compute rolling skew for all instruments ──
    print("[PASS 1] Computing rolling skewness...")
    rolling_skews   = {}
    inst_data_cache = {}

    for instrument in tqdm(mapping.index, desc="Rolling skew"):
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

        daily_ret = pd.Series(data["daily_return"], index=data["daily_dates"])
        rolling_skews[instrument] = daily_ret.rolling(SKEW_WINDOW, min_periods=128).skew()
        inst_data_cache[instrument] = (data, fx)

    # ── PASS 2: asset-class average skew ──
    print("[PASS 2] Computing asset-class skew aggregates...")
    asset_classes = mapping["asset_class"].dropna().unique()
    ac_skews = {}
    for ac in asset_classes:
        sk_members = [i for i in mapping.index[mapping["asset_class"] == ac] if i in rolling_skews]
        if sk_members:
            ac_skews[ac] = pd.DataFrame({m: rolling_skews[m] for m in sk_members}).mean(axis=1)

    # ── PASS 3: compute Skew_XS signal per instrument ──
    print("[PASS 3] Building Skew_XS signals...")
    inst_signals = {}
    for instrument in tqdm(list(inst_data_cache.keys()), desc=STRATEGY_NAME):
        data, fx = inst_data_cache[instrument]
        dates = data["daily_dates"]
        idx   = pd.DatetimeIndex(dates)
        ac    = mapping.loc[instrument, "asset_class"]
        vol   = blended_volatility(data["price_changes"])

        if ac in ac_skews and instrument in rolling_skews:
            inst_skew   = rolling_skews[instrument].reindex(dates).ffill()
            ac_avg_skew = ac_skews[ac].reindex(dates).ffill()
            raw_skew_xs = -(inst_skew - ac_avg_skew).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            forecast_scaled = cap_forecast(
                raw_skew_xs * rolling_forecast_scalar(raw_skew_xs, window=1280, min_periods=640)
            )
        else:
            forecast_scaled = pd.Series(0.0, index=idx)

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
