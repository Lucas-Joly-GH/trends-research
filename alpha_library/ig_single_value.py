"""Single Alpha: Value -- Cross-sectional relative price (mean-reversion)"""
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

STRATEGY_NAME = "Single_Alpha_Value"
OOS_START = 1280
VALUE_ROLLING_MEAN_WINDOW = 1280


def run_strategy():
    paths    = get_project_paths(_HERE)
    mapping  = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])

    # ── PASS 1: compute normalised prices for all instruments ──
    print("[PASS 1] Computing normalised prices...")
    norm_prices     = {}
    inst_data_cache = {}

    for instrument in tqdm(mapping.index, desc="Norm prices"):
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
        vol_norm_ret = (price_changes / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        norm_prices[instrument] = vol_norm_ret.cumsum()
        inst_data_cache[instrument] = (data, vol, fx)

    # ── PASS 2: asset-class average normalised price ──
    print("[PASS 2] Computing asset-class aggregates...")
    asset_classes  = mapping["asset_class"].dropna().unique()
    ac_norm_prices = {}
    for ac in asset_classes:
        members = [i for i in mapping.index[mapping["asset_class"] == ac] if i in norm_prices]
        if members:
            ac_norm_prices[ac] = pd.DataFrame({m: norm_prices[m] for m in members}).mean(axis=1)

    # ── PASS 3: compute Value signal per instrument ──
    print("[PASS 3] Building Value signals...")
    inst_signals = {}
    for instrument in tqdm(list(inst_data_cache.keys()), desc=STRATEGY_NAME):
        data, vol, fx = inst_data_cache[instrument]
        dates = data["daily_dates"]
        idx   = pd.DatetimeIndex(dates)
        ac    = mapping.loc[instrument, "asset_class"]

        if ac in ac_norm_prices and instrument in norm_prices:
            inst_norm = norm_prices[instrument].reindex(dates).ffill()
            ac_norm   = ac_norm_prices[ac].reindex(dates).ffill()
            rel_price = inst_norm - ac_norm
            roll_mean = rel_price.rolling(VALUE_ROLLING_MEAN_WINDOW, min_periods=640).mean()
            raw_value = -(rel_price - roll_mean).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            forecast_scaled = cap_forecast(
                raw_value * rolling_forecast_scalar(raw_value, window=1280, min_periods=640)
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
