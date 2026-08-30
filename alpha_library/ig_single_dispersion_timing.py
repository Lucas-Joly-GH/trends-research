"""
Single Alpha: Dispersion Timing -- Cross-sectional return dispersion
=====================================================================
When cross-sectional dispersion is high (instruments diverging),
trend/momentum signals are more valuable. When low (convergence),
fade existing trends.

Signal: sign(63d return) * z-score(XS_dispersion vs its 2yr average)
High dispersion amplifies trend conviction; low dispersion dampens it.

Reference: Morgan Stanley Counterpoint Global, "Dispersion and Alpha
Conversion" (2023); Stivers & Sun (2010) "Cross-Sectional Return
Dispersion and Time Variation in Value and Momentum Premiums".
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

STRATEGY_NAME = "Single_Alpha_Dispersion_Timing"
OOS_START = 1280


def run_strategy():
    paths    = get_project_paths(_HERE)
    mapping  = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])

    # ── PASS 1: load all instrument returns ──
    print("[PASS 1] Loading all instruments...")
    ret_dict = {}
    inst_data_cache = {}

    for instrument in tqdm(mapping.index, desc="Loading"):
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

        # Vol-normalised 20d returns
        ret_20 = data["close"].pct_change(20).reindex(idx).fillna(0.0)
        ret_dict[instrument] = ret_20
        inst_data_cache[instrument] = (data, vol, fx)

    # ── PASS 2: cross-sectional dispersion ──
    print("[PASS 2] Computing cross-sectional dispersion...")
    ret_panel = pd.DataFrame(ret_dict)
    xs_disp = ret_panel.std(axis=1)  # daily cross-sectional std

    # Z-score vs 2-year average
    disp_mean = xs_disp.rolling(504, min_periods=126).mean()
    disp_std  = xs_disp.rolling(504, min_periods=126).std().replace(0, np.nan)
    disp_z = ((xs_disp - disp_mean) / disp_std).fillna(0.0)

    # ── PASS 3: build signals ──
    print("[PASS 3] Building Dispersion Timing signals...")
    inst_signals = {}
    for instrument in tqdm(list(inst_data_cache.keys()), desc=STRATEGY_NAME):
        data, vol, fx = inst_data_cache[instrument]
        dates = data["daily_dates"]
        idx   = pd.DatetimeIndex(dates)

        close = data["close"]
        direction = np.sign(close.pct_change(63).fillna(0.0))
        dz = disp_z.reindex(idx).fillna(0.0)

        # High dispersion + trend = amplified signal
        raw = (direction * dz).fillna(0.0)
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
