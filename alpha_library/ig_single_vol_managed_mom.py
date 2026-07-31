"""
Single Alpha (transformation): Vol-Managed EWMAC Ensemble
==========================================================
Moreira & Muir (2017), "Volatility-Managed Portfolios", JF 72(4):1611
-- scaling positions by 1/sigma^2 (rather than the conventional
1/sigma) increases Sharpe out-of-sample for momentum, value, and
carry strategies.

Idea
----
EWMAC_Ensemble * scale, where scale = (sigma_target^2 / sigma_realised^2),
clipped to [0.25, 4.0] to limit leverage swings.

NOTE: this is a *transformation* of EWMAC_Ensemble -- compare against
EWMAC_Ensemble baseline (SR_Full = 0.574). The target vol is the
*instrument-level* vol target, not the portfolio target -- effectively
this rescales the forecast inversely to recent variance.
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
    align_fx, blended_volatility, compute_ewmac_raw,
    cap_forecast, run_compounded_portfolio, rolling_forecast_scalar,
)

STRATEGY_NAME = "Single_Alpha_Vol_Managed_Mom"
OOS_START     = 1280
RV_WINDOW     = 22       # ~1 month realised variance
SCALE_FLOOR   = 0.25
SCALE_CAP     = 4.0

TREND_SPEEDS = [
    {"fast": 16,  "slow": 64},
    {"fast": 64,  "slow": 256},
    {"fast": 128, "slow": 512},
]
TREND_FDM = 1.10


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

        # -- EWMAC Ensemble baseline --
        components = []
        for sp in TREND_SPEEDS:
            raw = compute_ewmac_raw(close, vol, sp["fast"], sp["slow"])
            scaled = cap_forecast(raw.fillna(0.0) * rolling_forecast_scalar(raw.fillna(0.0)))
            components.append(scaled * (1.0 / 3.0))
        ewmac_blend = (components[0] + components[1] + components[2]) * TREND_FDM
        ewmac_blend = cap_forecast(ewmac_blend.fillna(0.0))

        # -- Moreira-Muir scaling --
        # Use realised return variance over RV_WINDOW relative to its trailing
        # 5-year average ("target" var) so the scale is centered around 1.
        log_ret = np.log(close.replace(0, np.nan)).diff()
        rv_st   = (log_ret ** 2).rolling(RV_WINDOW, min_periods=RV_WINDOW // 2).mean()
        rv_lt   = rv_st.rolling(252 * 5, min_periods=252).mean()
        scale   = (rv_lt / rv_st.replace(0, np.nan))
        scale   = scale.replace([np.inf, -np.inf], np.nan).clip(SCALE_FLOOR, SCALE_CAP).fillna(1.0)
        scale   = scale.reindex(idx).fillna(1.0)

        gated = ewmac_blend.values * scale.values
        forecast_scaled = pd.Series(gated, index=idx).clip(-FORECAST_CAP, FORECAST_CAP)

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
