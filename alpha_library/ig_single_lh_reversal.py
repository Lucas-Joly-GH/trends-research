"""
Single Alpha: Long-Horizon Time-Series Reversal (48 months)
==============================================================
Sign-flipped 4-year return. Documented in:
  De Bondt & Thaler (1985), "Does the Stock Market Overreact?", JF 40:793
  Moskowitz, Ooi, Pedersen (2012), "Time Series Momentum", JFE 104(2):228
  Table 3 shows TS momentum PnL reverses sign beyond 12 months,
  with strongest contrarian profitability around 24-60 months.

S169's trend block runs EWMAC at spans 16/64/128 (i.e. half-lives
well under 4 months). There is zero overlap with a 48-month signal
by construction; this alpha is built to be *anti-correlated* to
the existing trend leg so that even a near-zero standalone SR adds
portfolio diversification value.

Signal:
    raw_t  =  - ( log(close_t) - log(close_{t-1008}) )    # 48m ~ 1008d
    norm   =  raw / trailing_1008d_vol_of_log_price
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

STRATEGY_NAME = "Single_Alpha_LongHorizon_Reversal"
OOS_START     = 1280
LOOKBACK      = 1008   # 48 months * 21 trading days


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
        if len(dates) < OOS_START + LOOKBACK + 5:
            continue
        ccy = mapping.loc[instrument, "currency"]
        fx  = align_fx(ccy, dates, fx_daily)
        if fx is None:
            continue

        close         = data["close"]
        price_changes = data["price_changes"]
        vol           = blended_volatility(price_changes)
        idx           = pd.DatetimeIndex(dates)

        log_p = np.log(close.replace(0, np.nan))
        lh_ret = log_p.diff(LOOKBACK)
        # Normalise by the running stdev of lh_ret itself so the signal
        # is O(1) across instruments with different volatilities
        norm  = lh_ret / lh_ret.rolling(LOOKBACK, min_periods=LOOKBACK // 2).std()
        raw   = -norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
