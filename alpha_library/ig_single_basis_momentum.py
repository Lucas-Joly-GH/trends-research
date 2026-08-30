"""
Single Alpha: Basis Momentum (Boons & Prado 2019, JF 74(1):239)
================================================================
Defined as the difference between the 12-month past return of the
front contract (roll-adjusted) and the 12-month past return of the
2nd-nearby contract (roll-adjusted). Captures the *dynamics* of the
term structure rather than its current slope:

    BM_t  =  r1_{t-252..t}  -  r2_{t-252..t}

Boons-Prado report a long-short SR around 0.5 on 21 commodity
futures, with correlation ~0.15 to slope-carry. Distinct from both:
  * Carry       (current front-back slope, level)
  * Carry_Momentum  (change in carry slope, Δlevel)

Mechanism: basis momentum isolates whether the convenience yield
curve has been steepening or flattening in a way the slope level
misses. Literature shows it carries independent information about
inventory/scarcity dynamics.

Data: uses Single_Alpha/_termstruct_helpers.load_term_structure()
to pull c1 and c2 closes on every date.
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
sys.path.insert(0, str(_HERE))
from ig_shared_config import (
    FORECAST_CAP, FORECAST_TARGET, VOL_TARGET,
    get_project_paths, load_mapping, load_fx_rates, load_instrument_data,
    align_fx, blended_volatility,
    cap_forecast, run_compounded_portfolio, rolling_forecast_scalar,
)
from _termstruct_helpers import load_term_structure

STRATEGY_NAME = "Single_Alpha_Basis_Momentum"
OOS_START = 1280
LOOKBACK = 252     # 12-month per Boons-Prado


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

        ts = load_term_structure(instrument, paths["panama"], paths["contracts"], idx, depth=2)
        if ts is None:
            # Instruments without a contracts dir (spot-like synth) cannot compute BM
            forecast_scaled = pd.Series(0.0, index=idx)
        else:
            c1 = ts["c1"]
            c2 = ts["c2"]
            # Roll-aware per-contract log returns: only keep diffs where the
            # front contract identity is the same day-over-day, otherwise set
            # to 0 (so the cumulative 252-day sum ignores roll jumps).
            # Panama provides the daily front-contract identity via the stats
            # payload; fall back to price-diff zeroing on NaN c1 segments.
            panama_file = str(Path(paths["panama"]) / f"{instrument}_continuous.csv")
            pan = pd.read_csv(panama_file, usecols=["Date","Contract"], parse_dates=["Date"])
            pan = pan.sort_values("Date").set_index("Date")
            front = pan["Contract"].reindex(idx).ffill()
            same_contract = (front == front.shift(1)).fillna(False)
            dl1 = np.log(c1).diff().where(same_contract, 0.0).fillna(0.0)
            dl2 = np.log(c2).diff().where(same_contract, 0.0).fillna(0.0)
            # 252-day cumulative roll-adjusted returns
            r1 = dl1.rolling(LOOKBACK, min_periods=LOOKBACK // 2).sum()
            r2 = dl2.rolling(LOOKBACK, min_periods=LOOKBACK // 2).sum()
            bm = (r1 - r2)
            # Normalise by 252-day realised vol of the difference series
            bm_std = (dl1 - dl2).rolling(LOOKBACK, min_periods=LOOKBACK // 2).std() * np.sqrt(LOOKBACK)
            bm_norm = bm / bm_std.replace(0, np.nan)
            bm_norm = bm_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            forecast_scaled = cap_forecast(
                bm_norm * rolling_forecast_scalar(bm_norm)
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
