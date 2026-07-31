"""
Single Alpha: Cross-Asset Lead-Lag
====================================
Documented commodity-currency and risk-asset/bond lead-lag pairs:
    - Copper (HG) leads AUD/USD (6A) -- China-cycle FX channel.
    - Realised SPX volatility leads US 10y bond futures (ZN) --
      flight-to-quality / VIX-bond correlation.
    - Oil (CL) leads CAD/USD (6C) -- petrocurrency channel.

References:
    Chen-Rogoff (2003) "Commodity currencies"; Connolly-Stivers-Sun
    (2005) "Stocks vs Bonds: VIX-conditional negative correlation".

Signal
------
For each (leader, follower) pair:
    leader_mom_t = sign(EMA_5(leader) - EMA_20(leader))         (+/-1)
    follower forecast at date t+lag:
        if leader is realised vol:  raw = leader_zscore   (long bonds when vol up)
        else:                       raw = leader_mom_zscore
After rolling-scalar normalisation. Lag = 2 trading days (Lo & MacKinlay
1990 lead-lag horizon).
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

STRATEGY_NAME = "Single_Alpha_LeadLag"
OOS_START     = 1280
LAG_DAYS      = 2

# (leader_inst, follower_inst, signal_kind, sign)
PAIRS = [
    ("HG", "6A", "momentum", +1),  # copper -> AUD
    ("ES", "ZN", "vol",       +1),  # SPX realised vol -> ZN (high vol -> long bonds)
    ("CL", "6C", "momentum", +1),  # oil -> CAD
]


def _leader_signal_momentum(close: pd.Series) -> pd.Series:
    """Z-scored fast-minus-slow EMA momentum on a leader's close price."""
    log_p = np.log(close.replace(0, np.nan))
    fast = log_p.ewm(span=5,  adjust=False).mean()
    slow = log_p.ewm(span=20, adjust=False).mean()
    diff = fast - slow
    mu   = diff.rolling(252, min_periods=64).mean()
    sd   = diff.rolling(252, min_periods=64).std()
    z    = (diff - mu) / sd.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _leader_signal_vol(close: pd.Series) -> pd.Series:
    """Z-scored realised vol on the leader's close (e.g. SPX -> proxy for VIX)."""
    log_ret = np.log(close.replace(0, np.nan)).diff()
    rv      = log_ret.rolling(20, min_periods=10).std()
    mu      = rv.rolling(252, min_periods=64).mean()
    sd      = rv.rolling(252, min_periods=64).std()
    z       = (rv - mu) / sd.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def run_strategy():
    paths    = get_project_paths(_HERE)
    mapping  = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])

    # Pre-compute leader signals
    leader_cache = {}
    for leader, _, kind, _ in PAIRS:
        if leader in leader_cache:
            continue
        data = load_instrument_data(leader, paths["stats"], paths["panama"])
        if data is None:
            print(f"[WARN] Cannot load leader {leader}")
            continue
        idx = pd.DatetimeIndex(data["daily_dates"])
        close = data["close"].copy()
        close.index = idx
        if kind == "momentum":
            leader_cache[(leader, "momentum")] = _leader_signal_momentum(close)
        else:
            leader_cache[(leader, "vol")] = _leader_signal_vol(close)

    # Per-follower: aggregate leader signals (signed sum)
    follower_to_signals = {}
    for leader, follower, kind, sgn in PAIRS:
        key = (leader, kind)
        if key not in leader_cache:
            continue
        sig = leader_cache[key].shift(LAG_DAYS) * sgn
        follower_to_signals.setdefault(follower, []).append(sig)

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

        if instrument in follower_to_signals:
            sigs = follower_to_signals[instrument]
            # Equal-weight blend across the leaders applied to this follower
            blended = pd.concat([s.reindex(idx) for s in sigs], axis=1).mean(axis=1).fillna(0.0)
            forecast_scaled = cap_forecast(blended * rolling_forecast_scalar(blended))
        else:
            forecast_scaled = pd.Series(0.0, index=idx)

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
