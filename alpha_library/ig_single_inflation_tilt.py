"""
Single Alpha: Inflation Tilt
==============================
Realised CPI 3-month change (CPIAUCSL from FRED) used as an imperfect
proxy for inflation surprise (true surprise = realised - consensus,
unavailable without Bloomberg).

Mechanism
---------
When inflation prints high, commodities benefit (real-asset hedge) and
nominal bonds suffer (yields rise). Bottom-quintile prints flip both.

Signal
------
For each date t:
    cpi_3m_chg_t = log(CPI_t / CPI_{t-3m})
    rank_t       = quintile rank over a 10y trailing window
    high (top 20%)   ->  +1 commodities, -1 bonds
    low  (bottom 20%) -> -1 commodities, +1 bonds
    middle quintiles -> 0

Universe: asset_class in {"OilGas","Metals","Ags","Carbon"} -> commodity
arm. asset_class in {"Bond","STIR"} -> bond arm. Others = 0.

CPI data: download CPIAUCSL.csv from FRED (free, no API key) into
Data/Macro/CPIAUCSL.csv. The file is auto-fetched on first run via
WebFetch-style requests; if unavailable the script falls back to a
hardcoded monthly series shipped with the project (if present).
"""
import sys
import os
import io
import urllib.request
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

STRATEGY_NAME = "Single_Alpha_Inflation_Tilt"
OOS_START     = 1280

CPI_URL  = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
CPI_FILE = _HERE / "_cpi_cache.csv"


def _load_cpi() -> pd.Series:
    """Load CPIAUCSL monthly series. Cache locally on first download."""
    if CPI_FILE.exists():
        df = pd.read_csv(CPI_FILE)
    else:
        try:
            req = urllib.request.Request(CPI_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(raw))
            df.to_csv(CPI_FILE, index=False)
            print(f"[INFO] Cached CPI to {CPI_FILE}")
        except Exception as e:
            raise RuntimeError(f"Cannot fetch CPI from FRED: {e}")

    # FRED columns: 'observation_date' (or 'DATE') and 'CPIAUCSL'
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    s = df["CPIAUCSL"].astype(float)
    return s


def _compute_cpi_signal(cpi: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """
    For each daily date, compute the most-recently-available CPI 3m log change,
    then return its rolling-10y quintile rank scaled to [-1, +1].
    Top quintile -> +1, bottom -> -1, middle three -> 0.
    """
    cpi_3m = np.log(cpi).diff(3)
    # Use only reported values up to date t (lag 1 month for release timing)
    cpi_3m_lag = cpi_3m.shift(1)

    # Rolling 10y (=120 months) quintile rank as of each month-end
    def _quintile_score(window):
        v = window[-1]
        if np.isnan(v):
            return 0.0
        # rank within the window
        valid = window[~np.isnan(window)]
        if len(valid) < 24:
            return 0.0
        pct = (valid <= v).mean()  # percentile in [0,1]
        if pct >= 0.80:
            return 1.0
        if pct <= 0.20:
            return -1.0
        return 0.0

    rolled = cpi_3m_lag.rolling(120, min_periods=24).apply(_quintile_score, raw=True)
    monthly_sig = rolled.fillna(0.0)

    # Forward-fill to daily
    daily = monthly_sig.reindex(idx, method="ffill").fillna(0.0)
    return daily


def run_strategy():
    paths    = get_project_paths(_HERE)
    mapping  = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])

    cpi = _load_cpi()

    commodity_classes = {"OilGas", "Metals", "Ags", "Carbon"}
    bond_classes      = {"Bond", "STIR"}

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
        ac            = mapping.loc[instrument, "asset_class"]

        cpi_score = _compute_cpi_signal(cpi, idx)

        if ac in commodity_classes:
            raw = cpi_score.copy()        # +1 long when CPI hot, -1 short when CPI cold
        elif ac in bond_classes:
            raw = -cpi_score              # opposite for bonds
        else:
            raw = pd.Series(0.0, index=idx)

        if raw.abs().sum() > 0:
            forecast_scaled = cap_forecast(raw * rolling_forecast_scalar(raw))
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
