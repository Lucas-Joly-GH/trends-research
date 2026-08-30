"""
Single Alpha: COT Positioning
==============================
CFTC Disaggregated Commitments-of-Traders Report (weekly).
Free public data via Socrata SODA endpoint:
    https://publicreporting.cftc.gov/resource/72hh-3qpy.csv

Mechanism
---------
Speculator (managed-money) crowding is a contrarian indicator. When
managed-money net long is high relative to open interest, momentum is
exhausted -- fade. Symmetric on the short side.

Signal
------
For each instrument with a CFTC mapping:
    mm_net_t  = m_money_long - m_money_short
    raw_t     = mm_net_t / open_interest_all
    z_t       = (raw_t - mean_3y) / std_3y          (rolling 156 weeks)
    forecast  = -z_t        (fade crowding)

Forward-fill weekly z-score to daily index. Apply only to instruments
with a mapping entry below; others receive 0 forecast.

Coverage: ~commodities + financials only (~30 of 63 instruments).
"""
import sys
import os
import io
import time
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

STRATEGY_NAME = "Single_Alpha_COT_Positioning"
OOS_START     = 1280

COT_DIR     = _HERE / "_cot_cache"
COT_DIR.mkdir(exist_ok=True)
COT_API     = "https://publicreporting.cftc.gov/resource/72hh-3qpy.csv"

# Map our Norgate codes to a *substring* matching the CFTC contract_market_name
# field (case-insensitive). Where no clean Disaggregated entry exists (e.g.
# financials / single-stock / equity indices) we leave the entry out -- those
# instruments get a zero forecast.
COT_NAME_MAP = {
    # Ags
    "ZC": "CORN",
    "ZS": "SOYBEANS",
    "ZL": "SOYBEAN OIL",
    "ZM": "SOYBEAN MEAL",
    "ZW": "WHEAT-SRW",
    "CC": "COCOA",
    "KC": "COFFEE C",
    "SB": "SUGAR NO. 11",
    "CT": "COTTON NO. 2",
    "RS": "CANOLA",
    "LE": "LIVE CATTLE",
    "HE": "LEAN HOGS",
    "GF": "FEEDER CATTLE",
    # Metals
    "GC": "GOLD",
    "SI": "SILVER",
    "HG": "COPPER-",
    "PL": "PLATINUM",
    "PA": "PALLADIUM",
    # Energy
    "CL": "CRUDE OIL, LIGHT SWEET-WTI",
    "BRN": "CRUDE OIL, BRENT",
    "RB": "GASOLINE RBOB",
    "HO": "NY HARBOR ULSD",
    "NG": "NAT GAS NYME",
    # Equity (S&P, Nasdaq, Russell, etc.)
    "ES": "E-MINI S&P 500",
    "NQ": "NASDAQ-100 STOCK INDEX",
    "RTY": "E-MINI RUSSELL 2000",
    "EMD": "E-MINI S&P MIDCAP 400",
    # Bonds / rates
    "ZT": "2-YEAR U.S. TREASURY",
    "ZF": "5-YEAR U.S. TREASURY",
    "ZN": "10-YEAR U.S. TREASURY",
    "ZB": "U.S. TREASURY BONDS",
    "UB": "ULTRA U.S. TREASURY",
    "SR3": "3-MONTH SOFR",
    # FX
    "6A": "AUSTRALIAN DOLLAR",
    "6B": "BRITISH POUND",
    "6C": "CANADIAN DOLLAR",
    "6E": "EURO FX",
    "6J": "JAPANESE YEN",
    "6S": "SWISS FRANC",
    "6N": "NEW ZEALAND DOLLAR",
    "6M": "MEXICAN PESO",
    "DX": "USD INDEX",
    # Vol
    "VX": "VIX FUTURES",
}


def _fetch_cot_for_instrument(name_query: str) -> pd.DataFrame:
    """
    Fetch all weekly COT rows for a contract whose name matches `name_query`
    via the SODA $where clause. Cached locally as parquet.
    """
    safe = name_query.replace(" ", "_").replace("/", "_").replace(",", "")
    cache = COT_DIR / f"{safe}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    cols = ",".join([
        "report_date_as_yyyy_mm_dd",
        "contract_market_name",
        "open_interest_all",
        "m_money_positions_long_all",
        "m_money_positions_short_all",
    ])
    where = f"upper(contract_market_name) like upper('%{name_query}%')"
    url = (f"{COT_API}?$select={cols}&$where={urllib.request.quote(where)}"
           f"&$limit=20000&$order=report_date_as_yyyy_mm_dd")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            df = pd.read_csv(io.StringIO(r.read().decode("utf-8")))
    except Exception as e:
        print(f"[WARN] COT fetch failed for '{name_query}': {e}")
        return pd.DataFrame()

    if df.empty:
        cache.write_bytes(b"")  # mark empty
        return df
    df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    df = df.rename(columns={"report_date_as_yyyy_mm_dd": "date"})
    df = df.sort_values(["contract_market_name", "date"])
    try:
        df.to_parquet(cache)
    except Exception:
        pass
    return df


def _build_cot_signal(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    """Aggregate all matching contract-name rows by date (sum across multiple
    listings under same query), then compute z-score and forward-fill to idx."""
    if df is None or df.empty:
        return pd.Series(0.0, index=idx)

    # Take the row with largest OI per (date) -- typically the front
    df["mm_net"] = (df["m_money_positions_long_all"].astype(float)
                    - df["m_money_positions_short_all"].astype(float))
    df["oi"]    = df["open_interest_all"].astype(float).replace(0, np.nan)
    grp = df.groupby("date").agg(mm_net=("mm_net", "sum"), oi=("oi", "sum")).sort_index()
    grp["raw"] = grp["mm_net"] / grp["oi"]

    # Rolling 3y (156 weeks) z-score, then sign flip (fade crowding).
    mu = grp["raw"].rolling(156, min_periods=52).mean()
    sd = grp["raw"].rolling(156, min_periods=52).std()
    z  = (grp["raw"] - mu) / sd.replace(0, np.nan)
    z  = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sig_weekly = -z       # fade

    # ffill to daily index, with 1-day lag (release lag)
    daily = sig_weekly.reindex(idx, method="ffill").shift(1).fillna(0.0)
    return daily


def run_strategy():
    paths    = get_project_paths(_HERE)
    mapping  = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])

    # Pre-fetch COT for all mapped instruments (sequential -- be polite to the API)
    print(f"\n[COT] Fetching {len(COT_NAME_MAP)} COT series...")
    cot_data = {}
    for inst, query in tqdm(COT_NAME_MAP.items(), desc="COT fetch"):
        df = _fetch_cot_for_instrument(query)
        cot_data[inst] = df
        time.sleep(0.1)  # rate limit

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

        df_cot = cot_data.get(instrument)
        if df_cot is None or df_cot.empty:
            forecast_scaled = pd.Series(0.0, index=idx)
        else:
            raw = _build_cot_signal(df_cot, idx)
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
