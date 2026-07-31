"""
Term-structure and Open-Interest helpers for Single_Alpha prototypes.

These helpers read the per-contract files in Data/Contracts/<INST>/ and
build:
  (a) load_term_structure() -- 3-deep price stack (c1, c2, c3) + years_diff
  (b) load_total_oi()       -- aggregate open interest across all active
                               contracts on each date

Everything is on the daily Panama calendar supplied by the caller.

Intentionally kept as a Single_Alpha-local helper. Promote to
ig_shared_config only after at least one alpha built on top of it lands
in a strategy.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Contract-name parsing (mirrors ig_shared_config internals)
# ----------------------------------------------------------------------
_MONTH_CODES = {"F":1,"G":2,"H":3,"J":4,"K":5,"M":6,
                "N":7,"Q":8,"U":9,"V":10,"X":11,"Z":12}


def _parse_expiry(cname: str):
    """Parse 'SYM-YYYYM' -> (year, month). Returns None on failure."""
    try:
        tail = cname.split("-")[-1]
        yr = int(tail[:4])
        mo = _MONTH_CODES.get(tail[4])
        if mo is None:
            return None
        return yr, mo
    except Exception:
        return None


def _sort_key(cname: str):
    p = _parse_expiry(cname)
    return p if p else (9999, 12)


def _months_between(y1, m1, y2, m2):
    return (y2 - y1) * 12 + (m2 - m1)


# ----------------------------------------------------------------------
# Term-structure loader
# ----------------------------------------------------------------------
def load_term_structure(
    instrument: str,
    panama_folder: str,
    contracts_folder: str,
    daily_dates: pd.DatetimeIndex,
    depth: int = 3,
):
    """
    For each date in `daily_dates`, return the closing prices of the
    front contract (c1), the next contract (c2), and optionally the
    contract beyond that (c3), along with the maturity gap in years
    from c1 to each.

    Returns dict or None:
        {
            "c1":    pd.Series,
            "c2":    pd.Series,
            "c3":    pd.Series,        # only if depth >= 3
            "yrs_12": pd.Series,       # years from c1 to c2
            "yrs_13": pd.Series,       # years from c1 to c3
        }
    None is returned if the instrument has no contracts dir or no Panama file.
    """
    panama_file = os.path.join(panama_folder, f"{instrument}_continuous.csv")
    if not os.path.exists(panama_file):
        return None

    df_pan = pd.read_csv(panama_file, usecols=["Date", "Contract"], parse_dates=["Date"])
    df_pan = df_pan.sort_values("Date").set_index("Date")
    front = df_pan["Contract"].reindex(daily_dates).ffill()

    inst_dir = os.path.join(contracts_folder, instrument)
    if not os.path.isdir(inst_dir):
        return None

    files = sorted(
        [f for f in os.listdir(inst_dir) if f.endswith(".csv") and "-" in f],
        key=lambda f: _sort_key(f.replace(".csv", "")),
    )
    names = [f.replace(".csv", "") for f in files]
    next_of = {names[i]: names[i + 1] for i in range(len(names) - 1)}
    next2_of = {names[i]: names[i + 2] for i in range(len(names) - 2)}

    # Identify which contracts we need
    needed = set()
    for c in front.dropna().unique():
        needed.add(c)
        if c in next_of:
            needed.add(next_of[c])
        if depth >= 3 and c in next2_of:
            needed.add(next2_of[c])

    prices = {}
    for c in needed:
        f = os.path.join(inst_dir, f"{c}.csv")
        if not os.path.exists(f):
            continue
        df_c = pd.read_csv(f, usecols=["Date", "Close"])
        df_c["Date"] = pd.to_datetime(df_c["Date"], format="%Y%m%d")
        df_c = df_c.dropna(subset=["Close"]).sort_values("Date").set_index("Date")
        prices[c] = df_c["Close"]

    # Years-diff maps
    y12 = {}
    y13 = {}
    for c in names:
        e1 = _parse_expiry(c)
        if e1 is None:
            continue
        if c in next_of:
            e2 = _parse_expiry(next_of[c])
            if e2:
                y12[c] = max(_months_between(*e1, *e2), 1) / 12.0
        if c in next2_of:
            e3 = _parse_expiry(next2_of[c])
            if e3:
                y13[c] = max(_months_between(*e1, *e3), 1) / 12.0

    idx = daily_dates
    c1_s = pd.Series(np.nan, index=idx)
    c2_s = pd.Series(np.nan, index=idx)
    c3_s = pd.Series(np.nan, index=idx)
    y12_s = pd.Series(np.nan, index=idx)
    y13_s = pd.Series(np.nan, index=idx)

    for cname in front.dropna().unique():
        mask = front == cname
        if cname in prices:
            c1_s[mask] = prices[cname].reindex(idx[mask]).values
        c2 = next_of.get(cname)
        if c2 and c2 in prices:
            c2_s[mask] = prices[c2].reindex(idx[mask]).values
        if cname in y12:
            y12_s[mask] = y12[cname]
        if depth >= 3:
            c3 = next2_of.get(cname)
            if c3 and c3 in prices:
                c3_s[mask] = prices[c3].reindex(idx[mask]).values
            if cname in y13:
                y13_s[mask] = y13[cname]

    out = {"c1": c1_s, "c2": c2_s, "yrs_12": y12_s}
    if depth >= 3:
        out["c3"] = c3_s
        out["yrs_13"] = y13_s
    return out


# ----------------------------------------------------------------------
# Aggregate Open-Interest loader
# ----------------------------------------------------------------------
def load_total_oi(
    instrument: str,
    contracts_folder: str,
    daily_dates: pd.DatetimeIndex,
    max_contracts: int = 200,
):
    """
    Sum Open Interest across every contract file that has data on each
    date. Returns a pd.Series aligned to `daily_dates`, or None if the
    contracts directory is missing.

    This is a proxy for aggregate speculator + commercial activity --
    distinct from nearby-only OI momentum (which was negative-SR in
    the existing Single_Alpha library).
    """
    inst_dir = os.path.join(contracts_folder, instrument)
    if not os.path.isdir(inst_dir):
        return None

    files = sorted(
        [f for f in os.listdir(inst_dir) if f.endswith(".csv") and "-" in f],
        key=lambda f: _sort_key(f.replace(".csv", "")),
    )[:max_contracts]

    total = pd.Series(0.0, index=daily_dates)
    count = pd.Series(0,   index=daily_dates)

    for f in files:
        path = os.path.join(inst_dir, f)
        try:
            df_c = pd.read_csv(path, usecols=["Date", "Open Interest"])
        except Exception:
            continue
        df_c["Date"] = pd.to_datetime(df_c["Date"], format="%Y%m%d")
        df_c = df_c.dropna(subset=["Open Interest"]).set_index("Date")
        s = df_c["Open Interest"].reindex(daily_dates).fillna(0.0)
        alive = (s > 0).astype(int)
        total = total + s
        count = count + alive

    # Mask dates with no contracts alive
    total[count == 0] = np.nan
    return total
