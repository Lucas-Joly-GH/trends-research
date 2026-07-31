"""
Smoothing half-life robustness sweep for Strategy 183.

Sweeps the master-forecast EWMA smoothing half-life across {1, 2, 3, 5, 10}
trading days.  The structural choice (hl=1 = rebalance cadence) is the
MASTER anchor.

ARCHITECTURE (post-15-Apr refactor):  this script is a *thin wrapper*
around `ig_strategy_183.build_master_inst_signals(cfg)` with a single
cfg override per variant:

    hl=N : {"smooth_mode": "halflife", "smooth_halflife": N}

The canonical anchor (hl=1) is BYTE-IDENTICAL to the canonical master
checkpoint by construction -- no parallel signal implementation can drift.

Emits per-variant checkpoints AND a summary CSV consumed by the appendix
generator (`_regen_appendix.py`).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
STRATEGY_DIR = _HERE.parent
sys.path.insert(0, str(STRATEGY_DIR))
sys.path.insert(0, str(STRATEGY_DIR.parent))

from ig_strategy_183 import build_master_inst_signals
from ig_shared_config import run_compounded_portfolio, load_irx

TRADING_DAYS    = 256
POST2010        = pd.Timestamp("2010-01-01")
HALFLIFE_VALUES = [1, 2, 3, 5, 10]


def _excess_sharpe(daily_ret, rf=None):
    r = np.asarray(daily_ret, dtype=float)
    if rf is not None:
        rf = np.asarray(rf, dtype=float)
        r = r - rf[:len(r)]
    if len(r) < 30:
        return 0.0
    s = r.std(ddof=1)
    if s < 1e-12:
        return 0.0
    return float((r.mean() * TRADING_DAYS) / (s * np.sqrt(TRADING_DAYS)))


def run_variant(smooth_halflife, irx_series=None, label=""):
    """Rerun canonical S183 with a custom smoothing halflife."""
    tag = f"hl={smooth_halflife}"
    name = f"S183_HL_{tag}"
    print(f"\n{'='*72}")
    print(f"  VARIANT: {name}  ({label})")
    print(f"{'='*72}")

    cfg = {"smooth_mode": "halflife", "smooth_halflife": smooth_halflife}
    inst_signals, paths = build_master_inst_signals(cfg)
    sweep_paths = {**paths, "output": str(_HERE)}
    ck = run_compounded_portfolio(
        inst_signals, name, sweep_paths, save_per_inst_pnl=False
    )
    if ck is None:
        return None

    dates_arr = pd.DatetimeIndex(ck["dates"])
    dr = pd.Series(ck["daily_ret"], index=dates_arr)
    rf = irx_series.reindex(dates_arr).fillna(0.0) if irx_series is not None else None

    def _sr_sub(mask):
        seg = dr[mask]
        rf_seg = rf[mask].values if rf is not None else None
        return round(_excess_sharpe(seg.values, rf_seg), 4)

    return {
        "variant":    tag,
        "halflife":   smooth_halflife,
        "sr_full":    _sr_sub(np.ones(len(dr), dtype=bool)),
        "sr_10":      _sr_sub(dates_arr >= POST2010),
        "cagr":       round(ck["cagr"], 4),
        "max_dd":     round(ck["max_dd"], 4),
        "calmar":     round(ck["calmar"], 4),
        "trades_yr":  round(float(ck.get("trades_yr", 0.0)), 1),
    }


def main():
    t0 = time.time()

    all_dates = pd.bdate_range("1982-01-01", "2027-12-31")
    try:
        irx_series = pd.Series(load_irx(all_dates), index=all_dates)
        print(f"  IRX loaded (mean {irx_series.mean() * 256:.2%} ann.)")
    except Exception as exc:
        print(f"  [WARN] Could not load IRX: {exc}  (using raw Sharpe)")
        irx_series = None

    results = []
    for hl in HALFLIFE_VALUES:
        marker = " (MASTER)" if hl == 1 else ""
        res = run_variant(hl, irx_series, label=f"halflife = {hl}{marker}")
        if res is not None:
            results.append(res)

    elapsed = time.time() - t0
    print(f"\n\n{'='*80}")
    print(f"  SMOOTHING HALF-LIFE SWEEP -- Strategy 183")
    print(f"  (elapsed: {elapsed/60:.1f} min)")
    print(f"{'='*80}\n")

    df = pd.DataFrame(results).set_index("variant")
    cols = ["halflife", "sr_full", "sr_10", "cagr", "max_dd", "calmar", "trades_yr"]
    print(df[cols].to_string())

    # Emit summary CSV consumed by _regen_appendix.py
    summary_path = STRATEGY_DIR / "S183_HL_sweep_summary.csv"
    df.reset_index().to_csv(summary_path, index=False)
    print(f"\n  [OK] wrote {summary_path.relative_to(STRATEGY_DIR.parent)}")

    # Deltas vs MASTER (hl=1)
    if "hl=1" in df.index:
        master = df.loc["hl=1"]
        print(f"\nDeltas vs MASTER (hl=1):")
        print(f"{'variant':<10} {'dSR_full':>10} {'dSR_10':>10} {'dCAGR':>10} {'dMDD':>10} {'dCalmar':>10} {'dTrades':>10}")
        print("-" * 75)
        for idx in df.index:
            r = df.loc[idx]
            d_sr   = r["sr_full"]   - master["sr_full"]
            d_sr10 = r["sr_10"]     - master["sr_10"]
            d_cagr = r["cagr"]      - master["cagr"]
            d_mdd  = r["max_dd"]    - master["max_dd"]
            d_cal  = r["calmar"]    - master["calmar"]
            d_trd  = r["trades_yr"] - master["trades_yr"]
            marker = " <-- MASTER" if idx == "hl=1" else ""
            print(f"{idx:<10} {d_sr:>+10.4f} {d_sr10:>+10.4f} {d_cagr:>+10.4f} {d_mdd:>+10.4f} {d_cal:>+10.4f} {d_trd:>+10.1f}{marker}")

    print(f"\n  Total elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
