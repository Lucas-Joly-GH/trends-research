"""
VoV direction-window robustness sweep for Strategy 183.

Goal: demonstrate that the VoV direction weighting result is NOT specific
to the 64-day window -- it works across multiple lookback windows.

Sweeps direction_lb in {21, 42, 64, 126} plus a baseline with NO direction
weighting (direction = 1.0 everywhere).

ARCHITECTURE (post-15-Apr refactor):  this script is a *thin wrapper*
around `ig_strategy_183.build_master_inst_signals(cfg)`.  Every variant
reuses the canonical signal pipeline with a single cfg override:

    dir=None : {"vov_use_direction": False}
    dir=N    : {"vov_direction_lb": N, "vov_use_direction": True}

The canonical anchor (dir=64) is BYTE-IDENTICAL to the canonical master
checkpoint by construction -- no parallel signal implementation can
drift from it.  See `S183_CANONICAL_CFG` in ig_strategy_183.py for the
full knob surface.

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

TRADING_DAYS = 256
POST2010     = pd.Timestamp("2010-01-01")
DIRECTION_WINDOWS = [21, 42, 64, 126]


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


def run_variant(direction_lb, irx_series=None, label=""):
    """
    Rerun the canonical S183 pipeline with a custom direction_lb override.

    direction_lb : int or None
        Number of days for the VoV direction sign. None => direction=1.0
        everywhere (no direction weighting).
    """
    tag = f"dir={direction_lb}" if direction_lb is not None else "NO_DIR"
    name = f"S183_VoVDir_{tag}"
    print(f"\n{'='*72}")
    print(f"  VARIANT: {name}  ({label})")
    print(f"{'='*72}")

    if direction_lb is None:
        cfg = {"vov_use_direction": False}
    else:
        cfg = {"vov_use_direction": True, "vov_direction_lb": direction_lb}

    inst_signals, paths = build_master_inst_signals(cfg)
    # Route sweep output into RobustnessTests/ dir so we don't clobber the
    # canonical top-level checkpoint when the MASTER-equivalent dir=64 runs.
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
        "variant":      tag,
        "direction_lb": direction_lb if direction_lb is not None else "NONE",
        "sr_full":      _sr_sub(np.ones(len(dr), dtype=bool)),
        "sr_10":        _sr_sub(dates_arr >= POST2010),
        "cagr":         round(ck["cagr"], 4),
        "max_dd":       round(ck["max_dd"], 4),
        "calmar":       round(ck["calmar"], 4),
        "trades_yr":    round(float(ck.get("trades_yr", 0.0)), 1),
    }


def main():
    t0 = time.time()

    # Preload IRX over a broad window (sweep variants all trade the same universe)
    all_dates = pd.bdate_range("1982-01-01", "2027-12-31")
    try:
        irx_series = pd.Series(load_irx(all_dates), index=all_dates)
        print(f"  IRX loaded (mean {irx_series.mean() * 256:.2%} ann.)")
    except Exception as exc:
        print(f"  [WARN] Could not load IRX: {exc}  (using raw Sharpe)")
        irx_series = None

    results = []

    res = run_variant(None, irx_series, label="BASELINE: no direction weighting")
    if res is not None:
        results.append(res)
    for dlb in DIRECTION_WINDOWS:
        marker = " (MASTER)" if dlb == 64 else ""
        res = run_variant(dlb, irx_series, label=f"direction lookback = {dlb}{marker}")
        if res is not None:
            results.append(res)

    elapsed = time.time() - t0
    print(f"\n\n{'='*80}")
    print(f"  VoV DIRECTION WINDOW ROBUSTNESS SWEEP -- Strategy 183")
    print(f"  (elapsed: {elapsed/60:.1f} min)")
    print(f"{'='*80}\n")

    df = pd.DataFrame(results).set_index("variant")
    cols = ["direction_lb", "sr_full", "sr_10", "cagr", "max_dd", "calmar", "trades_yr"]
    print(df[cols].to_string())

    # Emit summary CSV consumed by _regen_appendix.py
    summary_path = STRATEGY_DIR / "S183_VoVDir_sweep_summary.csv"
    df.reset_index().to_csv(summary_path, index=False)
    print(f"\n  [OK] wrote {summary_path.relative_to(STRATEGY_DIR.parent)}")

    # Deltas vs MASTER (dir=64)
    if "dir=64" in df.index:
        master = df.loc["dir=64"]
        print(f"\nDeltas vs MASTER (dir=64):")
        print(f"{'variant':<12} {'dSR_full':>10} {'dSR_10':>10} {'dCAGR':>10} {'dMDD':>10} {'dCalmar':>10}")
        print("-" * 65)
        for idx in df.index:
            r = df.loc[idx]
            d_sr   = r["sr_full"] - master["sr_full"]
            d_sr10 = r["sr_10"]   - master["sr_10"]
            d_cagr = r["cagr"]    - master["cagr"]
            d_mdd  = r["max_dd"]  - master["max_dd"]
            d_cal  = r["calmar"]  - master["calmar"]
            marker = " <-- MASTER" if idx == "dir=64" else ""
            print(f"{idx:<12} {d_sr:>+10.4f} {d_sr10:>+10.4f} {d_cagr:>+10.4f} {d_mdd:>+10.4f} {d_cal:>+10.4f}{marker}")

    print(f"\n  Total elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
