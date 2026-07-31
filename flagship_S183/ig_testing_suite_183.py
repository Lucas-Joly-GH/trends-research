"""
Strategy 183 -- Full Thesis-Grade & Live-Deployment Statistical Testing Suite
==============================================================================

Single-file, self-contained statistical testing artifact for Strategy 183, the
test-informed parsimonious+sharpened successor to S182. This suite is a direct
adaptation of ig_testing_suite_182.py and inherits the post-bug-fix engine
(patches 1-4) that surfaces subperiod dSRs and interaction effects.

S183 deltas vs S182 (all directly measured in the S182 suite's GROUP 10):
  * use_conviction    : True -> False   (conviction ramp removed)
  * w_ts / w_xs       : 0.5/0.5 -> 1.0/0.0  (trend = 100% TS-EWMAC)
  * sigmoid_steepness : 1.0 -> 10.0     ("S179 cliff-edge")
  * smooth_span       : 5 -> 3
Load-bearing retentions: pooled 4x4 FDM, VoV direction overlay, 3-speed
TS-EWMAC ensemble, Quad 25/25/25/25 blend.

What differs from the S182 suite (adaptation checklist)
-------------------------------------------------------
  1. S183 defaults reflect the S183 architecture (see above).

  2. The master-forecast pipeline in dm_build_variant_signals and
     abl_build_inst_signals is BYTE-UNCHANGED from the S182 suite; it was
     already the pipeline that ig_strategy_183.py mirrors (VoV 252/63,
     NaN fills on every alpha, sum-then-cap trend blend).

  3. ABL_VARIANTS GROUP 1a rewritten: instead of "revert each S182 delta
     toward S179", we now have "revert each S183 delta toward S182"
     (R_S183_ADD_CONV, R_S183_ADD_XS, R_S183_STEEP1, R_S183_SMOOTH5,
     R_S183_FULL_REVERT). GROUP 1b retains FDM-cap and shifted-sigmoid
     forward-direction probes plus "middle ground" values (STEEP5,
     SMOOTH7, STEEP20, SMOOTH1).

  4. GROUP 9/10 combos rewritten for S183.  GROUP 9 tests pairwise and
     three-way reverts of S183's four deltas (do they interact? is any
     single delta the main driver?).  GROUP 10 tests S183 paired with
     removals of S182/S179 load-bearing components (vov direction,
     pooled FDM, overlays, smoothing).

  5. All paths, prefixes, and the STRATEGY_NAME_MASTER constant point
     at Strategy_183/TestingSuite/ and the S183 checkpoint name.

Suite phases
------------
  1. DATAMINING  -- 18-axis (A..R) one-at-a-time parameter sweep around
                    the S183 MASTER anchor. Axes inherited from S182 suite.

  2. ABLATION    -- 70-variant attribution matrix: GROUP 1a/1b delta
                    reverts, GROUP 2 FDM mode, GROUP 3 alpha components,
                    GROUP 4 infrastructure, GROUP 5 overlay sensitivity,
                    GROUP 6 weight sensitivity, GROUP 7 cost sensitivity,
                    GROUP 8 adversarial, GROUP 9 S183 delta interactions,
                    GROUP 10 S183 x load-bearing component interactions.

  3. ANALYSIS    -- Thesis-grade statistics battery run over MASTER + DM +
                    Ablation checkpoints: summary CSVs (with dsr_10 /
                    dsr_15 / dsr_20 columns), JKM paired z (full + post-2010
                    + post-2015 subperiods), LW block bootstrap, subperiod
                    stability, per-asset-class SR decomposition, Deflated
                    Sharpe (BHY/BLP), stability scatter + per-axis heatmaps,
                    sneak-under detection table, combo interaction table,
                    final verdict.

  4. PARSIMONY   -- T23-T28 progressive parsimony + random-parameter MC
                    + identity strategy + coarse rounding + parameter scramble.

  5. ROBUSTNESS  -- T29-T34 T+1 lag, instrument leave-one-out, signal noise
                    injection, leverage scaling, IRX attribution, decade SRs.

Statistical stack
-----------------
  * Newey-West HAC standard errors for Sharpe ratios
  * Jobson-Korkie (1981) / Memmel (2003) paired SR z-test (full + subperiods)
  * Ledoit-Wolf (2008) circular block bootstrap, B=5000, block=21
  * Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio
  * IRX risk-free excess returns for all SRs

Output tree  (all under Strategy_183/TestingSuite/)
  Datamining/ : DM183_<ID>_checkpoint.pkl, _portfolio_returns.csv
  Ablation/   : S183_Ablation_<ID>_checkpoint.pkl, _portfolio_returns.csv
  Analysis/   : datamining_summary.csv, ablation_summary.csv,
                dm_significance.csv, ablation_significance.csv,
                master_stability.csv, master_by_asset_class.csv,
                deflated_sharpe.csv, datamining_stability_scatter.png,
                datamining_heatmaps.png
  Parsimony/  : t23*..t28* outputs
  Robustness/ : t29*..t34* outputs

Usage
-----
  py ig_testing_suite_183.py --all                 # everything (default)
  py ig_testing_suite_183.py --dm                  # datamining only
  py ig_testing_suite_183.py --ablation            # ablation only
  py ig_testing_suite_183.py --analysis            # stats from checkpoints
  py ig_testing_suite_183.py --all --skip-existing # resume
  py ig_testing_suite_183.py --metrics-only        # analysis from ckpts only
  py ig_testing_suite_183.py --variants R_S183_ADD_CONV R_S183_STEEP1
  py ig_testing_suite_183.py --seed 20260411 --boot-B 5000 --block 21

Lineage
-------
  S183 = S182 - conviction - XS + sigmoid steepness 1->10 + smooth 5->3.
         Every delta measured in COMBO_NOCONV_NOXS_STEEP10_SMOOTH3 (S182
         ablation GROUP 10).  Pooled FDM retained, VoV direction retained.
  S182 = S179 with 3 cost-free deltas (TREND_FDM=1.0, SMOOTH_SPAN=5,
         DD_THRESHOLD=-VOL_TARGET/2).  Frozen replica of the S180 Apr-7
         16:36 checkpoint.
  S179 = S172 + J_tighter risk overlay bundle.
  S172 = S169 + FDM_CAP 1.50 -> 1.80.
  S169 = S168 + J_tighter risk overlay bundle.
  S168 = S161 + universe-pooled 4-way FDM.
"""
from __future__ import annotations

__version__ = "1.0.0-S183"
# Adapted from ig_testing_suite_180.py 2026-04-10 for Strategy_183.

import argparse
import json
import math
import multiprocessing
import os
import pickle
import sys
import traceback
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE      = Path(__file__).resolve().parent          # Strategy_183/
_BACKTEST  = _HERE.parent                              # IG_Backtest/
sys.path.insert(0, str(_BACKTEST))

from ig_shared_config import (
    STARTING_CAPITAL, VOL_TARGET, FORECAST_TARGET, FORECAST_CAP,
    ANNUALISE_DAILY,
    get_project_paths, load_mapping, load_fx_rates, load_instrument_data,
    align_fx, blended_volatility, compute_ewmac_raw,
    compute_carry, cap_forecast, rolling_forecast_scalar,
    run_compounded_portfolio, load_irx,
)

# Post-15-Apr refactor: every S183 signal build goes through the canonical
# `assemble_signals_from_alphas` in ig_strategy_183.  The DM and ABL
# variant builders below project their pre-baked libraries into the
# expected alpha_bundle shape and call this function -- they no longer
# re-implement Stage B (XS z-score, trend sub-blend, pooled FDM) or
# Stage C (smoothing, vol/DD gates).  See _s183cfg_to_canonical() for
# the S183_CFG -> canonical cfg knob mapping.
from ig_strategy_183 import (
    S183_CANONICAL_CFG,
    assemble_signals_from_alphas,
    self_test_canonical,
)

from scipy import stats as sp_stats

TRADING_DAYS = 256

# ---------------------------------------------------------------------------
# Output tree
# ---------------------------------------------------------------------------
SUITE_DIR      = _HERE / "TestingSuite"
DM_DIR         = SUITE_DIR / "Datamining"
ABLATION_DIR   = SUITE_DIR / "Ablation"
ANALYSIS_DIR   = SUITE_DIR / "Analysis"
FAILED_LOG     = SUITE_DIR / "suite_failures.log"


def ensure_dirs():
    for d in (SUITE_DIR, DM_DIR, ABLATION_DIR, ANALYSIS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _log_failure(stage: str, vid: str, exc: BaseException):
    msg = f"[{stage}] {vid} FAILED: {type(exc).__name__}: {exc}\n"
    tb  = traceback.format_exc()
    ensure_dirs()
    with open(str(FAILED_LOG), "a", encoding="utf-8") as fh:
        fh.write(msg)
        fh.write(tb)
        fh.write("\n")
    print("  " + msg.strip())


def _log_failure_raw(stage: str, vid: str, exc_type: str, exc_msg: str, tb_str: str):
    """Log a failure that was caught in a worker process (no live exception object)."""
    msg = f"[{stage}] {vid} FAILED: {exc_type}: {exc_msg}\n"
    ensure_dirs()
    with open(str(FAILED_LOG), "a", encoding="utf-8") as fh:
        fh.write(msg)
        fh.write(tb_str)
        fh.write("\n")
    print("  " + msg.strip())


# ---------------------------------------------------------------------------
# Parallel worker infrastructure
# (globals are populated once per worker process by the pool initialiser)
# ---------------------------------------------------------------------------

_W_LIBRARY  = None   # Phase-0 DM library  (DM + Parsimony workers)
_W_ABL_SIGS = None   # Pre-computed ablation raw signals (Ablation workers)
_W_PATHS    = None   # Project-paths dict


def _pool_init_dm(library, paths):
    """Initialiser for datamining / parsimony worker processes."""
    global _W_LIBRARY, _W_PATHS
    _W_LIBRARY = library
    _W_PATHS   = paths


def _pool_init_abl(raw_signals, paths):
    """Initialiser for ablation worker processes."""
    global _W_ABL_SIGS, _W_PATHS
    _W_ABL_SIGS = raw_signals
    _W_PATHS    = paths


def _dm_worker(task):
    """
    Execute one DM variant in a worker process.
    task = (vid, cfg, skip_existing)
    Returns (vid, status, err_triple_or_None)
    """
    vid, cfg, skip_existing = task
    if skip_existing and _dm_ck_path(vid).exists():
        return vid, "skip", None
    try:
        inst_signals = dm_build_variant_signals(_W_LIBRARY, cfg, variant_id=vid)
        dm_paths = {**_W_PATHS, "output": str(DM_DIR)}
        run_compounded_portfolio(inst_signals, f"DM183_{vid}", dm_paths)
        return vid, "ok", None
    except Exception as exc:
        return vid, "fail", (type(exc).__name__, str(exc), traceback.format_exc())


def _abl_worker(task):
    """
    Execute one ablation variant in a worker process.
    task = (vid, cfg, skip_existing)
    Returns (vid, status, err_triple_or_None)
    """
    vid, cfg, skip_existing = task
    if skip_existing and _abl_ck_path(vid).exists():
        return vid, "skip", None
    try:
        inst_signals = abl_build_inst_signals(_W_ABL_SIGS, cfg, variant_id=vid)
        abl_paths = {**_W_PATHS, "output": str(ABLATION_DIR)}
        run_compounded_portfolio(inst_signals, f"S183_Ablation_{vid}", abl_paths)
        return vid, "ok", None
    except Exception as exc:
        return vid, "fail", (type(exc).__name__, str(exc), traceback.format_exc())


def _par_worker(task):
    """
    Execute one parsimony variant (T24/T25) in a worker process.
    task = (vid, cfg)
    Returns (vid, ck_dict_or_None, err_triple_or_None)
    """
    vid, cfg = task
    try:
        inst_signals = dm_build_variant_signals(_W_LIBRARY, cfg, variant_id=vid)
        par_paths = {**_W_PATHS, "output": str(PARSIMONY_DIR)}
        ck = run_compounded_portfolio(
            inst_signals, f"PAR_{vid}", par_paths,
            save_per_inst_pnl=False,
        )
        return vid, ck, None
    except Exception as exc:
        return vid, None, (type(exc).__name__, str(exc), traceback.format_exc())


# ===========================================================================
# SECTION 2 -- S183 DEFAULTS
# Cross-checked against Strategy_183/ig_strategy_183.py.
# S183 = S182 - conviction ramp - XS momentum + sigmoid steep 1->10
#        + smooth_span 5->3; pooled 4x4 FDM retained; VoV direction retained.
# Every delta is directly measured as COMBO_NOCONV_NOXS_STEEP10_SMOOTH3
# (GROUP 10) in the S182 ablation matrix.
# ===========================================================================

FDM_MODE_PERINST = "per_inst"
FDM_MODE_POOLED  = "pooled"
FDM_MODE_SHRINK  = "shrink"
FDM_MODE_CLIP    = "clip"

S183_CFG = dict(
    speed_pairs       = [(16, 64), (32, 128), (64, 256)],   # v3.4 Triple (L&P 2016 canonical)
    speed_weights     = [1.0 / 3, 1.0 / 3, 1.0 / 3],
    # S183: conviction ramp REMOVED (was ON in S182). Inside COMBO_NOCONV_NOXS
    # plus STEEP10 + SMOOTH3 the removal is a net positive (dSR_full=+0.053
    # vs S182 TOP-LEVEL, directly measured).
    use_conviction    = False,
    conviction_window = 256,           # unused when use_conviction=False
    trend_fdm         = 1.0,           # identity (no intra-alpha diversif bonus)
    # S183 v2: 50/50 TS/XS trend sub-blend RETAINED from S182 -- the
    # S183 v1 XS removal was reverted after the v1 suite showed
    # R_S183_ADD_XS produced the highest sr_10 in the 70-variant matrix
    # (1.010, dSR_10 = +0.039 vs v1 MASTER).
    w_ts              = 0.5,
    w_xs              = 0.5,
    xs_lookback       = 256,           # base-2 canonical year
    w_trend           = 0.25,
    w_carry           = 0.25,
    w_skew            = 0.25,
    w_vov             = 0.25,
    # Pooled 4x4 FDM RETAINED. D_100 ablation (static FDM=1.0) showed
    # dSR_15 = -0.055 (jkm_p=0.002) -- load-bearing post-2015 even though
    # the full-sample cost looks free.
    fdm_corr_span     = 512,
    fdm_min_periods   = 256,
    fdm_floor         = 1.0,
    fdm_cap           = 2.0,           # = sqrt(4) = sqrt(N_QUAD_ALPHAS), zero-corr benchmark cap
    fdm_mode          = FDM_MODE_POOLED,
    skew_window       = 256,           # base-2 canonical year
    # S183 v3.2: SMOOTH_SPAN = 3 (ablation-selected optimum).  The
    # v3 suite R_S183_SMOOTH3 showed +0.015 on sr_10 vs smooth=5, and
    # v2 R_S183_SMOOTH5 showed the same direction (smooth=3 preferred).
    # Neither integer value is structurally simpler so no parsimony is
    # sacrificed by retaining the data-optimal 3.
    smooth_span       = 3,
    # S183 delta: SIGMOID_STEEPNESS 1.0 -> 10.0 ("S179 cliff-edge").
    # The sharp logistic approximates binary on/off for the vol/DD gates.
    sigmoid_steepness = 10.0,
    use_shifted_sigmoid = False,       # standard (non-shifted) sigmoid
    # Overlay bundle -- unchanged vs S182.
    vol_trigger       = 1.0,
    vol_dampen        = 0.50,
    dd_threshold      = -(0.20 / 2.0), # = -0.10 (derived from VOL_TARGET)
    dd_scale          = 0.50,
    vov_window        = 64,            # base-2 canonical quarter
    # VoV direction overlay RETAINED. R_NO_VOVDIR: dSR_10 = -0.28 -- single
    # largest cliff in the parsimony sweep.
    vov_use_direction = True,
)

OOS_START           = 1280
LOCAL_SCALAR_WINDOW = OOS_START
VOL_SCALE_LOOKBACK  = 64
DD_LOOKBACK         = 64
SIGMOID_STEEPNESS   = 10.0            # S183 default (cliff-edge)
XS_MIN_INSTS        = 3

POST2000 = pd.Timestamp("2000-01-01")
POST2005 = pd.Timestamp("2005-01-01")
POST2010 = pd.Timestamp("2010-01-01")
POST2015 = pd.Timestamp("2015-01-01")
POST2020 = pd.Timestamp("2020-01-01")

STRATEGY_NAME_MASTER = "Strategy_183_IG_VoV_Quad_Sharpened"

# Runtime config populated from CLI
RT_SEED   = 20260405
RT_BOOT_B = 5000
RT_BLOCK  = 21


# ===========================================================================
# SECTION 6 -- Statistical helpers (self-contained)
# ===========================================================================

def newey_west_se(x, lags=None):
    """Newey-West HAC standard error of the sample mean of x."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    if T < 10:
        return float(np.sqrt(max(np.var(x, ddof=1) / max(T, 1), 1e-20)))
    if lags is None:
        lags = max(1, int(np.floor(0.75 * T ** (1.0 / 3.0))))
    d = x - x.mean()
    gamma0 = np.mean(d * d)
    s = gamma0
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        gj = np.mean(d[j:] * d[:-j])
        s += 2.0 * w * gj
    s = max(s / T, 1e-20)
    return float(np.sqrt(s))


def sharpe_with_hac(excess_ret):
    """Annualised Sharpe + Newey-West HAC SE, t-stat, one-sided p."""
    r = np.asarray(excess_ret, dtype=float)
    T = len(r)
    if T < 30 or r.std() < 1e-12:
        return dict(sr=0.0, se_hac=np.nan, t_hac=np.nan, p_hac=np.nan, n=T)
    mu    = r.mean()
    sigma = r.std(ddof=1)
    sr_ann = (mu / sigma) * np.sqrt(TRADING_DAYS)
    se_mu  = newey_west_se(r)
    se_sr_ann = (se_mu / sigma) * np.sqrt(TRADING_DAYS)
    t_hac = sr_ann / se_sr_ann if se_sr_ann > 0 else np.nan
    p_hac = 1.0 - sp_stats.norm.cdf(t_hac) if not np.isnan(t_hac) else np.nan
    return dict(sr=float(sr_ann), se_hac=float(se_sr_ann),
                t_hac=float(t_hac) if not np.isnan(t_hac) else np.nan,
                p_hac=float(p_hac) if not np.isnan(p_hac) else np.nan,
                n=T)


def jkm_paired_z(ra, rb):
    """
    Jobson-Korkie (1981) paired Sharpe test with Memmel (2003) correction.
    ra = candidate, rb = reference (MASTER). Returns annualised dSR, z, p, rho.
    """
    ra = np.asarray(ra, dtype=float)
    rb = np.asarray(rb, dtype=float)
    n = len(ra)
    if n < 30 or ra.std() < 1e-12 or rb.std() < 1e-12:
        return dict(sr_a=0.0, sr_b=0.0, dsr_ann=0.0, rho=0.0,
                    z=np.nan, p=np.nan, se_ann=np.nan, n=n)
    mu1, mu2 = ra.mean(), rb.mean()
    v1, v2   = ra.var(ddof=0), rb.var(ddof=0)
    s1, s2   = np.sqrt(v1), np.sqrt(v2)
    cov      = np.cov(ra, rb, ddof=0)[0, 1]
    rho      = cov / (s1 * s2) if (s1 > 0 and s2 > 0) else 0.0
    sr1_d    = mu1 / s1 if s1 > 0 else 0.0
    sr2_d    = mu2 / s2 if s2 > 0 else 0.0
    # Memmel (2003): includes rho^2 in the (sr1*sr2) cross-term
    theta = (
        2.0 * (1.0 - rho)
        + 0.5 * (sr1_d ** 2 + sr2_d ** 2 - 2.0 * sr1_d * sr2_d * rho ** 2)
    ) / n
    se = np.sqrt(theta) if theta > 0 else np.nan
    z  = (sr1_d - sr2_d) / se if se and se > 0 else np.nan
    if not np.isnan(z):
        p = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))
    else:
        p = np.nan
    return dict(
        sr_a=float(sr1_d * np.sqrt(TRADING_DAYS)),
        sr_b=float(sr2_d * np.sqrt(TRADING_DAYS)),
        dsr_ann=float((sr1_d - sr2_d) * np.sqrt(TRADING_DAYS)),
        rho=float(rho),
        z=float(z) if not np.isnan(z) else np.nan,
        p=float(p) if not np.isnan(p) else np.nan,
        se_ann=float(se * np.sqrt(TRADING_DAYS)) if se and not np.isnan(se) else np.nan,
        n=n,
    )


def lw_block_bootstrap(ra, rb, B=None, block=None, seed=None):
    """
    Ledoit-Wolf (2008) circular block bootstrap on the paired SR
    difference (annualised). Returns dict with dsr_ann, ci_lo, ci_hi,
    p_boot, block, B.
    """
    if B is None:
        B = RT_BOOT_B
    if seed is None:
        seed = RT_SEED
    ra = np.asarray(ra, dtype=float)
    rb = np.asarray(rb, dtype=float)
    n  = len(ra)
    if n < 100:
        return dict(dsr_ann=0.0, ci_lo=np.nan, ci_hi=np.nan,
                    p_boot=np.nan, block=0, B=B)
    if block is None:
        block = RT_BLOCK if RT_BLOCK else max(5, int(np.floor(n ** (1.0 / 3.0))))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    pair = np.column_stack([ra, rb])

    def _sr_ann(x):
        s = x.std(ddof=0)
        return (x.mean() / s) * np.sqrt(TRADING_DAYS) if s > 1e-12 else 0.0

    dsr0 = _sr_ann(ra) - _sr_ann(rb)
    diffs = np.empty(B)
    pool = np.arange(n)
    for b in tqdm(range(B), desc="LW Bootstrap", leave=False, unit="draw"):
        starts = rng.choice(pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        samp = pair[idx]
        diffs[b] = _sr_ann(samp[:, 0]) - _sr_ann(samp[:, 1])
    ci_lo, ci_hi = np.quantile(diffs, [0.025, 0.975])
    p_boot = 2.0 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return dict(
        dsr_ann=float(dsr0),
        ci_lo=float(ci_lo),
        ci_hi=float(ci_hi),
        p_boot=float(p_boot),
        block=int(block),
        B=int(B),
    )


def single_sr_bootstrap_ci(r, B=None, block=None, seed=None):
    """Circular-block bootstrap CI for a single annualised SR."""
    if B is None:
        B = RT_BOOT_B
    if seed is None:
        seed = RT_SEED
    r = np.asarray(r, dtype=float)
    T = len(r)
    if T < 100 or r.std() < 1e-12:
        return dict(sr=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    if block is None:
        block = RT_BLOCK if RT_BLOCK else max(5, int(np.floor(T ** (1.0 / 3.0))))
    rng = np.random.default_rng(seed)
    obs_sr = (r.mean() / r.std(ddof=0)) * np.sqrt(TRADING_DAYS)
    boot = np.empty(B)
    n_blocks = int(np.ceil(T / block))
    for b in tqdm(range(B), desc="SR CI Boot", leave=False, unit="draw"):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) % T for s in starts])[:T]
        rr = r[idx]
        s = rr.std(ddof=0)
        boot[b] = (rr.mean() / s) * np.sqrt(TRADING_DAYS) if s > 1e-12 else 0.0
    return dict(
        sr=float(obs_sr),
        ci_lo=float(np.quantile(boot, 0.025)),
        ci_hi=float(np.quantile(boot, 0.975)),
    )


def deflated_sharpe(sr, K, T, skew=0.0, kurt=3.0):
    """
    Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    Following BLP (2014, JPM), the expected maximum of K i.i.d.
    *studentized* Sharpe-ratio estimates is

        E[max z]_K = (1 - gamma) * Phi^-1(1 - 1/K)
                    +       gamma * Phi^-1(1 - 1/(K*e))

    where z_i = sr_i / sigma(sr_i) is the Mertens-adjusted studentized
    Sharpe. The deflation z-statistic compares the *observed*
    studentized Sharpe with this expected maximum, i.e.

        sigma(sr) = sqrt( (1 - skew*sr + ((kurt-1)/4)*sr^2) / (T-1) )
        z_obs     = sr / sigma(sr)
        dsr_z     = z_obs - E[max z]_K
        dsr_p     = 1 - Phi(dsr_z)

    sr, skew, kurt are expressed in DAILY units (unnormalised). Returned
    fields:

        sr          : observed daily Sharpe (copy of input)
        sigma_sr    : SE of the daily Sharpe estimator (Mertens)
        z_obs       : studentized observed Sharpe
        e_max       : expected max of K studentized SRs (unit-less z)
        e_max_sr    : e_max expressed back in daily-SR units
                      (= e_max * sigma_sr; useful for reporting)
        dsr_z, dsr_p: deflated-Sharpe test statistic and p-value
    """
    out = dict(sr=float(sr), sigma_sr=np.nan, z_obs=np.nan,
               e_max=np.nan, e_max_sr=np.nan,
               dsr_z=np.nan, dsr_p=np.nan, K=int(K), T=int(T))
    if K < 2 or T < 30:
        return out
    gamma = 0.5772156649015329  # Euler-Mascheroni
    try:
        z1 = sp_stats.norm.ppf(1.0 - 1.0 / K)
        z2 = sp_stats.norm.ppf(1.0 - 1.0 / (K * np.e))
    except Exception:
        return out
    e_max = (1 - gamma) * z1 + gamma * z2       # unit-less z-score
    denom = max(1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr * sr, 1e-12)
    sigma_sr = float(np.sqrt(denom / max(T - 1, 1)))  # daily SR SE
    if sigma_sr <= 0:
        return out
    z_obs = float(sr / sigma_sr)                 # studentized observed SR
    dsr_z = float(z_obs - e_max)
    dsr_p = float(1.0 - sp_stats.norm.cdf(dsr_z))
    out.update(dict(
        sigma_sr = sigma_sr,
        z_obs    = z_obs,
        e_max    = float(e_max),
        e_max_sr = float(e_max * sigma_sr),
        dsr_z    = dsr_z,
        dsr_p    = dsr_p,
    ))
    return out


def max_drawdown(equity):
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (peak - eq) / np.where(peak > 0, peak, np.nan)
    return float(np.nanmax(dd)) if np.any(np.isfinite(dd)) else 0.0


def subperiod_metrics(daily_ret, periods, irx=None):
    """
    Compute per-subperiod SR (annualised), HAC-SE, t, CAGR, MDD, LW bootstrap
    CI. `periods` = OrderedDict[label -> (lo_ts, hi_ts)] with None open ends.
    """
    dr = pd.Series(daily_ret).copy()
    dr.index = pd.DatetimeIndex(dr.index)
    if irx is not None:
        rf = pd.Series(irx).reindex(dr.index).fillna(0.0)
        ex = dr - rf
    else:
        ex = dr
    rows = []
    for lbl, (lo, hi) in periods.items():
        seg = ex.copy()
        if lo is not None:
            seg = seg[seg.index >= lo]
        if hi is not None:
            seg = seg[seg.index < hi]
        if len(seg) < 60:
            rows.append(dict(period=lbl, n_days=len(seg),
                             sr_ann=np.nan, nw_hac_se=np.nan, sharpe_t=np.nan,
                             cagr=np.nan, max_dd=np.nan,
                             lw_sr_ci_lo=np.nan, lw_sr_ci_hi=np.nan))
            continue
        h = sharpe_with_hac(seg.values)
        eq = (1.0 + seg).cumprod()
        if len(eq) > 20 and eq.iloc[0] > 0:
            ny = len(eq) / TRADING_DAYS
            cagr = float(eq.iloc[-1] ** (1.0 / ny) - 1.0)
        else:
            cagr = np.nan
        mdd = max_drawdown(eq.values)
        b = single_sr_bootstrap_ci(seg.values)
        rows.append(dict(
            period=lbl, n_days=int(len(seg)),
            sr_ann=round(h["sr"], 4),
            nw_hac_se=round(h["se_hac"], 4) if not np.isnan(h["se_hac"]) else np.nan,
            sharpe_t=round(h["t_hac"], 4) if not np.isnan(h["t_hac"]) else np.nan,
            cagr=round(cagr, 4) if not np.isnan(cagr) else np.nan,
            max_dd=round(mdd, 4),
            lw_sr_ci_lo=round(b["ci_lo"], 4) if not np.isnan(b["ci_lo"]) else np.nan,
            lw_sr_ci_hi=round(b["ci_hi"], 4) if not np.isnan(b["ci_hi"]) else np.nan,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 6a -- Supplementary block-bootstrap primitives
#
# Politis-White (2004) automatic block length, Romano-Wolf (2005) stepdown,
# Hansen (2005) SPA test, block-bootstrap CI on max drawdown. These are
# invoked from run_analysis() via _save_extra_bootstrap() and complement the
# JKM + LW + DSR primary battery above.
# ---------------------------------------------------------------------------

def _circular_block_indices(n, block, rng):
    """Circular block bootstrap index generator (Politis-Romano 1992)."""
    if block < 1:
        block = 1
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.empty(n_blocks * block, dtype=np.int64)
    for k in range(n_blocks):
        s = int(starts[k])
        for j in range(block):
            idx[k * block + j] = (s + j) % n
    return idx[:n]


def politis_white_optimal_block(x):
    """
    Politis & White (2004) "Automatic Block-Length Selection for the
    Dependent Bootstrap". Plug-in estimator of the stationary-bootstrap
    optimal block length b_opt for the series `x`. Returns (b_opt, info).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    var0 = float(np.dot(x, x) / n) if n > 0 else 0.0
    if var0 < 1e-18 or n < 20:
        return 1, dict(var0=var0, m_hat=0, n=n)

    Kn = max(5, int(math.ceil(2.0 * math.sqrt(math.log10(max(n, 10))))))
    max_lag = min(n - 1, int(math.ceil(math.sqrt(n))) + 2 * Kn)

    rho = np.empty(max_lag + 1)
    rho[0] = 1.0
    for k in range(1, max_lag + 1):
        rho[k] = float(np.dot(x[:-k], x[k:]) / ((n - k) * var0))

    c = 2.0 * math.sqrt(math.log10(max(n, 10)) / n)
    m_hat = max_lag - Kn
    for m in range(1, max_lag - Kn):
        if np.all(np.abs(rho[m + 1: m + 1 + Kn]) < c):
            m_hat = m
            break

    M = 2 * m_hat
    ks = np.arange(-M, M + 1)

    def lam_flat(t):
        a = abs(t)
        if a <= 0.5:
            return 1.0
        if a <= 1.0:
            return 2.0 * (1.0 - a)
        return 0.0

    rho_sym = np.array([rho[abs(int(k))] for k in ks])
    lam = np.array([lam_flat(k / max(M, 1)) for k in ks])
    G = float(np.sum(lam * np.abs(ks) * rho_sym))
    D_sb = 2.0 * (float(np.sum(lam * rho_sym))) ** 2
    if D_sb < 1e-18:
        b_opt = 1
    else:
        b_opt = int(round((2.0 * G * G / D_sb) ** (1.0 / 3.0) * (n ** (1.0 / 3.0))))
        b_opt = max(1, min(b_opt, n // 4))
    return b_opt, dict(var0=var0, Kn=Kn, m_hat=int(m_hat), G=G, D_sb=D_sb, n=n)


def romano_wolf_stepdown(diffs, block, B, seed, alpha=0.05):
    """
    Romano & Wolf (2005) stepwise multiple-testing control of FWER at level
    `alpha`. One-sided test H0_k: E[diffs[k]] <= 0. Studentised statistic
    t_k = sqrt(n) * mean / std. Bootstrap max-t distribution constructed by
    circular block bootstrap on the CENTRED differentials (least-favourable
    null). Returns a DataFrame sorted by descending observed t.
    """
    names = list(diffs.keys())
    K = len(names)
    if K == 0:
        return pd.DataFrame(columns=["variant", "t_obs", "rw_pvalue",
                                     "rejected_at_5pct"])
    n = len(next(iter(diffs.values())))
    rng = np.random.default_rng(int(seed) + 2)

    t_obs = np.empty(K)
    for k, nm in enumerate(names):
        x = diffs[nm]
        sd = x.std(ddof=1)
        t_obs[k] = (math.sqrt(n) * x.mean() / sd) if sd > 1e-18 else 0.0

    stacked = np.column_stack([diffs[nm] - diffs[nm].mean() for nm in names])
    boot_t = np.empty((B, K))
    for b in tqdm(range(B), desc="Romano-Wolf", leave=False, unit="draw"):
        idx = _circular_block_indices(n, block, rng)
        bs = stacked[idx, :]
        mu_b = bs.mean(axis=0)
        sd_b = bs.std(axis=0, ddof=1)
        sd_b = np.where(sd_b < 1e-18, np.inf, sd_b)
        boot_t[b, :] = math.sqrt(n) * mu_b / sd_b

    order = np.argsort(-t_obs)
    sorted_names = [names[i] for i in order]
    sorted_tobs  = t_obs[order]
    rw_p = np.ones(K)
    active = list(range(K))
    while active:
        max_t = boot_t[:, [order[i] for i in active]].max(axis=1)
        lead = active[0]
        p_lead = float(np.mean(max_t >= sorted_tobs[lead]))
        rw_p[lead] = p_lead
        if p_lead < alpha:
            active.pop(0)
        else:
            for i in active:
                rw_p[i] = max(rw_p[i], p_lead)
            break

    return pd.DataFrame({
        "variant":          sorted_names,
        "t_obs":            [float(sorted_tobs[i]) for i in range(K)],
        "rw_pvalue":        [float(rw_p[i]) for i in range(K)],
        "rejected_at_5pct": [bool(rw_p[i] < alpha) for i in range(K)],
    })


def hansen_spa_test(diffs, block, B, seed):
    """
    Hansen (2005) Superior Predictive Ability test. H0: max_k E[diffs[k]] <= 0.
    Reports SPA_l (lower), SPA_c (consistent, recommended), SPA_u (upper)
    p-values. Uses circular block bootstrap on centred differentials.
    """
    names = list(diffs.keys())
    K = len(names)
    if K == 0:
        return dict(T_spa=0.0, p_spa_c=np.nan, p_spa_l=np.nan,
                    p_spa_u=np.nan, K=0, n=0, block=int(block), B=int(B),
                    best_variant=None, best_t=np.nan)
    n = len(next(iter(diffs.values())))

    mu = np.array([diffs[nm].mean() for nm in names])
    sd = np.array([diffs[nm].std(ddof=1) for nm in names])
    sd_safe = np.where(sd < 1e-18, np.inf, sd)
    t_obs = math.sqrt(n) * mu / sd_safe
    T_spa = max(0.0, float(np.max(t_obs)))

    log_log_n = math.log(max(math.log(max(n, 3)), 1.0001))
    thresh = -math.sqrt(2.0 * log_log_n) * sd / math.sqrt(max(n, 1))

    # recentrings for the three Hansen variants
    mu_c = np.where(mu > thresh, 0.0, mu)  # consistent
    mu_l = mu.copy()                       # lower (no correction)
    mu_u = np.maximum(mu, 0.0)             # upper (set negatives to 0)

    def _boot_max(mu_shift, offset):
        rng = np.random.default_rng(int(seed) + int(offset))
        centred = np.column_stack([diffs[nm] - mu[k] + mu_shift[k]
                                   for k, nm in enumerate(names)])
        out = np.empty(B)
        for b in tqdm(range(B), desc="Hansen SPA", leave=False, unit="draw"):
            idx = _circular_block_indices(n, block, rng)
            bs = centred[idx, :]
            mb = bs.mean(axis=0)
            sb = bs.std(axis=0, ddof=1)
            sb = np.where(sb < 1e-18, np.inf, sb)
            out[b] = max(0.0, float(np.max(math.sqrt(n) * mb / sb)))
        return out

    boot_c = _boot_max(mu_c, 3)
    boot_l = _boot_max(mu_l, 4)
    boot_u = _boot_max(mu_u, 5)

    return dict(
        T_spa        = T_spa,
        p_spa_c      = float(np.mean(boot_c >= T_spa)),
        p_spa_l      = float(np.mean(boot_l >= T_spa)),
        p_spa_u      = float(np.mean(boot_u >= T_spa)),
        K            = K,
        n            = int(n),
        block        = int(block),
        B            = int(B),
        best_variant = names[int(np.argmax(t_obs))],
        best_t       = float(np.max(t_obs)),
    )


def mdd_block_bootstrap_ci(daily_ret, block, B, seed, alpha=0.05):
    """Block-bootstrap 95% CI on maximum drawdown from a daily-return array."""
    r = np.asarray(daily_ret, dtype=float)
    n = len(r)
    if n < 100:
        return dict(mdd_point=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    block=int(block), B=int(B))
    rng = np.random.default_rng(int(seed) + 6)
    vals = np.empty(B)
    for b in tqdm(range(B), desc="MDD Boot", leave=False, unit="draw"):
        idx = _circular_block_indices(n, block, rng)
        nav = np.cumprod(1.0 + r[idx])
        peak = np.maximum.accumulate(nav)
        dd = 1.0 - nav / peak
        vals[b] = float(np.max(dd))
    nav0 = np.cumprod(1.0 + r)
    peak0 = np.maximum.accumulate(nav0)
    mdd0 = float(np.max(1.0 - nav0 / peak0))
    return dict(
        mdd_point = mdd0,
        ci_lo     = float(np.quantile(vals, alpha / 2)),
        ci_hi     = float(np.quantile(vals, 1 - alpha / 2)),
        mdd_boot_mean = float(np.mean(vals)),
        block     = int(block),
        B         = int(B),
    )


# ---------------------------------------------------------------------------
# Section 6c -- Thesis-grade statistical battery primitives
#
# References:
#   * Jarque & Bera (1980) "Efficient tests for normality..." Econ. Letters
#   * Ljung & Box (1978) Biometrika
#   * Engle (1982) Econometrica (ARCH-LM)
#   * Lo (2002) "The Statistics of Sharpe Ratios" FAJ
#   * Mertens (2002) "Comments on variance of the IID estimator..."; Opdyke
#     (2007) "Comparing Sharpe Ratios: so where are the p-values?" J. Asset Mgmt
#   * Bailey & Lopez de Prado (2012) "The Sharpe Ratio Efficient Frontier"
#     J. Risk  (Minimum Track Record Length)
#   * Harvey & Liu (2015) "Backtesting" J. Portfolio Mgmt  (Haircut Sharpe)
#   * Bailey, Borwein, Lopez de Prado & Zhu (2016) "The Probability of Backtest
#     Overfitting" J. Computational Finance  (PBO via CSCV)
#   * Hansen, Lunde & Nason (2011) "The Model Confidence Set" Econometrica
#   * Henriksson & Merton (1981) "On Market Timing..." J. Business
#   * Treynor & Mazuy (1966) "Can mutual funds outguess the market?" HBR
#   * Fung & Hsieh (2001, 2004) "The Risk in Hedge Fund Strategies" RFS/FAJ
#   * Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum" JFE
# ---------------------------------------------------------------------------

def jarque_bera_test(x):
    """Jarque-Bera test for normality. Returns (JB, p, skew, kurt_excess)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return dict(jb=np.nan, p=np.nan, skew=np.nan, kurt_excess=np.nan, n=n)
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd <= 0:
        return dict(jb=np.nan, p=np.nan, skew=0.0, kurt_excess=0.0, n=n)
    z = (x - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt_ex = float(np.mean(z ** 4) - 3.0)
    jb = n / 6.0 * (skew ** 2 + 0.25 * kurt_ex ** 2)
    p = float(1.0 - sp_stats.chi2.cdf(jb, df=2))
    return dict(jb=float(jb), p=p, skew=skew, kurt_excess=kurt_ex, n=int(n))


def ljung_box_test(x, lags=20):
    """Ljung-Box Q-test for residual autocorrelation up to `lags`."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < lags + 10:
        return dict(q=np.nan, p=np.nan, lags=int(lags), n=n)
    denom = float(np.sum(x * x))
    if denom <= 0:
        return dict(q=np.nan, p=np.nan, lags=int(lags), n=n)
    q = 0.0
    for k in range(1, lags + 1):
        num = float(np.sum(x[k:] * x[:-k]))
        rho_k = num / denom
        q += rho_k * rho_k / (n - k)
    q *= n * (n + 2)
    p = float(1.0 - sp_stats.chi2.cdf(q, df=lags))
    return dict(q=float(q), p=p, lags=int(lags), n=int(n))


def arch_lm_test(x, lags=5):
    """Engle (1982) ARCH-LM test for conditional heteroskedasticity."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < lags + 30:
        return dict(lm=np.nan, p=np.nan, lags=int(lags), n=n)
    e2 = (x - x.mean()) ** 2
    Y = e2[lags:]
    T = len(Y)
    Xm = np.ones((T, lags + 1), dtype=float)
    for k in range(1, lags + 1):
        Xm[:, k] = e2[lags - k: n - k]
    try:
        beta, *_ = np.linalg.lstsq(Xm, Y, rcond=None)
        yhat = Xm @ beta
        ss_res = float(np.sum((Y - yhat) ** 2))
        ss_tot = float(np.sum((Y - Y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except Exception:
        return dict(lm=np.nan, p=np.nan, lags=int(lags), n=int(n))
    lm = T * r2
    p = float(1.0 - sp_stats.chi2.cdf(lm, df=lags))
    return dict(lm=float(lm), p=p, lags=int(lags), n=int(n))


def lo_sharpe_se(r, q=TRADING_DAYS):
    """
    Lo (2002) autocorrelation-adjusted Sharpe ratio and SE for aggregation
    from daily to q-period returns.

      eta(q) = q / sqrt(q + 2 * sum_{k=1}^{q-1} (q-k) * rho_k)

    Returns the annualised SR = eta(q) * (mu/sigma_daily) and its Lo SE.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 60:
        return dict(sr_iid=np.nan, sr_lo=np.nan, se_lo=np.nan, eta=np.nan, n=n)
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return dict(sr_iid=0.0, sr_lo=0.0, se_lo=np.nan, eta=np.nan, n=n)
    sr_d = mu / sd
    # autocorrelations up to q-1 (capped by n/4 to avoid ghost cycles)
    max_lag = int(min(q - 1, n // 4))
    x = r - mu
    rho = []
    den = float(np.sum(x * x))
    for k in range(1, max_lag + 1):
        rho_k = float(np.sum(x[k:] * x[:-k])) / den if den > 0 else 0.0
        rho.append(rho_k)
    inner = float(q + 2.0 * sum((q - (k + 1)) * rho[k] for k in range(len(rho))))
    if inner <= 0:
        return dict(sr_iid=sr_d * np.sqrt(q), sr_lo=np.nan, se_lo=np.nan,
                    eta=np.nan, n=int(n))
    eta = q / np.sqrt(inner)
    sr_iid  = sr_d * np.sqrt(q)
    sr_lo   = sr_d * eta
    # Lo (2002) eqn(22): var(SR_lo) ~ (1 + SR_lo^2/2) / n (iid bound;
    # Lo shows this is still valid under GMM for the annualised ratio).
    se_lo = float(np.sqrt((1.0 + 0.5 * sr_lo ** 2) / n))
    return dict(sr_iid=float(sr_iid), sr_lo=float(sr_lo), se_lo=se_lo,
                eta=float(eta), n=int(n))


def mertens_sharpe_se(r):
    """
    Mertens (2002) / Opdyke (2007) fat-tail Sharpe SE.

      var(SR) = (1 + SR^2/2 - g3*SR + (g4-3)/4 * SR^2) / n

    where g3, g4 are sample skewness and kurtosis (not excess). SR here is
    the daily Sharpe; annualised SR = SR_daily * sqrt(TRADING_DAYS) and the
    annualised SE = se_d * sqrt(TRADING_DAYS).
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 60:
        return dict(sr_ann=np.nan, se_ann=np.nan, skew=np.nan, kurt=np.nan, n=n)
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return dict(sr_ann=0.0, se_ann=np.nan, skew=0.0, kurt=3.0, n=n)
    sr_d = mu / sd
    z = (r - mu) / sd
    g3 = float(np.mean(z ** 3))
    g4 = float(np.mean(z ** 4))  # raw kurtosis, not excess
    var_d = (1.0 + 0.5 * sr_d ** 2 - g3 * sr_d + 0.25 * (g4 - 3.0) * sr_d ** 2) / n
    var_d = max(var_d, 0.0)
    se_d = float(np.sqrt(var_d))
    sr_a = sr_d * np.sqrt(TRADING_DAYS)
    se_a = se_d * np.sqrt(TRADING_DAYS)
    return dict(sr_ann=float(sr_a), se_ann=se_a,
                t_stat=float(sr_a / se_a) if se_a > 0 else np.nan,
                skew=g3, kurt=g4, n=int(n))


def min_track_record_length(sr_ann, sr_bench_ann, n, skew=0.0, kurt=3.0,
                             alpha=0.05):
    """
    Bailey & Lopez de Prado (2012) Minimum Track Record Length (MinTRL):
    the number of observations needed to reject H_0: SR <= SR* at level alpha.

      MinTRL = 1 + (1 - g3*SR_d + (g4-1)/4 * SR_d^2) * (z / (SR_d - SR*_d))^2
    """
    if sr_ann <= sr_bench_ann:
        return dict(mintrl=np.inf, current_n=int(n), feasible=False,
                    sr_ann=float(sr_ann), sr_bench_ann=float(sr_bench_ann))
    sr_d   = sr_ann / np.sqrt(TRADING_DAYS)
    sr_b_d = sr_bench_ann / np.sqrt(TRADING_DAYS)
    z = sp_stats.norm.ppf(1.0 - alpha)
    num = 1.0 - skew * sr_d + 0.25 * (kurt - 1.0) * sr_d ** 2
    mintrl = 1.0 + max(num, 0.0) * (z / (sr_d - sr_b_d)) ** 2
    return dict(mintrl=float(mintrl), mintrl_years=float(mintrl / TRADING_DAYS),
                current_n=int(n), feasible=bool(n >= mintrl),
                sr_ann=float(sr_ann), sr_bench_ann=float(sr_bench_ann),
                alpha=float(alpha))


def harvey_liu_haircut(sr_ann, t_stat, N_trials):
    """
    Harvey & Liu (2015) multiple-testing haircut Sharpe ratio.

    Applies Bonferroni, Holm, and Benjamini-Yekutieli adjustments to the
    observed p-value, then back-solves the haircut SR that matches the
    adjusted p-value. Returns original and three haircut SRs.
    """
    if t_stat <= 0 or N_trials < 1:
        return dict(sr_orig=float(sr_ann), sr_bonf=0.0, sr_holm=0.0,
                    sr_bhy=0.0, p_orig=1.0, p_bonf=1.0, p_holm=1.0, p_bhy=1.0,
                    haircut_bonf=1.0, haircut_holm=1.0, haircut_bhy=1.0)
    p_obs = float(1.0 - sp_stats.norm.cdf(t_stat))
    p_obs = max(p_obs, 1e-300)
    # Bonferroni
    p_bonf = min(1.0, p_obs * N_trials)
    # Holm (best case for the single best trial is Holm step 1 = Bonferroni)
    p_holm = p_bonf
    # Benjamini-Yekutieli (BHY) under arbitrary dependence:
    c_M = float(sum(1.0 / k for k in range(1, int(N_trials) + 1)))
    p_bhy = min(1.0, p_obs * N_trials * c_M)
    def _back_solve(p_adj):
        z_adj = sp_stats.norm.ppf(max(1.0 - p_adj, 1e-12))
        return float(sr_ann * z_adj / t_stat)
    sr_bonf = _back_solve(p_bonf)
    sr_holm = _back_solve(p_holm)
    sr_bhy  = _back_solve(p_bhy)
    return dict(
        sr_orig = float(sr_ann),
        t_orig  = float(t_stat),
        N       = int(N_trials),
        p_orig  = p_obs,
        p_bonf  = float(p_bonf),
        p_holm  = float(p_holm),
        p_bhy   = float(p_bhy),
        sr_bonf = sr_bonf,
        sr_holm = sr_holm,
        sr_bhy  = sr_bhy,
        haircut_bonf = float(1.0 - sr_bonf / sr_ann) if sr_ann > 0 else 1.0,
        haircut_holm = float(1.0 - sr_holm / sr_ann) if sr_ann > 0 else 1.0,
        haircut_bhy  = float(1.0 - sr_bhy  / sr_ann) if sr_ann > 0 else 1.0,
    )


def _combinations_half(S):
    """Generate all C(S, S/2) subsets of range(S) as tuples."""
    from itertools import combinations
    return list(combinations(range(S), S // 2))


def pbo_cscv(returns_matrix, S=16, metric="sharpe"):
    """
    Bailey, Borwein, Lopez de Prado & Zhu (2016) Probability of Backtest
    Overfitting via Combinatorially-Symmetric Cross-Validation (CSCV).

    Parameters
    ----------
    returns_matrix : np.ndarray, shape (T, N)
        Daily returns for N candidate strategies over T common dates.
    S : int, even
        Number of sub-periods. Default 16 -> C(16, 8) = 12,870 splits.

    Returns
    -------
    dict with pbo, median_logit, distribution, and diagnostics.
    """
    R = np.asarray(returns_matrix, dtype=float)
    T, N = R.shape
    if T < S * 10 or N < 4 or S % 2 != 0:
        return dict(pbo=np.nan, median_logit=np.nan, n_splits=0,
                    T=int(T), N=int(N), S=int(S),
                    note="insufficient data or bad S")
    # Trim T to a multiple of S, partition contiguously.
    T_eff = (T // S) * S
    R = R[:T_eff, :]
    blk = T_eff // S
    blocks = [R[s * blk:(s + 1) * blk, :] for s in range(S)]

    def _score(mat):
        mu = mat.mean(axis=0)
        sd = mat.std(axis=0, ddof=1)
        sd = np.where(sd > 1e-12, sd, np.nan)
        return mu / sd

    logits = []
    ranks_in_oos = []
    splits = _combinations_half(S)
    for J in splits:
        J_set = set(J)
        is_idx  = [i for i in range(S) if i in J_set]
        oos_idx = [i for i in range(S) if i not in J_set]
        is_mat  = np.vstack([blocks[i] for i in is_idx])
        oos_mat = np.vstack([blocks[i] for i in oos_idx])
        is_score  = _score(is_mat)
        oos_score = _score(oos_mat)
        if not np.any(np.isfinite(is_score)):
            continue
        n_star = int(np.nanargmax(is_score))
        # OOS rank of the IS winner (1 = worst, N = best); normalise to (0,1)
        valid = np.isfinite(oos_score)
        if not valid.any():
            continue
        oos_ranks = sp_stats.rankdata(np.where(valid, oos_score, -np.inf),
                                        method="average")
        w_bar = oos_ranks[n_star] / (N + 1.0)
        if w_bar <= 0 or w_bar >= 1:
            continue
        logit = float(np.log(w_bar / (1.0 - w_bar)))
        logits.append(logit)
        ranks_in_oos.append(float(w_bar))
    if not logits:
        return dict(pbo=np.nan, median_logit=np.nan, n_splits=0,
                    T=int(T), N=int(N), S=int(S))
    logits = np.asarray(logits)
    pbo = float(np.mean(logits < 0.0))
    return dict(
        pbo            = pbo,
        median_logit   = float(np.median(logits)),
        mean_logit     = float(np.mean(logits)),
        n_splits       = int(len(logits)),
        T              = int(T_eff),
        N              = int(N),
        S              = int(S),
        median_oos_rank= float(np.median(ranks_in_oos)),
    )


def model_confidence_set(loss_matrix, block=21, B=5000, alpha=0.10, seed=0):
    """
    Hansen, Lunde & Nason (2011) Model Confidence Set (MCS) via the T_max
    statistic with a stationary block bootstrap, stepwise elimination.

    Parameters
    ----------
    loss_matrix : np.ndarray, shape (T, N)
        Per-period loss for each model. Higher loss = worse model.
        For a Sharpe-style comparison, pass L = -daily_return.
    block : int
        Circular block bootstrap block length.
    B : int
        Bootstrap replications.
    alpha : float
        Significance level (MCS confidence = 1 - alpha).

    Returns
    -------
    dict with kept_set (indices), eliminated sequence, p-values.
    """
    L = np.asarray(loss_matrix, dtype=float)
    T, N = L.shape
    if T < 50 or N < 2:
        return dict(kept=list(range(N)), eliminated=[], p_values=[],
                    T=int(T), N=int(N), note="too small")
    rng = np.random.default_rng(int(seed))
    # Precompute bootstrap index sets once (shared across elimination rounds)
    boot_idx = np.stack([_circular_block_indices(T, int(block), rng)
                          for _ in range(int(B))], axis=0)  # (B, T)

    alive = list(range(N))
    eliminated = []
    p_vals = []
    while len(alive) > 1:
        sub = L[:, alive]                 # (T, k)
        k = sub.shape[1]
        mean_l = sub.mean(axis=0)          # (k,)
        # d_{ij} = L_i - L_mean  (relative to current set mean)
        d_rel = sub - sub.mean(axis=1, keepdims=True)    # (T, k)
        d_mean = d_rel.mean(axis=0)  # (k,)  observed
        # Bootstrap variance of d_mean via resampling the T rows with blocks
        boot_means = np.empty((B, k), dtype=float)
        for b in tqdm(range(B), desc="MCS Boot", leave=False, unit="draw"):
            boot_means[b, :] = d_rel[boot_idx[b], :].mean(axis=0)
        var_d = boot_means.var(axis=0, ddof=1)
        sd_d  = np.sqrt(np.maximum(var_d, 1e-24))
        t_obs = d_mean / sd_d
        t_max_obs = float(np.max(t_obs))
        # Bootstrap distribution of t_max under H_0 (recentre)
        centred = boot_means - d_mean[None, :]
        t_boot = centred / sd_d[None, :]
        t_max_boot = t_boot.max(axis=1)
        p = float(np.mean(t_max_boot >= t_max_obs))
        p_vals.append(p)
        if p >= alpha:
            break
        # Eliminate the worst (largest t_obs = highest relative loss)
        drop_local = int(np.argmax(t_obs))
        drop_global = alive[drop_local]
        eliminated.append(drop_global)
        alive.pop(drop_local)
    return dict(
        kept_idx    = alive,
        eliminated  = eliminated,
        p_values    = p_vals,
        alpha       = float(alpha),
        block       = int(block),
        B           = int(B),
        T           = int(T),
        N           = int(N),
    )


def _ols_hac(y, X, lags=None):
    """OLS with Newey-West HAC standard errors. Returns beta, se, t, p, r2."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    T, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    if lags is None:
        lags = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    lags = max(lags, 0)
    # HAC (Newey-West) meat
    S0 = (X * resid[:, None]).T @ (X * resid[:, None])
    S = S0.copy()
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        u = X[l:, :] * resid[l:, None]
        v = X[:-l, :] * resid[:-l, None]
        G = u.T @ v
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    # Finite-sample DoF correction (HC1-style, matches statsmodels default)
    if T > k:
        cov = cov * (T / (T - k))
    se  = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t   = np.where(se > 0, beta / se, np.nan)
    p   = 2.0 * (1.0 - sp_stats.norm.cdf(np.abs(t)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return dict(beta=beta, se=se, t=t, p=p, r2=float(r2), n=int(T))


def henriksson_merton(r_strategy, r_market, rf=None, lags=None):
    """
    Henriksson-Merton (1981) market timing regression:
      r_s - rf = alpha + beta*(r_m - rf) + gamma*max(0, r_m - rf) + eps
    """
    r_s = np.asarray(r_strategy, dtype=float)
    r_m = np.asarray(r_market, dtype=float)
    if rf is None:
        rf = np.zeros_like(r_s)
    rf = np.asarray(rf, dtype=float)
    n = min(len(r_s), len(r_m), len(rf))
    r_s, r_m, rf = r_s[:n], r_m[:n], rf[:n]
    if n < 60:
        return dict(n=n, note="too few obs")
    y = r_s - rf
    xm = r_m - rf
    X = np.column_stack([np.ones(n), xm, np.maximum(0.0, xm)])
    res = _ols_hac(y, X, lags=lags)
    alpha_ann = float(res["beta"][0] * TRADING_DAYS)
    return dict(
        n         = int(n),
        alpha_daily = float(res["beta"][0]),
        alpha_ann = alpha_ann,
        alpha_t   = float(res["t"][0]),
        alpha_p   = float(res["p"][0]),
        beta      = float(res["beta"][1]),
        beta_t    = float(res["t"][1]),
        gamma     = float(res["beta"][2]),
        gamma_t   = float(res["t"][2]),
        gamma_p   = float(res["p"][2]),
        r2        = float(res["r2"]),
    )


def treynor_mazuy(r_strategy, r_market, rf=None, lags=None):
    """
    Treynor-Mazuy (1966) quadratic market timing:
      r_s - rf = alpha + beta*(r_m - rf) + gamma*(r_m - rf)^2 + eps
    """
    r_s = np.asarray(r_strategy, dtype=float)
    r_m = np.asarray(r_market, dtype=float)
    if rf is None:
        rf = np.zeros_like(r_s)
    rf = np.asarray(rf, dtype=float)
    n = min(len(r_s), len(r_m), len(rf))
    r_s, r_m, rf = r_s[:n], r_m[:n], rf[:n]
    if n < 60:
        return dict(n=n, note="too few obs")
    y = r_s - rf
    xm = r_m - rf
    X = np.column_stack([np.ones(n), xm, xm ** 2])
    res = _ols_hac(y, X, lags=lags)
    return dict(
        n         = int(n),
        alpha_ann = float(res["beta"][0] * TRADING_DAYS),
        alpha_t   = float(res["t"][0]),
        alpha_p   = float(res["p"][0]),
        beta      = float(res["beta"][1]),
        beta_t    = float(res["t"][1]),
        gamma     = float(res["beta"][2]),
        gamma_t   = float(res["t"][2]),
        gamma_p   = float(res["p"][2]),
        r2        = float(res["r2"]),
    )


def factor_regression_hac(r_strategy, factors_df, rf=None, lags=None,
                           periods_per_year=None):
    """
    Generic multi-factor regression with Newey-West HAC SE.
      r_s - rf = alpha + sum_k beta_k * F_k + eps

    factors_df : pd.DataFrame, aligned index to r_strategy (after outer align).
    Returns alpha (annualised), each factor loading with t-stat, R^2.
    """
    if not isinstance(factors_df, pd.DataFrame):
        return dict(note="factors_df must be DataFrame")
    # Coerce r_strategy to a Series. If it's already a Series, keep its index;
    # if it's a raw ndarray/list, assume it is pre-aligned to factors_df.index.
    if isinstance(r_strategy, pd.Series):
        y_ser = r_strategy
    else:
        y_arr = np.asarray(r_strategy, dtype=float).ravel()
        if len(y_arr) == len(factors_df):
            y_ser = pd.Series(y_arr, index=factors_df.index)
        else:
            y_ser = pd.Series(y_arr)  # will yield empty join
    aligned = pd.DataFrame({"_y": y_ser}).join(factors_df, how="inner")
    if len(aligned) == 0:
        return dict(n=0, alpha_ann=float("nan"), alpha_t=float("nan"),
                    alpha_p=float("nan"), r2=float("nan"),
                    factors=OrderedDict(),
                    note="empty alignment between r_strategy and factors")
    if rf is not None:
        rf_s = pd.Series(rf).reindex(aligned.index).fillna(0.0)
        y = (aligned["_y"] - rf_s).values.astype(float)
    else:
        y = aligned["_y"].values.astype(float)
    cols = [c for c in aligned.columns if c != "_y"]
    X = np.column_stack([np.ones(len(y))] + [aligned[c].values.astype(float)
                                                for c in cols])
    res = _ols_hac(y, X, lags=lags)
    ppy = float(periods_per_year) if periods_per_year is not None else float(TRADING_DAYS)
    out = dict(
        n         = int(len(y)),
        alpha_ann = float(res["beta"][0] * ppy),
        alpha_t   = float(res["t"][0]),
        alpha_p   = float(res["p"][0]),
        r2        = float(res["r2"]),
        factors   = OrderedDict(),
    )
    for i, c in enumerate(cols, start=1):
        out["factors"][c] = dict(
            beta = float(res["beta"][i]),
            se   = float(res["se"][i]),
            t    = float(res["t"][i]),
            p    = float(res["p"][i]),
        )
    return out


# ===========================================================================
# SECTION 6b -- Small numeric utilities shared by DM + Ablation builders
# ===========================================================================

def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _shifted_sigmoid(x):
    """Zero-drag sigmoid: 0 at trigger, approaches 1 for large x."""
    return np.clip(2.0 * _sigmoid(x) - 1.0, 0.0, 1.0)


def _s183cfg_to_canonical(cfg):
    """Translate an S183_CFG-shaped dict into the canonical cfg schema.

    S183_CFG was authored before the 15-Apr refactor introduced
    `ig_strategy_183.S183_CANONICAL_CFG` as the single source of truth.
    The two dicts share most keys verbatim (w_ts, w_xs, w_trend, ...,
    vov_window, vov_use_direction, fdm_corr_span, ...) but differ in
    smoothing and sigmoid encoding:

      S183_CFG         canonical
      --------         ---------
      smooth_span      smooth_mode='span', smooth_span=N
      (no halflife)    smooth_mode='halflife', smooth_halflife=N
      use_shifted_sig  use_shifted_sigmoid   (same meaning)
      fdm_mode='pooled'/'per_inst'/...       (same)

    To minimise drift between the suite's ablation MASTER and the canonical
    master, this translator rewrites smooth_span=3 into smooth_mode='halflife',
    smooth_halflife=1 (both give alpha=0.5 but pandas computes them with
    slightly different numerics).  Any variant that explicitly sets
    smooth_span != 3 preserves span-mode semantics.

    Also surfaces the VoV direction lookback, which in S183_CFG is implicitly
    tied to vov_window (the pre-bake always uses `close.pct_change(vov_window)`
    via _build_vov_raw).  When use_direction=True, vov_direction_lb follows
    vov_window.
    """
    params = dict(cfg)

    out = dict(S183_CANONICAL_CFG)   # start from canonical defaults
    # Direct key mappings (identical semantics)
    direct = [
        "w_ts", "w_xs", "w_trend", "w_carry", "w_skew", "w_vov",
        "use_conviction", "conviction_window", "trend_fdm",
        "xs_lookback", "skew_window", "vov_window", "vov_use_direction",
        "fdm_mode", "fdm_corr_span", "fdm_min_periods", "fdm_floor", "fdm_cap",
        "sigmoid_steepness", "use_shifted_sigmoid",
        "vol_trigger", "vol_dampen", "dd_threshold", "dd_scale",
        "speed_pairs", "speed_weights",
    ]
    for k in direct:
        if k in params:
            out[k] = params[k]

    # VoV direction lookback: in the suite's _build_vov_raw it's hardcoded
    # equal to `window` (the vov_window).  Mirror that here so canonical's
    # parameterised direction_lb tracks the suite's behaviour.
    out["vov_direction_lb"] = out["vov_window"]

    # Smoothing: S183_CFG uses span; canonical prefers halflife for the
    # canonical master (halflife=1 == span=3 at alpha=0.5).  Rewrite the
    # span=3 default case to halflife=1 so the suite MASTER matches the
    # canonical master bit-for-bit.
    if "smooth_span" in params:
        span = params["smooth_span"]
        if span == 3:
            out["smooth_mode"]     = "halflife"
            out["smooth_halflife"] = 1
            out["smooth_span"]     = None
        elif span == 0:
            out["smooth_mode"]     = "none"
            out["smooth_halflife"] = None
            out["smooth_span"]     = None
        else:
            out["smooth_mode"]     = "span"
            out["smooth_halflife"] = None
            out["smooth_span"]     = int(span)

    # xs_min_insts is hardcoded in the suite (XS_MIN_INSTS=3) -- use canonical default.
    return out


def _transform_corr(ct, mode):
    """Pooling-mode transform on an nxn correlation matrix."""
    ct = np.clip(ct, -1.0, 1.0)
    if mode == FDM_MODE_SHRINK:
        I = np.eye(ct.shape[0])
        ct = 0.5 * ct + 0.5 * I
    elif mode == FDM_MODE_CLIP:
        n = ct.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off = ct[mask]
        m_off = float(off.mean()) if off.size else 0.0
        new_ct = np.eye(n)
        new_ct[mask] = m_off
        ct = new_ct
    return ct


def _pooled_fdm_from_panels(panel_dict, w_dict, corr_span, min_per,
                             floor, cap, mode):
    """Universe-pooled N-way FDM from dict of DataFrames."""
    names = list(panel_dict.keys())
    n = len(names)
    if n == 0:
        raise ValueError("No active alphas")
    means = {nm: panel_dict[nm].mean(axis=1) for nm in names}
    fc_df = pd.DataFrame(means)
    if n == 1:
        return pd.Series(1.0, index=fc_df.index)
    w_vec = np.array([w_dict[nm] for nm in names], dtype=float)
    if w_vec.sum() > 0:
        w_vec = w_vec / w_vec.sum()
    T = len(fc_df)
    fdm = np.ones(T)
    ewm = fc_df.ewm(span=corr_span, min_periods=min_per).corr()
    for t in range(min_per, T):
        try:
            ct = ewm.loc[fc_df.index[t]].values
            if ct.shape != (n, n) or np.any(np.isnan(ct)):
                continue
            ct = _transform_corr(ct, mode)
            var = w_vec @ ct @ w_vec
            if var > 0.01:
                fdm[t] = np.clip(1.0 / np.sqrt(var), floor, cap)
        except (KeyError, ValueError):
            continue
    return pd.Series(fdm, index=fc_df.index)


def _perinst_fdm(fc_dict, w_dict, corr_span, min_per, floor, cap):
    """Per-instrument 4-way FDM (S161 legacy path)."""
    names = list(fc_dict.keys())
    n = len(names)
    if n == 0:
        raise ValueError("No active alphas")
    if n == 1:
        return pd.Series(1.0, index=list(fc_dict.values())[0].index)
    w_vec = np.array([w_dict[nm] for nm in names], dtype=float)
    if w_vec.sum() > 0:
        w_vec = w_vec / w_vec.sum()
    fc_df = pd.DataFrame({nm: fc_dict[nm] for nm in names})
    T = len(fc_df)
    fdm = np.ones(T)
    ewm = fc_df.ewm(span=corr_span, min_periods=min_per).corr()
    for t in range(min_per, T):
        try:
            ct = ewm.loc[fc_df.index[t]].values
            if ct.shape != (n, n) or np.any(np.isnan(ct)):
                continue
            ct = _transform_corr(ct, FDM_MODE_PERINST)
            var = w_vec @ ct @ w_vec
            if var > 0.01:
                fdm[t] = np.clip(1.0 / np.sqrt(var), floor, cap)
        except (KeyError, ValueError):
            continue
    return pd.Series(fdm, index=fc_df.index)


def _apply_conviction(scaled, window):
    rstd = scaled.rolling(window, min_periods=min(window // 2, 128)).std()
    conv = (scaled.abs() / rstd.fillna(1.0)).clip(0.0, 1.0)
    return scaled * conv


def _vol_gate(vol_ratio, trigger, dampen, steepness):
    return 1.0 - (1.0 - dampen) * _shifted_sigmoid(steepness * (vol_ratio - trigger))


def _dd_gate(dd_proxy, threshold, dd_scale, steepness):
    return 1.0 - (1.0 - dd_scale) * _shifted_sigmoid(steepness * (threshold - dd_proxy))


def _build_vov_raw(price_changes, close, window, use_direction):
    daily_vol = price_changes.rolling(21, min_periods=10).std()
    # min_periods is capped at `window` so DM_ALL_VOV_WINDOWS values < 21
    # (used by run_extended_robustness T29-T34) do not violate pandas'
    # constraint that min_periods <= window.
    vov = daily_vol.rolling(window, min_periods=min(window, max(21, window // 3))).std()
    # S183 v3: base-2 canonical lookbacks 256 / 64 (not 252 / 63).
    # Matches ig_strategy_183.py top-level pipeline exactly so the
    # ablation-engine MASTER checkpoint reproduces the strategy checkpoint.
    vov_avg = vov.rolling(256, min_periods=64).mean()
    vov_norm = -(vov / vov_avg - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if use_direction:
        direction = np.where(close.pct_change(64).fillna(0.0) >= 0, 1.0, -1.0)
        return (pd.Series(direction, index=close.index) * vov_norm).fillna(0.0)
    return vov_norm.fillna(0.0)


# ===========================================================================
# SECTION 3 -- DATAMINING module
# ===========================================================================

# All windows / lookbacks any downstream consumer might request must
# appear in these lists, because dm_precompute_library pre-bakes each
# signal at the listed values and dm_build_variant_signals silently
# falls back to pd.Series(0.0, ...) when a requested key is missing.
#
# Audited consumers: S183 MASTER, every DM sweep variant, every ABL
# variant, T24 random-parameter MC discrete sampling grid, T28
# parameter-scramble adversarial values, and the T26/T27 hard-coded
# values. The union of required values is:
#
#   VOV_WINDOWS  : {5, 16, 21, 42, 63, 64, 90, 126, 2048}
#   XS_LOOKBACKS : {16, 126, 192, 252, 256, 378, 512, 2048}
#   SKEW_WINDOWS : {16, 64, 128, 192, 256, 384, 512, 2048}
#   CONV_WINDOWS : {16, 128, 192, 256, 384, 512, 2048}
#
# The S180 suite pre-baked at {21, 42, 63, 90, 126} and {126, 252, 378}
# which silently zeroed the VoV and XS contributions for every DM and
# parsimony variant that used S183_CFG's base-2 canonical {64, 256} values.
# This list is now a superset of every value any test requests.

# All speeds referenced across any A-variant -- needs pre-baking.
DM_ALL_SPEED_PAIRS = [
    (8, 32), (16, 64), (32, 128), (64, 256), (128, 512),
]
DM_ALL_CONV_WINDOWS = [16, 128, 192, 256, 384, 512, 2048]
DM_ALL_SKEW_WINDOWS = [16, 64, 128, 192, 256, 384, 512, 2048]
DM_ALL_XS_LOOKBACKS = [16, 126, 192, 252, 256, 378, 512, 2048]
DM_ALL_VOV_WINDOWS  = [5, 16, 21, 42, 63, 64, 90, 126, 2048]


def _build_dm_variants():
    """
    Full S183_CFG datamining registry (A..R). Every variant is a one-at-a-time
    perturbation of the S183 MASTER anchor.

    Four axes are re-centred vs the S180 suite to reflect S183_CFG's actual
    baseline architecture:

      B (Conviction ramp)    : MASTER = ON (window 256). Sweep adds 'OFF'.
      E (FDM cap)            : MASTER = inf. Sweep tests imposing caps
                                 at {1.20, 1.50, 1.65, 1.80, 2.00, 2.20,
                                 2.40, 3.00}. Tests whether an actual
                                 cap would bind given the downstream
                                 renormalisation.
      I (Sigmoid steepness)  : MASTER = 1.0 (identity). Sweep covers
                                 {0.5, 1.0*, 2.0, 3.0, 5.0, 7.0, 10.0}.
                                 Notable points: 5.0 = 1/VOL_TARGET
                                 (S180 report claim); 10.0 = S179
                                 cliff-edge.
      N (VoV direction)      : MASTER = ON. Sweep adds 'OFF'.
    """
    V = OrderedDict()

    # -- A. EWMAC Speed Combinations --
    # MASTER under v3.4: 3-speed Triple {(16,64),(32,128),(64,256)} --
    # three consecutive L&P 2016 canonical speeds, all with factor-4
    # internal ratio and factor-2 between-pair spacing.
    speed_combos = [
        # Triple (MASTER) -- all three L&P canonical speeds
        ("3spd_triple",        [(16, 64), (32, 128), (64, 256)]),   # MASTER
        # 2-speed subsets of the Triple
        ("2spd_fast_mid",      [(16, 64), (32, 128)]),
        ("2spd_fast_slow",     [(16, 64), (64, 256)]),   # = v3.3 pre-Triple master
        ("2spd_mid_slow",      [(32, 128), (64, 256)]),
        # Single speeds from the Triple
        ("1spd_16_64",         [(16, 64)]),
        ("1spd_32_128",        [(32, 128)]),
        ("1spd_64_256",        [(64, 256)]),
        # Cross-direction probes (outside the Triple)
        ("3spd_incl_fastest",  [(8, 32), (16, 64), (32, 128)]),      # pull faster
        ("3spd_incl_slowest",  [(32, 128), (64, 256), (128, 512)]),  # pull slower
        ("4spd_8_to_256",      [(8, 32), (16, 64), (32, 128), (64, 256)]),  # full L&P set
    ]
    for label, pairs in speed_combos:
        w = [1.0 / len(pairs)] * len(pairs)
        V[f"A_{label}"] = dict(speed_pairs=pairs, speed_weights=w,
                                label=label, dim="A: EWMAC Speeds")

    # -- B. Conviction (S183 default = OFF) --
    # S183 removes the S179/S182 conviction ramp.  The sweep re-adds it
    # at a range of windows to confirm that none of them beats OFF.
    V["B_conv_off"] = dict(use_conviction=False,
                            label="Conviction OFF (S183*)",
                            dim="B: Conviction")
    for cw in [128, 192, 256, 384, 512]:
        V[f"B_conv_{cw}"] = dict(use_conviction=True, conviction_window=cw,
                                  label=f"Conviction ON, win={cw}",
                                  dim="B: Conviction")

    # -- C. Trend FDM -- (centered on S180 default 1.0)
    for val in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]:
        V[f"C_tfdm_{val:.2f}"] = dict(trend_fdm=val,
                                        label=f"TREND_FDM={val:.2f}",
                                        dim="C: Trend FDM")

    # -- D. Alpha Weights --
    alpha_weights = [
        (0.25, 0.25, 0.25, 0.25, "25/25/25/25"),  # MASTER
        (0.40, 0.20, 0.20, 0.20, "40/20/20/20"),
        (0.20, 0.40, 0.20, 0.20, "20/40/20/20"),
        (0.20, 0.20, 0.40, 0.20, "20/20/40/20"),
        (0.20, 0.20, 0.20, 0.40, "20/20/20/40"),
        (0.3333, 0.2222, 0.2222, 0.2223, "33/22/22/22"),
        (0.30, 0.30, 0.30, 0.10, "30/30/30/10"),
    ]
    for wt, wc, ws, wv, lbl in alpha_weights:
        key = (f"D_t{int(round(wt*100))}_c{int(round(wc*100))}"
               f"_s{int(round(ws*100))}_v{int(round(wv*100))}")
        V[key] = dict(w_trend=wt, w_carry=wc, w_skew=ws, w_vov=wv,
                      label=f"T/C/S/V={lbl}", dim="D: Alpha Weights")

    # -- E. FDM_CAP -- Cap-sensitivity axis --
    # S183 v3.3+ MASTER has FDM_CAP = sqrt(N_QUAD_ALPHAS) = 2.0, the
    # zero-correlation benchmark cap.  Empirical max pre-cap is ~3.6, so
    # the cap actively binds on a minority of days when the EWM(512)
    # correlation estimate dips into the negative tail.  This axis sweeps
    # cap values both tighter (1.20, 1.50, 1.65, 1.80) and looser (2.20,
    # 2.40, 3.00, inf) than the derived sqrt(N) cap.
    for cap in [1.20, 1.50, 1.65, 1.80, 2.00, 2.20, 2.40, 3.00]:
        is_master = abs(cap - 2.00) < 1e-9
        V[f"E_fdmcap_{cap:.2f}"] = dict(
            fdm_cap=cap,
            label=f"FDM_CAP={cap:.2f}" + (" (S183*)" if is_master else ""),
            dim="E: FDM Cap",
        )
    V["E_fdmcap_inf"] = dict(
        fdm_cap=float("inf"),
        label="FDM_CAP=inf (uncapped)",
        dim="E: FDM Cap",
    )

    # -- F. FDM_CORR_SPAN --
    for span in [256, 512, 1024]:
        V[f"F_fdmspan_{span}"] = dict(fdm_corr_span=span,
                                       label=f"FDM_SPAN={span}",
                                       dim="F: FDM Corr Span")

    # -- G. Skew Window --
    for sw in [128, 192, 256, 384, 512]:
        V[f"G_skew_{sw}"] = dict(skew_window=sw,
                                  label=f"SkewWin={sw}",
                                  dim="G: Skew Window")

    # -- H. Smooth Span -- (S183 default = 3, sharpened from S182's 5)
    for ss in [0, 2, 3, 5, 8, 10, 15, 21]:
        suffix = " (S183*)" if ss == 3 else ""
        V[f"H_smooth_{ss}"] = dict(
            smooth_span=ss,
            label=(f"Smooth={ss}{suffix}" if ss > 0 else "No Smoothing"),
            dim="H: Smooth Span",
        )

    # -- I. Sigmoid Steepness -- (centered on S183 default 10.0)
    # S183 uses the sharpened logistic at steepness 10.0 ("S179 cliff-edge").
    # Sweep covers both lower values (reverting toward S182's identity 1.0)
    # and a higher value (20.0) to probe plateau vs cliff.
    for steep in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]:
        is_master = abs(steep - 10.0) < 1e-9
        V[f"I_steep_{steep:.1f}"] = dict(
            sigmoid_steepness=float(steep),
            label=f"Steepness={steep:.1f}" + (" (S183*)" if is_master else ""),
            dim="I: Sigmoid Steepness")

    # -- J. Risk Overlays --
    V["J_tighter"] = dict(vol_trigger=1.0, vol_dampen=0.50,
                           dd_threshold=-0.10, dd_scale=0.50,
                           label="Tighter Overlays (S183*)",
                           dim="J: Risk Overlays")
    V["J_tight"]   = dict(vol_trigger=1.5, vol_dampen=0.75,
                           dd_threshold=-0.02, dd_scale=0.50,
                           label="Tight Overlays (=S168)",
                           dim="J: Risk Overlays")
    V["J_default"] = dict(vol_trigger=2.0, vol_dampen=0.85,
                           dd_threshold=-0.04, dd_scale=0.65,
                           label="Default Overlays (=S158)",
                           dim="J: Risk Overlays")
    V["J_loose"]   = dict(vol_trigger=2.5, vol_dampen=0.90,
                           dd_threshold=-0.06, dd_scale=0.75,
                           label="Loose Overlays",
                           dim="J: Risk Overlays")
    V["J_off"]     = dict(vol_trigger=999, vol_dampen=1.0,
                           dd_threshold=-999, dd_scale=1.0,
                           label="No Overlays",
                           dim="J: Risk Overlays")

    # -- K. Trend Sub-Weights (TS/XS only; S183 default = 100% TS) --
    trend_sub = [
        (1.00, 0.00, "100/0 (TS only, S183*)"),
        (0.75, 0.25, "75/25 (TS-heavy)"),
        (0.50, 0.50, "50/50 (S182)"),
        (0.25, 0.75, "25/75 (XS-heavy)"),
        (0.00, 1.00, "0/100 (XS only)"),
    ]
    for wts, wxs, lbl in trend_sub:
        V[f"K_ts{int(round(wts*100))}_xs{int(round(wxs*100))}"] = dict(
            w_ts=wts, w_xs=wxs,
            label=f"TS/XS={lbl}", dim="K: Trend Sub-Weights")

    # -- L. XS Lookback -- (S183 MASTER = 256, base-2 canonical year)
    for lb in [126, 192, 252, 256, 378, 512]:
        is_master = (lb == 256)
        V[f"L_xslb_{lb}"] = dict(xs_lookback=lb,
                                   label=f"XS_lookback={lb}"
                                         + (" (S183*)" if is_master else ""),
                                   dim="L: XS Lookback")

    # -- M. VoV Window -- (S183 MASTER = 64, base-2 canonical quarter)
    for vw in [21, 42, 63, 64, 90, 126]:
        is_master = (vw == 64)
        V[f"M_vovwin_{vw}"] = dict(vov_window=vw,
                                    label=f"VoV_window={vw}"
                                          + (" (S183*)" if is_master else ""),
                                    dim="M: VoV Window")

    # -- N. VoV Direction Overlay (S183 default = ON) --
    # S183_CFG keeps the S179 VoV direction overlay: removing it is the only
    # element that flips daily return signs and costs post-2010 Sharpe.
    V["N_vov_dir_on"]  = dict(vov_use_direction=True,
                                label="VoV Direction ON (S183*)",
                                dim="N: VoV Direction")
    V["N_vov_dir_off"] = dict(vov_use_direction=False,
                                label="VoV Direction OFF (S180 report delta 6)",
                                dim="N: VoV Direction")

    # -- O. VoV Weight --
    vov_weights = [
        (1.0/3, 1.0/3, 1.0/3, 0.00, "0% (no VoV)"),
        (0.30, 0.30, 0.30, 0.10, "10%"),
        (0.2833, 0.2833, 0.2834, 0.15, "15%"),
        (0.2667, 0.2667, 0.2666, 0.20, "20%"),
        (0.25, 0.25, 0.25, 0.25, "25% (S180)"),
        (0.2333, 0.2333, 0.2334, 0.30, "30%"),
        (0.20, 0.20, 0.20, 0.40, "40%"),
    ]
    for wt, wc, ws, wv, lbl in vov_weights:
        V[f"O_vovw_{int(round(wv*100)):02d}"] = dict(
            w_trend=wt, w_carry=wc, w_skew=ws, w_vov=wv,
            label=f"VoV_w={lbl}", dim="O: VoV Weight")

    # -- P. FDM Pooling Mode --
    V["P_fdm_per_inst"] = dict(fdm_mode=FDM_MODE_PERINST,
                                label="FDM per-instrument (=S161)",
                                dim="P: FDM Pooling Mode")
    V["P_fdm_pooled"]   = dict(fdm_mode=FDM_MODE_POOLED,
                                label="FDM pooled (S180)",
                                dim="P: FDM Pooling Mode")
    V["P_fdm_shrink"]   = dict(fdm_mode=FDM_MODE_SHRINK,
                                label="FDM pooled + LW shrink a=0.5",
                                dim="P: FDM Pooling Mode")
    V["P_fdm_clip"]     = dict(fdm_mode=FDM_MODE_CLIP,
                                label="FDM pooled + off-diag clip at mean",
                                dim="P: FDM Pooling Mode")

    # -- Q. Pooled-FDM Correlation Span --
    for span in [128, 256, 512, 1024, 2048]:
        V[f"Q_pooled_span_{span}"] = dict(
            fdm_mode=FDM_MODE_POOLED, fdm_corr_span=span,
            label=f"Pooled FDM span={span}",
            dim="Q: Pooled FDM Corr Span")

    # -- R. FDM Min Periods --
    for mp in [128, 256, 384]:
        V[f"R_fdm_min_{mp}"] = dict(
            fdm_mode=FDM_MODE_POOLED, fdm_min_periods=mp,
            label=f"Pooled FDM min_periods={mp}",
            dim="R: FDM Min Periods")

    return V


DM_VARIANTS = _build_dm_variants()


# ---------------------------------------------------------------------------
# DM Phase 0 -- forecast library shared across all DM variants
# ---------------------------------------------------------------------------

def dm_precompute_library(mapping, fx_daily, paths):
    print("[DM PHASE 0] Building forecast library (S180 anchor) ...")
    inst_data_cache = {}
    inst_list = []
    for inst in tqdm(mapping.index, desc="Loading"):
        data = load_instrument_data(inst, paths["stats"], paths["panama"])
        if data is None or len(data["daily_dates"]) < OOS_START:
            continue
        inst_data_cache[inst] = data
        inst_list.append(inst)

    library = {}
    xs_raw_by_lb = {lb: {} for lb in DM_ALL_XS_LOOKBACKS}

    for instrument in tqdm(inst_list, desc="Library"):
        data = inst_data_cache[instrument]
        dates = data["daily_dates"]
        if len(dates) < OOS_START + 2:
            continue
        ccy = mapping.loc[instrument, "currency"]
        fx = align_fx(ccy, dates, fx_daily)
        if fx is None:
            continue

        close = data["close"]
        price_changes = data["price_changes"]
        vol = blended_volatility(price_changes)
        idx = pd.DatetimeIndex(dates)

        ewmac = {}
        for fast, slow in DM_ALL_SPEED_PAIRS:
            raw = compute_ewmac_raw(close, vol, fast, slow)
            scaled = cap_forecast(raw * rolling_forecast_scalar(raw))
            ewmac[(fast, slow)] = pd.Series(
                scaled.reindex(idx).fillna(0.0).values, index=idx)

        ewmac_ramped = {}
        for (fast, slow), ungated in ewmac.items():
            for cw in DM_ALL_CONV_WINDOWS:
                ewmac_ramped[(fast, slow, cw)] = _apply_conviction(ungated, cw)

        raw_carry = compute_carry(instrument, paths["panama"],
                                   paths["contracts"], dates, vol)
        if raw_carry is not None and raw_carry.dropna().any():
            carry_fc = cap_forecast(
                raw_carry.fillna(0.0) * rolling_forecast_scalar(raw_carry.fillna(0.0)))
        else:
            carry_fc = pd.Series(0.0, index=idx)
        carry_fc = pd.Series(carry_fc.reindex(idx).fillna(0.0).values, index=idx)

        skew = {}
        for sw in DM_ALL_SKEW_WINDOWS:
            raw_skew = -price_changes.rolling(
                sw, min_periods=min(sw // 2, 128)).skew().fillna(0.0)
            skew_fc = cap_forecast(raw_skew * rolling_forecast_scalar(raw_skew))
            skew[sw] = pd.Series(skew_fc.reindex(idx).fillna(0.0).values, index=idx)

        norm_price = (price_changes / vol.replace(0.0, np.nan)).fillna(0.0).cumsum()
        for lb in DM_ALL_XS_LOOKBACKS:
            xs_raw_by_lb[lb][instrument] = norm_price.diff(lb).reindex(idx)

        vov = {}
        for vw in DM_ALL_VOV_WINDOWS:
            for use_dir in (True, False):
                vov_raw = _build_vov_raw(price_changes, close, vw, use_dir)
                vov_fc = cap_forecast(vov_raw * rolling_forecast_scalar(vov_raw))
                vov[(vw, use_dir)] = pd.Series(
                    vov_fc.reindex(idx).fillna(0.0).values, index=idx)

        library[instrument] = dict(
            ewmac=ewmac, ewmac_ramped=ewmac_ramped,
            carry=carry_fc, skew=skew, xs_fc={}, vov=vov,
            vol=vol.reindex(idx), fx=fx.reindex(idx),
            close=data["close"].reindex(idx),
            open=data["open"].reindex(idx),
            cost_rt=float(mapping.loc[instrument, "total_avg_cost_rt"]),
            pointsize=float(mapping.loc[instrument, "pointsize"]),
            dates=dates, idx=idx,
            oos_date_set=set(dates[OOS_START:]),
            daily_ret=close.pct_change(),
        )

    print("  Cross-sectional z-scoring at all XS lookbacks ...")
    for lb in DM_ALL_XS_LOOKBACKS:
        xs_df = pd.DataFrame(xs_raw_by_lb[lb])
        n_valid = xs_df.notna().sum(axis=1)
        cs_mean = xs_df.mean(axis=1)
        cs_std = xs_df.std(axis=1).replace(0.0, np.nan)
        xs_z = xs_df.subtract(cs_mean, axis=0).divide(cs_std, axis=0)
        xs_z[n_valid < XS_MIN_INSTS] = np.nan
        for inst in library:
            if inst not in xs_z.columns:
                continue
            z = xs_z[inst].fillna(0.0)
            xsf = cap_forecast(z * rolling_forecast_scalar(z))
            library[inst]["xs_fc"][lb] = pd.Series(
                xsf.reindex(library[inst]["idx"]).fillna(0.0).values,
                index=library[inst]["idx"])

    print(f"  Library: {len(library)} instruments.")
    return library


def _dm_preflight_coverage(library, params, variant_id):
    """Validate that the variant's requested windows were pre-baked.

    dm_precompute_library bakes a fixed superset.  If a variant asks for
    a window outside that set, the old code silently fell back to
    pd.Series(0.0) -- a documented bug that contaminated the S180 suite.
    This check is kept byte-equal to the pre-refactor logic and raises
    loudly if any requested window is missing.
    """
    wt = params["w_trend"]; wc = params["w_carry"]
    ws = params["w_skew"];  wv = params["w_vov"]
    coverage_errors = []
    if wt > 0 and params.get("w_ts", 0) > 0:
        pairs = params["speed_pairs"]
        cw = params["conviction_window"]
        use_con = params["use_conviction"]
        for fast, slow in pairs:
            if (fast, slow) not in (library.get(next(iter(library)), {})
                                    .get("ewmac", {}) if library else {}):
                coverage_errors.append(
                    f"EWMAC speed pair ({fast},{slow}) not in library "
                    f"(baked: DM_ALL_SPEED_PAIRS={DM_ALL_SPEED_PAIRS})")
            if use_con and (fast, slow, cw) not in (
                library.get(next(iter(library)), {}).get("ewmac_ramped", {})
                if library else {}):
                coverage_errors.append(
                    f"Conviction-ramped EWMAC ({fast},{slow},cw={cw}) not "
                    f"in library (baked cw values: {DM_ALL_CONV_WINDOWS})")
    if wt > 0 and params["xs_lookback"] not in DM_ALL_XS_LOOKBACKS:
        coverage_errors.append(
            f"xs_lookback={params['xs_lookback']} not in "
            f"DM_ALL_XS_LOOKBACKS={DM_ALL_XS_LOOKBACKS}")
    if ws > 0 and params["skew_window"] not in DM_ALL_SKEW_WINDOWS:
        coverage_errors.append(
            f"skew_window={params['skew_window']} not in "
            f"DM_ALL_SKEW_WINDOWS={DM_ALL_SKEW_WINDOWS}")
    if wv > 0 and params["vov_window"] not in DM_ALL_VOV_WINDOWS:
        coverage_errors.append(
            f"vov_window={params['vov_window']} not in "
            f"DM_ALL_VOV_WINDOWS={DM_ALL_VOV_WINDOWS}")
    if coverage_errors:
        msg = (f"[dm_build_variant_signals] variant {variant_id!r}: "
               f"cfg requests windows not in the pre-baked library. "
               f"Fix by adding these values to the corresponding DM_ALL_* "
               f"list and rerunning dm_precompute_library. Errors:\n  - "
               + "\n  - ".join(coverage_errors))
        raise ValueError(msg)


def _dm_build_alpha_bundle(library, params):
    """Project the pre-baked DM library into the `alpha_bundle` shape
    consumed by `assemble_signals_from_alphas`.

    Picks EWMAC speeds, XS, skew, VoV at the variant's requested windows
    from the library; no recomputation.  Stage B (trend sub-blend, FDM)
    and Stage C (smoothing, gates) are then handled by the canonical
    assembler.
    """
    wt = params["w_trend"]; wc = params["w_carry"]
    ws = params["w_skew"];  wv = params["w_vov"]
    pairs   = params["speed_pairs"]
    weights = params["speed_weights"]
    cw      = params["conviction_window"]
    use_con = params["use_conviction"]
    w_ts_l  = params["w_ts"]
    xs_lb   = params["xs_lookback"]
    sw      = params["skew_window"]
    vw_p    = params["vov_window"]
    vdir    = params["vov_use_direction"]

    bundle = {}
    for instrument, lib in library.items():
        idx = lib["idx"]

        # TS-trend: multi-speed EWMAC blend, matches canonical Stage A.
        if wt > 0 and w_ts_l > 0:
            ts_components = []
            for i, (fast, slow) in enumerate(pairs):
                if use_con:
                    fc = lib["ewmac_ramped"].get((fast, slow, cw))
                    if fc is None:
                        fc = lib["ewmac"].get((fast, slow), pd.Series(0.0, index=idx))
                else:
                    fc = lib["ewmac"].get((fast, slow), pd.Series(0.0, index=idx))
                ts_components.append(fc * weights[i])
            ts_trend = cap_forecast(sum(ts_components) * params["trend_fdm"])
        else:
            ts_trend = pd.Series(0.0, index=idx)

        # XS: pre-baked cap-scaled forecast at the variant's xs_lookback
        xs_fc = lib["xs_fc"].get(xs_lb, pd.Series(0.0, index=idx))
        xs_fc = pd.Series(xs_fc.reindex(idx).fillna(0.0).values, index=idx)

        # Carry: library stores already cap-scaled series (wc==0 zeros it)
        carry_fc = lib["carry"] if wc > 0 else pd.Series(0.0, index=idx)
        carry_fc = pd.Series(carry_fc.reindex(idx).fillna(0.0).values, index=idx)

        # Skew, VoV: pre-baked at (window, direction) keys
        skew_fc = (lib["skew"].get(sw, pd.Series(0.0, index=idx))
                   if ws > 0 else pd.Series(0.0, index=idx))
        skew_fc = pd.Series(skew_fc.reindex(idx).fillna(0.0).values, index=idx)

        if wv > 0:
            vov_fc = lib["vov"].get((vw_p, vdir), pd.Series(0.0, index=idx))
        else:
            vov_fc = pd.Series(0.0, index=idx)
        vov_fc = pd.Series(vov_fc.reindex(idx).fillna(0.0).values, index=idx)

        # Build the bundle.  "xs_fc" key signals to the assembler that
        # Stage B.1 cross-sectional z-scoring is ALREADY done (the suite
        # pre-bakes it per lookback); no re-z-scoring.
        dates = list(idx)
        bundle[instrument] = dict(
            idx=idx, dates=dates,
            ts_trend=ts_trend,
            final_carry=carry_fc,
            final_skew=skew_fc,
            final_vov=vov_fc,
            xs_fc=xs_fc,
            vol=lib["vol"], fx=lib["fx"],
            close=lib["close"], open=lib["open"],
            pointsize=lib["pointsize"],
            cost_rt=lib["cost_rt"],
        )
    return bundle


def _dm_compute_fdm_override(bundle, params):
    """Compute the FDM series the suite's mode requires.

    Returns one of:
      None               -> assembler uses its own pooled FDM (mode="pooled" only)
      pd.Series          -> universe-pooled FDM (pooled/shrink/clip)
      dict[inst, Series] -> per-instrument FDM (fdm_mode="per_inst")
      pd.Series (const)  -> static (constant fdm_cap)

    The non-pooled modes reuse the suite's legacy helpers
    (_pooled_fdm_from_panels with mode!=POOLED, _perinst_fdm) rather than
    re-deriving them in `assemble_signals_from_alphas`.  This preserves
    the suite's original numeric behaviour on shrink / clip / per_inst
    ablation variants while keeping every downstream computation inside
    the unified assembler.
    """
    wt = params["w_trend"]; wc = params["w_carry"]
    ws = params["w_skew"];  wv = params["w_vov"]
    fdm_mode = params.get("fdm_mode", FDM_MODE_POOLED)

    if fdm_mode == FDM_MODE_POOLED:
        return None                           # let assembler compute canonical pooled-4 FDM

    if fdm_mode in (FDM_MODE_SHRINK, FDM_MODE_CLIP):
        panel_dict = OrderedDict()
        w_dict = OrderedDict()
        insts = list(bundle.keys())
        if wt > 0:
            # assembler expects a combined trend = w_ts*ts + w_xs*xs but
            # for FDM correlation estimation the suite pools ts_trend + xs
            # separately.  We rebuild the combined trend here via the same
            # formula the assembler uses (cap_forecast(w_ts*ts + w_xs*xs)).
            def _tr(i):
                return cap_forecast(params["w_ts"] * bundle[i]["ts_trend"]
                                    + params["w_xs"] * bundle[i]["xs_fc"])
            panel_dict["trend"] = pd.DataFrame({i: _tr(i) for i in insts})
            w_dict["trend"] = wt
        if wc > 0:
            panel_dict["carry"] = pd.DataFrame({i: bundle[i]["final_carry"] for i in insts})
            w_dict["carry"] = wc
        if ws > 0:
            panel_dict["skew"] = pd.DataFrame({i: bundle[i]["final_skew"] for i in insts})
            w_dict["skew"] = ws
        if wv > 0:
            panel_dict["vov"] = pd.DataFrame({i: bundle[i]["final_vov"] for i in insts})
            w_dict["vov"] = wv
        if len(panel_dict) >= 2:
            return _pooled_fdm_from_panels(
                panel_dict, w_dict,
                params["fdm_corr_span"], params["fdm_min_periods"],
                params["fdm_floor"], params["fdm_cap"], fdm_mode)
        first_idx = bundle[insts[0]]["idx"]
        return pd.Series(1.0, index=first_idx)

    if fdm_mode == FDM_MODE_PERINST:
        # Build per-instrument FDM: each instrument gets its own FDM series
        # from its own cross-alpha correlation (not universe-pooled).
        out = {}
        for inst, b in bundle.items():
            active_fc = OrderedDict()
            active_w = OrderedDict()
            if wt > 0:
                trend_combined = cap_forecast(params["w_ts"] * b["ts_trend"]
                                              + params["w_xs"] * b["xs_fc"])
                active_fc["trend"] = trend_combined; active_w["trend"] = wt
            if wc > 0:
                active_fc["carry"] = b["final_carry"]; active_w["carry"] = wc
            if ws > 0:
                active_fc["skew"]  = b["final_skew"];  active_w["skew"]  = ws
            if wv > 0:
                active_fc["vov"]   = b["final_vov"];   active_w["vov"]   = wv
            if len(active_fc) >= 2:
                out[inst] = _perinst_fdm(
                    active_fc, active_w,
                    params["fdm_corr_span"], params["fdm_min_periods"],
                    params["fdm_floor"], params["fdm_cap"])
            else:
                out[inst] = pd.Series(1.0, index=b["idx"])
        return out

    raise NotImplementedError(f"Unknown fdm_mode: {fdm_mode!r}")


def dm_build_variant_signals(library, cfg, variant_id=""):
    """Build inst_signals for a DM variant via the UNIFIED canonical pipeline.

    Post-15-Apr refactor: this function no longer re-implements Stage B+C.
    It (1) validates the pre-bake coverage, (2) projects the library into
    an `alpha_bundle`, (3) computes the FDM series if the variant uses a
    non-pooled mode, (4) calls `assemble_signals_from_alphas` with a
    canonical-schema cfg that mirrors S183_CFG's knob overrides.  Any
    drift between this path and canonical is now impossible by
    construction -- both funnel through the same assembler.
    """
    params = {**S183_CFG, **{k: v for k, v in cfg.items() if k not in ("label", "dim")}}
    _dm_preflight_coverage(library, params, variant_id)

    print(f"[DM {variant_id or '(anon)'}] projecting pre-baked library -> alpha_bundle "
          f"(fdm_mode={params.get('fdm_mode', FDM_MODE_POOLED)}, smooth_span={params['smooth_span']})")
    bundle = _dm_build_alpha_bundle(library, params)

    # FDM override for non-pooled modes; None lets the assembler compute it.
    fdm_override = _dm_compute_fdm_override(bundle, params)

    canonical_cfg = _s183cfg_to_canonical(params)
    if fdm_override is not None:
        canonical_cfg["pooled_fdm_override"] = fdm_override

    return assemble_signals_from_alphas(bundle, canonical_cfg)


def _dm_ck_path(vid):
    return DM_DIR / f"DM183_{vid}_checkpoint.pkl"


def _dm_load_ck(vid):
    p = _dm_ck_path(vid)
    if p.exists():
        with open(str(p), "rb") as fh:
            return pickle.load(fh)
    return None


def run_datamining(paths, args):
    print("\n" + "=" * 78)
    print("  DATAMINING SWEEP -- Strategy 183 (S180 Apr-7 16:36 frozen build)")
    print("=" * 78)
    ensure_dirs()
    mapping = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])
    library = dm_precompute_library(mapping, fx_daily, paths)

    # MASTER anchor -- S180 defaults
    print("\n[DM] Running S183 MASTER anchor ...")
    ck = None
    if args.skip_existing and _dm_ck_path("ANCHOR").exists():
        ck = _dm_load_ck("ANCHOR")
        print("  [SKIP] MASTER checkpoint exists.")
    else:
        try:
            inst_signals = dm_build_variant_signals(library, {}, variant_id="ANCHOR")
            dm_paths = {**paths, "output": str(DM_DIR)}
            ck = run_compounded_portfolio(inst_signals, "DM183_ANCHOR", dm_paths)
        except Exception as exc:
            _log_failure("DM", "ANCHOR", exc)

    # Sweep
    variant_keys = list(DM_VARIANTS.keys())
    if args.variants:
        variant_keys = [v for v in variant_keys if v in set(args.variants)]

    n_workers = int(getattr(args, "workers", 1))
    if n_workers > 1:
        tasks = [(vid, DM_VARIANTS[vid], args.skip_existing)
                 for vid in variant_keys]
        with multiprocessing.Pool(
            processes=n_workers,
            initializer=_pool_init_dm,
            initargs=(library, paths),
        ) as pool:
            for vid, status, err in tqdm(
                pool.imap_unordered(_dm_worker, tasks),
                total=len(tasks),
                desc="DM sweep",
                unit="variant",
            ):
                if status == "fail":
                    _log_failure_raw("DM", vid, *err)
    else:
        for vid in tqdm(variant_keys, desc="DM sweep", unit="variant"):
            if args.skip_existing and _dm_ck_path(vid).exists():
                continue
            try:
                inst_signals = dm_build_variant_signals(library, DM_VARIANTS[vid],
                                                         variant_id=vid)
                dm_paths = {**paths, "output": str(DM_DIR)}
                run_compounded_portfolio(inst_signals, f"DM183_{vid}", dm_paths)
            except Exception as exc:
                _log_failure("DM", vid, exc)


# ===========================================================================
# SECTION 4 -- ABLATION module
# ===========================================================================

def _abl_cfg(**kw):
    base = dict(
        w_ts=S183_CFG["w_ts"], w_xs=S183_CFG["w_xs"],
        speeds=[0, 1, 2], use_conviction=S183_CFG["use_conviction"],  # v3.4 Triple {(16,64),(32,128),(64,256)} (L&P 2016 canonical 3-speed)
        w_trend=S183_CFG["w_trend"], w_carry=S183_CFG["w_carry"],
        w_skew=S183_CFG["w_skew"], w_vov=S183_CFG["w_vov"],
        trend_fdm=S183_CFG["trend_fdm"],
        use_smooth=True, use_overlays=True,
        static_fdm=None, cost_mult=1.0,
        vov_window=S183_CFG["vov_window"],
        vov_use_direction=S183_CFG["vov_use_direction"],
        fdm_mode=S183_CFG["fdm_mode"],
        fdm_cap=S183_CFG["fdm_cap"],
        fdm_corr_span=S183_CFG["fdm_corr_span"],
        fdm_min_periods=S183_CFG["fdm_min_periods"],
        fdm_floor=S183_CFG["fdm_floor"],
        smooth_span=S183_CFG["smooth_span"],
        vol_trigger=S183_CFG["vol_trigger"],
        vol_dampen=S183_CFG["vol_dampen"],
        dd_threshold=S183_CFG["dd_threshold"],
        dd_scale=S183_CFG["dd_scale"],
        sigmoid_steepness=S183_CFG["sigmoid_steepness"],
        use_shifted_sigmoid=S183_CFG["use_shifted_sigmoid"],  # S183 default: False (standard logistic)
    )
    base.update(kw)
    return base


ABL_VARIANTS = {

    # =====================================================================
    # GROUP 1a: S183 DEFINING DELTAS (revert each toward S182)
    # Each variant reverts exactly ONE of the four S183 deltas back to
    # its S182 value. These are the core "did the S183 combo stack hold
    # when measured directly, one delta at a time?" tests.  Collectively
    # they retrace the interior of GROUP 10 from the S182 suite.
    # =====================================================================
    "R_S183_ADD_CONV":   _abl_cfg(
        label="S183 + conviction ramp ON (revert use_conviction False->True)",
        use_conviction=True),
    "R_S183_NO_XS":      _abl_cfg(
        label="S183 - XS (drop w_ts=0.5/0.5 -> 1.0/0.0, = S183 v1)",
        w_ts=1.0, w_xs=0.0),
    "R_S183_STEEP1":     _abl_cfg(
        label="S183 + sigmoid steepness 10 -> 1 (revert to S182 identity)",
        sigmoid_steepness=1.0),
    "R_S183_SMOOTH5":    _abl_cfg(
        label="S183 + smooth_span 3 -> 5 (revert to S182 one-trading-week)",
        smooth_span=5),
    "R_S183_2SPEED":     _abl_cfg(
        label="S183 -> revert to v3.3 2-speed {(16,64),(64,256)} (drop middle)",
        speeds=[0, 2]),
    "R_S183_TFDM110":    _abl_cfg(
        label="S183 + TREND_FDM = 1.10 (revert identity removal)",
        trend_fdm=1.10),
    "R_S183_FULL_REVERT": _abl_cfg(
        label="Revert S183 v3.4 deltas -> S182-like (conv ON + steep=1), Triple speeds retained",
        use_conviction=True,
        sigmoid_steepness=1.0),
        # Note: cannot exactly reproduce S182's original (16,64)+(64,256)+(128,512)
        # 3-speed without extending TREND_SPEEDS_ABL because (128,512) is no
        # longer in the precomputed speed library.  This variant tests the
        # conviction+steepness reverts with the current Triple speeds held fixed.

    # =====================================================================
    # GROUP 1b: S183 CROSS-DIRECTION ROBUSTNESS
    # Each of these pushes a single S183 delta FURTHER in the same
    # direction (more aggressive).  Intended to probe whether S183 sits
    # on a plateau or at the edge of a cliff.
    # =====================================================================
    "R_S183_STEEP20":     _abl_cfg(
        label="Push sigmoid steepness 10 -> 20 (harder cliff-edge)",
        sigmoid_steepness=20.0),
    "R_S183_SMOOTH1":     _abl_cfg(
        label="Push smooth_span 3 -> 1 (no smoothing)",
        smooth_span=1),
    "R_S183_SMOOTH8":     _abl_cfg(
        label="Push smooth_span 3 -> 8 (much longer)",
        smooth_span=8),
    "R_S183_STEEP5":      _abl_cfg(
        label="Middle ground sigmoid steepness 10 -> 5 (halfway to S182)",
        sigmoid_steepness=5.0),
    "R_FDMCAP180":        _abl_cfg(
        label="Tighter FDM_CAP = 1.80 (S172 datamined value)",
        fdm_cap=1.80),
    "R_FDMCAP_INF":       _abl_cfg(
        label="Revert FDM_CAP sqrt(N)=2 -> inf (uncapped)",
        fdm_cap=float("inf")),
    "R_SHIFTED":          _abl_cfg(
        label="Switch to shifted sigmoid (zero-drag below trigger)",
        use_shifted_sigmoid=True),
    "R_NO_VOVDIR":        _abl_cfg(
        label="Remove VoV direction weighting (S180 report delta 6)",
        vov_use_direction=False),
    "R_TFDM110":          _abl_cfg(
        label="Revert TREND_FDM 1.0 -> 1.10 (S179 bonus)",
        trend_fdm=1.10),

    # =====================================================================
    # GROUP 2: FDM ARCHITECTURE
    # =====================================================================
    "R_PERFDM":   _abl_cfg(label="Revert FDM pooling -> per-instrument (=S161)",
                            fdm_mode=FDM_MODE_PERINST),
    "FDM_SHRINK": _abl_cfg(label="Pooled FDM + LW shrinkage to I (a=0.5)",
                            fdm_mode=FDM_MODE_SHRINK),
    "FDM_CLIP":   _abl_cfg(label="Pooled FDM with off-diag clipped at mean",
                            fdm_mode=FDM_MODE_CLIP),
    "D":          _abl_cfg(label="Static FDM = 1.20", static_fdm=1.20),
    "D_150":      _abl_cfg(label="Static FDM = 1.50 (Carver heuristic)", static_fdm=1.50),
    "D_200":      _abl_cfg(label="Static FDM = 2.00 (= S180 cap)", static_fdm=2.00),

    # =====================================================================
    # GROUP 3: ALPHA COMPONENT ABLATIONS
    # =====================================================================
    # VoV weight
    "VOV0":   _abl_cfg(label="Drop VoV: revert to Trinity 1/3 T/C/S",
                        w_trend=1.0/3, w_carry=1.0/3, w_skew=1.0/3, w_vov=0.0),
    "VOV_H":  _abl_cfg(label="VoV weight halved (T/C/S 29.17%, VoV 12.5%)",
                        w_trend=0.2917, w_carry=0.2917, w_skew=0.2916, w_vov=0.125),
    "VOV_D":  _abl_cfg(label="VoV weight doubled (T/C/S 16.67%, VoV 50%)",
                        w_trend=1.0/6, w_carry=1.0/6, w_skew=1.0/6, w_vov=0.50),
    "VOV_W21":  _abl_cfg(label="VoV window = 21d (short)", vov_window=21),
    "VOV_W42":  _abl_cfg(label="VoV window = 42d", vov_window=42),
    "VOV_W126": _abl_cfg(label="VoV window = 126d (long)", vov_window=126),

    # Component drops
    "B2": _abl_cfg(label="Drop Carry (T/S/V at 1/3, 3-way FDM)",
                    w_trend=1.0/3, w_carry=0.00, w_skew=1.0/3, w_vov=1.0/3),
    "B3": _abl_cfg(label="Drop Skew (T/C/V at 1/3, 3-way FDM)",
                    w_trend=1.0/3, w_carry=1.0/3, w_skew=0.00, w_vov=1.0/3),
    "B4": _abl_cfg(label="Drop Trend (C/S/V at 1/3) -- mandate breach",
                    w_ts=0.00, w_xs=0.00, speeds=[],
                    w_trend=0.00, w_carry=1.0/3, w_skew=1.0/3, w_vov=1.0/3),
    "B5": _abl_cfg(label="100% Trend Only (no carry, skew, vov)",
                    w_trend=1.00, w_carry=0.00, w_skew=0.00, w_vov=0.00),

    # Trend sub-blend (S183 default = 100% TS; XS0 dropped as it equals
    # the S183 master).  TS0 tests the opposite extreme (100% XS); the two
    # intermediate blends sit between S182 and S183.
    "TS0": _abl_cfg(label="No TS-EWMAC (100% XS in trend)", w_ts=0.0, w_xs=1.0),
    "TS75_XS25": _abl_cfg(label="75/25 TS/XS trend blend", w_ts=0.75, w_xs=0.25),
    "TS25_XS75": _abl_cfg(label="25/75 TS/XS trend blend", w_ts=0.25, w_xs=0.75),

    # EWMAC speeds
    "A1": _abl_cfg(label="Drop fast (16,64): alt 2-speed (32,128)+(64,256)", speeds=[1, 2]),
    "A2": _abl_cfg(label="Drop mid (32,128): alt 2-speed (16,64)+(64,256) = v3.3 pre-Triple", speeds=[0, 2]),
    "A3": _abl_cfg(label="Drop slow (64,256): alt 2-speed (16,64)+(32,128)", speeds=[0, 1]),

    # =====================================================================
    # GROUP 4: INFRASTRUCTURE
    # =====================================================================
    "C1": _abl_cfg(label="No Turnover Reduction (no EWM smoothing)",
                    use_smooth=False),
    "C2": _abl_cfg(label="No Risk Overlays (no sigmoid vol/DD)",
                    use_overlays=False),
    "C3": _abl_cfg(label="No Infrastructure (no smooth/overlays)",
                    use_smooth=False, use_overlays=False),

    # =====================================================================
    # GROUP 5: OVERLAY SENSITIVITY
    # =====================================================================
    "R_S158":    _abl_cfg(label="Revert refinements -> S158 smoothing+overlays",
                            smooth_span=3,
                            vol_trigger=2.0, vol_dampen=0.85,
                            dd_threshold=-0.04, dd_scale=0.65),
    "R_JDEF":    _abl_cfg(label="Revert overlays only (J_tighter -> J_default)",
                            vol_trigger=2.0, vol_dampen=0.85,
                            dd_threshold=-0.04, dd_scale=0.65),
    "R_JTIGHT":  _abl_cfg(label="Revert J_tighter -> S168 J_tight bundle",
                            vol_trigger=1.5, vol_dampen=0.75,
                            dd_threshold=-0.02, dd_scale=0.50),
    "OV_DAMP25": _abl_cfg(label="Overlay dampen 25% (lighter defence)",
                            vol_dampen=0.25, dd_scale=0.25),
    "OV_DAMP75": _abl_cfg(label="Overlay dampen 75% (heavier defence)",
                            vol_dampen=0.75, dd_scale=0.75),

    # =====================================================================
    # GROUP 6: WEIGHT SENSITIVITY
    # =====================================================================
    "E": _abl_cfg(label="Trend-Heavy 40/20/20/20",
                   w_trend=0.40, w_carry=0.20, w_skew=0.20, w_vov=0.20),
    "F": _abl_cfg(label="Skew-Heavy 20/20/40/20",
                   w_trend=0.20, w_carry=0.20, w_skew=0.40, w_vov=0.20),
    "E2": _abl_cfg(label="Carry-Heavy 20/40/20/20",
                    w_trend=0.20, w_carry=0.40, w_skew=0.20, w_vov=0.20),
    "E3": _abl_cfg(label="VoV-Heavy 20/20/20/40",
                    w_trend=0.20, w_carry=0.20, w_skew=0.20, w_vov=0.40),

    # =====================================================================
    # GROUP 7: COST SENSITIVITY
    # =====================================================================
    "G":      _abl_cfg(label="Frictionless (cost_rt = $0)", cost_mult=0.0),
    "G_150":  _abl_cfg(label="1.5x costs", cost_mult=1.5),
    "G_200":  _abl_cfg(label="2x costs", cost_mult=2.0),
    "G_300":  _abl_cfg(label="3x costs", cost_mult=3.0),

    # =====================================================================
    # GROUP 8: ADVERSARIAL ARCHITECTURE TESTS
    # =====================================================================
    "D_100":  _abl_cfg(label="Static FDM = 1.00 (no diversification correction)",
                        static_fdm=1.00),
    # Note: S183_CFG's MASTER uses the standard (non-shifted) sigmoid, so
    # R_OLDSIG would be a no-op. R_SHIFTED in GROUP 1b tests the opposite
    # direction (enabling the zero-drag shifted form).

    # =====================================================================
    # GROUP 9: S183 v2 PAIRWISE DELTA INTERACTION TESTS
    # =====================================================================
    # S183 v2 has THREE deltas vs S182 (conviction off, steep 10, smooth 3).
    # Each variant reverts TWO deltas simultaneously to probe pairwise
    # interactions.  Three additional probes test what happens if we
    # layer the S183 v1 mistake (XS removal) on top of a v2 revert.
    "COMBO_ADDCONV_STEEP1": _abl_cfg(
        label="Revert 2: conviction ON + sigmoid steepness 10->1",
        use_conviction=True, sigmoid_steepness=1.0),
    "COMBO_ADDCONV_SMOOTH5": _abl_cfg(
        label="Revert 2: conviction ON + smooth_span 3->5",
        use_conviction=True, smooth_span=5),
    "COMBO_STEEP1_SMOOTH5": _abl_cfg(
        label="Revert 2: sigmoid 10->1 + smooth_span 3->5 (= S182 infra)",
        sigmoid_steepness=1.0, smooth_span=5),

    # Keep-only-one-delta combos
    "COMBO_KEEP_NOCONV_ONLY": _abl_cfg(
        label="Keep only no-conviction; revert steep, smooth to S182",
        sigmoid_steepness=1.0, smooth_span=5),
    "COMBO_KEEP_STEEP10_ONLY": _abl_cfg(
        label="Keep only sigmoid=10; revert conviction, smooth to S182",
        use_conviction=True, smooth_span=5),
    "COMBO_KEEP_SMOOTH3_ONLY": _abl_cfg(
        label="Keep only smooth=3; revert conviction, steep to S182",
        use_conviction=True, sigmoid_steepness=1.0),

    # S183 v1 probes: apply the XS removal on top of v2 reverts
    "COMBO_NOXS_NOCONV": _abl_cfg(
        label="Drop XS + keep no-conviction (= S183 v1 candidate)",
        w_ts=1.0, w_xs=0.0),
    "COMBO_NOXS_STEEP1": _abl_cfg(
        label="Drop XS + sigmoid 10->1",
        w_ts=1.0, w_xs=0.0, sigmoid_steepness=1.0),
    "COMBO_NOXS_SMOOTH5": _abl_cfg(
        label="Drop XS + smooth_span 3->5",
        w_ts=1.0, w_xs=0.0, smooth_span=5),

    # =====================================================================
    # GROUP 10: S183 LOAD-BEARING COMPONENT INTERACTION TESTS
    # =====================================================================
    # Each variant pairs the S183 defaults with the removal of one S182/S179
    # load-bearing component (VoV direction, pooled FDM, vol/DD overlays,
    # smoothing, VoV alpha).  Purpose: verify that the components that were
    # load-bearing under S182 remain load-bearing under the S183 regime.
    "COMBO_S183_NO_VOVDIR": _abl_cfg(
        label="S183 base + drop VoV direction overlay",
        vov_use_direction=False),
    "COMBO_S183_STATIC_FDM": _abl_cfg(
        label="S183 base + static FDM=1.0 (drop pooled 4x4 FDM)",
        static_fdm=1.0),
    "COMBO_S183_PERINST_FDM": _abl_cfg(
        label="S183 base + per-instrument FDM (instead of pooled)",
        fdm_mode=FDM_MODE_PERINST),
    "COMBO_S183_NO_VOV": _abl_cfg(
        label="S183 base + drop VoV alpha entirely (Trinity T/C/S)",
        w_trend=1.0/3, w_carry=1.0/3, w_skew=1.0/3, w_vov=0.0),
    "COMBO_S183_NO_OVERLAYS": _abl_cfg(
        label="S183 base + no vol/DD overlays (use_overlays=False)",
        use_overlays=False),
    "COMBO_S183_NO_SMOOTH": _abl_cfg(
        label="S183 base + no smoothing (use_smooth=False)",
        use_smooth=False),
}


TREND_SPEEDS_ABL = [
    {"fast": 16, "slow":  64, "weight": 1.0 / 3},
    {"fast": 32, "slow": 128, "weight": 1.0 / 3},
    {"fast": 64, "slow": 256, "weight": 1.0 / 3},
]
ABL_VOV_WINDOWS = [21, 42, 64, 90, 126]


def abl_precompute(mapping, fx_daily, paths):
    print("[ABL PHASE 0] Computing per-instrument raw signals ...")
    inst_data_cache = {}
    inst_list = []
    for inst in tqdm(mapping.index, desc="Loading"):
        data = load_instrument_data(inst, paths["stats"], paths["panama"])
        if data is None or len(data["daily_dates"]) < OOS_START:
            continue
        inst_data_cache[inst] = data
        inst_list.append(inst)

    raw_signals = {}
    for instrument in tqdm(inst_list, desc="Raw signals"):
        data = inst_data_cache[instrument]
        dates = data["daily_dates"]
        if len(dates) < OOS_START + 2:
            continue
        ccy = mapping.loc[instrument, "currency"]
        fx = align_fx(ccy, dates, fx_daily)
        if fx is None:
            continue

        close = data["close"]
        price_changes = data["price_changes"]
        vol = blended_volatility(price_changes)
        idx = pd.DatetimeIndex(dates)

        ewmac_ramped = []
        ewmac_flat = []
        for speed in TREND_SPEEDS_ABL:
            raw = compute_ewmac_raw(close, vol, speed["fast"], speed["slow"])
            scaled = cap_forecast(raw * rolling_forecast_scalar(raw))
            flat = pd.Series(scaled.reindex(idx).fillna(0.0).values, index=idx)
            ewmac_flat.append(flat)
            rolling_std = scaled.rolling(S183_CFG["conviction_window"],
                                          min_periods=128).std()
            conviction = (scaled.abs() / rolling_std.fillna(1.0)).clip(0.0, 1.0)
            ramped = scaled * conviction
            ewmac_ramped.append(
                pd.Series(ramped.reindex(idx).fillna(0.0).values, index=idx))

        raw_carry = compute_carry(instrument, paths["panama"],
                                   paths["contracts"], dates, vol)
        if raw_carry is not None and raw_carry.dropna().any():
            carry_fc = cap_forecast(
                raw_carry.fillna(0.0) * rolling_forecast_scalar(raw_carry.fillna(0.0)))
        else:
            carry_fc = pd.Series(0.0, index=idx)
        carry_fc = pd.Series(carry_fc.reindex(idx).fillna(0.0).values, index=idx)

        raw_skew = -price_changes.rolling(
            S183_CFG["skew_window"], min_periods=128).skew().fillna(0.0)
        skew_fc = cap_forecast(raw_skew * rolling_forecast_scalar(raw_skew))
        skew_fc = pd.Series(skew_fc.reindex(idx).fillna(0.0).values, index=idx)

        norm_price = (price_changes / vol.replace(0.0, np.nan)).fillna(0.0).cumsum()
        xs_raw = norm_price.diff(S183_CFG["xs_lookback"]).reindex(idx)

        vov_library = {}
        for w in ABL_VOV_WINDOWS:
            for use_dir in (True, False):
                vov_raw = _build_vov_raw(price_changes, close, w, use_dir)
                vov_fc = cap_forecast(vov_raw * rolling_forecast_scalar(vov_raw))
                vov_library[(w, use_dir)] = pd.Series(
                    vov_fc.reindex(idx).fillna(0.0).values, index=idx)

        raw_signals[instrument] = dict(
            ewmac_ramped=ewmac_ramped, ewmac_flat=ewmac_flat,
            carry_fc=carry_fc, skew_fc=skew_fc, xs_raw=xs_raw, xs_fc=None,
            vov_library=vov_library,
            vol=vol.reindex(idx), fx=fx.reindex(idx),
            close=data["close"].reindex(idx), open=data["open"].reindex(idx),
            cost_rt=float(mapping.loc[instrument, "total_avg_cost_rt"]),
            pointsize=float(mapping.loc[instrument, "pointsize"]),
            dates=dates, idx=idx,
            oos_date_set=set(dates[OOS_START:]),
            daily_ret=close.pct_change(),
        )

    print("  Cross-sectional z-scoring xs_raw ...")
    xs_df = pd.DataFrame({i: raw_signals[i]["xs_raw"] for i in raw_signals})
    n_valid = xs_df.notna().sum(axis=1)
    cs_mean = xs_df.mean(axis=1)
    cs_std = xs_df.std(axis=1).replace(0.0, np.nan)
    xs_z = xs_df.subtract(cs_mean, axis=0).divide(cs_std, axis=0)
    xs_z[n_valid < XS_MIN_INSTS] = np.nan
    for inst in raw_signals:
        z = xs_z[inst].fillna(0.0)
        xsf = cap_forecast(z * rolling_forecast_scalar(z))
        raw_signals[inst]["xs_fc"] = pd.Series(
            xsf.reindex(raw_signals[inst]["idx"]).fillna(0.0).values,
            index=raw_signals[inst]["idx"])

    print(f"  {len(raw_signals)} instruments precomputed.")
    return raw_signals


def _abl_build_alpha_bundle(raw_signals, cfg):
    """Project the ABL pre-baked raw_signals into the canonical alpha_bundle."""
    w_ts_l  = cfg["w_ts"]
    speeds  = cfg["speeds"]
    use_con = cfg["use_conviction"]
    wt      = cfg["w_trend"]
    wc      = cfg["w_carry"]
    ws      = cfg["w_skew"]
    wv      = cfg["w_vov"]
    t_fdm   = cfg["trend_fdm"]
    vovw    = cfg["vov_window"]
    vdir    = cfg["vov_use_direction"]

    bundle = {}
    for instrument, rs in raw_signals.items():
        idx = rs["idx"]

        # TS-trend: uniform-weighted blend over `speeds` indices into ABL_TREND_SPEEDS_KEYS
        if speeds and w_ts_l > 0 and wt > 0:
            source = rs["ewmac_ramped"] if use_con else rs["ewmac_flat"]
            n_speeds = len(speeds)
            ts_blend = sum(source[i] / n_speeds for i in speeds)
            ts_trend = cap_forecast(ts_blend * t_fdm)
        else:
            ts_trend = pd.Series(0.0, index=idx)

        xs_fc    = rs["xs_fc"]
        carry_fc = rs["carry_fc"] if wc > 0 else pd.Series(0.0, index=idx)
        skew_fc  = rs["skew_fc"]  if ws > 0 else pd.Series(0.0, index=idx)
        if wv > 0:
            vov_fc = rs["vov_library"].get((vovw, vdir), pd.Series(0.0, index=idx))
        else:
            vov_fc = pd.Series(0.0, index=idx)

        bundle[instrument] = dict(
            idx=idx, dates=list(idx),
            ts_trend=ts_trend,
            final_carry=carry_fc,
            final_skew=skew_fc,
            final_vov=vov_fc,
            xs_fc=xs_fc,
            vol=rs["vol"], fx=rs["fx"],
            close=rs["close"], open=rs["open"],
            pointsize=rs["pointsize"],
            cost_rt=rs["cost_rt"],   # cost_mult applied by assembler via canonical cfg
        )
    return bundle


def _abl_compute_fdm_override(bundle, cfg):
    """Pick the FDM override series / dict for a given ABL cfg.

    ABL-specific: `static_fdm` (float) forces a constant FDM; otherwise
    falls back to fdm_mode handling similar to DM.
    """
    stat_f = cfg.get("static_fdm")
    if stat_f is not None:
        any_idx = next(iter(bundle.values()))["idx"]
        return pd.Series(float(stat_f), index=any_idx)

    wt = cfg["w_trend"]; wc = cfg["w_carry"]
    ws = cfg["w_skew"];  wv = cfg["w_vov"]
    fdm_mode = cfg.get("fdm_mode", FDM_MODE_POOLED)

    if fdm_mode == FDM_MODE_POOLED:
        return None  # assembler computes canonical pooled-4 FDM

    if fdm_mode in (FDM_MODE_SHRINK, FDM_MODE_CLIP):
        insts = list(bundle.keys())
        panel_dict = OrderedDict(); w_dict = OrderedDict()
        def _tr(i):
            return cap_forecast(cfg["w_ts"] * bundle[i]["ts_trend"]
                                + cfg["w_xs"] * bundle[i]["xs_fc"])
        if wt > 0:
            panel_dict["trend"] = pd.DataFrame({i: _tr(i) for i in insts})
            w_dict["trend"] = wt
        if wc > 0:
            panel_dict["carry"] = pd.DataFrame({i: bundle[i]["final_carry"] for i in insts})
            w_dict["carry"] = wc
        if ws > 0:
            panel_dict["skew"] = pd.DataFrame({i: bundle[i]["final_skew"] for i in insts})
            w_dict["skew"] = ws
        if wv > 0:
            panel_dict["vov"] = pd.DataFrame({i: bundle[i]["final_vov"] for i in insts})
            w_dict["vov"] = wv
        if len(panel_dict) >= 2:
            return _pooled_fdm_from_panels(
                panel_dict, w_dict,
                cfg.get("fdm_corr_span", S183_CFG["fdm_corr_span"]),
                cfg.get("fdm_min_periods", S183_CFG["fdm_min_periods"]),
                cfg.get("fdm_floor", S183_CFG["fdm_floor"]),
                cfg.get("fdm_cap", S183_CFG["fdm_cap"]),
                fdm_mode)
        return pd.Series(1.0, index=bundle[insts[0]]["idx"])

    if fdm_mode == FDM_MODE_PERINST:
        out = {}
        for inst, b in bundle.items():
            active_fc = OrderedDict(); active_w = OrderedDict()
            if wt > 0:
                trend_combined = cap_forecast(cfg["w_ts"] * b["ts_trend"]
                                              + cfg["w_xs"] * b["xs_fc"])
                active_fc["trend"] = trend_combined; active_w["trend"] = wt
            if wc > 0: active_fc["carry"] = b["final_carry"]; active_w["carry"] = wc
            if ws > 0: active_fc["skew"]  = b["final_skew"];  active_w["skew"]  = ws
            if wv > 0: active_fc["vov"]   = b["final_vov"];   active_w["vov"]   = wv
            if len(active_fc) >= 2:
                out[inst] = _perinst_fdm(
                    active_fc, active_w,
                    cfg.get("fdm_corr_span", S183_CFG["fdm_corr_span"]),
                    cfg.get("fdm_min_periods", S183_CFG["fdm_min_periods"]),
                    cfg.get("fdm_floor", S183_CFG["fdm_floor"]),
                    cfg.get("fdm_cap", S183_CFG["fdm_cap"]))
            else:
                out[inst] = pd.Series(1.0, index=b["idx"])
        return out

    raise NotImplementedError(f"Unknown fdm_mode: {fdm_mode!r}")


def abl_build_inst_signals(raw_signals, cfg, variant_id=""):
    """Build inst_signals for an ABL variant via the UNIFIED canonical pipeline.

    Post-15-Apr refactor: mirrors `dm_build_variant_signals`.  The pre-flight
    coverage check + alpha-bundle projection stay here (ABL-specific: the
    pre-bake only covers ABL_VOV_WINDOWS).  The Stage B+C math is delegated
    entirely to `ig_strategy_183.assemble_signals_from_alphas`.
    """
    wv   = cfg["w_trend"] and cfg["w_vov"]  # irrelevant; kept for readability
    vovw = cfg["vov_window"]
    if cfg["w_vov"] > 0 and vovw not in ABL_VOV_WINDOWS:
        raise ValueError(
            f"[abl_build_inst_signals] variant {variant_id!r}: "
            f"vov_window={vovw} not in ABL_VOV_WINDOWS={ABL_VOV_WINDOWS}. "
            f"Add this value to ABL_VOV_WINDOWS and rerun abl_precompute, "
            f"or use dm_build_variant_signals (which bakes a superset of "
            f"VoV windows via DM_ALL_VOV_WINDOWS).")

    print(f"[ABL {variant_id or '(master)'}] projecting raw_signals -> alpha_bundle "
          f"(fdm_mode={cfg.get('fdm_mode', FDM_MODE_POOLED)}, "
          f"static_fdm={cfg.get('static_fdm')}, "
          f"smooth_span={cfg['smooth_span']}, "
          f"overlays={cfg['use_overlays']}, cost_mult={cfg['cost_mult']})")
    bundle = _abl_build_alpha_bundle(raw_signals, cfg)
    fdm_override = _abl_compute_fdm_override(bundle, cfg)

    # Translate ABL cfg to canonical cfg schema
    canonical_cfg = _s183cfg_to_canonical(cfg)

    # ABL-specific overrides not covered by _s183cfg_to_canonical defaults
    canonical_cfg["use_overlays"] = bool(cfg["use_overlays"])
    canonical_cfg["cost_mult"]    = float(cfg["cost_mult"])
    # `use_smooth=False` overrides smooth_mode to "none"
    if not cfg.get("use_smooth", True):
        canonical_cfg["smooth_mode"] = "none"

    if fdm_override is not None:
        canonical_cfg["pooled_fdm_override"] = fdm_override

    return assemble_signals_from_alphas(bundle, canonical_cfg)


def _abl_ck_path(vid):
    return ABLATION_DIR / f"S183_Ablation_{vid}_checkpoint.pkl"


def _abl_load_ck(vid):
    p = _abl_ck_path(vid)
    if p.exists():
        with open(str(p), "rb") as fh:
            return pickle.load(fh)
    return None


def run_ablation(paths, args):
    print("\n" + "=" * 78)
    print("  ABLATION MATRIX -- Strategy 183 (S180 Apr-7 16:36 frozen build)")
    print("=" * 78)
    ensure_dirs()
    mapping = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])
    raw_signals = abl_precompute(mapping, fx_daily, paths)

    abl_paths = {**paths, "output": str(ABLATION_DIR)}

    # MASTER
    master_name = "S183_Ablation_MASTER"
    print("\n[ABL] Running S183 MASTER (full replay) ...")
    if args.skip_existing and _abl_ck_path("MASTER").exists():
        print("  [SKIP] MASTER checkpoint exists.")
    else:
        try:
            master_cfg = _abl_cfg(label="MASTER")
            inst_signals = abl_build_inst_signals(raw_signals, master_cfg,
                                                    variant_id="MASTER")
            run_compounded_portfolio(inst_signals, master_name, abl_paths,
                                      save_per_inst_pnl=True)
        except Exception as exc:
            _log_failure("ABL", "MASTER", exc)

    # Variants
    keys = list(ABL_VARIANTS.keys())
    if args.variants:
        keys = [k for k in keys if k in set(args.variants)]

    n_workers = int(getattr(args, "workers", 1))
    if n_workers > 1:
        tasks = [(vid, ABL_VARIANTS[vid], args.skip_existing) for vid in keys]
        with multiprocessing.Pool(
            processes=n_workers,
            initializer=_pool_init_abl,
            initargs=(raw_signals, paths),
        ) as pool:
            for vid, status, err in tqdm(
                pool.imap_unordered(_abl_worker, tasks),
                total=len(tasks),
                desc="Ablation variants",
                unit="variant",
            ):
                if status == "fail":
                    _log_failure_raw("ABL", vid, *err)
    else:
        for vid in tqdm(keys, desc="Ablation variants", unit="variant"):
            if args.skip_existing and _abl_ck_path(vid).exists():
                continue
            try:
                cfg = ABL_VARIANTS[vid]
                inst_signals = abl_build_inst_signals(raw_signals, cfg,
                                                        variant_id=vid)
                run_compounded_portfolio(inst_signals, f"S183_Ablation_{vid}", abl_paths)
            except Exception as exc:
                _log_failure("ABL", vid, exc)


# ===========================================================================
# SECTION 5 -- ANALYSIS module
# ===========================================================================

_IRX_SERIES = None


def _load_ck_any(path: Path):
    if not path.exists():
        return None
    try:
        with open(str(path), "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        _log_failure("ANALYSIS_LOAD", str(path.name), exc)
        return None


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


def _metrics_from_ck(ck):
    dates = pd.DatetimeIndex(ck["dates"])
    dr = pd.Series(ck["daily_ret"], index=dates)
    rf = _IRX_SERIES.reindex(dates).fillna(0.0) if _IRX_SERIES is not None else None

    def _sr_mask(mask):
        seg = dr[mask]
        rf_seg = rf[mask].values if rf is not None else None
        return round(_excess_sharpe(seg.values, rf_seg), 4)

    eq = np.asarray(ck.get("equity", (1.0 + dr).cumprod().values))
    mdd = max_drawdown(eq)
    try:
        cagr = float(ck.get("cagr", np.nan))
    except Exception:
        cagr = np.nan
    try:
        sr_full_raw = float(ck.get("sr", np.nan))
    except Exception:
        sr_full_raw = np.nan
    return dict(
        n_days=len(dr),
        sr_full=_sr_mask(np.ones(len(dr), dtype=bool)),
        sr_10=_sr_mask(dates >= POST2010),
        sr_15=_sr_mask(dates >= POST2015),
        sr_20=_sr_mask(dates >= POST2020),
        cagr=round(cagr, 4) if not np.isnan(cagr) else np.nan,
        max_dd=round(mdd, 4),
        calmar=round(cagr / mdd, 4) if (not np.isnan(cagr) and mdd > 1e-9) else np.nan,
        ann_vol=round(float(dr.std(ddof=1) * np.sqrt(TRADING_DAYS)), 4)
                if len(dr) > 1 else np.nan,
        sr_full_raw=round(sr_full_raw, 4) if not np.isnan(sr_full_raw) else np.nan,
    )


def _paired_stats(ref_ck, cand_ck):
    """
    Compute JKM + LW bootstrap for a candidate vs MASTER (reference),
    on full sample AND on post-2010 / post-2015 subperiod slices.

    The subperiod paired tests are critical: the S183 failure mode
    showed that some ablations look free on the full-sample dSR
    (e.g., D_100 at -0.011) while dropping post-2010 SR by 3.6 pp
    and post-2015 SR by 5.5 pp. Full-sample significance alone is
    not sufficient to catch these "sneaks under" -- the engine
    must verify that every removal is ALSO free on the subperiods.

    LW block bootstrap is only computed on the full sample to keep
    runtime bounded; subperiod tests use JKM only (cheap).
    """
    m_dt = pd.DatetimeIndex(ref_ck["dates"])
    c_dt = pd.DatetimeIndex(cand_ck["dates"])
    common = m_dt.intersection(c_dt)
    if len(common) < 100:
        return None

    mi_full = pd.Series(ref_ck["daily_ret"], index=m_dt).reindex(common).values
    ci_full = pd.Series(cand_ck["daily_ret"], index=c_dt).reindex(common).values

    # Full-sample JKM + LW (existing path)
    jkm_full = jkm_paired_z(ci_full, mi_full)
    lw       = lw_block_bootstrap(ci_full, mi_full)

    out = dict(
        n_days=int(len(common)),
        rho=round(jkm_full["rho"], 4),
        dsr_ann=round(jkm_full["dsr_ann"], 4),
        jkm_z=round(jkm_full["z"], 4) if not np.isnan(jkm_full["z"]) else np.nan,
        jkm_p=round(jkm_full["p"], 4) if not np.isnan(jkm_full["p"]) else np.nan,
        lw_ci_lo=round(lw["ci_lo"], 4) if not np.isnan(lw["ci_lo"]) else np.nan,
        lw_ci_hi=round(lw["ci_hi"], 4) if not np.isnan(lw["ci_hi"]) else np.nan,
        lw_p_boot=round(lw["p_boot"], 4) if not np.isnan(lw["p_boot"]) else np.nan,
    )

    # Subperiod JKM paired tests (post-2010, post-2015)
    for period_tag, lo in [("10", POST2010), ("15", POST2015)]:
        mask = np.asarray(common >= lo)
        n_p = int(mask.sum())
        if n_p < 30:
            out[f"n_days_{period_tag}"] = n_p
            out[f"rho_{period_tag}"] = np.nan
            out[f"dsr_ann_{period_tag}"] = np.nan
            out[f"jkm_z_{period_tag}"] = np.nan
            out[f"jkm_p_{period_tag}"] = np.nan
            continue

        mi_p = mi_full[mask]
        ci_p = ci_full[mask]

        if mi_p.std(ddof=1) < 1e-12 or ci_p.std(ddof=1) < 1e-12:
            out[f"n_days_{period_tag}"] = n_p
            out[f"rho_{period_tag}"] = np.nan
            out[f"dsr_ann_{period_tag}"] = np.nan
            out[f"jkm_z_{period_tag}"] = np.nan
            out[f"jkm_p_{period_tag}"] = np.nan
            continue

        jkm_p_stats = jkm_paired_z(ci_p, mi_p)
        out[f"n_days_{period_tag}"] = n_p
        out[f"rho_{period_tag}"] = round(jkm_p_stats["rho"], 4)
        out[f"dsr_ann_{period_tag}"] = round(jkm_p_stats["dsr_ann"], 4)
        out[f"jkm_z_{period_tag}"] = (round(jkm_p_stats["z"], 4)
                                       if not np.isnan(jkm_p_stats["z"]) else np.nan)
        out[f"jkm_p_{period_tag}"] = (round(jkm_p_stats["p"], 4)
                                       if not np.isnan(jkm_p_stats["p"]) else np.nan)

    return out


def _collect_dm_results(master_ck):
    results = OrderedDict()
    stats = OrderedDict()
    # Anchor
    anchor_ck = _load_ck_any(_dm_ck_path("ANCHOR"))
    if anchor_ck is not None:
        results["ANCHOR"] = _metrics_from_ck(anchor_ck)
    for vid in DM_VARIANTS.keys():
        ck = _load_ck_any(_dm_ck_path(vid))
        if ck is None:
            continue
        results[vid] = _metrics_from_ck(ck)
        if master_ck is not None:
            s = _paired_stats(master_ck, ck)
            if s is not None:
                stats[vid] = s
    return results, stats


def _collect_abl_results(master_ck):
    results = OrderedDict()
    stats = OrderedDict()
    if master_ck is not None:
        results["MASTER"] = _metrics_from_ck(master_ck)
    for vid in ABL_VARIANTS.keys():
        ck = _load_ck_any(_abl_ck_path(vid))
        if ck is None:
            continue
        results[vid] = _metrics_from_ck(ck)
        if master_ck is not None:
            s = _paired_stats(master_ck, ck)
            if s is not None:
                stats[vid] = s
    return results, stats


def _rho_vs_master(master_ck, cand_ck):
    if master_ck is None or cand_ck is None:
        return np.nan
    m_dt = pd.DatetimeIndex(master_ck["dates"])
    c_dt = pd.DatetimeIndex(cand_ck["dates"])
    common = m_dt.intersection(c_dt)
    if len(common) < 100:
        return np.nan
    a = pd.Series(master_ck["daily_ret"], index=m_dt).reindex(common).values
    b = pd.Series(cand_ck["daily_ret"], index=c_dt).reindex(common).values
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def _save_summary_csv(results, stats, master_ck, path, prefix, registry):
    """
    Emit a variant summary CSV with full-sample AND subperiod metrics.

    Critical columns for catching the S183-style "sneak under" failure:
      dsr_vs_master    - full-sample dSR (existing)
      dsr_10           - post-2010 dSR  (NEW)
      dsr_15           - post-2015 dSR  (NEW)
      dsr_20           - post-2020 dSR  (NEW)

    A variant that looks free on dsr_vs_master but has dsr_10 < -0.02
    or dsr_15 < -0.02 is a load-bearing removal masquerading as
    cost-free. See GROUP 9 combination variants in ABL_VARIANTS.
    """
    rows = []
    master_m = None
    if "MASTER" in results:
        master_m = results["MASTER"]
    elif "ANCHOR" in results:
        master_m = results["ANCHOR"]
    master_sr_full = master_m.get("sr_full") if master_m else None
    master_sr_10   = master_m.get("sr_10")   if master_m else None
    master_sr_15   = master_m.get("sr_15")   if master_m else None
    master_sr_20   = master_m.get("sr_20")   if master_m else None

    for vid, m in results.items():
        if vid == "MASTER":
            label = "MASTER S183_CFG (S180 Apr-7 16:36 frozen build)"
            dim = "---"
        elif vid == "ANCHOR":
            label = "DM ANCHOR S180"
            dim = "---"
        else:
            meta = registry.get(vid, {})
            label = meta.get("label", vid)
            dim = meta.get("dim", "---")
        ck_path = (_abl_ck_path(vid) if prefix == "ablation"
                   else (_dm_ck_path(vid) if vid != "ANCHOR" else _dm_ck_path("ANCHOR")))
        cand_ck = _load_ck_any(ck_path) if vid not in ("MASTER",) else master_ck
        rho = _rho_vs_master(master_ck, cand_ck) if master_ck is not None else np.nan

        def _d(cand_v, ref_v):
            if ref_v is None or cand_v is None:
                return np.nan
            try:
                return round(float(cand_v) - float(ref_v), 4)
            except (TypeError, ValueError):
                return np.nan

        d_sr_full = _d(m.get("sr_full"), master_sr_full)
        d_sr_10   = _d(m.get("sr_10"),   master_sr_10)
        d_sr_15   = _d(m.get("sr_15"),   master_sr_15)
        d_sr_20   = _d(m.get("sr_20"),   master_sr_20)

        row = {
            "variant": vid, "dim": dim, "label": label,
            "n_days": m["n_days"],
            "sr_full": m["sr_full"], "sr_10": m["sr_10"],
            "sr_15": m["sr_15"], "sr_20": m["sr_20"],
            "cagr": m["cagr"], "max_dd": m["max_dd"],
            "calmar": m["calmar"], "ann_vol": m["ann_vol"],
            "rho_vs_master": rho,
            "dsr_vs_master": d_sr_full,
            "dsr_10": d_sr_10,
            "dsr_15": d_sr_15,
            "dsr_20": d_sr_20,
        }
        if vid in stats:
            row.update({f"stat_{k}": v for k, v in stats[vid].items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(str(path), index=False)
    print(f"[INFO] {prefix} summary -> {path}  ({len(df)} rows)")
    return df


def _save_significance_csv(stats, path, prefix):
    if not stats:
        return None
    rows = []
    for vid, s in stats.items():
        row = {"variant": vid}
        row.update(s)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(str(path), index=False)
    print(f"[INFO] {prefix} significance -> {path}  ({len(df)} rows)")
    return df


def _save_master_stability(master_ck, path):
    if master_ck is None:
        return None
    dates = pd.DatetimeIndex(master_ck["dates"])
    dr = pd.Series(master_ck["daily_ret"], index=dates)
    periods = OrderedDict([
        ("PRE-2000",  (None, POST2000)),
        ("2000-2004", (POST2000, POST2005)),
        ("2005-2009", (POST2005, POST2010)),
        ("2010-2014", (POST2010, POST2015)),
        ("2015-2019", (POST2015, POST2020)),
        ("2020-NOW",  (POST2020, None)),
        ("FULL",      (None, None)),
        ("OOS_POST2005", (POST2005, None)),
    ])
    irx = _IRX_SERIES.reindex(dates) if _IRX_SERIES is not None else None
    df = subperiod_metrics(dr, periods, irx=irx)
    df.to_csv(str(path), index=False)
    print(f"[INFO] Master stability -> {path}")
    return df


def _save_master_by_asset_class(master_ck, mapping, path):
    if master_ck is None:
        return None
    ac_col = None
    for c in ("asset_class", "AssetClass", "assetclass", "class"):
        if c in mapping.columns:
            ac_col = c
            break
    if ac_col is None:
        df = pd.DataFrame([{"asset_class": "ALL",
                             "note": "mapping has no asset_class column"}])
        df.to_csv(str(path), index=False)
        print(f"[INFO] Master per-asset-class (no class col) -> {path}")
        return df

    pnl = master_ck.get("per_inst_pnl", None)
    rows = []
    total_sr = None
    if isinstance(pnl, pd.DataFrame) and not pnl.empty:
        groups = mapping.groupby(ac_col)
        total_pnl = pnl.sum(axis=1)
        if total_pnl.std() > 1e-12:
            total_sr = (total_pnl.mean() / total_pnl.std()) * np.sqrt(TRADING_DAYS)
        for ac, sub in groups:
            cols = [c for c in sub.index if c in pnl.columns]
            if not cols:
                continue
            ac_pnl = pnl[cols].sum(axis=1)
            if ac_pnl.std() < 1e-12:
                continue
            sr = (ac_pnl.mean() / ac_pnl.std()) * np.sqrt(TRADING_DAYS)
            se_mu = newey_west_se(ac_pnl.values)
            se_sr = (se_mu / ac_pnl.std()) * np.sqrt(TRADING_DAYS)
            t = sr / se_sr if se_sr > 0 else np.nan
            b = single_sr_bootstrap_ci(ac_pnl.values)
            contrib = (sr / total_sr) if total_sr and total_sr > 0 else np.nan
            rows.append(dict(
                asset_class=str(ac), n_insts=len(cols),
                sr_ann=round(float(sr), 4),
                nw_se=round(float(se_sr), 4),
                t_stat=round(float(t), 4) if not np.isnan(t) else np.nan,
                lw_ci_lo=round(b["ci_lo"], 4) if not np.isnan(b["ci_lo"]) else np.nan,
                lw_ci_hi=round(b["ci_hi"], 4) if not np.isnan(b["ci_hi"]) else np.nan,
                contrib_to_total=round(float(contrib), 4)
                                  if not np.isnan(contrib) else np.nan,
            ))
    else:
        rows.append({"asset_class": "N/A",
                     "note": "per_inst_pnl missing from MASTER checkpoint"})
    df = pd.DataFrame(rows)
    if "sr_ann" in df.columns:
        df = df.sort_values("sr_ann", ascending=False)
    df.to_csv(str(path), index=False)
    print(f"[INFO] Master per-asset-class -> {path}")
    return df


def _save_deflated_sharpe(master_ck, dm_results, abl_results, path):
    """
    Compute Bailey-Lopez de Prado DSR:
      1. MASTER vs passive (K = # DM variants + # ABL variants + 1).
      2. Each candidate (DM and ABL) vs MASTER (same K).
    """
    if master_ck is None:
        return None
    dates = pd.DatetimeIndex(master_ck["dates"])
    m_dr = np.asarray(master_ck["daily_ret"], dtype=float)
    T = len(m_dr)
    sigma_m = m_dr.std(ddof=1) if T > 1 else np.nan
    if not (T >= 30 and sigma_m and sigma_m > 1e-12):
        return None
    sr_master_daily = float(m_dr.mean() / sigma_m)
    skew_m = float(pd.Series(m_dr).skew()) if T > 3 else 0.0
    kurt_m = float(pd.Series(m_dr).kurt() + 3.0) if T > 3 else 3.0

    K = 1 + len(dm_results) + len(abl_results)
    rows = []

    master_dsr = deflated_sharpe(sr_master_daily, K, T,
                                  skew=skew_m, kurt=kurt_m)
    def _safe(d, key, ndigits=4):
        v = d.get(key, np.nan)
        return round(float(v), ndigits) if np.isfinite(v) else np.nan

    rows.append(dict(
        target="MASTER_vs_passive",
        K=int(K), T=int(T),
        sr_daily=round(sr_master_daily, 6),
        sr_ann=round(sr_master_daily * np.sqrt(TRADING_DAYS), 4),
        sigma_sr=_safe(master_dsr, "sigma_sr", 6),
        z_obs=_safe(master_dsr, "z_obs", 4),
        e_max=_safe(master_dsr, "e_max", 6),
        e_max_sr_ann=round(_safe(master_dsr, "e_max_sr", 6) * np.sqrt(TRADING_DAYS), 4)
                     if np.isfinite(_safe(master_dsr, "e_max_sr", 6)) else np.nan,
        dsr_z=_safe(master_dsr, "dsr_z", 4),
        dsr_p=_safe(master_dsr, "dsr_p", 6),
    ))

    # Each candidate vs MASTER uses the dSR (difference in SR) as the
    # observed statistic. We treat K = total trials for multi-testing.
    def _dsr_for_cand(cand_name, cand_ck_path):
        ck = _load_ck_any(cand_ck_path)
        if ck is None:
            return
        c_dt = pd.DatetimeIndex(ck["dates"])
        common = dates.intersection(c_dt)
        if len(common) < 100:
            return
        ca = pd.Series(ck["daily_ret"], index=c_dt).reindex(common).values
        mb = pd.Series(master_ck["daily_ret"], index=dates).reindex(common).values
        diff = ca - mb
        Td = len(diff)
        sd = diff.std(ddof=1)
        if sd < 1e-12 or Td < 30:
            return
        sr_d_daily = float(diff.mean() / sd)
        sk = float(pd.Series(diff).skew()) if Td > 3 else 0.0
        ku = float(pd.Series(diff).kurt() + 3.0) if Td > 3 else 3.0
        d = deflated_sharpe(sr_d_daily, K, Td, skew=sk, kurt=ku)
        rows.append(dict(
            target=f"{cand_name}_minus_MASTER",
            K=int(K), T=int(Td),
            sr_daily=round(sr_d_daily, 6),
            sr_ann=round(sr_d_daily * np.sqrt(TRADING_DAYS), 4),
            sigma_sr=_safe(d, "sigma_sr", 6),
            z_obs=_safe(d, "z_obs", 4),
            e_max=_safe(d, "e_max", 6),
            e_max_sr_ann=round(_safe(d, "e_max_sr", 6) * np.sqrt(TRADING_DAYS), 4)
                         if np.isfinite(_safe(d, "e_max_sr", 6)) else np.nan,
            dsr_z=_safe(d, "dsr_z", 4),
            dsr_p=_safe(d, "dsr_p", 6),
        ))

    for vid in DM_VARIANTS.keys():
        _dsr_for_cand(f"DM_{vid}", _dm_ck_path(vid))
    for vid in ABL_VARIANTS.keys():
        _dsr_for_cand(f"ABL_{vid}", _abl_ck_path(vid))

    df = pd.DataFrame(rows)
    df.to_csv(str(path), index=False)
    print(f"[INFO] Deflated Sharpe table -> {path}  (K={K}, T={T})")
    return df


def _collect_aligned_pool_diffs(master_ck):
    """
    Build {variant_id: diff_array} over DM + Ablation checkpoints, aligned to
    MASTER's OOS index. Used by Romano-Wolf and Hansen-SPA over the full pool.
    """
    if master_ck is None:
        return OrderedDict()
    m_dt = pd.DatetimeIndex(master_ck["dates"])
    m_ret = pd.Series(master_ck["daily_ret"], index=m_dt).astype(float)
    diffs = OrderedDict()

    def _add(vid, ck):
        if ck is None:
            return
        c_dt = pd.DatetimeIndex(ck["dates"])
        common = m_dt.intersection(c_dt)
        if len(common) < 100:
            return
        a = pd.Series(ck["daily_ret"], index=c_dt).reindex(common).values
        b = m_ret.reindex(common).values
        diffs[vid] = (a - b).astype(float)

    # Datamining pool (skip ANCHOR which duplicates MASTER)
    for vid in DM_VARIANTS.keys():
        _add(f"DM_{vid}", _load_ck_any(_dm_ck_path(vid)))
    # Ablation pool
    for vid in ABL_VARIANTS.keys():
        _add(f"ABL_{vid}", _load_ck_any(_abl_ck_path(vid)))
    return diffs


def _save_extra_bootstrap(master_ck, out_dir):
    """
    Six supplementary block-bootstrap tests (integrated analysis stage):

      1. Politis-White (2004) optimal block length on MASTER + differentials
      2. Block-length sensitivity grid on pre-registered ablation hurdles
      3. Romano-Wolf (2005) stepdown over the full variant pool
      4. Hansen (2005) SPA test over the full variant pool
      5. One-sample circular block bootstrap CI on S172 absolute Sharpe
         (both gross and excess-of-IRX)
      6. Block-bootstrap CI on S172 maximum drawdown

    Writes:
      - extra_bootstrap_report.json  (top-level summary)
      - extra_block_length_grid.csv  (Test 2 rows)
      - extra_romano_wolf.csv        (Test 3 ranking)

    References: Politis & White (2004) Econometric Reviews; Politis & Romano
    (1994) JASA; Romano & Wolf (2005) Econometrica; Hansen (2005) JBES;
    White (2000) Econometrica; Ledoit & Wolf (2008) JEF.
    """
    if master_ck is None:
        print("  [EXTRA] skipped -- MASTER checkpoint missing")
        return
    print("\n[EXTRA] Supplementary block-bootstrap battery "
          f"(B={RT_BOOT_B}, seed={RT_SEED})")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m_dt  = pd.DatetimeIndex(master_ck["dates"])
    m_ret = pd.Series(master_ck["daily_ret"], index=m_dt).astype(float).fillna(0.0)
    # Trim to OOS window (first non-zero return) for cleanness
    nz = m_ret.loc[m_ret.abs() > 0]
    if len(nz) == 0:
        print("  [EXTRA] MASTER has no non-zero returns; skipping")
        return
    oos_start = nz.index.min()
    oos_end   = nz.index.max()
    m_ret = m_ret.loc[oos_start:oos_end]
    m_arr = m_ret.values.astype(float)
    n = len(m_arr)
    print(f"  [EXTRA] OOS window {oos_start.date()} -> {oos_end.date()}  (n={n})")

    report = {
        "meta": {
            "seed": int(RT_SEED),
            "B":    int(RT_BOOT_B),
            "n":    int(n),
            "oos_start": str(oos_start.date()),
            "oos_end":   str(oos_end.date()),
        }
    }

    # ---------- collect aligned differentials pool ----------
    pool = _collect_aligned_pool_diffs(master_ck)
    # Align each variant's diffs to the trimmed OOS window
    pool_trim = OrderedDict()
    for vid, arr in pool.items():
        cand_full = pd.Series(arr, index=pd.DatetimeIndex(master_ck["dates"])[:len(arr)])
        cand_full = cand_full.reindex(m_ret.index).fillna(0.0).values
        if len(cand_full) == len(m_arr):
            pool_trim[vid] = cand_full.astype(float)
    # Drop any duplicate-of-master (identical) series
    pool_trim = OrderedDict((k, v) for k, v in pool_trim.items()
                            if float(np.std(v)) > 1e-12)
    print(f"  [EXTRA] pool size = {len(pool_trim)} variants")
    report["meta"]["pool_size"] = len(pool_trim)

    # ---------- Test 1: Politis-White optimal block length ----------
    b_master, info_master = politis_white_optimal_block(m_arr)
    print(f"  [T1] PW optimal block: MASTER b_opt = {b_master}  "
          f"(m_hat={info_master.get('m_hat')}, n={n})")
    t1 = {"master_b_opt": int(b_master),
          "master_info":  {k: float(v) if isinstance(v, (int, float)) else v
                           for k, v in info_master.items()},
          "diff_b_opts":  {}}
    probe_keys = ("ABL_R_FDMCAP180", "ABL_R_PERFDM", "ABL_C2", "ABL_VOV0",
                  "ABL_R_JDEF", "ABL_R_S158")
    for key in probe_keys:
        if key in pool_trim:
            bi, _ = politis_white_optimal_block(pool_trim[key])
            t1["diff_b_opts"][key] = int(bi)
            print(f"  [T1] {key}: b_opt = {bi}")
    report["test1_politis_white"] = t1
    PW_BLOCK = max(1, int(b_master))

    # Per pre-registration we run all bootstrap-based tests at the
    # EARLIER suite block length (RT_BLOCK, default 21), which is
    # conservative vs the PW optimum and matches the existing JKM/LW
    # outputs. Block-length sensitivity is tested explicitly in Test 2.
    RUN_BLOCK = int(RT_BLOCK)
    print(f"  [EXTRA] primary block for RW/SPA/SR/MDD tests = {RUN_BLOCK} "
          f"(RT_BLOCK); PW optimum = {PW_BLOCK}")

    # ---------- Test 2: Block-length sensitivity grid ----------
    PRE_REG = [
        ("R_FDMCAP180", "Revert FDM_CAP 2.0 -> 1.80 (S180 defining delta)"),
        ("R_PERFDM",    "Revert pooled FDM -> per-instrument"),
        ("FDM_SHRINK",  "Pooled FDM + LW shrinkage"),
        ("VOV0",        "Drop VoV (Trinity)"),
        ("B2",          "Drop Carry"),
        ("B3",          "Drop Skew"),
        ("B4",          "Drop Trend"),
        ("C1",          "Drop EWM smoothing"),
        ("C2",          "Drop risk overlays"),
        ("C3",          "Drop smoothing + overlays"),
        ("R_JDEF",      "Revert overlays -> J_default"),
        ("R_JTIGHT",    "Revert overlays -> J_tight (S168)"),
        ("R_S158",      "Revert post-S158 refinements"),
    ]
    BLOCK_GRID = [5, 10, 21, 42, 63, 126, 252]
    print(f"  [T2] block-length grid {BLOCK_GRID} on "
          f"{len(PRE_REG)} pre-registered ablations ...")
    rows2 = []
    for code, label in PRE_REG:
        key = f"ABL_{code}"
        if key not in pool_trim:
            print(f"  [T2] skip {code}: not in pool")
            continue
        diff = pool_trim[key]
        # Reconstruct cand = master + diff to feed lw_block_bootstrap
        cand = m_arr + diff
        for blk in BLOCK_GRID:
            lw = lw_block_bootstrap(cand, m_arr, B=RT_BOOT_B,
                                    block=int(blk), seed=RT_SEED)
            rows2.append(dict(
                variant = code,
                label   = label,
                block   = int(blk),
                dsr_ann = round(float(lw["dsr_ann"]), 6),
                ci_lo   = round(float(lw["ci_lo"]), 6),
                ci_hi   = round(float(lw["ci_hi"]), 6),
                p_boot  = round(float(lw["p_boot"]), 6),
            ))
    df2 = pd.DataFrame(rows2)
    grid_path = out_dir / "extra_block_length_grid.csv"
    df2.to_csv(grid_path, index=False)
    report["test2_block_length_grid"] = {"file": grid_path.name,
                                         "rows": int(len(df2))}
    print(f"  [T2] -> {grid_path.name}  ({len(df2)} rows)")

    # ---------- Test 3: Romano-Wolf stepdown ----------
    print(f"  [T3] Romano-Wolf stepdown on K={len(pool_trim)} variants "
          f"(block={RUN_BLOCK}, B={RT_BOOT_B}) ...")
    rw_df = romano_wolf_stepdown(pool_trim, block=RUN_BLOCK,
                                 B=RT_BOOT_B, seed=RT_SEED)
    rw_path = out_dir / "extra_romano_wolf.csv"
    rw_df.to_csv(rw_path, index=False)
    n_rej = int(rw_df["rejected_at_5pct"].sum()) if len(rw_df) else 0
    report["test3_romano_wolf"] = {
        "file":         rw_path.name,
        "K":            int(len(rw_df)),
        "block":        int(RUN_BLOCK),
        "n_rejected":   n_rej,
        "best_variant": rw_df.iloc[0]["variant"] if len(rw_df) else None,
        "best_t_obs":   float(rw_df.iloc[0]["t_obs"]) if len(rw_df) else np.nan,
        "best_rw_p":    float(rw_df.iloc[0]["rw_pvalue"]) if len(rw_df) else np.nan,
    }
    print(f"  [T3] {n_rej}/{len(rw_df)} rejected at FWER=5%  "
          f"best={rw_df.iloc[0]['variant'] if len(rw_df) else 'n/a'}  "
          f"t={rw_df.iloc[0]['t_obs'] if len(rw_df) else float('nan'):.3f}  "
          f"RW_p={rw_df.iloc[0]['rw_pvalue'] if len(rw_df) else float('nan'):.4f}")

    # ---------- Test 4: Hansen SPA ----------
    print(f"  [T4] Hansen SPA test (block={RUN_BLOCK}, B={RT_BOOT_B}) ...")
    spa = hansen_spa_test(pool_trim, block=RUN_BLOCK, B=RT_BOOT_B, seed=RT_SEED)
    report["test4_hansen_spa"] = {k: (float(v) if isinstance(v, (int, float))
                                       else v) for k, v in spa.items()}
    print(f"  [T4] T_SPA = {spa['T_spa']:.3f}  "
          f"p_c={spa['p_spa_c']:.4f}  p_l={spa['p_spa_l']:.4f}  "
          f"p_u={spa['p_spa_u']:.4f}  best={spa['best_variant']}")

    # ---------- Test 5: One-sample Sharpe CI (gross + excess) ----------
    print(f"  [T5] One-sample block bootstrap SR CI (block={RUN_BLOCK}) ...")
    res5_gross = single_sr_bootstrap_ci(m_arr, B=RT_BOOT_B,
                                         block=RUN_BLOCK, seed=RT_SEED)
    if _IRX_SERIES is not None:
        irx_vals = _IRX_SERIES.reindex(m_ret.index).fillna(0.0).values
        m_excess = m_arr - irx_vals.astype(float)
    else:
        m_excess = m_arr.copy()
    res5_excess = single_sr_bootstrap_ci(m_excess, B=RT_BOOT_B,
                                          block=RUN_BLOCK, seed=RT_SEED + 10)
    report["test5_abs_sr_ci"] = {
        "gross":         {k: float(v) for k, v in res5_gross.items()},
        "excess_of_irx": {k: float(v) for k, v in res5_excess.items()},
        "note": ("gross = NAV daily return; excess = NAV daily return minus "
                 "IRX daily fraction, matching ablation_summary.csv SR."),
    }
    print(f"  [T5] GROSS  SR={res5_gross['sr']:.4f}  "
          f"CI [{res5_gross['ci_lo']:.4f}, {res5_gross['ci_hi']:.4f}]")
    print(f"  [T5] EXCESS SR={res5_excess['sr']:.4f}  "
          f"CI [{res5_excess['ci_lo']:.4f}, {res5_excess['ci_hi']:.4f}]")

    # ---------- Test 6: Block-bootstrap CI on max drawdown ----------
    print(f"  [T6] Block-bootstrap MDD CI (block={RUN_BLOCK}) ...")
    res6 = mdd_block_bootstrap_ci(m_arr, block=RUN_BLOCK,
                                   B=RT_BOOT_B, seed=RT_SEED)
    report["test6_mdd_ci"] = {k: (float(v) if isinstance(v, (int, float)) else v)
                               for k, v in res6.items()}
    print(f"  [T6] MDD = {res6['mdd_point']:.4f}  "
          f"CI [{res6['ci_lo']:.4f}, {res6['ci_hi']:.4f}]")

    # ---------- write JSON report ----------
    json_path = out_dir / "extra_bootstrap_report.json"
    with open(str(json_path), "w", encoding="ascii") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"  [EXTRA] -> {json_path.name}")


def _collect_aligned_pool_returns(master_ck):
    """
    Build a (T, K+1) matrix of daily returns aligned to MASTER's index:
    column 0 = MASTER, columns 1..K = every DM + Ablation variant that has
    a checkpoint and a non-trivial variance. Used by PBO-CSCV and MCS.
    Returns (date_index, names, matrix).
    """
    if master_ck is None:
        return None, [], np.zeros((0, 0))
    m_dt  = pd.DatetimeIndex(master_ck["dates"])
    m_ret = pd.Series(master_ck["daily_ret"], index=m_dt).astype(float).fillna(0.0)
    nz = m_ret.loc[m_ret.abs() > 0]
    if len(nz) == 0:
        return None, [], np.zeros((0, 0))
    idx = m_ret.loc[nz.index.min():nz.index.max()].index
    cols = OrderedDict()
    cols["MASTER"] = m_ret.reindex(idx).fillna(0.0).values

    def _add(vid, ck):
        if ck is None:
            return
        dt = pd.DatetimeIndex(ck["dates"])
        s  = pd.Series(ck["daily_ret"], index=dt).reindex(idx).fillna(0.0).values
        if float(np.std(s)) > 1e-12:
            cols[vid] = s

    for vid in DM_VARIANTS.keys():
        _add(f"DM_{vid}", _load_ck_any(_dm_ck_path(vid)))
    for vid in ABL_VARIANTS.keys():
        if vid == "MASTER":
            continue
        _add(f"ABL_{vid}", _load_ck_any(_abl_ck_path(vid)))
    names = list(cols.keys())
    mat = np.column_stack([cols[k] for k in names])
    return idx, names, mat


def _build_passive_equity_benchmark(master_ck, mapping):
    """
    Build a synthetic passive-long equity-futures benchmark using the equity
    instruments' returns embedded in MASTER's per_inst_pnl. We recover the
    underlying instrument daily return from the PnL normalised by its own
    rolling sigma (a common practice in CTA evaluation when raw futures
    closes are not immediately at hand). If that fails, falls back to
    scaling the sum of equity per_inst_pnl.
    """
    if master_ck is None:
        return None
    pnl = master_ck.get("per_inst_pnl", None)
    if not isinstance(pnl, pd.DataFrame) or pnl.empty:
        return None
    ac_col = None
    for c in ("asset_class", "AssetClass", "assetclass", "class"):
        if c in mapping.columns:
            ac_col = c
            break
    if ac_col is None:
        return None
    eq_inst = mapping[mapping[ac_col].astype(str).str.lower().str.startswith("equit")]
    if eq_inst.empty:
        return None
    cols = [c for c in eq_inst.index if c in pnl.columns]
    if not cols:
        # try using a name column if mapping is not keyed by instrument code
        for name_col in ("symbol", "Symbol", "code", "Code", "instrument"):
            if name_col in mapping.columns:
                codes = eq_inst[name_col].astype(str).tolist()
                cols = [c for c in codes if c in pnl.columns]
                if cols:
                    break
    if not cols:
        return None
    eq_pnl = pnl[cols].fillna(0.0)
    # Equal-vol weight each equity instrument's PnL stream.
    sds = eq_pnl.std(ddof=1).replace(0.0, np.nan)
    inv_vol = (1.0 / sds).fillna(0.0).values
    if inv_vol.sum() <= 0:
        return None
    w = inv_vol / inv_vol.sum()
    bench = (eq_pnl.values * w[None, :]).sum(axis=1)
    # Rescale to a canonical 15% annualised vol so HM/TM regressions are
    # dimensionally consistent with (r_m - rf) in return units.
    bench_sd = float(np.std(bench, ddof=1))
    if bench_sd <= 0:
        return None
    TARGET_ANN_VOL = 0.15
    target_daily_vol = TARGET_ANN_VOL / np.sqrt(TRADING_DAYS)
    bench = bench * (target_daily_vol / bench_sd)
    return pd.Series(bench, index=eq_pnl.index, name="equity_bench")


def _build_passive_multi_asset_benchmark(master_ck, mapping):
    """
    Build a synthetic passive-long multi-asset benchmark covering equity,
    bond, and commodity (Metals + Ags + OilGas) instruments from MASTER's
    per_inst_pnl.  Within each bucket, equal-vol weight. Across buckets,
    60/20/20 (equity / bond / commodity).  Rescale the combined basket to
    15% annualised vol.
    """
    if master_ck is None:
        return None
    pnl = master_ck.get("per_inst_pnl", None)
    if not isinstance(pnl, pd.DataFrame) or pnl.empty:
        return None
    ac_col = None
    for c in ("asset_class", "AssetClass", "assetclass", "class"):
        if c in mapping.columns:
            ac_col = c
            break
    if ac_col is None:
        return None

    # Map low-level asset classes to three buckets
    bucket_map = {
        "equity": "equity", "equities": "equity",
        "bond": "bond", "bonds": "bond",
        "metals": "commodity", "ags": "commodity", "oilgas": "commodity",
        "oil": "commodity", "gas": "commodity", "carbon": "commodity",
    }
    bucket_weights = {"equity": 0.60, "bond": 0.20, "commodity": 0.20}

    def _get_cols(ac_pattern):
        mask = mapping[ac_col].astype(str).str.lower().apply(
            lambda x: any(x.startswith(p) for p in ac_pattern))
        insts = mapping[mask]
        cols = [c for c in insts.index if c in pnl.columns]
        if not cols:
            for name_col in ("symbol", "Symbol", "code", "Code", "instrument"):
                if name_col in mapping.columns:
                    codes = insts[name_col].astype(str).tolist()
                    cols = [c for c in codes if c in pnl.columns]
                    if cols:
                        break
        return cols

    equity_cols = _get_cols(["equit"])
    bond_cols   = _get_cols(["bond"])
    commod_cols = _get_cols(["metal", "ags", "oilgas", "oil", "gas", "carbon"])

    if not equity_cols or not bond_cols or not commod_cols:
        return None

    def _equal_vol_basket(cols):
        sub = pnl[cols].fillna(0.0)
        sds = sub.std(ddof=1).replace(0.0, np.nan)
        inv_vol = (1.0 / sds).fillna(0.0).values
        if inv_vol.sum() <= 0:
            return None
        w = inv_vol / inv_vol.sum()
        return (sub.values * w[None, :]).sum(axis=1)

    eq_b = _equal_vol_basket(equity_cols)
    bd_b = _equal_vol_basket(bond_cols)
    cm_b = _equal_vol_basket(commod_cols)
    if eq_b is None or bd_b is None or cm_b is None:
        return None

    # Normalise each bucket to unit vol, then apply 60/20/20
    def _unit_vol(arr):
        sd = float(np.std(arr, ddof=1))
        return arr / sd if sd > 0 else arr
    eq_n = _unit_vol(eq_b) * bucket_weights["equity"]
    bd_n = _unit_vol(bd_b) * bucket_weights["bond"]
    cm_n = _unit_vol(cm_b) * bucket_weights["commodity"]
    bench = eq_n + bd_n + cm_n

    # Rescale to 15% ann vol
    bench_sd = float(np.std(bench, ddof=1))
    if bench_sd <= 0:
        return None
    target_daily_vol = 0.15 / np.sqrt(TRADING_DAYS)
    bench = bench * (target_daily_vol / bench_sd)
    return pd.Series(bench, index=pnl.index, name="multi_asset_bench")


def _load_optional_factor_csv(stem):
    """
    Look for external factor CSV (Fung-Hsieh, TSMOM, etc.) under
      Strategy_183/TestingSuite/ExtData/<stem>.csv
    Also falls back to Strategy_180/TestingSuite/ExtData/ so that the
    user does not need to duplicate the factor files already placed
    under the S180 tree.
    Format: first column = date (YYYY-MM-DD), remaining columns = factors
    in either daily or monthly frequency. Returns DataFrame or None.
    """
    candidates = [
        SUITE_DIR / "ExtData" / f"{stem}.csv",
        _HERE / "ExtData" / f"{stem}.csv",
        _BACKTEST / "ExtData" / f"{stem}.csv",
        # Fallback to Strategy_180's ExtData tree for shared factor files
        _HERE.parent / "Strategy_180" / "TestingSuite" / "ExtData" / f"{stem}.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(str(p))
                dcol = df.columns[0]
                df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
                df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()
                df = df.apply(pd.to_numeric, errors="coerce")
                return df
            except Exception as exc:
                _log_failure("ANALYSIS", f"factor_load:{stem}", exc)
                return None
    return None


def _save_thesis_battery(master_ck, mapping, paths, out_dir):
    """
    Additional thesis-grade statistical battery (tests 7..15):

      T7  Distributional sanity: Jarque-Bera, Ljung-Box(Q), Engle ARCH-LM
      T8  Lo (2002) + Mertens (2002) Sharpe SE corrections
      T9  Bailey-Lopez de Prado (2012) Minimum Track Record Length
      T10 Harvey & Liu (2015) haircut Sharpe (Bonferroni, Holm, BHY) over
          the full 118-variant trial pool
      T11 Bailey-Borwein-LdP-Zhu (2016) Probability of Backtest Overfitting
          via Combinatorially-Symmetric Cross-Validation
      T12 Hansen-Lunde-Nason (2011) Model Confidence Set over the pool
      T13 Crisis-window conditional performance (dotcom, GFC, COVID crash,
          2022 inflation shock, sovereign crisis, Volmageddon)
      T14 Henriksson-Merton (1981) + Treynor-Mazuy (1966) timing ability
          against a passive long-only equity-futures benchmark
      T15 Optional factor-model regressions (Fung-Hsieh 7F, TSMOM) if
          external factor CSVs are provided under TestingSuite/ExtData/

    Writes:
      thesis_battery_report.json        (top-level summary)
      thesis_crisis_windows.csv         (T13 table)
      thesis_factor_regressions.csv     (T15 table, if factors present)

    All tests use MASTER's excess-of-IRX returns where appropriate to match
    the canonical S180 Sharpe reported in ablation_summary.csv.
    """
    if master_ck is None:
        print("  [THESIS] skipped -- MASTER checkpoint missing")
        return
    print("\n[THESIS] Additional thesis-grade statistical battery "
          f"(B={RT_BOOT_B}, seed={RT_SEED})")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m_dt  = pd.DatetimeIndex(master_ck["dates"])
    m_ret = pd.Series(master_ck["daily_ret"], index=m_dt).astype(float).fillna(0.0)
    nz = m_ret.loc[m_ret.abs() > 0]
    if len(nz) == 0:
        print("  [THESIS] MASTER has no non-zero returns; skipping")
        return
    oos_start = nz.index.min()
    oos_end   = nz.index.max()
    m_ret = m_ret.loc[oos_start:oos_end]
    m_arr = m_ret.values.astype(float)
    n = len(m_arr)
    # Excess-of-IRX version used everywhere for "canonical" SR
    if _IRX_SERIES is not None:
        irx_daily = _IRX_SERIES.reindex(m_ret.index).fillna(0.0).values
        m_ex = (m_arr - irx_daily).astype(float)
    else:
        irx_daily = np.zeros_like(m_arr)
        m_ex = m_arr.copy()
    sr_ex_d = float(m_ex.mean() / m_ex.std(ddof=1)) if m_ex.std(ddof=1) > 0 else 0.0
    sr_ex_a = sr_ex_d * np.sqrt(TRADING_DAYS)
    print(f"  [THESIS] OOS window {oos_start.date()} -> {oos_end.date()}  "
          f"(n={n})  excess SR_ann = {sr_ex_a:.4f}")

    report = {
        "meta": {
            "seed":        int(RT_SEED),
            "B":           int(RT_BOOT_B),
            "n":           int(n),
            "oos_start":   str(oos_start.date()),
            "oos_end":     str(oos_end.date()),
            "sr_excess_ann": float(sr_ex_a),
        }
    }

    # -------------- T7  Distributional sanity --------------
    print("  [T7]  Distributional sanity (JB + Ljung-Box + ARCH-LM) ...")
    jb = jarque_bera_test(m_ex)
    lb_r  = ljung_box_test(m_ex, lags=20)
    lb_r2 = ljung_box_test(m_ex ** 2, lags=20)
    arch = arch_lm_test(m_ex, lags=5)
    report["test7_distribution"] = {
        "jarque_bera":     jb,
        "ljung_box_ret":   lb_r,
        "ljung_box_ret2":  lb_r2,
        "arch_lm":         arch,
    }
    print(f"  [T7]  JB={jb['jb']:.2f} p={jb['p']:.3g}  "
          f"skew={jb['skew']:.3f}  excess_kurt={jb['kurt_excess']:.3f}")
    print(f"  [T7]  LB(r,20)={lb_r['q']:.2f} p={lb_r['p']:.3g}   "
          f"LB(r^2,20)={lb_r2['q']:.2f} p={lb_r2['p']:.3g}   "
          f"ARCH(5)={arch['lm']:.2f} p={arch['p']:.3g}")

    # -------------- T8  Lo + Mertens Sharpe SE --------------
    print("  [T8]  Lo (2002) + Mertens (2002) Sharpe SE ...")
    lo = lo_sharpe_se(m_ex, q=TRADING_DAYS)
    me = mertens_sharpe_se(m_ex)
    report["test8_sharpe_inference"] = {
        "lo_2002":       lo,
        "mertens_2002":  me,
    }
    print(f"  [T8]  Lo     SR_iid={lo['sr_iid']:.4f}  SR_Lo={lo['sr_lo']:.4f}  "
          f"eta={lo['eta']:.4f}  se={lo['se_lo']:.4f}")
    print(f"  [T8]  Mertens SR_ann={me['sr_ann']:.4f}  "
          f"SE={me['se_ann']:.4f}  t={me.get('t_stat', float('nan')):.3f}  "
          f"(skew={me['skew']:.3f}, kurt={me['kurt']:.3f})")

    # -------------- T9  Minimum Track Record Length --------------
    print("  [T9]  Minimum Track Record Length (Bailey-LdP 2012) ...")
    mtr_rows = []
    sr_d = sr_ex_d
    for sr_bench in (0.0, 0.25, 0.50, 0.75, 1.00):
        out_m = min_track_record_length(sr_ex_a, sr_bench, n,
                                         skew=me.get("skew", 0.0),
                                         kurt=me.get("kurt", 3.0),
                                         alpha=0.05)
        mtr_rows.append({"sr_bench_ann": sr_bench, **out_m})
    report["test9_min_trl"] = mtr_rows
    for row in mtr_rows:
        mtrl = row["mintrl"] if np.isfinite(row["mintrl"]) else float('inf')
        mtrly = row.get("mintrl_years", float('nan'))
        print(f"  [T9]  vs SR*={row['sr_bench_ann']:.2f}: "
              f"MinTRL={mtrl:.0f} days ({mtrly:.2f}y)  "
              f"n={row['current_n']}  feasible={row['feasible']}")

    # -------------- T10  Harvey-Liu haircut --------------
    N_trials = 118  # full DM + ablation pool
    t_stat_daily = sr_ex_d * np.sqrt(n)
    print(f"  [T10] Harvey-Liu haircut (N={N_trials} trials, t={t_stat_daily:.3f}) ...")
    hl = harvey_liu_haircut(sr_ex_a, t_stat_daily, N_trials)
    report["test10_harvey_liu"] = hl
    print(f"  [T10] orig SR={hl['sr_orig']:.4f}   "
          f"Bonf={hl['sr_bonf']:.4f} (cut {hl['haircut_bonf']*100:.1f}%)  "
          f"Holm={hl['sr_holm']:.4f}  BHY={hl['sr_bhy']:.4f} "
          f"(cut {hl['haircut_bhy']*100:.1f}%)")

    # -------------- T11  PBO via CSCV --------------
    # NOTE: CSCV/PBO is reported for completeness but should NOT be used as
    # evidence for or against overfitting.  Empirically, the more datamined
    # S172 (FDM_CAP=1.80 from sweep, sigmoid=10.0 fitted) scored PBO=6.7%,
    # while S180 (all parameters structurally derived, zero fitted values)
    # scores PBO~29%.  The test rewards low equity-curve variance — which
    # S172's permanent sigmoid drag produced — not parameter parsimony.
    # When all DM variants sit on a flat SR plateau, the IS "champion" is
    # effectively random across CSCV folds, mechanically inflating PBO.
    # Harvey-Liu BHY (T10) and the parameter perturbation grid (robustness
    # T3) are the correct anti-overfit instruments for this strategy.
    print("  [T11] PBO via CSCV (Bailey et al. 2016) ...")
    print("  [T11] WARNING: PBO is reported but deprecated for S180 --")
    print("         S172 (datamined) scored PBO=6.7% vs S180 (structural) ~29%.")
    print("         The test measures equity-curve smoothness, not overfitting.")
    _, pool_names, pool_mat = _collect_aligned_pool_returns(master_ck)
    if pool_mat.shape[1] >= 4:
        pbo = pbo_cscv(pool_mat, S=16)
        report["test11_pbo_cscv"] = pbo
        report["test11_pbo_cscv"]["pool_names_count"] = int(pool_mat.shape[1])
        report["test11_pbo_cscv"]["deprecated"] = True
        report["test11_pbo_cscv"]["deprecation_note"] = (
            "CSCV/PBO rewards equity-curve smoothness, not parameter parsimony. "
            "S172 (datamined) scored PBO=6.7% vs S180 (structural) ~29%. "
            "Use Harvey-Liu BHY (T10) and robustness T3 grid instead."
        )
        pbo_val = pbo.get('pbo', float('nan'))
        med_log = pbo.get('median_logit', float('nan'))
        n_spl = pbo.get('n_splits', 0)
        print(f"  [T11] PBO={pbo_val:.4f}   "
              f"median_logit={med_log:.3f}   "
              f"splits={n_spl}   "
              f"pool N={pool_mat.shape[1]}")
    else:
        report["test11_pbo_cscv"] = {"note": "insufficient pool"}
        print("  [T11] skipped -- insufficient pool")

    # -------------- T12  Model Confidence Set --------------
    print(f"  [T12] Model Confidence Set (HLN 2011, block={RT_BLOCK}, B={RT_BOOT_B}) ...")
    if pool_mat.shape[1] >= 2:
        # Loss = -daily return  (higher SR -> lower loss, so MASTER should survive)
        loss = -pool_mat
        mcs = model_confidence_set(loss, block=RT_BLOCK, B=RT_BOOT_B,
                                    alpha=0.10, seed=RT_SEED)
        kept_idx = mcs.get("kept_idx", [])
        kept_names = [pool_names[i] for i in kept_idx]
        master_in = ("MASTER" in kept_names)
        rank_master = kept_names.index("MASTER") + 1 if master_in else None
        report["test12_mcs"] = {
            "alpha":         mcs.get("alpha"),
            "block":         mcs.get("block"),
            "B":             mcs.get("B"),
            "T":             mcs.get("T"),
            "N":             mcs.get("N"),
            "kept_count":    len(kept_names),
            "kept":          kept_names[:25],   # cap for JSON readability
            "master_kept":   bool(master_in),
            "n_eliminated":  len(mcs.get("eliminated", [])),
            "p_values":      mcs.get("p_values", []),
        }
        print(f"  [T12] MCS_{int((1-mcs.get('alpha',0.1))*100)}% has "
              f"{len(kept_names)} / {pool_mat.shape[1]} models   "
              f"MASTER kept = {master_in}")
    else:
        report["test12_mcs"] = {"note": "insufficient pool"}
        print("  [T12] skipped -- insufficient pool")

    # -------------- T13  Crisis-window conditional performance --------------
    print("  [T13] Crisis-window conditional Sharpes ...")
    crises = [
        ("Gulf_War_1990",        "1990-07-01", "1991-03-31"),
        ("LTCM_1998",            "1998-07-01", "1998-12-31"),
        ("Dotcom_bust",          "2000-03-10", "2002-10-09"),
        ("GFC",                  "2007-10-09", "2009-03-09"),
        ("Euro_sov_2011",        "2011-05-01", "2011-12-31"),
        ("Volmageddon_2018",     "2018-01-26", "2018-04-02"),
        ("Q4_2018_shock",        "2018-10-01", "2018-12-24"),
        ("COVID_crash",          "2020-02-19", "2020-03-23"),
        ("COVID_recovery",       "2020-03-24", "2020-12-31"),
        ("Inflation_shock_2022", "2022-01-03", "2022-10-14"),
        ("AI_rally_2023",        "2023-01-01", "2023-12-31"),
        ("Full_2024",            "2024-01-01", "2024-12-31"),
    ]
    def _window_metrics(s):
        if len(s) < 5:
            return dict(n=len(s), sr_ann=np.nan, cagr=np.nan,
                        max_dd=np.nan, mean_bp=np.nan, worst_day=np.nan,
                        best_day=np.nan)
        sd = float(s.std(ddof=1))
        sr = (float(s.mean()) / sd) * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
        eq = (1.0 + s).cumprod()
        ny = len(s) / TRADING_DAYS
        cagr = float(eq.iloc[-1] ** (1.0 / ny) - 1.0) if ny > 0 and eq.iloc[0] > 0 else np.nan
        mdd = max_drawdown(eq.values)
        return dict(n=int(len(s)),
                    sr_ann=round(sr, 4) if np.isfinite(sr) else np.nan,
                    cagr=round(cagr, 4) if np.isfinite(cagr) else np.nan,
                    max_dd=round(float(mdd), 4),
                    mean_bp=round(float(s.mean() * 1e4), 3),
                    worst_day=round(float(s.min()), 5),
                    best_day=round(float(s.max()), 5))
    crows = []
    m_ex_s = pd.Series(m_ex, index=m_ret.index)
    for lbl, d0, d1 in crises:
        seg = m_ex_s.loc[d0:d1]
        row = {"window": lbl, "start": d0, "end": d1}
        row.update(_window_metrics(seg))
        crows.append(row)
    crisis_df = pd.DataFrame(crows)
    crisis_path = out_dir / "thesis_crisis_windows.csv"
    crisis_df.to_csv(crisis_path, index=False)
    report["test13_crisis_windows"] = {"file": crisis_path.name,
                                         "rows": int(len(crisis_df))}
    print(f"  [T13] -> {crisis_path.name}  ({len(crisis_df)} windows)")
    for _, row in crisis_df.iterrows():
        sr_v = row.get("sr_ann")
        try:
            sr_str = f"{float(sr_v):+.3f}" if pd.notna(sr_v) else "n/a"
        except Exception:
            sr_str = "n/a"
        print(f"  [T13]   {row['window']:<22} {row['start']} .. "
              f"{row['end']}  n={row['n']:4d}  SR={sr_str}")

    # -------------- T14  Henriksson-Merton + Treynor-Mazuy --------------
    print("  [T14] Henriksson-Merton + Treynor-Mazuy timing tests ...")
    bench = None
    try:
        bench = _build_passive_equity_benchmark(master_ck, mapping)
    except Exception as exc:
        _log_failure("ANALYSIS", "equity_bench", exc)
        bench = None
    if bench is not None and len(bench) >= 500:
        b_al = bench.reindex(m_ret.index).fillna(0.0).values.astype(float)
        hm = henriksson_merton(m_ex, b_al, rf=None)
        tm = treynor_mazuy  (m_ex, b_al, rf=None)
        report["test14_timing"] = {
            "benchmark":      "passive_equity_futures_equalvol",
            "benchmark_n":    int(len(b_al)),
            "benchmark_sr":   float((b_al.mean() / b_al.std(ddof=1))
                                      * np.sqrt(TRADING_DAYS))
                              if b_al.std(ddof=1) > 0 else float('nan'),
            "henriksson_merton": hm,
            "treynor_mazuy":     tm,
        }
        print(f"  [T14] HM  alpha_ann={hm.get('alpha_ann', float('nan')):.4f}  "
              f"t={hm.get('alpha_t', float('nan')):.2f}  "
              f"gamma={hm.get('gamma', float('nan')):.3f}  "
              f"t_gamma={hm.get('gamma_t', float('nan')):.2f}  "
              f"R2={hm.get('r2', float('nan')):.3f}")
        print(f"  [T14] TM  alpha_ann={tm.get('alpha_ann', float('nan')):.4f}  "
              f"t={tm.get('alpha_t', float('nan')):.2f}  "
              f"gamma={tm.get('gamma', float('nan')):.3g}  "
              f"t_gamma={tm.get('gamma_t', float('nan')):.2f}  "
              f"R2={tm.get('r2', float('nan')):.3f}")
    else:
        report["test14_timing"] = {"note": "no equity benchmark available"}
        print("  [T14] skipped -- no equity benchmark")

    # -------------- T15  Fung-Hsieh 7F + TSMOM factor regressions --------------
    print("  [T15] Optional factor regressions (Fung-Hsieh 7F, TSMOM) ...")
    fr_rows = []
    for stem, label in (("fh_factors",   "Fung-Hsieh 7F"),
                         ("tsmom_factor", "MOP TSMOM"),
                         ("ff5_factors",  "Fama-French 5F")):
        fac = _load_optional_factor_csv(stem)
        if fac is None or fac.empty:
            print(f"  [T15] {label}: no CSV at TestingSuite/ExtData/{stem}.csv -- skipped")
            continue
        # Align: if monthly, resample MASTER to monthly sums
        is_daily = (fac.index.to_series().diff().dropna().median()
                     <= pd.Timedelta(days=3))
        if is_daily:
            y_ser = pd.Series(m_ex, index=m_ret.index)
            fac_al = fac.reindex(y_ser.index).dropna(how="all")
            common = y_ser.index.intersection(fac_al.index)
            y_ser = y_ser.reindex(common)
            fac_al = fac_al.reindex(common).fillna(0.0)
            freq = "daily"
        else:
            m_m = pd.Series(m_ex, index=m_ret.index).resample("ME").sum()
            fac_al = fac.resample("ME").last() if fac.index.freq is None else fac
            common = m_m.index.intersection(fac_al.index)
            y_ser = m_m.reindex(common)
            fac_al = fac_al.reindex(common).fillna(0.0)
            freq = "monthly"
        if len(y_ser) < 30:
            print(f"  [T15] {label}: too few aligned obs ({len(y_ser)}) -- skipped")
            continue
        ppy = float(TRADING_DAYS) if freq == "daily" else 12.0
        fr = factor_regression_hac(y_ser, fac_al, rf=None,
                                     periods_per_year=ppy)
        fr_rows.append(dict(model=label, stem=stem, freq=freq,
                             n=fr.get("n"),
                             alpha_ann=round(fr.get("alpha_ann", np.nan), 4),
                             alpha_t=round(fr.get("alpha_t", np.nan), 3),
                             alpha_p=round(fr.get("alpha_p", np.nan), 4),
                             r2=round(fr.get("r2", np.nan), 4),
                             loadings=";".join(f"{k}={v['beta']:+.3f}(t={v['t']:.2f})"
                                                 for k, v in fr.get("factors", {}).items())))
        print(f"  [T15] {label:<16} ({freq}, n={fr.get('n')})  "
              f"alpha_ann={fr.get('alpha_ann', float('nan')):+.4f}  "
              f"t={fr.get('alpha_t', float('nan')):.2f}  "
              f"R^2={fr.get('r2', float('nan')):.3f}")
    if fr_rows:
        fr_df = pd.DataFrame(fr_rows)
        fr_path = out_dir / "thesis_factor_regressions.csv"
        fr_df.to_csv(fr_path, index=False)
        report["test15_factor_regressions"] = {"file": fr_path.name,
                                                 "rows": int(len(fr_df))}
        print(f"  [T15] -> {fr_path.name}  ({len(fr_df)} models)")
    else:
        report["test15_factor_regressions"] = {
            "note": ("No external factor CSVs found. Drop Fung-Hsieh factors "
                     "(fh_factors.csv) or AQR TSMOM (tsmom_factor.csv) into "
                     "Strategy_183/TestingSuite/ExtData/ and re-run to enable.")
        }

    # -------------- T16  Temporal Holdout (post-2015) --------------------------
    print("  [T16] Temporal holdout: post-2015 performance ...")
    holdout_start = pd.Timestamp("2015-01-01")
    mask_ho = pd.DatetimeIndex(master_ck["dates"]) >= holdout_start
    if mask_ho.sum() > 252:
        ho_ret = master_ck["daily_ret"][mask_ho]
        ho_exc = ho_ret  # already excess in checkpoint context
        ho_sr = float(ho_ret.mean() / ho_ret.std() * np.sqrt(256)) if ho_ret.std() > 0 else 0.0
        ho_cum = np.cumprod(1 + ho_ret)
        ho_peak = np.maximum.accumulate(ho_cum)
        ho_mdd = float(((ho_peak - ho_cum) / ho_peak).max())
        ho_n = int(mask_ho.sum())
        ho_years = ho_n / 256
        ho_cagr = float((ho_cum[-1]) ** (1.0 / ho_years) - 1.0) if ho_years > 0 else 0.0
        report["test16_temporal_holdout"] = {
            "start": "2015-01-01", "n_days": ho_n, "years": round(ho_years, 1),
            "sr": round(ho_sr, 4), "cagr": round(ho_cagr, 4),
            "max_dd": round(ho_mdd, 4),
        }
        print(f"  [T16] Post-2015: SR={ho_sr:.4f}  CAGR={ho_cagr:.2%}  "
              f"MDD={ho_mdd:.2%}  ({ho_years:.1f}yr, n={ho_n})")
    else:
        report["test16_temporal_holdout"] = {"note": "insufficient post-2015 data"}
        print("  [T16] skipped -- insufficient post-2015 data")

    # -------------- T17  Signal Shuffle (null-distribution) ------------------
    # The correct implementation (per-instrument forecast permutation with
    # full pipeline re-simulation) is run by the standalone script
    #   Strategy_183/t17_signal_shuffle.py
    # because it requires the DM library, which is not in scope here.
    # We pick up its cached JSON result if present.
    t17_json = ANALYSIS_DIR / "test17_signal_shuffle.json"
    if t17_json.exists():
        with open(t17_json, "r", encoding="utf-8") as fh:
            t17_res = json.load(fh)
        # New dual-null schema: {real_sr, permutation:{...}, whitenoise:{...}}
        if "permutation" in t17_res or "whitenoise" in t17_res:
            slim = {"real_sr": t17_res.get("real_sr")}
            for k in ("permutation", "whitenoise"):
                if k in t17_res:
                    slim[k] = {kk: vv for kk, vv in t17_res[k].items()
                               if kk != "shuffle_srs"}
            report["test17_signal_shuffle"] = slim
            for k in ("permutation", "whitenoise"):
                if k in t17_res:
                    d = t17_res[k]
                    print(f"  [T17 {k}] real_sr={d.get('real_sr'):+.4f}  "
                          f"shuffle_mean={d.get('shuffle_mean'):+.4f}  "
                          f"pct>=real={d.get('pct_ge_real'):.4f}  "
                          f"z={d.get('n_sigma_above_null')}")
        else:
            # Legacy single-null schema
            report["test17_signal_shuffle"] = {
                k: v for k, v in t17_res.items() if k != "shuffle_srs"
            }
            print(f"  [T17] Loaded cached result: real_sr="
                  f"{t17_res.get('real_sr'):+.4f}  "
                  f"shuffle_mean={t17_res.get('shuffle_mean'):+.4f}  "
                  f"pct>=real={t17_res.get('pct_ge_real'):.4f}  "
                  f"z={t17_res.get('n_sigma_above_null')}")
    else:
        report["test17_signal_shuffle"] = {
            "note": ("Not run. Execute `py t17_signal_shuffle.py` in "
                     "Strategy_183/ to generate "
                     "Analysis/test17_signal_shuffle.json, then re-run "
                     "analysis."),
        }
        print("  [T17] skipped -- run t17_signal_shuffle.py to generate "
              "Analysis/test17_signal_shuffle.json")

    # -------------- T18  Break-Even Cost Multiplier --------------------------
    print("  [T18] Break-even cost multiplier ...")
    # Collect cost-variant SRs from ablation results if available
    _master_sr_fallback = float(master_ck.get("sr", 0.0))
    cost_points = [(1.0, master_ck.get("sr", _master_sr_fallback))]
    abl_results_path = out_dir / "ablation_summary.csv"
    if abl_results_path.exists():
        abl_df = pd.read_csv(str(abl_results_path))
        for mult_label, mult_val in [("G", 0.0), ("G_150", 1.5), ("G_200", 2.0), ("G_300", 3.0)]:
            row = abl_df[abl_df["variant"] == mult_label]
            if len(row):
                cost_points.append((mult_val, float(row.iloc[0]["sr_full"])))
        cost_points.sort(key=lambda x: x[0])
        if len(cost_points) >= 2:
            x_pts = np.array([p[0] for p in cost_points])
            y_pts = np.array([p[1] for p in cost_points])
            # Linear interpolation to find SR=0 crossing
            for i in range(len(y_pts) - 1):
                if y_pts[i] > 0 >= y_pts[i + 1]:
                    breakeven = x_pts[i] + (0 - y_pts[i]) * (x_pts[i+1] - x_pts[i]) / (y_pts[i+1] - y_pts[i])
                    report["test18_breakeven_cost"] = {
                        "breakeven_multiplier": round(float(breakeven), 2),
                        "cost_points": [(float(x), round(float(y), 4)) for x, y in zip(x_pts, y_pts)],
                    }
                    print(f"  [T18] Break-even cost multiplier: {breakeven:.2f}x")
                    break
            else:
                # SR never crosses zero in range
                report["test18_breakeven_cost"] = {
                    "breakeven_multiplier": f">{x_pts[-1]:.1f}x (SR still positive)",
                    "cost_points": [(float(x), round(float(y), 4)) for x, y in zip(x_pts, y_pts)],
                }
                print(f"  [T18] SR still positive at {x_pts[-1]:.0f}x costs — breakeven > {x_pts[-1]:.0f}x")
    else:
        report["test18_breakeven_cost"] = {"note": "ablation_summary.csv not found; run ablation first"}
        print("  [T18] skipped -- run ablation first")

    # -------------- Master daily-return array (shared by T19..T22) ----------
    # Previously defined in the pre-refactor T17 block; re-established here
    # as a named variable used by the rolling-SR, look-ahead and vol-scaled
    # cost tests below.
    master_ret = np.asarray(master_ck["daily_ret"], dtype=float)

    # -------------- T19  Rolling 5-Year SR -----------------------------------
    print("  [T19] Rolling 5-year SR stability ...")
    roll_window = 1260  # ~5 years of trading days
    master_ret_s = pd.Series(master_ck["daily_ret"],
                              index=pd.DatetimeIndex(master_ck["dates"]))
    if len(master_ret_s) > roll_window:
        roll_mean = master_ret_s.rolling(roll_window).mean()
        roll_std = master_ret_s.rolling(roll_window).std()
        roll_sr = (roll_mean / roll_std * np.sqrt(256)).dropna()
        sr_trend_slope = float(np.polyfit(np.arange(len(roll_sr)), roll_sr.values, 1)[0]) * 256
        report["test19_rolling_sr"] = {
            "window_days": roll_window,
            "min_sr": round(float(roll_sr.min()), 4),
            "max_sr": round(float(roll_sr.max()), 4),
            "median_sr": round(float(roll_sr.median()), 4),
            "current_sr": round(float(roll_sr.iloc[-1]), 4),
            "trend_slope_per_year": round(sr_trend_slope, 6),
            "min_date": str(roll_sr.idxmin().date()),
            "max_date": str(roll_sr.idxmax().date()),
        }
        print(f"  [T19] 5yr rolling SR: min={roll_sr.min():.3f} ({roll_sr.idxmin().date()})  "
              f"max={roll_sr.max():.3f} ({roll_sr.idxmax().date()})  "
              f"current={roll_sr.iloc[-1]:.3f}  trend={sr_trend_slope:+.4f}/yr")
    else:
        report["test19_rolling_sr"] = {"note": "insufficient data for 5-year window"}
        print("  [T19] skipped -- insufficient data")

    # -------------- T20  CTA Benchmark Regression ----------------------------
    print("  [T20] CTA benchmark regression ...")
    cta_csv = out_dir.parent / "ExtData" / "cta_benchmark.csv"
    if cta_csv.exists():
        cta_df = pd.read_csv(str(cta_csv), parse_dates=["Date"], index_col="Date")
        cta_col = [c for c in cta_df.columns if c.lower() not in ("date",)][0]
        cta_ret = cta_df[cta_col].dropna()
        # Align with master returns
        y_ser = master_ret_s.reindex(cta_ret.index).dropna()
        x_ser = cta_ret.reindex(y_ser.index).dropna()
        y_ser = y_ser.reindex(x_ser.index)
        if len(y_ser) > 60:
            x_arr = x_ser.values
            y_arr = y_ser.values
            X_ols = np.column_stack([np.ones_like(x_arr), x_arr])
            beta_hat = np.linalg.lstsq(X_ols, y_arr, rcond=None)[0]
            resid = y_arr - X_ols @ beta_hat
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            # Annualise alpha depending on frequency (daily vs monthly)
            obs_per_year = 256 if len(y_ser) > 500 else 12
            alpha_ann = float(beta_hat[0]) * obs_per_year
            se_alpha = float(np.sqrt(ss_res / (len(y_arr) - 2) / np.sum((X_ols[:, 1] - X_ols[:, 1].mean())**2 + 1e-30)))
            t_alpha = float(beta_hat[0]) / se_alpha if se_alpha > 0 else 0.0
            report["test20_cta_benchmark"] = {
                "benchmark": cta_col, "n_obs": len(y_ser), "freq": "daily" if obs_per_year > 100 else "monthly",
                "alpha_ann": round(alpha_ann, 4), "t_alpha": round(t_alpha, 2),
                "beta": round(float(beta_hat[1]), 4), "r2": round(r2, 4),
            }
            print(f"  [T20] CTA benchmark ({cta_col}, n={len(y_ser)}):  "
                  f"alpha={alpha_ann:.2%} (t={t_alpha:.2f})  beta={beta_hat[1]:.3f}  R2={r2:.3f}")
        else:
            report["test20_cta_benchmark"] = {"note": f"too few aligned obs ({len(y_ser)})"}
            print(f"  [T20] skipped -- too few aligned obs ({len(y_ser)})")
    else:
        report["test20_cta_benchmark"] = {
            "note": "No cta_benchmark.csv in ExtData/. Download SG CTA or BTOP50 index returns."
        }
        print("  [T20] skipped -- no cta_benchmark.csv in ExtData/")

    # -------------- T21  Look-Ahead Bias Audit (+1 lag) ----------------------
    print("  [T21] Look-ahead bias audit (forecast lag +1) ...")
    # Compare SR of original vs lagged forecasts from the checkpoint
    # This is a structural check: if lagging all forecasts by 1 day doesn't
    # meaningfully change SR, there's no look-ahead leakage.
    # We approximate by lagging daily returns by +1 and comparing the
    # auto-correlation structure.
    if len(master_ret) > 500:
        orig_sr = float(master_ret.mean() / master_ret.std() * np.sqrt(256))
        # The actual lag test requires re-running with shifted forecasts.
        # As a proxy, test whether day-t return predicts day-(t+1) return,
        # which would indicate look-ahead in the forecast->position pipeline.
        ret_lag1 = np.correlate(master_ret[:-1] - master_ret[:-1].mean(),
                                 master_ret[1:] - master_ret[1:].mean(),
                                 mode='valid')[0]
        ret_lag1 /= (len(master_ret) - 1) * master_ret.std() ** 2
        # Also check if forecast-return correlation is abnormally high at lag 0 vs lag 1
        report["test21_lookahead_audit"] = {
            "ret_autocorr_lag1": round(float(ret_lag1), 6),
            "note": ("Return autocorrelation at lag-1. Values near 0 indicate no "
                     "look-ahead. Values > 0.05 would be suspicious. For a full "
                     "audit, re-run the strategy with all forecasts shifted +1 day."),
        }
        print(f"  [T21] Return autocorrelation lag-1: {ret_lag1:.6f}  "
              f"({'OK — no look-ahead signal' if abs(ret_lag1) < 0.05 else 'WARNING — investigate'})")
    else:
        report["test21_lookahead_audit"] = {"note": "insufficient data"}
        print("  [T21] skipped -- insufficient data")

    # -------------- T22  Vol-Scaled Transaction Costs ----------------------------
    print("  [T22] Vol-scaled transaction costs (adversarial, x10 cap) ...")
    if len(master_ret) > 256:
        # Compute daily realised vol ratio: rolling 21d std annualised / VOL_TARGET
        vol_target = 0.20
        realised_vol = pd.Series(master_ret).rolling(21, min_periods=10).std() * np.sqrt(256)
        vol_ratio = (realised_vol / vol_target).fillna(1.0).values

        # Cost multiplier: floor at 1.0, linear scale with vol_ratio, cap at 10.0
        cost_mult = np.clip(vol_ratio, 1.0, 10.0)

        # Estimate daily cost drag from checkpoint
        total_comm = master_ck.get("total_comm", 0.0)
        n_days = len(master_ret)
        base_daily_cost = total_comm / n_days if n_days > 0 else 0.0

        # Original daily cost drag as fraction of equity
        equity = master_ck.get("equity", None)
        if equity is not None and len(equity) == n_days:
            # Base cost drag per day as return impact
            base_cost_ret = base_daily_cost / equity
            base_cost_ret = np.where(np.isfinite(base_cost_ret), base_cost_ret, 0.0)

            # Vol-scaled additional cost: (multiplier - 1) * base_cost
            # The base cost is already in daily_ret. We only ADD the extra.
            extra_cost_ret = (cost_mult - 1.0) * base_cost_ret
            adjusted_ret = master_ret - extra_cost_ret

            # Compute adjusted SR
            adj_mean = np.mean(adjusted_ret)
            adj_std = np.std(adjusted_ret)
            adj_sr = (adj_mean / adj_std * np.sqrt(256)) if adj_std > 0 else 0.0
            orig_sr = float(np.mean(master_ret) / np.std(master_ret) * np.sqrt(256))
            sr_decay = (adj_sr - orig_sr) / orig_sr * 100 if orig_sr != 0 else 0.0

            # Stats on the multiplier
            pct_days_above_1 = float(np.mean(cost_mult > 1.0) * 100)
            pct_days_above_3 = float(np.mean(cost_mult > 3.0) * 100)
            pct_days_above_5 = float(np.mean(cost_mult > 5.0) * 100)
            mean_mult = float(np.mean(cost_mult))
            max_mult = float(np.max(cost_mult))

            report["test22_vol_scaled_costs"] = {
                "description": "Transaction costs scaled by realised_vol/vol_target, floor 1.0x, cap 10.0x",
                "vol_target": vol_target,
                "cost_floor": 1.0,
                "cost_cap": 10.0,
                "mean_multiplier": round(mean_mult, 3),
                "max_multiplier": round(max_mult, 3),
                "pct_days_above_1x": round(pct_days_above_1, 1),
                "pct_days_above_3x": round(pct_days_above_3, 1),
                "pct_days_above_5x": round(pct_days_above_5, 1),
                "sr_original": round(orig_sr, 4),
                "sr_vol_scaled_costs": round(adj_sr, 4),
                "sr_decay_pct": round(sr_decay, 2),
            }
            print(f"  [T22] Vol-scaled costs: mean mult={mean_mult:.2f}x, max={max_mult:.2f}x")
            print(f"  [T22] Days above 1x: {pct_days_above_1:.1f}% | >3x: {pct_days_above_3:.1f}% | >5x: {pct_days_above_5:.1f}%")
            print(f"  [T22] SR: {orig_sr:.4f} -> {adj_sr:.4f} ({sr_decay:+.2f}% decay)")
        else:
            report["test22_vol_scaled_costs"] = {"note": "equity array not available in checkpoint"}
            print("  [T22] skipped -- equity array not in checkpoint")
    else:
        report["test22_vol_scaled_costs"] = {"note": "insufficient data"}
        print("  [T22] skipped -- insufficient data")

    # -------------- write JSON report --------------
    json_path = out_dir / "thesis_battery_report.json"
    with open(str(json_path), "w", encoding="ascii") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"  [THESIS] -> {json_path.name}")


def _plot_stability_scatter(dm_df, path):
    if dm_df is None or dm_df.empty:
        return
    if "sr_10" not in dm_df.columns or "sr_20" not in dm_df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 9))

    def _axis_letter(vid):
        if vid == "ANCHOR":
            return "*"
        return vid[0] if vid and vid[0].isalpha() else "?"

    letters = sorted({_axis_letter(v) for v in dm_df["variant"].tolist()})
    cmap = plt.get_cmap("tab20", max(len(letters), 1))
    color_map = {lt: cmap(i) for i, lt in enumerate(letters)}

    for lt in letters:
        sub = dm_df[dm_df["variant"].apply(_axis_letter) == lt]
        ax.scatter(sub["sr_10"], sub["sr_20"],
                    c=[color_map[lt]], label=f"Axis {lt}",
                    s=40, alpha=0.8, edgecolors="k", linewidths=0.3)
    anchor = dm_df[dm_df["variant"] == "ANCHOR"]
    if len(anchor):
        ax.scatter(anchor["sr_10"], anchor["sr_20"], c="red", marker="*",
                    s=350, edgecolors="k", linewidths=0.8,
                    label="MASTER S180", zorder=10)

    lim_lo = float(min(dm_df["sr_10"].min(), dm_df["sr_20"].min())) - 0.05
    lim_hi = float(max(dm_df["sr_10"].max(), dm_df["sr_20"].max())) + 0.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
             color="gray", linestyle=":", alpha=0.5)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("SR 2010+ (training-adjacent)")
    ax.set_ylabel("SR 2020+ (most-recent OOS)")
    ax.set_title("S180 DM Stability Scatter -- SR_10 vs SR_20")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"[INFO] Stability scatter -> {path}")


def _plot_heatmaps(dm_df, path):
    if dm_df is None or dm_df.empty:
        return
    dims = [d for d in dm_df["dim"].unique() if d and d != "---"]
    if not dims:
        return
    n = len(dims)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8 * n), tight_layout=True)
    if n == 1:
        axes = [axes]
    anchor_sr = (dm_df.loc[dm_df["variant"] == "ANCHOR", "sr_full"].values[0]
                 if "ANCHOR" in dm_df["variant"].values else 0.0)
    for ax, dim in zip(axes, dims):
        sub = dm_df[dm_df["dim"] == dim].sort_values("sr_full")
        cols = sub["sr_full"].values
        if len(cols) == 0:
            continue
        norm = (cols - anchor_sr)
        vmax = max(abs(norm).max(), 1e-6)
        colors = plt.cm.RdYlGn(0.5 + 0.5 * (norm / vmax))
        ax.barh(sub["label"].tolist(), cols, color=colors)
        ax.axvline(anchor_sr, color="red", linestyle="--", alpha=0.7)
        ax.set_title(dim)
        ax.set_xlabel("SR_full")
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"[INFO] Heatmaps -> {path}")


def _print_final_verdict(master_ck, dm_results, dm_stats, abl_results, abl_stats,
                          dsr_df):
    print("\n" + "=" * 78)
    print("  FINAL VERDICT -- Strategy 183 Thesis/Live Promotion Evidence")
    print("=" * 78)
    if master_ck is None:
        print("  [WARN] MASTER checkpoint missing -- analysis incomplete.")
        return

    # 1. DM challengers beating MASTER (JKM p<0.05 AND LW CI lo>0 AND DSR p<0.15)
    print("\n  (1) DM variants beating MASTER after full correction")
    dsr_lookup = {}
    if dsr_df is not None and not dsr_df.empty:
        for _, r in dsr_df.iterrows():
            dsr_lookup[r["target"]] = r
    challengers = []
    for vid, s in dm_stats.items():
        if s.get("jkm_p") is None or np.isnan(s.get("jkm_p", np.nan)):
            continue
        ci_lo = s.get("lw_ci_lo", np.nan)
        p = s.get("jkm_p", np.nan)
        if (not np.isnan(p)) and p < 0.05 and (not np.isnan(ci_lo)) and ci_lo > 0:
            dsr_row = dsr_lookup.get(f"DM_{vid}_minus_MASTER")
            dsr_p = float(dsr_row["dsr_p"]) if (dsr_row is not None
                                                  and not pd.isna(dsr_row.get("dsr_p"))) else np.nan
            if np.isnan(dsr_p) or dsr_p < 0.15:
                challengers.append((vid, s["dsr_ann"], p, ci_lo, dsr_p))
    if challengers:
        print("  " + "-" * 74)
        print(f"  {'Variant':<28} {'dSR':>8} {'JKM p':>8} {'LW CI lo':>9} {'DSR p':>8}")
        for vid, dsr, p, ci_lo, dp in sorted(challengers, key=lambda x: -x[1]):
            dp_s = f"{dp:.4f}" if not np.isnan(dp) else "  --"
            print(f"  {vid:<28} {dsr:>+8.4f} {p:>8.4f} {ci_lo:>+9.4f} {dp_s:>8}")
    else:
        print("  None. S183 MASTER survives the full DM multi-testing bar.")

    # 2. Ablation variants whose removal costs >3 sigma of SR
    print("\n  (2) Ablation removals that cost MASTER > 3 sigma of SR")
    robust_components = []
    for vid, s in abl_stats.items():
        if s.get("jkm_z") is None or np.isnan(s.get("jkm_z", np.nan)):
            continue
        # Removal costs MASTER means variant SR is worse => dSR_ann < 0, z < 0
        if s["dsr_ann"] < 0 and abs(s["jkm_z"]) > 3.0:
            robust_components.append((vid, s["dsr_ann"], s["jkm_z"]))
    if robust_components:
        print("  " + "-" * 58)
        print(f"  {'Variant':<28} {'dSR':>9} {'JKM z':>9}")
        for vid, dsr, z in sorted(robust_components, key=lambda x: x[1]):
            print(f"  {vid:<28} {dsr:>+9.4f} {z:>+9.3f}")
    else:
        print("  None. No single component removal is >3 sigma significant.")

    # 3. Top-3 fragility flags: DM axes with |dSR|>0.10 between neighbours
    print("\n  (3) Top-3 DM fragility flags (|dSR_full| > 0.10 between neighbours)")
    flags = []
    by_dim = {}
    for vid, m in dm_results.items():
        if vid == "ANCHOR":
            continue
        meta = DM_VARIANTS.get(vid, {})
        d = meta.get("dim", "---")
        by_dim.setdefault(d, []).append((vid, m.get("sr_full", np.nan)))
    for d, lst in by_dim.items():
        srs = [v for _, v in lst if v is not None and not np.isnan(v)]
        if len(srs) < 2:
            continue
        srs_sorted = sorted(srs)
        max_gap = max(
            abs(srs_sorted[i + 1] - srs_sorted[i]) for i in range(len(srs_sorted) - 1))
        flags.append((d, max_gap, max(srs) - min(srs)))
    flags.sort(key=lambda x: -x[1])
    shown = [f for f in flags if f[1] > 0.10][:3]
    if shown:
        print("  " + "-" * 66)
        print(f"  {'Axis':<32} {'max neighbour gap':>19} {'range':>10}")
        for d, g, r in shown:
            print(f"  {d:<32} {g:>+19.4f} {r:>+10.4f}")
    else:
        print("  No axis shows |dSR| > 0.10 between neighbouring values (ROBUST).")

    # 4. SUBPERIOD DAMAGE TABLE -- catches the "sneak under" failure mode
    # where a variant looks free on dsr_full but silently degrades
    # post-2010 or post-2015 SR.  Carried over from the S182 suite, where
    # this check was added after the first S183 candidate (no-conv + no-XS
    # + static FDM) showed a flat full-sample dSR but a 5 pp post-2010 loss
    # that full-sample ablation could not surface.
    print("\n  (4) Ablation variants with post-2010/post-2015 'sneak under' damage")
    print("      (dsr_full > -0.02 but dsr_10 or dsr_15 < -0.02)")
    master_m = abl_results.get("MASTER") if abl_results else None
    sneak_rows = []
    if master_m is not None:
        m_full = master_m.get("sr_full")
        m_10 = master_m.get("sr_10")
        m_15 = master_m.get("sr_15")
        for vid, m in abl_results.items():
            if vid == "MASTER":
                continue
            try:
                d_f = float(m["sr_full"]) - float(m_full)
                d_10 = float(m["sr_10"]) - float(m_10)
                d_15 = float(m["sr_15"]) - float(m_15)
            except (TypeError, ValueError):
                continue
            # Sneak-under threshold: full-sample looks free (> -0.02)
            # but at least one subperiod takes a >2 pp hit.
            full_looks_free = d_f > -0.02
            sub_hit = (d_10 < -0.02) or (d_15 < -0.02)
            if full_looks_free and sub_hit:
                sneak_rows.append((vid, d_f, d_10, d_15))
    if sneak_rows:
        sneak_rows.sort(key=lambda r: min(r[2], r[3]))  # worst subperiod first
        print("  " + "-" * 72)
        print(f"  {'Variant':<28} {'dSR_full':>10} {'dSR_10':>10} {'dSR_15':>10}")
        for vid, df, d10, d15 in sneak_rows[:15]:
            tag_10 = " **" if d10 < -0.03 else ""
            tag_15 = " **" if d15 < -0.03 else ""
            print(f"  {vid:<28} {df:>+10.4f} {d10:>+10.4f}{tag_10}"
                  f" {d15:>+10.4f}{tag_15}")
        print(f"  ** = subperiod drop > 3 pp (material, likely load-bearing)")
    else:
        print("  None. No ablation variant hides subperiod damage behind dsr_full.")

    # 5. COMBINATION INTERACTION TABLE -- compares each GROUP 9 combination
    # to the additive prediction from its constituent single-delta ablations.
    # A large interaction term means the combination is more-than-the-sum:
    # stacking individually-free removals is genuinely unsafe.
    print("\n  (5) GROUP 9 combination interaction effects (measured - additive)")
    def _get_dsr(vid, tag):
        m = abl_results.get(vid)
        if m is None or master_m is None:
            return None
        ref_key = {"full": "sr_full", "10": "sr_10", "15": "sr_15"}[tag]
        try:
            return float(m[ref_key]) - float(master_m[ref_key])
        except (TypeError, ValueError, KeyError):
            return None

    # S183 suite combos: each key is a multi-delta variant from GROUP 9/10,
    # and the value is the list of single-delta revert variants that
    # should sum to the same dSR (pure additivity).  If abs(measured -
    # additive) is larger than sampling noise, the combo has a genuine
    # interaction effect.
    combos = {
        # GROUP 9 pairwise reverts of the three S183 v2 deltas
        "COMBO_ADDCONV_STEEP1":  ["R_S183_ADD_CONV", "R_S183_STEEP1"],
        "COMBO_ADDCONV_SMOOTH5": ["R_S183_ADD_CONV", "R_S183_SMOOTH5"],
        "COMBO_STEEP1_SMOOTH5":  ["R_S183_STEEP1",   "R_S183_SMOOTH5"],
        # GROUP 9 keep-only-one-delta combos
        "COMBO_KEEP_NOCONV_ONLY":  ["R_S183_STEEP1",  "R_S183_SMOOTH5"],
        "COMBO_KEEP_STEEP10_ONLY": ["R_S183_ADD_CONV", "R_S183_SMOOTH5"],
        "COMBO_KEEP_SMOOTH3_ONLY": ["R_S183_ADD_CONV", "R_S183_STEEP1"],
        # S183 v1 probes (drop XS + revert one v2 delta)
        "COMBO_NOXS_NOCONV":   ["R_S183_NO_XS"],
        "COMBO_NOXS_STEEP1":   ["R_S183_NO_XS", "R_S183_STEEP1"],
        "COMBO_NOXS_SMOOTH5":  ["R_S183_NO_XS", "R_S183_SMOOTH5"],
        # GROUP 10 S183 x load-bearing-component interactions
        "COMBO_S183_NO_VOVDIR":   ["R_NO_VOVDIR"],
        "COMBO_S183_STATIC_FDM":  ["D_100"],
        "COMBO_S183_NO_VOV":      ["VOV0"],
        "COMBO_S183_NO_OVERLAYS": ["C2"],
        "COMBO_S183_NO_SMOOTH":   ["C1"],
    }
    combo_rows = []
    for combo_vid, parts in combos.items():
        if combo_vid not in abl_results:
            continue
        row = [combo_vid]
        for tag in ("full", "10", "15"):
            measured = _get_dsr(combo_vid, tag)
            additive = None
            parts_dsrs = [_get_dsr(p, tag) for p in parts]
            if measured is not None and all(x is not None for x in parts_dsrs):
                additive = sum(parts_dsrs)
                interaction = measured - additive
            else:
                interaction = None
            row.extend([measured, additive, interaction])
        combo_rows.append(row)
    if combo_rows:
        print("  " + "-" * 78)
        print(f"  {'Variant':<26}  "
              f"{'full meas':>9}{'add':>8}{'int':>8}  "
              f"{'p10 meas':>9}{'add':>8}{'int':>8}  "
              f"{'p15 meas':>9}{'add':>8}{'int':>8}")
        for row in combo_rows:
            vid = row[0]
            cells = []
            for i in range(1, len(row)):
                v = row[i]
                cells.append(f"{v:>+8.4f}" if isinstance(v, (int, float)) else "     --")
            # Group cells into (meas, add, int) triples for each period
            print(f"  {vid:<26}  "
                  f"{cells[0]:>9}{cells[1]:>8}{cells[2]:>8}  "
                  f"{cells[3]:>9}{cells[4]:>8}{cells[5]:>8}  "
                  f"{cells[6]:>9}{cells[7]:>8}{cells[8]:>8}")
        print("  (int = measured - additive; large negative int = destructive interaction)")
    else:
        print("  No GROUP 9 combination checkpoints available; run --ablation first.")

    print("\n  Seed used           : %d" % RT_SEED)
    print("  Bootstrap B         : %d" % RT_BOOT_B)
    print("  Bootstrap block     : %d" % RT_BLOCK)
    print("=" * 78 + "\n")


def run_analysis(paths, args):
    print("\n" + "=" * 78)
    print("  ANALYSIS -- Strategy 183 Thesis-Grade Statistics Battery")
    print("=" * 78)
    ensure_dirs()

    # Drift guardrail: rebuild canonical twice and compare against the
    # pinned digest.  Fails loudly if any refactor / dependency change
    # perturbs the canonical master forecasts.  Opt out with --skip-self-test.
    if not getattr(args, "skip_self_test", False):
        try:
            self_test_canonical(strict=True, verbose=False)
            print("  [GUARDRAIL] self_test_canonical PASSED "
                  "(canonical master forecasts match the pinned digest).")
        except AssertionError as exc:
            print(f"  [GUARDRAIL] self_test_canonical FAILED:\n{exc}\n"
                  f"  Refusing to run analysis.  Fix canonical first, then re-pin "
                  f"CANONICAL_FORECAST_DIGEST in ig_strategy_183.py, or pass "
                  f"--skip-self-test to override.")
            raise SystemExit(1)

    master_ck = _load_ck_any(_abl_ck_path("MASTER"))
    if master_ck is None:
        alt = _HERE / f"{STRATEGY_NAME_MASTER}_checkpoint.pkl"
        master_ck = _load_ck_any(alt)
        if master_ck is not None:
            print(f"  [INFO] Using top-level S180 checkpoint {alt.name}")
    if master_ck is None:
        print("  [WARN] No MASTER checkpoint available. Run --ablation first.")

    # IRX
    global _IRX_SERIES
    if master_ck is not None:
        all_dates = pd.DatetimeIndex(master_ck["dates"])
    else:
        all_dates = pd.date_range("1980-01-01", "2027-01-01", freq="B")
    try:
        irx_arr = load_irx(all_dates)
        _IRX_SERIES = pd.Series(irx_arr, index=all_dates)
        print(f"[INFO] Loaded IRX risk-free (mean {_IRX_SERIES.mean() * 256:.2%} ann.)")
    except Exception as exc:
        _log_failure("ANALYSIS", "load_irx", exc)
        _IRX_SERIES = None

    dm_results, dm_stats = _collect_dm_results(master_ck)
    abl_results, abl_stats = _collect_abl_results(master_ck)

    # CSVs
    dm_df = _save_summary_csv(
        dm_results, dm_stats, master_ck,
        ANALYSIS_DIR / "datamining_summary.csv",
        prefix="datamining", registry=DM_VARIANTS)
    abl_df = _save_summary_csv(
        abl_results, abl_stats, master_ck,
        ANALYSIS_DIR / "ablation_summary.csv",
        prefix="ablation", registry=ABL_VARIANTS)
    _save_significance_csv(dm_stats,
                             ANALYSIS_DIR / "dm_significance.csv",
                             prefix="dm")
    _save_significance_csv(abl_stats,
                             ANALYSIS_DIR / "ablation_significance.csv",
                             prefix="ablation")
    _save_master_stability(master_ck, ANALYSIS_DIR / "master_stability.csv")

    # Asset class decomposition
    try:
        mapping = load_mapping(paths["mapping"])
        _save_master_by_asset_class(master_ck, mapping,
                                      ANALYSIS_DIR / "master_by_asset_class.csv")
    except Exception as exc:
        _log_failure("ANALYSIS", "asset_class", exc)

    # Deflated Sharpe Ratio (combined DM + Ablation multi-testing correction)
    dsr_df = _save_deflated_sharpe(master_ck, dm_results, abl_results,
                                     ANALYSIS_DIR / "deflated_sharpe.csv")

    # Supplementary block-bootstrap battery (Politis-White, Romano-Wolf,
    # Hansen SPA, block-length grid, one-sample SR CI, MDD CI).
    try:
        _save_extra_bootstrap(master_ck, ANALYSIS_DIR)
    except Exception as exc:
        _log_failure("ANALYSIS", "extra_bootstrap", exc)

    # Thesis-grade battery (distribution, Lo/Mertens SR SE, MinTRL,
    # Harvey-Liu haircut, PBO-CSCV, Model Confidence Set, crisis windows,
    # Henriksson-Merton/Treynor-Mazuy, optional factor regressions).
    try:
        mapping_loc = None
        try:
            mapping_loc = load_mapping(paths["mapping"])
        except Exception:
            mapping_loc = None
        _save_thesis_battery(master_ck, mapping_loc, paths, ANALYSIS_DIR)
    except Exception as exc:
        _log_failure("ANALYSIS", "thesis_battery", exc)

    # Plots
    try:
        _plot_stability_scatter(dm_df,
                                 ANALYSIS_DIR / "datamining_stability_scatter.png")
        _plot_heatmaps(dm_df, ANALYSIS_DIR / "datamining_heatmaps.png")
    except Exception as exc:
        _log_failure("ANALYSIS", "plots", exc)

    _print_final_verdict(master_ck, dm_results, dm_stats,
                          abl_results, abl_stats, dsr_df)


# ===========================================================================
# SECTION 7 -- PARSIMONY & RANDOM-PARAMETER BATTERY (T23-T28)
# ===========================================================================
#
# The core T1-T22 battery establishes (a) that MASTER's SR is distinguishable
# from null models, (b) that alpha survives known risk factors, (c) that the
# strategy is locally stable around MASTER on each DM axis one-at-a-time, and
# (d) that entire components are individually load-bearing. It does NOT
# answer two orthogonal questions:
#
#   1. Can MASTER be SIMPLIFIED further while retaining performance?
#      Formally: is MASTER the minimum specification that produces the
#      reported SR, or could an even smaller-parameter strategy match it?
#      -> T23 progressive parsimony (cumulative component drops)
#      -> T26 identity-parameter strategy (all lookbacks unified)
#      -> T27 coarse-rounding robustness (verifies MASTER is already on a
#             coarse grid with no "fitted digits")
#
#   2. Would an ARBITRARY choice of parameters produce similar performance?
#      Formally: where does MASTER sit in the empirical SR distribution
#      of strategies drawn from reasonable parameter ranges? If MASTER is
#      at the median, the specific parameter values are empirically free
#      (= not a tuned choice). If MASTER is in the 99th percentile, the
#      reported SR depends on the specific values and would not survive
#      any small perturbation.
#      -> T24 random-parameter Monte Carlo (uniform over DM axis ranges)
#      -> T25 random alpha-weight simplex (Dirichlet over 4-simplex)
#      -> T28 parameter-scramble catastrophe test (adversarial swaps)
#
# All six tests share the Phase-0 DM library (dm_precompute_library) and
# evaluate variants via dm_build_variant_signals + run_compounded_portfolio.
# Outputs land under Strategy_183/TestingSuite/Parsimony/.
#
# Runtime note. T24 and T25 are the expensive ones: each draws N variants
# and runs N full backtests. N defaults to 1000 (CLI --mc-N to override).
# At ~1 minute per variant that is ~1.5 hours per test. The cheap tests
# (T23, T26, T27, T28) together add ~30 more variants.
# ===========================================================================

PARSIMONY_DIR = SUITE_DIR / "Parsimony"


def _ensure_parsimony_dir():
    PARSIMONY_DIR.mkdir(parents=True, exist_ok=True)


def _sr_from_ck(ck):
    """Excess-of-IRX Sharpe on full sample, matching _metrics_from_ck."""
    if ck is None:
        return float("nan")
    dates = pd.DatetimeIndex(ck["dates"])
    dr = np.asarray(ck["daily_ret"], dtype=float)
    if _IRX_SERIES is not None:
        rf = _IRX_SERIES.reindex(dates).fillna(0.0).values
        ex = dr - rf
    else:
        ex = dr
    if len(ex) < 30:
        return 0.0
    s = ex.std(ddof=1)
    if s < 1e-12:
        return 0.0
    return float((ex.mean() * TRADING_DAYS) / (s * np.sqrt(TRADING_DAYS)))


def _p10_sr_from_ck(ck):
    """Post-2010 excess-of-IRX Sharpe."""
    if ck is None:
        return float("nan")
    dates = pd.DatetimeIndex(ck["dates"])
    dr = pd.Series(ck["daily_ret"], index=dates).astype(float)
    if _IRX_SERIES is not None:
        rf = _IRX_SERIES.reindex(dates).fillna(0.0)
        ex = dr - rf
    else:
        ex = dr
    ex = ex[ex.index >= POST2010]
    if len(ex) < 30:
        return 0.0
    s = float(ex.std(ddof=1))
    if s < 1e-12:
        return 0.0
    return float((ex.mean() * TRADING_DAYS) / (s * np.sqrt(TRADING_DAYS)))


def _ck_metrics_row(ck, variant_id, cfg_note=""):
    """Build a CSV-ready row of common metrics from a checkpoint."""
    if ck is None:
        return dict(variant_id=variant_id, sr_full=np.nan, sr_10=np.nan,
                    cagr=np.nan, max_dd=np.nan, ann_vol=np.nan, n_days=0,
                    cfg_note=cfg_note)
    dr = np.asarray(ck["daily_ret"], dtype=float)
    return dict(
        variant_id=variant_id,
        sr_full=round(_sr_from_ck(ck), 4),
        sr_10=round(_p10_sr_from_ck(ck), 4),
        cagr=round(float(ck.get("cagr", np.nan)), 4),
        max_dd=round(float(ck.get("max_dd", np.nan)), 4),
        ann_vol=round(float(ck.get("ann_vol", np.nan)), 4),
        n_days=int(len(dr)),
        cfg_note=cfg_note,
    )


def _run_parsimony_variant(library, cfg, variant_id, paths):
    """
    Build signals from cfg, run_compounded_portfolio writing under
    PARSIMONY_DIR with the given variant_id, return the checkpoint dict.
    `paths` is the full project-paths dict from get_project_paths;
    we override only paths["output"] so checkpoints land in the parsimony
    subdirectory.
    """
    inst_signals = dm_build_variant_signals(library, cfg, variant_id=variant_id)
    par_paths = {**paths, "output": str(PARSIMONY_DIR)}
    try:
        ck = run_compounded_portfolio(
            inst_signals, f"PAR_{variant_id}", par_paths,
            save_per_inst_pnl=False,
        )
    except Exception as exc:
        _log_failure("PARSIMONY", variant_id, exc)
        ck = None
    return ck


# ---------------------------------------------------------------------------
# T23 -- Progressive parsimony sweep
# ---------------------------------------------------------------------------

def t23_progressive_parsimony(library, master_ck, paths, out_dir,
                                abl_results=None):
    """
    Progressive parsimony sweep. Starting from MASTER, cumulatively drop
    components in the order they were individually least impactful in the
    ablation matrix. At each cumulative-drop step run a full backtest and
    record the resulting SR. The point is to trace how the BHY-corrected
    SR degrades as we progressively simplify the architecture, and to
    identify the minimum-spec variant that still delivers SR above a
    pre-registered threshold.

    Drop order is defined *a priori* by the ablation variants most similar
    to identity removals (conviction ramp, VoV direction, smoothing,
    overlays, individual speeds). The order is:

      1. conviction ramp       (use_conviction False)
      2. VoV direction         (vov_use_direction False)
      3. smoothing             (smooth_span 0)
      4. overlays              (use_overlays False)
      5. fastest EWMAC speed   (drop 16/64)
      6. slowest EWMAC speed   (drop 128/512)
      7. XS-momentum           (w_xs = 0, w_ts = 1)
      8. VoV alpha entirely    (drop to Trinity 1/3 T/C/S)

    Each step is CUMULATIVE: step 3 applies drops 1+2+3 together.
    """
    print("\n  [T23] Progressive parsimony sweep ...")
    _ensure_parsimony_dir()

    drop_sequence = [
        ("drop_conviction",
         dict(use_conviction=False),
         "Conviction ramp removed"),
        ("drop_vov_direction",
         dict(vov_use_direction=False),
         "VoV direction overlay removed"),
        ("drop_smoothing",
         dict(smooth_span=0),
         "EWM5 smoothing removed"),
        ("drop_overlays",
         dict(vol_trigger=999.0, vol_dampen=1.0,
              dd_threshold=-999.0, dd_scale=1.0),
         "Vol + DD overlays neutralised"),
        ("drop_fastest_ewmac",
         dict(speed_pairs=[(32, 128), (64, 256)],
              speed_weights=[0.5, 0.5]),
         "Fastest EWMAC speed (16/64) dropped"),
        ("drop_slowest_ewmac",
         dict(speed_pairs=[(32, 128)],
              speed_weights=[1.0]),
         "Slowest EWMAC speed (64/256) dropped (1-speed TS, middle only)"),
        ("drop_xs_momentum",
         dict(w_ts=1.0, w_xs=0.0),
         "XS momentum removed (100% TS trend)"),
        ("drop_vov_alpha",
         dict(w_trend=1.0/3, w_carry=1.0/3, w_skew=1.0/3, w_vov=0.0),
         "VoV alpha dropped (Trinity T/C/S)"),
    ]

    master_sr = round(_sr_from_ck(master_ck), 4)
    master_p10 = round(_p10_sr_from_ck(master_ck), 4)
    rows = [dict(
        step=0, variant_id="MASTER",
        cumulative_drops="(none)",
        sr_full=master_sr, sr_10=master_p10,
        cagr=round(float(master_ck.get("cagr", np.nan)), 4) if master_ck else np.nan,
        ann_vol=round(float(master_ck.get("ann_vol", np.nan)), 4) if master_ck else np.nan,
        max_dd=round(float(master_ck.get("max_dd", np.nan)), 4) if master_ck else np.nan,
        note="Reference baseline (S183 MASTER)",
    )]

    cumulative_cfg = {}
    cumulative_labels = []
    for step_idx, (name, cfg_delta, note) in enumerate(drop_sequence, start=1):
        cumulative_cfg.update(cfg_delta)
        cumulative_labels.append(name)
        vid = f"T23_STEP{step_idx:02d}_{name}"
        print(f"    step {step_idx}: cumulative = {' + '.join(cumulative_labels)}")
        ck = _run_parsimony_variant(library, dict(cumulative_cfg), vid, paths)
        rows.append(dict(
            step=step_idx, variant_id=vid,
            cumulative_drops=" + ".join(cumulative_labels),
            sr_full=round(_sr_from_ck(ck), 4),
            sr_10=round(_p10_sr_from_ck(ck), 4),
            cagr=round(float(ck.get("cagr", np.nan)), 4) if ck else np.nan,
            ann_vol=round(float(ck.get("ann_vol", np.nan)), 4) if ck else np.nan,
            max_dd=round(float(ck.get("max_dd", np.nan)), 4) if ck else np.nan,
            note=note,
        ))

    df = pd.DataFrame(rows)
    fp = out_dir / "t23_progressive_parsimony.csv"
    df.to_csv(str(fp), index=False)
    print(f"  [T23] -> {fp.name}")

    # Minimum-spec threshold lookup (SR >= 0.8 * MASTER)
    threshold = 0.80 * master_sr
    survivors = df[df["sr_full"] >= threshold]
    min_spec = survivors.iloc[-1] if len(survivors) > 0 else df.iloc[0]
    print(f"  [T23] Minimum-spec variant with SR >= 0.80 x MASTER "
          f"(={threshold:.3f}): step {int(min_spec['step'])} "
          f"({min_spec['variant_id']}) SR={min_spec['sr_full']:.3f}")
    return df


# ---------------------------------------------------------------------------
# T24 -- Random-parameter Monte Carlo
# ---------------------------------------------------------------------------

# Parameter sampling spec. Ranges are deliberately broad: not so extreme
# that everything fails, but wide enough that the median draw is a plausible
# "default" rather than a nearby-neighbour of MASTER. All sampling is done
# with a seeded numpy RandomState so the result is reproducible.
T24_CONTINUOUS = {
    # name -> (low, high, scale) where scale is "uniform" or "log"
    "trend_fdm":        (0.85, 1.20, "uniform"),
    "vol_trigger":      (0.80, 2.00, "uniform"),
    "vol_dampen":       (0.30, 0.80, "uniform"),
    "dd_threshold":     (-0.15, -0.03, "uniform"),
    "dd_scale":         (0.30, 0.80, "uniform"),
    "sigmoid_steepness":(0.5, 10.0, "log"),
    "fdm_cap":          (1.20, 3.00, "uniform"),
}

T24_DISCRETE = {
    "smooth_span":       [1, 2, 3, 5, 8, 10],
    "xs_lookback":       [126, 192, 256, 378, 512],
    "vov_window":        [21, 42, 64, 90, 126],
    "skew_window":       [128, 192, 256, 384, 512],
    "conviction_window": [128, 192, 256, 384, 512],
    "use_shifted_sigmoid": [True, False],
}


def _sample_t24_cfg(rng):
    """Draw one random parameter vector for T24."""
    cfg = {}
    for k, (lo, hi, scale) in T24_CONTINUOUS.items():
        if scale == "log":
            cfg[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            cfg[k] = float(rng.uniform(lo, hi))
    for k, options in T24_DISCRETE.items():
        cfg[k] = options[rng.randint(len(options))]
    return cfg


def t24_random_parameter_mc(library, master_ck, paths, out_dir, N, seed,
                              n_workers=1):
    """
    Draw N parameter vectors uniformly from the T24 spec, build and
    backtest each, and record the resulting SR. The question is where
    MASTER sits in the empirical SR distribution. If MASTER is at the
    median, the specific parameter values are empirically free. If MASTER
    is in the 99th percentile, the reported SR depends on the specific
    values and would not survive any small perturbation.
    """
    print(f"\n  [T24] Random-parameter Monte Carlo (N={N}, seed={seed}, "
          f"workers={n_workers}) ...")
    _ensure_parsimony_dir()
    rng = np.random.RandomState(seed)

    master_sr = _sr_from_ck(master_ck)
    master_p10 = _p10_sr_from_ck(master_ck)

    # Pre-generate all configs so the RNG sequence is deterministic
    # regardless of worker count.
    tasks = [(f"T24_MC{i:04d}", _sample_t24_cfg(rng)) for i in range(N)]

    if n_workers > 1:
        ck_map = {}
        with multiprocessing.Pool(
            processes=n_workers,
            initializer=_pool_init_dm,
            initargs=(library, paths),
        ) as pool:
            for vid, ck, err in tqdm(
                pool.imap_unordered(_par_worker, tasks),
                total=N,
                desc="T24 MC",
                unit="sample",
            ):
                if err:
                    _log_failure_raw("T24", vid, *err)
                ck_map[vid] = ck
        rows = []
        for vid, cfg in tasks:
            ck = ck_map.get(vid)
            rows.append(_ck_metrics_row(
                ck, vid,
                cfg_note=json.dumps(cfg, sort_keys=True, default=str)))
    else:
        rows = []
        for i, (vid, cfg) in enumerate(
            tqdm(tasks, desc="T24 MC", unit="sample")
        ):
            ck = _run_parsimony_variant(library, cfg, vid, paths)
            rows.append(_ck_metrics_row(
                ck, vid,
                cfg_note=json.dumps(cfg, sort_keys=True, default=str)))
            if (i + 1) % 10 == 0:
                current = [r["sr_full"] for r in rows
                           if not np.isnan(r["sr_full"])]
                if current:
                    med = float(np.median(current))
                    print(f"    [T24] {i+1}/{N} done  "
                          f"running median SR={med:.3f}  "
                          f"master SR={master_sr:.3f}")

    df = pd.DataFrame(rows)
    fp = out_dir / "t24_random_parameter_mc.csv"
    df.to_csv(str(fp), index=False)
    print(f"  [T24] -> {fp.name}")

    srs = df["sr_full"].dropna().values
    if len(srs) == 0:
        print("  [T24] No valid samples -- skipping summary")
        return df

    pct_rank = float(np.mean(srs >= master_sr))
    q = {str(int(100 * p)): round(float(np.quantile(srs, p)), 4)
         for p in (0.05, 0.25, 0.50, 0.75, 0.95)}
    summary = dict(
        n_samples=int(len(srs)),
        master_sr_full=round(master_sr, 4),
        master_p10_sr=round(master_p10, 4),
        random_sr_mean=round(float(srs.mean()), 4),
        random_sr_median=round(float(np.median(srs)), 4),
        random_sr_std=round(float(srs.std(ddof=1)), 4),
        random_sr_min=round(float(srs.min()), 4),
        random_sr_max=round(float(srs.max()), 4),
        pct_rank_master=round(pct_rank, 4),
        quantiles=q,
        pct_above_0_5=round(float(np.mean(srs > 0.5)), 4),
        pct_above_0_7=round(float(np.mean(srs > 0.7)), 4),
        seed=int(seed),
    )
    fp_json = out_dir / "t24_random_parameter_mc_summary.json"
    with open(str(fp_json), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [T24] -> {fp_json.name}")
    print(f"  [T24] Master SR={master_sr:.3f}  "
          f"random-sample median={summary['random_sr_median']:.3f}  "
          f"pct >= master = {pct_rank:.1%}  "
          f"pct SR>0.5 = {summary['pct_above_0_5']:.1%}")
    return df


# ---------------------------------------------------------------------------
# T25 -- Random alpha-weight simplex (Dirichlet sampling)
# ---------------------------------------------------------------------------

def _sample_dirichlet_simplex(rng, alpha=(1.0, 1.0, 1.0, 1.0)):
    """Draw one point from Dirichlet(alpha) on the 4-simplex."""
    return rng.dirichlet(alpha)


def t25_random_simplex(library, master_ck, paths, out_dir, N, seed,
                        n_workers=1):
    """
    Draw N alpha-weight vectors from a symmetric Dirichlet(1) (= uniform
    on the 4-simplex), rerun the backtest with each, and compare to the
    equal-weight MASTER. If the equal-weight default is indistinguishable
    from a random draw, the choice is empirically free (not a tuned pick).
    """
    print(f"\n  [T25] Random alpha-weight simplex Monte Carlo "
          f"(N={N}, seed={seed}, workers={n_workers}) ...")
    _ensure_parsimony_dir()
    rng = np.random.RandomState(seed)

    master_sr = _sr_from_ck(master_ck)
    master_p10 = _p10_sr_from_ck(master_ck)

    # Pre-generate all weight vectors so RNG sequence is deterministic.
    raw_weights = [_sample_dirichlet_simplex(rng) for _ in range(N)]
    tasks = []
    for i, w in enumerate(raw_weights):
        cfg = dict(
            w_trend=float(w[0]),
            w_carry=float(w[1]),
            w_skew=float(w[2]),
            w_vov=float(w[3]),
        )
        tasks.append((f"T25_SMX{i:04d}", cfg, w))

    def _make_row(vid, cfg, w, ck):
        row = _ck_metrics_row(
            ck, vid,
            cfg_note=f"w=T{w[0]:.3f}/C{w[1]:.3f}/S{w[2]:.3f}/V{w[3]:.3f}")
        row.update(dict(w_trend=round(float(w[0]), 4),
                        w_carry=round(float(w[1]), 4),
                        w_skew=round(float(w[2]), 4),
                        w_vov=round(float(w[3]), 4)))
        return row

    if n_workers > 1:
        worker_tasks = [(vid, cfg) for vid, cfg, _ in tasks]
        ck_map = {}
        with multiprocessing.Pool(
            processes=n_workers,
            initializer=_pool_init_dm,
            initargs=(library, paths),
        ) as pool:
            for vid, ck, err in tqdm(
                pool.imap_unordered(_par_worker, worker_tasks),
                total=N,
                desc="T25 simplex",
                unit="sample",
            ):
                if err:
                    _log_failure_raw("T25", vid, *err)
                ck_map[vid] = ck
        rows = [_make_row(vid, cfg, w, ck_map.get(vid))
                for vid, cfg, w in tasks]
    else:
        rows = []
        for i, (vid, cfg, w) in enumerate(
            tqdm(tasks, desc="T25 simplex", unit="sample")
        ):
            ck = _run_parsimony_variant(library, cfg, vid, paths)
            rows.append(_make_row(vid, cfg, w, ck))
            if (i + 1) % 10 == 0:
                print(f"    [T25] {i+1}/{N} done")

    df = pd.DataFrame(rows)
    fp = out_dir / "t25_random_simplex_mc.csv"
    df.to_csv(str(fp), index=False)
    print(f"  [T25] -> {fp.name}")

    srs = df["sr_full"].dropna().values
    if len(srs) == 0:
        print("  [T25] No valid samples -- skipping summary")
        return df

    pct_rank = float(np.mean(srs >= master_sr))
    summary = dict(
        n_samples=int(len(srs)),
        master_sr_full=round(master_sr, 4),
        master_p10_sr=round(master_p10, 4),
        equal_weight=(0.25, 0.25, 0.25, 0.25),
        random_sr_median=round(float(np.median(srs)), 4),
        random_sr_mean=round(float(srs.mean()), 4),
        random_sr_std=round(float(srs.std(ddof=1)), 4),
        random_sr_min=round(float(srs.min()), 4),
        random_sr_max=round(float(srs.max()), 4),
        pct_rank_master=round(pct_rank, 4),
        seed=int(seed),
    )
    fp_json = out_dir / "t25_random_simplex_summary.json"
    with open(str(fp_json), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [T25] -> {fp_json.name}")
    print(f"  [T25] Equal-weight master SR={master_sr:.3f}  "
          f"random simplex median={summary['random_sr_median']:.3f}  "
          f"pct >= master = {pct_rank:.1%}")
    return df


# ---------------------------------------------------------------------------
# T26 -- Identity-parameter strategy
# ---------------------------------------------------------------------------

def t26_identity_strategy(library, master_ck, paths, out_dir):
    """
    Build a single "maximally identity" variant where every free scalar
    is at its minimal/identity value and every lookback is unified to a
    single canonical base-2 window (256). This tests whether a genuinely
    minimal-parameter variant matches MASTER. If it does, MASTER has no
    "fitted precision" in its per-lookback tuning.

    Identity overrides applied simultaneously:
      speed_pairs        -> single (64, 256)         [single canonical speed]
      speed_weights      -> [1.0]                    [degenerate, identity]
      trend_fdm          -> 1.0                      [identity (same as S183_CFG)]
      xs_lookback        -> 256                      [unified base-2 year]
      conviction_window  -> 256                      [unified base-2 year]
      skew_window        -> 256                      [unified base-2 year]
      vov_window         -> 64                       [unified base-2 quarter]
      smooth_span        -> 5                        [one trading week]
      sigmoid_steepness  -> 1.0                      [identity (same as S183_CFG)]
      vol_trigger        -> 1.0                      [identity (same as S183_CFG)]
      vol_dampen         -> 0.50                     [half (same as S183_CFG)]
      dd_threshold       -> -VOL_TARGET / 2          [derived (same as S183_CFG)]
      dd_scale           -> 0.50                     [half (same as S183_CFG)]
      w_trend/carry/skew/vov -> 0.25 each            [equal]
      w_ts / w_xs        -> 0.50 / 0.50              [equal]

    The key non-trivial change vs S183 MASTER is the speed_pairs collapse
    to a single (64, 256) speed. MASTER has three speeds; this tests
    whether the multi-speed ensemble is load-bearing or if a single
    canonical speed suffices.
    """
    print("\n  [T26] Identity-parameter strategy (single canonical speed) ...")
    _ensure_parsimony_dir()

    identity_cfg = dict(
        speed_pairs=[(64, 256)],
        speed_weights=[1.0],
        trend_fdm=1.0,
        xs_lookback=256,
        conviction_window=256,
        skew_window=256,
        vov_window=64,
        smooth_span=5,
        sigmoid_steepness=1.0,
        vol_trigger=1.0,
        vol_dampen=0.50,
        dd_threshold=-(VOL_TARGET / 2.0),
        dd_scale=0.50,
        w_trend=0.25, w_carry=0.25, w_skew=0.25, w_vov=0.25,
        w_ts=0.50, w_xs=0.50,
    )
    ck = _run_parsimony_variant(library, identity_cfg, "T26_IDENTITY", paths)

    master_sr = round(_sr_from_ck(master_ck), 4)
    identity_sr = round(_sr_from_ck(ck), 4)
    dsr = round(identity_sr - master_sr, 4)

    row = dict(
        variant_id="T26_IDENTITY",
        master_sr_full=master_sr,
        identity_sr_full=identity_sr,
        dsr=dsr,
        master_p10_sr=round(_p10_sr_from_ck(master_ck), 4),
        identity_p10_sr=round(_p10_sr_from_ck(ck), 4),
        identity_cagr=round(float(ck.get("cagr", np.nan)), 4) if ck else np.nan,
        identity_ann_vol=round(float(ck.get("ann_vol", np.nan)), 4) if ck else np.nan,
        note="Single 64/256 EWMAC speed; all lookbacks unified to 256",
    )
    df = pd.DataFrame([row])
    fp = out_dir / "t26_identity_strategy.csv"
    df.to_csv(str(fp), index=False)
    print(f"  [T26] -> {fp.name}")
    print(f"  [T26] master SR={master_sr:.3f}  identity SR={identity_sr:.3f}  "
          f"dSR={dsr:+.3f}")
    return df


# ---------------------------------------------------------------------------
# T27 -- Coarse-rounding robustness
# ---------------------------------------------------------------------------

def _round_to_pow2(x):
    """Round an integer to the nearest power of two (>= 2)."""
    if x <= 2: return 2
    lo = 2 ** int(np.floor(np.log2(x)))
    hi = 2 ** int(np.ceil(np.log2(x)))
    return lo if (x - lo) <= (hi - x) else hi


def _round_to_grid(x, step):
    return round(round(x / step) * step, 6)


def t27_coarse_rounding(library, master_ck, paths, out_dir):
    """
    Round every numeric parameter in S183 MASTER to a coarse grid:
      - all lookbacks to the nearest power of two (>= 2)
      - all multipliers and scale factors to the nearest 0.10
      - all thresholds to the nearest 0.02
    and rerun. If the rounded MASTER equals MASTER (i.e., MASTER is
    already on the grid), the test confirms MASTER has no "fitted digits"
    in any parameter. If the rounded MASTER differs meaningfully from
    MASTER's performance, MASTER relies on fine-grained tuning.
    """
    print("\n  [T27] Coarse-rounding robustness ...")
    _ensure_parsimony_dir()

    # Pull S183 MASTER's parameters and round each one.
    # (Alpha weights already lie on the 0.05 grid at 0.25 each.)
    rounded_cfg = dict(
        trend_fdm=_round_to_grid(S183_CFG["trend_fdm"], 0.10),
        smooth_span=_round_to_pow2(S183_CFG["smooth_span"]),   # 5 -> 4
        sigmoid_steepness=_round_to_grid(S183_CFG["sigmoid_steepness"], 0.10),
        vol_trigger=_round_to_grid(S183_CFG["vol_trigger"], 0.10),
        vol_dampen=_round_to_grid(S183_CFG["vol_dampen"], 0.10),
        dd_threshold=_round_to_grid(S183_CFG["dd_threshold"], 0.02),
        dd_scale=_round_to_grid(S183_CFG["dd_scale"], 0.10),
        xs_lookback=_round_to_pow2(S183_CFG["xs_lookback"]),
        vov_window=_round_to_pow2(S183_CFG["vov_window"]),
        skew_window=_round_to_pow2(S183_CFG["skew_window"]),
        conviction_window=_round_to_pow2(S183_CFG["conviction_window"]),
    )
    # Build a "diff" note so the CSV shows which parameters actually moved
    deltas = []
    for k, new_v in rounded_cfg.items():
        old_v = S183_CFG[k]
        if isinstance(new_v, float):
            changed = abs(new_v - old_v) > 1e-9
        else:
            changed = new_v != old_v
        if changed:
            deltas.append(f"{k}: {old_v} -> {new_v}")
    note = " | ".join(deltas) if deltas else "IDENTICAL to MASTER (already on coarse grid)"

    ck = _run_parsimony_variant(library, rounded_cfg, "T27_COARSE", paths)

    master_sr = round(_sr_from_ck(master_ck), 4)
    rounded_sr = round(_sr_from_ck(ck), 4)
    dsr = round(rounded_sr - master_sr, 4)

    row = dict(
        variant_id="T27_COARSE",
        master_sr_full=master_sr,
        rounded_sr_full=rounded_sr,
        dsr=dsr,
        master_p10_sr=round(_p10_sr_from_ck(master_ck), 4),
        rounded_p10_sr=round(_p10_sr_from_ck(ck), 4),
        rounded_cagr=round(float(ck.get("cagr", np.nan)), 4) if ck else np.nan,
        rounded_ann_vol=round(float(ck.get("ann_vol", np.nan)), 4) if ck else np.nan,
        deltas=note,
    )
    df = pd.DataFrame([row])
    fp = out_dir / "t27_coarse_rounding.csv"
    df.to_csv(str(fp), index=False)
    print(f"  [T27] -> {fp.name}")
    print(f"  [T27] master SR={master_sr:.3f}  rounded SR={rounded_sr:.3f}  "
          f"dSR={dsr:+.3f}")
    if not deltas:
        print("  [T27] Rounded MASTER == MASTER (no parameters off the coarse grid).")
    else:
        print(f"  [T27] Moved parameters: {len(deltas)}")
    return df


# ---------------------------------------------------------------------------
# T28 -- Parameter-scramble catastrophe test
# ---------------------------------------------------------------------------

def t28_parameter_scramble(library, master_ck, paths, out_dir):
    """
    Adversarial test. Build ~15 scrambles that swap parameter values
    between axes or push them to architecturally absurd values. Most
    should fail catastrophically (SR << 0); the few that don't reveal
    unintended flexibility in the strategy. This is the mirror image of
    T24: instead of asking "do arbitrary reasonable parameters work?",
    T28 asks "do deliberately bad parameters break the strategy?".

    Expected outcome: median scramble SR near 0, with no scramble
    exceeding MASTER's SR.
    """
    print("\n  [T28] Parameter-scramble catastrophe test ...")
    _ensure_parsimony_dir()

    scrambles = [
        ("swap_smooth_vov",
         dict(smooth_span=64, vov_window=5),
         "smooth_span <-> vov_window (absurd smoothing)"),
        ("swap_skew_dd",
         dict(skew_window=64, conviction_window=16),
         "skew_window <-> conviction_window shrunken"),
        ("mega_fdm",
         dict(fdm_cap=10.0, trend_fdm=10.0),
         "Huge FDM_CAP and TREND_FDM multipliers"),
        ("binary_sigmoid",
         dict(sigmoid_steepness=100.0, use_shifted_sigmoid=True),
         "Near-binary cliff-edge sigmoid, zero-drag"),
        ("positive_dd_threshold",
         dict(dd_threshold=0.50, dd_scale=0.10),
         "Positive DD threshold (always engaged)"),
        ("trigger_zero",
         dict(vol_trigger=0.0, vol_dampen=0.05),
         "Vol trigger at 0 (always clipping)"),
        ("trigger_huge",
         dict(vol_trigger=100.0, vol_dampen=0.0),
         "Vol trigger at 100 (never engaged)"),
        ("tiny_smooth_window",
         dict(smooth_span=100),
         "Massive smoothing (100 trading days)"),
        ("zero_smooth",
         dict(smooth_span=0),
         "No smoothing (raw master forecast)"),
        ("invert_alpha_weights_t",
         dict(w_trend=-0.25, w_carry=0.5, w_skew=0.5, w_vov=0.25),
         "Negative trend weight (short the trend signal)"),
        ("100pct_vov",
         dict(w_trend=0.0, w_carry=0.0, w_skew=0.0, w_vov=1.0),
         "100% VoV alpha"),
        ("100pct_carry",
         dict(w_trend=0.0, w_carry=1.0, w_skew=0.0, w_vov=0.0),
         "100% Carry alpha"),
        ("all_identical_lookbacks",
         dict(conviction_window=16, vov_window=16, skew_window=16,
              xs_lookback=16),
         "All lookbacks collapsed to 16 days"),
        ("extreme_slow",
         dict(conviction_window=2048, vov_window=2048, skew_window=2048,
              xs_lookback=2048),
         "All lookbacks extended to 2048 days"),
        ("no_xs_plus_no_convict",
         dict(w_ts=1.0, w_xs=0.0, use_conviction=False,
              speed_pairs=[(128, 512)], speed_weights=[1.0]),
         "Single slow TS speed, no conviction, no XS"),
    ]

    rows = []
    master_sr = round(_sr_from_ck(master_ck), 4)
    for i, (name, cfg, note) in enumerate(scrambles, start=1):
        vid = f"T28_{name}"
        print(f"    scramble {i}/{len(scrambles)}: {name}")
        ck = _run_parsimony_variant(library, cfg, vid, paths)
        rows.append(dict(
            scramble_id=name,
            variant_id=vid,
            sr_full=round(_sr_from_ck(ck), 4),
            sr_10=round(_p10_sr_from_ck(ck), 4),
            cagr=round(float(ck.get("cagr", np.nan)), 4) if ck else np.nan,
            ann_vol=round(float(ck.get("ann_vol", np.nan)), 4) if ck else np.nan,
            max_dd=round(float(ck.get("max_dd", np.nan)), 4) if ck else np.nan,
            note=note,
        ))

    df = pd.DataFrame(rows)
    fp = out_dir / "t28_parameter_scramble.csv"
    df.to_csv(str(fp), index=False)

    srs = df["sr_full"].dropna().values
    median_scramble = float(np.median(srs)) if len(srs) else float("nan")
    n_above_master = int(np.sum(srs >= master_sr))
    n_positive = int(np.sum(srs > 0))

    summary = dict(
        n_scrambles=int(len(rows)),
        master_sr_full=master_sr,
        scramble_sr_median=round(median_scramble, 4),
        scramble_sr_mean=round(float(srs.mean()), 4) if len(srs) else float("nan"),
        scramble_sr_min=round(float(srs.min()), 4) if len(srs) else float("nan"),
        scramble_sr_max=round(float(srs.max()), 4) if len(srs) else float("nan"),
        n_above_master=n_above_master,
        n_positive=n_positive,
    )
    fp_json = out_dir / "t28_parameter_scramble_summary.json"
    with open(str(fp_json), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [T28] -> {fp.name}")
    print(f"  [T28] -> {fp_json.name}")
    print(f"  [T28] master SR={master_sr:.3f}  "
          f"scramble median={median_scramble:.3f}  "
          f"n_above_master={n_above_master}/{len(rows)}  "
          f"n_positive={n_positive}/{len(rows)}")
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_parsimony_battery(paths, args):
    """
    Run the full T23-T28 parsimony & random-parameter battery.
    Requires the Phase-0 DM library; if run_datamining has not been
    executed yet, this function rebuilds the library itself.
    """
    print("\n" + "=" * 78)
    print("  PARSIMONY & RANDOM-PARAMETER BATTERY -- Strategy 183 (T23-T28)")
    print("=" * 78)
    ensure_dirs()
    _ensure_parsimony_dir()

    # 1. Load MASTER checkpoint (same logic as run_analysis)
    master_ck = _load_ck_any(_abl_ck_path("MASTER"))
    if master_ck is None:
        alt = _HERE / f"{STRATEGY_NAME_MASTER}_checkpoint.pkl"
        master_ck = _load_ck_any(alt)
        if master_ck is not None:
            print(f"  [INFO] Using top-level S183_CFG checkpoint {alt.name}")
    if master_ck is None:
        print("  [WARN] No MASTER checkpoint. Run --ablation or S183_CFG master run first.")
        print("  [WARN] Skipping parsimony battery.")
        return

    # 2. IRX (needed by _sr_from_ck / _p10_sr_from_ck)
    global _IRX_SERIES
    all_dates = pd.DatetimeIndex(master_ck["dates"])
    try:
        irx_arr = load_irx(all_dates)
        _IRX_SERIES = pd.Series(irx_arr, index=all_dates)
    except Exception as exc:
        _log_failure("PARSIMONY", "load_irx", exc)
        _IRX_SERIES = None

    # 3. Build the Phase-0 library (shared across all 6 tests)
    print("\n[PAR PHASE 0] Building DM library for parsimony tests ...")
    mapping = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])
    library = dm_precompute_library(mapping, fx_daily, paths)

    # 4. Run tests
    n_mc      = int(getattr(args, "mc_N",    1000))
    seed      = int(getattr(args, "seed",    20260405))
    n_workers = int(getattr(args, "workers", 1))

    try:
        t23_progressive_parsimony(library, master_ck, paths, ANALYSIS_DIR)
    except Exception as exc:
        _log_failure("PARSIMONY", "T23", exc)

    try:
        t24_random_parameter_mc(library, master_ck, paths, ANALYSIS_DIR,
                                  N=n_mc, seed=seed, n_workers=n_workers)
    except Exception as exc:
        _log_failure("PARSIMONY", "T24", exc)

    try:
        t25_random_simplex(library, master_ck, paths, ANALYSIS_DIR,
                            N=n_mc, seed=seed + 1, n_workers=n_workers)
    except Exception as exc:
        _log_failure("PARSIMONY", "T25", exc)

    try:
        t26_identity_strategy(library, master_ck, paths, ANALYSIS_DIR)
    except Exception as exc:
        _log_failure("PARSIMONY", "T26", exc)

    try:
        t27_coarse_rounding(library, master_ck, paths, ANALYSIS_DIR)
    except Exception as exc:
        _log_failure("PARSIMONY", "T27", exc)

    try:
        t28_parameter_scramble(library, master_ck, paths, ANALYSIS_DIR)
    except Exception as exc:
        _log_failure("PARSIMONY", "T28", exc)

    print("\n[PARSIMONY] Battery complete. See Strategy_183/TestingSuite/Analysis/"
          "t23_*..t28_*.csv")


# ===========================================================================
# SECTION 8 -- EXTENDED ROBUSTNESS BATTERY (T29-T34)
# ===========================================================================
#
# Six tests filling gaps in the T1-T22 analysis battery and the T23-T28
# parsimony battery. Each test addresses a specific defensive claim the
# thesis needs:
#
#   T29  T+1 execution lag         Defends the "live-deployable under
#                                   realistic latency" claim. One backtest
#                                   with forecasts shifted +1 day.
#
#   T30  Instrument leave-one-out  Tests whether MASTER's SR is robust to
#                                   removing any single instrument. 62
#                                   backtests; the strongest parsimony
#                                   evidence available for cross-instrument
#                                   generalisation.
#
#   T31  Signal noise injection    Adds Gaussian noise to the master
#                                   forecast at 4 SNR levels. A fitted
#                                   strategy collapses under noise; a
#                                   robust one degrades gracefully.
#
#   T32  Leverage scaling          Scales the master forecast by k in
#                                   {0.8, 1.0, 1.2}. SR should be invariant
#                                   to k in the cap-unconstrained regime.
#                                   Confirms the S181 linear-scaling claim.
#
#   T33  IRX attribution           Analytical decomposition (no new backtest)
#                                   of reported SR into cash-yield and
#                                   trading-alpha components.
#
#   T34  Calendar-decade SRs       Analytical cut (no new backtest) of
#                                   MASTER by 1990s / 2000s / 2010s / 2020s.
#
# Runtime at default settings: ~70 new backtests (62 T30 + 4 T31 + 3 T32 +
# 1 T29 + 0 T33/T34), ~50 min serial on a typical single-machine baseline.
# With --workers 6 the parallelised T30 drops to ~10 min and the whole
# battery runs in under 15 min.
# ===========================================================================

ROBUSTNESS_DIR = SUITE_DIR / "Robustness"


def _ensure_robustness_dir():
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)


def _run_robustness_variant(library, cfg, variant_id, paths, postprocess=None):
    """
    Like _run_parsimony_variant but writes under ROBUSTNESS_DIR and
    optionally applies a `postprocess` hook to the inst_signals dict
    between signal-building and portfolio simulation. `postprocess` is
    a callable taking `inst_signals` and returning (possibly the same)
    dict with modified forecasts -- used by T29 (shift forecasts), T31
    (add noise), and T32 (scale forecasts).
    """
    inst_signals = dm_build_variant_signals(library, cfg, variant_id=variant_id)
    if postprocess is not None:
        inst_signals = postprocess(inst_signals)
    rb_paths = {**paths, "output": str(ROBUSTNESS_DIR)}
    try:
        ck = run_compounded_portfolio(
            inst_signals, f"RB_{variant_id}", rb_paths,
            save_per_inst_pnl=False,
        )
    except Exception as exc:
        _log_failure("ROBUSTNESS", variant_id, exc)
        ck = None
    return ck


# ---------------------------------------------------------------------------
# T29 -- T+1 execution lag
# ---------------------------------------------------------------------------

def t29_t_plus_1_lag(library, master_ck, paths, out_dir):
    """
    Rerun MASTER with all instrument forecasts shifted forward one day
    (the strategy trades on information that is 1 day stale). The S180
    report's live-execution section claims T+1 lag costs only ~0.4% of
    Sharpe but does not actually run this test in the suite. This closes
    that gap.
    """
    print("\n  [T29] T+1 execution lag ...")
    _ensure_robustness_dir()

    def shift_forecasts(inst_signals):
        for inst, sig in inst_signals.items():
            sig["forecast"] = sig["forecast"].shift(1).fillna(0.0)
        return inst_signals

    ck = _run_robustness_variant(library, {}, "T29_T_PLUS_1_LAG",
                                   paths, postprocess=shift_forecasts)
    master_sr = _sr_from_ck(master_ck)
    lagged_sr = _sr_from_ck(ck)

    row = dict(
        variant_id="T29_T_PLUS_1_LAG",
        master_sr_full=round(master_sr, 4),
        lagged_sr_full=round(lagged_sr, 4),
        dsr=round(lagged_sr - master_sr, 4),
        pct_sr_retained=(round(lagged_sr / master_sr, 4)
                          if master_sr else 0.0),
        master_p10_sr=round(_p10_sr_from_ck(master_ck), 4),
        lagged_p10_sr=round(_p10_sr_from_ck(ck), 4),
        note="All forecasts shifted +1 day (forecast.shift(1).fillna(0))",
    )
    df = pd.DataFrame([row])
    fp = out_dir / "t29_t_plus_1_lag.csv"
    df.to_csv(str(fp), index=False)
    print(f"  [T29] -> {fp.name}")
    print(f"  [T29] master SR={master_sr:.4f}  lagged SR={lagged_sr:.4f}  "
          f"dSR={lagged_sr - master_sr:+.4f}  "
          f"retained={row['pct_sr_retained']:.1%}")
    return df


# ---------------------------------------------------------------------------
# T30 -- Instrument leave-one-out jackknife
# ---------------------------------------------------------------------------

def t30_instrument_leave_one_out(library, master_ck, paths, out_dir):
    """
    Leave-one-out jackknife over the instrument universe. Drop each
    instrument in turn and rerun MASTER on the sub-universe. Produces
    a distribution of N_inst Sharpe ratios showing how much MASTER's
    reported SR depends on any single instrument.

    A strategy whose SR depends on one or two instruments (large
    max-drop |dSR|) is fragile. S183_CFG's parsimony story requires all
    instruments to contribute approximately equally.
    """
    print("\n  [T30] Instrument leave-one-out jackknife ...")
    _ensure_robustness_dir()

    all_insts = sorted(library.keys())
    master_sr = _sr_from_ck(master_ck)
    master_p10 = _p10_sr_from_ck(master_ck)
    print(f"    [T30] {len(all_insts)} instruments to drop")

    rows = []
    for i, dropped in enumerate(all_insts):
        sub_library = {k: v for k, v in library.items() if k != dropped}
        vid = f"T30_DROP_{dropped}"
        ck = _run_robustness_variant(sub_library, {}, vid, paths)
        row = _ck_metrics_row(ck, vid, cfg_note=f"dropped: {dropped}")
        row["dropped_instrument"] = dropped
        row["dsr"] = round(row["sr_full"] - master_sr, 4) if not np.isnan(
            row["sr_full"]) else np.nan
        rows.append(row)
        if (i + 1) % 10 == 0 or (i + 1) == len(all_insts):
            print(f"    [T30] {i + 1}/{len(all_insts)} instruments done")

    df = pd.DataFrame(rows)
    fp = out_dir / "t30_instrument_leave_one_out.csv"
    df.to_csv(str(fp), index=False)
    print(f"  [T30] -> {fp.name}")

    srs = df["sr_full"].dropna().values
    if len(srs) == 0:
        print("  [T30] No valid samples; skipping summary")
        return df

    dsrs = df["dsr"].dropna().values
    dropped_names = df.loc[df["dsr"].notna(), "dropped_instrument"].values

    min_idx = int(np.argmin(dsrs))
    max_idx = int(np.argmax(dsrs))

    summary = dict(
        n_drops=int(len(srs)),
        master_sr_full=round(master_sr, 4),
        master_p10_sr=round(master_p10, 4),
        jackknife_sr_mean=round(float(srs.mean()), 4),
        jackknife_sr_std=round(float(srs.std(ddof=1)), 4),
        jackknife_sr_min=round(float(srs.min()), 4),
        jackknife_sr_max=round(float(srs.max()), 4),
        max_disruption_dsr=round(float(dsrs[min_idx]), 4),
        max_disruption_instrument=str(dropped_names[min_idx]),
        max_boost_dsr=round(float(dsrs[max_idx]), 4),
        max_boost_instrument=str(dropped_names[max_idx]),
        n_above_0_8_master=int(np.sum(srs >= 0.8 * master_sr)),
        n_above_0_9_master=int(np.sum(srs >= 0.9 * master_sr)),
    )
    fp_json = out_dir / "t30_instrument_loo_summary.json"
    with open(str(fp_json), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  [T30] -> {fp_json.name}")
    print(f"  [T30] master SR={master_sr:.4f}  "
          f"jackknife mean={summary['jackknife_sr_mean']:.4f}  "
          f"std={summary['jackknife_sr_std']:.4f}")
    print(f"  [T30] most disruptive drop: {summary['max_disruption_instrument']} "
          f"(dSR={summary['max_disruption_dsr']:+.4f})")
    print(f"  [T30] largest-boost drop:   {summary['max_boost_instrument']} "
          f"(dSR={summary['max_boost_dsr']:+.4f})")
    return df


# ---------------------------------------------------------------------------
# T31 -- Signal noise injection
# ---------------------------------------------------------------------------

def t31_signal_noise_injection(library, master_ck, paths, out_dir, seed=20260405):
    """
    Add Gaussian noise to the master forecast at four SNR levels:
    sigma_noise in {0.1, 0.25, 0.5, 1.0} * sigma_forecast per instrument.
    Rerun MASTER with forecast + N(0, sigma_noise) clipped to [-20, 20]
    and record the resulting SR.

    A fitted strategy collapses under signal noise; a robust one
    degrades gracefully. This complements T24 (random parameters in
    parameter space) by testing robustness in signal space.
    """
    print(f"\n  [T31] Signal noise injection (seed={seed}) ...")
    _ensure_robustness_dir()

    master_sr = _sr_from_ck(master_ck)
    rows = [dict(
        variant_id="T31_NOISE_0.00",
        noise_level=0.0,
        sr_full=round(master_sr, 4),
        sr_10=round(_p10_sr_from_ck(master_ck), 4),
        dsr=0.0,
        note="Baseline (no noise injection)",
    )]

    for level in [0.10, 0.25, 0.50, 1.00]:
        # Deterministic per-level seed so a partial rerun is reproducible
        rng = np.random.RandomState(seed + int(round(level * 1000)))

        def make_noiser(lvl, the_rng):
            def inject(inst_signals):
                for inst, sig in inst_signals.items():
                    f = sig["forecast"]
                    s = float(f.std()) * lvl
                    if s < 1e-12:
                        continue
                    noise = the_rng.normal(0.0, s, size=len(f))
                    sig["forecast"] = (f + pd.Series(noise, index=f.index)
                                        ).clip(-FORECAST_CAP, FORECAST_CAP)
                return inst_signals
            return inject

        vid = f"T31_NOISE_{level:.2f}"
        ck = _run_robustness_variant(
            library, {}, vid, paths,
            postprocess=make_noiser(level, rng),
        )
        sr = _sr_from_ck(ck)
        rows.append(dict(
            variant_id=vid,
            noise_level=level,
            sr_full=round(sr, 4),
            sr_10=round(_p10_sr_from_ck(ck), 4),
            dsr=round(sr - master_sr, 4),
            note=f"Gaussian noise: sigma = {level} x sigma_forecast per instrument",
        ))

    df = pd.DataFrame(rows)
    fp = out_dir / "t31_signal_noise_injection.csv"
    df.to_csv(str(fp), index=False)
    print(f"  [T31] -> {fp.name}")
    for r in rows:
        if r["noise_level"] == 0:
            continue
        print(f"  [T31] noise={r['noise_level']:.2f}*sigma  "
              f"SR={r['sr_full']:.4f}  dSR={r['dsr']:+.4f}")
    return df


# ---------------------------------------------------------------------------
# T32 -- Leverage scaling invariance
# ---------------------------------------------------------------------------

def t32_leverage_scaling(library, master_ck, paths, out_dir):
    """
    Scale the master forecast by k in {0.8, 1.0, 1.2} and rerun. In the
    cap-unconstrained regime, SR should be invariant to k (positions
    scale linearly, returns scale linearly, SR = mean/vol is scale-
    invariant). Deviation from invariance flags either:
      - FORECAST_CAP biting (nonlinear truncation at |fc| > 20)
      - implicit dependence on an absolute magnitude elsewhere

    This test confirms the thesis's "scale linearly with leverage"
    claim from the S181 verification (S181 = S183_CFG at 1.5x leverage).
    """
    print("\n  [T32] Leverage scaling invariance ...")
    _ensure_robustness_dir()

    master_sr = _sr_from_ck(master_ck)
    rows = []

    for k in [0.8, 1.0, 1.2]:
        def make_scaler(scale):
            def apply_scale(inst_signals):
                for inst, sig in inst_signals.items():
                    sig["forecast"] = (sig["forecast"] * scale
                                        ).clip(-FORECAST_CAP, FORECAST_CAP)
                return inst_signals
            return apply_scale

        vid = f"T32_LEV_{int(round(k * 100)):03d}"
        ck = _run_robustness_variant(
            library, {}, vid, paths,
            postprocess=make_scaler(k),
        )
        sr = _sr_from_ck(ck)
        cagr = (round(float(ck.get("cagr", np.nan)), 4)
                if ck is not None else np.nan)
        vol = (round(float(ck.get("ann_vol", np.nan)), 4)
               if ck is not None else np.nan)
        mdd = (round(float(ck.get("max_dd", np.nan)), 4)
               if ck is not None else np.nan)
        rows.append(dict(
            variant_id=vid,
            leverage_k=k,
            sr_full=round(sr, 4),
            sr_10=round(_p10_sr_from_ck(ck), 4),
            cagr=cagr,
            ann_vol=vol,
            max_dd=mdd,
            dsr_vs_master=round(sr - master_sr, 4),
        ))

    df = pd.DataFrame(rows)
    fp = out_dir / "t32_leverage_scaling.csv"
    df.to_csv(str(fp), index=False)
    print(f"  [T32] -> {fp.name}")

    sr_values = df["sr_full"].values
    sr_range = float(sr_values.max() - sr_values.min())
    vol_values = df["ann_vol"].values
    for i, r in enumerate(rows):
        print(f"  [T32] k={r['leverage_k']:.2f}: SR={r['sr_full']:.4f}  "
              f"CAGR={r['cagr']*100:.2f}%  vol={r['ann_vol']*100:.2f}%")
    print(f"  [T32] SR range across k: {sr_range:+.4f} "
          f"(expect ~0 in cap-unconstrained regime)")
    return df


# ---------------------------------------------------------------------------
# T33 -- IRX attribution (analytical)
# ---------------------------------------------------------------------------

def t33_irx_attribution(master_ck, out_dir):
    """
    Analytical decomposition (no new backtest) of the reported Sharpe
    into gross-return and excess-of-IRX components. Shows what fraction
    of the reported SR comes from the T-bill cash yield on collateral
    vs pure trading alpha. A thesis that claims a high SR largely
    sourced from IRX is vulnerable to a rate-cut regime.
    """
    print("\n  [T33] IRX attribution (analytical) ...")
    _ensure_robustness_dir()

    dates = pd.DatetimeIndex(master_ck["dates"])
    dr = np.asarray(master_ck["daily_ret"], dtype=float)

    if len(dr) < 30 or dr.std(ddof=1) < 1e-12:
        print("  [T33] MASTER checkpoint too small; skipping")
        return None

    sr_gross = float(
        dr.mean() * TRADING_DAYS / (dr.std(ddof=1) * np.sqrt(TRADING_DAYS)))

    if _IRX_SERIES is not None:
        rf = _IRX_SERIES.reindex(dates).fillna(0.0).values
        excess = dr - rf
        if excess.std(ddof=1) < 1e-12:
            sr_excess = 0.0
        else:
            sr_excess = float(
                excess.mean() * TRADING_DAYS /
                (excess.std(ddof=1) * np.sqrt(TRADING_DAYS)))
        ann_irx = float(rf.mean() * TRADING_DAYS)
    else:
        sr_excess = sr_gross
        ann_irx = 0.0

    delta = sr_gross - sr_excess
    pct_from_cash = delta / sr_gross if sr_gross else 0.0

    row = dict(
        variant_id="T33_IRX_ATTRIBUTION",
        sr_gross=round(sr_gross, 4),
        sr_excess_of_irx=round(sr_excess, 4),
        delta_sr_from_irx=round(delta, 4),
        annual_irx_rate=round(ann_irx, 4),
        pct_sr_from_cash=round(pct_from_cash, 4),
        note="Gross SR uses total NAV return; excess SR subtracts daily IRX",
    )
    df = pd.DataFrame([row])
    fp = out_dir / "t33_irx_attribution.csv"
    df.to_csv(str(fp), index=False)
    print(f"  [T33] -> {fp.name}")
    print(f"  [T33] gross SR={sr_gross:.4f}  excess SR={sr_excess:.4f}  "
          f"cash contribution={delta:+.4f} ({pct_from_cash:+.1%})")
    return df


# ---------------------------------------------------------------------------
# T34 -- Calendar-decade SRs (analytical)
# ---------------------------------------------------------------------------

def t34_decade_sharpe(master_ck, out_dir):
    """
    Analytical cut (no new backtest) of MASTER by calendar decade:
    1990s, 2000s, 2010s, 2020s. Reports SR, CAGR, annualised vol, and
    max drawdown per decade on the excess-of-IRX return series.

    Complements the existing `master_stability.csv` 5-year buckets
    with cleaner decade boundaries.
    """
    print("\n  [T34] Calendar-decade SRs (analytical) ...")
    _ensure_robustness_dir()

    dates = pd.DatetimeIndex(master_ck["dates"])
    dr = pd.Series(master_ck["daily_ret"], index=dates).astype(float)

    if _IRX_SERIES is not None:
        rf = _IRX_SERIES.reindex(dates).fillna(0.0)
        excess = dr - rf
    else:
        excess = dr

    decades = [
        ("1990s", pd.Timestamp("1990-01-01"), pd.Timestamp("2000-01-01")),
        ("2000s", pd.Timestamp("2000-01-01"), pd.Timestamp("2010-01-01")),
        ("2010s", pd.Timestamp("2010-01-01"), pd.Timestamp("2020-01-01")),
        ("2020s", pd.Timestamp("2020-01-01"), pd.Timestamp("2030-01-01")),
    ]

    rows = []
    for name, start, end in decades:
        mask = (excess.index >= start) & (excess.index < end)
        seg = excess[mask]
        if len(seg) < 30 or seg.std(ddof=1) < 1e-12:
            rows.append(dict(
                decade=name,
                n_days=int(mask.sum()),
                sr_full=np.nan, cagr=np.nan, ann_vol=np.nan, max_dd=np.nan,
                note="insufficient data",
            ))
            continue
        sr = float(
            seg.mean() * TRADING_DAYS /
            (seg.std(ddof=1) * np.sqrt(TRADING_DAYS)))
        eq = (1.0 + seg).cumprod()
        n_years = len(seg) / TRADING_DAYS
        cagr = float(eq.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
        vol = float(seg.std(ddof=1) * np.sqrt(TRADING_DAYS))
        peak = eq.cummax()
        mdd = float(((eq - peak) / peak).min())
        rows.append(dict(
            decade=name,
            n_days=int(len(seg)),
            sr_full=round(sr, 4),
            cagr=round(cagr, 4),
            ann_vol=round(vol, 4),
            max_dd=round(mdd, 4),
            note="",
        ))

    df = pd.DataFrame(rows)
    fp = out_dir / "t34_decade_sharpe.csv"
    df.to_csv(str(fp), index=False)
    print(f"  [T34] -> {fp.name}")
    for r in rows:
        if not np.isnan(r["sr_full"]):
            print(f"  [T34] {r['decade']}: SR={r['sr_full']:>7.4f}  "
                  f"CAGR={r['cagr']*100:>6.2f}%  "
                  f"vol={r['ann_vol']*100:>6.2f}%  "
                  f"MDD={r['max_dd']*100:>7.2f}%  n={r['n_days']}")
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_extended_robustness(paths, args):
    """
    Run the full T29-T34 extended robustness battery. Requires the
    Phase-0 DM library; rebuilds it if not already built.
    """
    print("\n" + "=" * 78)
    print("  EXTENDED ROBUSTNESS BATTERY -- Strategy 183 (T29-T34)")
    print("=" * 78)
    ensure_dirs()
    _ensure_robustness_dir()

    # 1. Load MASTER
    master_ck = _load_ck_any(_abl_ck_path("MASTER"))
    if master_ck is None:
        alt = _HERE / f"{STRATEGY_NAME_MASTER}_checkpoint.pkl"
        master_ck = _load_ck_any(alt)
        if master_ck is not None:
            print(f"  [INFO] Using top-level S183_CFG checkpoint {alt.name}")
    if master_ck is None:
        print("  [WARN] No MASTER checkpoint. Run --ablation or --parsimony "
              "first (or run the S183_CFG master directly).")
        return

    # 2. IRX
    global _IRX_SERIES
    all_dates = pd.DatetimeIndex(master_ck["dates"])
    try:
        irx_arr = load_irx(all_dates)
        _IRX_SERIES = pd.Series(irx_arr, index=all_dates)
    except Exception as exc:
        _log_failure("ROBUSTNESS", "load_irx", exc)
        _IRX_SERIES = None

    # 3. Build Phase-0 library (skipped for T33/T34 which are analytical)
    print("\n[RB PHASE 0] Building DM library for robustness tests ...")
    mapping = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])
    library = dm_precompute_library(mapping, fx_daily, paths)

    # 4. Run tests in order: analytical first (fast), then backtests
    test_plan = [
        ("T33", "IRX attribution",
         lambda: t33_irx_attribution(master_ck, ANALYSIS_DIR)),
        ("T34", "Decade SRs",
         lambda: t34_decade_sharpe(master_ck, ANALYSIS_DIR)),
        ("T29", "T+1 execution lag",
         lambda: t29_t_plus_1_lag(library, master_ck, paths, ANALYSIS_DIR)),
        ("T32", "Leverage scaling",
         lambda: t32_leverage_scaling(library, master_ck, paths, ANALYSIS_DIR)),
        ("T31", "Signal noise injection",
         lambda: t31_signal_noise_injection(library, master_ck, paths,
                                             ANALYSIS_DIR,
                                             seed=int(getattr(args, "seed", 20260405)))),
        ("T30", "Instrument leave-one-out",
         lambda: t30_instrument_leave_one_out(library, master_ck, paths,
                                                ANALYSIS_DIR)),
    ]

    for name, label, fn in test_plan:
        try:
            fn()
        except Exception as exc:
            _log_failure("ROBUSTNESS", name, exc)
            print(f"  [{name}] {label} FAILED: {type(exc).__name__}: {exc}")

    print("\n[ROBUSTNESS] Extended battery complete. See "
          "Strategy_183/TestingSuite/Analysis/t29_*..t34_*.csv")


# ===========================================================================
# SECTION 1 -- CLI + Section 7/8 -- Orchestration
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Strategy 183 Full Thesis/Live Testing Suite "
                    "(Datamining + Ablation + Analysis + Parsimony + Robustness)")
    p.add_argument("--dm", action="store_true",
                    help="Run datamining sweep only")
    p.add_argument("--ablation", action="store_true",
                    help="Run ablation matrix only")
    p.add_argument("--analysis", action="store_true",
                    help="Run statistical analysis only (from checkpoints)")
    p.add_argument("--parsimony", action="store_true",
                    help="Run T23-T28 parsimony + random-parameter battery only")
    p.add_argument("--robustness", action="store_true",
                    help="Run T29-T34 extended robustness battery only "
                         "(T+1 lag, instrument leave-one-out, noise injection, "
                         "leverage scaling, IRX attribution, decade SRs)")
    p.add_argument("--all", action="store_true",
                    help="Run DM + Ablation + Analysis + Parsimony + Robustness (default)")
    p.add_argument("--skip-existing", action="store_true",
                    help="Resume from existing checkpoints")
    p.add_argument("--skip-self-test", action="store_true",
                    dest="skip_self_test",
                    help="Skip the canonical-drift self-test at analysis start")
    p.add_argument("--metrics-only", action="store_true",
                    help="Rebuild summary CSVs + analysis from checkpoints only")
    p.add_argument("--variants", nargs="+", default=None,
                    help="Restrict DM/Ablation runs to these variant IDs")
    p.add_argument("--seed", type=int, default=20260405,
                    help="Bootstrap seed (default 20260405)")
    p.add_argument("--boot-B", type=int, default=5000,
                    help="Bootstrap resample count (default 5000)")
    p.add_argument("--block", type=int, default=21,
                    help="Bootstrap block length (default 21)")
    p.add_argument("--mc-N", type=int, default=1000, dest="mc_N",
                    help="Monte Carlo sample count for parsimony tests "
                         "T24 (random parameter) and T25 (random simplex). "
                         "Default 1000: at 0 beats out of 1000, the Wilson "
                         "95%% upper-bound on the 'true rate of random "
                         "draws beating MASTER' is 0.37%%, i.e. MASTER is "
                         "in the top 0.4%% of the parameter/weight space "
                         "at 95%% confidence.  Each sample is a full 36-year "
                         "backtest so runtime scales linearly (~45s per "
                         "sample single-threaded; use --workers to parallelise).")
    p.add_argument("--workers", type=int, default=1,
                    help="Number of parallel worker processes for the DM "
                         "sweep, Ablation matrix, and T24/T25 MC loops "
                         "(default 1 = serial). Recommended: leave 1-2 "
                         "cores free, e.g. --workers 6 on an 8-core machine.")
    return p.parse_args()


def main():
    args = parse_args()

    global RT_SEED, RT_BOOT_B, RT_BLOCK
    RT_SEED   = int(args.seed)
    RT_BOOT_B = int(args.boot_B)
    RT_BLOCK  = int(args.block)

    # Default = run everything
    if not (args.dm or args.ablation or args.analysis or args.parsimony
            or args.robustness or args.all or args.metrics_only):
        args.all = True

    if args.metrics_only:
        args.dm = False
        args.ablation = False
        args.parsimony = False
        args.robustness = False
        args.analysis = True

    paths = get_project_paths(_HERE)
    ensure_dirs()

    print(f"[SUITE] Strategy 183 testing suite v{__version__}")
    print(f"[SUITE] seed={RT_SEED}  boot_B={RT_BOOT_B}  block={RT_BLOCK}")
    print(f"[SUITE] FDM_CAP (MASTER) = {S183_CFG['fdm_cap']}  (S183_CFG: uncapped by design, see master_fc pipeline note)")
    print(f"[SUITE] DM variants  : {len(DM_VARIANTS)}")
    print(f"[SUITE] ABL variants : {len(ABL_VARIANTS)}")
    print(f"[SUITE] MC sample N  : {getattr(args, 'mc_N', 1000)}  (T24/T25)")
    n_workers = int(getattr(args, "workers", 1))
    cpu_count = os.cpu_count() or 1
    print(f"[SUITE] workers      : {n_workers}  (logical CPUs available: {cpu_count})")

    run_dm    = args.dm         or args.all
    run_abl   = args.ablation   or args.all
    run_an    = args.analysis   or args.all or args.metrics_only
    run_par   = args.parsimony  or args.all
    run_rob   = args.robustness or args.all

    if run_dm and not args.metrics_only:
        try:
            run_datamining(paths, args)
        except Exception as exc:
            _log_failure("DM_MAIN", "datamining", exc)

    if run_abl and not args.metrics_only:
        try:
            run_ablation(paths, args)
        except Exception as exc:
            _log_failure("ABL_MAIN", "ablation", exc)

    if run_an:
        try:
            run_analysis(paths, args)
        except Exception as exc:
            _log_failure("ANALYSIS_MAIN", "analysis", exc)

    if run_par and not args.metrics_only:
        try:
            run_parsimony_battery(paths, args)
        except Exception as exc:
            _log_failure("PARSIMONY_MAIN", "parsimony", exc)

    if run_rob and not args.metrics_only:
        try:
            run_extended_robustness(paths, args)
        except Exception as exc:
            _log_failure("ROBUSTNESS_MAIN", "robustness", exc)

    print("\n[SUITE] Done.")


if __name__ == "__main__":
    main()
