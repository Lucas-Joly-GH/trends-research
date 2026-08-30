"""
IG_Backtest / ig_shared_config.py
=============================================================================
Institutional-Grade (IG) compounding infrastructure shared by all
ig_strategy_XX.py files.

Re-exports everything from the project's shared_config.py and adds:
  - Daily-compounding NAV engine  (run_compounded_portfolio)
  - IRX risk-free cash yield      (load_irx)
  - Geometric metrics             (CAGR, compounded drawdown, Sharpe vs rf)

ARCHITECTURE (mirrors Strategy 75):
  Phase 1 — Pre-compute all per-instrument signals.
             Each strategy file owns this phase (signal math untouched).
  Phase 2 — Build joint OOS date universe & align arrays.
  Phase 3 — Joint walk-forward: compound equity daily, dynamic weight.
  Phase 4 — Geometric metrics, checkpoint, CSV export.

USAGE IN EACH STRATEGY FILE (base universe):
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parent))
  from ig_shared_config import (
      *,
      IDM_IG, OOS_START_IG, TRADING_START,
      run_compounded_portfolio,
  )
  ...
  # Phase 1: build inst_signals dict (signal math untouched from original)
  ...
  run_compounded_portfolio(inst_signals, STRATEGY_NAME, paths)

EXTENDED UNIVERSE (opt-in — add synthetic spreads/butterflies):
  from ig_shared_config import (
      load_extended_mapping,           # returns (mapping_73, synth_panama_dir)
      load_instrument_data_extended,   # auto-routes base vs synthetic
  )
  mapping, synth_dir = load_extended_mapping(paths)
  for instrument in mapping.index:
      data = load_instrument_data_extended(instrument, paths, synth_dir)
      ...

  The base universe (63 instruments) is loaded by default via load_mapping().
  Synthetics live in Data/Synthetic/ and are NEVER included unless the strategy
  explicitly calls load_extended_mapping().

inst_signals FORMAT:
  {
    instrument: {
      "oos_date_set" : set of pd.Timestamp  — dates this instrument is OOS,
      "forecast"     : pd.Series(float)     — final blended forecast [-20,+20],
      "vol"          : pd.Series(float)     — blended price-change volatility,
      "fx"           : pd.Series(float)     — USD-per-local-currency FX rate,
      "close"        : pd.Series(float)     — Panama-adjusted close,
      "open"         : pd.Series(float)     — Panama-adjusted open,
      "pointsize"    : float,
      "cost_rt"      : float,
    }
  }
"""

import sys
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Locate and import shared_config.py from the project root ─────────────────
_HERE = Path(__file__).resolve().parent   # IG_Backtest/

def _find_shared_config():
    """Walk up the directory tree (and siblings at each level) for shared_config.py."""
    for parent in [_HERE] + list(_HERE.parents):
        if (parent / "shared_config.py").exists():
            return parent
        # Also check immediate subdirectories (handles sibling project folders)
        try:
            for sibling in parent.iterdir():
                if sibling.is_dir() and (sibling / "shared_config.py").exists():
                    return sibling
        except PermissionError:
            continue
    raise FileNotFoundError(
        "Cannot locate shared_config.py — ensure IG_Backtest is reachable from "
        "the Robert Carver project directory."
    )

_PROJECT_ROOT = _find_shared_config()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared_config import *   # noqa: F401, F403  — re-export everything

# ── IG-specific constants ────────────────────────────────────────────────────
IDM_IG        = 3.0                         # Legacy fixed IDM — kept for reference only.
                                             # run_compounded_portfolio now uses idm_from_n_active.
TRADING_START = pd.Timestamp("1990-01-01")  # No positions opened before this date;
                                             # equity earns cash yield from simulation start.
OOS_START_IG  = max(512, SCALAR_WINDOW)     # Default warmup bars per instrument
                                             # (512 = EWMAC-128 slow span; override per-strategy
                                             # for shorter lookback signals).

# Drawdown scaling constants (used by S77+ when dd_scaling=True).
DD_SCALE_THRESHOLD  = 0.05    # 5% drawdown from peak → scaling begins
DD_SCALE_FLOOR      = 0.50    # Minimum scale factor at deep drawdown
DD_SCALE_FULL_AT    = 0.10    # 10% drawdown → floor reached

# Carver IDM lookup table (Systematic Trading, Table 7.8).
# Linear interpolation between breakpoints; capped at 4.0 above n=200.
_IDM_N   = np.array([1,   2,    3,    4,    5,    6,    7,    8,
                      9,   10,   15,   20,   25,   30,   40,   50,
                      60,  70,  100,  200], dtype=float)
_IDM_VAL = np.array([1.0, 1.20, 1.48, 1.56, 1.70, 1.90, 1.98, 2.11,
                      2.18, 2.24, 2.53, 2.87, 3.01, 3.12, 3.27, 3.39,
                      3.47, 3.54, 3.75, 4.0])


def idm_from_n_active(n: int) -> float:
    """Return the Carver IDM for n simultaneously active instruments.

    Uses linear interpolation between the published breakpoints and caps at 4.0.
    Guarantees a floor of 1.0 (single-instrument case).
    """
    if n <= 1:
        return 1.0
    return float(np.interp(n, _IDM_N, _IDM_VAL))


# ── Locate %IRX cash-yield file ──────────────────────────────────────────────
def _find_irx_path():
    for parent in [_HERE] + list(_HERE.parents):
        candidate = parent / "Data" / "Cash-Yield" / "%IRX.csv"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Cannot locate Data/Cash-Yield/%%IRX.csv — ensure the Data folder is "
        "accessible from IG_Backtest."
    )

IRX_PATH = _find_irx_path()


def load_irx(all_dates):
    """
    Load the 13-week T-Bill annualised yield (%IRX, in percent) and return a
    daily-fraction array aligned to all_dates.

      IRX = 4.52  →  daily_fraction = (4.52 / 100) / 256 = 0.000177

    T-bills are used as futures margin collateral, so the full NAV earns
    this yield every trading day.
    """
    def _read_irx(path):
        df = pd.read_csv(path, parse_dates=["Date"], date_format="%Y%m%d")
        return df.set_index("Date")[["Close"]]

    irx_df = parquet_cache(str(IRX_PATH), _read_irx, cache_tag="_irx")
    irx_raw = irx_df["Close"]
    irx_daily = (irx_raw / 100.0) / 256.0          # daily fraction
    return irx_daily.reindex(all_dates).ffill().fillna(0.0).values


# ── Extended universe: opt-in synthetic instruments ─────────────────────────
#
# By default every backtest runs on the BASE universe (63 instruments from
# Général/instrument_mapping.csv, prices in Data/PanamaMethod/).
#
# Calling load_extended_mapping() / load_synthetic_instrument_data() adds the
# synthetic spread/butterfly contracts from Data/Synthetic/.  Strategies must
# explicitly opt in; nothing changes for strategies that don't.
#
# Usage in a strategy that wants synthetics:
#
#   paths = get_project_paths(...)
#   mapping_base = load_mapping(paths["mapping"])
#   mapping_ext, synth_panama_dir = load_extended_mapping(paths)
#   # mapping_ext = 63 base + 10 synthetic instruments
#   # For base instruments:  load_instrument_data(inst, paths["stats"], paths["panama"])
#   # For synthetic instruments: load_instrument_data(inst, paths["stats"], synth_panama_dir)
#   # Or use the convenience wrapper:
#   data = load_instrument_data_extended(inst, paths, synth_panama_dir)

def _find_synthetic_dir():
    """Locate Data/Synthetic/ relative to the project root."""
    for parent in [_HERE] + list(_HERE.parents):
        candidate = parent / "Data" / "Synthetic"
        if candidate.exists() and (candidate / "synthetic_mapping.csv").exists():
            return candidate
    return None

_SYNTHETIC_DIR = _find_synthetic_dir()


def load_extended_mapping(paths):
    """
    Load the base mapping (63 instruments) + synthetic mapping (10 instruments).

    Returns
    -------
    (mapping_extended, synth_panama_dir)
        mapping_extended : pd.DataFrame  — concatenation of base + synthetic
        synth_panama_dir : str           — path to Data/Synthetic/ (use as
                                           panama_folder for synthetic instruments)
    Raises FileNotFoundError if the synthetic mapping is not found.
    """
    base = load_mapping(paths["mapping"])

    if _SYNTHETIC_DIR is None:
        raise FileNotFoundError(
            "Data/Synthetic/synthetic_mapping.csv not found. "
            "Run build_synthetic.py first."
        )

    synth = pd.read_csv(_SYNTHETIC_DIR / "synthetic_mapping.csv").set_index("norgate_code")
    print(f"Loaded synthetic mapping: {len(synth)} instruments from Data/Synthetic/")

    # Ensure no column mismatches (synthetics may lack some optional columns)
    for col in base.columns:
        if col not in synth.columns:
            synth[col] = np.nan
    synth = synth.reindex(columns=base.columns)

    extended = pd.concat([base, synth])
    print(f"Extended universe: {len(extended)} instruments "
          f"({len(base)} base + {len(synth)} synthetic)")
    return extended, str(_SYNTHETIC_DIR)


def load_instrument_data_extended(instrument, paths, synth_panama_dir=None):
    """
    Load instrument data, automatically routing to the correct panama folder.

    For base instruments (present in Data/PanamaMethod/):
        Loads from paths["panama"] as usual.
    For synthetic instruments (NOT in Data/PanamaMethod/):
        Loads from synth_panama_dir (Data/Synthetic/).

    Parameters
    ----------
    instrument : str
        Norgate code (e.g. "ES" or "CL_BRN").
    paths : dict
        From get_project_paths().
    synth_panama_dir : str or None
        Path to Data/Synthetic/.  If None, only base instruments are loadable.

    Returns
    -------
    dict or None — same format as load_instrument_data().
    """
    # Try the base universe first
    data = load_instrument_data(instrument, paths["stats"], paths["panama"])
    if data is not None:
        return data

    # Fall back to synthetic folder
    if synth_panama_dir is not None:
        data = load_instrument_data(instrument, paths["stats"], synth_panama_dir)
        return data

    return None


# ── GAP 2: Realized Variance Scaling — Barroso & Santa-Clara (2013) ──────────
#    Source: Momentum_has_its_moments.pdf, Eqs. 1-6, Table 1.
#    Key finding: momentum variance is highly predictable (AR(1) R²=60.38%).
#    Scaling positions by target_vol / realized_vol nearly doubles SR (0.53→0.97)
#    and reduces worst monthly return from -78.96% to -28.40%.

def compute_realized_variance_scale(
    daily_returns,
    target_vol_ann=0.12,
    rv_window=126,
    scale_floor=0.25,
    scale_cap=4.0,
):
    """
    Realized-variance position scaling per Barroso & Santa-Clara (2013).

    Formula:
        RV_t = sum_{i=1}^{rv_window} r_{t-i}^2    (Eq. 1)
        w_t  = sigma_target / sqrt(RV_t * annualise)

    Source: Momentum_has_its_moments.pdf, Eqs. 1-6, Table 1.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily return series of the alpha or instrument.
    target_vol_ann : float
        Target annualized volatility (paper uses 12%).
    rv_window : int
        Days for realized variance (126 = ~6 months).
    scale_floor : float
        Minimum scale factor.
    scale_cap : float
        Maximum scale factor.

    Returns
    -------
    pd.Series of daily multiplicative scale factors.
    """
    r = daily_returns.fillna(0.0)
    rv = (r ** 2).rolling(rv_window, min_periods=max(rv_window // 2, 10)).sum()
    realized_vol_ann = np.sqrt(rv * (256.0 / rv_window))
    scale = target_vol_ann / realized_vol_ann.replace(0, np.nan)
    scale = scale.clip(scale_floor, scale_cap).fillna(1.0)
    return scale


# ── GAP 6: AR(1) Variance Forecast — Barroso & Santa-Clara (2013) ───────────
#    Source: Momentum_has_its_moments.pdf, Table 2.
#    AR(1) on realized variance: in-sample R²=60.38%, OOS R²=57.82%.
#    Momentum variance is MORE predictable than market variance (41.31%).

def compute_ar1_variance_forecast(
    daily_returns,
    rv_window=126,
    ar1_lookback=504,
    blend_with_current=0.5,
):
    """
    AR(1) forecast of next-period realized variance.

    Model: RV_{t+1} = alpha + beta * RV_t + epsilon_t
    Walk-forward: retrain every month on rolling window.

    Source: Momentum_has_its_moments.pdf, Table 2 (AR(1) results).

    Parameters
    ----------
    daily_returns : pd.Series
        Daily return series.
    rv_window : int
        Window for RV computation (126 = 6 months).
    ar1_lookback : int
        Rolling training window for AR(1) regression (504 = 2 years).
    blend_with_current : float
        Blend weight with current RV (0.5 = 50% forecast + 50% current).

    Returns
    -------
    pd.Series of AR(1)-predicted annualized volatility (blended).
    """
    r = daily_returns.fillna(0.0)
    rv = (r ** 2).rolling(rv_window, min_periods=max(rv_window // 2, 10)).sum()
    rv_vals = rv.values
    N = len(rv_vals)
    rv_forecast = np.full(N, np.nan)

    refit_every = 21  # monthly refit
    alpha_coeff, beta_coeff = 0.0, 1.0  # initialise as identity (forecast = current)

    for t in range(ar1_lookback, N - 1):
        # Refit AR(1) monthly
        if t == ar1_lookback or (t - ar1_lookback) % refit_every == 0:
            window = rv_vals[max(0, t - ar1_lookback):t]
            valid = ~np.isnan(window)
            if valid.sum() > 20:
                X = window[valid][:-1]
                y = window[valid][1:]
                if len(X) >= 20 and np.std(X) > 1e-12:
                    x_mean = X.mean()
                    y_mean = y.mean()
                    beta_coeff = np.sum((X - x_mean) * (y - y_mean)) / max(np.sum((X - x_mean) ** 2), 1e-12)
                    alpha_coeff = y_mean - beta_coeff * x_mean

        current_rv = rv_vals[t]
        if not np.isnan(current_rv):
            predicted = alpha_coeff + beta_coeff * current_rv
            rv_forecast[t] = max(predicted, 0.0)
        else:
            rv_forecast[t] = current_rv

    # Annualized volatility from forecast RV
    forecast_vol = np.sqrt(np.maximum(rv_forecast, 0) * (256.0 / rv_window))
    current_vol  = np.sqrt(np.maximum(rv_vals, 0) * (256.0 / rv_window))

    # Blend: forecast + current (conservative)
    blended = np.where(
        np.isnan(forecast_vol),
        current_vol,
        blend_with_current * forecast_vol + (1.0 - blend_with_current) * current_vol,
    )
    return pd.Series(blended, index=daily_returns.index).ffill().fillna(0.0)


# ── Core compounding runner ──────────────────────────────────────────────────
def run_compounded_portfolio(
    inst_signals,
    strategy_name,
    paths,
    idm=IDM_IG,
    trading_start=TRADING_START,
    starting_capital=STARTING_CAPITAL,
    dd_scaling=False,
    save_per_inst_pnl=False,
):
    """
    Phase 2-3-4: Joint walk-forward with daily NAV compounding.

    Identical to the Phase 2-3-4 architecture of Strategy 75 (Compounded
    Trinity CTA), parameterised so that any strategy's inst_signals dict can
    be passed in.

    Parameters
    ----------
    inst_signals : dict
        See module docstring for format.
    strategy_name : str
        Used for checkpoint/CSV filenames.
    paths : dict
        From get_project_paths(); "output" key is used for saved files.
    idm : float
        Instrument Diversification Multiplier (default IDM_IG = 3.0).
    trading_start : pd.Timestamp
        Positions are only opened from this date onward.
    starting_capital : float
        Initial portfolio NAV in USD.

    Returns
    -------
    dict  — checkpoint dict with keys:
        dates, pnl, equity, daily_ret, ann_ret, ann_vol, sr,
        cagr, max_dd, calmar, total_comm, total_cash_yield, trades_yr, n_years
    None  — if inst_signals is empty.
    """
    from tqdm import tqdm

    if not inst_signals:
        print("[ERROR] No instruments loaded — cannot run portfolio.")
        return None

    # =========================================================================
    # PHASE 2 — Build joint date universe & align arrays
    # =========================================================================
    print(f"\n[PHASE 2] Building joint date universe ({len(inst_signals)} instruments)...")

    all_dates = pd.DatetimeIndex(sorted(
        {d for sig in inst_signals.values() for d in sig["oos_date_set"]}
    ))
    N = len(all_dates)
    date_to_i = {d: i for i, d in enumerate(all_dates)}

    print(f"  OOS period : {all_dates[0].date()} to {all_dates[-1].date()} ({N} days)")
    print(f"  Trading    : {trading_start.date()} to {all_dates[-1].date()}")
    print(f"  IDM        : dynamic (Carver lookup by n_active)")

    irx_arr = load_irx(all_dates)   # daily fraction of NAV earned from T-bills

    # Pre-align each instrument's Series to the joint calendar.
    # reindex fills NaN for dates outside the instrument's own data range.
    for sig in inst_signals.values():
        sig["fc_arr"]    = sig["forecast"].reindex(all_dates).values
        sig["vol_arr"]   = sig["vol"].reindex(all_dates).values
        sig["fx_arr"]    = sig["fx"].reindex(all_dates).values
        sig["close_arr"] = sig["close"].reindex(all_dates).values
        sig["open_arr"]  = sig["open"].reindex(all_dates).values

        # Boolean mask: True on dates the instrument is in-sample for this OOS run
        active = np.zeros(N, dtype=bool)
        for d in sig["oos_date_set"]:
            if d in date_to_i:
                active[date_to_i[d]] = True
        sig["active"] = active

    # =========================================================================
    # PHASE 3 — Joint walk-forward with daily NAV compounding
    # =========================================================================

    # ── Try C++ accelerated path first, fall back to Python ───────────────
    try:
        from fast_backtest_bridge import run_phase3_cpp, CPP_AVAILABLE
    except ImportError:
        CPP_AVAILABLE = False

    # If caller wants per-instrument PnL panel, force Python path — the C++
    # bridge only surfaces aggregated portfolio PnL.
    if save_per_inst_pnl:
        CPP_AVAILABLE = False
        print("[INFO] save_per_inst_pnl=True -> forcing Python fallback")

    if CPP_AVAILABLE:
        print("\n[PHASE 3] Running compounded simulation (C++ accelerated)...")
        _cpp = run_phase3_cpp(
            inst_signals, all_dates, irx_arr,
            trading_start, starting_capital, dd_scaling,
        )
        equity_arr  = _cpp["equity_arr"]
        port_pnl    = _cpp["port_pnl"]
        port_comm   = _cpp["port_comm"]
        port_trades = _cpp["port_trades"]
        port_cash   = _cpp["port_cash"]
        # C++ sets pre-trading equity to starting_capital (0.0 at index 0 is
        # overwritten); replace 0.0 entries before trading with NaN so Phase 4
        # valid-mask works identically to the Python path.
        equity_arr[0] = starting_capital
        mask_pretrade = equity_arr == 0.0
        equity_arr[mask_pretrade] = np.nan
        equity = float(equity_arr[~np.isnan(equity_arr)][-1])
    else:
        print("\n[PHASE 3] Running compounded simulation (Python fallback)...")

        equity        = float(starting_capital)
        peak_equity   = float(starting_capital)   # for dd_scaling
        current_pos   = {inst: 0.0 for inst in inst_signals}

        port_pnl    = np.zeros(N)
        port_comm   = np.zeros(N)
        port_trades = np.zeros(N, dtype=int)
        port_cash   = np.zeros(N)          # T-bill cash yield credited each day
        equity_arr  = np.full(N, np.nan)
        equity_arr[0] = starting_capital

        # Per-instrument PnL panel (optional, enables asset-class decomposition)
        if save_per_inst_pnl:
            inst_list       = list(inst_signals.keys())
            inst_idx        = {nm: k for k, nm in enumerate(inst_list)}
            per_inst_pnl_mx = np.zeros((N, len(inst_list)))
        else:
            per_inst_pnl_mx = None

        for i in tqdm(range(N - 1), desc="Joint Simulation"):

            # ── Before TRADING_START: hold cash, no positions ─────────────
            if all_dates[i] < trading_start:
                equity_arr[i + 1] = equity
                continue

            # ── Cash yield: entire NAV earns T-bill rate every day ────────
            day_cash = irx_arr[i] * equity
            day_pnl  = day_cash
            port_cash[i + 1] = day_cash
            day_comm   = 0.0
            day_trades = 0

            # ── Dynamic weight: 1/n_active ────────────────────────────────
            n_active = sum(
                1 for s in inst_signals.values()
                if (s["active"][i]
                    and not np.isnan(s["fc_arr"][i])
                    and s["fc_arr"][i] != 0.0
                    and not np.isnan(s["vol_arr"][i])
                    and s["vol_arr"][i] >= 1e-10
                    and not np.isnan(s["fx_arr"][i]))
            )
            dynamic_weight = 1.0 / max(n_active, 1)

            # ── Drawdown scaling (opt-in via dd_scaling=True) ─────────────
            if dd_scaling:
                peak_equity = max(peak_equity, equity)
                dd_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                if dd_pct < DD_SCALE_THRESHOLD:
                    dd_scale = 1.0
                elif dd_pct >= DD_SCALE_FULL_AT:
                    dd_scale = DD_SCALE_FLOOR
                else:
                    dd_scale = 1.0 - (1.0 - DD_SCALE_FLOOR) * (
                        (dd_pct - DD_SCALE_THRESHOLD) / (DD_SCALE_FULL_AT - DD_SCALE_THRESHOLD)
                    )
            else:
                dd_scale = 1.0

            # ── Per-instrument P&L ────────────────────────────────────────
            for instrument, sig in inst_signals.items():
                if not sig["active"][i]:
                    continue

                f_t  = sig["fc_arr"][i]
                v_t  = sig["vol_arr"][i]
                fx_t = sig["fx_arr"][i]
                ps   = sig["pointsize"]
                cost = sig["cost_rt"]
                cur  = current_pos[instrument]

                if (np.isnan(f_t) or np.isnan(v_t) or np.isnan(fx_t)
                        or v_t < 1e-10 or f_t == 0.0):
                    new_pos = 0.0
                else:
                    ann_vol_usd = v_t * ps * ANNUALISE_DAILY * fx_t
                    if ann_vol_usd < 1.0:
                        new_pos = cur
                    else:
                        risk_budget = equity * VOL_TARGET * dynamic_weight * idm_from_n_active(n_active) * dd_scale
                        target      = (f_t / FORECAST_TARGET) * (risk_budget / ann_vol_usd)
                        new_pos     = buffered_position(target, cur, compute_buffer(target))

                current_pos[instrument] = new_pos

                close_t  = sig["close_arr"][i]
                open_t1  = sig["open_arr"][i + 1]
                close_t1 = sig["close_arr"][i + 1]
                fx_t1    = sig["fx_arr"][i + 1]

                if (np.isnan(close_t) or np.isnan(open_t1)
                        or np.isnan(close_t1) or np.isnan(fx_t1)):
                    continue

                pnl, comm, traded = compute_pnl_day(
                    cur, new_pos, close_t, open_t1, close_t1, ps, fx_t1, cost)

                day_pnl    += pnl
                day_comm   += comm
                day_trades += traded

                if per_inst_pnl_mx is not None:
                    per_inst_pnl_mx[i + 1, inst_idx[instrument]] = pnl

            # ── Book P&L at i+1 and compound equity ──────────────────────
            port_pnl[i + 1]    = day_pnl
            port_comm[i + 1]   = day_comm
            port_trades[i + 1] = day_trades

            equity            += day_pnl
            equity_arr[i + 1]  = equity

    # =========================================================================
    # PHASE 4 — Geometric metrics, checkpoint, CSV export
    # =========================================================================
    print("\n[PHASE 4] Computing metrics...")

    valid        = ~np.isnan(equity_arr)
    dates_all_v  = all_dates[valid]
    equity_all_v = equity_arr[valid]
    pnl_all_v    = port_pnl[valid]
    cash_all_v   = port_cash[valid]

    # Trim to TRADING_START for metric computation only.
    # Pre-trading dates have zero PnL and would dilute return / vol stats.
    trade_idx = np.searchsorted(dates_all_v, trading_start)
    dates_v   = dates_all_v[trade_idx:]
    equity_v  = equity_all_v[trade_idx:]
    pnl_v     = pnl_all_v[trade_idx:]
    cash_v    = cash_all_v[trade_idx:]

    # Daily geometric returns: pnl_t / equity_{t-1}
    eq_lag     = np.empty_like(equity_v)
    eq_lag[0]  = equity_v[0]       # equity at TRADING_START = STARTING_CAPITAL
    eq_lag[1:] = equity_v[:-1]
    daily_ret  = np.where(eq_lag > 0, pnl_v / eq_lag, 0.0)

    # Risk-free component (T-bill yield included in pnl_v; subtract for Sharpe)
    rf_daily   = np.where(eq_lag > 0, cash_v / eq_lag, 0.0)
    excess_ret = daily_ret - rf_daily

    ann_ret  = daily_ret.mean() * 256
    ann_rf   = rf_daily.mean()  * 256            # annualised T-bill yield
    ann_vol  = daily_ret.std()  * ANNUALISE_DAILY
    sr       = (excess_ret.mean() * 256 / ann_vol) if ann_vol > 0 else 0.0

    n_years   = len(daily_ret) / 256
    total_ret = equity_v[-1] / starting_capital   # gross multiple
    cagr      = (total_ret ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0

    # Max drawdown on compounded NAV (percentage from rolling peak)
    peak   = np.maximum.accumulate(equity_v)
    dd     = (peak - equity_v) / peak
    max_dd = dd.max()
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0

    total_comm       = float(port_comm.sum())
    total_trades     = int(port_trades.sum())
    trades_yr        = total_trades / n_years if n_years > 0 else 0.0
    total_cash_yield = float(port_cash[trade_idx:].sum())

    # ── Print tearsheet ────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  {strategy_name} [IG Compounded]")
    print(f"{'='*62}")
    print(f"  Period       : {dates_v[0].date()} to {dates_v[-1].date()} ({n_years:.1f} yrs)")
    print(f"  Start NAV    : ${starting_capital:>15,.0f}")
    print(f"  End NAV      : ${equity_v[-1]:>15,.0f}")
    print(f"  Total Return : {total_ret - 1:>13.1%}  (geometric)")
    print(f"  CAGR         : {cagr:>13.2%}")
    print(f"  Ann. Return  : {ann_ret:>13.2%}  (arithmetic)")
    print(f"  Risk-Free    : {ann_rf:>13.2%}  (avg T-bill)")
    print(f"  Volatility   : {ann_vol:>13.2%}")
    print(f"  Sharpe       : {sr:>13.2f}  (excess return vs rf)")
    print(f"  Calmar       : {calmar:>13.2f}")
    print(f"  Max Drawdown : {max_dd:>13.2%}  (from rolling peak)")
    print(f"  Trades/yr    : {trades_yr:>13,.0f}")
    print(f"  Total Costs  : ${total_comm:>14,.0f}")
    print(f"  Cash Yield   : ${total_cash_yield:>14,.0f}  (T-bill, compounded)")
    print(f"{'='*62}")

    # ── Save checkpoint & CSV ──────────────────────────────────────────────
    os.makedirs(paths["output"], exist_ok=True)

    checkpoint = {
        "dates":            dates_v,
        "pnl":              pnl_v,
        "equity":           equity_v,
        "daily_ret":        daily_ret,
        "ann_ret":          ann_ret,
        "ann_vol":          ann_vol,
        "sr":               sr,
        "cagr":             cagr,
        "max_dd":           max_dd,
        "calmar":           calmar,
        "total_comm":       total_comm,
        "total_cash_yield": total_cash_yield,
        "trades_yr":        trades_yr,
        "n_years":          n_years,
    }

    # Optional per-instrument PnL panel (Python path only, for asset-class decomposition)
    if save_per_inst_pnl and 'per_inst_pnl_mx' in locals() and per_inst_pnl_mx is not None:
        per_inst_df = pd.DataFrame(
            per_inst_pnl_mx[valid][trade_idx:],
            index=dates_v,
            columns=inst_list,
        )
        checkpoint["per_inst_pnl"] = per_inst_df

    ckpt_file = os.path.join(paths["output"], f"{strategy_name}_checkpoint.pkl")
    with open(ckpt_file, "wb") as f:
        pickle.dump(checkpoint, f)

    pd.DataFrame({
        "Date":         dates_all_v,
        "NAV":          equity_all_v,
        "Daily_PnL":    pnl_all_v,
        "Daily_Return": np.concatenate([np.zeros(trade_idx), daily_ret]),
    }).to_csv(
        os.path.join(paths["output"], f"{strategy_name}_portfolio_returns.csv"),
        index=False,
    )

    print(f"\n[INFO] Checkpoint : {ckpt_file}")
    print(f"[INFO] CSV        : {strategy_name}_portfolio_returns.csv")
    return checkpoint
