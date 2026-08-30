"""
fast_backtest_bridge.py
=======================
Python bridge that converts the ig_shared_config inst_signals dict into the
packed matrix format expected by the fast_backtest C++ extension, runs the
simulation, and returns the results in the same format as the Python loop.

Usage (drop-in replacement for the Phase 3 loop):

    from fast_backtest_bridge import run_phase3_cpp

    result = run_phase3_cpp(
        inst_signals, all_dates, irx_arr,
        trading_start, starting_capital, dd_scaling
    )
    # result keys: equity_arr, port_pnl, port_comm, port_trades, port_cash
"""

import numpy as np

try:
    import fast_backtest as _fb
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False


def run_phase3_cpp(inst_signals, all_dates, irx_arr,
                   trading_start, starting_capital, dd_scaling=False):
    """
    Run Phase 3 (daily compounding walk-forward) using the C++ extension.

    Parameters
    ----------
    inst_signals : dict
        Keyed by instrument name. Each value must already have the aligned
        arrays: fc_arr, vol_arr, fx_arr, close_arr, open_arr, active
        (populated by Phase 2 in ig_shared_config.py).
    all_dates : pd.DatetimeIndex
        Joint OOS date universe.
    irx_arr : np.ndarray[float64, (N,)]
        Daily T-bill yield fraction.
    trading_start : pd.Timestamp
        First date to open positions.
    starting_capital : float
        Initial NAV.
    dd_scaling : bool
        Enable drawdown scaling.

    Returns
    -------
    dict with keys:
        equity_arr  : np.ndarray[float64, (N,)]
        port_pnl    : np.ndarray[float64, (N,)]
        port_comm   : np.ndarray[float64, (N,)]
        port_trades : np.ndarray[int32,   (N,)]
        port_cash   : np.ndarray[float64, (N,)]
    """
    if not CPP_AVAILABLE:
        raise ImportError(
            "fast_backtest C++ extension not available. "
            "Build it with: python setup_fast_backtest.py build_ext --inplace"
        )

    N = len(all_dates)
    instruments = list(inst_signals.keys())
    n_inst = len(instruments)

    # ── Pack per-instrument arrays into (n_inst, N) matrices ──────────────
    fc_matrix     = np.empty((n_inst, N), dtype=np.float64)
    vol_matrix    = np.empty((n_inst, N), dtype=np.float64)
    fx_matrix     = np.empty((n_inst, N), dtype=np.float64)
    close_matrix  = np.empty((n_inst, N), dtype=np.float64)
    open_matrix   = np.empty((n_inst, N), dtype=np.float64)
    active_matrix = np.empty((n_inst, N), dtype=np.uint8)
    pointsizes    = np.empty(n_inst, dtype=np.float64)
    costs         = np.empty(n_inst, dtype=np.float64)

    for j, inst in enumerate(instruments):
        sig = inst_signals[inst]
        fc_matrix[j]     = np.nan_to_num(sig["fc_arr"], nan=np.nan)
        vol_matrix[j]    = sig["vol_arr"]
        fx_matrix[j]     = sig["fx_arr"]
        close_matrix[j]  = sig["close_arr"]
        open_matrix[j]   = sig["open_arr"]
        active_matrix[j] = sig["active"].astype(np.uint8)
        pointsizes[j]    = sig["pointsize"]
        costs[j]         = sig["cost_rt"]

    # ── Ensure C-contiguous arrays (pybind11 unchecked<> requires it) ─────
    fc_matrix     = np.ascontiguousarray(fc_matrix)
    vol_matrix    = np.ascontiguousarray(vol_matrix)
    fx_matrix     = np.ascontiguousarray(fx_matrix)
    close_matrix  = np.ascontiguousarray(close_matrix)
    open_matrix   = np.ascontiguousarray(open_matrix)
    active_matrix = np.ascontiguousarray(active_matrix)
    irx_arr       = np.ascontiguousarray(irx_arr, dtype=np.float64)

    # ── Compute trading_start_idx ─────────────────────────────────────────
    trading_start_idx = int(np.searchsorted(all_dates, trading_start))

    # ── Call C++ ──────────────────────────────────────────────────────────
    result = _fb.run_simulation(
        N, n_inst, float(starting_capital),
        trading_start_idx, bool(dd_scaling),
        irx_arr,
        fc_matrix, vol_matrix, fx_matrix,
        close_matrix, open_matrix, active_matrix,
        pointsizes, costs
    )

    return {
        "equity_arr":  result["equity"],
        "port_pnl":    result["pnl"],
        "port_comm":   result["commission"],
        "port_trades": result["trades"],
        "port_cash":   result["cash_yield"],
    }
