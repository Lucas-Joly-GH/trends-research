/*
 * fast_backtest.cpp
 * =================
 * pybind11 C++ extension for the IG_Backtest daily compounding loop.
 *
 * Translates the Phase 3 walk-forward from ig_shared_config.py into C++.
 * Accepts pre-computed NumPy arrays (from Phase 2) and returns equity/PnL
 * arrays back to Python, eliminating the Python for-loop bottleneck.
 *
 * Build:
 *   pip install pybind11
 *   python setup_fast_backtest.py build_ext --inplace
 *
 * Usage:
 *   import fast_backtest
 *   result = fast_backtest.run_simulation(...)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace py = pybind11;

// ═══════════════════════════════════════════════════════════════════════════
// Constants — mirrored from shared_config.py
// ═══════════════════════════════════════════════════════════════════════════
static constexpr double VOL_TARGET       = 0.20;
static constexpr double FORECAST_TARGET  = 10.0;
static constexpr double ANNUALISE_DAILY  = 16.0;  // sqrt(256)
static constexpr double BUFFER_FRACTION  = 0.10;
static constexpr double MIN_BUFFER       = 0.5;

// Drawdown scaling constants
static constexpr double DD_SCALE_THRESHOLD = 0.05;
static constexpr double DD_SCALE_FLOOR     = 0.50;
static constexpr double DD_SCALE_FULL_AT   = 0.10;

// ═══════════════════════════════════════════════════════════════════════════
// Carver IDM lookup table (Systematic Trading, Table 7.8)
// ═══════════════════════════════════════════════════════════════════════════
static const double IDM_N[]   = {1,  2,   3,   4,   5,   6,   7,   8,
                                  9,  10,  15,  20,  25,  30,  40,  50,
                                  60, 70, 100, 200};
static const double IDM_VAL[] = {1.0, 1.20, 1.48, 1.56, 1.70, 1.90, 1.98, 2.11,
                                  2.18, 2.24, 2.53, 2.87, 3.01, 3.12, 3.27, 3.39,
                                  3.47, 3.54, 3.75, 4.0};
static constexpr int IDM_LEN = 20;

static double idm_from_n_active(int n) {
    if (n <= 1) return 1.0;
    double x = static_cast<double>(n);
    // Linear interpolation (same as np.interp)
    if (x <= IDM_N[0]) return IDM_VAL[0];
    if (x >= IDM_N[IDM_LEN - 1]) return IDM_VAL[IDM_LEN - 1];
    for (int k = 0; k < IDM_LEN - 1; ++k) {
        if (x >= IDM_N[k] && x <= IDM_N[k + 1]) {
            double t = (x - IDM_N[k]) / (IDM_N[k + 1] - IDM_N[k]);
            return IDM_VAL[k] + t * (IDM_VAL[k + 1] - IDM_VAL[k]);
        }
    }
    return IDM_VAL[IDM_LEN - 1];  // unreachable, but safe fallback
}

// ═══════════════════════════════════════════════════════════════════════════
// Carver buffering
// ═══════════════════════════════════════════════════════════════════════════
static inline double compute_buffer(double target) {
    double buf = std::abs(target) * BUFFER_FRACTION;
    return buf > MIN_BUFFER ? buf : MIN_BUFFER;
}

static inline double buffered_position(double target, double current, double buffer) {
    if (std::abs(target - current) > buffer) {
        return std::round(target);
    }
    return current;
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-instrument struct — holds raw pointers to the NumPy data
// ═══════════════════════════════════════════════════════════════════════════
struct InstrumentData {
    const double* fc;       // forecast array [N]
    const double* vol;      // vol array [N]
    const double* fx;       // fx array [N]
    const double* close;    // close array [N]
    const double* open;     // open array [N]
    const uint8_t* active;  // active mask [N]
    double pointsize;
    double cost_rt;
};

// ═══════════════════════════════════════════════════════════════════════════
// Main simulation loop
// ═══════════════════════════════════════════════════════════════════════════

/*
 * run_simulation(
 *     N              : int          — number of trading days
 *     n_instruments   : int          — number of instruments
 *     starting_capital: double       — initial NAV
 *     trading_start_idx: int         — index of first trading day (skip before)
 *     dd_scaling      : bool         — enable drawdown scaling
 *     irx_arr         : np.ndarray[float64, N]   — daily T-bill fraction
 *     fc_matrix       : np.ndarray[float64, n_inst x N]  — forecasts
 *     vol_matrix      : np.ndarray[float64, n_inst x N]  — volatilities
 *     fx_matrix       : np.ndarray[float64, n_inst x N]  — FX rates
 *     close_matrix    : np.ndarray[float64, n_inst x N]  — close prices
 *     open_matrix     : np.ndarray[float64, n_inst x N]  — open prices
 *     active_matrix   : np.ndarray[uint8,   n_inst x N]  — active masks
 *     pointsizes      : np.ndarray[float64, n_inst]      — point sizes
 *     costs           : np.ndarray[float64, n_inst]      — round-trip costs
 * )
 *
 * Returns: dict with keys "equity", "pnl", "commission", "trades", "cash_yield"
 *          each a np.ndarray[float64, N] (or int32 for trades).
 */
py::dict run_simulation(
    int N,
    int n_instruments,
    double starting_capital,
    int trading_start_idx,
    bool dd_scaling,
    py::array_t<double> irx_arr_py,
    py::array_t<double> fc_matrix_py,
    py::array_t<double> vol_matrix_py,
    py::array_t<double> fx_matrix_py,
    py::array_t<double> close_matrix_py,
    py::array_t<double> open_matrix_py,
    py::array_t<uint8_t> active_matrix_py,
    py::array_t<double> pointsizes_py,
    py::array_t<double> costs_py
) {
    // ── Unchecked access for speed (we trust Phase 2 alignment) ──────────
    auto irx        = irx_arr_py.unchecked<1>();
    auto fc_mat     = fc_matrix_py.unchecked<2>();
    auto vol_mat    = vol_matrix_py.unchecked<2>();
    auto fx_mat     = fx_matrix_py.unchecked<2>();
    auto close_mat  = close_matrix_py.unchecked<2>();
    auto open_mat   = open_matrix_py.unchecked<2>();
    auto active_mat = active_matrix_py.unchecked<2>();
    auto ps_arr     = pointsizes_py.unchecked<1>();
    auto cost_arr   = costs_py.unchecked<1>();

    // ── Allocate output arrays ───────────────────────────────────────────
    auto equity_out  = py::array_t<double>(N);
    auto pnl_out     = py::array_t<double>(N);
    auto comm_out    = py::array_t<double>(N);
    auto trades_out  = py::array_t<int32_t>(N);
    auto cash_out    = py::array_t<double>(N);

    auto eq_buf  = equity_out.mutable_unchecked<1>();
    auto pnl_buf = pnl_out.mutable_unchecked<1>();
    auto cm_buf  = comm_out.mutable_unchecked<1>();
    auto tr_buf  = trades_out.mutable_unchecked<1>();
    auto ca_buf  = cash_out.mutable_unchecked<1>();

    // Zero-initialise
    for (int i = 0; i < N; ++i) {
        eq_buf(i)  = 0.0;
        pnl_buf(i) = 0.0;
        cm_buf(i)  = 0.0;
        tr_buf(i)  = 0;
        ca_buf(i)  = 0.0;
    }

    // ── State ────────────────────────────────────────────────────────────
    double equity      = starting_capital;
    double peak_equity = starting_capital;
    eq_buf(0)          = starting_capital;

    std::vector<double> current_pos(n_instruments, 0.0);

    // ── Main loop ────────────────────────────────────────────────────────
    for (int i = 0; i < N - 1; ++i) {

        // Before trading start: hold cash, no positions
        if (i < trading_start_idx) {
            eq_buf(i + 1) = equity;
            continue;
        }

        // Cash yield: entire NAV earns T-bill rate
        double day_cash = irx(i) * equity;
        double day_pnl  = day_cash;
        ca_buf(i + 1)   = day_cash;
        double day_comm  = 0.0;
        int day_trades   = 0;

        // ── Count active instruments with valid forecast/vol/fx ──────────
        int n_active = 0;
        for (int j = 0; j < n_instruments; ++j) {
            if (active_mat(j, i) == 0) continue;
            double f = fc_mat(j, i);
            double v = vol_mat(j, i);
            double fx = fx_mat(j, i);
            if (std::isnan(f) || f == 0.0) continue;
            if (std::isnan(v) || v < 1e-10) continue;
            if (std::isnan(fx)) continue;
            ++n_active;
        }
        double dynamic_weight = 1.0 / std::max(n_active, 1);
        double idm = idm_from_n_active(n_active);

        // ── Drawdown scaling ─────────────────────────────────────────────
        double dd_scale = 1.0;
        if (dd_scaling) {
            peak_equity = std::max(peak_equity, equity);
            double dd_pct = (peak_equity > 0.0)
                ? (peak_equity - equity) / peak_equity
                : 0.0;
            if (dd_pct >= DD_SCALE_FULL_AT) {
                dd_scale = DD_SCALE_FLOOR;
            } else if (dd_pct >= DD_SCALE_THRESHOLD) {
                dd_scale = 1.0 - (1.0 - DD_SCALE_FLOOR) *
                    (dd_pct - DD_SCALE_THRESHOLD) /
                    (DD_SCALE_FULL_AT - DD_SCALE_THRESHOLD);
            }
        }

        // ── Per-instrument rebalance and P&L ─────────────────────────────
        for (int j = 0; j < n_instruments; ++j) {
            if (active_mat(j, i) == 0) continue;

            double f_t  = fc_mat(j, i);
            double v_t  = vol_mat(j, i);
            double fx_t = fx_mat(j, i);
            double ps   = ps_arr(j);
            double cost = cost_arr(j);
            double cur  = current_pos[j];

            // Position sizing with rolling equity
            double new_pos;
            if (std::isnan(f_t) || std::isnan(v_t) || std::isnan(fx_t)
                    || v_t < 1e-10 || f_t == 0.0) {
                new_pos = 0.0;
            } else {
                double ann_vol_usd = v_t * ps * ANNUALISE_DAILY * fx_t;
                if (ann_vol_usd < 1.0) {
                    new_pos = cur;  // keep existing if vol unreliable
                } else {
                    double risk_budget = equity * VOL_TARGET * dynamic_weight * idm * dd_scale;
                    double target = (f_t / FORECAST_TARGET) * (risk_budget / ann_vol_usd);
                    double buf = compute_buffer(target);
                    new_pos = buffered_position(target, cur, buf);
                }
            }

            current_pos[j] = new_pos;

            // P&L: close[i] -> open[i+1] on old, open[i+1] -> close[i+1] on new
            double close_t  = close_mat(j, i);
            double open_t1  = open_mat(j, i + 1);
            double close_t1 = close_mat(j, i + 1);
            double fx_t1    = fx_mat(j, i + 1);

            if (std::isnan(close_t) || std::isnan(open_t1)
                    || std::isnan(close_t1) || std::isnan(fx_t1)) {
                continue;
            }

            double delta     = new_pos - cur;
            double comm      = std::abs(delta) * (cost / 2.0) * fx_t1;
            double overnight = cur * (open_t1 - close_t) * ps * fx_t1;
            double intraday  = new_pos * (close_t1 - open_t1) * ps * fx_t1;
            double pnl       = overnight + intraday - comm;

            day_pnl  += pnl;
            day_comm += comm;
            if (delta != 0.0) ++day_trades;
        }

        // ── Book P&L and compound equity ─────────────────────────────────
        pnl_buf(i + 1) = day_pnl;
        cm_buf(i + 1)  = day_comm;
        tr_buf(i + 1)  = day_trades;

        equity        += day_pnl;
        eq_buf(i + 1)  = equity;
    }

    // ── Return results ───────────────────────────────────────────────────
    py::dict result;
    result["equity"]     = equity_out;
    result["pnl"]        = pnl_out;
    result["commission"] = comm_out;
    result["trades"]     = trades_out;
    result["cash_yield"] = cash_out;
    return result;
}


// ═══════════════════════════════════════════════════════════════════════════
// pybind11 module definition
// ═══════════════════════════════════════════════════════════════════════════
PYBIND11_MODULE(fast_backtest, m) {
    m.doc() = R"doc(
        fast_backtest — C++ extension for IG_Backtest daily compounding loop.

        Accelerates the Phase 3 walk-forward simulation by running the
        stateful day-by-day equity compounding in C++ instead of Python.
    )doc";

    m.def("run_simulation", &run_simulation,
        R"doc(
        Run the daily compounding simulation in C++.

        Parameters
        ----------
        N : int
            Number of trading days.
        n_instruments : int
            Number of instruments.
        starting_capital : float
            Initial portfolio NAV in USD.
        trading_start_idx : int
            Index of the first day to open positions (days before this earn cash only).
        dd_scaling : bool
            Enable Carver drawdown scaling.
        irx_arr : np.ndarray[float64, (N,)]
            Daily T-bill yield fraction.
        fc_matrix : np.ndarray[float64, (n_inst, N)]
            Forecast arrays, row-major (instrument × day).
        vol_matrix : np.ndarray[float64, (n_inst, N)]
            Volatility arrays.
        fx_matrix : np.ndarray[float64, (n_inst, N)]
            FX rate arrays (USD per local currency).
        close_matrix : np.ndarray[float64, (n_inst, N)]
            Close price arrays.
        open_matrix : np.ndarray[float64, (n_inst, N)]
            Open price arrays.
        active_matrix : np.ndarray[uint8, (n_inst, N)]
            Active boolean masks.
        pointsizes : np.ndarray[float64, (n_inst,)]
            Point sizes per instrument.
        costs : np.ndarray[float64, (n_inst,)]
            Round-trip costs per instrument.

        Returns
        -------
        dict with keys:
            "equity"     : np.ndarray[float64, (N,)]
            "pnl"        : np.ndarray[float64, (N,)]
            "commission" : np.ndarray[float64, (N,)]
            "trades"     : np.ndarray[int32,   (N,)]
            "cash_yield" : np.ndarray[float64, (N,)]
        )doc",
        py::arg("N"),
        py::arg("n_instruments"),
        py::arg("starting_capital"),
        py::arg("trading_start_idx"),
        py::arg("dd_scaling"),
        py::arg("irx_arr"),
        py::arg("fc_matrix"),
        py::arg("vol_matrix"),
        py::arg("fx_matrix"),
        py::arg("close_matrix"),
        py::arg("open_matrix"),
        py::arg("active_matrix"),
        py::arg("pointsizes"),
        py::arg("costs")
    );
}
