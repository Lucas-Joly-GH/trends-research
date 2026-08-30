"""Single Alpha: Network_Momentum -- Li & Ferreira (2025) Levy-area graph learning"""
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
    align_fx, blended_volatility, compute_ewmac_raw,
    compute_carry, cap_forecast, run_compounded_portfolio,
    rolling_forecast_scalar,
)

STRATEGY_NAME = "Single_Alpha_Network_Momentum"
OOS_START = 1280

# NMM parameters (S86 identical)
ENSEMBLE_WINDOWS    = [22, 44, 66, 88, 110, 132]
REFIT_EVERY         = 22
GRAPH_ALPHA         = 1.0
GRAPH_BETA          = 0.1
GRAPH_ITERS         = 300
OSCILLATOR_K_VALUES = [1, 2, 3, 4, 5, 6]
M_RATIO             = 4
N_NMM_SPEEDS        = len(OSCILLATOR_K_VALUES)
RESPONSE_LAMBDA     = np.sqrt(2.0)
VOL_SPAN            = 22


# ── NMM helpers (from S86) ──────────────────────────────────────────────────

def _compute_response_constant(lam=RESPONSE_LAMBDA, n=2_000_000):
    rng = np.random.default_rng(42)
    x   = rng.standard_normal(n)
    return 1.0 / np.std(x * np.exp(-lam**2 * x**2 / 2.0))


def _compute_target_scalar(lam=RESPONSE_LAMBDA, n=2_000_000):
    rng  = np.random.default_rng(42)
    x    = rng.standard_normal(n)
    raw  = x * np.exp(-lam**2 * x**2 / 2.0)
    std  = raw / np.std(raw)
    return FORECAST_TARGET / np.mean(np.abs(std))


RESPONSE_C    = _compute_response_constant()
TARGET_SCALAR = _compute_target_scalar()


def _response_fn(x):
    return RESPONSE_C * x * np.exp(-RESPONSE_LAMBDA**2 * x**2 / 2.0)


def _pairwise_sq_dist(V):
    sq = np.sum(V**2, axis=1)
    Z  = sq[:, None] + sq[None, :] - 2.0 * V @ V.T
    return np.maximum(Z, 0.0)


def _graph_learning(V):
    Z = _pairwise_sq_dist(V)
    z_max = Z.max()
    if z_max > 0:
        Z = Z / z_max
    W = 1.0 / (Z + 1.0)
    np.fill_diagonal(W, 0.0)
    W = (W + W.T) / 2.0
    W *= 0.1 / (W.max() + 1e-12)
    step = 0.3 / (2.0 * GRAPH_BETA + 1e-8)
    for _ in range(GRAPH_ITERS):
        d     = np.maximum(W.sum(axis=1), 1e-6)
        inv_d = GRAPH_ALPHA / d
        grad  = Z + 2.0 * GRAPH_BETA * W - inv_d[:, None] - inv_d[None, :]
        W_new = np.maximum(W - step * grad, 0.0)
        np.fill_diagonal(W_new, 0.0)
        W_new = (W_new + W_new.T) / 2.0
        if np.abs(W_new - W).max() < 1e-6:
            break
        W = W_new
    return W


def _normalise_adj(A):
    d = A.sum(axis=1)
    d_i = np.where(d > 1e-10, 1.0 / np.sqrt(d), 0.0)
    return d_i[:, None] * A * d_i[None, :]


def _levy_area(deltas):
    p = np.vstack([np.zeros(deltas.shape[1]), np.cumsum(deltas, axis=0)])
    return 0.5 * (p[1:].T @ p[:-1] - p[:-1].T @ p[1:])


def _compute_adjacency(vol_scaled_deltas, t, valid_mask):
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 5:
        return np.zeros((vol_scaled_deltas.shape[1],) * 2)
    A_ens, n_w = np.zeros((len(valid_idx),) * 2), 0
    for delta in ENSEMBLE_WINDOWS:
        if t < delta:
            continue
        win = np.nan_to_num(vol_scaled_deltas[t - delta:t, :][:, valid_idx], nan=0.0)
        if np.isnan(win).sum() > 0.3 * win.size:
            continue
        A_ens += _graph_learning(_levy_area(win))
        n_w   += 1
    if n_w == 0:
        return np.zeros((vol_scaled_deltas.shape[1],) * 2)
    A_s = _normalise_adj(A_ens / n_w)
    M   = vol_scaled_deltas.shape[1]
    A_f = np.zeros((M, M))
    for ii, vi in enumerate(valid_idx):
        for jj, vj in enumerate(valid_idx):
            A_f[vi, vj] = A_s[ii, jj]
    return A_f


def _ema_matrix(series, alpha):
    out = np.zeros_like(series)
    out[0] = series[0]
    for t in range(1, len(series)):
        out[t] = alpha * series[t] + (1.0 - alpha) * out[t - 1]
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def run_strategy():
    paths    = get_project_paths(_HERE)
    mapping  = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])

    # ── PHASE 1A: Load instruments & build global matrices ──
    print(f"\n[PHASE 1A] Loading instruments for {STRATEGY_NAME} ...")
    inst_data_cache = {}
    inst_list       = []

    for inst in tqdm(mapping.index, desc="Loading"):
        data = load_instrument_data(inst, paths["stats"], paths["panama"])
        if data is None or len(data["daily_dates"]) < OOS_START:
            continue
        inst_data_cache[inst] = data
        inst_list.append(inst)

    M           = len(inst_list)
    inst_to_idx = {inst: i for i, inst in enumerate(inst_list)}

    # Build common date universe
    all_dates_set = set()
    for inst in inst_list:
        all_dates_set.update(inst_data_cache[inst]["daily_dates"])
    common_dates = np.array(sorted(all_dates_set))
    T_global     = len(common_dates)
    date_to_t    = {d: i for i, d in enumerate(common_dates)}
    print(f"  {M} instruments, {T_global} common days.")

    # Align close prices to global grid
    raw_close = np.full((T_global, M), np.nan)
    for inst in tqdm(inst_list, desc="Align close"):
        i    = inst_to_idx[inst]
        data = inst_data_cache[inst]
        for k_idx, d in enumerate(data["daily_dates"]):
            if d in date_to_t:
                raw_close[date_to_t[d], i] = data["close"].values[k_idx]

    # Vol-scaled deltas
    price_delta = np.full((T_global, M), np.nan)
    price_delta[1:, :] = np.diff(raw_close, axis=0)

    alpha_ewm = 2.0 / (VOL_SPAN + 1)
    ewm_var   = np.zeros(M)
    sigma22   = np.full((T_global, M), np.nan)
    for t in range(T_global):
        dp = price_delta[t, :]
        ok = ~np.isnan(dp)
        ewm_var[ok] = alpha_ewm * dp[ok]**2 + (1.0 - alpha_ewm) * ewm_var[ok]
        sigma22[t, ok] = np.sqrt(np.maximum(ewm_var[ok], 0.0))

    valid_m = (sigma22 > 1e-8) & ~np.isnan(price_delta)
    vol_scaled_delta = np.zeros((T_global, M))
    vol_scaled_delta[valid_m] = np.clip(
        price_delta[valid_m] / sigma22[valid_m], -20.0, 20.0
    )
    vol_scaled_price = np.cumsum(vol_scaled_delta, axis=0)

    # ── PHASE 1B: NMM oscillators ──
    print(f"\n[PHASE 1B] Computing {N_NMM_SPEEDS} NMM oscillators ...")
    ts_oscillators = []
    for k in OSCILLATOR_K_VALUES:
        ts_oscillators.append(
            _ema_matrix(vol_scaled_price, 1.0 / (2.0 * k))
            - _ema_matrix(vol_scaled_price, 1.0 / (M_RATIO * 2.0 * k))
        )

    # ── PHASE 1C: Graph learning at refit points ──
    print("\n[PHASE 1C] Graph learning at refit points ...")
    refit_times      = list(range(max(ENSEMBLE_WINDOWS), T_global, REFIT_EVERY))
    adjacency_series = {}
    for t in tqdm(refit_times, desc="Graph"):
        start    = max(0, t - max(ENSEMBLE_WINDOWS))
        coverage = np.sum(~np.isnan(raw_close[start:t, :]), axis=0)
        nontriv  = np.sum(np.abs(vol_scaled_delta[start:t, :]), axis=0)
        valid    = (coverage > (t - start) * 0.7) & (nontriv > 1e-6)
        adjacency_series[t] = _compute_adjacency(vol_scaled_delta, t, valid)

    refit_arr    = np.array(sorted(adjacency_series.keys()))
    latest_refit = np.zeros(T_global, dtype=int)
    for t in range(T_global):
        idx_r = np.searchsorted(refit_arr, t, side="right") - 1
        if idx_r >= 0:
            latest_refit[t] = refit_arr[idx_r]

    # ── PHASE 1D: Build NMM forecast matrix ──
    print("\n[PHASE 1D] Building NMM forecast matrix ...")
    all_osc       = np.stack(ts_oscillators, axis=0)
    nmm_fc_matrix = np.zeros((T_global, M))
    for t in tqdm(range(T_global), desc="NMM signals"):
        refit_t = latest_refit[t]
        A       = adjacency_series.get(refit_t) if refit_t > 0 else None
        osc_t   = all_osc[:, t, :]
        net_t   = osc_t @ A.T if A is not None else osc_t
        nmm_fc_matrix[t, :] = _response_fn(net_t).mean(axis=0) * TARGET_SCALAR

    # ── PHASE 2: Build per-instrument signals ──
    print("\n[PHASE 2] Building per-instrument NMM signals...")
    inst_signals = {}
    common_idx = pd.DatetimeIndex(common_dates)

    for instrument in tqdm(inst_list, desc=STRATEGY_NAME):
        data  = inst_data_cache[instrument]
        dates = data["daily_dates"]
        if len(dates) < OOS_START + 2:
            continue

        ccy = mapping.loc[instrument, "currency"]
        fx  = align_fx(ccy, dates, fx_daily)
        if fx is None:
            continue

        idx     = pd.DatetimeIndex(dates)
        i_glob  = inst_to_idx[instrument]
        vol     = blended_volatility(data["price_changes"])

        # NMM forecast for this instrument
        fc_glob = pd.Series(nmm_fc_matrix[:, i_glob], index=common_idx)
        nmm_raw = fc_glob.reindex(idx).fillna(0.0)
        forecast_scaled = cap_forecast(nmm_raw)

        final_fc = forecast_scaled.reindex(idx).fillna(0.0)
        final_fc.iloc[:OOS_START] = 0.0

        inst_signals[instrument] = {
            "oos_date_set": set(dates[OOS_START:]),
            "forecast":     final_fc,
            "vol":          vol.reindex(idx),
            "fx":           fx.reindex(idx),
            "close":        data["close"].reindex(idx),
            "open":         data["open"].reindex(idx),
            "pointsize":    float(mapping.loc[instrument, "pointsize"]),
            "cost_rt":      float(mapping.loc[instrument, "total_avg_cost_rt"]),
        }

    run_compounded_portfolio(inst_signals, STRATEGY_NAME, paths)


if __name__ == "__main__":
    run_strategy()
