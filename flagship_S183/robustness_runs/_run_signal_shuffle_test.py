"""
White-Noise Signal Shuffle Test (fix for T17)
=============================================================================
The original T17 in the testing suite permutes master daily returns, which
leaves mean/std unchanged and therefore produces identical Sharpe ratios on
every shuffle (the test measures nothing).

This corrected version shuffles each instrument's daily forecast time series
independently, destroying the temporal predictive relationship between
signal and subsequent return while preserving the marginal distribution of
forecasts. The portfolio is then rerun with shuffled forecasts and the
resulting SR distribution is compared to the realised SR.

Null hypothesis: if the signals have no predictive power, the realised SR
should be consistent with the distribution of shuffled SRs.

Run from IG_Backtest root:
  python Strategy_183/RobustnessTests/_run_signal_shuffle_test.py
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
STRATEGY_DIR = _HERE.parent
sys.path.insert(0, str(STRATEGY_DIR.parent))

from ig_shared_config import (
    FORECAST_CAP, FORECAST_TARGET, VOL_TARGET,
    get_project_paths, load_mapping, load_fx_rates, load_instrument_data,
    align_fx, blended_volatility, compute_ewmac_raw,
    compute_carry, cap_forecast, run_compounded_portfolio,
    rolling_forecast_scalar, load_irx,
)

# ── S183 constants ────────────────────────────────────────────────────────────
TRADING_DAYS_YEAR    = 256
TRADING_DAYS_HALF    = 128
TRADING_DAYS_QUARTER = 64

N_QUAD_ALPHAS = 4
W_QUAD = 1.0 / N_QUAD_ALPHAS
W_VEC  = np.full(N_QUAD_ALPHAS, W_QUAD)

TREND_SPEED_PAIRS = [(16, 64), (32, 128), (64, 256)]
N_SPEEDS = len(TREND_SPEED_PAIRS)
W_SPEED = 1.0 / N_SPEEDS
TREND_SPEEDS = [{"fast": f, "slow": s, "weight": W_SPEED}
                for (f, s) in TREND_SPEED_PAIRS]

W_TS = W_XS = 0.5
SKEW_WINDOW = TRADING_DAYS_YEAR
VOV_WINDOW = TRADING_DAYS_QUARTER
VOV_AVG_WINDOW = TRADING_DAYS_YEAR
XS_LOOKBACK = TRADING_DAYS_YEAR
XS_MIN_INSTS = 3
LOCAL_SCALAR_WINDOW = 5 * TRADING_DAYS_YEAR
OOS_START = LOCAL_SCALAR_WINDOW

FDM_CORR_SPAN = 2 * TRADING_DAYS_YEAR
FDM_MIN_PERIODS = TRADING_DAYS_YEAR
FDM_FLOOR = 1.0
FDM_CAP = float(np.sqrt(N_QUAD_ALPHAS))
SMOOTH_HALFLIFE = 1

OVERLAY_FLOOR = 0.50
VOL_SCALE_TRIGGER = 1.0
VOL_SCALE_DAMPEN = OVERLAY_FLOOR
VOL_SCALE_LOOKBACK = TRADING_DAYS_QUARTER
DD_THRESHOLD = -(VOL_TARGET / 2.0)
DD_SCALE = OVERLAY_FLOOR
DD_LOOKBACK = TRADING_DAYS_QUARTER
SIGMOID_STEEPNESS = 2.0 / VOL_TARGET

N_SHUFFLES = 100
SEED_BASE = 42


def _sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _pooled_fdm_4(tp, cp, sp, vp):
    fc_df = pd.DataFrame({"t": tp.mean(1), "c": cp.mean(1),
                           "s": sp.mean(1), "v": vp.mean(1)})
    T = len(fc_df)
    fdm = np.ones(T)
    ewm_corr = fc_df.ewm(span=FDM_CORR_SPAN, min_periods=FDM_MIN_PERIODS).corr()
    for t in range(FDM_MIN_PERIODS, T):
        try:
            c = ewm_corr.loc[fc_df.index[t]].values
            if c.shape != (4, 4) or np.any(np.isnan(c)):
                continue
            c = np.clip(c, -1, 1)
            var = W_VEC @ c @ W_VEC
            if var > 0.01:
                fdm[t] = np.clip(1 / np.sqrt(var), FDM_FLOOR, FDM_CAP)
        except:
            continue
    return pd.Series(fdm, index=fc_df.index)


def build_forecasts(paths, mapping, fx_daily, inst_data_cache, inst_list,
                    shuffle_seed=None):
    """
    Build per-instrument final forecasts.

    If shuffle_seed is provided, the final smoothed forecast for each
    instrument is shuffled (permuted along the time axis) after the overlay
    step, destroying the temporal relationship between signal and return.

    Returns: dict of inst_signals compatible with run_compounded_portfolio.
    """
    rng = np.random.RandomState(shuffle_seed) if shuffle_seed is not None else None

    raw_cache = {}
    for instrument in inst_list:
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

        ts_components = []
        for speed in TREND_SPEEDS:
            raw = compute_ewmac_raw(close, vol, speed["fast"], speed["slow"])
            scaled = cap_forecast(raw * rolling_forecast_scalar(raw))
            ts_components.append(pd.Series(scaled.reindex(idx).fillna(0.0).values, index=idx))
        ts_blend = sum(c / N_SPEEDS for c in ts_components)
        ts_trend = cap_forecast(ts_blend)

        raw_carry = compute_carry(instrument, paths["panama"], paths["contracts"], dates, vol)
        if raw_carry is not None and raw_carry.dropna().any():
            carry_fc = cap_forecast(raw_carry.fillna(0.0) * rolling_forecast_scalar(raw_carry.fillna(0.0)))
        else:
            carry_fc = pd.Series(0.0, index=idx)
        final_carry = pd.Series(carry_fc.reindex(idx).fillna(0.0).values, index=idx)

        raw_skew = -price_changes.rolling(SKEW_WINDOW, min_periods=TRADING_DAYS_HALF).skew().fillna(0.0)
        final_skew = pd.Series(
            cap_forecast(raw_skew * rolling_forecast_scalar(raw_skew)).reindex(idx).fillna(0.0).values, index=idx)

        daily_vol = price_changes.rolling(21, min_periods=10).std()
        vov_mp = min(VOV_WINDOW, max(21, VOV_WINDOW // 3))
        vov = daily_vol.rolling(VOV_WINDOW, min_periods=vov_mp).std()
        vov_avg = vov.rolling(VOV_AVG_WINDOW, min_periods=VOV_WINDOW).mean()
        vov_norm = -(vov / vov_avg - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        direction = np.where(close.pct_change(VOV_WINDOW).fillna(0.0) >= 0, 1.0, -1.0)
        vov_raw = (pd.Series(direction, index=close.index) * vov_norm).fillna(0.0)
        final_vov = pd.Series(
            cap_forecast(vov_raw * rolling_forecast_scalar(vov_raw)).reindex(idx).fillna(0.0).values, index=idx)

        norm_price = (price_changes / vol.replace(0.0, np.nan)).fillna(0.0).cumsum()
        xs_raw = norm_price.diff(XS_LOOKBACK).reindex(idx)

        raw_cache[instrument] = {
            "idx": idx, "dates": dates, "ts_trend": ts_trend,
            "final_carry": final_carry, "final_skew": final_skew, "final_vov": final_vov,
            "xs_raw": xs_raw, "vol": vol.reindex(idx), "fx": fx.reindex(idx),
            "close": close.reindex(idx), "open": data["open"].reindex(idx),
            "pointsize": float(mapping.loc[instrument, "pointsize"]),
            "cost_rt": float(mapping.loc[instrument, "total_avg_cost_rt"]),
        }

    all_instruments = list(raw_cache.keys())

    # XS z-score
    xs_raw_df = pd.DataFrame({i: raw_cache[i]["xs_raw"] for i in all_instruments})
    n_valid = xs_raw_df.notna().sum(axis=1)
    cs_mean = xs_raw_df.mean(axis=1)
    cs_std = xs_raw_df.std(axis=1).replace(0.0, np.nan)
    xs_zscored = xs_raw_df.subtract(cs_mean, axis=0).divide(cs_std, axis=0)
    xs_zscored[n_valid < XS_MIN_INSTS] = np.nan
    xs_forecasts = {}
    for inst in all_instruments:
        z = xs_zscored[inst].fillna(0.0)
        xs_forecasts[inst] = pd.Series(
            cap_forecast(z * rolling_forecast_scalar(z)).reindex(raw_cache[inst]["idx"]).fillna(0.0).values,
            index=raw_cache[inst]["idx"])

    # Build final alpha blends
    final_trend_map, final_carry_map, final_skew_map, final_vov_map = {}, {}, {}, {}
    for inst in all_instruments:
        c = raw_cache[inst]
        xs = xs_forecasts[inst]
        final_trend_map[inst] = cap_forecast(W_TS * c["ts_trend"] + W_XS * xs)
        final_carry_map[inst] = c["final_carry"]
        final_skew_map[inst] = c["final_skew"]
        final_vov_map[inst] = c["final_vov"]

    pooled_fdm = _pooled_fdm_4(
        pd.DataFrame(final_trend_map), pd.DataFrame(final_carry_map),
        pd.DataFrame(final_skew_map), pd.DataFrame(final_vov_map))

    # Apply FDM + smoothing + overlays
    inst_signals = {}
    for instrument in all_instruments:
        c = raw_cache[instrument]
        idx = c["idx"]
        dates = c["dates"]
        fdm_s = pooled_fdm.reindex(idx).ffill().fillna(1.0)
        raw_blend = (W_QUAD * final_trend_map[instrument] + W_QUAD * c["final_carry"]
                     + W_QUAD * c["final_skew"] + W_QUAD * c["final_vov"])
        fdm_scaled = raw_blend * fdm_s
        master_fc = cap_forecast(fdm_scaled * rolling_forecast_scalar(fdm_scaled))
        smoothed = master_fc.ewm(halflife=SMOOTH_HALFLIFE, min_periods=1).mean()
        fc_smooth = cap_forecast(smoothed).reindex(idx).fillna(0.0).values

        daily_ret = c["close"].pct_change()
        realised_vol = daily_ret.rolling(VOL_SCALE_LOOKBACK).std() * np.sqrt(TRADING_DAYS_YEAR)
        vol_ratio = (realised_vol / VOL_TARGET).fillna(1.0).reindex(idx).fillna(1.0).values
        dd_proxy = c["close"].pct_change(DD_LOOKBACK).shift(1).reindex(idx).fillna(0.0).values

        T_len = len(idx)
        final_fc = np.zeros(T_len)
        for t in range(OOS_START, T_len):
            fc = fc_smooth[t]
            if np.isnan(fc):
                continue
            vol_gate = 1.0 - (1.0 - VOL_SCALE_DAMPEN) * _sigmoid(
                SIGMOID_STEEPNESS * (vol_ratio[t] - VOL_SCALE_TRIGGER))
            dd_gate = 1.0 - (1.0 - DD_SCALE) * _sigmoid(
                SIGMOID_STEEPNESS * (DD_THRESHOLD - dd_proxy[t]))
            final_fc[t] = float(np.clip(fc * float(vol_gate) * float(dd_gate),
                                        -FORECAST_CAP, FORECAST_CAP))

        # ── SIGNAL SHUFFLE: permute the OOS portion of the final forecast ──
        # Preserves marginal distribution, destroys temporal relationship.
        if rng is not None:
            oos_slice = final_fc[OOS_START:].copy()
            rng.shuffle(oos_slice)
            final_fc[OOS_START:] = oos_slice

        forecast = pd.Series(final_fc, index=idx)
        inst_signals[instrument] = {
            "oos_date_set": set(dates[OOS_START:]),
            "forecast": forecast, "vol": c["vol"], "fx": c["fx"],
            "close": c["close"], "open": c["open"],
            "pointsize": c["pointsize"], "cost_rt": c["cost_rt"],
        }

    return inst_signals


def main():
    SEP = "=" * 72
    print(f"\n{SEP}")
    print("  White-Noise Signal Shuffle Test (corrected T17)")
    print(f"  N = {N_SHUFFLES} shuffles")
    print(f"{SEP}\n")

    paths = get_project_paths(STRATEGY_DIR)
    mapping = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])

    # Load instrument data once
    print("[PHASE 0] Loading instruments...")
    inst_data_cache = {}
    inst_list = []
    for inst in tqdm(mapping.index, desc="Loading"):
        data = load_instrument_data(inst, paths["stats"], paths["panama"])
        if data is None or len(data["daily_dates"]) < OOS_START:
            continue
        inst_data_cache[inst] = data
        inst_list.append(inst)
    print(f"  {len(inst_list)} instruments loaded.\n")

    # Run N_SHUFFLES with shuffled signals
    shuffle_srs = []
    t_start = time.time()

    for i in range(N_SHUFFLES):
        seed = SEED_BASE + i
        inst_signals = build_forecasts(
            paths, mapping, fx_daily, inst_data_cache, inst_list,
            shuffle_seed=seed)
        ck = run_compounded_portfolio(
            inst_signals, f"S183_shuffle_{i:03d}", paths,
            save_per_inst_pnl=False)
        if ck is None:
            print(f"  [{i+1}/{N_SHUFFLES}] FAILED")
            continue
        sr = float(ck["sr"])
        shuffle_srs.append(sr)
        elapsed = time.time() - t_start
        eta = elapsed / (i+1) * (N_SHUFFLES - i - 1)
        print(f"  [{i+1:3d}/{N_SHUFFLES}] Shuffle SR = {sr:+.4f}  "
              f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

    shuffle_srs = np.array(shuffle_srs)

    # Load master SR
    import pickle
    with open(str(STRATEGY_DIR / "Strategy_183_IG_VoV_Quad_Sharpened_checkpoint.pkl"), "rb") as f:
        master_ck = pickle.load(f)
    real_sr = float(master_ck["sr"])

    # Compute stats
    pct_rank = float(np.mean(shuffle_srs >= real_sr))

    print(f"\n{SEP}")
    print("  RESULTS")
    print(f"{SEP}")
    print(f"  Real SR:              {real_sr:.4f}")
    print(f"  Shuffle N:            {len(shuffle_srs)}")
    print(f"  Shuffle SR mean:      {shuffle_srs.mean():+.4f}")
    print(f"  Shuffle SR std:       {shuffle_srs.std():.4f}")
    print(f"  Shuffle SR min:       {shuffle_srs.min():+.4f}")
    print(f"  Shuffle SR max:       {shuffle_srs.max():+.4f}")
    print(f"  Shuffle SR 95th pct:  {np.percentile(shuffle_srs, 95):+.4f}")
    print(f"  Shuffle SR 99th pct:  {np.percentile(shuffle_srs, 99):+.4f}")
    print(f"  pct_rank (shuffle >= real): {pct_rank:.4f}")
    print(f"{SEP}\n")

    # Save JSON
    result = {
        "n_shuffles": len(shuffle_srs),
        "real_sr": round(real_sr, 4),
        "shuffle_mean": round(float(shuffle_srs.mean()), 4),
        "shuffle_std": round(float(shuffle_srs.std()), 4),
        "shuffle_min": round(float(shuffle_srs.min()), 4),
        "shuffle_max": round(float(shuffle_srs.max()), 4),
        "shuffle_p95": round(float(np.percentile(shuffle_srs, 95)), 4),
        "shuffle_p99": round(float(np.percentile(shuffle_srs, 99)), 4),
        "pct_rank": round(pct_rank, 4),
        "all_shuffle_srs": [round(float(s), 4) for s in shuffle_srs],
        "note": ("Corrected T17 white-noise test: each instrument's final "
                 "forecast is permuted along the time axis, destroying the "
                 "temporal predictive relationship with future returns. The "
                 "portfolio is rerun with shuffled forecasts."),
    }
    out_path = _HERE / "t17_signal_shuffle_result.json"
    with open(str(out_path), "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path.name}")

    # Clean up temp checkpoints
    import os
    for pat in ["S183_shuffle_*"]:
        for fp in Path(paths["output"]).glob(pat):
            os.remove(fp)


if __name__ == "__main__":
    main()
