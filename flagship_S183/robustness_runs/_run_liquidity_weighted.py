"""
Liquidity-weighted position sizing robustness test for Strategy 183.

Goal: compare equal-weight (1/N) sizing vs log(ADV)-weighted sizing.
Instead of re-running the full strategy, this test modifies the forecasts
by scaling them proportionally to each instrument's liquidity, then runs
the portfolio with the standard equal-weight sizer. The effect is equivalent
to a liquidity-weighted risk allocation.

Approach:
  1. Compute each instrument's ADV from raw contract data (last 5 years)
  2. Compute log(ADV) weights normalised to sum to 1
  3. Scale each instrument's forecast by (liq_weight / equal_weight) so that
     the equal-weight sizer effectively allocates proportional to liquidity
  4. Run the portfolio and compare to master

This is a clean approximation: scaling the forecast is equivalent to scaling
the position weight when the sizer is linear in forecast (which it is).
"""

import sys, os, pickle, time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent          # .../RobustnessTests
STRATEGY_DIR = _HERE.parent                      # .../Strategy_183
sys.path.insert(0, str(STRATEGY_DIR.parent))     # .../IG_Backtest

from ig_shared_config import (
    get_project_paths, load_mapping, load_instrument_data,
    run_compounded_portfolio,
)

paths = get_project_paths(STRATEGY_DIR)
CONTRACTS_DIR = Path(paths["contracts"])


def compute_adv(instrument, last_n_years=5):
    """Compute average daily volume from raw contract files (last N years)."""
    inst_dir = CONTRACTS_DIR / instrument
    if not inst_dir.exists():
        return np.nan

    all_vol = []
    for fp in sorted(inst_dir.glob("*.csv")):
        try:
            df = pd.read_csv(fp, usecols=["Date", "Volume"], parse_dates=["Date"])
            all_vol.append(df)
        except Exception:
            continue

    if not all_vol:
        return np.nan

    combined = pd.concat(all_vol).sort_values("Date").drop_duplicates("Date", keep="last")
    combined = combined.set_index("Date")

    # Last N years
    cutoff = combined.index.max() - pd.DateOffset(years=last_n_years)
    recent = combined.loc[combined.index >= cutoff, "Volume"]
    recent = recent[recent > 0]

    if len(recent) < 100:
        return np.nan

    return float(recent.mean())


def main():
    SEP = "=" * 72
    print(f"\n{SEP}")
    print("  Liquidity-Weighted Sizing Robustness Test")
    print(f"{SEP}")

    # Load master checkpoint
    ck_path = STRATEGY_DIR / "Strategy_183_IG_VoV_Quad_Sharpened_checkpoint.pkl"
    with open(ck_path, "rb") as f:
        master_ck = pickle.load(f)

    master_sr = master_ck["sr"]
    master_trades = master_ck["trades_yr"]
    print(f"\n  Master: SR={master_sr:.4f}, trades/yr={master_trades:.0f}")

    # Load mapping
    mapping = load_mapping(paths["mapping"])

    # Compute ADV for each instrument
    print("\n  Computing ADV per instrument...")
    adv_dict = {}
    for inst in tqdm(mapping.index, desc="ADV"):
        adv = compute_adv(inst)
        if not np.isnan(adv):
            adv_dict[inst] = adv

    print(f"  {len(adv_dict)} instruments with valid ADV")

    # Log-ADV weights
    log_adv = {inst: np.log(max(v, 1)) for inst, v in adv_dict.items()}
    total_log = sum(log_adv.values())
    n_insts = len(log_adv)
    equal_w = 1.0 / n_insts

    liq_weights = {inst: v / total_log for inst, v in log_adv.items()}

    # Print weight comparison for extremes
    sorted_by_adv = sorted(adv_dict.items(), key=lambda x: x[1])
    print(f"\n  {'Instrument':>12} {'ADV':>10} {'Equal W':>9} {'Liq W':>9} {'Ratio':>7}")
    print(f"  {'-'*50}")
    for inst, adv in sorted_by_adv[:5]:
        lw = liq_weights.get(inst, 0)
        print(f"  {inst:>12} {adv:>10,.0f} {equal_w:>9.4f} {lw:>9.4f} {lw/equal_w:>7.2f}x")
    print(f"  {'...':>12}")
    for inst, adv in sorted_by_adv[-5:]:
        lw = liq_weights.get(inst, 0)
        print(f"  {inst:>12} {adv:>10,.0f} {equal_w:>9.4f} {lw:>9.4f} {lw/equal_w:>7.2f}x")

    # Now we need to run the strategy with modified forecasts.
    # The cleanest way: import the strategy, run it, then scale forecasts
    # before passing to run_compounded_portfolio.

    # Re-run the strategy signal pipeline
    print("\n  Running S183 signal pipeline...")
    sys.path.insert(0, str(STRATEGY_DIR))

    # We need to use the halflife sweep's approach — import and run
    from ig_shared_config import (
        FORECAST_CAP, VOL_TARGET, load_fx_rates,
        align_fx, blended_volatility, compute_ewmac_raw,
        compute_carry, cap_forecast, rolling_forecast_scalar,
    )

    TRADING_DAYS_YEAR = 256
    TRADING_DAYS_HALF = 128
    TRADING_DAYS_QUARTER = 64
    N_QUAD_ALPHAS = 4
    W_QUAD = 1.0 / N_QUAD_ALPHAS
    W_VEC = np.full(N_QUAD_ALPHAS, W_QUAD)
    TREND_SPEED_PAIRS = [(16, 64), (32, 128), (64, 256)]
    N_SPEEDS = len(TREND_SPEED_PAIRS)
    W_SPEED = 1.0 / N_SPEEDS
    TREND_SPEEDS = [{"fast": f, "slow": s, "weight": W_SPEED} for f, s in TREND_SPEED_PAIRS]
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

    def _sigmoid(x):
        return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

    def _pooled_fdm_4(tp, cp, sp, vp):
        fc_df = pd.DataFrame({"t": tp.mean(1), "c": cp.mean(1), "s": sp.mean(1), "v": vp.mean(1)})
        T = len(fc_df); fdm = np.ones(T)
        ewm_corr = fc_df.ewm(span=FDM_CORR_SPAN, min_periods=FDM_MIN_PERIODS).corr()
        for t in range(FDM_MIN_PERIODS, T):
            try:
                c = ewm_corr.loc[fc_df.index[t]].values
                if c.shape != (4,4) or np.any(np.isnan(c)): continue
                c = np.clip(c, -1, 1)
                var = W_VEC @ c @ W_VEC
                if var > 0.01: fdm[t] = np.clip(1/np.sqrt(var), FDM_FLOOR, FDM_CAP)
            except: continue
        return pd.Series(fdm, index=fc_df.index)

    fx_daily = load_fx_rates(paths["panama"])

    # Load instruments
    inst_data_cache = {}
    inst_list = []
    for inst in tqdm(mapping.index, desc="Loading"):
        data = load_instrument_data(inst, paths["stats"], paths["panama"])
        if data is None or len(data["daily_dates"]) < OOS_START:
            continue
        inst_data_cache[inst] = data
        inst_list.append(inst)

    # Phase 1a: signals
    raw_cache = {}
    for instrument in tqdm(inst_list, desc="Signals"):
        data = inst_data_cache[instrument]
        dates = data["daily_dates"]
        if len(dates) < OOS_START + 2: continue
        ccy = mapping.loc[instrument, "currency"]
        fx = align_fx(ccy, dates, fx_daily)
        if fx is None: continue

        close = data["close"]; price_changes = data["price_changes"]
        vol = blended_volatility(price_changes); idx = pd.DatetimeIndex(dates)

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
        final_skew = pd.Series(cap_forecast(raw_skew * rolling_forecast_scalar(raw_skew)).reindex(idx).fillna(0.0).values, index=idx)

        daily_vol = price_changes.rolling(21, min_periods=10).std()
        vov_mp = min(VOV_WINDOW, max(21, VOV_WINDOW // 3))
        vov = daily_vol.rolling(VOV_WINDOW, min_periods=vov_mp).std()
        vov_avg = vov.rolling(VOV_AVG_WINDOW, min_periods=VOV_WINDOW).mean()
        vov_norm = -(vov / vov_avg - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        direction = np.where(close.pct_change(VOV_WINDOW).fillna(0.0) >= 0, 1.0, -1.0)
        vov_raw = (pd.Series(direction, index=close.index) * vov_norm).fillna(0.0)
        final_vov = pd.Series(cap_forecast(vov_raw * rolling_forecast_scalar(vov_raw)).reindex(idx).fillna(0.0).values, index=idx)

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

    # Phase 1b: XS
    xs_raw_df = pd.DataFrame({i: raw_cache[i]["xs_raw"] for i in all_instruments})
    n_valid = xs_raw_df.notna().sum(axis=1)
    cs_mean = xs_raw_df.mean(axis=1); cs_std = xs_raw_df.std(axis=1).replace(0.0, np.nan)
    xs_zscored = xs_raw_df.subtract(cs_mean, axis=0).divide(cs_std, axis=0)
    xs_zscored[n_valid < XS_MIN_INSTS] = np.nan
    xs_forecasts = {}
    for inst in all_instruments:
        z = xs_zscored[inst].fillna(0.0)
        xs_forecasts[inst] = pd.Series(cap_forecast(z * rolling_forecast_scalar(z)).reindex(raw_cache[inst]["idx"]).fillna(0.0).values, index=raw_cache[inst]["idx"])

    # Phase 1c: blend + FDM + smoothing + overlays
    final_trend_map, final_carry_map, final_skew_map, final_vov_map = {}, {}, {}, {}
    for inst in all_instruments:
        c = raw_cache[inst]; xs = xs_forecasts[inst]
        final_trend_map[inst] = cap_forecast(W_TS * c["ts_trend"] + W_XS * xs)
        final_carry_map[inst] = c["final_carry"]
        final_skew_map[inst] = c["final_skew"]
        final_vov_map[inst] = c["final_vov"]

    pooled_fdm = _pooled_fdm_4(pd.DataFrame(final_trend_map), pd.DataFrame(final_carry_map),
                                pd.DataFrame(final_skew_map), pd.DataFrame(final_vov_map))

    # Build final forecasts for BOTH variants
    def build_signals(weight_scale=None):
        """Build inst_signals dict. If weight_scale is provided, scale forecasts."""
        inst_signals = {}
        for instrument in all_instruments:
            c = raw_cache[instrument]; idx = c["idx"]; dates = c["dates"]
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

            T_len = len(idx); final_fc = np.zeros(T_len)
            for t in range(OOS_START, T_len):
                fc = fc_smooth[t]
                if np.isnan(fc): continue
                vol_gate = 1.0 - (1.0 - VOL_SCALE_DAMPEN) * _sigmoid(SIGMOID_STEEPNESS * (vol_ratio[t] - VOL_SCALE_TRIGGER))
                dd_gate = 1.0 - (1.0 - DD_SCALE) * _sigmoid(SIGMOID_STEEPNESS * (DD_THRESHOLD - dd_proxy[t]))
                fc *= float(vol_gate) * float(dd_gate)
                final_fc[t] = float(np.clip(fc, -FORECAST_CAP, FORECAST_CAP))

            # Apply liquidity weight scaling if provided
            if weight_scale is not None and instrument in weight_scale:
                scale = weight_scale[instrument]
                final_fc = np.clip(final_fc * scale, -FORECAST_CAP, FORECAST_CAP)

            forecast = pd.Series(final_fc, index=idx)
            inst_signals[instrument] = {
                "oos_date_set": set(dates[OOS_START:]),
                "forecast": forecast, "vol": c["vol"], "fx": c["fx"],
                "close": c["close"], "open": c["open"],
                "pointsize": c["pointsize"], "cost_rt": c["cost_rt"],
            }
        return inst_signals

    # Compute scale factors: liq_weight / equal_weight
    n_with_adv = len([i for i in all_instruments if i in liq_weights])
    eq_w = 1.0 / n_with_adv
    scale_factors = {}
    for inst in all_instruments:
        if inst in liq_weights:
            scale_factors[inst] = liq_weights[inst] / eq_w
        else:
            scale_factors[inst] = 1.0  # keep equal if no ADV data

    # Run both variants
    print(f"\n  Running equal-weight variant (master replication)...")
    t0 = time.time()
    ck_equal = run_compounded_portfolio(
        build_signals(weight_scale=None),
        "S183_EqualWeight", paths, save_per_inst_pnl=False)

    print(f"\n  Running liquidity-weighted variant...")
    ck_liq = run_compounded_portfolio(
        build_signals(weight_scale=scale_factors),
        "S183_LiquidityWeighted", paths, save_per_inst_pnl=False)
    elapsed = time.time() - t0

    # Results
    print(f"\n{SEP}")
    print("  RESULTS: Equal-Weight vs Liquidity-Weighted Sizing")
    print(f"{SEP}")
    print(f"\n  {'Metric':<20} {'Equal-Weight':>14} {'Liq-Weighted':>14} {'Delta':>10}")
    print(f"  {'-'*60}")
    for metric in ['sr', 'cagr', 'max_dd', 'calmar', 'ann_vol', 'trades_yr']:
        v_eq = ck_equal[metric]
        v_lq = ck_liq[metric]
        delta = v_lq - v_eq
        if metric in ['cagr', 'max_dd', 'ann_vol']:
            print(f"  {metric:<20} {v_eq:>14.2%} {v_lq:>14.2%} {delta:>+10.2%}")
        elif metric == 'trades_yr':
            print(f"  {metric:<20} {v_eq:>14,.0f} {v_lq:>14,.0f} {delta:>+10,.0f}")
        else:
            print(f"  {metric:<20} {v_eq:>14.4f} {v_lq:>14.4f} {delta:>+10.4f}")

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
