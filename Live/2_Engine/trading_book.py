from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
import warnings
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
FC = HERE.parent / "1_Roll" / "Front_Contract" / "front_contract.py"
CYCLES = HERE.parent / "1_Roll" / "contract_cycles.csv"
MAPPING = HERE.parent / "instrument_mapping.csv"
BOOK = HERE / "Trading_book"

HOLD_FOR = {
    "auto_roll": "auto_roll_hold",
    "forced_roll": "forced_roll_hold",
    "+1_auto_roll": "+1_auto_roll_hold",
    "RS_forced_roll": "confirm_forced_roll_hold",
    "LT_forced_roll": "confirm_forced_roll_hold",
    "CS_forced_roll": "confirm_forced_roll_hold",
}
KEEP = ["date", "symbol", "open", "close"]
CONT = ["Continuous_O", "Continuous_C"]


CACHE = HERE / ".cache" / "worksheets"


def _fingerprint(fc, inst: str, start: str, end: str, as_of) -> str:
    h = hashlib.sha256()
    h.update(FC.read_bytes())
    h.update(f"gate={fc.gate(inst)}|"
             f"dead={','.join(sorted(fc.dead_months(inst)))}|".encode())
    h.update(f"{start}|{end}|{as_of}".encode())
    for f in (fc.NOTICE,):
        st = f.stat()
        h.update(f"{f.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    d = fc.CONTRACTS / inst
    for f in sorted(d.glob(f"{inst}-[0-9][0-9][0-9][0-9][A-Z].csv")):
        st = f.stat()
        h.update(f"{f.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def cached_worksheet(fc, inst: str, start: str, end: str, as_of,
                     use_cache: bool = True):
    if not use_cache:
        return fc.worksheet(inst, start, end, as_of=as_of), False
    key = _fingerprint(fc, inst, start, end, as_of)
    pq, kf = CACHE / f"{inst}.parquet", CACHE / f"{inst}.key"
    if pq.is_file() and kf.is_file() and kf.read_text().strip() == key:
        return pl.read_parquet(pq), True
    w = fc.worksheet(inst, start, end, as_of=as_of)
    CACHE.mkdir(parents=True, exist_ok=True)
    w.write_parquet(pq)
    kf.write_text(key)
    return w, False


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rules() -> dict[str, str]:
    t = pl.read_csv(CYCLES, infer_schema_length=0)
    out = {}
    for r in t.iter_rows(named=True):
        rule = (r.get("Roll_Rule") or "").strip()
        if rule:
            out[r["instrument"]] = rule
    return out


EWMAC_PAIRS = [(16, 64), (32, 128), (64, 256)]

VOL_EWM_SPAN = 32
VOL_EWM_MIN = 32
VOL_ROLL_WINDOW = 2560
VOL_ROLL_MIN = 256
VOL_W_SHORT = 0.70
VOL_W_LONG = 0.30

VOL_INNER_WINDOW = 21
VOL_INNER_MIN = 10

GVOL_SPAN = 64
GVOL_STEEPNESS = 10.0
GVOL_TRIGGER = 1.0
GVOL_FLOOR = 0.50

VOV_INNER_WINDOW = 21

VOV_OUTER_WINDOW = 64

VOV_AVG_WINDOW = 256

VOV_DIR_LOOKBACK = 64

SIGNAL_PHI = 10.0
SIGNAL_W = 256
SIGNAL_CAP = 20.0

FDM_CORR_SPAN = 512
FDM_MIN_PERIODS = 256
FDM_FLOOR = 1.0
FDM_CAP = 2.0
FDM_VAR_FLOOR = 0.01

XS_LOOKBACK = 256
XS_MIN_INSTS = 3

TRADING_DAYS_YEAR = 256

SKEW_WINDOW = 256
SKEW_MIN = SKEW_WINDOW


def ewmac(close: pl.Series) -> dict:
    out = {}
    for f, s in EWMAC_PAIRS:
        fast = close.ewm_mean(alpha=2.0 / (f + 1), adjust=False)
        slow = close.ewm_mean(alpha=2.0 / (s + 1), adjust=False)
        out[f"{f}-{s}"] = fast - slow

    total = None
    for name in (f"{f}-{s}" for f, s in EWMAC_PAIRS):
        total = out[name] if total is None else total + out[name]
    out["TS_trend"] = total / len(EWMAC_PAIRS)
    return out


def _blended_std(x: pl.Series) -> pl.Series:
    short = x.ewm_std(span=VOL_EWM_SPAN, adjust=True, bias=False,
                      min_samples=VOL_EWM_MIN, ignore_nulls=False)
    long = x.rolling_std(VOL_ROLL_WINDOW, min_samples=VOL_ROLL_MIN, ddof=1)
    return VOL_W_SHORT * short + VOL_W_LONG * long


def daily_vol_abs(cont_close: pl.Series) -> pl.Series:
    return _blended_std(cont_close.diff())


def s_g_vol(cont_close: pl.Series, vol_abs: pl.Series) -> pl.Series:
    fast = cont_close.diff().ewm_std(span=GVOL_SPAN, adjust=True, bias=False,
                                     min_samples=GVOL_SPAN, ignore_nulls=False)
    ratio = fast / vol_abs
    z = GVOL_STEEPNESS * (ratio - GVOL_TRIGGER)
    gate = 1.0 - (1.0 - GVOL_FLOOR) * (1.0 / (1.0 + (-z).exp()))
    return gate.fill_null(1.0)


def daily_vol(daily_ret: pl.Series,
              window: int = VOL_INNER_WINDOW,
              min_samples: int = VOL_INNER_MIN) -> pl.Series:
    return daily_ret.rolling_std(window, min_samples=min_samples, ddof=1)


def skew(daily_ret: pl.Series, window: int = SKEW_WINDOW,
         min_samples: int = SKEW_MIN) -> pl.Series:
    return -daily_ret.rolling_skew(window, bias=False, min_samples=min_samples)


def vov_inner(daily_ret: pl.Series,
              window: int = VOV_INNER_WINDOW) -> pl.Series:
    return daily_ret.rolling_std(window, min_samples=window, ddof=1)


def vov_smooth(inner: pl.Series,
               window: int = VOV_OUTER_WINDOW) -> pl.Series:
    return inner.rolling_std(window, min_samples=window, ddof=1)


def vov_mean_ann(smooth: pl.Series,
            window: int = VOV_AVG_WINDOW) -> pl.Series:
    return smooth.rolling_mean(window, min_samples=window)


def vov_signal(smooth: pl.Series, mean_ann: pl.Series,
               cont_close: pl.Series,
               lookback: int = VOV_DIR_LOOKBACK) -> pl.Series:
    ratio = -((smooth / mean_ann) - 1.0)
    prev = cont_close.shift(lookback)
    diff = (cont_close - prev).to_numpy().astype(np.float64)
    scale = np.maximum(np.abs(cont_close.to_numpy().astype(np.float64)),
                       np.abs(prev.to_numpy().astype(np.float64)))
    diff = np.where(np.abs(diff) <= 1e-9 * np.maximum(scale, 1.0), 0.0, diff)
    direction = np.where(np.isnan(diff), np.nan, np.where(diff >= 0, 1.0, -1.0))
    return pl.Series(direction).fill_nan(None) * ratio


def normalise_signal(f: pl.Series, phi: float = SIGNAL_PHI,
                     window: int = SIGNAL_W,
                     cap: float | None = SIGNAL_CAP) -> pl.Series:
    x = f.to_numpy().astype(np.float64)
    fin = np.isfinite(x)
    if not fin.any():
        return pl.Series(np.full(x.size, np.nan)).fill_nan(None)
    filled = x.copy()
    first = int(np.argmax(fin))
    filled[first:] = np.where(fin[first:], filled[first:], 0.0)
    present = np.isfinite(filled)
    num = pl.Series(np.where(present, np.abs(filled), 0.0)).rolling_sum(
        window, min_samples=1).to_numpy()
    den = pl.Series(present.astype(np.float64)).rolling_sum(
        window, min_samples=1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_abs = np.where(den >= window, num / np.maximum(den, 1.0), np.nan)
        out = filled * (phi / mean_abs)
        if cap is not None:
            out = np.clip(out, -cap, cap)
    return pl.Series(np.where(np.isfinite(out), out, np.nan)).fill_nan(None)


def trend_sign(ts_sign: pl.Series, xs_sign: pl.Series,
               w_ts: float = 0.5, w_xs: float = 0.5) -> pl.Series:
    return (w_ts * ts_sign + w_xs * xs_sign).clip(-SIGNAL_CAP, SIGNAL_CAP)


def _recursive_sum(v: np.ndarray, decay: float) -> np.ndarray:
    from scipy.signal import lfilter
    return lfilter([1.0], [1.0, -decay], np.asarray(v, dtype=np.float64), axis=0)


def ewm_corr_4(X: np.ndarray, span: float, min_periods: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    T, K = X.shape
    alpha = 2.0 / (span + 1.0)
    decay = 1.0 - alpha
    mask = np.isfinite(X)
    vals = np.where(mask, X, 0.0)
    mf = mask.astype(np.float64)

    cov = np.full((T, K, K), np.nan)
    for i in range(K):
        for j in range(i, K):
            both = mf[:, i] * mf[:, j]
            xi, xj = vals[:, i] * both, vals[:, j] * both
            sw = _recursive_sum(both, decay)
            sw2 = _recursive_sum(both, decay * decay)
            sx = _recursive_sum(xi, decay)
            sy = _recursive_sum(xj, decay)
            sxy = _recursive_sum(xi * xj / np.where(both > 0, 1.0, 1.0), decay)
            with np.errstate(invalid="ignore", divide="ignore"):
                mx, my = sx / sw, sy / sw
                c = sxy / sw - mx * my
                denom = sw * sw - sw2
                c = np.where(denom > 0, c * (sw * sw) / denom, np.nan)
            n_obs = np.cumsum(both)
            c = np.where(n_obs >= max(min_periods, 1), c, np.nan)
            cov[:, i, j] = c
            cov[:, j, i] = c

    d = np.sqrt(np.clip(np.einsum("tii->ti", cov), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / (d[:, :, None] * d[:, None, :])
    corr = np.clip(corr, -1.0, 1.0)
    idx = np.arange(K)
    corr[:, idx, idx] = np.where(np.isfinite(corr[:, idx, idx]), 1.0, np.nan)
    return corr


def pooled_fdm(mean_signals: np.ndarray) -> np.ndarray:
    T = mean_signals.shape[0]
    fdm = np.ones(T, dtype=np.float64)
    corr = ewm_corr_4(mean_signals, FDM_CORR_SPAN, FDM_MIN_PERIODS)
    w = np.full(4, 0.25)
    ok = ~np.isnan(corr).any(axis=(1, 2))
    var_blend = np.einsum("i,tij,j->t", w, np.clip(corr, -1.0, 1.0), w)
    good = ok & (var_blend > FDM_VAR_FLOOR)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.clip(1.0 / np.sqrt(var_blend), FDM_FLOOR, FDM_CAP)
    fdm[good] = vals[good]
    return fdm


_POINTSIZE: dict[str, float] | None = None


def pointsize_of(inst: str) -> float:
    global _POINTSIZE
    if _POINTSIZE is None:
        t = pl.read_csv(MAPPING, infer_schema_length=0)
        _POINTSIZE = {r["norgate_code"]: float(r["pointsize"])
                      for r in t.iter_rows(named=True)
                      if (r.get("pointsize") or "").strip()}
    if inst not in _POINTSIZE:
        raise SystemExit(f"[ABORT] {inst}: no pointsize in {MAPPING.name}")
    return _POINTSIZE[inst]


_COST_RT = None


def cost_rt_of(inst: str) -> float:
    global _COST_RT
    if _COST_RT is None:
        t = pl.read_csv(MAPPING, infer_schema_length=0)
        _COST_RT = {r["norgate_code"]: float(r["total_avg_cost_rt_LocalCurrency"])
                    for r in t.iter_rows(named=True)
                    if (r.get("total_avg_cost_rt_LocalCurrency") or "").strip()}
    if inst not in _COST_RT:
        raise SystemExit(f"[ABORT] {inst}: no total_avg_cost_rt_LocalCurrency "
                         f"in {MAPPING.name}")
    return _COST_RT[inst]


_CURRENCY = None


def currency_of(inst: str) -> str:
    global _CURRENCY
    if _CURRENCY is None:
        t = pl.read_csv(MAPPING, infer_schema_length=0)
        _CURRENCY = {r["norgate_code"]: (r.get("currency") or "").strip().upper()
                     for r in t.iter_rows(named=True)
                     if (r.get("currency") or "").strip()}
    if inst not in _CURRENCY:
        raise SystemExit(f"[ABORT] {inst}: no currency in {MAPPING.name}")
    c = _CURRENCY[inst]
    if c not in FX_CCY:
        raise SystemExit(f"[ABORT] {inst}: currency {c} has no rate -- add it to "
                         f"FX_CCY in {Path(__file__).name}")
    return c


def price_vol_curr_ann(vol_abs: pl.Series, pointsize: float) -> pl.Series:
    return vol_abs * pointsize * np.sqrt(TRADING_DAYS_YEAR)


def xs_return(cont_close: pl.Series, raw_close: pl.Series,
              lookback: int = XS_LOOKBACK) -> pl.Series:
    return (cont_close - cont_close.shift(lookback)) / raw_close.shift(lookback)


def cross_sectional_z(panel: dict[str, pl.Series], dates: pl.Series,
                      min_insts: int = XS_MIN_INSTS) -> dict:
    insts = sorted(panel)
    M = np.column_stack([panel[i].to_numpy().astype(float) for i in insts])
    finite = np.isfinite(M)
    n = finite.sum(axis=1)

    mean = np.full(M.shape[0], np.nan)
    std = np.full(M.shape[0], np.nan)
    usable = n >= max(min_insts, 2)
    if usable.any():
        Mm = np.where(finite, M, np.nan)
        with np.errstate(invalid="ignore"):
            mean[usable] = np.nanmean(Mm[usable], axis=1)
            std[usable] = np.nanstd(Mm[usable], axis=1, ddof=1)
    good = usable & np.isfinite(std) & (std != 0.0)
    z = np.full(M.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        z[good] = (M[good] - mean[good, None]) / std[good, None]

    return ({i: pl.Series(z[:, k]).fill_nan(None) for k, i in enumerate(insts)},
            dates, int(good.sum()), int(M.shape[0]))


def panama(w: pl.DataFrame, col: str) -> tuple[dict, list]:
    sess = (w.select(["date", col])
             .unique(subset=["date"], keep="first")
             .sort("date"))
    order = [str(x) for x in sess.get_column("date").to_list()]
    holds = [h or None for h in sess.get_column(col).to_list()]

    close_of = {}
    for d, sym, c in zip(w.get_column("date").cast(pl.Utf8).to_list(),
                         w.get_column("symbol").to_list(),
                         w.get_column("close").to_list()):
        close_of[(d, sym)] = c

    adj = {d: 0.0 for d in order}
    unmeasured = []
    run = 0.0
    for i in range(len(order) - 1, 0, -1):
        cur, prev = order[i], order[i - 1]
        h_cur, h_prev = holds[i], holds[i - 1]
        if h_cur and h_prev and h_cur != h_prev:
            c_new = close_of.get((cur, h_cur))
            c_old = close_of.get((cur, h_prev))
            if c_new is not None and c_old is not None:
                run += float(c_new) - float(c_old)
            else:
                unmeasured.append((cur, h_prev, h_cur))
        adj[prev] = run
    return adj, unmeasured


def carry_contract(w: pl.DataFrame, col: str) -> tuple[pl.DataFrame, int]:
    exp = next((c for c in ("last_trade", "first_notice") if c in w.columns), None)
    if exp is None:
        raise SystemExit("[ABORT] worksheet has neither 'last_trade' nor "
                         "'first_notice'; cannot order contracts by expiry")
    base = (w.select(["date", "symbol", exp, col])
             .drop_nulls(exp)
             .filter(pl.col(exp) != ""))
    held = (base.filter(pl.col("symbol") == pl.col(col))
                .select(["date", pl.col(exp).alias("_hold_exp")])
                .unique(subset=["date"], keep="first"))
    nxt = (base.join(held, on="date", how="inner")
               .filter(pl.col(exp) > pl.col("_hold_exp"))
               .group_by("date")
               .agg(pl.col("symbol").sort_by([exp, "symbol"]).first()
                      .alias("carry_hold"),
                    pl.col(exp).sort_by([exp, "symbol"]).first().alias("_far_exp"),
                    pl.col("_hold_exp").first()))

    nxt = nxt.with_columns(
        (pl.col("_far_exp").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
         - pl.col("_hold_exp").str.strptime(pl.Date, "%Y-%m-%d", strict=False))
        .dt.total_days().cast(pl.Float64).alias("_dT")).drop("_far_exp", "_hold_exp")

    prices = (w.select(["date",
                        pl.col("symbol").alias("carry_hold"),
                        pl.col("open").cast(pl.Float64).alias("Carry_hold_O"),
                        pl.col("close").cast(pl.Float64).alias("Carry_hold_C")])
               .unique(subset=["date", "carry_hold"], keep="first"))
    nxt = nxt.join(prices, on=["date", "carry_hold"], how="left")

    n_sessions = held.height
    return nxt, n_sessions - nxt.height


def book_one(fc, inst: str, rule: str, start: str, end: str, *,
             as_of: str | None = "auto", held_only: bool = True,
             uniform: bool = True, use_cache: bool = True):
    col = HOLD_FOR.get(rule)
    if col is None:
        raise SystemExit(f"[ABORT] {inst}: Roll_Rule {rule!r} has no hold "
                         f"column in HOLD_FOR")
    w, hit = cached_worksheet(fc, inst, start, end, as_of, use_cache=use_cache)
    if col not in w.columns:
        raise SystemExit(f"[ABORT] {inst}: worksheet has no column {col!r} "
                         f"-- has front_contract.py been renamed under it?")
    n_sessions = w.get_column("date").n_unique()

    adj, unmeasured = panama(w, col)
    carry, n_no_next = carry_contract(w, col)
    out = w.select(KEEP + [col]).with_columns(
        pl.col("date").cast(pl.Utf8).replace_strict(adj, default=0.0)
          .alias("_adj"))
    out = out.with_columns(
        (pl.col("open").cast(pl.Float64) + pl.col("_adj")).alias("Continuous_O"),
        (pl.col("close").cast(pl.Float64) + pl.col("_adj")).alias("Continuous_C"),
    ).drop("_adj")
    held = out.filter(pl.col("symbol") == pl.col(col)).sort("date")
    sig = pl.DataFrame({"date": held.get_column("date")})
    for name, series in ewmac(held.get_column("Continuous_C")).items():
        sig = sig.with_columns(series.alias(name))

    sig = sig.with_columns(
        ((held.get_column("Continuous_C")
          - held.get_column("Continuous_C").shift(1))
         / held.get_column("close").shift(1)).alias("daily_ret"))

    sig = sig.with_columns(skew(sig.get_column("daily_ret")).alias("-Skew"))

    sig = sig.with_columns(
        daily_vol_abs(held.get_column("Continuous_C")).alias("daily_vol_abs"))
    sig = sig.with_columns(
        s_g_vol(held.get_column("Continuous_C"),
                sig.get_column("daily_vol_abs")).alias("s_g_vol"))
    sig = sig.with_columns(
        daily_vol(sig.get_column("daily_ret")).alias("daily_vol"))

    sig = sig.with_columns(
        price_vol_curr_ann(sig.get_column("daily_vol_abs"),
                           pointsize_of(inst)).alias("price_vol_curr_ann"))


    sig = sig.with_columns(
        vov_inner(sig.get_column("daily_ret")).alias("VoV_inner"))
    sig = sig.with_columns(
        vov_smooth(sig.get_column("VoV_inner")).alias("VoV_smooth"))
    sig = sig.with_columns(
        vov_mean_ann(sig.get_column("VoV_smooth")).alias("VoV_mean_ann"))
    sig = sig.with_columns(
        vov_signal(sig.get_column("VoV_smooth"),
                   sig.get_column("VoV_mean_ann"),
                   held.get_column("Continuous_C")).alias("VoV"))


    sig = sig.with_columns(
        xs_return(held.get_column("Continuous_C"),
                  held.get_column("close").cast(pl.Float64)).alias("r256"))

    sig = sig.join(carry, on="date", how="left")

    f1 = held.get_column("close").cast(pl.Float64)
    sig = sig.with_columns(f1.alias("_F1"))
    sig = sig.with_columns(
        pl.when((pl.col("_F1") > 0) & pl.col("Carry_hold_C").is_not_null()
                & (pl.col("_dT") > 0))
          .then((pl.col("_F1") - pl.col("Carry_hold_C")) / pl.col("_F1")
                * (TRADING_DAYS_YEAR / pl.col("_dT")))
          .otherwise(None).alias("Carry")).drop("_F1", "_dT")

    sig = sig.with_columns(
        pl.when(pl.col("Carry").is_null()).then(None)
          .when(pl.col("Carry") > 0).then(pl.lit("B"))
          .when(pl.col("Carry") < 0).then(pl.lit("C"))
          .otherwise(pl.lit("F")).alias("Carry_State"))

    sig = sig.with_columns(
        normalise_signal(sig.get_column("Carry")).alias("Carry_sign"),
        normalise_signal(sig.get_column("-Skew")).alias("Skew_sign"),
        normalise_signal(sig.get_column("VoV")).alias("VoV_sign"),
    )

    if held_only:
        out = held
    out = out.join(sig, on="date", how="left")
    if uniform:
        out = out.rename({col: "hold"})
    return out, n_sessions, unmeasured, hit, n_no_next


def _one(args_tuple):
    (inst, rule, start, end, as_of, held_only, uniform, out_dir,
     use_cache) = args_tuple
    fc = _load(FC, "fc")
    try:
        d, n_avail, unmeasured, hit, n_no_next = book_one(
            fc, inst, rule, start, end, as_of=as_of, held_only=held_only,
            uniform=uniform, use_cache=use_cache)
    except SystemExit as exc:
        return inst, None, 0, 0, [], str(exc), False, None, 0
    return (inst, d.height, d.get_column("date").n_unique(), n_avail,
            unmeasured, None, hit, d, n_no_next)


FX_DIR = HERE / "FX"
FX_CACHE = CACHE.parent / "fx"
HKD_PEG = 7.80


def _src_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

FX_CCY: dict[str, dict] = {
    "USD": dict(inst=None, scale=1.0,   ndu=None,  yf=None,       inv=False,
                fixed=1.0),
    "EUR": dict(inst="6E", scale=1.0,   ndu="&6E", yf="EURUSD=X", inv=False,
                fixed=None),
    "GBP": dict(inst="6B", scale=1.0,   ndu="&6B", yf="GBPUSD=X", inv=False,
                fixed=None),
    "CAD": dict(inst="6C", scale=1.0,   ndu="&6C", yf="CADUSD=X", inv=False,
                fixed=None),
    "AUD": dict(inst="6A", scale=1.0,   ndu="&6A", yf="AUDUSD=X", inv=False,
                fixed=None),
    "JPY": dict(inst="6J", scale=100.0, ndu="&6J", yf="JPYUSD=X", inv=False,
                fixed=None),
    "HKD": dict(inst=None, scale=1.0,   ndu=None,  yf="USDHKD=X", inv=True,
                fixed=1.0 / HKD_PEG),
}

FX_WATCH_BP = 100.0
FX_ALERT_BP = 300.0


def _prefer_parquet(base: Path) -> Path:
    pq, csv = base.with_suffix(".parquet"), base.with_suffix(".csv")
    if pq.is_file() and (not csv.is_file()
                         or pq.stat().st_mtime_ns >= csv.stat().st_mtime_ns):
        return pq
    return csv


FX_SCHEMA = {"date": pl.Utf8, "Derived_Rate": pl.Float64, "NDU_Rate": pl.Float64,
             "YF_Rate": pl.Float64, "NDU_diff_bp": pl.Float64,
             "YF_diff_bp": pl.Float64, "Status": pl.Utf8}


def load_fx(ccy: str, d: Path | None = None) -> pl.DataFrame:
    p = _prefer_parquet((d or FX_DIR) / ccy)
    return (pl.read_parquet(p) if p.suffix == ".parquet"
            else pl.read_csv(p, schema=FX_SCHEMA))


BOOK_TEXT_COLS = {"date", "symbol", "hold", "carry_hold", "Carry_State"}


def load_book(inst: str, d: Path | None = None) -> pl.DataFrame:
    p = _prefer_parquet((d or BOOK) / inst)
    if p.suffix == ".parquet":
        return pl.read_parquet(p)
    with open(p, "r", encoding="utf-8") as fh:
        cols = fh.readline().rstrip("\r\n").split(",")
    return pl.read_csv(p, schema={
        c: (pl.Utf8 if c in BOOK_TEXT_COLS or c.endswith("_hold") else pl.Float64)
        for c in cols})


def fx_derived(w: pl.DataFrame, scale: float) -> pl.DataFrame:
    d = (w.select(["date", "close", "till_last_trade_cd"])
          .drop_nulls()
          .filter((pl.col("close") > 0) & (pl.col("till_last_trade_cd") >= 0))
          .sort(["date", "till_last_trade_cd"])
          .group_by("date", maintain_order=True)
          .agg([pl.col("close").head(2).alias("f"),
                pl.col("till_last_trade_cd").head(2).alias("t")])
          .filter(pl.col("f").list.len() == 2))
    if not d.height:
        return pl.DataFrame({"date": [], "Derived_Rate": []},
                            schema={"date": pl.Utf8, "Derived_Rate": pl.Float64})
    f1 = pl.col("f").list.get(0) / scale
    f2 = pl.col("f").list.get(1) / scale
    t1 = pl.col("t").list.get(0).cast(pl.Float64) / 365.25
    t2 = pl.col("t").list.get(1).cast(pl.Float64) / 365.25
    slope = (f2.log() - f1.log()) / (t2 - t1)
    return (d.with_columns(
                pl.when(t2 > t1)
                  .then((f1.log() - slope * t1).exp())
                  .otherwise(None).alias("Derived_Rate"))
             .select(["date", "Derived_Rate"]))


def _fx_cached(name: str, key: str, build):
    pq, kf = FX_CACHE / f"{name}.parquet", FX_CACHE / f"{name}.key"
    if pq.is_file() and kf.is_file() and kf.read_text().strip() == key:
        return pl.read_parquet(pq), True
    df = build()
    if df.height:
        FX_CACHE.mkdir(parents=True, exist_ok=True)
        df.write_parquet(pq)
        kf.write_text(key)
    return df, False


def fx_ndu(symbol: str, scale: float, col: str) -> pl.DataFrame:
    import platform
    n = platform.node()
    if not n.isascii():
        platform.node = lambda _v=n.encode("ascii", "ignore").decode(): _v
    import norgatedata as nd

    r = nd.price_timeseries(symbol, timeseriesformat="numpy-recarray",
                            datetimeformat="datetime64ns",
                            padding_setting=nd.PaddingType.NONE)
    if r is None or not len(r):
        return pl.DataFrame({"date": [], col: []},
                            schema={"date": pl.Utf8, col: pl.Float64})
    ds = np.datetime_as_string(np.asarray(r["Date"]).astype("datetime64[D]"),
                               unit="D")
    cl = np.asarray(r["Close"], dtype=float) / scale
    return (pl.DataFrame({"date": [str(x) for x in ds], col: cl})
              .filter(pl.col(col).is_finite() & (pl.col(col) > 0)))


def fx_yf(ticker: str, invert: bool, col: str) -> pl.DataFrame:
    import yfinance as yf
    h = yf.Ticker(ticker).history(period="max", interval="1d")
    if h is None or not len(h):
        return pl.DataFrame({"date": [], col: []},
                            schema={"date": pl.Utf8, col: pl.Float64})
    v = np.asarray(h["Close"], dtype=float)
    df = pl.DataFrame({"date": [str(i)[:10] for i in h.index], col: v})
    df = df.filter(pl.col(col).is_finite() & (pl.col(col) > 0))
    return df.with_columns((1.0 / pl.col(col)).alias(col)) if invert else df


IRX_DIR = HERE / "IRX"
IRX_SYMBOL = "%IRX"
IRX_SCALE = 100.0
IRX_BILL_DAYS = 91


def irx_series(col: str = "irx_pct") -> pl.DataFrame:
    import platform
    n = platform.node()
    if not n.isascii():
        platform.node = lambda _v=n.encode("ascii", "ignore").decode(): _v
    import norgatedata as nd

    r = nd.price_timeseries(IRX_SYMBOL, timeseriesformat="numpy-recarray",
                            datetimeformat="datetime64ns",
                            padding_setting=nd.PaddingType.NONE)
    if r is None or not len(r):
        return pl.DataFrame({"date": [], col: []},
                            schema={"date": pl.Utf8, col: pl.Float64})
    ds = np.datetime_as_string(np.asarray(r["Date"]).astype("datetime64[D]"),
                               unit="D")
    c = np.asarray(r["Close"], dtype=float)
    return (pl.DataFrame({"date": [str(x) for x in ds], col: c})
              .filter(pl.col(col).is_finite()))


def build_irx(grid: pl.Series | None, out_dir: Path,
              use_cache: bool = True) -> pl.DataFrame | None:
    if grid is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    key = f"{_src_hash()}|{IRX_SYMBOL}|{grid[-1] if len(grid) else 'none'}"
    try:
        raw, hit = (_fx_cached("irx", key, irx_series) if use_cache
                    else (irx_series(), False))
    except Exception as exc:
        print(f"  [SKIP] IRX ({IRX_SYMBOL}): {type(exc).__name__}: {exc}")
        return None
    if not raw.height:
        print(f"  [SKIP] IRX ({IRX_SYMBOL}): vendor returned nothing")
        return None

    d = pl.col("irx_pct") / IRX_SCALE
    denom = 360.0 - d * IRX_BILL_DAYS
    df = (pl.DataFrame({"date": grid.to_list()}).sort("date")
            .join_asof(raw.sort("date"), on="date", strategy="backward")
            .with_columns([
                (365.0 * d / denom * IRX_SCALE).alias("irx_bey_pct"),
                (d / denom).alias("rf_cal_day"),
                (pl.col("date").str.to_date().shift(-1)
                 - pl.col("date").str.to_date()).dt.total_days()
                .alias("cal_days_to_next")]))
    df = df.with_columns(
        (pl.col("rf_cal_day") * pl.col("cal_days_to_next"))
        .alias("rf_accrual_next"))
    df.write_csv(out_dir / "IRX.csv")
    df.write_parquet(out_dir / "IRX.parquet")
    v = df.get_column("irx_pct")
    print(f"\n  IRX -> {out_dir}    {df.height:,} sessions"
          f"   {'cache hit' if hit else 'fetched'}")
    print(f"    {IRX_SYMBOL}  {raw.get_column('date').min()} .. "
          f"{raw.get_column('date').max()}   {raw.height:,} vendor bars")
    print(f"    on grid: mean {v.mean():.3f}%   min {v.min():.3f}%   "
          f"max {v.max():.3f}%   last {v[-1]:.3f}%")
    b = df.get_column("irx_bey_pct")
    print(f"    BEY    : mean {b.mean():.3f}%   last {b[-1]:.3f}%"
          f"   (+{b[-1] - v[-1]:.3f}pp over the quoted discount)")
    print(f"    rf_cal_day last {df.get_column('rf_cal_day')[-1]:.10f}"
          f"   x 365 = {df.get_column('rf_cal_day')[-1] * 365:.3%}")
    cd = df.get_column("cal_days_to_next").drop_nulls()
    print(f"    calendar gap to next session: mean {cd.mean():.2f}d"
          f"   max {cd.max()}d   (1 midweek, 3 over a weekend)")
    return df


def build_fx(fc, as_of, start: str, end: str, grid: pl.Series | None,
             out_dir: Path, use_cache: bool = True,
             checks: bool = True) -> dict[str, pl.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    edge = as_of or "none"
    frames: dict[str, pl.DataFrame] = {}
    rows = []
    for ccy, m in FX_CCY.items():
        if m["inst"] is not None:
            w, _ = cached_worksheet(fc, m["inst"], start, end, as_of,
                                    use_cache=use_cache)
            nat = fx_derived(w, m["scale"])
        else:
            base = grid if grid is not None else pl.Series("date", [])
            nat = pl.DataFrame({"date": base.to_list()}).with_columns(
                pl.lit(m["fixed"], dtype=pl.Float64).alias("Derived_Rate"))
        nat = nat.sort("date")

        for col, key, fn in (
                ("NDU_Rate", m["ndu"],
                 (lambda mm=m: fx_ndu(mm["ndu"], mm["scale"], "NDU_Rate"))),
                ("YF_Rate", m["yf"],
                 (lambda mm=m: fx_yf(mm["yf"], mm["inv"], "YF_Rate")))):
            got = None
            if checks and key:
                try:
                    got, _hit = _fx_cached(f"{col[:3].lower()}_{ccy}",
                                           f"{key}|{m['scale']}|{m['inv']}|{edge}",
                                           fn) if use_cache else (fn(), False)
                except Exception as exc:
                    print(f"  [SKIP] {ccy} {col} ({key}): "
                          f"{type(exc).__name__}: {exc}")
                    got = None
            nat = (nat.join(got.sort("date"), on="date", how="left")
                   if got is not None and got.height
                   else nat.with_columns(pl.lit(None, dtype=pl.Float64).alias(col)))

        nat = nat.with_columns([
            ((pl.col("Derived_Rate") / pl.col("NDU_Rate") - 1.0) * 1e4)
                .alias("NDU_diff_bp"),
            ((pl.col("Derived_Rate") / pl.col("YF_Rate") - 1.0) * 1e4)
                .alias("YF_diff_bp")])
        worst = pl.max_horizontal(pl.col("NDU_diff_bp").abs(),
                                  pl.col("YF_diff_bp").abs())
        nat = nat.with_columns(
            pl.when(pl.col("Derived_Rate").is_null()).then(pl.lit("NO_DERIVED"))
              .when(worst.is_null()).then(pl.lit("UNCHECKED"))
              .when(worst >= FX_ALERT_BP).then(pl.lit("ALERT"))
              .when(worst >= FX_WATCH_BP).then(pl.lit("WATCH"))
              .otherwise(pl.lit("OK")).alias("Status"))

        if grid is not None and m["inst"] is not None:
            nat = (pl.DataFrame({"date": grid.to_list()}).sort("date")
                     .join_asof(nat, on="date", strategy="backward"))
            nat = nat.with_columns(
                pl.col("Status").fill_null(pl.lit("NO_DERIVED")))

        nat = nat.select(["date", "Derived_Rate", "NDU_Rate", "YF_Rate",
                          "NDU_diff_bp", "YF_diff_bp", "Status"])

        if m["inst"] is not None:
            nat = nat.filter(
                pl.col("date") >= pl.lit(
                    nat.filter(pl.col("Derived_Rate").is_not_null())
                       .get_column("date").min() or ""))
        nat.write_csv(out_dir / f"{ccy}.csv")
        nat.write_parquet(out_dir / f"{ccy}.parquet")
        frames[ccy] = nat

        nb = nat.get_column("NDU_diff_bp").drop_nulls().abs()
        yb = nat.get_column("YF_diff_bp").drop_nulls().abs()
        both = pl.concat([nb, yb])
        rows.append((ccy, nat.height,
                     nat.get_column("Derived_Rate").drop_nulls().len(),
                     len(nb), len(yb),
                     both.mean() if len(both) else None,
                     both.quantile(0.95) if len(both) else None,
                     both.max() if len(both) else None,
                     nat.filter(pl.col("Status") == "WATCH").height,
                     nat.filter(pl.col("Status") == "ALERT").height,
                     nat.get_column("Derived_Rate").drop_nulls().tail(1)))

    hdr = (f"\n{'ccy':<6}{'rows':>9}{'derived':>9}{'NDU':>8}{'yf':>8}"
           f"{'mean bp':>10}{'p95 bp':>9}{'max bp':>10}{'WATCH':>7}{'ALERT':>7}"
           f"{'last rate':>14}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    dash = lambda v, w, p=1: (f"{'-':>{w}}" if v is None or v != v
                              else f"{v:>{w},.{p}f}")
    for (c, n, nd_, a, b, mu, p95, mx, wt, al, last) in rows:
        print(f"{c:<6}{n:>9,}{nd_:>9,}{a:>8,}{b:>8,}{dash(mu, 10)}"
              f"{dash(p95, 9)}{dash(mx, 10)}{wt:>7,}{al:>7,}"
              f"{(last[0] if len(last) else float('nan')):>14.6f}")
    print("-" * (len(hdr) - 1))
    n_alert = sum(r[9] for r in rows)
    print(f"  FX -> {out_dir}    {len(frames)} rates, one file each")
    print(f"  the book reads Derived_Rate; NDU_Rate and YF_Rate are checks")
    if n_alert:
        print(f"  [WATCH] {n_alert:,} rows disagree by >= {FX_ALERT_BP:.0f}bp "
              f"-- inspect with Status == 'ALERT'")
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default=None,
                    help="one instrument; default is every ruled instrument")
    ap.add_argument("--start", default="1900-01-01")
    ap.add_argument("--end", default="2100-01-01")
    ap.add_argument("--out", default=str(BOOK))
    ap.add_argument("--all-rows", action="store_true",
                    help="every listed month, not just the held contract")
    ap.add_argument("--keep-rule-name", action="store_true",
                    help="hold column keeps the rule's own name per instrument")
    ap.add_argument("--as-of", default="auto", dest="as_of",
                    help="square the panel off at this date; 'none' to disable")
    ap.add_argument("--no-cache", action="store_true",
                    help="rebuild every worksheet, ignoring the cache")
    ap.add_argument("--clear-cache", action="store_true",
                    help="delete the worksheet cache and exit")
    ap.add_argument("--no-fx", action="store_true",
                    help="skip the FX rate files entirely")
    ap.add_argument("--no-fx-checks", action="store_true",
                    help="FX from the local panel only -- no vendor, no network")
    ap.add_argument("--jobs", type=int, default=2,
                    help="worker processes. DEFAULT 2. Bound by MEMORY, not "
                         "cores -- see the note in main()")
    args = ap.parse_args()

    if args.clear_cache:
        import shutil
        if CACHE.is_dir():
            n = len(list(CACHE.glob("*.parquet")))
            shutil.rmtree(CACHE)
            print(f"cache cleared: {n} worksheet(s) removed from {CACHE}")
        else:
            print(f"no cache at {CACHE}")
        return 0

    fc = _load(FC, "fc")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    as_of = None if args.as_of == "none" else args.as_of
    if as_of == "auto":
        as_of, edge = fc.panel_as_of()
        print("PANEL EDGE")
        fc.report_edge(as_of, edge)
        print("")

    rule_of = rules()
    if args.instrument:
        if args.instrument not in rule_of:
            raise SystemExit(f"[ABORT] {args.instrument} has no Roll_Rule in "
                             f"{CYCLES.name}")
        rule_of = {args.instrument: rule_of[args.instrument]}

    print(f"trading book -> {out_dir}    as_of {as_of or 'none'}")
    n_cached = len(list(CACHE.glob("*.parquet"))) if CACHE.is_dir() else 0
    print(f"  building {len(rule_of)} instrument(s) on {max(1, args.jobs)} "
          f"worker(s); worksheet cache holds {n_cached}"
          + ("  (cold -- first rows in a few seconds, full run ~80s+)"
             if n_cached < len(rule_of) else "  (warm)"),
          flush=True)
    hdr = (f"\n{'inst':<8}{'rule':<18}{'hold column':<26}{'rows':>10}"
           f"{'sessions':>10}{'sec':>6}{'done':>9}")
    print(hdr, flush=True)
    print("-" * (len(hdr) - 1), flush=True)

    tot_rows = tot_sess = 0
    written, holes, unmeasured_all = [], [], []
    jobs = max(1, args.jobs)
    if jobs > 1:
        os.environ.setdefault("POLARS_MAX_THREADS", "2")
    tasks = [(inst, rule, args.start, args.end, as_of, not args.all_rows,
              not args.keep_rule_name, str(out_dir), not args.no_cache)
             for inst, rule in sorted(rule_of.items())]

    from contextlib import nullcontext
    HEARTBEAT = 5.0

    def _serial():
        for t in tasks:
            yield _one(t)

    def _parallel(pool):
        from concurrent.futures import wait as _fwait
        futs = [pool.submit(_one, t) for t in tasks]
        for i, f in enumerate(futs):
            while not f.done():
                _fwait([f], timeout=HEARTBEAT)
                if not f.done():
                    print(f"  ... building, {i} of {len(futs)} done, "
                          f"{time.time() - t_build:.0f}s elapsed", flush=True)
            yield f.result()

    hits = 0
    frames = {}
    no_next = []
    t_build = time.time()
    n_task = len(tasks)
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        _pool = ProcessPoolExecutor(max_workers=jobs)
        ctx, stream = _pool, _parallel(_pool)
    else:
        ctx, stream = nullcontext(), _serial()
    with ctx:
        for k, (inst, h, n_sess, n_avail, unm, err, hit, frame,
                nnn) in enumerate(stream, 1):
            if err:
                print(f"{inst:<8}{err}", flush=True)
                continue
            rule = rule_of[inst]
            print(f"{inst:<8}{rule:<18}{HOLD_FOR[rule]:<26}{h:>10,}"
                  f"{n_sess:>10,}{time.time() - t_build:>6.0f}"
                  f"{k:>5}/{n_task}", flush=True)
            if unm:
                unmeasured_all.append((inst, unm))
            if not args.all_rows and n_sess < n_avail:
                holes.append((inst, n_avail - n_sess, n_avail))
            tot_rows += h
            tot_sess += n_sess
            hits += int(hit)
            frames[inst] = frame
            if nnn:
                no_next.append((inst, nnn, n_sess))
            written.append(inst)

    print("-" * (len(hdr) - 1))

    xs_rows = xs_total = 0
    grid = None
    if frames:
        grid = pl.Series("date", sorted(
            set().union(*(set(f.get_column("date").to_list())
                          for f in frames.values()))))
        panel = {}
        for inst, f in frames.items():
            per = (f.select(["date", "r256"])
                    .unique(subset=["date"], keep="first"))
            aligned = (pl.DataFrame({"date": grid})
                       .join(per, on="date", how="left"))
            panel[inst] = aligned.get_column("r256")
        z, _, xs_rows, xs_total = cross_sectional_z(panel, grid)
        for inst in frames:
            zf = pl.DataFrame({"date": grid, "XS_trend": z[inst]})
            f = frames[inst].join(zf, on="date", how="left").drop("r256")
            per = (f.select(["date", "TS_trend", "XS_trend"])
                    .unique(subset=["date"], keep="first").sort("date"))
            per = per.with_columns(
                normalise_signal(per.get_column("TS_trend"),
                                 cap=None).alias("TS_trend_sign_UNCAPPED"),
                normalise_signal(per.get_column("XS_trend"),
                                 cap=None).alias("XS_trend_sign_UNCAPPED"))
            per = per.with_columns(
                trend_sign(per.get_column("TS_trend_sign_UNCAPPED"),
                           per.get_column("XS_trend_sign_UNCAPPED")).alias("Trend_sign"))
            f = f.join(
                per.select(["date", "TS_trend_sign_UNCAPPED", "XS_trend_sign_UNCAPPED", "Trend_sign"]),
                on="date", how="left")

            frames[inst] = f.with_columns(
                (0.25 * pl.col("Trend_sign") + 0.25 * pl.col("Carry_sign")
                 + 0.25 * pl.col("Skew_sign") + 0.25 * pl.col("VoV_sign"))
                .alias("Sign_raw"))

        gd = grid.to_list()
        pos = {d: k for k, d in enumerate(gd)}
        panels = {c: np.full((len(gd), len(frames)), np.nan)
                  for c in ("Trend_sign", "Carry_sign", "Skew_sign", "VoV_sign")}
        for col, inst in enumerate(sorted(frames)):
            fr = (frames[inst].select(["date"] + list(panels))
                  .unique(subset=["date"], keep="first"))
            rows = [pos.get(d) for d in fr.get_column("date").to_list()]
            for c in panels:
                vals = fr.get_column(c).to_numpy().astype(float)
                for r, v in zip(rows, vals):
                    if r is not None:
                        panels[c][r, col] = v
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.filterwarnings("ignore", message="Mean of empty slice",
                                    category=RuntimeWarning)
            means = np.column_stack([np.nanmean(panels[c], axis=1)
                                     for c in ("Trend_sign", "Carry_sign",
                                               "Skew_sign", "VoV_sign")])
        fdm = pooled_fdm(means)
        fdm_f = pl.DataFrame({"date": grid, "fdm_raw": fdm})
        for inst in frames:
            frames[inst] = frames[inst].join(fdm_f, on="date", how="left")

        for inst in frames:
            frames[inst] = frames[inst].with_columns(
                (pl.col("Sign_raw") * pl.col("fdm_raw")).alias("fdm_norm"))

        for inst in frames:
            frames[inst] = frames[inst].with_columns(
                normalise_signal(
                    frames[inst].get_column("fdm_norm")).alias("FDM_MASTER"))

        for inst in frames:
            frames[inst] = frames[inst].with_columns(
                pl.col("FDM_MASTER")
                  .ewm_mean(alpha=0.5, adjust=False, min_samples=1,
                            ignore_nulls=False)
                  .alias("FDM_MASTER_smooth"))

        for inst in frames:
            frames[inst] = frames[inst].with_columns(
                pl.col("FDM_MASTER_smooth").clip(-SIGNAL_CAP, SIGNAL_CAP)
                  .alias("SIGNAL"))
        n_active = int((fdm > FDM_FLOOR).sum())

    fx_frames = {}
    if not args.no_fx:
        try:
            fx_frames = build_fx(fc, as_of, args.start, args.end, grid, FX_DIR,
                                 use_cache=not args.no_cache,
                                 checks=not args.no_fx_checks)

        except Exception as exc:
            print(f"\n  [WARN] FX build failed: {type(exc).__name__}: {exc}")
            print("         Books will be written WITHOUT FX_rate. "
                  "Rerun, or use --no-fx to skip deliberately.")

    if not (args.no_fx or args.no_fx_checks):
        try:
            build_irx(grid, IRX_DIR, use_cache=not args.no_cache)
        except Exception as exc:
            print(f"\n  [WARN] IRX build failed: "
                  f"{type(exc).__name__}: {exc}")
            print("         Books and FX are unaffected; stage 3 has no rate.")

    fx_missing = []
    if fx_frames:
        for inst in list(frames):
            ccy = currency_of(inst)
            r = (fx_frames[ccy].select(["date", pl.col("Derived_Rate")
                                                  .alias("FX_rate")]))
            frames[inst] = frames[inst].join(r, on="date", how="left")
            frames[inst] = frames[inst].with_columns(
                (pl.col("price_vol_curr_ann") * pl.col("FX_rate"))
                .alias("price_vol_USD_ann"))
            n = frames[inst].get_column("FX_rate").null_count()
            if n:
                fx_missing.append((inst, ccy, n, frames[inst].height))

    for inst, f in frames.items():
        f.write_csv(out_dir / f"{inst}.csv")
        f.write_parquet(out_dir / f"{inst}.parquet")

    if fx_frames:
        by_ccy = {}
        for inst in frames:
            by_ccy[currency_of(inst)] = by_ccy.get(currency_of(inst), 0) + 1
        print(f"\n  FX_rate attached to {len(frames)} book(s): "
              + "  ".join(f"{c} {n}" for c, n in sorted(by_ccy.items(),
                                                        key=lambda kv: -kv[1])))
        if fx_missing:
            print(f"  [NOTE] FX_rate is null on some sessions -- the book "
                  f"predates the currency's future, which has no rate to carry:")
            for inst, ccy, n, tot in sorted(fx_missing,
                                            key=lambda x: -x[2]):
                print(f"     {inst:<10}{ccy}  {n:,} of {tot:,} sessions")

    print("")
    print(f"{len(written)} instruments, {tot_rows:,} rows, "
          f"{tot_sess:,} instrument-sessions")
    if frames:
        pct = 100.0 * xs_rows / xs_total if xs_total else 0.0
        print(f"  XS_trend: {xs_rows:,} of {xs_total:,} union sessions scored "
              f"({pct:.1f}%)")
        if len(frames) < XS_MIN_INSTS:
            print(f"  [WARN] only {len(frames)} instrument(s) in this run, so "
                  f"there is no cross-section to score against and XS_trend is "
                  f"ENTIRELY NULL. Eq 3.13 needs >= {XS_MIN_INSTS}; run the "
                  f"full book for a usable column.")
    print(f"  worksheet cache: {hits} hit, {len(written)-hits} rebuilt")
    if no_next:
        n = sum(k for _, k, _ in no_next)
        print("")
        print(f"  [WARN] carry_hold is NULL on {n:,} session(s) across "
              f"{len(no_next)} instrument(s): the held contract was the "
              f"furthest-dated month listed, so there is no next leg:")
        for inst, k, tot in sorted(no_next, key=lambda r: -r[1])[:8]:
            print(f"     {inst:<8}{k:>7,} of {tot:,} sessions")
    if holes:
        print("")
        print(f"  [WARN] {len(holes)} instrument(s) HOLD NOTHING on some "
              f"sessions -- those sessions have no row at all:")
        for inst, n, tot in holes:
            print(f"     {inst:<8}{n:>7,} of {tot:,} sessions missing")
    elif not args.all_rows:
        print("  every session has a held contract; no holes")
    if unmeasured_all:
        n = sum(len(u) for _, u in unmeasured_all)
        print("")
        print(f"  [WARN] {n} roll(s) across {len(unmeasured_all)} instrument(s) "
              f"had no quote for the outgoing contract on the roll date, so the "
              f"gap could not be measured and the jump stays in the series:")
        for inst, u in unmeasured_all[:5]:
            for dt, o, nw in u[:3]:
                print(f"     {inst:<8}{dt}  {o} -> {nw}")
    if not args.instrument:
        missing = sorted(set(pl.read_csv(CYCLES, infer_schema_length=0)
                             .get_column("instrument").to_list()) - set(written))
        if missing:
            print(f"NOT written ({len(missing)}), no Roll_Rule: "
                  f"{', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
