from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
BOOK_PY = HERE.parent / "2_Engine" / "trading_book.py"
POS = HERE / "Positions"
PORTFOLIO = HERE / "Portfolio.csv"
IRX_FILE = BOOK_PY.parent / "IRX" / "IRX.parquet"

NAV_0 = 100_000_000.0

TAU = 0.20

IDM_CAP = 4.0
IDM_CORR_SPAN = 512
IDM_MIN_PERIODS = 256
IDM_VAR_FLOOR = 0.01

GDD_LOOKBACK = 64
GDD_THRESHOLD = -0.10
GDD_STEEPNESS = 10.0
GDD_FLOOR = 0.50

BUFFER = 0.10

START_DATE = "2026-01-02"
MIN_SESSIONS = 256 * 5

SIGNAL_PHI = 10.0


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ewm_corr_sum(X: np.ndarray, active: np.ndarray, span: float,
                 min_periods: int, chunk: int = 128):
    from scipy.signal import lfilter

    T, K = X.shape
    alpha = 2.0 / (span + 1.0)
    decay = 1.0 - alpha
    mask = np.isfinite(X)
    vals = np.where(mask, X, 0.0)
    mf = mask.astype(np.float64)

    def acc(a, d):
        return lfilter([1.0], [1.0, -d], a, axis=0)

    sw = acc(mf, decay)
    sw2 = acc(mf, decay * decay)
    sx = acc(vals * mf, decay)
    sxx = acc(vals * vals * mf, decay)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = sx / sw
        var = sxx / sw - m * m
        den = sw * sw - sw2
        var = np.where(den > 0, var * (sw * sw) / den, np.nan)
    n_obs = np.cumsum(mf, axis=0)
    var = np.where(n_obs >= max(min_periods, 1), var, np.nan)
    sd = np.sqrt(np.clip(var, 0.0, None))
    ok = np.isfinite(sd) & (sd > 0)
    act = active & ok

    n_act = act.sum(axis=1).astype(np.float64)
    pair_sum = np.zeros(T, dtype=np.float64)
    n_undef = np.zeros(T, dtype=np.float64)

    iu, ju = np.triu_indices(K, k=1)
    for s in range(0, len(iu), chunk):
        i = iu[s:s + chunk]
        j = ju[s:s + chunk]
        both = mf[:, i] * mf[:, j]
        xi = vals[:, i] * both
        xj = vals[:, j] * both
        b_sw = acc(both, decay)
        b_sw2 = acc(both, decay * decay)
        b_sx = acc(xi, decay)
        b_sy = acc(xj, decay)
        b_sxy = acc(xi * xj, decay)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = b_sx / b_sw
            my = b_sy / b_sw
            cov = b_sxy / b_sw - mx * my
            den = b_sw * b_sw - b_sw2
            cov = np.where(den > 0, cov * (b_sw * b_sw) / den, np.nan)
            corr = cov / (sd[:, i] * sd[:, j])
        corr = np.where(np.cumsum(both, axis=0) >= max(min_periods, 1),
                        corr, np.nan)
        corr = np.clip(corr, -1.0, 1.0)
        live = act[:, i] & act[:, j]
        bad = live & ~np.isfinite(corr)
        n_undef += bad.sum(axis=1)
        pair_sum += np.where(live, np.where(np.isfinite(corr), corr, 1.0),
                             0.0).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        wcw = (n_act + 2.0 * pair_sum) / (n_act * n_act)
    wcw = np.where(n_act > 0, wcw, np.nan)
    return wcw, n_act, n_undef, act


def idm_from(wcw: np.ndarray) -> np.ndarray:
    out = np.ones_like(wcw)
    good = np.isfinite(wcw) & (wcw > IDM_VAR_FLOOR)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[good] = np.minimum(IDM_CAP, 1.0 / np.sqrt(wcw[good]))
    return out


def panel(tb, insts: list[str]):
    cols = ["date", "symbol", "SIGNAL", "price_vol_USD_ann", "daily_ret",
            "Continuous_C", "Continuous_O", "close", "FX_rate", "s_g_vol"]
    per = {i: tb.load_book(i).select(cols) for i in insts}
    dates = sorted(set().union(*(set(d.get_column("date").to_list())
                                 for d in per.values())))
    pos = {d: k for k, d in enumerate(dates)}
    T, K = len(dates), len(insts)
    out = {c: np.full((T, K), np.nan) for c in
           ("SIGNAL", "sigma", "ret", "contc", "conto", "raw", "fx", "gvol")}
    sym = np.full((T, K), None, dtype=object)
    src = {"SIGNAL": "SIGNAL", "sigma": "price_vol_USD_ann", "ret": "daily_ret",
           "contc": "Continuous_C", "conto": "Continuous_O", "raw": "close",
           "fx": "FX_rate", "gvol": "s_g_vol"}
    for k, i in enumerate(insts):
        d = per[i]
        idx = np.array([pos[x] for x in d.get_column("date").to_list()])
        for name, col in src.items():
            v = d.get_column(col).to_numpy().astype(np.float64)
            out[name][idx, k] = v
        for r, s in zip(idx, d.get_column("symbol").to_list()):
            sym[r, k] = s
    return dates, out, sym


def _ffill(a: np.ndarray) -> np.ndarray:
    m = np.isfinite(a)
    idx = np.where(m, np.arange(a.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return a[idx, np.arange(a.shape[1])[None, :]]


def simulate(dates, P, sym, insts, tb, nav0: float, span: int,
             tau: float = TAU, compound: bool = True,
             use_gvol: bool = True, use_gdd: bool = True,
             use_buffer: bool = True, buffer: float = BUFFER,
             start_date: str = START_DATE):
    T, K = P["SIGNAL"].shape
    ps = np.array([tb.pointsize_of(i) for i in insts], dtype=np.float64)

    seen = np.cumsum(np.isfinite(P["contc"]), axis=0)
    old_enough = seen >= MIN_SESSIONS
    after_start = np.array([d >= start_date for d in dates])[:, None]
    tradable = old_enough & after_start

    sizeable_today = (np.isfinite(P["SIGNAL"]) & np.isfinite(P["sigma"])
                      & (P["sigma"] > 0) & np.isfinite(P["fx"]))
    began = np.cumsum(sizeable_today, axis=0) >= 1
    has_bar = np.isfinite(P["contc"])
    last_bar = np.where(has_bar.any(axis=0), has_bar.shape[0] - 1
                        - np.argmax(has_bar[::-1], axis=0), -1)
    alive = np.arange(len(dates))[:, None] <= last_bar[None, :]
    in_universe = tradable & began & alive
    started_t = after_start[:, 0]

    wcw, n_act, n_undef, uni = ewm_corr_sum(P["ret"], in_universe, span,
                                            IDM_MIN_PERIODS)
    act = uni & sizeable_today
    idm = idm_from(wcw)

    gvol = (np.where(np.isfinite(P["gvol"]), P["gvol"], 1.0)
            if use_gvol else np.ones_like(P["gvol"]))
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(n_act > 0, 1.0 / n_act, 0.0)[:, None]
        k = (np.where(act, P["SIGNAL"], 0.0) / SIGNAL_PHI) * (
            tau * w * idm[:, None]) / np.where(P["sigma"] > 0, P["sigma"], np.nan)
        k = k * gvol
    k = np.where(np.isfinite(k) & act, k, 0.0)

    _C = _ffill(P["contc"])
    _O = _ffill(P["conto"])
    dP = np.diff(_C, axis=0, prepend=np.nan)
    _bar = np.array([[x is not None for x in row] for row in sym], dtype=bool)
    gapP = np.full_like(_C, np.nan)
    gapP[1:] = _O[1:] - _C[:-1]
    dayP = _C - _O
    gapP = np.where(_bar, gapP, 0.0)
    dayP = np.where(_bar, dayP, 0.0)
    fx = _ffill(P["fx"])

    symf = np.empty_like(sym)
    _last = np.array([None] * K, dtype=object)
    for _t in range(T):
        _row = sym[_t]
        _last = np.where(np.equal(_row, None), _last, _row)
        symf[_t] = _last
    rolled = np.zeros((T, K), dtype=bool)
    rolled[1:] = (np.not_equal(symf[1:], symf[:-1])
                  & ~np.equal(symf[1:], None) & ~np.equal(symf[:-1], None))

    has_bar = _bar
    gapN = np.zeros((T, K))
    _own_last = np.zeros(K)
    _own_prev = np.zeros(K)
    _cost_pending = np.zeros(K)
    cost_chg = np.zeros((T, K))

    nav = np.full(T, np.nan)
    equity = np.full(T, np.nan)
    pnl = np.zeros(T)
    N = np.zeros((T, K))
    cum = np.zeros((T + 1, K))
    gdd = np.ones((T, K))
    tgt = np.zeros((T, K))
    cost_rt = np.array([tb.cost_rt_of(i) for i in insts],
                       dtype=np.float64) / 2.0
    cost_m = np.zeros((T, K))
    rf_next = np.zeros(T)
    if IRX_FILE.is_file():
        _ix = pl.read_parquet(IRX_FILE)
        _m = dict(zip(_ix.get_column('date').to_list(),
                      _ix.get_column('rf_accrual_next').to_list()))
        rf_next = np.array([(_m.get(d) or 0.0) for d in dates],
                           dtype=np.float64)
    interest = np.zeros(T)
    E = nav0
    EQ = nav0
    prev = np.zeros(K)
    for t in range(T):
        nav[t] = E
        equity[t] = EQ
        if use_gdd and t >= 2:
            lo = max(0, t - 1 - GDD_LOOKBACK)
            span = (cum[t - 1] - cum[lo]) if (t - 1 - lo) == GDD_LOOKBACK else 0.0
            den = E * w[t, 0] * idm[t]
            dd = np.divide(span, den, out=np.zeros(K),
                           where=np.isfinite(den) & (den > 0))
            gdd[t] = 1.0 - (1.0 - GDD_FLOOR) / (
                1.0 + np.exp(-GDD_STEEPNESS * (GDD_THRESHOLD - dd)))
        if E > 0:
            want = np.trunc(E * k[t] * gdd[t])
            tgt[t] = want
            if use_buffer:
                want = np.where(np.abs(want - prev) > buffer * np.abs(prev),
                                want, prev)
            N[t] = np.where(act[t], want, prev)
            traded = np.where(rolled[t], np.abs(prev) + np.abs(N[t]),
                              np.abs(N[t] - prev))
            tr = traded * cost_rt * fx[t]
            cost_m[t] = np.where(np.isfinite(tr), tr, 0.0)
            prev = N[t]
        _b = has_bar[t]
        _own_prev = np.where(_b, _own_last, _own_prev)
        _own_last = np.where(_b, N[t], _own_last)
        if t + 1 < T:
            gapN[t + 1] = _own_prev
        _base = E if compound else EQ
        _cost_pending = _cost_pending + cost_m[t]
        if t + 1 < T:
            _fill = np.where(has_bar[t + 1], _cost_pending, 0.0)
            cost_chg[t + 1] = _fill
            _cost_pending = _cost_pending - _fill
            _c = float(_fill.sum())
        else:
            _c = 0.0
        EQ -= _c
        if compound:
            E -= _c
        if t + 1 < T and started_t[t]:
            interest[t + 1] = _base * rf_next[t]
        if t + 1 < T:
            g = ((gapN[t + 1] * gapP[t + 1] + N[t] * dayP[t + 1])
                 * ps * fx[t + 1])
            gi = np.where(np.isfinite(g), g, 0.0)
            cum[t + 1] = cum[t] + gi
            p = float(np.nansum(gi))
            pnl[t + 1] = p
            EQ += p + interest[t + 1]
            E = (E + p + interest[t + 1]) if compound else nav0
    cost_t = cost_m.sum(axis=1)
    cost_chg_t = cost_chg.sum(axis=1)
    cost_lag = cost_chg_t
    net_pnl = pnl - cost_lag

    gross_ret = np.zeros(T)
    net_ret = np.zeros(T)
    total_ret = np.zeros(T)
    ok = np.isfinite(nav) & (np.abs(nav) > 0)
    base = np.where(ok[:-1], nav[:-1], 1.0)
    gross_ret[1:] = np.where(ok[:-1], pnl[1:] / base, 0.0)
    net_ret[1:] = np.where(ok[:-1], net_pnl[1:] / base, 0.0)
    total_ret[1:] = np.where(ok[:-1],
                             (net_pnl[1:] + interest[1:]) / base, 0.0)
    ret = gross_ret

    wanted = np.abs(nav[:, None] * k) >= 1e-12
    floored = int(np.sum(wanted & (N == 0) & act))

    px = _ffill(P["raw"])
    notional = np.abs(N) * px * ps * fx
    dN = np.abs(np.diff(N, axis=0, prepend=0.0))
    traded = np.nansum(np.where(np.isfinite(dN * px * ps * fx),
                                dN * px * ps * fx, 0.0), axis=1)
    def _pnl_i(kk: int) -> np.ndarray:
        n_day = N[:-1, kk]
        n_gap = gapN[1:, kk]
        z = lambda x: np.where(np.isfinite(x), x, 0.0)
        g = z(n_gap * gapP[1:, kk] * ps[kk] * fx[1:, kk])
        d = z(n_day * dayP[1:, kk] * ps[kk] * fx[1:, kk])
        return g + d, g, d

    frames = {}
    for kk, i in enumerate(insts):
        _tot, _gp, _dy = _pnl_i(kk)
        _pv = np.concatenate([[0.0], _tot])
        keep = (has_bar[:, kk] | np.isfinite(P["SIGNAL"][:, kk])
                | (N[:, kk] != 0) | (_pv != 0.0))
        frames[i] = pl.DataFrame({
            "date": [dates[t] for t in range(T) if keep[t]],
            "symbol": [sym[t, kk] for t in range(T) if keep[t]],
            "SIGNAL": P["SIGNAL"][keep, kk],
            "price_vol_USD_ann": P["sigma"][keep, kk],
            "s_g_vol": P["gvol"][keep, kk],
            "s_g_dd": gdd[keep, kk],
            "w_i": np.where(act[keep, kk], w[keep, 0], np.nan),
            "IDM": idm[keep],
            "NAV": nav[keep],
            "tradable": in_universe[keep, kk],
            "sized": act[keep, kk],
            "N_raw": np.where(act[:, kk], nav * k[:, kk] * gdd[:, kk],
                              np.nan)[keep],
            "N_target": tgt[keep, kk],
            "N_contracts": N[keep, kk],
            "notional_USD": notional[keep, kk],
            "pnl_USD": _pv[keep],
            "pnl_gap_USD": np.concatenate([[0.0], _gp])[keep],
            "pnl_day_USD": np.concatenate([[0.0], _dy])[keep],
            "cost_USD": cost_m[keep, kk],
            "cost_lag_USD": cost_chg[keep, kk],
            "net_pnl_USD": (_pv - cost_chg[:, kk])[keep],
            "cum_cost_USD": np.cumsum(cost_m[:, kk])[keep],
        })
    port = pl.DataFrame({
        "date": dates,
        "started": started_t,
        "n_active": n_act,
        "wCw": wcw,
        "IDM": idm,
        "n_undefined_pairs": n_undef,
        "NAV": nav,
        "equity_USD": equity,
        "pnl_USD": pnl,
        "cost_USD": cost_t,
        "cost_lag_USD": cost_lag,
        "net_pnl_USD": net_pnl,
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        "rf_accrual_next": rf_next,
        "rf_accrual_applied": np.concatenate([[0.0], rf_next[:-1]]),
        "interest_USD": interest,
        "total_ret": total_ret,
        "gross_notional_USD": np.nansum(np.where(np.isfinite(notional),
                                                 notional, 0.0), axis=1),
        "n_positions": (N != 0).sum(axis=1).astype(np.float64),
        "traded_notional_USD": traded,
    })
    return frames, port, floored


def stats(port: pl.DataFrame, label: str) -> dict:
    r = port.get_column("gross_ret").to_numpy()
    nr = (port.get_column("net_ret").to_numpy()
          if "net_ret" in port.columns else r)
    nav = port.get_column("NAV").to_numpy()
    idm = port.get_column("IDM").to_numpy()
    npos = (port.get_column("n_positions").to_numpy()
            if "n_positions" in port.columns else np.ones(len(r)))
    held = np.flatnonzero(npos > 0)
    t0 = int(held[0]) if len(held) else 1
    live = np.isfinite(r) & (np.arange(len(r)) >= max(t0, 1))
    rr = r[live]
    ann = float(np.std(rr, ddof=0) * math.sqrt(tb_days()))
    mu = float(np.mean(rr) * tb_days())
    eq = (port.get_column("equity_USD").to_numpy()
          if "equity_USD" in port.columns else nav)
    dd = eq / np.maximum.accumulate(np.where(np.isfinite(eq), eq, -np.inf))
    tr = port.get_column("traded_notional_USD").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        turn = np.where(np.isfinite(nav) & (nav > 0), tr / nav, np.nan)
    n_yr = live.sum() / tb_days()
    e0 = eq[max(t0 - 1, 0)]
    cagr = (((eq[-1] / e0) ** (1.0 / n_yr) - 1.0)
            if n_yr > 0 and e0 > 0 else float("nan"))
    return {
        "label": label,
        "sessions": int(live.sum()),
        "t0": t0,
        "NAV_end": float(eq[-1]),
        "ann_ret": mu,
        "net_ann_ret": float(np.mean(nr[live]) * tb_days()),
        "net_ann_vol": float(np.std(nr[live], ddof=0) * math.sqrt(tb_days())),
        "net_sharpe": (float(np.mean(nr[live]) * tb_days())
                       / float(np.std(nr[live], ddof=0) * math.sqrt(tb_days())))
                      if float(np.std(nr[live], ddof=0)) else float("nan"),
        "cost_ann": cost_ann(r, nr, live),
        "rf_ann": (float(np.mean(
            (port.get_column("total_ret").to_numpy() - nr)[live]) * tb_days())
            if "total_ret" in port.columns else float("nan")),
        "cagr": cagr,
        "ann_vol": ann,
        "sharpe": mu / ann if ann else float("nan"),
        "max_dd": float(1.0 - np.nanmin(dd[np.isfinite(dd)])) if len(dd) else float("nan"),
        "idm_mean": float(np.nanmean(idm[live])),
        "idm_sd": float(np.nanstd(idm[live])),
        "idm_dod": float(np.nanmean(np.abs(np.diff(idm[live])))),
        "idm_at_cap": float(np.mean(np.isclose(idm[live], IDM_CAP))),
        "turnover": float(np.nanmean(turn[live]) * tb_days()),
    }


def tb_days() -> int:
    return 256


def cost_ann(gross_ret, net_ret, live=None) -> float:
    g = np.asarray(gross_ret, dtype=float)
    n = np.asarray(net_ret, dtype=float)
    if live is not None:
        g, n = g[live], n[live]
    return float(np.mean(g - n) * tb_days())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", type=float, default=NAV_0)
    ap.add_argument("--tau", type=float, default=TAU,
                    help="risk budget: annualised volatility the "
                         "PORTFOLIO targets (default 0.20)")
    ap.add_argument("--idm-span", type=int, default=IDM_CORR_SPAN)
    ap.add_argument("--compare-spans", default=None,
                    help="comma-separated spans to score against each other")
    ap.add_argument("--no-gvol", action="store_true",
                    help="ablation: disable the 3.29 volatility gate")
    ap.add_argument("--no-gdd", action="store_true",
                    help="ablation: disable the 3.30 drawdown gate")
    ap.add_argument("--no-buffer", action="store_true",
                    help="ablation: disable the 3.36 no-trade buffer")
    ap.add_argument("--start-date", default=START_DATE,
                    help="first session the book may hold anything "
                         "(default 1990-01-01)")
    ap.add_argument("--buffer", type=float, default=BUFFER,
                    help="3.36 band width b (default 0.10)")
    ap.add_argument("--fixed-nav", action="store_true",
                    help="size off a constant NAV -- no compounding")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    tb = _load(BOOK_PY, "tb")
    insts = sorted(p.stem for p in tb.BOOK.glob("*.csv"))
    if not insts:
        raise SystemExit(f"[ABORT] no books in {tb.BOOK}; run stage 2 first")
    t0 = time.time()
    print(f"portfolio -> {HERE}    NAV {args.nav:,.0f}   tau {args.tau:.0%}"
          f"   {'FIXED notional' if args.fixed_nav else 'compounding'}"
          f"   from {args.start_date}")
    print(f"  loading {len(insts)} books ...", flush=True)
    dates, P, sym = panel(tb, insts)
    print(f"  panel {len(dates):,} sessions x {len(insts)} instruments "
          f"({time.time() - t0:.0f}s)", flush=True)

    if args.compare_spans:
        spans = [int(x) for x in args.compare_spans.split(",")]
        rows = []
        for s in spans:
            t1 = time.time()
            _f, port, fl = simulate(dates, P, sym, insts, tb, args.nav, s,
                                    args.tau, not args.fixed_nav,
                                    not args.no_gvol, not args.no_gdd,
                                    not args.no_buffer, args.buffer,
                                    args.start_date)
            st = stats(port, f"span {s}")
            st["floored"] = fl
            st["secs"] = time.time() - t1
            rows.append(st)
            print(f"  span {s:<5} done in {st['secs']:.0f}s", flush=True)
        hdr = (f"\n{'IDM span':<10}{'mean':>8}{'sd':>8}{'|d/day|':>9}"
               f"{'at cap':>8}{'turnover':>10}{'ann vol':>9}{'ann ret':>9}"
               f"{'Sharpe':>8}{'max DD':>8}")
        print(hdr)
        print("-" * (len(hdr) - 1))
        for st in rows:
            print(f"{st['label']:<10}{st['idm_mean']:>8.3f}{st['idm_sd']:>8.3f}"
                  f"{st['idm_dod']:>9.4f}{st['idm_at_cap']:>8.1%}"
                  f"{st['turnover']:>9.1f}x{st['ann_vol']:>9.2%}"
                  f"{st['ann_ret']:>9.2%}{st['sharpe']:>8.2f}"
                  f"{st['max_dd']:>8.1%}")
        return 0

    frames, port, floored = simulate(dates, P, sym, insts, tb, args.nav,
                                     args.idm_span, args.tau,
                                     not args.fixed_nav, not args.no_gvol,
                                     not args.no_gdd, not args.no_buffer,
                                     args.buffer, args.start_date)
    st = stats(port, "run")
    print(f"\n  IDM      mean {st['idm_mean']:.3f}   sd {st['idm_sd']:.3f}   "
          f"at cap {st['idm_at_cap']:.1%}")
    eq = port.get_column("equity_USD")[-1]
    d0 = port.get_column("date")[st["t0"]]
    print(f"  traded   {d0} .. {port.get_column('date')[-1]}"
          f"   {st['sessions']:,} sessions"
          f"   ({st['sessions']/256:.1f} years)")
    print(f"  equity   {args.nav:,.0f} -> {eq:,.0f}"
          f"   ({eq/args.nav:,.1f}x)")
    print(f"  gross    ret {st['ann_ret']:>7.2%}   vol {st['ann_vol']:>7.2%}"
          f"   Sharpe {st['sharpe']:.3f}")
    print(f"  costs        {st['cost_ann']:>7.2%} of NAV per year")
    print(f"  NET      ret {st['net_ann_ret']:>7.2%}   vol {st['net_ann_vol']:>7.2%}"
          f"   Sharpe {st['net_sharpe']:.3f}   (excess of IRX)")
    print(f"  interest     {st['rf_ann']:>7.2%} of NAV per year, earned on cash "
          f"and NOT in NET above")
    print(f"  max DD       {st['max_dd']:>7.1%} on equity "
          f"(after costs, after interest)")
    print(f"  CAGR     {st['cagr']:>7.2%}     (equity growth rate; differs from "
          f"ann ret under fixed notional)")
    und = float(port.get_column("n_undefined_pairs").sum())
    print(f"  undefined correlation pairs among active markets: {und:,.0f}")
    print(f"  positions truncated to zero: {floored:,}")

    if not args.no_write:
        POS.mkdir(parents=True, exist_ok=True)
        for i, f in frames.items():
            f.write_csv(POS / f"{i}.csv")
            f.write_parquet(POS / f"{i}.parquet")
        port.write_csv(PORTFOLIO)
        port.write_parquet(PORTFOLIO.with_suffix(".parquet"))
        print(f"\n  wrote {len(frames)} position files -> {POS}")
        print(f"  wrote {PORTFOLIO.name}   {port.height:,} sessions")
    print(f"  {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
