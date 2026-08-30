from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
LIVE = HERE.parent
REPO = LIVE.parent
BK = LIVE / "4_Bookkeeping"
PORT = LIVE / "3_Portfolio" / "Portfolio.parquet"
POS = LIVE / "3_Portfolio" / "Positions"
STMT = BK / "statement.parquet"
RUN_STAMP = LIVE / ".pipeline_run.json"
BOOK_PY = LIVE / "2_Engine" / "trading_book.py"
PORT_PY = LIVE / "3_Portfolio" / "portfolio.py"
JOURNAL = BK / "Journal"
DOCS = REPO / "docs"
OUT = DOCS / "data"
PAGES = ["index.html", "journal.html", "pnl.html", "mapping.html",
         "qa.html", "expectations.html"]
ASSETS = ["app.js", "site.css"]
TAGS = {
    "app.js": (re.compile(r'<script src="app\.js(?:\?v=[A-Za-z0-9]+)?"></script>'),
               '<script src="app.js?v={v}"></script>'),
    "site.css": (re.compile(r'<link rel="stylesheet" '
                            r'href="site\.css(?:\?v=[A-Za-z0-9]+)?">'),
                 '<link rel="stylesheet" href="site.css?v={v}">'),
}

WINDOW_START = "2026-01-02"
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

BACKTEST_START = "1990-01-02"

EXECUTED_COLS = ["instrument", "contract", "action", "quantity", "kind",
                 "fill_open", "decision_close", "commission_USD",
                 "realised_pnl_USD"]
GIVEN_COLS = ["instrument", "contract", "action", "quantity", "kind",
              "decision_close"]
INDEX_COLS = ["date", "n_given", "n_executed"]
PENDING_COLS = ["instrument", "contract", "action", "quantity", "kind",
                "decision_close"]
OUTSTANDING_COLS = ["instrument", "contract", "action", "quantity",
                    "carried_sessions", "reason"]
DAILY_COLS = ["date", "equity_USD", "drawdown", "gross_pnl_USD",
              "commission_USD", "interest_USD"]
PNL_COLS = ["instrument", "gross_pnl_USD", "session", "held"]
BOOK_COLS = ["opening_equity_USD", "gross_pnl_USD", "commission_USD",
             "interest_USD", "closing_equity_USD", "interest_base_USD",
             "interest_from_date", "calendar_days", "rate_annual_pct"]
PNL_INDEX_COLS = ["date", "gross_pnl_USD", "net_total_USD"]
MAPPING_COLS = ["instrument", "description", "asset_class", "pointsize",
                "currency", "exchange", "cost_rt_local", "tick_size"]
QA_ATTRIB_COLS = ["asset_class", "gross_pnl_USD", "share"]
QA_CUM_COLS = ["date", "asset_class", "cum_USD"]
QA_POS_COLS = ["instrument", "asset_class", "contracts", "notional_USD",
               "side", "share_of_gross"]
QA_WORST_COLS = ["date", "bench_ret", "book_ret"]
QA_VOL_COLS = ["date", "realised"]
BH_BUFFER = 0.10
QA_BENCH_COLS = ["date", "book", "bh", "spx"]
QA_BSTAT_COLS = ["key", "name", "total", "vol", "sharpe", "max_dd"]
EXPECT_MOM_COLS = ["horizon", "n", "mean", "median", "sd", "skew", "kurt",
                   "up", "min", "max"]
EXPECT_VAR_COLS = ["level", "normal", "cf", "hist", "normal_cvar", "hist_cvar",
                   "ratio_normal", "ratio_hist", "understate"]
EXPECT_BREACH_COLS = ["level", "threshold", "expected", "observed",
                      "kupiec_lr", "reject_5pct"]
EXPECT_ANNUAL_COLS = ["year", "ret"]
EXPECT_CURVE_COLS = ["level", "normal", "cf", "hist", "normal_cvar",
                     "hist_cvar"]
EXPECT_LIVE_COLS = ["date", "ret"]
QA_KEYS = ["as_of", "bench", "bench_name", "n", "corr", "corr_lo", "corr_hi",
           "beta", "r2", "worst", "attribution", "attribution_cum",
           "positions", "exposure", "gross_notional_USD", "leverage",
           "n_positions", "long", "short",
           "vol", "vol_window", "vol_target", "vol_mean", "realised",
           "bench_curves", "bench_stats", "bench_scaled",
           "key", "name", "total", "vol", "sharpe", "max_dd",
           "book", "bh", "spx", "vol_ratio", "scaled_total", "scaled_dd"]

FORBIDDEN = ["norgate"]
META_KEYS = ["as_of", "window_start", "sessions", "equity_start", "equity_end",
             "net_ann_ret", "net_cagr", "cagr_trading", "window_ret_trading",
             "window_ret_interest", "net_ann_vol", "net_sharpe",
             "max_drawdown", "cost_ann", "cost_window_USD", "interest_ann",
             "n_positions", "updated_at",
             "generated_at"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _f(x):
    return None if x is None or x != x else float(x)


def _guard(name: str, rows: list[dict], allowed: list[str]) -> list[dict]:
    for r in rows:
        extra = sorted(set(r) - set(allowed))
        if extra:
            raise SystemExit(
                f"[ABORT] {name}: column(s) not on the publish whitelist: "
                f"{extra}\n         Add them to {Path(__file__).name} "
                f"deliberately, or drop them. Nothing was written.")
    return rows


def _dates_guard(name: str, rows: list[dict], key: str = "date") -> None:
    bad = [r[key] for r in rows if key in r and r[key] < WINDOW_START]
    if bad:
        raise SystemExit(
            f"[ABORT] {name}: {len(bad)} row(s) dated before {WINDOW_START} "
            f"(earliest {min(bad)}).\n         The published window is 2026 "
            f"only. Nothing was written.")


_WINDOW_EXEMPT = {
    "interest_from_date",
    "backtest_start", "backtest_end",
}


def _window_sweep(name: str, payload) -> None:
    bad = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in _WINDOW_EXEMPT:
                    continue
                walk(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, str) and _DATE_RE.fullmatch(o) and o < WINDOW_START:
            bad.append(f"{path} = {o}")

    walk(payload, name)
    if bad:
        nl = chr(10) + " " * 9
        raise SystemExit(
            f"[ABORT] {name}: {len(bad)} pre-{WINDOW_START} date(s) in the "
            f"payload:" + nl + nl.join(sorted(bad)[:6]) + nl
            + "The published window is 2026 only. Either the data is wrong, "
            + "or the field belongs in _WINDOW_EXEMPT with a reason beside it."
            + nl + "Nothing was written.")


def run_stamp() -> str:
    if not RUN_STAMP.is_file():
        raise SystemExit(
            f"[ABORT] no pipeline run stamp at {RUN_STAMP}. The site reports "
            f"when the data was last rebuilt, and that is recorded by "
            f"Update.py. Run the pipeline first. Nothing was written.")
    st = json.loads(RUN_STAMP.read_text(encoding="utf-8"))
    if st.get("failures"):
        raise SystemExit(
            f"[ABORT] the last pipeline run finished with {st['failures']} "
            f"verification failure(s) at {st.get('completed_at')}. Publishing "
            f"would put unverified numbers on a public page. Nothing was "
            f"written.")
    if not st.get("verified", True):
        raise SystemExit(
            f"[ABORT] the last pipeline run at {st.get('completed_at')} was "
            f"made with --no-verify, so nothing was checked. A run nobody "
            f"verified is not a run worth publishing. Nothing was written.")
    if not st.get("full_run", True):
        raise SystemExit(
            f"[ABORT] the last pipeline run was partial (a --no-* flag was "
            f"used). The published figures come from every stage, so they must "
            f"be rebuilt by every stage. Nothing was written.")
    return st["completed_at"]


def _pages_guard() -> None:
    for name in PAGES:
        f = DOCS / name
        if not f.is_file():
            raise SystemExit(f"[ABORT] missing {f}")
        page = f.read_text(encoding="utf-8")
        for asset, (rx, _) in TAGS.items():
            if not rx.search(page):
                raise SystemExit(
                    f"[ABORT] {name}: no tag loading {asset} that this can "
                    f"stamp. The cache version rides on those tags and every "
                    f"data fetch inherits it. Nothing was written.")


_WALL = ("generated_at", "updated_at")


def build_stamp(latest: dict) -> str:
    payload = json.loads(json.dumps(latest))
    for _wall in _WALL:
        payload.get("meta", {}).pop(_wall, None)
    h = hashlib.sha256()
    for asset in ASSETS:
        h.update((DOCS / asset).read_bytes())
    h.update(json.dumps(payload, sort_keys=True,
                        separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()[:8]


def stamp_pages(stamp: str) -> list[str]:
    touched = []
    for name in PAGES:
        f = DOCS / name
        old = f.read_text(encoding="utf-8")
        new = old
        for rx, tpl in TAGS.values():
            new = rx.sub(tpl.format(v=stamp), new, count=1)
        if new != old:
            f.write_text(new, encoding="utf-8")
            touched.append(name)
    return touched


def build() -> tuple[dict, dict, list, dict]:
    for f in (PORT, BK / "executed.parquet", BK / "pending.parquet"):
        if not f.is_file():
            raise SystemExit(f"[ABORT] missing {f}; run the pipeline first")
    _pages_guard()
    tb = _load(BOOK_PY, "tb")
    pf = _load(PORT_PY, "pf")
    P = pl.read_parquet(PORT).filter(pl.col("started"))
    if not P.height:
        raise SystemExit("[ABORT] portfolio has no started sessions")
    d = P.get_column("date").to_list()
    if d[0] < WINDOW_START:
        raise SystemExit(
            f"[ABORT] the portfolio starts {d[0]}, before the published window "
            f"{WINDOW_START}.\n         This publisher is for the 2026 run. "
            f"Nothing was written.")

    eq = P.get_column("equity_USD").to_numpy().astype(float)
    dd = eq / np.maximum.accumulate(eq) - 1.0
    nr = np.nan_to_num(P.get_column("net_ret").to_numpy().astype(float))
    tot = np.nan_to_num(P.get_column("total_ret").to_numpy().astype(float))
    gr = np.nan_to_num(P.get_column("gross_ret").to_numpy().astype(float))
    cost = np.nan_to_num(P.get_column("cost_lag_USD").to_numpy().astype(float))
    ist = np.nan_to_num(P.get_column("interest_USD").to_numpy().astype(float))
    gpnl = np.nan_to_num(P.get_column("pnl_USD").to_numpy().astype(float))
    nav = np.nan_to_num(P.get_column("NAV").to_numpy().astype(float), nan=1.0)
    npnl = gpnl - cost
    sd = float(nr.std(ddof=0))

    exe = pl.read_parquet(BK / "executed.parquet")
    opens: dict[str, dict] = {}
    ex_rows = []
    for r in exe.iter_rows(named=True):
        inst = r["instrument"]
        if inst not in opens:
            b = tb.load_book(inst).select(["date", "open"])
            opens[inst] = dict(zip(b.get_column("date").to_list(),
                                   b.get_column("open").to_list()))
        ex_rows.append({
            "instrument": inst, "contract": r["contract"],
            "action": r["action"], "quantity": _f(r["quantity"]),
            "kind": r["kind"],
            "fill_open": _f(opens[inst].get(r["execute_at"])),
            "decision_close": _f(r["decision_close"]),
            "commission_USD": _f(r["commission_USD"]),
            "realised_pnl_USD": _f(r["realised_pnl_USD"])})
    _guard("executed", ex_rows, EXECUTED_COLS)

    pend = pl.read_parquet(BK / "pending.parquet")
    pd_rows = [{"instrument": r["instrument"], "contract": r["contract"],
                "action": r["action"], "quantity": _f(r["quantity"]),
                "kind": r["kind"],
                "decision_close": _f(r["decision_close"])}
               for r in pend.iter_rows(named=True)]
    _guard("pending", pd_rows, PENDING_COLS)

    as_of = d[-1]
    ot_rows: list[dict] = []
    op = JOURNAL / "outstanding" / as_of[:4] / f"{as_of}.parquet"
    if op.is_file():
        ot_rows = [{"instrument": r["instrument"], "contract": r["contract"],
                    "action": r["action"], "quantity": _f(r["quantity"]),
                    "carried_sessions": _f(r["carried_sessions"]),
                    "reason": r["reason"]}
                   for r in pl.read_parquet(op).iter_rows(named=True)]
    _guard("outstanding", ot_rows, OUTSTANDING_COLS)

    daily = [{"date": d[k], "equity_USD": round(float(eq[k]), 2),
              "drawdown": round(float(dd[k]), 6),
              "gross_pnl_USD": round(float(gpnl[k]), 2),
              "commission_USD": round(float(cost[k]), 2),
              "interest_USD": round(float(ist[k]), 2)}
             for k in range(len(d))]
    _guard("daily", daily, DAILY_COLS)
    _dates_guard("daily", daily)

    npos = P.get_column("n_positions").to_numpy()
    meta = {
        "as_of": as_of,
        "window_start": d[0],
        "sessions": len(d),
        "equity_start": round(float(eq[0]), 2),
        "equity_end": round(float(eq[-1]), 2),
        "net_ann_ret": round(float(nr.mean() * 256), 6),
        "net_cagr": round(float((eq[-1] / eq[0]) ** (256.0 / len(d)) - 1.0), 6)
                    if eq[0] > 0 and len(d) else None,
        "cagr_trading": round(float(np.prod(1.0 + nr) ** (256.0 / len(d)) - 1.0), 6)
                        if len(d) else None,
        "window_ret_trading": round(float(npnl.sum() / eq[0]), 6) if eq[0] else None,
        "window_ret_interest": round(float(ist.sum() / eq[0]), 6) if eq[0] else None,
        "net_ann_vol": round(float(sd * math.sqrt(256)), 6),
        "net_sharpe": round(float(nr.mean() / sd * math.sqrt(256)), 4) if sd else None,
        "max_drawdown": round(float(dd.min()), 6),
        "cost_ann": round(pf.cost_ann(gr, nr), 6),
        "cost_window_USD": round(float(cost.sum()), 2),
        "interest_ann": round(float((tot - nr).mean() * 256), 6),
        "n_positions": int(npos[-1]) if len(npos) else 0,
        "updated_at": run_stamp(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    extra = sorted(set(meta) - set(META_KEYS))
    if extra:
        raise SystemExit(f"[ABORT] meta: key(s) not whitelisted: {extra}")

    index, days = build_days(tb, pl.read_parquet(BK / "Orders.parquet"), opens)
    latest = {"meta": meta, "executed": ex_rows, "pending": pd_rows,
              "outstanding": ot_rows}
    history = {"meta": {"as_of": as_of, "sessions": len(d)}, "daily": daily}
    return latest, history, index, days


def build_days(tb, led, opens: dict) -> tuple[list[dict], dict]:
    dec, exe = {}, {}
    for r in led.iter_rows(named=True):
        dec.setdefault(r["decision_date"], []).append(r)
        if r["execute_at"]:
            exe.setdefault(r["execute_at"], []).append(r)

    def _open(inst: str, date: str):
        if inst not in opens:
            b = tb.load_book(inst).select(["date", "open"])
            opens[inst] = dict(zip(b.get_column("date").to_list(),
                                   b.get_column("open").to_list()))
        return _f(opens[inst].get(date))

    days, index = {}, []
    for d in sorted(set(dec) | set(exe)):
        given = [{"instrument": r["instrument"], "contract": r["contract"],
                  "action": r["action"], "quantity": _f(r["quantity"]),
                  "kind": r["kind"],
                  "decision_close": _f(r["decision_close"])}
                 for r in dec.get(d, [])]
        filled = [{"instrument": r["instrument"], "contract": r["contract"],
                   "action": r["action"], "quantity": _f(r["quantity"]),
                   "kind": r["kind"], "fill_open": _open(r["instrument"], d),
                   "decision_close": _f(r["decision_close"]),
                   "commission_USD": _f(r["commission_USD"]),
                   "realised_pnl_USD": _f(r["realised_pnl_USD"])}
                  for r in exe.get(d, [])]
        _guard(f"given {d}", given, GIVEN_COLS)
        _guard(f"executed {d}", filled, EXECUTED_COLS)
        days[d] = {"date": d, "given": given, "executed": filled}
        index.append({"date": d, "n_given": len(given),
                      "n_executed": len(filled)})
    _guard("index", index, INDEX_COLS)
    _dates_guard("index", index)
    return index, days


def build_mapping() -> list[dict]:
    f = LIVE / "instrument_mapping.csv"
    if not f.is_file():
        raise SystemExit(f"[ABORT] missing {f}")
    m = pl.read_csv(f)
    rows = [{"instrument": r["norgate_code"],
             "description": r["description"],
             "asset_class": r["asset_class"],
             "pointsize": _f(r["pointsize"]),
             "currency": r["currency"],
             "exchange": r["exchange"],
             "cost_rt_local": _f(r["total_avg_cost_rt_LocalCurrency"]),
             "tick_size": _f(r["tick_size"])}
            for r in m.iter_rows(named=True)]
    _guard("mapping", rows, MAPPING_COLS)
    blob = json.dumps(rows).lower()
    for word in FORBIDDEN:
        if word in blob:
            raise SystemExit(
                f"[ABORT] mapping: the published payload contains {word!r}. "
                f"The site never names the data provider. Nothing was written.")
    return rows


def _bh_book() -> tuple[dict, dict]:
    import csv as _csv
    spec = {r["norgate_code"]: r for r in
            _csv.DictReader((LIVE / "instrument_mapping.csv").open(encoding="utf-8"))}
    tb = _load(BOOK_PY, "tb_bh")
    pnl: dict[str, float] = {}
    cost: dict[str, float] = {}
    for sym, row in spec.items():
        ps = float(row["pointsize"])
        rt = float(row["total_avg_cost_rt_LocalCurrency"])
        bk = pl.read_parquet(tb.BOOK / f"{sym}.parquet",
                             columns=["date", "Continuous_O", "Continuous_C",
                                      "FX_rate", "hold"])
        po = pl.read_parquet(POS / f"{sym}.parquet",
                             columns=["date", "N_raw", "SIGNAL", "sized"])
        d = (po.join(bk, on="date", how="left").sort("date")
               .with_columns(pl.col("hold").forward_fill().alias("h")))
        dts = d.get_column("date").to_list()
        CO = d.get_column("Continuous_O").to_numpy().astype(float)
        CC = d.get_column("Continuous_C").to_numpy().astype(float)
        FX = d.get_column("FX_rate").to_numpy().astype(float)
        raw = d.get_column("N_raw").to_numpy().astype(float)
        sig = d.get_column("SIGNAL").to_numpy().astype(float)
        act = np.nan_to_num(
            d.get_column("sized").to_numpy().astype(float)).astype(bool)
        H = np.array(d.get_column("h").to_list(), dtype=object)

        bar = np.isfinite(CC)
        ff = lambda v: pl.Series(v).fill_nan(None).forward_fill().to_numpy()
        prevC = np.concatenate([[np.nan], ff(CC)[:-1]])
        CO = np.where(bar, CO, prevC)
        CC = np.where(bar, CC, prevC)
        FX = ff(FX)

        with np.errstate(divide="ignore", invalid="ignore"):
            want = np.where(np.abs(sig) > 1e-9, raw * (10.0 / sig), np.nan)

        n = len(want)
        N = np.zeros(n)
        prev = 0.0
        for t in range(n):
            if act[t] and np.isfinite(want[t]):
                w = np.trunc(want[t])
                if not abs(w - prev) > BH_BUFFER * abs(prev):
                    w = prev
                prev = w
            N[t] = prev

        N1 = np.concatenate([[0.0], N[:-1]])
        rowpnl = np.zeros(n)
        b_ix = np.flatnonzero(bar)
        if b_ix.size:
            Nb, COb, CCb, FXb = N[b_ix], CO[b_ix], CC[b_ix], FX[b_ix]
            Nb1 = np.concatenate([[0.0], Nb[:-1]])
            Nb2 = np.concatenate([[0.0, 0.0], Nb[:-2]])
            CCbp = np.concatenate([[np.nan], CCb[:-1]])
            rowpnl[b_ix] = (np.nan_to_num(Nb2 * (COb - CCbp) * ps * FXb)
                            + np.nan_to_num(Nb1 * (CCb - COb) * ps * FXb))

        roll = np.zeros(n, bool)
        roll[1:] = (H[1:] != H[:-1]) & (H[1:] != None) & (H[:-1] != None)
        units = np.where(roll, np.abs(N1) + np.abs(N), np.abs(N - N1))
        c = np.nan_to_num(units * (rt / 2.0) * FX)
        pend = 0.0
        for t in range(n):
            pend += c[t]
            if t + 1 < n and bar[t + 1]:
                if dts[t + 1] >= WINDOW_START:
                    cost[dts[t + 1]] = cost.get(dts[t + 1], 0.0) + pend
                pend = 0.0
        for t, dt in enumerate(dts):
            if dt >= WINDOW_START:
                pnl[dt] = pnl.get(dt, 0.0) + rowpnl[t]
    return pnl, cost


def build_qa() -> dict:
    import math

    port = pl.read_parquet(PORT).sort("date")
    port = port.filter(pl.col("started")).filter(pl.col("date") >= WINDOW_START)
    dates = port.get_column("date").to_list()
    eq = port.get_column("equity_USD").to_numpy()
    ret = port.get_column("net_ret").to_numpy()
    cls = {r["instrument"]: r["asset_class"] for r in build_mapping()}

    tbf = _load(BOOK_PY, "tb_qa").BOOK / "ES.parquet"
    es = (pl.read_parquet(tbf, columns=["date", "Continuous_C"])
            .sort("date").filter(pl.col("date") >= WINDOW_START))
    ed, ec = es.get_column("date").to_list(), es.get_column("Continuous_C").to_numpy()
    bench = {}
    for i in range(1, len(ed)):
        if np.isfinite(ec[i]) and np.isfinite(ec[i - 1]) and ec[i - 1]:
            bench[ed[i]] = float(ec[i] / ec[i - 1] - 1.0)
    pair = [(float(r), bench[d]) for r, d in zip(ret, dates) if d in bench]
    B = np.array([x[0] for x in pair]); E = np.array([x[1] for x in pair])
    n = len(B)
    r = float(np.corrcoef(B, E)[0, 1])
    z, se = math.atanh(r), 1.0 / math.sqrt(n - 3)
    beta = float(np.polyfit(E, B, 1)[0])
    order = np.argsort(E)[:10]
    worst = _guard("qa worst", [
        {"date": [d for d in dates if d in bench][int(i)],
         "bench_ret": round(float(E[i]), 6), "book_ret": round(float(B[i]), 6)}
        for i in order], QA_WORST_COLS)

    tot: dict[str, float] = {}
    cum_rows, running = [], {}
    for f in sorted(POS.glob("*.parquet")):
        k = cls.get(f.stem, "?")
        d = (pl.read_parquet(f, columns=["date", "pnl_USD"])
               .filter(pl.col("date") >= WINDOW_START).sort("date"))
        for dt, v in zip(d.get_column("date").to_list(),
                         d.get_column("pnl_USD").to_list()):
            tot[k] = tot.get(k, 0.0) + float(v or 0.0)
            running.setdefault(dt, {})
            running[dt][k] = running[dt].get(k, 0.0) + float(v or 0.0)
    gross_abs = sum(abs(v) for v in tot.values()) or 1.0
    attribution = _guard("qa attribution", [
        {"asset_class": k, "gross_pnl_USD": round(v, 2),
         "share": round(v / gross_abs, 6)}
        for k, v in sorted(tot.items(), key=lambda kv: -kv[1])], QA_ATTRIB_COLS)
    acc: dict[str, float] = {}
    for dt in sorted(running):
        for k in sorted(tot):
            acc[k] = acc.get(k, 0.0) + running[dt].get(k, 0.0)
            cum_rows.append({"date": dt, "asset_class": k,
                             "cum_USD": round(acc[k], 2)})
    _guard("qa attribution_cum", cum_rows[:1], QA_CUM_COLS)

    last = dates[-1]
    pos = []
    for f in sorted(POS.glob("*.parquet")):
        d = (pl.read_parquet(f, columns=["date", "N_contracts", "notional_USD"])
               .filter(pl.col("date") == last))
        if not d.height:
            continue
        nq = d.get_column("N_contracts")[0]
        if nq is None or not np.isfinite(nq) or nq == 0:
            continue
        nt = d.get_column("notional_USD")[0]
        nt = float(nt) if nt is not None and np.isfinite(nt) else 0.0
        pos.append({"instrument": f.stem, "asset_class": cls.get(f.stem, "?"),
                    "contracts": int(nq), "notional_USD": round(nt, 2),
                    "side": "LONG" if nq > 0 else "SHORT", "share_of_gross": 0.0})
    gn = sum(abs(x["notional_USD"]) for x in pos) or 1.0
    for x in pos:
        x["share_of_gross"] = round(abs(x["notional_USD"]) / gn, 6)
    pos.sort(key=lambda x: -abs(x["notional_USD"]))
    _guard("qa positions", pos, QA_POS_COLS)

    W = 63
    vol = _guard("qa vol", [
        {"date": dates[i],
         "realised": round(float(ret[i - W:i].std(ddof=0) * math.sqrt(256)), 6)}
        for i in range(W, len(ret) + 1) if i < len(dates)], QA_VOL_COLS)

    bh_pnl, bh_cost = _bh_book()
    eq_bh, e, itr = [], float(eq[0]), 0.0
    for i, dd_ in enumerate(dates):
        if i:
            e += bh_pnl.get(dd_, 0.0) - bh_cost.get(dd_, 0.0) + itr
        itr = e * float(port.get_column("rf_accrual_next").to_numpy()[i] or 0.0)
        eq_bh.append(e)

    spx, lvl = [], float(eq[0])
    for i, dd_ in enumerate(dates):
        if i and dd_ in bench:
            lvl *= (1.0 + bench[dd_])
        spx.append(lvl)

    curves = _guard("qa bench", [
        {"date": d, "book": round(float(eq[i]), 2),
         "bh": round(eq_bh[i], 2), "spx": round(spx[i], 2)}
        for i, d in enumerate(dates)], QA_BENCH_COLS)

    rf_in = np.concatenate([[0.0], np.nan_to_num(
        port.get_column("rf_accrual_next").to_numpy())[:-1]])

    def _st(series):
        a_ = np.asarray(series, dtype=float)
        rr = np.diff(a_) / a_[:-1]
        ex = rr - rf_in[1:]
        v = float(ex.std(ddof=0) * math.sqrt(256))
        return (float(a_[-1] / a_[0] - 1.0), v,
                float((ex.mean() * 256) / v) if v else float("nan"),
                float((a_ / np.maximum.accumulate(a_) - 1).min()))

    bstats, raw_st = [], {}
    for k, nm, series in (("book", "The book", eq),
                          ("bh", "Buy & hold, same universe", eq_bh),
                          ("spx", "S&P 500, via ES", spx)):
        t_, v_, sh, dd_ = _st(series)
        raw_st[k] = (t_, v_, sh, dd_)
        bstats.append({"key": k, "name": nm, "total": round(t_, 6),
                       "vol": round(v_, 6), "sharpe": round(sh, 4),
                       "max_dd": round(dd_, 6)})
    _guard("qa bench stats", bstats, QA_BSTAT_COLS)

    kx = raw_st["bh"][1] / raw_st["book"][1] if raw_st["book"][1] else 1.0
    scaled = {"vol_ratio": round(kx, 3),
              "scaled_total": round(raw_st["book"][0] * kx, 6),
              "scaled_dd": round(raw_st["book"][3] * kx, 6)}

    return {
        "as_of": last,
        "bench": "ES", "bench_name": "S&P 500 futures",
        "bench_curves": curves, "bench_stats": bstats, "bench_scaled": scaled,
        "vol": vol, "vol_window": W, "vol_target": 0.20,
        "vol_mean": round(float(np.mean([v["realised"] for v in vol])), 6),
        "n": n, "corr": round(r, 4),
        "corr_lo": round(math.tanh(z - 1.96 * se), 4),
        "corr_hi": round(math.tanh(z + 1.96 * se), 4),
        "beta": round(beta, 4), "r2": round(r * r, 4), "worst": worst,
        "attribution": attribution, "attribution_cum": cum_rows,
        "positions": pos,
        "exposure": {
            "gross_notional_USD": round(gn, 2),
            "leverage": round(gn / float(eq[-1]), 3),
            "n_positions": len(pos),
            "long": sum(1 for x in pos if x["side"] == "LONG"),
            "short": sum(1 for x in pos if x["side"] == "SHORT")},
    }


def build_expectations() -> dict:
    import math
    from statistics import NormalDist

    ND = NormalDist()
    pf = _load(PORT_PY, "pf_expect")
    tb = _load(BOOK_PY, "tb_expect")
    insts = sorted(p.stem for p in tb.BOOK.glob("*.csv"))
    dates, P, sym = pf.panel(tb, insts)
    _f, port, _fl = pf.simulate(dates, P, sym, insts, tb, pf.NAV_0,
                                pf.IDM_CORR_SPAN, pf.TAU, True, True, True,
                                True, pf.BUFFER, BACKTEST_START)
    port = port.filter(pl.col("started"))
    bd = port.get_column("date").to_list()
    br = port.get_column("net_ret").to_numpy()
    keep = [i for i, d in enumerate(bd) if d < WINDOW_START]
    bt = br[keep]
    bt_dates = [bd[i] for i in keep]
    bt_eq = port.get_column("equity_USD").to_numpy()[keep]

    live_p = pl.read_parquet(PORT).sort("date")
    live_p = live_p.filter(pl.col("started")).filter(pl.col("date") >= WINDOW_START)
    lv = live_p.get_column("net_ret").to_numpy()
    lv_dates = live_p.get_column("date").to_list()
    lv_eq = live_p.get_column("equity_USD").to_numpy()

    def _mom(x, label):
        x = np.asarray(x, dtype=float)
        sd = float(x.std(ddof=0))
        z = (x - x.mean()) / sd if sd else x * 0
        return {"horizon": label, "n": int(len(x)), "mean": round(float(x.mean()), 8),
                "median": round(float(np.median(x)), 8),
                "sd": round(sd, 8), "skew": round(float((z ** 3).mean()), 4),
                "kurt": round(float((z ** 4).mean() - 3), 4),
                "up": round(float((x > 0).mean()), 4),
                "min": round(float(x.min()), 6), "max": round(float(x.max()), 6)}

    def _agg(rets, ds, key):
        b: dict = {}
        for d, v in zip(ds, rets):
            b.setdefault(key(d), []).append(v)
        return np.array([float(np.prod([1 + q for q in v]) - 1)
                         for _, v in sorted(b.items())])

    def _iso_week(d):
        return datetime.fromisoformat(d).isocalendar()[:2]

    moments = _guard("expect moments", [
        _mom(bt, "daily"),
        _mom(_agg(bt, bt_dates, _iso_week), "weekly"),
        _mom(_agg(bt, bt_dates, lambda d: d[:7]), "monthly"),
        _mom(_agg(bt, bt_dates, lambda d: d[:4]), "annual"),
    ], EXPECT_MOM_COLS)
    live_moments = _guard("expect live moments", [
        _mom(lv, "daily"),
        _mom(_agg(lv, lv_dates, _iso_week), "weekly"),
    ], EXPECT_MOM_COLS)

    lo, hi = -0.045, 0.045
    edges = [lo + (hi - lo) * i / 60 for i in range(61)]
    bc, _e = np.histogram(np.clip(bt, lo, hi), bins=edges)
    lc, _e = np.histogram(np.clip(lv, lo, hi), bins=edges)

    s = np.sort(bt)
    ls = np.sort(lv)
    emp = np.searchsorted(s, ls, side="right") / len(s)
    d_stat = float(np.max(np.abs(emp - np.arange(1, len(ls) + 1) / len(ls))))
    crit = 1.36 / math.sqrt(len(ls))

    mu, sd = float(bt.mean()), float(bt.std(ddof=0))
    zz = (bt - mu) / sd
    S, K = float((zz ** 3).mean()), float((zz ** 4).mean() - 3)
    var_rows, breaches = [], []
    for a in (0.95, 0.99, 0.995, 0.999):
        q = ND.inv_cdf(1 - a)
        nv = mu + sd * q
        zc = (q + (q * q - 1) * S / 6 + (q ** 3 - 3 * q) * K / 24
              - (2 * q ** 3 - 5 * q) * S * S / 36)
        hv = float(np.quantile(bt, 1 - a))
        ncv = mu - sd * ND.pdf(q) / (1 - a)
        hcv = float(bt[bt <= hv].mean())
        var_rows.append({
            "level": a, "normal": round(nv, 6), "cf": round(mu + sd * zc, 6),
            "hist": round(hv, 6), "normal_cvar": round(ncv, 6),
            "hist_cvar": round(hcv, 6),
            "ratio_normal": round(ncv / nv, 4), "ratio_hist": round(hcv / hv, 4),
            "understate": round(hcv / ncv - 1, 4)})
        obs = int((lv < hv).sum())
        exp = len(lv) * (1 - a)
        p_hat = obs / len(lv) if len(lv) else 0.0
        if 0 < p_hat < 1:
            lr = -2 * (((len(lv) - obs) * math.log(a) + obs * math.log(1 - a))
                       - ((len(lv) - obs) * math.log(1 - p_hat)
                          + obs * math.log(p_hat)))
        else:
            lr = -2 * ((len(lv) - obs) * math.log(a) + obs * math.log(1 - a)
                       - (len(lv) * math.log(1 - p_hat) if p_hat else 0.0))
        breaches.append({"level": a, "threshold": round(hv, 6),
                         "expected": round(exp, 2), "observed": obs,
                         "kupiec_lr": round(float(lr), 3),
                         "reject_5pct": bool(lr > 3.841)})
    _guard("expect var", var_rows, EXPECT_VAR_COLS)
    _guard("expect breaches", breaches, EXPECT_BREACH_COLS)

    def _dd(equity):
        e = np.asarray(equity, dtype=float)
        return e / np.maximum.accumulate(e) - 1.0

    bdd, ldd = _dd(bt_eq), _dd(lv_eq)
    pct = float((bdd < ldd.min()).mean())

    def _bins(x, lo, hi, k):
        x = np.asarray(x, dtype=float)
        e = [lo + (hi - lo) * i / k for i in range(k + 1)]
        c, _ = np.histogram(np.clip(x, lo, hi), bins=e)
        m = int(np.argmax(c))
        return {"bins": [round(v, 6) for v in e], "counts": [int(v) for v in c],
                "mean": round(float(x.mean()), 6),
                "median": round(float(np.median(x)), 6),
                "mode": round((e[m] + e[m + 1]) / 2, 6)}

    wk = _agg(bt, bt_dates, _iso_week)
    mo = _agg(bt, bt_dates, lambda d: d[:7])
    yr_key = sorted({d[:4] for d in bt_dates})
    yr = _agg(bt, bt_dates, lambda d: d[:4])
    horizon_hist = {
        "daily": _bins(bt, -0.045, 0.045, 46),
        "weekly": _bins(wk, -0.09, 0.09, 40),
        "monthly": _bins(mo, -0.15, 0.15, 34),
        "annual": _bins(yr, -0.20, 0.50, 24),
    }
    annual = _guard("expect annual",
                    [{"year": int(y), "ret": round(float(v), 6)}
                     for y, v in zip(yr_key, yr)], EXPECT_ANNUAL_COLS)

    n_l = len(lv)
    qq = [[round(float(np.quantile(bt, (i + 0.5) / n_l)), 6),
           round(float(v), 6)] for i, v in enumerate(np.sort(lv))]

    curve = []
    for i in range(46):
        a = 0.90 + (0.999 - 0.90) * i / 45
        qn = ND.inv_cdf(1 - a)
        zc = (qn + (qn * qn - 1) * S / 6 + (qn ** 3 - 3 * qn) * K / 24
              - (2 * qn ** 3 - 5 * qn) * S * S / 36)
        hq = float(np.quantile(bt, 1 - a))
        curve.append({"level": round(a, 5), "normal": round(mu + sd * qn, 6),
                      "cf": round(mu + sd * zc, 6), "hist": round(hq, 6),
                      "normal_cvar": round(mu - sd * ND.pdf(qn) / (1 - a), 6),
                      "hist_cvar": round(float(bt[bt <= hq].mean()), 6)})
    _guard("expect var curve", curve, EXPECT_CURVE_COLS)

    live_rows = _guard("expect live series",
                       [{"date": d, "ret": round(float(v), 8)}
                        for d, v in zip(lv_dates, lv)], EXPECT_LIVE_COLS)
    _dates_guard("expect live series", live_rows)

    dd_hist = _bins(bdd, -0.20, 0.0, 40)

    return {
        "backtest_start": bt_dates[0], "backtest_end": bt_dates[-1],
        "live_start": lv_dates[0], "live_end": lv_dates[-1],
        "moments": moments, "live_moments": live_moments,
        "bins": [round(x, 5) for x in edges],
        "backtest_counts": [int(x) for x in bc],
        "live_counts": [int(x) for x in lc],
        "ks_d": round(d_stat, 4), "ks_crit": round(crit, 4),
        "ks_reject": bool(d_stat > crit),
        "var": var_rows, "breaches": breaches,
        "dd_backtest_worst": round(float(bdd.min()), 6),
        "dd_live_worst": round(float(ldd.min()), 6),
        "dd_percentile": round(pct, 4),
        "dd_quantiles": [round(float(np.quantile(bdd, q)), 6)
                         for q in (0.5, 0.1, 0.01, 0.0)],
        "dd_underwater": round(float((bdd < -0.0001).mean()), 4),
        "horizon_hist": horizon_hist, "annual": annual, "qq": qq,
        "curve": curve, "live_returns": live_rows, "dd_hist": dd_hist,
    }


def build_pnl() -> tuple[list[dict], dict]:
    for f in (STMT,):
        if not f.is_file():
            raise SystemExit(f"[ABORT] missing {f}; run stage 4 first")
    if not POS.is_dir():
        raise SystemExit(f"[ABORT] missing {POS}; run stage 3 first")

    tb = _load(BOOK_PY, "tb_pnl")
    per: dict[str, list[dict]] = {}
    for f in sorted(POS.glob("*.parquet")):
        bars = set(tb.load_book(f.stem).get_column("date").to_list())
        q = pl.read_parquet(f).select(["date", "pnl_USD", "N_contracts"])

        own = [(d, n) for d, n in zip(q.get_column("date").to_list(),
                                      q.get_column("N_contracts").to_list())
               if d in bars]
        held = {}
        for i, (d, _) in enumerate(own):
            n1 = own[i - 1][1] if i >= 1 else 0.0
            n2 = own[i - 2][1] if i >= 2 else 0.0
            held[d] = bool((n1 or 0.0) != 0.0 or (n2 or 0.0) != 0.0)

        w = q.filter(pl.col("date") >= WINDOW_START)
        for d, g in zip(w.get_column("date").to_list(),
                        w.get_column("pnl_USD").to_list()):
            per.setdefault(d, []).append(
                {"instrument": f.stem, "gross_pnl_USD": round(float(g or 0.0), 2),
                 "session": d in bars, "held": held.get(d, False)})

    st = pl.read_parquet(STMT).filter(pl.col("date") >= WINDOW_START)
    days, index = {}, []
    for r in st.iter_rows(named=True):
        d = r["date"]
        rows = sorted(per.get(d, []),
                      key=lambda x: (-x["gross_pnl_USD"], x["instrument"]))
        tot = sum(x["gross_pnl_USD"] for x in rows)
        if abs(tot - r["gross_pnl_USD"]) > 1.0:
            raise SystemExit(
                f"[ABORT] {d}: instruments sum to {tot:,.2f} but the statement "
                f"says {r['gross_pnl_USD']:,.2f}. Nothing was written.")
        book = {k: (r[k] if k in ("interest_from_date", "calendar_days")
                    else _f(r[k])) for k in BOOK_COLS}
        _guard(f"pnl {d}", rows, PNL_COLS)
        _guard(f"book {d}", [book], BOOK_COLS)
        days[d] = {"date": d, "instruments": rows, "book": book}
        index.append({"date": d,
                      "gross_pnl_USD": round(float(r["gross_pnl_USD"]), 2),
                      "net_total_USD": round(float(r["closing_equity_USD"]
                                                   - r["opening_equity_USD"]), 2)})
    _guard("pnl index", index, PNL_INDEX_COLS)
    _dates_guard("pnl index", index)
    return index, days


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="build and validate, write nothing")
    ap.add_argument("--stdout", action="store_true")
    a = ap.parse_args()
    latest, history, index, days = build()
    pnl_index, pnl_days = build_pnl()
    mapping = build_mapping()
    m = latest["meta"]
    print(f"  as of {m['as_of']}   {m['sessions']} sessions from "
          f"{m['window_start']}")
    print(f"  executed {len(latest['executed'])}   pending "
          f"{len(latest['pending'])}   outstanding {len(latest['outstanding'])}")
    print(f"  equity {m['equity_start']:,.0f} -> {m['equity_end']:,.0f}   "
          f"net {m['net_ann_ret']:.2%}/yr   Sharpe {m['net_sharpe']}   "
          f"maxDD {m['max_drawdown']:.2%}")
    if a.stdout:
        print(json.dumps(latest, indent=2)[:2000])
    print(f"  archive {len(days)} session(s)")
    print(f"  mapping {len(mapping)} instruments")
    print(f"  attribution {len(pnl_days)} session(s), "
          f"{len(pnl_days[pnl_index[-1]['date']]['instruments'])} instruments "
          f"on {pnl_index[-1]['date']}")
    qa = build_qa()
    print(f"  Q&A  corr vs {qa['bench']} {qa['corr']:+.3f} on {qa['n']} sessions"
          f"   {len(qa['positions'])} positions   "
          f"{qa['exposure']['leverage']:.2f}x gross")
    ex = build_expectations()
    d = next(r for r in ex["moments"] if r["horizon"] == "daily")
    print(f"  expectations  backtest {ex['backtest_start']}..."
          f"{ex['backtest_end']} {d['n']:,} sessions   "
          f"skew {d['skew']:+.2f} vs live {ex['live_moments'][0]['skew']:+.2f}"
          f"   KS {ex['ks_d']:.4f} vs {ex['ks_crit']:.4f} crit"
          f" -> {'REJECT' if ex['ks_reject'] else 'cannot reject'}")
    _window_sweep("latest.json", latest)
    _window_sweep("history.json", history)
    _window_sweep("index.json", index)
    _window_sweep("mapping.json", mapping)
    _window_sweep("pnl_index.json", pnl_index)
    _window_sweep("qa.json", qa)
    _window_sweep("expectations.json", ex)
    for _d, _pl in days.items():
        _window_sweep(f"days/{_d}.json", _pl)
    for _d, _pl in pnl_days.items():
        _window_sweep(f"pnl/{_d}.json", _pl)
    print(f"  window sweep  {7 + len(days) + len(pnl_days)} payload(s) clean, "
          f"nothing dated before {WINDOW_START}")
    if a.check:
        print("  --check: whitelist, window and page guards passed, "
              "nothing written")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "days").mkdir(exist_ok=True)
    (OUT / "pnl").mkdir(exist_ok=True)
    latest_json = json.dumps(latest, indent=1)
    (OUT / "latest.json").write_text(latest_json, encoding="utf-8")
    (OUT / "history.json").write_text(json.dumps(history, separators=(",", ":")),
                                      encoding="utf-8")
    (OUT / "index.json").write_text(
        json.dumps({"days": index, "sessions": int(m["sessions"])},
                   separators=(",", ":")),
        encoding="utf-8")
    for d, payload in days.items():
        (OUT / "days" / f"{d}.json").write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    (OUT / "mapping.json").write_text(
        json.dumps({"instruments": mapping}, separators=(",", ":")),
        encoding="utf-8")
    (OUT / "pnl_index.json").write_text(
        json.dumps({"days": pnl_index}, separators=(",", ":")), encoding="utf-8")
    (OUT / "qa.json").write_text(json.dumps(qa, separators=(",", ":")),
                                 encoding="utf-8")
    (OUT / "expectations.json").write_text(
        json.dumps(ex, separators=(",", ":")), encoding="utf-8")
    for d, payload in pnl_days.items():
        (OUT / "pnl" / f"{d}.json").write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    stamp = build_stamp(latest)
    touched = stamp_pages(stamp)
    print(f"  cache version {stamp}   "
          + (f"stamped into {', '.join(touched)}" if touched
             else "already current in all pages"))
    tot = sum(f.stat().st_size for f in OUT.rglob("*.json"))
    print(f"  wrote latest.json, history.json, index.json, pnl_index.json, "
          f"{len(days)} day file(s) and {len(pnl_days)} attribution file(s)"
          f"   {tot / 1024:,.0f} KB total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
