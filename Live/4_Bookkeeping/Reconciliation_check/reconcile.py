from __future__ import annotations

import argparse
import glob
import importlib.util
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
LIVE = HERE.parent.parent
POS = LIVE / "3_Portfolio" / "Positions"
PORT = LIVE / "3_Portfolio" / "Portfolio.parquet"
BOOK_PY = LIVE / "2_Engine" / "trading_book.py"
PORT_PY = LIVE / "3_Portfolio" / "portfolio.py"
IRX = LIVE / "2_Engine" / "IRX" / "IRX.parquet"
LEDGER = LIVE / "4_Bookkeeping" / "Orders.parquet"
STATEMENT = LIVE / "4_Bookkeeping" / "statement.parquet"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _f(x) -> float:
    return 0.0 if x is None or x != x else float(x)


def _ffill(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    for k in range(1, len(out)):
        if not np.isfinite(out[k]):
            out[k] = out[k - 1]
    return out


class Ties:
    def __init__(self) -> None:
        self.rows: list[tuple] = []
        self.secs = 0.0

    def add(self, name: str, a: float, b: float, tol: float = 1e-9,
            unit: str = "$", note: str = "") -> None:
        scale = max(abs(a), abs(b), 1.0)
        rel = abs(a - b) / scale
        self.rows.append((rel <= tol, name, a, b, rel, unit, note))

    def report(self, quiet: bool, secs: float | None = None) -> int:
        secs = self.secs if secs is None else secs
        w = max(len(r[1]) for r in self.rows)
        bar = "=" * (w + 58)
        print(bar)
        print("  RECONCILIATION")
        print(bar)
        for ok, name, a, b, rel, unit, note in self.rows:
            if unit == "$":
                va, vb = f"{a / 1e9:>13,.4f}B", f"{b / 1e9:>13,.4f}B"
            elif unit == "ct":
                va, vb = f"{a:>14,.0f}", f"{b:>14,.0f}"
            else:
                va, vb = f"{a:>14,.0f}", f"{b:>14,.0f}"
            print(f"  [{'OK  ' if ok else 'FAIL'}] {name:<{w}} {va} {vb}"
                  f"  rel {rel:.1e}")
            if note and not quiet:
                print(f"         {'':<{w}} {note}")
        bad = sum(1 for r in self.rows if not r[0])
        print("-" * (w + 58))
        print(f"  {len(self.rows) - bad}/{len(self.rows)} ties"
              + (f"   {bad} BROKEN" if bad else "") + f"   ({secs:.0f}s)")
        return bad


def ties() -> Ties | None:
    t0 = time.time()
    for f in (PORT, LEDGER, STATEMENT):
        if not f.is_file():
            return None
    tb = _load(BOOK_PY, "tb")
    start_date = _load(PORT_PY, "pf").START_DATE

    P = pl.read_parquet(PORT)
    led = pl.read_parquet(LEDGER)
    S = pl.read_parquet(STATEMENT)
    pdates = P.get_column("date").to_list()
    T = Ties()

    led_q: dict[tuple, float] = defaultdict(float)
    led_c: dict[tuple, float] = defaultdict(float)
    for i, d, q, c in zip(led.get_column("instrument").to_list(),
                          led.get_column("decision_date").to_list(),
                          led.get_column("quantity").to_list(),
                          led.get_column("commission_USD").to_list()):
        led_q[(i, d)] += q
        led_c[(i, d)] += _f(c)

    pnl_recomp = pnl_file = 0.0
    comm_recomp = comm_s3 = qty_s3 = notional = 0.0
    day_pnl: dict[str, float] = defaultdict(float)
    earned: dict[tuple, float] = defaultdict(float)
    live: set[tuple] = set()
    worst = (0.0, "")

    for path in sorted(POS.glob("*.parquet")):
        inst = path.stem
        t = pl.read_parquet(path, columns=["date", "symbol", "N_contracts",
                                           "pnl_USD", "cost_USD",
                                           "pnl_gap_USD", "pnl_day_USD"])
        bk = tb.load_book(inst).select(["date", "Continuous_C", "Continuous_O",
                                        "close", "FX_rate"])
        m = {d: (a, b, c, e) for d, a, b, c, e in zip(
            bk.get_column("date").to_list(),
            bk.get_column("Continuous_C").to_list(),
            bk.get_column("close").to_list(),
            bk.get_column("FX_rate").to_list(),
            bk.get_column("Continuous_O").to_list())}
        dts = t.get_column("date").to_list()
        sy = t.get_column("symbol").to_list()
        nan3 = (np.nan, np.nan, np.nan, np.nan)
        C = _ffill(np.array([m.get(d, nan3)[0] for d in dts], float))
        R = _ffill(np.array([m.get(d, nan3)[1] for d in dts], float))
        X = _ffill(np.array([m.get(d, nan3)[2] for d in dts], float))
        O = _ffill(np.array([m.get(d, nan3)[3] for d in dts], float))
        N = np.array([_f(x) for x in t.get_column("N_contracts").to_list()])
        pf = np.array([_f(x) for x in t.get_column("pnl_USD").to_list()])
        cf = np.array([_f(x) for x in t.get_column("cost_USD").to_list()])
        ps = tb.pointsize_of(inst)
        one_way = tb.cost_rt_of(inst) / 2.0

        bar = np.array([x is not None for x in
                        t.get_column("symbol").to_list()], dtype=bool)
        Cb, Ob, Xb, Nb = C[bar], O[bar], X[bar], N[bar]
        gap = np.full_like(Cb, np.nan); gap[1:] = Ob[1:] - Cb[:-1]
        day = Cb - Ob
        g = (np.concatenate([[0.0], Nb[:-2]]) * gap[1:]
             + Nb[:-1] * day[1:]) * ps * Xb[1:]
        r = float(np.nansum(np.where(np.isfinite(g), g, 0.0)))
        pnl_recomp += r
        pnl_file += float(pf.sum())
        e = abs(r - float(pf.sum())) / max(abs(r), 1.0)
        if e > worst[0]:
            worst = (e, inst)

        v = np.abs(N) * R * ps * X
        notional += float(np.nansum(np.where(np.isfinite(v), v, 0.0)))

        for k in range(1, len(dts)):
            day_pnl[dts[k]] += pf[k]

        own = t.filter(pl.col("symbol").is_not_null())
        osy = own.get_column("symbol").to_list()
        ogp = [_f(x) for x in own.get_column("pnl_gap_USD").to_list()]
        ody = [_f(x) for x in own.get_column("pnl_day_USD").to_list()]
        oN = [_f(x) for x in own.get_column("N_contracts").to_list()]
        for k in range(1, len(osy)):
            if k >= 2 and oN[k - 2]:
                earned[(inst, osy[k - 2])] += ogp[k]
            if oN[k - 1]:
                earned[(inst, osy[k - 1])] += ody[k]

        for k, d in enumerate(dts):
            fx = m.get(d, nan3)[2]
            q = led_q.get((inst, d), 0.0)
            if q and fx and fx == fx:
                comm_recomp += q * one_way * fx
            if cf[k] and fx and fx == fx and one_way:
                qty_s3 += cf[k] / (one_way * fx)
        comm_s3 += float(cf.sum())

        held = t.filter(pl.col("symbol").is_not_null())
        if held.height and held.get_column("N_contracts")[-1]:
            live.add((inst, held.get_column("symbol")[-1]))

    T.add("A  P&L recomputed from books x positions", pnl_recomp, pnl_file,
          note=f"worst instrument {worst[1]}, rel {worst[0]:.1e}")
    T.add("B  instrument P&L sums to the portfolio", sum(day_pnl.values()),
          float(np.nansum(P.get_column("pnl_USD").to_numpy().astype(float))))
    T.add("C  commission recomputed from the mapping", comm_recomp, comm_s3)
    T.add("D  turnover: ledger legs vs stage 3's billed qty",
          float(led.get_column("quantity").sum()), qty_s3, 1e-6, "ct",
          note="differed by 143.5M contracts before the roll fix")

    sc = {d: _f(x) for d, x in zip(S.get_column("date").to_list(),
                                   S.get_column("commission_USD").to_list())}
    day_led: dict[str, float] = defaultdict(float)
    for (_i, d), c in led_c.items():
        day_led[d] += c
    mis = 0
    tot_l = tot_s = 0.0
    for k in range(1, len(pdates)):
        a, b = day_led.get(pdates[k - 1], 0.0), sc.get(pdates[k], 0.0)
        tot_l += a
        tot_s += b
        if abs(a - b) > max(0.01, abs(b) * 1e-9):
            mis += 1
    T.add("E  ledger commission vs statement, per day", tot_l, tot_s, 1e-9, "$",
          note=f"{mis} of {len(pdates) - 1:,} days disagree; the total sits "
               f"below C by the last session's cost, which cost_lag has not "
               f"yet paid")

    rf = {d: _f(x) for d, x in zip(
        pl.read_parquet(IRX).get_column("date").to_list(),
        pl.read_parquet(IRX).get_column("rf_accrual_next").to_list())}
    eq = np.array([_f(x) for x in P.get_column("equity_USD").to_list()])
    cost = np.array([_f(x) for x in P.get_column("cost_USD").to_list()])
    ist = np.array([_f(x) for x in P.get_column("interest_USD").to_list()])
    if "started" in P.columns:
        gate = [bool(x) for x in P.get_column("started").to_list()]
        src = "the run's own `started` column"
    else:
        gate = [d >= start_date for d in pdates]
        src = f"START_DATE {start_date} (no `started` column)"
    rec = 0.0
    for k in range(len(eq) - 1):
        if eq[k] and gate[k]:
            rec += eq[k] * rf.get(pdates[k], 0.0)
    first = next((pdates[k] for k in range(len(gate)) if gate[k]), "-")
    T.add("F  interest recomputed from IRX x balance", rec, float(ist.sum()),
          1e-6, note=f"last night's closing equity, accruing from {first}, "
                     f"gated by {src}")

    pnl_p = np.array([_f(x) for x in P.get_column("pnl_USD").to_list()])
    start = int(np.argmax(eq > 0))
    flows = pnl_p - np.concatenate([[0.0], cost[:-1]]) + ist
    T.add("G  equity == NAV0 + cumulative flows",
          float(eq[start]) + float(flows[start + 1:].sum()), float(eq[-1]))

    booked = 0.0
    for i, c, v in zip(led.get_column("instrument").to_list(),
                       led.get_column("contract").to_list(),
                       led.get_column("realised_pnl_USD").to_list()):
        if (i, c) not in live:
            booked += _f(v)
    closed = sum(v for k, v in earned.items() if k not in live)
    T.add("H  realised P&L == P&L of contracts now closed", booked, closed,
          note=f"{len(earned) - len(live):,} closed contracts, "
               f"{len(live)} still open and excluded from both sides")

    bad = 0
    for path in sorted(POS.glob("*.parquet")):
        inst = path.stem
        t = (pl.read_parquet(path, columns=["date", "symbol", "N_contracts"])
             .filter(pl.col("symbol").is_not_null()))
        by: dict[str, list] = defaultdict(list)
        for r in led.filter(pl.col("instrument") == inst).iter_rows(named=True):
            by[r["decision_date"]].append(r)
        held: dict[str, float] = {}
        for d, s, n in zip(t.get_column("date").to_list(),
                           t.get_column("symbol").to_list(),
                           t.get_column("N_contracts").to_list()):
            for r in by.get(d, ()):
                held[r["contract"]] = held.get(r["contract"], 0.0) + (
                    r["quantity"] if r["action"] == "BUY" else -r["quantity"])
            held = {c: v for c, v in held.items() if v}
            n = _f(n)
            if held.get(s, 0.0) != n or len(held) > bool(n):
                bad += 1
    T.add("I  ledger replay reproduces every position", 0.0, float(bad), 0.0,
          "err", note="divergences, over every instrument-session")

    T.add("J  notional recomputed from raw closes", notional,
          float(np.nansum(
              P.get_column("gross_notional_USD").to_numpy().astype(float))))

    jrl = LIVE / "4_Bookkeeping" / "Journal" / "orders"
    if jrl.is_dir() and any(jrl.glob("*/*.parquet")):
        J = pl.concat([pl.read_parquet(f) for f in sorted(jrl.glob("*/*.parquet"))])
        ok_causal = int((J.get_column("decision_date")
                         <= J.get_column("panel_edge")).sum())
        led_key = {(r["decision_date"], r["instrument"], r["contract"]):
                   (r["action"], r["quantity"], r["kind"])
                   for r in led.iter_rows(named=True)}
        common = agree = 0
        for r in J.iter_rows(named=True):
            k = (r["decision_date"], r["instrument"], r["contract"])
            v = led_key.get(k)
            if v is None:
                continue
            common += 1
            if (v[0] == r["action"] and abs(v[1] - r["quantity"]) < 1e-9
                    and v[2] == r["kind"]):
                agree += 1
        pct = (100.0 * agree / common) if common else 100.0
        T.add("K  journal: no order predates its own panel edge",
              float(ok_causal), float(J.height), 0.0, "n",
              note=f"{J.height:,} journalled; {common:,} also in today's ledger, "
                   f"{agree:,} identical ({pct:.1f}%) -- drift is expected and "
                   f"is not a failure")
    else:
        T.add("K  journal: no order predates its own panel edge", 0.0, 0.0,
              0.0, "n", note="no journal store yet")

    T.secs = time.time() - t0
    return T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true",
                    help="one line per tie, no explanatory notes")
    args = ap.parse_args()
    T = ties()
    if T is None:
        print("[ABORT] missing an upstream artifact; run stages 3 and 4 first")
        return 2
    return 1 if T.report(args.quiet) else 0


if __name__ == "__main__":
    sys.exit(main())
