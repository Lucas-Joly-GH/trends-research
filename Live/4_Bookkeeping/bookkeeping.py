from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
POS = HERE.parent / "3_Portfolio" / "Positions"
PORT = HERE.parent / "3_Portfolio" / "Portfolio.parquet"
BOOK_PY = HERE.parent / "2_Engine" / "trading_book.py"
IRX = HERE.parent / "2_Engine" / "IRX" / "IRX.parquet"
LEDGER = HERE / "Orders.csv"
PENDING = HERE / "pending.csv"
EXECUTED = HERE / "executed.csv"
STATEMENT = HERE / "statement.csv"

COLS = ["decision_date", "execute_at", "instrument", "contract", "action",
        "quantity", "kind", "position_before", "position_after",
        "decision_close", "commission_USD", "realised_pnl_USD"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def orders_for(inst: str, f: Path, tb) -> list[tuple]:
    t = (pl.read_parquet(f)
         .select(["date", "symbol", "N_contracts", "pnl_USD",
                  "pnl_gap_USD", "pnl_day_USD"])
         .filter(pl.col("symbol").is_not_null()))
    dts = t.get_column("date").to_list()
    sym = t.get_column("symbol").to_list()
    N = t.get_column("N_contracts").to_list()
    pgap = t.get_column("pnl_gap_USD").to_list()
    pday = t.get_column("pnl_day_USD").to_list()

    bk = tb.load_book(inst).select(["date", "close", "FX_rate"])
    px = dict(zip(bk.get_column("date").to_list(),
                  zip(bk.get_column("close").to_list(),
                      bk.get_column("FX_rate").to_list())))
    one_way = tb.cost_rt_of(inst) / 2.0

    bucket: dict[str, float] = {}
    out: list[tuple] = []
    pend: list[tuple[int, str, float, float]] = []

    def _take(c: str, q0: float, q1: float) -> float:
        b = bucket.get(c, 0.0)
        if q0 == 0.0:
            return 0.0
        if q1 == 0.0:
            bucket.pop(c, None)
            return b
        if abs(q1) < abs(q0):
            got = b * (abs(q0) - abs(q1)) / abs(q0)
            bucket[c] = b - got
            return got
        return 0.0

    def _settle() -> None:
        for idx, c, q0, q1 in pend:
            r = out[idx]
            out[idx] = r[:11] + (_take(c, q0, q1),)
        pend.clear()

    for k in range(len(dts)):
        n0 = N[k - 1] if k else 0.0
        s0 = sym[k - 1] if k else None
        n1, s1 = N[k], sym[k]
        n0 = 0.0 if n0 is None else n0
        n1 = 0.0 if n1 is None else n1
        if k:
            if k >= 2 and sym[k - 2] is not None:
                g = pgap[k]
                if g is not None and g == g:
                    bucket[sym[k - 2]] = bucket.get(sym[k - 2], 0.0) + g
            if s0 is not None:
                d = pday[k]
                if d is not None and d == d:
                    bucket[s0] = bucket.get(s0, 0.0) + d
        _settle()
        if n0 == 0.0 and n1 == 0.0:
            continue
        dec = dts[k]
        nxt = dts[k + 1] if k + 1 < len(dts) else None
        raw, fx = px.get(dec, (None, None))
        fee = (lambda q: (q * one_way * fx) if fx is not None and fx == fx
               else None)

        rolled = (s0 is not None) and (s1 is not None) and (s0 != s1)
        if rolled:
            if n0:
                out.append((dec, nxt, inst, s0, "SELL" if n0 > 0 else "BUY",
                            abs(n0), "ROLL_OUT", n0, 0.0, raw, fee(abs(n0)),
                            0.0))
                pend.append((len(out) - 1, s0, n0, 0.0))
            if n1:
                out.append((dec, nxt, inst, s1, "BUY" if n1 > 0 else "SELL",
                            abs(n1), "ROLL_IN", 0.0, n1, raw, fee(abs(n1)), 0.0))
        elif n0 != n1:
            d = n1 - n0
            kind = ("OPEN" if n0 == 0.0 else
                    "CLOSE" if n1 == 0.0 else "RESIZE")
            c = s1 or s0
            out.append((dec, nxt, inst, c, "BUY" if d > 0 else "SELL",
                        abs(d), kind, n0, n1, raw, fee(abs(d)), 0.0))
            pend.append((len(out) - 1, c, n0, n1))
    _settle()
    return out


def build(tb) -> pl.DataFrame:
    files = sorted(POS.glob("*.parquet"))
    if not files:
        raise SystemExit(f"[ABORT] no position files in {POS}; run stage 3 first")
    rows: list[tuple] = []
    for f in files:
        rows += orders_for(f.stem, f, tb)
    if not rows:
        return pl.DataFrame({c: [] for c in COLS})
    return (pl.DataFrame(rows, orient="row", schema={
                "decision_date": pl.Utf8, "execute_at": pl.Utf8,
                "instrument": pl.Utf8, "contract": pl.Utf8, "action": pl.Utf8,
                "quantity": pl.Float64, "kind": pl.Utf8,
                "position_before": pl.Float64, "position_after": pl.Float64,
                "decision_close": pl.Float64, "commission_USD": pl.Float64,
                "realised_pnl_USD": pl.Float64})
              .sort(["decision_date", "instrument", "kind"]))


def statement() -> pl.DataFrame:
    P = pl.read_parquet(PORT)
    d = P.get_column("date").to_list()
    g = lambda c: (P.get_column(c).to_list() if c in P.columns
                   else [None] * len(d))
    eq, pnl = g("equity_USD"), g("pnl_USD")
    cost, ist = g("cost_lag_USD"), g("interest_USD")
    appl = g("rf_accrual_applied")
    rate = {}
    if IRX.is_file():
        ix = pl.read_parquet(IRX)
        rate = dict(zip(ix.get_column("date").to_list(),
                        zip(ix.get_column("irx_bey_pct").to_list(),
                            ix.get_column("cal_days_to_next").to_list())))
    n = len(d)
    f = lambda v: 0.0 if v is None or v != v else float(v)
    base, frm, days, ann = [], [], [], []
    for k in range(n):
        a = f(appl[k])
        base.append(f(ist[k]) / a if a else None)
        frm.append(d[k - 1] if k else None)
        r = rate.get(d[k - 1]) if k else None
        ann.append(r[0] if r else None)
        days.append(r[1] if r else None)
    return pl.DataFrame({
        "date": d,
        "opening_equity_USD": [None] + [f(x) for x in eq[:-1]],
        "gross_pnl_USD": [f(x) for x in pnl],
        "commission_USD": [f(x) for x in cost],
        "interest_USD": [f(x) for x in ist],
        "closing_equity_USD": [f(x) for x in eq],
        "interest_base_USD": base,
        "interest_from_date": frm,
        "rate_cal_day": [f(x) for x in appl],
        "calendar_days": days,
        "rate_annual_pct": ann,
    })

def views(led: pl.DataFrame, asof: str | None):
    dec = led.get_column("decision_date")
    if not led.height:
        return led, led
    asof = asof or dec.max()
    exe = led.filter(pl.col("execute_at") == asof)
    pend = led.filter((pl.col("decision_date") <= asof)
                      & (pl.col("execute_at").is_null()
                         | (pl.col("execute_at") > asof)))
    return pend, exe


def report(led: pl.DataFrame, pend: pl.DataFrame, exe: pl.DataFrame,
           asof: str) -> None:
    print(f"\n  ledger : {led.height:,} orders   "
          f"{led.get_column('decision_date').min()} .. "
          f"{led.get_column('decision_date').max()}")
    k = led.group_by("kind").len().sort("len", descending=True)
    print("  by kind: " + "   ".join(f"{r['kind']} {r['len']:,}"
                                     for r in k.iter_rows(named=True)))
    rolls = led.filter(pl.col("kind").is_in(["ROLL_OUT", "ROLL_IN"]))
    if rolls.height:
        n_ev = rolls.select(["decision_date", "instrument"]).n_unique()
        print(f"  rolls  : {n_ev:,} events, "
              f"{rolls.get_column('quantity').sum():,.0f} contracts -- the "
              f"trading a dN-based ledger would have hidden")
    for lab, v in (("PENDING  (send for the next open)", pend),
                   ("EXECUTED (filled at this open)", exe)):
        print(f"\n  {lab}   as of {asof}   {v.height} order(s)")
        if not v.height:
            print("    (none)")
            continue
        with pl.Config(tbl_rows=14, tbl_width_chars=175, fmt_str_lengths=12):
            print(v.select(["instrument", "contract", "action", "quantity",
                            "kind", "decision_close", "commission_USD",
                            "realised_pnl_USD", "execute_at"]))
        print(f"    commission {v.get_column('commission_USD').sum():,.0f} USD"
              f"    realised P&L "
              f"{v.get_column('realised_pnl_USD').sum():,.0f} USD")


def cash_line(st: pl.DataFrame, asof: str) -> None:
    row = st.filter(pl.col("date") == asof)
    if not row.height:
        return
    r = row.to_dicts()[0]
    print(f"\n  CASH  {asof}")
    print(f"    opening equity   {r['opening_equity_USD'] or 0:>20,.0f}")
    print(f"    gross P&L        {r['gross_pnl_USD']:>20,.0f}")
    print(f"    commission       {-r['commission_USD']:>20,.0f}")
    print(f"    interest         {r['interest_USD']:>20,.0f}   on "
          f"{r['interest_base_USD'] or 0:,.0f} carried from "
          f"{r['interest_from_date']}")
    print(f"    {'':<21}{'':>20}   at {r['rate_annual_pct'] or 0:.3f}% annual "
          f"over {r['calendar_days'] or 0:.0f} calendar day(s)"
          f"  = {r['rate_cal_day']:.8f}")
    print(f"    closing equity   {r['closing_equity_USD']:>20,.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None,
                    help="treat this session as 'today' (default: the newest)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    tb = _load(BOOK_PY, "tb")
    led = build(tb)
    dts = sorted(set(led.get_column("decision_date").to_list()))
    asof = args.date or (dts[-1] if dts else "-")
    if args.date and args.date not in dts:
        print(f"  [WARN] no orders decided on {args.date}; the book may have "
              f"been unchanged that session.")
    pend, exe = views(led, args.date)
    report(led, pend, exe, asof)
    st = statement() if PORT.is_file() else pl.DataFrame()
    if st.height:
        cash_line(st, asof)

    if not args.no_write:
        HERE.mkdir(parents=True, exist_ok=True)
        out = [(led, LEDGER), (pend, PENDING), (exe, EXECUTED)]
        if st.height:
            out.append((st, STATEMENT))
        for df, p in out:
            df.write_csv(p)
            df.write_parquet(p.with_suffix(".parquet"))
        names = ", ".join(q.name for _d, q in out)
        print(f"\n  wrote {names} -> {HERE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
