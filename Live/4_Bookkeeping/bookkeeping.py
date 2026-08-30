"""
Bookkeeping: positions into orders.  Stage 4 of the pipeline.

Stage 3 says what to HOLD.  This says what to SEND -- and the two are not the
same statement, because a position is a level and an order is a change.

    Orders.csv        the full derived ledger, one row per (session, contract)
    pending.csv       what to send for the next open        <- the actionable one
    executed.csv      what filled at the last open          <- the reconciliation one
    statement.csv     the daily cash account, one row per session

Every order row is PRICED: `decision_close` (the raw contract close on the decision
session -- raw, not Panama, because Panama goes negative on 14 of these books),
`commission_USD` for that leg, and `realised_pnl_USD`, the P&L the transaction
crystallised.

Interest is NOT on an order row, and could not be: it is earned on a balance
over the gap between two sessions whether or not anything traded, so on a quiet
day it would have nowhere to go and on a busy one it would have to be divided
among the day's orders by an invented rule.  It gets `statement.csv` instead,
where the day reconciles as

    opening_equity + gross_pnl - commission + interest == closing_equity

and the interest line names the balance it was earned on, the session that
balance came from, the calendar days it accrued over and the bill yield behind
it.  Reads stage 3's Portfolio.parquet and stage 2's IRX.

------------------------------------------------------------------------------
WHAT THE PRICED COLUMNS REVEALED, AND STAGE 3 THEN FIXED
------------------------------------------------------------------------------

Stage 3 charges `|N[t] - N[t-1]| . (cost_rt/2) . FX`.  This file prices every
LEG.  Off a roll the two agree to the cent -- 172,760 sessions -- and on a roll
they cannot, because a roll trades both months while `dN` sees only the
difference:

    stage 3 billed, whole history      $2.624 B
    every leg priced                   $3.707 B
    UNBILLED                           $1.083 B      +41.3%

The sharp part is not the average, it is the tail: **3,924 of 9,377 roll events
(41.8%) have |dN| == 0 exactly** -- the new month is the same size as the old --
so stage 3 charges NOTHING AT ALL for a full two-leg roll.  Not an
underestimate, a zero.

FIXED IN STAGE 3 on 2026-08-29: `portfolio.py` now charges both legs on a roll.
Cost drag went 1.30% -> 1.93% of NAV per year and net Sharpe 1.108 -> 1.054.
The figures above are what the ledger measured on the OLD model, kept because
they are what located the bug.  `verify_bookkeeping` now asserts that the two
implementations agree on rolls -- $0.999B both ways -- rather than reporting how
far apart they are, so the correction cannot be lost to a later edit.

------------------------------------------------------------------------------
WHY THE LEDGER STARTS IN 1990
------------------------------------------------------------------------------

It does not: stage 3 does.  `portfolio.py` sets `START_DATE = "1990-01-01"`, so
the first non-zero position in any book is 1990-01-02, and `orders_for` skips
every session where the position is flat on both sides -- no position, no order.
The panel itself reaches back to 1978 and the books are built over all of it;
there is simply nothing to trade before the window opens.  Stage 4 contains no
date logic whatsoever.  Move `START_DATE` and the ledger follows.

ORDERS ARE KEYED ON (INSTRUMENT, CONTRACT), NOT ON INSTRUMENT, and that is the
whole design.  `N_contracts` carries one number per instrument, so a roll --
which is a CLOSE of one delivery month and an OPEN of the next -- appears as a
single small difference.  Measured on this book: 33,693 of 448,235
position-sessions since 1990 change contract (7.52%), and on 32,214 of them
(95.6%) `|dN|` understates the true traded quantity by more than half.  A ledger
built from `dN` would be wrong on those, quietly, in the direction of reporting
less trading than happens.

    kind      what it means                        rows emitted
    OPEN      flat -> a position                   1
    CLOSE     a position -> flat                   1
    RESIZE    same contract, different size        1
    ROLL_OUT  close the expiring month             1 } always
    ROLL_IN   open the new month                   1 } together

`kind` exists because reconciliation questions are almost always "why did we
trade this", and a roll and a signal change are entirely different answers.

------------------------------------------------------------------------------
TIMING
------------------------------------------------------------------------------

The convention is the one the pipeline already runs on: data arrives after the
close, the stages run in the evening, orders go to the next open.  So on the
evening of session t, once stage 3 has produced N[t]:

    pending   N[t] - N[t-1]      to be sent for the open of t+1
    executed  N[t-1] - N[t-2]    filled at t's open, this morning

Both are functions of the position series alone -- the pricing columns read the
books and the portfolio series, but nothing here decides anything, so the stage
still needs no state, no vendor and no network.

`decision_date` is the session whose data produced the order; `execute_at` is
the session at whose OPEN it is meant to fill.  Keeping both means a row can be
reconciled against either the model or the broker without inferring a lag.

THE POSITION FILES ARE ON THE PANEL'S UNION GRID, NOT ON EACH INSTRUMENT'S OWN
CALENDAR.  A row exists for 6A on Presidents' Day because some other market
traded; `symbol` is null there and the position is carried forward, which is
correct -- a holiday does not flatten a book -- but it means the previous ROW is
not the previous SESSION.  Two consequences, both measured on this book and both
handled by iterating only the sessions where the instrument actually has a bar:

    578 rolls sit across a gap (6A rolled 2001U -> 2001Z over the 9/11
        closure).  Reading `symbol` from the adjacent row reads the gap's null,
        the roll fails to register, and the pair collapses into one RESIZE --
        the precise failure this module exists to prevent.
  5,105 order-sessions (2.80%) would take an `execute_at` naming a day that
        market is shut.

For the same reason the two views key on `execute_at`, not on `decision_date`.
An instrument shut today had its last order decided yesterday evening, and that
order did NOT fill at this morning's open, because there was no open: it is
still pending.  Selecting on the decision date would report it as filled.

------------------------------------------------------------------------------
DERIVED, NOT A JOURNAL -- AND THE DIFFERENCE MATTERS BEFORE GOING LIVE
------------------------------------------------------------------------------

This ledger is RECOMPUTED from Positions on every run.  That makes it
idempotent and reproducible, which is what research wants: the same panel gives
the same orders, always.

IT IS THEREFORE NOT A RECORD OF WHAT WAS SENT.  A vendor revision to history, a
changed roll rule, a different tau -- any of these rewrites the whole ledger
retroactively, including rows describing orders that were genuinely transmitted
last week.  The buffer makes this sharper than it sounds: N[t] depends on
N[t-1] all the way back, so a single upstream change re-derives every order
after it, not just the affected day.

Live trading needs an append-only journal that is written once and never
recomputed.  The schema here is deliberately journal-shaped and now carries a
price -- add `sent_at` and the ACTUAL `fill_price` and `fill_qty` the broker
reports, beside the modelled ones, and it becomes one.  Building that now would
add state before there is anything to protect; the gap between the modelled
fill price and a real one is also where slippage would first become visible,
and nothing in this pipeline models slippage yet.

    python bookkeeping.py
    python bookkeeping.py --date 2026-08-27    the two views as of that session
    python bookkeeping.py --no-write
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import polars as pl

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
    """Every order this instrument implies, priced, over its whole history.

    ONE PASS, COMPARING CONSECUTIVE SESSIONS.  The position series is already
    the executed one -- stage 3 applied the truncation and the 3.36 buffer -- so
    a difference between two sessions IS an order.  Nothing is re-decided here.

    A SESSION WHERE THE CONTRACT CHANGED EMITS TWO ROWS, always, even when the
    two sizes are similar.  Selling 1,234 of the expiring month and buying 1,240
    of the next is 2,474 contracts through the market, not the 6 that `dN`
    would report.  The one exception is a roll out of, or into, flat -- then one
    of the two legs is zero and is not emitted, because an order for nothing is
    not an order.

    ------------------------------------------------------------------------
    THE THREE PRICED COLUMNS
    ------------------------------------------------------------------------

    `decision_close` IS NOT A FILL PRICE, and was called `fill_price` until the
    name was challenged.  Nothing in this pipeline is ever priced at an open:
    it is the close of the DECISION session, which is the price the backtest
    attributes the trade to under its one-day-lag convention.  Naming it after a
    fill claimed a precision the model does not have, and hid the very gap that
    `execute_at` exists to make visible -- the row says "fills at the open of
    t+1" while the money says "priced at the close of t".  The name now pairs
    with `decision_date` so the mismatch is legible in the schema itself.

    RAW close, not the Panama close.  Two reasons and both matter: the raw close
    is the number a human can check against a screen, and the Panama close is
    negative on 14 of these books, which would print a short at -29.11.

    `commission_USD` = quantity . (cost_rt / 2) . FX_rate, one-way, exactly the
    unit price stage 3 uses -- but applied to THIS LEG rather than to the net
    change in the instrument's position.  On a roll the two disagree, and the
    disagreement is the point: see the module docstring.

    `realised_pnl_USD` is PROPORTIONAL CRYSTALLISATION.  Each contract carries a
    bucket of mark-to-market accumulated while it was held; a trade that cuts
    the position by fraction f realises f x bucket, a trade that grows it
    realises nothing, and closing realises the rest.  Per instrument,

        sum(realised) + sum(open buckets) == sum(pnl_USD)

    to 1.8e-13 -- every dollar the backtest booked is either crystallised by a
    transaction or still sitting in a position.  Average-cost basis on the
    price would NOT reconcile, because stage 3 converts each session's P&L at
    that session's FX and a single basis price cannot carry that.
    """
    t = (pl.read_parquet(f)
         .select(["date", "symbol", "N_contracts", "pnl_USD",
                  "pnl_gap_USD", "pnl_day_USD"])
         .filter(pl.col("symbol").is_not_null()))
    dts = t.get_column("date").to_list()
    sym = t.get_column("symbol").to_list()
    N = t.get_column("N_contracts").to_list()
    pgap = t.get_column("pnl_gap_USD").to_list()
    pday = t.get_column("pnl_day_USD").to_list()

    # stage 2's book, for the fill price and the FX the cost is converted at
    bk = tb.load_book(inst).select(["date", "close", "FX_rate"])
    px = dict(zip(bk.get_column("date").to_list(),
                  zip(bk.get_column("close").to_list(),
                      bk.get_column("FX_rate").to_list())))
    one_way = tb.cost_rt_of(inst) / 2.0

    bucket: dict[str, float] = {}
    out: list[tuple] = []
    # ORDERS ARE EMITTED WHEN DECIDED, BUT REALISE WHEN THEY FILL, and under
    # open execution those are different sessions.  A ROLL_OUT decided at k
    # leaves the expiring month held overnight and exited at the open of k+1, so
    # that month is still earning the gap leg of session k+1.  Realising at k --
    # which is what a decision-timeline walk does -- books the contract closed
    # before its last P&L arrives, and the per-contract attribution never ties.
    #
    # So today's rows are parked and settled on the NEXT session, after that
    # session's two legs have been credited.  `pend` holds (row index, contract,
    # qty_before, qty_after) for the orders decided at the previous session.
    pend: list[tuple[int, str, float, float]] = []

    def _take(c: str, q0: float, q1: float) -> float:
        """Crystallise the share of `c`'s bucket this trade closes."""
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
        """Apply the parked orders now that they have filled."""
        for idx, c, q0, q1 in pend:
            r = out[idx]
            out[idx] = r[:11] + (_take(c, q0, q1),)   # 12 cols: 0..10 + realised
        pend.clear()

    for k in range(len(dts)):
        n0 = N[k - 1] if k else 0.0
        s0 = sym[k - 1] if k else None
        n1, s1 = N[k], sym[k]
        n0 = 0.0 if n0 is None else n0
        n1 = 0.0 if n1 is None else n1
        # CREDIT EACH LEG TO THE CONTRACT THAT ACTUALLY EARNED IT.  Under open
        # execution those are two different contracts on a roll session: the
        # overnight gap was earned on the month held at k-2 (the fill had not
        # happened), the rest of the session on the month held at k-1.  Off a
        # roll they coincide and this reduces to a single credit.
        if k:
            if k >= 2 and sym[k - 2] is not None:
                g = pgap[k]
                if g is not None and g == g:
                    bucket[sym[k - 2]] = bucket.get(sym[k - 2], 0.0) + g
            if s0 is not None:
                d = pday[k]
                if d is not None and d == d:
                    bucket[s0] = bucket.get(s0, 0.0) + d
        # ...THEN settle yesterday's orders, which fill at this session's open.
        _settle()
        if n0 == 0.0 and n1 == 0.0:
            continue
        dec = dts[k]
        # `execute_at` is the NEXT session in this instrument's own calendar --
        # which is what `dts` now holds.  Deliberately per-instrument: "the next
        # open" is not one moment across a book spanning Hong Kong to Chicago,
        # and pretending otherwise would put a fill time on a row that no
        # exchange agrees with.  Null on the last session: the order is real and
        # actionable, its fill date simply is not known yet.
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
    # The last session's orders have not filled.  Settle them anyway: the
    # position series already says the contract is gone, so leaving the bucket
    # unrealised would strand P&L on a contract the book reports as closed.
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
    """The daily cash statement.  One row per session, and the day reconciles.

    WHY THIS IS NOT A COLUMN IN executed.csv.  Interest is not a property of a
    transaction -- it is earned on a balance, over a gap between two sessions,
    whether or not anything traded.  Putting it on an order row would force it
    to be divided among that day's orders by some rule, and every such rule is
    invented.  On a day with no orders it would have nowhere to go at all.

    So it gets the object it actually belongs to: an account statement.

        opening_equity + gross_pnl - commission + interest == closing_equity

    checked to 6.9e-15 relative.  `commission` here is `cost_lag_USD`, the cost
    that left the account today, not the cost of today's decision -- see
    `Portfolio_Journal.md`.

    THE INTEREST LINE IS STATED WITH ITS BASE AND ITS SOURCE, which is the
    question this file exists to answer: `interest_base_USD` is the balance the
    rate was applied to, `interest_from_date` is the session that balance and
    that rate were observed on, `calendar_days` is the gap it accrued over (3
    across a weekend), and `rate_annual_pct` is the bill yield behind it.
    Interest credited today was computed last night: the base is last night's
    CLOSING equity, exactly as stage 3 computes it.  It is not reduced by
    today's commission -- that cash leaves at today's open, at the far end of
    the window the credit accrued over.
    """
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
        # The base is recovered from the credit and the rate that produced it,
        # which is exact and needs no assumption about where equity stood.
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
    """(pending, executed) as of the evening of `asof`.

    BOTH SELECT ON `execute_at`, NOT ON `decision_date`, because the two are not
    interchangeable once each market keeps its own calendar:

        executed   execute_at == asof            filled at this morning's open
        pending    execute_at > asof, or null    not filled yet

    `pending` is bounded by `decision_date <= asof` so that replaying a past
    session cannot leak orders decided on information that did not exist yet.
    Within that bound each instrument contributes at most one row, since
    `execute_at` is strictly its next session: the newest decision still unfilled.

    A null `execute_at` is the live case -- decided on the last session in the
    panel, filling at an open that has not happened.  An instrument shut on
    `asof` lands here too: its order was decided yesterday for an open that
    never came, so it carries forward rather than being reported as filled.
    """
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
        # COUNT EVENTS, NOT ROWS // 2.  Five rolls in this history have a single
        # leg because the other side was flat, so halving the row count is off
        # by three and drifts further every time one occurs.
        n_ev = rolls.select(["decision_date", "instrument"]).n_unique()
        print(f"  rolls  : {n_ev:,} events, "
              f"{rolls.get_column('quantity').sum():,.0f} contracts -- the "
              f"trading a dN-based ledger would have hidden")
    for lab, v in (("PENDING  (send for the next open)", pend),
                   ("EXECUTED (filled at this open)", exe)):
        # printed with execute_at, because with per-market calendars the fill
        # session is the non-obvious half and a blank means "next open, TBD"
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
    """The one row of the statement a person actually reads at the close."""
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
