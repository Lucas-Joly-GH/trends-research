"""
Closing the books.  A periodic reconciliation of stages 3 and 4.

    python reconcile.py
    python reconcile.py --quiet      one line per tie, no notes

WHY THIS IS SEPARATE FROM `verify_bookkeeping`.  The nightly checks test the
artifacts AGAINST EACH OTHER.  This does something different: it recomputes each
quantity FROM THE PRIMARY SOURCES -- the trading books, instrument_mapping.csv,
IRX -- and requires the derived files to agree.  A control you run after changing
anything in the money path.

It is NOT separate because it is slow.  It takes about 3 seconds, which is a
correction to an earlier guess of ninety; the 63 books are parquet and the walk
is trivial.  It is separate because it answers a different question.

`Update.py` runs it LAST, through `verify_reconciliation`, which calls `ties()`
and renders the rows in the pipeline's own report format -- two copies of a
reconciliation is two reconciliations, and the second one is always the stale
one.  This file stays runnable on its own, which is the point: a reconciliation
that only ever runs unattended stops being read.

A RECONCILIATION IS ONLY WORTH THE PAPER IT IS ON IF THE TWO SIDES ARE
INDEPENDENT.  Every tie below recomputes one side from source data; none of them
re-runs the code that produced the other side.  Where an identity could only be
written circularly it was rewritten until it could not be -- see H.

    A  P&L        recomputed from books x positions   -> Positions.pnl_USD
    B  P&L        instrument sum                      -> Portfolio.pnl_USD
    C  commission recomputed from the mapping         -> stage 3 cost_USD
    D  turnover   ledger legs                         -> stage 3's billed qty
    E  commission ledger, per day                     -> statement.csv
    F  interest   recomputed from IRX x balance       -> Portfolio.interest_USD
    G  equity     NAV0 + cumulative flows             -> Portfolio.equity_USD
    H  P&L        realised on closed contracts        -> P&L those contracts earned
    I  positions  ledger replay                       -> N_contracts
    J  notional   recomputed from raw closes          -> gross_notional_USD

D IS THE ONE THAT EARNS ITS KEEP.  Stage 3's traded quantity is backed out of
its own `cost_USD`; the ledger's is the sum of order legs.  Until 2026-08-29
those differed by 143.5M contracts, which was the roll under-billing.  Any
future regression in the cost model shows up here as a contract count, before it
shows up anywhere as money.

TWO OF THESE TIES WERE WRONG BEFORE THEY WERE RIGHT, both times because the
IDENTITY was wrong rather than the pipeline.  F accrued interest across the
whole grid, ignoring that stage 3 gates accrual on `started_t` -- so it credited
twelve pre-1990 years of double-digit bill yields on an idle $100M and called
the $109M a break.  (F was later wrong a SECOND time, in the opposite
direction, and that one is worse: it deducted the session's commission from the
base exactly as stage 3 did, so the tie passed while both sides were wrong.  See
the note at F.)  H compared all realised P&L against the P&L of
closed contracts, when a contract still open has already realised P&L on every
partial reduction; two populations, $648M apart, nothing wrong underneath.  Both
notes are kept in the code beside the fix, because the failure mode of an
auditor is writing a false identity confidently.

WHAT A CLEAN RUN DOES NOT MEAN.  It means every derived number is reproducible
from the panel and the metadata, and that the artifacts describe one consistent
set of books.  It says nothing about the conventions being right: close-to-close
P&L attributed against an `execute_at` at the next open is an open
inconsistency, slippage is unmodelled so the commission figure is a floor rather
than an estimate, and the vendor panel is taken as given.  A book can reconcile
perfectly and still measure the wrong thing.
"""
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
    """None and NaN are both 'no number here', and both mean zero to a total."""
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
    """Compute every tie.  None when an upstream artifact is missing.

    SEPARATE FROM THE RENDERING so `Update.py` can fold these into its own
    report format rather than printing a second, differently-shaped table
    beneath the other six.  The arithmetic has one home either way.
    """
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

    # ---- per-instrument pass: A, B, C, D, H, J ---------------------------
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

        # (A) P&L from first principles, ON THE OPEN-EXECUTION SPLIT: the
        # overnight gap belongs to the position held BEFORE the fill, the rest
        # of the session to the one held after.  Panama series throughout,
        # because its differences are the true price move with the roll gap
        # removed.  Recomputed here from the books, so it would catch stage 3
        # reverting to close-to-close.
        # ON THIS MARKET'S OWN SESSIONS.  The gap holder is the target decided
        # TWO own-sessions back, and on the union grid that is not `u-2`: a
        # holiday row sits between and N is carried across it.  Filtering to bar
        # rows first makes the lag mean what it says -- and costs nothing,
        # because both legs are zero on a no-bar row anyway.
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

        # (J) notional off the RAW close -- the Panama series is negative on 14
        # of these books and would report a short position as negative notional.
        v = np.abs(N) * R * ps * X
        notional += float(np.nansum(np.where(np.isfinite(v), v, 0.0)))

        for k in range(1, len(dts)):
            day_pnl[dts[k]] += pf[k]

        # (H) credit each session's P&L to the contract that EARNED it, ON THIS
        # MARKET'S OWN SESSIONS.  Doing it on the union grid loses the P&L of
        # every session that FOLLOWS a holiday: `symbol` is null on the gap row,
        # so the previous ROW names no contract while the previous SESSION names
        # one perfectly well.  That is the same union-grid trap that hid 578
        # rolls in stage 4 and would have mispriced rolls in stage 3; here it
        # dropped $1.74B of earned P&L and broke this tie by 8.5%.  Third time
        # in one pipeline, and the first two were not enough to stop the third.
        # AND TWO LEGS, TWO CONTRACTS on a roll: under open execution the
        # overnight gap was earned on the month held at k-2, the rest of the
        # session on the month held at k-1.
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
            # (C) commission repriced from the mapping, on the ledger's legs
            q = led_q.get((inst, d), 0.0)
            if q and fx and fx == fx:
                comm_recomp += q * one_way * fx
            # (D) stage 3's billed quantity, backed out of its own cost
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

    # ---- E: the statement is the ledger, one session later ---------------
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

    # ---- F: interest --------------------------------------------------------
    #
    # THE BASE IS `equity_USD` ITSELF -- last night's closing equity, nothing
    # subtracted.  This line used to read `(eq[k] - cost[k])` on the reasoning
    # that cash which has paid for a trade is not there to earn, and that
    # reasoning is sound but the timing is not: `cost_USD[k]` is the cost
    # DECIDED at k, which fills at k+1's OPEN, so it leaves at the far end of
    # the window this credit accrues over.  2026-01-05 exposed it -- an account
    # that had sat flat on 100,000,000 across a 3-day weekend, never having
    # traded, was credited interest on 99,917,719.09.
    #
    # WORTH KNOWING HOW THIS SURVIVED: stage 3 and this check encoded the SAME
    # false idea, so F passed and agreed to 1e-6 while both were wrong.  A tie
    # only catches a mistake the two sides do not share, which is the argument
    # for recomputing from primary sources rather than from the other side's
    # intermediate -- and the reason the header says an auditor's failure mode
    # is writing a false identity confidently.
    #
    # The second correction here still stands: NOTHING accrues before the book
    # opens, which stage 3 gates with `started_t`.  Dropping the gate credits
    # twelve pre-1990 years of double-digit bill yields on an idle $100M: a
    # $109M "break" that was the check's fault, not the pipeline's.
    rf = {d: _f(x) for d, x in zip(
        pl.read_parquet(IRX).get_column("date").to_list(),
        pl.read_parquet(IRX).get_column("rf_accrual_next").to_list())}
    eq = np.array([_f(x) for x in P.get_column("equity_USD").to_list()])
    cost = np.array([_f(x) for x in P.get_column("cost_USD").to_list()])
    ist = np.array([_f(x) for x in P.get_column("interest_USD").to_list()])
    # THE GATE COMES FROM THE RUN, NOT FROM THE MODULE DEFAULT.  `--start-date`
    # is a run parameter; reading `START_DATE` instead reconciles against a
    # window the run never used.  Stage 3 now records `started` in its own
    # output for exactly this reason -- the fallback is only for a Portfolio
    # written before that column existed.
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

    # ---- G: the equity curve is its own flows ------------------------------
    pnl_p = np.array([_f(x) for x in P.get_column("pnl_USD").to_list()])
    start = int(np.argmax(eq > 0))
    flows = pnl_p - np.concatenate([[0.0], cost[:-1]]) + ist
    T.add("G  equity == NAV0 + cumulative flows",
          float(eq[start]) + float(flows[start + 1:].sum()), float(eq[-1]))

    # ---- H: realised P&L ----------------------------------------------------
    #
    # BOTH SIDES RANGE OVER THE SAME CONTRACTS, which the first version of this
    # did not: a contract still open has already realised P&L on every partial
    # reduction, so comparing ALL realised against CLOSED-contract P&L compares
    # two populations and breaks by $648M with nothing wrong underneath.
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

    # ---- I: the ledger IS the position path --------------------------------
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

    # ---- K: the journal ----------------------------------------------------
    #
    # WHAT CAN AND CANNOT BE TIED HERE.  Every other tie recomputes a derived
    # number and demands agreement.  The journal is not derived -- it is the
    # record of what was sent -- so there is no quantity to recompute and no
    # invariant that survives a pipeline change.  Asserting journal == ledger
    # would put this report red on every day anyone fixes anything, and a check
    # that is red by default is a check nobody reads.
    #
    # So the tie asserts the one thing that must hold whatever changes:
    # A ROW CANNOT HAVE BEEN DECIDED ON DATA THAT DID NOT EXIST.  Every
    # journalled order carries the panel edge its run saw, and decision_date can
    # never exceed it.  That catches a fabricated, back-dated or corrupted store,
    # which is the failure mode worth having a hard check for.
    #
    # The agreement with today's ledger is REPORTED in the note, not asserted.
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
