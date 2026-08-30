"""
Front-contract worksheet: every listed month, every session, side by side.

Replicates the hand-made `zc_19890508.csv` and generalises it to all 63
instruments. The point is to make the front-contract decision auditable one
session at a time -- which month each candidate rule picks, and where the rules
disagree.

Merged 2026-08-27 from the three sheets it replaces:

    has_notice_front_contract.py       22 markets, is_deliverable AND has_notice
    not_has_notice_front_contract.py   19 markets, is_deliverable, no notice
    not_deliverable_front_contract.py  22 markets, cash-settled

THE SPLIT WAS NEVER A TRICHOTOMY. Measured with comments stripped the three were
~211/214/224 lines and ~85% identical. The only real axes were a BINARY date
source and one variant column carrying three different names. The extra
`held2` filter in the notice sheet was checked over 1,421,027 candidate sets and
binds 0 times, so the three had no behavioural divergence at all.

*** THE FINDINGS ARE IN Roll_Journal.md, NOT HERE. ***
Every rule below has a session behind it, and the journal names it. Before
deleting anything that "looks unnecessary", read section 2 (negative results)
and section 3 (incidents that decided a rule). This docstring says WHAT the
columns are; the journal says WHY, and what happened when they were absent.

Columns
-------
date, symbol, open, close, volume, open_interest
                    raw, from Contracts/<INST>/<SYMBOL>.csv in the private repo
first_notice        NOTICE markets only. From Data/Contract-Metadata/contracts.csv.
till_notice_cd      NOTICE markets only. first_notice - date, CALENDAR days,
                    signed. Negative once notice has passed.
last_trade          EVERY OTHER MARKET. The scheduled final trading day.
till_last_trade_cd  EVERY OTHER MARKET. last_trade - date, same convention.

                    The label is per group ON PURPOSE. A generic name would hide
                    which gate applied, and that is the first thing a reader of
                    the sheet wants to know. `GATE` below picks the pair.

is_passed           till <= 0. NOTE the <=: for a notice market delivery can be
                    assigned ON the first notice day, so the contract is already
                    unholdable that morning. For the others, termination IS the
                    delivery event, so the boundary is the last trading day.
Best_Vol            Highest volume among contracts eligible today. Eligible
                    carries the window gate: a contract within BEST_VOL_MIN_CD
                    of its gate date cannot win. Reads FALSE there, not blank --
                    it traded, it is simply not somewhere a rule may leave you.
Best_Oi             Highest open interest, SAME window gate. Does NOT require
                    volume, and that asymmetry is deliberate: open interest is
                    what survives a session the vendor recorded no volume for.
B_V_3               Best_Vol true for this contract 3 CONSECUTIVE sessions.
auto_roll           Nearest expiry, stepped on once that contract comes within
                    AUTO_ROLL_CD days of its gate date. Reads no volume and no
                    open interest. SESSION-level.
Auto_Best_V         Whether auto_roll is also the volume leader. A SCORE of that
                    rule, not an input to it.
Forced_roll_V       The alternative where auto_roll is in doubt: nearest
                    contract by gate date that is not auto_roll's own pick and
                    is clear of the window. Blank where the rules already agree.
Forced_Best_V       Whether that alternative is the volume leader. BLANK where
                    Forced_roll_V is blank -- the rule was not asked.
+2_Forced_Roll_V    One step further out, both earlier answers struck off.
+2_Forced_Best_V    Whether that third contract is the leader.
auto_roll_hold      auto_roll verbatim, beside the other hold series so the
                    baseline and the rules read as a pair. A COPY, not a variant.
+1_auto_roll_hold   auto_roll's decision taken one contract further out -- the
                    front month is never held. Same ordering, same window, reads
                    no volume. Built for the compounded-in-arrears STIRs.
forced_roll_hold    auto_roll where Auto_Best_V, else Forced_roll_V where
                    Forced_Best_V, else Best_Oi, else Best_Vol. RATCHETED.
f_r_h_Best_V        Whether forced_roll_hold is the volume leader. Never blank.
                    Expected to read LOWER than the branch scores around it, and
                    not because the rule is worse: three of its four branches are
                    not aiming at Best_Vol.
confirm_forced_roll_hold
                    forced_roll_hold held back until the same contract has been
                    the answer ROLL_CONFIRM_SESSIONS running. Guards against the
                    ratchet making a one-session mistake permanent.
                    ONE COLUMN, replacing RS_/LT_/CS_forced_roll_hold, which
                    were the same algorithm under three names. The Roll_Rule
                    VALUES in contract_cycles.csv deliberately keep the old
                    names: those record WHY each market was cleared.
Test_Hold           auto_roll / Forced_roll_V / +2_Forced_Roll_V, first branch
                    that fires, ratcheted. The third is a fallback and is NOT
                    gated on +2_Forced_Best_V.
Test_Best_V         Whether Test_Hold is the leader. NEVER blank: it is asked
                    every session, so a session it cannot answer is a LOSS.

BLANK IS NOT FALSE. A passed contract gets a blank flag -- out of the running,
not losing it -- and `means` reads blank as null, dropping it from the
denominator. Session-level columns are never blanked per row.

WARM-UP IS LOAD-BEARING. B_V_3 and every hold series carry state across
sessions, so the count accumulates from BEFORE the window. `worksheet` iterates
from the start of contract history and only emits inside it.

    python front_contract.py --instrument ZC --start 1989-05-08 --end 1989-05-23
    python front_contract.py --instrument SO3 --start 2019-11-20 --end 2019-12-13
    python front_contract.py --instrument ZC --means
    python front_contract.py --instrument ZC --check <reference.csv>
"""
from __future__ import annotations

import argparse
import datetime as _dt
from datetime import date as _date
from pathlib import Path

import numpy as np
import polars as pl


def _private() -> Path:
    """Locate the private data repo by walking up, not by counting parents.

    A fixed `parents[N]` breaks the moment the script moves a directory deeper,
    silently resolving to a path that does not exist.
    """
    for p in Path(__file__).resolve().parents:
        cand = p / "LJOLY_Memoire_INSEEC_Msc2"
        if cand.is_dir():
            return cand
    raise SystemExit("[ABORT] cannot find LJOLY_Memoire_INSEEC_Msc2 above "
                     f"{Path(__file__).resolve()}")


PRIVATE = _private()
CONTRACTS = PRIVATE / "Data" / "Paper-trading" / "Contracts"
NOTICE = PRIVATE / "Data" / "Contract-Metadata" / "contracts.csv"
WORKING = Path(__file__).resolve().parent / "worksheet.csv"
CYCLES = Path(__file__).resolve().parents[1] / "contract_cycles.csv"
MONTH_ORDER = "FGHJKMNQUVXZ"

# Rows the DATA PROVIDER should never have published.  Not a rule problem and
# not something a rule can be made to handle -- the bar simply should not exist.
#
#   HO 2007-09-03: US Labor Day.  NYMEX was CLOSED, yet the panel carries a
#   single bar for HO-2008G (563 lots, 12,662 OI, five months out) and nothing
#   else.  The sessions either side carry 36 contracts each.  With one row in
#   the session, any nearest-expiry rule must name it, so the chain jumps from
#   HO-2007V out to HO-2008G and back the next day -- the only backward roll in
#   the entire 19-instrument deliverable-no-notice group.  Deleting the row
#   removes the cause rather than teaching the rule to tolerate a holiday it
#   cannot detect.
BAD_ROWS = {
    ("HO", "2007-09-03", "HO-2008G"),
}

# Whole SESSIONS the provider should not have published -- every contract in
# them prints zero volume, so no rule can choose anything meaningful.
#
#   LFT9 2025-05-06 .. 2025-05-30: SEVENTEEN sessions carrying only 2025Z and
#   2026H, both at zero, while LFT9-2025M -- the front month, trading ~62,000
#   lots and 18 days from last trade -- has NO ROW AT ALL.  The FTSE 100 future
#   did not stop trading for a month; the bars are simply missing.  Any
#   nearest-expiry rule must name one of the two zero rows and then step back
#   when the real contract returns, which is the backward roll this removes.
#   (The pre-merge comment said "ten sessions from 05-16" -- it was written
#   against a truncated listing and never updated. Seventeen, from 05-06.)
BAD_SESSIONS = {
    ("LFT9", d) for d in (
        "2025-05-06", "2025-05-07", "2025-05-08", "2025-05-12", "2025-05-13",
        "2025-05-14", "2025-05-15", "2025-05-16", "2025-05-19", "2025-05-20",
        "2025-05-21", "2025-05-22", "2025-05-23", "2025-05-27", "2025-05-28",
        "2025-05-29", "2025-05-30",
    )
}

BV3_SESSIONS = 3            # consecutive leading sessions to arm B_V_3
AUTO_ROLL_CD = 5            # step off the front this close to its gate date
BEST_VOL_MIN_CD = 5         # a contract this close cannot win Best_Vol/Best_Oi
FORCED_ROLL_MIN_CD = 5      # Forced_roll_V will not park you this close either
ROLL_CONFIRM_SESSIONS = 2   # confirm_forced_roll_hold waits for a repeat
INCEPTION_VOLUME = 1000     # the market is not really trading until some
                            # SELECTABLE contract prints this

BOOL_COLS = ["is_passed", "Best_Vol", "Best_Oi", "B_V_3", "Auto_Best_V",
             "Forced_Best_V", "+2_Forced_Best_V", "Test_Best_V", "f_r_h_Best_V"]

# The two gate labels.  Which pair an instrument gets is decided by `gate`
# below; the names are per group deliberately -- see the docstring.
GATE = {"notice": ("first_notice", "till_notice_cd"),
        "last_trade": ("last_trade", "till_last_trade_cd")}


def sort_key(sym: str) -> int:
    """'ZC-1989K' -> sortable expiry index."""
    tail = sym.split("-")[1]
    return int(tail[:4]) * 12 + MONTH_ORDER.index(tail[4])


def ratchet(cand: str | None, prev: str | None, till: dict) -> str | None:
    """A roll goes one way: refuse a step onto a contract NEARER to its gate
    date than the one already held, and stay put instead.

    `till` is TODAY's distance by symbol, and both sides are read from it.
    Comparing across sessions would score an ordinary hold as a step backwards,
    since every contract's own figure falls by a day overnight.

    Staying put is always the safe side: a backward step means the candidate is
    closer to the gate than the incumbent, so the incumbent is the further of
    the two.  An incumbent no longer listed has no figure today and does not
    block.
    """
    if cand is None or prev is None or cand == prev:
        return cand
    a, b = till.get(prev), till.get(cand)
    return prev if (a is not None and b is not None and b < a) else cand


def gate(inst: str) -> str:
    """'notice' or 'last_trade' -- which date this market is gated on.

    BINARY, not three-way.  has_notice markets gate on first_notice; everything
    else gates on last trade, whether it is deliverable or cash-settled.  That
    is why merging the three sheets lost nothing: the deliverable-no-notice and
    cash-settled sheets were already the same rule on the same field.

    Energy is why the distinction exists at all.  The vendor returns None for
    first_notice on every contract in those markets -- CL 0 of 645, NG 0 of 568,
    HO 8 of 609, and those 8 are 1979 records with notice AFTER last trade --
    because the concept does not exist in the contract structure: trading
    TERMINATES and delivery is assigned afterwards.
    """
    if not CYCLES.exists():
        return "last_trade"
    t = pl.read_csv(CYCLES, infer_schema_length=0)
    row = t.filter(pl.col("instrument") == inst)
    if not row.height:
        return "last_trade"
    return ("notice" if row.get_column("has_notice").to_list()[0] == "true"
            else "last_trade")


def gate_map(inst: str, which: str) -> dict:
    """symbol -> the gate date for this market, as an ISO string.

    One reader for both fields.  Last trade is forward-looking in the vendor's
    data, so a live contract carries its scheduled date rather than "today".
    """
    if not NOTICE.exists():
        return {}
    col = "first_notice" if which == "notice" else "last_trade"
    t = (pl.read_csv(NOTICE, infer_schema_length=0)
         .filter(pl.col("instrument") == inst)
         .select(["symbol", col]))
    return {r["symbol"]: r[col] for r in t.iter_rows(named=True) if r[col]}


def load(inst: str) -> pl.DataFrame:
    rows = []
    for f in sorted((CONTRACTS / inst).glob(f"{inst}-*.csv")):
        t = (pl.read_csv(f, infer_schema_length=0)
             .select(["Date", "Open", "Close", "Volume", "Open Interest"])
             .with_columns(
                 pl.col("Date").cast(pl.Utf8),
                 pl.col("Open").cast(pl.Float64, strict=False),
                 pl.col("Close").cast(pl.Float64, strict=False),
                 pl.col("Volume").cast(pl.Float64, strict=False).fill_null(0.0),
                 pl.col("Open Interest").cast(pl.Float64, strict=False).fill_null(0.0),
                 pl.lit(f.stem).alias("symbol")))
        rows.append(t)
    if not rows:
        raise SystemExit(f"[ABORT] no contracts under {CONTRACTS / inst}")
    d = pl.concat(rows).rename({"Open": "open", "Close": "close",
                                "Volume": "volume",
                                "Open Interest": "open_interest"})
    d = d.with_columns(
        pl.col("Date").str.strptime(pl.Date, "%Y%m%d").alias("date")).drop("Date")
    drop = [_date.fromisoformat(dt) for (i, dt) in BAD_SESSIONS if i == inst]
    if drop:
        d = d.filter(~pl.col("date").is_in(drop))
    for (i, dt, sym) in BAD_ROWS:
        if i != inst:
            continue
        d = d.filter(~((pl.col("date") == _date.fromisoformat(dt))
                       & (pl.col("symbol") == sym)))
    return d


def dead_months(inst: str) -> set:
    """Delivery months this market lists but does not trade.

    Curated in contract_cycles.csv.  ZC lists F and X but holding them gives a
    median 0.8% and 0.4% of session volume against 28.8-62.4% for H/K/N/U/Z, so
    a purely calendar-driven roll walks straight into them.  Empty means NOT YET
    MEASURED for that instrument, not "none".
    """
    if not CYCLES.exists():
        return set()
    t = pl.read_csv(CYCLES, infer_schema_length=0)
    if "Dead_contracts" not in t.columns:
        return set()
    row = t.filter(pl.col("instrument") == inst)
    return set(row.get_column("Dead_contracts").to_list()[0] or "") if row.height else set()


def inception(d: pl.DataFrame, dead: set) -> object:
    """First session on which a SELECTABLE contract printed INCEPTION_VOLUME.

    Measured on the same candidate set selection uses -- dead months excluded.
    Measuring it on every contract starts the panel before anything tradable
    exists: SI's 1980F cleared 1,000 lots on 1978-03-07 and opened the panel,
    but F is dead in silver, so nothing was selectable and auto_roll was empty
    for 37 sessions until 1980H appeared trading 2 lots.

    Before that the instrument is listed but not traded, and the panel shows it:
    CT in 1978 carries a single contract 522 days from notice printing 5-10 lots
    on the sessions it prints at all.

    KNOWN TOO WEAK FOR YOUNG MARKETS.  1,000 lots on ANY contract admitted SO3
    from 2018-08 on a 5,103-lot book with 9.8 months trading, which is 39% of
    that panel and where every SO3 roll pathology lives.  Not yet swept across
    the book -- see Roll_Journal.md section 4.
    """
    sel = (d.filter(~pl.col("symbol").str.slice(-1).is_in(list(dead)))
           if dead else d)
    by_day = (sel.group_by("date").agg(pl.col("volume").max().alias("mx"))
               .filter(pl.col("mx") >= INCEPTION_VOLUME).sort("date"))
    return by_day.get_column("date").to_list()[0] if by_day.height else None


# ----------------------------------------------------------------------------
# AS-OF ALIGNMENT
#
# The panel does not end on the same date for every market.  ASX closes earliest
# in the global day, so YAP4 and YXT4 -- the only two `Day`-session instruments
# in the book -- carry date D while the other 61 are still on D-1, whenever the
# vendor's update lands between the ASX close (~08:30 CEST) and the US close
# (~22:00 CEST).  On 2026-08-25 the updater ran at 17:42 CEST, squarely inside
# that gap.
#
# THIS IS A COLLECTION ARTEFACT, NOT A MARKET FACT, and it must not reach the
# signals.  Using each instrument's own newest bar would make the output a
# function of WHEN the refresh happened to run: the same pipeline an hour later
# gives a different alignment, no backtest reproduces the lead-lag, and every
# cross-sectional estimate -- correlation, vol target, portfolio weight --
# quietly assumes contemporaneous observations it no longer has.
#
# So the panel is squared off at the newest date EVERY instrument has, and bars
# beyond it wait for the others to catch up.  That costs nothing at execution:
# signals computed on date D trade each market's next open, and the ASX close is
# then ~14h old against ES's ~17.5h.  Every instrument trades its next open off
# a same-dated close, which is the convention a backtest already assumes.
#
# Switching YAP4/YXT4 to their All-Sessions twins does NOT fix this -- YAP and
# YXT end on the same date as YAP4 and YXT4.  Measured, not assumed.
# ----------------------------------------------------------------------------

STALE_SESSIONS = 3      # further behind as_of than this is a broken feed,
                        # not the ordinary ragged edge
QUORUM_FRAC = 0.15      # share of the book that must have traded a date for it
QUORUM_MIN = 5          # to count as a real session -- see panel_as_of
EDGE_WINDOW = 40        # recent sessions read per contract to decide the above


def _tail_dates(path: Path, window: int = EDGE_WINDOW) -> list[str]:
    """The last `window` YYYYMMDD values in a contract file, read from the end.

    Seeks rather than reading whole files: this runs over 15,231 files to decide
    one date, and reading each in full is the kind of cost that gets a safety
    check quietly removed from a pipeline.
    """
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 64 * window))
            tail = fh.read().splitlines()
    except OSError:
        return []
    out = []
    for line in reversed(tail):
        head = line.decode("utf-8", "ignore").split(",", 1)[0]
        if len(head) == 8 and head.isdigit():
            out.append(head)
            if len(out) >= window:
                break
    return out


EDGE_LOOKBACK_DAYS = 15     # quorum window.  Must stay well inside the tail
                            # EDGE_WINDOW reads, or dates near the far end are
                            # undercounted by contracts whose tail does not
                            # reach them -- which read as phantom holidays.


def _contract_last_dates(instruments: list[str] | None = None) -> dict:
    """{instrument: {file: its last date}}.  One date per contract, cheap."""
    out = {}
    for d in sorted(CONTRACTS.iterdir()):
        if not d.is_dir():
            continue
        if instruments is not None and d.name not in instruments:
            continue
        got = {}
        for f in d.glob(f"{d.name}-[0-9][0-9][0-9][0-9][A-Z].csv"):
            v = _tail_dates(f, window=1)
            if v:
                got[f] = v[0]
        if got:
            out[d.name] = got
    return out


def panel_sessions(instruments: list[str] | None = None) -> dict:
    """{instrument: set of session dates inside the quorum window}.

    WINDOWED, and the window is short on purpose.  as_of only ever needs the
    last few sessions, and a long window is actively misleading: with a 90-day
    window the counts said 30 markets were shut in late June, which was not a
    holiday at all but contracts whose 40-bar tail simply did not reach that
    far back.  Kept inside the tail, every live contract covers the whole
    window and the counts mean what they say.

    NOTE this deliberately does NOT decide `edge`.  An instrument whose feed
    stopped a month ago has nothing in the window and would vanish from here --
    taking it out of the stale check, which is the one place it must appear.
    """
    per = _contract_last_dates(instruments)
    newest = max((d for g in per.values() for d in g.values()), default="")
    if not newest:
        return {}
    cut = (_dt.date(int(newest[:4]), int(newest[4:6]), int(newest[6:]))
           - _dt.timedelta(days=EDGE_LOOKBACK_DAYS)).strftime("%Y%m%d")
    out = {}
    for inst, files in per.items():
        seen: set = set()
        for f, last in files.items():
            if last < cut:
                continue                      # expired before the window
            seen.update(d for d in _tail_dates(f) if d >= cut)
        if seen:
            out[inst] = seen
    return out


def panel_edge(instruments: list[str] | None = None) -> dict:
    """Each instrument's newest bar date.  {instrument: 'YYYYMMDD'}.

    From the COMPLETE scan, not the quorum window: an instrument that stopped
    reporting has to keep appearing here or the stale check cannot see it.
    """
    return {i: max(g.values())
            for i, g in _contract_last_dates(instruments).items()}


_AS_OF: tuple[str, dict] | None = None


def panel_as_of(instruments: list[str] | None = None, *,
                refresh: bool = False) -> tuple[str, dict]:
    """The latest date that is plausibly a COMPLETE session, and the full edge.

    QUORUM, not min.  The panel ends on different dates for two unrelated
    reasons and only one of them should hold the pipeline back:

      COLLECTION LAG.  ASX closes earliest in the global day, so YAP4 and YXT4
      carry a date the other 61 do not have yet whenever the vendor's update
      lands between the two closes.  On 2026-08-25, 2 instruments of 63.  This
      is an artefact of WHEN the refresh ran -- a state no backtest ever sees,
      because a backtest reads a completed panel -- and it must not reach the
      signals.

      DIFFERENT TRADING CALENDARS.  ASX trades through US holidays and the CME
      does not.  2026-06-19 (Juneteenth) has 19 instruments of 63; 2026-01-19,
      02-16, 05-25 and 07-03 are the same shape.  This is not lag: those markets
      genuinely traded and the others were genuinely shut.  Holding everyone
      back here would defer real signals roughly six times a year for nothing.

    The two are cleanly separable and the data says so: a holiday leaves 16-50
    instruments trading, collection lag leaves 2, and nothing observed lands in
    between.  So a date counts as a real session once a QUORUM of the book has
    it, and as_of is the latest such date.  `min` -- the first version of this --
    could not tell them apart and suppressed both.

    Instruments that were SHUT on as_of legitimately sit behind it.  That is why
    last_date is reported per instrument rather than assumed uniform.

    CACHED for the life of the process: resolving this walks every contract
    file, and `rule_scores` asks for 63 worksheets in one run.  Pass refresh=True
    after writing bars -- the pipeline does that once, in stage 1.
    """
    global _AS_OF
    if _AS_OF is not None and not refresh and instruments is None:
        return _AS_OF

    edge = panel_edge(instruments)
    sess = panel_sessions(instruments)
    if not edge:
        return "", {}
    counts: dict[str, int] = {}
    for s in sess.values():
        for d in s:
            counts[d] = counts.get(d, 0) + 1
    need = max(QUORUM_MIN, int(round(len(edge) * QUORUM_FRAC)))
    quorate = [d for d, n in counts.items() if n >= need]
    # Fall back to the old min() rule if nothing clears quorum -- a book of one
    # instrument, or a window short enough that no date is shared.
    as_of = max(quorate) if quorate else min(edge.values())
    got = as_of, edge
    if instruments is None:
        _AS_OF = got
    return got


def report_edge(as_of: str, edge: dict, *, stale: int = STALE_SESSIONS) -> list:
    """Print the ragged edge and return anything too far behind to be ordinary.

    AHEAD of as_of means the vendor's update caught that market and not the
    others -- held back until the rest arrive.  BEHIND means either the market
    was shut on as_of, which is normal and silent, or the feed has stopped,
    which is not.  Only the second is worth waking someone for, and the only
    thing separating them is how far behind it has fallen.
    """
    ahead = sorted(i for i, d in edge.items() if d > as_of)
    behind = sorted((d, i) for i, d in edge.items() if d < as_of)
    print(f"  as_of {as_of}   {len(edge)} instruments")
    if ahead:
        print(f"    {len(ahead)} ahead, held back to as_of: "
              f"{', '.join(f'{i} ({edge[i]})' for i in ahead)}")
    lagging = []
    if behind:
        # Calendar days, not sessions: this needs no exchange calendar to be
        # right about the only thing it is asked, which is whether a feed has
        # stopped rather than merely closed for a day.
        a = _dt.date(int(as_of[:4]), int(as_of[4:6]), int(as_of[6:]))
        for d, i in behind:
            b = _dt.date(int(d[:4]), int(d[4:6]), int(d[6:]))
            gap = (a - b).days
            if gap > stale * 2:
                lagging.append((i, d, gap))
        closed = len(behind) - len(lagging)
        if closed:
            print(f"    {closed} behind as_of -- market shut that session")
        for i, d, gap in lagging:
            print(f"      [STALE] {i} last traded {d}, {gap} calendar days back")
    return lagging


def worksheet(inst: str, start: str, end: str,
              as_of: str | None = "auto") -> pl.DataFrame:
    """One instrument's worksheet.

    `as_of` squares the panel off at a common date -- see the AS-OF ALIGNMENT
    note above.  It is applied to the DATA, not to the emit window, so every
    stateful column (B_V_3, the ratchets, each hold series) accumulates exactly
    as it would have on that date.  Capping only the output would let state
    build from bars the run is meant not to know about.

    DEFAULTS TO "auto", which resolves to the newest date every instrument has.
    On by default because the alternative was opt-in, and an opt-in safety rule
    is one an unrelated caller forgets: a signal stage that omitted it would
    silently get the ragged edge back and nothing would say so.  Pass None to
    disable it deliberately.
    """
    which = gate(inst)
    date_col, till_col = GATE[which]
    gates = gate_map(inst, which)

    d = load(inst)
    if as_of == "auto":
        as_of, _ = panel_as_of()
    if as_of:
        cap = _dt.date(int(as_of[:4]), int(as_of[4:6]), int(as_of[6:]))             if as_of.isdigit() else _dt.date.fromisoformat(as_of)
        d = d.filter(pl.col("date") <= cap)
        if not d.height:
            raise SystemExit(f"[ABORT] {inst}: no bars at or before {as_of}")
    dead = dead_months(inst)
    # Dead months are dropped from the DATA, not merely from selection.  Keeping
    # them made the sheet unreadable where most of the listing is dead: EUA
    # trades only December, so a 9-session window carried 178 rows of which 9
    # mattered.  What was skipped is recorded in contract_cycles.csv, which is a
    # better place for it than 19 inert rows per session.
    if dead:
        d = d.filter(~pl.col("symbol").str.slice(-1).is_in(list(dead)))
    start_at = inception(d, dead)
    if start_at is not None:
        d = d.filter(pl.col("date") >= start_at)
    lo, hi = np.datetime64(start, "D"), np.datetime64(end, "D")

    # The streak has to accumulate from BEFORE the window, or B_V_3 reads false
    # on the first rows of every sheet when it should read true.
    # Sorted by (date, symbol) ONCE, not per session.  The loop used to call
    # g.sort("symbol") on every group -- 12,209 separate polars calls for ZC,
    # each paying full lazy-frame collect overhead to order ~20 rows, and 1.7s
    # of a 6.2s build.  group_by(maintain_order=True) preserves the frame's row
    # order inside each group, so pre-sorting gives the identical ordering.
    #
    # The order is LOAD-BEARING, not cosmetic: `max()` returns the first of any
    # tie, so symbol order decides Best_Vol and Best_Oi when two contracts share
    # a volume.  Changing it would silently change the book.
    warm = (d.filter(pl.col("date") <= pl.lit(hi).cast(pl.Date))
             .sort(["date", "symbol"]))
    streak: dict[str, int] = {}
    # Carried across sessions like `streak`: a ratchet that starts empty at the
    # window edge accepts a step it should refuse.
    prev_hold: str | None = None
    prev_forced: str | None = None
    cf_hold: str | None = None
    cf_pending: str | None = None
    cf_pending_n = 0
    rank = {sym: i for i, sym in enumerate(
        sorted(d.get_column("symbol").unique().to_list(), key=sort_key))}
    out = []

    for day, g in warm.group_by("date", maintain_order=True):
        day = day[0]
        recs = []
        for r in g.iter_rows(named=True):
            fn = gates.get(r["symbol"], "")
            till = (int((np.datetime64(fn, "D") - np.datetime64(day, "D")).astype(int))
                    if fn else None)
            passed = (till is not None and till <= 0)
            recs.append({**r, "gate_date": fn, "till": till,
                         "is_passed": passed})

        # Dead months are excluded from SELECTION but their rows stay in the
        # sheet -- a worksheet that hides what it ignored cannot be audited.
        # Both "best" columns take the window gate; they part company on what
        # counts as trading.  `is None or` is load-bearing: a contract with no
        # gate date has no expiry to be close to and stays eligible.
        def in_window(r):
            return r["till"] is None or r["till"] > BEST_VOL_MIN_CD

        # Best_Vol needs volume by definition: a session nothing traded in has
        # no volume leader, and saying so is correct rather than unhelpful.
        best_v_pool = [r for r in recs
                       if not r["is_passed"] and r["volume"] > 0 and in_window(r)]
        best_v = (max(best_v_pool, key=lambda r: r["volume"])["symbol"]
                  if best_v_pool else None)

        # Best_Oi does NOT: open interest is the field that SURVIVES a session
        # the vendor recorded no volume for, which is exactly when a fallback
        # needs it.  Gating it on volume made a missing field disqualify every
        # real contract -- CT 1998-01-16.
        best_o_pool = [r for r in recs
                       if not r["is_passed"] and r["open_interest"] > 0
                       and in_window(r)]
        best_o = (max(best_o_pool, key=lambda r: r["open_interest"])["symbol"]
                  if best_o_pool else None)

        # B_V_3: consecutive sessions as the volume leader.  Every contract that
        # is not the leader today has its count reset, so a single lost session
        # costs the full streak.
        for r in recs:
            sym = r["symbol"]
            streak[sym] = streak.get(sym, 0) + 1 if sym == best_v else 0
        best_3 = {s for s, n in streak.items() if n >= BV3_SESSIONS}

        # auto_roll: the symbol closest to its gate date; if that one is within
        # AUTO_ROLL_CD days of it, the next closest instead.  Nothing else.  It
        # does NOT read volume, open interest or is_passed -- an earlier version
        # took its candidates from a pool carrying a volume > 0 filter, and a
        # single quiet session then emptied the column (CGB on Christmas Eve
        # 1999, holding 29,027 open interest 63 days from notice).  is_passed
        # needs no test of its own either: a passed contract has till <= 0,
        # which the <= AUTO_ROLL_CD rule already skips.
        cal = sorted(recs, key=lambda r: rank[r["symbol"]])
        auto = None
        if cal:
            first = cal[0]
            if first["till"] is not None and first["till"] <= AUTO_ROLL_CD:
                auto = cal[1]["symbol"] if len(cal) > 1 else None
            else:
                auto = first["symbol"]

        # +1_auto_roll_hold: auto_roll's decision, taken one contract further
        # out.  Where auto_roll holds cal[0] this holds cal[1]; where auto_roll
        # steps to cal[1] this steps to cal[2].  Same ordering, same window,
        # same blindness to volume -- the front month is simply never held.
        #
        # For a compounded-in-arrears STIR that is the whole game: SO3 and SR3
        # settle to daily rates compounded over the contract's own reference
        # quarter, so by the time one is the front month its price is largely
        # already fixed and the volume has gone.  Holding the front means under
        # 1,000 lots on 27% (SO3) and 9% (SR3) of sessions against a book in the
        # hundreds of thousands; one step out that is 0% on both, with the same
        # 22 rolls, the same zero backward rolls and the same 44-day maturity
        # IQR.  SO3 median 4,716 -> 44,786 lots.
        #
        # NOT a new rule class -- it is auto_roll with the index shifted.  Only
        # position 2 and beyond would need the rule to step twice.
        plus1 = None
        if cal:
            i = 2 if (cal[0]["till"] is not None
                      and cal[0]["till"] <= AUTO_ROLL_CD) else 1
            plus1 = cal[i]["symbol"] if len(cal) > i else None

        # Hoisted out of the emit loop: Forced_roll_V gates on it, and the two
        # must not drift apart.  Both None must NOT compare equal -- no pick and
        # no leader is a failure of the rule, not agreement between absences.
        auto_best_v = auto is not None and best_v is not None and auto == best_v

        # Forced_roll_V: the alternative to auto_roll, offered only where
        # auto_roll is in doubt.  Nearest contract by gate date that is not
        # auto_roll's own pick, and clear of the window.  Without the "not equal
        # to auto_roll" exclusion the column was auto_roll verbatim on all 6,314
        # populated sessions across the notice group, 2005-2015.
        held = [r for r in recs
                if r["till"] is not None
                and r["till"] > FORCED_ROLL_MIN_CD
                and r["symbol"] != auto]
        forced = (min(held, key=lambda r: (r["till"], rank[r["symbol"]]))["symbol"]
                  if held and not auto_best_v else None)
        forced_best_v = (forced is not None and best_v is not None
                         and forced == best_v)

        # +2: one more step out, both earlier answers struck off -- the third
        # contract clear of the window.  It exists for the crop-year jump: the
        # first two columns reach the nearest and second-nearest and no further,
        # while grains roll old crop to new crop in one jump of two or three
        # months.  95% of the blanks Test_Hold carried before this column were
        # that jump.
        #
        # The > FORCED_ROLL_MIN_CD test is carried rather than assumed.  It
        # cannot bind -- every candidate surviving the two exclusions already
        # passed it for Forced_roll_V, verified over 1,421,027 candidate sets --
        # but a filter that states its own precondition does not quietly change
        # meaning when someone edits the line above it.
        held2 = [r for r in held
                 if r["symbol"] != forced and r["till"] > FORCED_ROLL_MIN_CD]
        forced2 = (min(held2, key=lambda r: (r["till"], rank[r["symbol"]]))["symbol"]
                   if held2 and not auto_best_v else None)
        forced2_best_v = (forced2 is not None and best_v is not None
                          and forced2 == best_v)

        # Today's distances, read by every ratchet below.
        till = {r["symbol"]: r["till"] for r in recs}

        # forced_roll_hold: four branches, ratcheted.  Best_Oi where neither
        # volume branch agrees, and Best_Vol where nothing eligible reports open
        # interest -- the newest session in any file carries volume and no OI,
        # because exchanges publish it the next morning.  Without the ratchet
        # this series reversed 1,605 times, 224 in LE alone.
        forced_hold = ratchet(
            (auto if auto_best_v
             else forced if forced_best_v
             else best_o if best_o is not None
             else best_v),
            prev_forced, till)
        if forced_hold is not None:
            prev_forced = forced_hold
        frh_best_v = (forced_hold is not None and best_v is not None
                      and forced_hold == best_v)

        # confirm_forced_roll_hold: the same answer, held back until it repeats.
        # The ratchet turns a one-session mistake into a permanent one -- RS
        # rolled to RS-2025N on 2025-03-10 on a single session where volume
        # touched N, open interest never moved, and the ratchet then refused to
        # go back because K is nearer: 21 sessions in the wrong contract off one
        # print.  So a move is confirmed before it is acted on.
        #
        # The wait is ABANDONED once the incumbent reaches the window or stops
        # being listed -- at that point there is nothing left to wait with, and
        # without the escape the delay pushed the hold INTO the notice window on
        # 6 RS sessions.  A one-day delay costs nothing mid-contract and
        # everything at the end of one.
        #
        # THE COST IS A LATE ROLL: every genuine move lands one session after
        # the signal, forever.
        #
        # One column, replacing RS_/LT_/CS_forced_roll_hold -- the same
        # algorithm under three names, each named for the market that wanted it
        # and computed everywhere regardless.
        cf_cand = ratchet(
            (auto if auto_best_v
             else forced if forced_best_v
             else best_o if best_o is not None
             else best_v),
            cf_hold, till)
        if cf_hold is None or cf_hold not in till:
            cf_forced_out = True
        else:
            v = till[cf_hold]
            cf_forced_out = v is not None and v <= FORCED_ROLL_MIN_CD
        if cf_cand is None or cf_cand == cf_hold:
            cf_pending, cf_pending_n = None, 0
        elif cf_cand == cf_pending:
            cf_pending_n += 1
        else:
            cf_pending, cf_pending_n = cf_cand, 1
        if cf_cand is not None and (cf_forced_out
                                    or cf_pending_n >= ROLL_CONFIRM_SESSIONS):
            cf_hold = cf_cand
            cf_pending, cf_pending_n = None, 0

        # Test_Hold: first branch that fires, ratcheted.  The third branch is NOT
        # gated on +2_Forced_Best_V -- it is a fallback, not a third opinion,
        # which is what ends the invariant the first two branches gave it.
        test_hold = ratchet(
            (auto if auto_best_v
             else forced if forced_best_v
             else forced2),
            prev_hold, till)
        if test_hold is not None:
            prev_hold = test_hold
        test_best_v = (test_hold is not None and best_v is not None
                       and test_hold == best_v)

        if np.datetime64(day, "D") < lo:
            continue                      # warm-up only, not emitted

        for r in recs:
            blank = r["is_passed"]
            out.append({
                "date": str(day), "symbol": r["symbol"],
                "open": r["open"], "close": r["close"],
                "volume": int(r["volume"]),
                "open_interest": int(r["open_interest"]),
                date_col: r["gate_date"],
                till_col: r["till"],
                "is_passed": "true" if r["is_passed"] else "false",
                "Best_Vol": "" if blank else str(r["symbol"] == best_v).lower(),
                "Best_Oi": "" if blank else str(r["symbol"] == best_o).lower(),
                "B_V_3": "" if blank else str(r["symbol"] in best_3).lower(),
                # NOT blanked on passed rows: auto_roll is a property of the
                # SESSION, not of the contract in this row, so the answer is the
                # same everywhere and holds even on the contract just rolled out
                # of.  The blanking convention belongs to the per-contract flags.
                "auto_roll": auto or "",
                "Auto_Best_V": str(auto_best_v).lower(),
                "Forced_roll_V": forced or "",
                "Forced_Best_V": ("" if forced is None
                                  else str(forced_best_v).lower()),
                "+2_Forced_Roll_V": forced2 or "",
                "+2_Forced_Best_V": ("" if forced2 is None
                                     else str(forced2_best_v).lower()),
                "auto_roll_hold": auto or "",
                "+1_auto_roll_hold": plus1 or "",
                "forced_roll_hold": forced_hold or "",
                "f_r_h_Best_V": str(frh_best_v).lower(),
                "confirm_forced_roll_hold": cf_hold or "",
                "Test_Hold": test_hold or "",
                "Test_Best_V": str(test_best_v).lower(),
            })

    return pl.DataFrame(out).sort(["date", "symbol"])


def means(df: pl.DataFrame) -> pl.DataFrame:
    """pl.mean() of every column that has one.  Booleans read as share true.

    Blank flags are null, NOT false, so they are excluded from the denominator
    rather than counted against: a passed contract is out of the running, not
    losing it.  date/symbol/gate date/the hold series are strings with no mean.

    Read row-wise here, which is what pl.mean() means, but the session-level
    columns are repeated on every row of a session -- so this weights each
    session by how many months were listed.  `session_means` gives the
    unweighted one; they are not the same statistic.
    """
    exprs = []
    for c in df.columns:
        if c in BOOL_COLS:
            exprs.append(pl.when(pl.col(c) == "true").then(1.0)
                           .when(pl.col(c) == "false").then(0.0)
                           .otherwise(None).mean().alias(c))
        elif df.schema[c] != pl.Utf8:
            exprs.append(pl.col(c).cast(pl.Float64).mean().alias(c))
    return df.select(exprs)


def session_means(df: pl.DataFrame) -> pl.DataFrame:
    """The session-level columns, one row per session instead of per contract."""
    one = df.unique(subset=["date"], keep="first").sort("date")
    return one.select([
        pl.when(pl.col("Auto_Best_V") == "true").then(1.0).otherwise(0.0)
          .mean().alias("Auto_Best_V"),
        pl.when(pl.col("f_r_h_Best_V") == "true").then(1.0).otherwise(0.0)
          .mean().alias("f_r_h_Best_V"),
        pl.col("date").n_unique().alias("sessions")])


def check(got: pl.DataFrame, ref_path: Path) -> int:
    """Compare against a reference worksheet, column by column."""
    ref = pl.read_csv(ref_path, infer_schema_length=0)
    got = got.with_columns([pl.col(c).cast(pl.Utf8) for c in got.columns])
    ref = ref.with_columns([pl.col(c).fill_null("").cast(pl.Utf8) for c in ref.columns])
    got = got.with_columns([pl.col(c).fill_null("") for c in got.columns])
    if got.height != ref.height:
        print(f"[FAIL] {got.height} rows against {ref.height}")
        return 1
    bad = 0
    for c in ref.columns:
        if c not in got.columns:
            print(f"[FAIL] missing column {c}")
            bad += 1
            continue
        # numeric columns: compare as numbers, not as formatted strings
        try:
            a = ref.get_column(c).cast(pl.Float64, strict=True)
            b = got.get_column(c).cast(pl.Float64, strict=True)
            n = int(((a - b).abs() > 1e-9).sum())
        except Exception:
            n = int((ref.get_column(c) != got.get_column(c)).sum())
        if n:
            print(f"[FAIL] {c}: {n} differing rows")
            bad += 1
    print("[OK] identical" if not bad else f"{bad} column(s) differ")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="ZC")
    ap.add_argument("--start", default="1989-05-08")
    ap.add_argument("--end", default="1989-05-23")
    ap.add_argument("--out", default=None)
    ap.add_argument("--means", action="store_true",
                    help="print pl.mean() of every column instead of the head")
    ap.add_argument("--check", default=None,
                    help="compare against a reference worksheet instead of writing")
    ap.add_argument("--as-of", default="auto", dest="as_of",
                    help="square the panel off at this date (YYYYMMDD or ISO). "
                         "Default 'auto': the newest date EVERY instrument has. "
                         "Pass 'none' to read the panel ragged.")
    ap.add_argument("--edge", action="store_true",
                    help="print each instrument's newest bar and stop")
    args = ap.parse_args()

    if args.edge:
        as_of, edge = panel_as_of()
        print("PANEL EDGE")
        report_edge(as_of, edge)
        return 0

    as_of = None if args.as_of == "none" else args.as_of
    if as_of == "auto":
        as_of, edge = panel_as_of()
        print("PANEL EDGE")
        report_edge(as_of, edge)
        print("")

    df = worksheet(args.instrument, args.start, args.end, as_of=as_of)
    if args.check:
        return check(df, Path(args.check))

    # One working file, overwritten each run.  A per-window filename accumulates
    # a directory of half-remembered snapshots; the window is an argument, not
    # an artefact.
    out = Path(args.out) if args.out else WORKING
    df.write_csv(out)
    with pl.Config(tbl_rows=30, tbl_cols=28, tbl_width_chars=320):
        print(means(df) if args.means else df.head(14))
        if args.means:
            print(session_means(df))
    print(f"\n{df.height} rows -> {out}")
    print(f"gate: {gate(args.instrument)}   "
          f"sessions {df.get_column('date').n_unique()}"
          + (f"   as_of {as_of}" if as_of else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
