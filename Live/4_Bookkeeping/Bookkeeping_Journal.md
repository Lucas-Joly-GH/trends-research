# Bookkeeping Journal

*Opened 2026-08-29.*

The durable record behind `bookkeeping.py` — stage 4, which turns positions into
orders. Same convention as the other journals: every claim names the number that
produced it.

---

## 2026-08-29 — Stage 4 exists because a position is a level and an order is a difference

Stage 3 says what to **hold**. This says what to **send**, and the two are not
the same statement.

```
Orders.csv      the full derived ledger, one row per (session, contract)
pending.csv     what to send for the next open       <- the actionable one
executed.csv    what filled at the last open         <- the reconciliation one
```

`python bookkeeping.py` takes ~1s. It reads only `3_Portfolio/Positions/` — no
vendor, no network, no state.

### Orders are keyed on (instrument, contract), and that is the whole design

`N_contracts` carries one number per instrument. A roll is a **close of one
delivery month and an open of the next**, so it appears in that one number as a
single small difference. Measured before writing a line of it:

```
position-sessions since 1990              448,235
of which the contract CHANGED              33,693   (7.52%)
where |dN| understates the true order      32,214   (95.6% of rolls)
```

A ledger built from `dN` would be wrong on 95.6% of rolls, quietly, and in the
direction of **reporting less trading than happens** — which is the direction
that makes a cost model look good. Hence one row per contract, and `kind` to say
why the row exists:

```
OPEN      flat -> a position                 1 row
CLOSE     a position -> flat                 1 row
RESIZE    same contract, different size      1 row
ROLL_OUT  close the expiring month           1 } always
ROLL_IN   open the new month                 1 } together
```

`kind` earns its place because reconciliation questions are almost always "why
did we trade this", and a roll and a signal change are entirely different
answers.

### Timing

Data arrives after the close, the stages run in the evening, orders go to the
next open. On the evening of session *t*:

```
pending    N[t] - N[t-1]      to be sent for the open of t+1
executed   N[t-1] - N[t-2]    filled at t's open, this morning
```

`decision_date` is the session whose data produced the order; `execute_at` is the
session at whose **open** it is meant to fill. Carrying both means a row
reconciles against either the model or the broker without anyone inferring a lag.

---

## 2026-08-29 — The bug: Positions are on the union grid, and I assumed otherwise

It did not work on the first run, and the defect was in the design rather than
the code.

`orders_for` compared each row against the row before it, on the assumption that
`Positions/<INST>.parquet` holds that instrument's own sessions. **It holds the
panel's union grid.** 6A has a row on Presidents' Day because some other market
traded; `symbol` is null there and the position is carried forward — which is
correct, a holiday does not flatten a book — but it means **the previous row is
not the previous session**.

Measured consequences:

```
(a) orders emitted on a session that market was shut        0
(b) rolls hidden behind a closure -> collapsed to a RESIZE  578
(c) execute_at naming a day that market is shut         5,105   (2.80%)
```

**(a) is the reassuring one**: `N_contracts` is always carried forward across a
no-bar row, so the ledger never invents a trade on a closed market.

**(b) is the module's own reason for existing, failing.** Reading `sym[k-1]`
reads the *gap's* null, `rolled` comes out false, and a two-legged roll collapses
into one small resize. 6A rolling 2001U → 2001Z across the 9/11 closure is one of
them. Of the 578: **366 surfaced as a wrong RESIZE, and 212 emitted nothing at
all** — the size happened to be unchanged across the gap, so a roll of thousands
of contracts was recorded as no order whatsoever.

The fix is one filter — iterate only the sessions where the instrument has a bar
— after which the original comparison is correct as written. Dropping the no-bar
rows is safe as well as necessary: the position never changes on one, and were it
ever to, the change would surface at the next real open, which is the only place
it could have been traded anyway.

```
                 before      after
ROLL_OUT          8,795      9,373
ROLL_IN           8,798      9,376
RESIZE          172,788    172,422
total           190,719    191,509
```

### The second half: the views had to move to `execute_at`

Fixing (c) surfaced a case the original design could not express. The two views
selected on `decision_date` — pending is decided today, executed was decided
yesterday. With per-market calendars that is wrong:

**An instrument shut today had its last order decided yesterday evening, for an
open that never came.** It did not fill. Selecting on the decision date reports
it as executed. Both views now key on `execute_at`:

```
executed   execute_at == asof             filled at this morning's open
pending    execute_at > asof, or null     not filled yet
```

bounded by `decision_date <= asof` so replaying a past session cannot leak orders
decided on information that did not exist yet. **4,969 orders (2.59%) straddle a
closure this way.** MLK Day 2026-01-19: 6B's order, decided 01-16, correctly
shows pending and not executed.

A null `execute_at` is the live case — decided on the last session in the panel,
filling at an open that has not happened. 24 of them today, one per instrument
that moved.

---

## 2026-08-29 — Verification is a replay, because nothing else would have caught it

`verify_bookkeeping` in `Update.py`, 21 checks, running on every pipeline.

**An order ledger cannot usefully be checked against itself.** Every arithmetic
identity inside it — `after - before == signed quantity`, flat sides, paired
legs — holds by construction, so a report made of those passes on a ledger that
describes entirely the wrong trades. The 578 collapsed rolls were well-formed
rows: right instrument, right date, plausible size.

So the central check replays. Apply all **191,509 orders in sequence from flat**
and the book must hold exactly `N_contracts`, in exactly one contract, on all
**496,053 instrument-sessions**. That is a statement that the ledger is a
*lossless encoding* of the position path rather than a plausible one, and it runs
in **0.63s**, which is why it can run nightly rather than on demand.

### Fault-injected, not observed passing

A check that has never fired is not evidence of anything. Five defects injected
into a copy, each confirmed to fail, then restored clean:

```
injected                                        caught by
roll collapsed into a resize (the shipped bug)  replay + flat-side
newest session's orders missing                 replay + both views
one order's action flipped                      signed-quantity + replay
one execute_at on the wrong session             execute_at
one fractional quantity                         whole-contract + signed + replay
```

The first is the one that matters. The replay names it explicitly — *held
ZT-2026U and ZT-2026Z simultaneously* — where every structural check passed.

`verify_stages` gained two lines for the failure the replay would catch but not
*name*: stage 3 rerunning without stage 4, which leaves a ledger that is
internally flawless and describes yesterday's book.

**That one fired in the wild within the hour**, unprompted — I reran
`portfolio.py` while gathering numbers for these journals and the next
cross-stage report said *ledger written BEFORE the positions it describes*. It
is the cheapest check in the file and it caught the most likely real-world
failure mode on its first day, which is roughly the opposite of how the
expensive checks earn their keep.

Found while writing this entry: `report()` counted roll events as `rows // 2`,
which is off by three because five rolls have a single leg. Now counted as
distinct `(session, instrument)` pairs. 9,377, not 9,374.

Current: **21/21**, and the pipeline is 17/17 books, 16/16 FX, 15/15 IRX, 23/23
portfolio, 21/21 bookkeeping, 7/7 cross-stage.

---

## 2026-08-29 — Does the order book match? Four questions, and two of them are free

"Are the orders given for execution always the orders executed" turns out to be
four separate questions, and the interesting part is that **the two obvious ones
are tautologies**.

*Executed today == pending last night* and *every order executes exactly once*
both follow directly from `execute_at` being the market's own next session,
which is asserted separately (191,485 dated, 0 wrong). They confirm the wiring
and nothing else. Reported as passing would be reporting a definition.

The two that carry content:

**The fill-timeline replay.** The existing replay applies orders when they are
*decided* and lands on `N_contracts`. Walking the same ledger on the *execution*
timeline must land on **yesterday's** position instead. That is a different
statement — the decision-side replay cannot see a fill at all — and it holds
across 496,053 market-sessions with 0 divergences.

**The trial balance.** Every contract ever traded must net to what is still held
in it: zero for the 9,380 that expired, the live position for the 63 that have
not. **9,443 contracts, 0 unbalanced.** A contract opened and never closed is the
one bookkeeping error a position-based replay can neither produce nor detect,
because the position it reconstructs would be right while the contract breakdown
underneath it is wrong.

Both are now in `verify_bookkeeping` (23 checks, 1.2s) — they fold into the walk
that was already happening, so they cost almost nothing.

### The half no snapshot can reach

A ledger that is re-derived nightly can be internally perfect on every snapshot
and still disagree with itself across time: announce an order last night,
re-derive it differently tonight, and the desk sent one thing while the book
records another. Every check above still passes.

Tested directly. `orders_for` rerun against Positions truncated at T — exactly
what stage 4 would have produced on the evening of T — diffed against what
today's ledger says about those sessions:

```
22 rebuild dates, 1992 to 2026      0 discrepancies of substance
492 execute_at values moved null -> dated
```

which is the only field permitted to move: at T that session had not happened.

That holds Positions fixed, so it leaves one link. Stage 3 is rerun nightly too,
and if `N[t]` changed when more data arrived, last night's order was announced
against a position path today's run no longer agrees with. Tested the same way —
the real `panel()` and `simulate()` against books truncated at T:

```
T             sessions   inst-sess   N differs   sym differs
1996-11-01       4,800     105,408           0             0
2011-02-14       8,510     267,785           0             0
2017-08-10      10,199     366,270           0             0
2024-05-08      11,953     471,802           0             0
2026-08-27      12,552     509,479           0             0
2026-08-28      12,553     509,542           0             0
```

**1.72M instrument-sessions, zero differences.** Stage 3 is causal, stage 4 is
stable given stage 3, so the chain closes: an order announced on any evening is
still, today, exactly the order the book says executed.

**What this does NOT prove** is stability under *revised* data, only under
*more* data. A vendor restatement of a past price re-derives `N[t]` and with it
every order after it. That is the append-only journal argument below, and no
amount of checking substitutes for it.

---

## 2026-08-29 — Pricing the ledger, and the 41% of commission stage 3 never bills

Every order row now carries `decision_close`, `commission_USD` and
`realised_pnl_USD`. Stage 4 reads stage 2's books (for the raw close and the FX
the cost converts at) and stage 3's Portfolio series, on top of Positions. It
still decides nothing, so it still needs no state, no vendor and no network.

**`decision_close` is not a fill price**, and was called `fill_price` until the
name was challenged. Nothing here is ever priced at an open: it is the close of
the *decision* session, which is what the backtest attributes the trade to under
its one-day-lag convention. The old name claimed a precision the model does not
have and hid the very gap `execute_at` exists to expose — the row says "fills at
the open of t+1" while the money says "priced at the close of t". Pairing the
name with `decision_date` puts that mismatch in the schema where it is legible,
rather than in a journal entry nobody reads at the moment they need it.

`price_close` was the suggestion and would have been an improvement too; it was
not taken because it still does not say *which* session's close, and which
session is exactly what the name has to settle.

**It is the RAW contract close, not the Panama close.** Two reasons
and both are sufficient: the raw close is the number a human can check against a
screen, and the Panama close is negative on 14 of these books — a short in CL
would print at -29.11.

**`realised_pnl_USD` is proportional crystallisation, not average-cost basis.**
Each contract carries a bucket of mark-to-market accumulated while held; a trade
that cuts the position by fraction f realises f × bucket, a trade that grows it
realises nothing, a close realises the rest. The identity

    sum(realised) + sum(open buckets) == sum(pnl_USD)

holds to **1.8e-13** per instrument. An average-cost basis on the *price* would
not reconcile, because stage 3 converts each session's P&L at that session's FX
and a single basis price cannot carry that. The check in `verify_bookkeeping`
deliberately reconciles against Positions rather than re-running this walk —
9,380 expired contracts closed out exactly, $261M unrealised across the 63 open
— because a check that re-derives the number the same way agrees with itself
whatever it does.

### The finding: commission is under-billed by 41.3%

Stage 3 charges `|N[t] - N[t-1]| . (cost_rt/2) . FX`. This ledger prices every
leg. **Off a roll they agree to the cent, across 172,760 sessions.** On a roll
they cannot, because a roll trades both months and `dN` sees only the difference:

```
stage 3 billed, whole history      $2.624 B
every leg priced                   $3.707 B
UNBILLED                           $1.083 B      +41.3%
```

The sharp part is the tail, not the average. **3,924 of 9,377 roll events
(41.8%) have |dN| == 0 exactly** — the new month is the same size as the old — so
stage 3 charges *nothing at all* for a full two-leg roll. Not an underestimate,
a zero. Carried through: cost drag 1.30% → **1.84% of NAV per year**, net Sharpe
1.108 → **~1.062**.

This is the same measurement that motivated the whole stage, arriving with a
price on it. The original note said `dN` understates traded quantity on 95.6% of
rolls; it does, by 143.5M contracts, and those contracts cost $1.08B.

**FIXED IN STAGE 3, later the same day.** `portfolio.py` now charges
`|N[t-1]| + |N[t]|` on a roll session and `|dN|` otherwise; the entry in
`Portfolio_Journal.md` has the arithmetic and the cost. Cost drag went 1.30% to
1.93%/yr, net Sharpe 1.108 to 1.054, and post-2010 crossed from just above the
paper's 0.991 to just below it.

The check that closed it is worth naming, because it is the reason to have built
stage 4 at all: **the two implementations share no code.** Stage 3 computes the
traded quantity inside the sizing loop from `N` and the symbol grid; stage 4
derives it from the order legs. They disagreed by $1.083B on rolls and agreed to
the cent off them, which located the bug precisely. Afterwards:

```
roll commission    ledger $0.999B    stage 3 $0.999B    $0M apart
```

`verify_bookkeeping` now ASSERTS that equality instead of reporting it, so the
correction cannot be lost to a later edit. The roll total fell $1.210B ->
$0.999B only because the book is poorer and therefore smaller.

*(A first pass at this put the gap at $491M / +18.7%. It was wrong: it computed
the shortfall as a ratio against `|dN|` and so silently skipped every session
where `|dN|` was zero — which is 41.8% of rolls and the worst 41.8%. Recorded
because the error has a moral: a relative correction cannot see a case where the
thing you are correcting is zero.)*

### Interest gets its own object

Interest is not a property of a transaction. It is earned on a balance, over a
gap between two sessions, whether or not anything traded — so on a quiet day it
would have nowhere to go on an order row, and on a busy one it would have to be
split across that day's orders by an invented rule.

`statement.csv`, one row per session, where the whole day reconciles:

```
opening_equity + gross_pnl - commission + interest == closing_equity
```

to **2.2e-16** relative over 12,552 sessions. The interest line names what the
question actually asked for — the balance, and where it came from:

```
CASH  2026-08-28
  opening equity         26,873,488,426
  gross P&L                 127,206,014
  commission                 -1,188,963
  interest                    2,771,218   on 26,872,299,464 carried from 2026-08-27
                                          at 3.764% annual over 1 calendar day = 0.00010313
  closing equity         27,002,276,695
```

`interest_base_USD` is recovered as `interest / rate`, which is exact and needs
no assumption about where equity stood mid-loop. `commission` is `cost_lag_USD`
— the cost that left the account today, not the cost of today's decision.

### Why the ledger starts in 1990

It does not; stage 3 does. `START_DATE = "1990-01-01"` in `portfolio.py` means
the first non-zero position anywhere is 1990-01-02, and `orders_for` skips every
session flat on both sides — no position, no order. The panel reaches back to
1978 and the books are built across all of it. **Stage 4 contains no date logic
whatsoever**; move `START_DATE` and the ledger follows.

---

## 2026-08-29 — Reconciliation: ten ties against primary sources, all clean

The 28 nightly checks test the artifacts against each other. This closes the
books a different way: **recompute each quantity from the primary sources** — the
trading books, `instrument_mapping.csv`, IRX — and require the derived files to
agree. Run after the commission fix.

```
  A  P&L recomputed from books x positions          20.0315B   20.0315B   0.0e+00
  B  instrument P&L sums to the portfolio           20.0315B   20.0315B   3.8e-16
  C  commission recomputed from the mapping          3.0560B    3.0560B   8.6e-15
  D  turnover: ledger legs vs stage 3's billed qty  433.052M   433.052M   0.0e+00
  E  ledger commission vs statement, per day         3.0554B    3.0554B   0.0e+00
  F  interest recomputed from IRX x balance          4.3333B    4.3333B   3.1e-15
  G  equity == NAV0 + cumulative flows              21.4093B   21.4093B   3.6e-16
  H  realised P&L == P&L of contracts now closed    20.4730B   20.4730B   3.7e-15
  I  ledger replay reproduces every position               0          0   0.0e+00
  J  notional recomputed from raw closes           445,312.7B 445,312.7B  0.0e+00
```

**D is the one that was worth waiting for.** Stage 3's billed quantity is backed
out of its own `cost_USD` by dividing by the unit cost and FX; the ledger's is
the sum of order legs. Before the commission fix these differed by 143.5M
contracts. They are now identical to the last contract — 433,051,927 either way.

E ties $3.0554B against C's $3.0560B, and the $0.6M is the expected boundary:
the statement carries `cost_lag`, so the final session's commission has not
landed yet. 0 of 12,552 days disagree.

### Two of the ten broke first, and both were my arithmetic

Recorded because the pattern is now three-for-three with the audit, and the
lesson is cheap to apply.

**F, interest, first read $4.442B against the file's $4.333B — a 2.45% break.**
I had accrued on `equity[t]` across the whole grid. Two errors: `equity[t]` is
recorded *before* that session's commission leaves, and stage 3 gates accrual on
`started_t`, so nothing earns before 1990. Ignoring the gate credits twelve years
of double-digit bill yields on an idle $100M, which is precisely the $109M my
check "found". The rate is ~1.2e-4 per day — a cost term could not have produced
a gap that size, and noticing that is what pointed at the gate.

**H, realised P&L, first read a $648M break.** I compared *all* realised P&L
against the P&L of *closed* contracts. A contract still open has already realised
P&L on every partial reduction, so the two sides covered different populations.
A defect in the identity, not the ledger.

Both times the code was right. The rule that keeps paying: **re-derive from the
object the code used, and make both sides of an identity range over the same
set.**

### It now lives in `Reconciliation_check/`, and it caught a third one on the way

Moved out of scratch into `4_Bookkeeping/Reconciliation_check/reconcile.py`,
`--quiet` for the short form, non-zero exit on a break. **3 seconds**, not the
ninety I guessed — the books are parquet and the walk is trivial. So the reason
it sits outside `Update.py` is that it answers a different question, not that it
is expensive; wiring it in would be cheap.

Rewriting it for the move broke H again, by 8.5% / $1.74B — and the cause was
**the union grid, for the third time in this pipeline.** Tidying the code moved
the P&L attribution from a per-market walk into the main loop over the full
grid, where the session *after* a holiday has `symbol` null on the previous ROW
while the previous SESSION names a contract perfectly well. Every such session's
P&L was dropped.

Same defect that hid 578 rolls in stage 4 and that stage 3's roll detection has
to forward-fill around. Two prior encounters, both documented in this file, and
neither stopped the third — which says the mitigation cannot be memory. Anything
that walks a Positions or book frame per instrument has to filter to
`symbol.is_not_null()` first, and the tie is what caught it.

### Wired into `Update.py` as the eighth and last report

`verify_reconciliation` calls `reconcile.ties()` and renders the rows in the
pipeline's own format. **The arithmetic is not reimplemented** — two copies of a
reconciliation is two reconciliations, and the second one is always the stale
one. `reconcile.py` stays runnable on its own, which is the point: a
reconciliation that only ever runs unattended stops being read.

It runs last because it is the broadest claim in the file, and it is the only
report that leaves the artifacts. Every other check, `verify_stages` included,
compares derived files with each other — so **a consistent misreading of the
panel passes all of them**, because every stage inherited the same misreading.
This one goes back to the books, the mapping and IRX. `--no-reconcile` skips it.

Fault-injected rather than observed passing:

```
injected                       broke
one interest value, +$1M       F, G                        ($1M in $4.3B, rel 2.3e-4)
one ROLL_OUT leg deleted       C, D, E, H, I               (10,516 contracts)
```

Pipeline now reads **17/17 · 16/16 · 15/15 · 23/23 · 28/28 · 7/7 · 10/10**.

### What this does and does not establish

It establishes that every derived number in stages 3 and 4 is reproducible from
the panel and the metadata, and that the four artifacts, the statement and the
ledger all describe one consistent set of books.

It establishes nothing about whether the *conventions* are right. Close-to-close
attribution against an `execute_at` at the next open is still an unresolved
inconsistency; slippage and market impact are still unmodelled, so `cost_rt/2`
remains a floor on execution cost rather than an estimate of it; and the vendor
panel is taken as given throughout. A book can reconcile perfectly and still be
measuring the wrong thing.

---

## 2026-08-29 — The close-vs-open gap is closed, and it moved this stage too

Stage 3 now prices on OPEN EXECUTION -- `Portfolio_Journal.md` has the
arithmetic and the five corrections it took. This entry records what it changed
HERE, because the gap it closes is the one this file has been flagging since it
was written: the ledger said "fills at the open of t+1" while the P&L said
"close of t". They now say the same thing, which is the precondition for
comparing a real fill against either.

### The realised-P&L walk moved onto the fill timeline

`orders_for` used to realise a contract's bucket when the trade was DECIDED. Under
open execution the position changes when it FILLS, one session later -- so a
ROLL_OUT decided at k leaves the expiring month held overnight and exited at the
open of k+1, still earning the gap leg of session k+1. Realising at k closed the
contract before its last P&L arrived, and the per-contract attribution never
tied.

Orders are now parked when decided and settled on the next session, after that
session's two legs have been credited. Off a roll nothing changes; on a roll it
is the difference between a tie that closes and one that does not.

### A session's P&L can belong to two contracts

Stage 3 publishes `pnl_gap_USD` and `pnl_day_USD`, and this stage credits them
separately: the gap to the month held at k-2, the day to the month held at k-1.
On a roll those differ. Off one they coincide and it reduces to the old single
credit.

**One definition, three consumers.** The split is computed once in stage 3 rather
than re-derived here, in `verify_bookkeeping` and in `reconcile.py` -- three
re-derivations of the same subtle lag is three chances to get it wrong, and this
session already spent four corrections proving that.

### `keep` is now a contract, not a convenience

`Positions` retains every session the market traded. It used to drop rows with a
null signal and a flat position -- harmless when nothing read a lag off the file,
fatal now that three objects do. **A lag can only be read off a file whose
consecutive bar rows are consecutive sessions**, and a dropped bar row silently
turns `k-2` into "two rows back", which is the union-grid trap one level down.

Ledger after the change: 4,305 orders over the 2026 window, all 28 checks and all
10 reconciliation ties clean.

---

## 2026-08-29 — The append-only journal, built and tested at both scales

`Journal/journal.py`, wired into `Update.py` as **stage 4b, non-blocking**.  The
entry below it has argued since it was written that the derived ledger cannot
hold a fact; this is the store that can.

### Three stores, because an immutable row can only hold what was known

```
orders/YYYY/DATE.parquet        the decision + its provenance, write-once
outstanding/YYYY/DATE.parquet   given earlier, still unfilled tonight, and why
fills/YYYY/DATE.parquet         what actually happened, written later
```

On the evening of *t* you know the order and every input that produced it. You
do **not** know the fill price — the open of *t+1* has not happened — nor, at the
live edge, the fill date. Forcing either into the order row forces a mutation,
which defeats the point. So they live in `fills/`, written the next morning.

The order row carries the nine inputs to 3.32 (`SIGNAL`, `price_vol_USD_ann`,
`s_g_vol`, `s_g_dd`, `w_i`, `IDM`, `NAV`, `N_raw`, `N_target`) plus
`code_commit`, `panel_edge` and `cycles_fingerprint`. That triple is what makes
a past order explainable after the code has moved on.

### Why provenance and not reproduction

On 2026-08-29, unchanged panel and unchanged window, the derived ledger went
**4,322 → 4,305 orders** across three fixes in one day. The 3.36 buffer turns any
change into a cascade — `N[t]` depends on `N[t-1]` all the way back — so a
rounding fix on 86 bars in the 1990s reached 2026. Send BUY 18 on Monday, fix a
bug on Tuesday, and the ledger says Monday was BUY 22 while the broker says 18.

Re-running the old code does not save you: the code is in git, the vendor panel
is not, and stage 1 refreshes in place. **Reproduction is not a substitute for a
record.**

### `execute_at`, and the calendar we did not add

Exact for every past session — an instrument's own bar history IS its trading
calendar, unscheduled closures included. NULL at the live edge, meaning "the next
session this market opens, whenever that is"; the fill record resolves it the
next morning.

`exchange_calendars` was checked and rejected: it needs pandas (excluded here),
has no calendar for **ICE Futures Europe, ICE Canada or Montreal — 10 of our 63
instruments** — and its holidays are user-contributed with no guarantee back to
1978. It would be a guess where the data gives a fact.

### Given once, carried visibly

`pending` deliberately carries an unfilled order forward, so a market shut four
days shows the same order on four evenings. Right for the view, fatal for a
record: **93 of 4,305 orders** in the 2026 window, and 4,950 of 191,445 over the
full history, would have been written more than once under the same `order_id`.

An order is now recorded **once** in `orders/` when given, and appears in
`outstanding/` each evening it remains unfilled, with a reason. Two counts, not
one: `own_sessions_since` is how many sessions THAT MARKET has opened since the
order was given — 0 is what justifies `MARKET_CLOSED` — and `carried_sessions`
is how long it has been sitting. The first version conflated them under one
misleading name.

**The reason is recorded, not inferred.** "The market was shut" can be recovered
from the bars afterwards; "the broker rejected it" or "we chose not to send"
cannot. Those are facts about one evening and nowhere else.

### Tested at both scales, and the small one was not enough

```
                        2026 window        full history
sessions                        170               9,494
orders journalled             4,305             191,445   = the ledger, exactly
fills (modelled)              4,284             191,421
carries                          95               5,265   all MARKET_CLOSED
files / checksums               353              19,700   VERIFIED
drift                             0                   0
backfill                        11s              30 min
```

**The longest carries in the whole history are 2001-09-10** — CC and DX orders
given the evening before the exchanges closed for four sessions after 9/11.
Recorded as given once, carried four evenings with the reason, filled on
reopening. No calendar library would have known that; the bars did.

**Two bugs the 170-session window passed and the 9,494-session one caught:**

- **Double-journalling across a multi-day closure.** August 2026 contains no
  multi-day closure, so the first 20-session replay was clean. Lunar New Year
  found it.
- **A quadratic drift report.** A 191,445-element set built INSIDE a generator
  expression, rebuilt once per candidate. Invisible at 4,305 orders, a ten-minute
  timeout at 191,445. Now 14s.

### Guards

`MANIFEST.json` carries `mode`. A `test` store is wiped by `--reset`; a `live`
store refuses and names the directory to move by hand. `--fills` refuses to write
modelled fills into a live store at all — they assume execution at exactly the
next open, the paper's documented limitation, and must never be mistaken for a
broker's report. Both verified.

`--backfill` exists because the nightly path is O(N) per session and quadratic
over 9,494 of them. It holds the caches in memory and calls the same builders,
and was **proven byte-identical to the one-at-a-time path over 116 sessions** —
a backfill on a different code path would be testing different code.

### Precedence, written down

```
what did we SEND?      the journal.  always.
what should we HOLD?   the derived ledger.
do we hold it?         neither -- reconcile against the broker.
```

---

## 2026-08-29 — This is a derived ledger, not a journal, and that matters before going live

Recorded prominently because the file is journal-*shaped* and will be mistaken
for one.

`Orders.csv` is **recomputed from Positions on every run**. That makes it
idempotent and reproducible, which is what research wants: the same panel gives
the same orders, always.

**It is therefore not a record of what was sent.** A vendor revision to history, a
changed roll rule, a different tau — any of these rewrites the whole ledger
retroactively, including rows describing orders that were genuinely transmitted
last week. The 3.36 buffer makes this sharper than it sounds: `N[t]` depends on
`N[t-1]` all the way back, so a single upstream change re-derives **every order
after it**, not just the affected day.

Live trading needs an append-only journal, written once and never recomputed. The
schema here is deliberately close to one — add `sent_at`, `fill_price` and
`fill_qty` and it becomes one — but building it now would add state before there
is anything to protect.

---

## 2026-08-29 — Tie F was wrong twice, and the second time it passed

F recomputes interest from IRX x balance against `Portfolio.interest_USD`. It has
now been wrong in both directions, and the two failures are not the same kind of
thing.

**First**, it accrued across the whole grid and ignored stage 3's `started_t`
gate, crediting twelve pre-1990 years of double-digit bill yields on an idle
$100M and calling the $109M result a break. That one **failed loudly**, which is
what a check is for.

**Second**, and worse: it read `(eq[k] - cost[k])`, deducting the session's
commission from the base exactly as stage 3 did. Both sides held the same false
idea — that cash which has paid for a trade is not there to earn, applied to a
cost that does not leave until the next open — so F **agreed to 1e-6 while both
were wrong**. It only surfaced because the base was questioned from outside, on
the published page, where 2026-01-05 showed a book that had never traded earning
interest on 99,917,719.09 of its 100,000,000.

```
                       before        after
F, relative error      1e-6        1.8e-16
```

The lesson is the one already in this file's header, now demonstrated rather than
asserted: **the failure mode of an auditor is writing a false identity
confidently**, and a tie between two sides that share an assumption tests
neither. F is the reason to recompute from primary sources — IRX and the equity
column — rather than from the other stage's intermediate. Where it took the
intermediate, it took the error with it.

Corrected to `eq[k]` alone, with the history written beside it. 11/11.

---

## 2026-08-30 — The journal was reset and re-baselined, and why that was allowed

Stage 4b started refusing to write:

```
[CONFLICT] 2026-08-28 is already journalled and the rebuild disagrees.
    stored 21 orders, rebuilt 22
    NOTHING WRITTEN.  The stored file is the record; the rebuild is today's
    opinion of it.
```

Correct behaviour, and not a bug. The cost audit changed `cost_rt` on 17
contracts, which changed NAV, which changed position sizes, which changed the
orders. `--drift` put a number on it:

```
journalled 4,305 orders over 170 sessions
  unchanged     4,104
  changed         183
  gone             18
  new              24     about 5% of the book
```

### The deciding fact was `mode test`, not the size of the drift

A LIVE journal must never be reset. Its whole value is that it records what was
actually sent, on the day, under whatever the model was then; a live store that
can be re-baselined after a model change records nothing at all, because it
would always agree with the current model by construction. `reset()` enforces
this in code -- it refuses on a live store and tells you to move the directory
by hand.

This store was `mode test`, created 2026-08-29 by BACKFILLING the whole of 2026
in one pass. It never captured a decision as it was made; it is a snapshot of a
model that has since been corrected. Keeping it would mean every future run
reporting a conflict about orders nobody ever placed, and that noise is what
hides the first conflict that matters.

So: reset, backfill, resolve fills. The store was copied out first, which cost
nothing and is the only reason this was a reversible decision.

```
before   4,305 orders, 170 sessions, created 08-29, drift 201 rows
after    4,311 orders, 170 sessions, created 08-30, drift 0
         4,289 fills, 96 carry-forward rows, 353/353 checksums VERIFIED
```

**The rule this establishes, for when it goes live:** a test store may be
re-baselined after a model change; a live store may not, and drift against it is
evidence rather than error. The moment `mode` is flipped to `live` is the moment
this entry stops applying.

---

## 2026-08-30 — A conflict is about the decision, not the commit it was rebuilt at

Stage 4b failed on the first real end-to-end run. The report said the store and
the rebuild disagreed on all 22 orders of 2026-08-28 — while `--drift` said
zero had changed, and a column-by-column comparison of the same 22 rows found
no field that differed. Two checks over one pair of files, disagreeing.

Exactly one column separated them:

```
                    stored          rebuilt
code_commit         a588aa5         81133b1     <- the whole disagreement
panel_edge          2026-08-28      2026-08-28
cycles_fingerprint  9dccdb3c1e85    9dccdb3c1e85
```

The commit had moved because we had committed. Nothing about the orders had.

### Why that made the warning worthless

The conflict test dropped `written_at` and compared everything else, so
`code_commit`, `panel_edge` and `cycles_fingerprint` were being read as part of
the decision. Since the commit changes every time anyone commits, **every rerun
of an already-journalled session after any commit was a conflict** — and we
commit constantly. A warning that always fires is one nobody reads, which costs
more than the warning was ever worth. That is the same argument as leaving CI
red, and it applies here for the same reason.

### The doctrine did not change; what counts as disagreement did

Never rewrite, the store is the record, a rebuild that disagrees is information
rather than a correction — all unchanged. What changed is that `CONTEXT`
(`code_commit`, `panel_edge`, `cycles_fingerprint`, `written_at`) is separated
from `DECISION` (the order fields plus the model inputs that produced them).

The reasoning is that a row written at one commit **IS** the record of what was
sent. That is exactly why it must not be overwritten when the same decision is
rebuilt at another commit — and equally why that is not a disagreement. Context
is evidence about the act; the decision is the claim. Only the claim can
conflict.

Context drift is now reported and never fatal:

```
2026-08-28: already journalled, decision identical (22 orders) -- no write
    context moved: code_commit a588aa5 -> 81133b1
```

### Tested in both directions

A check that cannot fail is not a check, and this one had just been made
*harder* to trip:

```
a quantity changed          -> CONFLICT, exit 1, "common but changed: 1"
a model input changed       -> CONFLICT, exit 1  (SIGNAL is in DECISION)
an order removed from store -> CONFLICT, exit 1, "only in rebuild: 1"
only panel_edge moved       -> no-op,    exit 0, reported as context
```

The second one matters most: `PROV` — SIGNAL, the gates, IDM, NAV, N_raw,
N_target — stays inside `DECISION`. Those are the inputs that produced the
order, so a row that reached the same quantity by different arithmetic is still
a disagreement. Only the environment moved out.

Store restored afterwards, 353/353 checksums VERIFIED.

---

## 2026-08-30 — Three modes, because two conflated two things

`mode` was `test` or `live`, and it governed two independent permissions:
whether the store can be wiped, and whether modelled fills may be written.

A paper-trading track needs one and not the other. It must NOT be wipeable — an
erasable forward record records nothing, because it can always be made to agree
with today's model — but it has no broker, so its fills are modelled and always
will be. Forced to choose, it stayed `test`, which meant **the thing most worth
protecting was the thing left deletable.** Three resets today, each legitimate
under that mode, each one proof of the exposure.

```
mode     wipeable   modelled fills
test        yes          yes        development; re-baselined freely
paper       NO           yes        a forward record, filled by model
live        NO           NO         a forward record, filled by a broker
```

`--promote` moves forward only. Each step removes a permission, and a command
that could hand one back could quietly undo protection somebody thought they had
turned on. Verified: `test -> live` succeeded, `live -> paper` and `live -> test`
were both refused, and `--reset` then refused as well.

### Promotion is gated on the store being backed up

The store is gitignored and 356 files live on one disk. That is fine while it is
`test`, because it is disposable. **Promotion is the exact moment it stops being
fine**, so the check lives there rather than in a note somebody reads later: a
promotion out of `test` refuses while `git check-ignore` matches the store, and
says what to do. `--status` now names the exposure on every run:

```
backup  NOT tracked by git -- one disk only   (fine while mode=test)
```

An unrewritable record that exists nowhere else cannot be rewritten OR
recovered, which is the failure the store exists to prevent arriving by a
different route.

*Tested carelessly on the real store rather than a copy, which promoted it to
`live`; the manifest was restored by hand — the escape the design names — and
`--verify` confirmed 353/353 checksums intact afterwards.*

---

## Conventions and decisions

- **`execute_at` is the next session in that instrument's OWN calendar.** "The
  next open" is not one moment across a book spanning Hong Kong to Chicago, and
  a single panel-wide next-session would put a fill time on 2.59% of rows that
  no exchange agrees with.
- **A roll emits two rows even when the two sizes are similar.** Selling 1,234
  of the expiring month and buying 1,240 of the next is 2,474 contracts through
  the market, not the 6 that `dN` reports.
- **Single-leg rolls are legitimate only when one side is flat.** 5 of 9,377.
  Asserted, not assumed.
- **Quantities are positive; direction lives in `action`.** A signed quantity
  and a BUY/SELL column encode the same thing twice, and the two disagree the
  moment anyone edits one.
- **Both csv and parquet are written, and both are checked.** The dual-write is
  worth nothing if only one half is ever read back.
- **A conflict is about the DECISION, never the context.** `code_commit`,
  `panel_edge` and `cycles_fingerprint` record the act, not the claim;
  they are reported when they move and never block a write. The model
  inputs stay inside the decision — the same quantity reached by different
  arithmetic is still a disagreement.
- **A test journal may be re-baselined; a live one may not.** Drift after a
  model change is expected, and against a LIVE store it is evidence, not
  error. `reset()` refuses on a live store in code rather than by
  convention. See the 2026-08-30 entry.
- **Nothing here re-decides anything.** The position series arrives with
  truncation and the buffer already applied, so a difference between two
  sessions *is* an order. Stage 4 does no sizing. It measured stage 3's
  commission error; it did not fix it, and should not have — the fix belongs
  where the model is.
- **`decision_close` is the raw close, `realised_pnl_USD` is proportional
  crystallisation, and interest is not on an order row at all.**
- **The statement's `interest_base_USD` is last night's CLOSING equity**, not
  last night's equity less today's commission. That cash leaves at today's open,
  at the far end of the window the credit accrued over. The base is recovered by
  inverting `interest / rate` rather than reconstructed from equity, so it states
  what stage 3 actually used and cannot drift from it.

## Not yet built

- **Order aggregation and working rules.** Every row is a market order for the
  full difference. Real execution would split the 713,049-contract maximum.
- **A pre-trade check against open interest.** The capacity work in
  `Portfolio_Journal.md` says the book breaches 100% of open interest under
  compounding from 1983; nothing in stage 4 refuses to emit such an order.
