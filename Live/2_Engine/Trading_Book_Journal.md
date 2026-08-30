# Trading Book Journal

*Opened 2026-08-28.*

The durable record behind `trading_book.py`: which equation each column
implements, where the paper and the code disagree, and the measurement that
settled each choice.

Convention, inherited from `1_Roll/Front_Contract/Roll_Journal.md`: every claim
names the instrument, session or figure that produced it. A finding with no case
attached is a guess someone wrote down confidently.

---

## 2026-08-28 — `price_vol_USD_ann`: equation 3.35, completed

```
price_vol_USD_ann = price_vol_curr_ann x FX_rate
```

`price_vol_curr_ann` is 3.35 with the FX leg deliberately left off —
`sigma_hat x pointsize x sqrt(256)`, in whatever currency the contract quotes
in. Useful on its own, but **not comparable across markets**. This multiplies
the last unit change through so one contract's annual risk is on one scale for
the whole book.

The last session shows why it is not cosmetic:

```
inst    ccy        curr_ann    FX_rate       USD_ann
ZC      USD           5,799   1.000000         5,799
CL      USD          35,987   1.000000        35,987
NIY     JPY       7,283,654   0.006271        45,678
SJB     JPY         404,142   0.006271         2,535
HSI     HKD         229,553   0.128205        29,430
FDAX9   EUR          70,213   1.164880        81,790
```

Read the middle column as dollars and NIY looks about a thousand times riskier
than CL, on the column a position sizer divides by.

**Computed in `main()`, not in `book_one` with the rest of the vol chain**, because
`FX_rate` does not exist until the rates are built — and those are built on the
session union of every book, which is only known once the workers have finished.

**Null where the rate is null**, which is the honest answer: YAP4 and YXT4 open
before the AUD future existed. Polars propagates the null through the product
rather than substituting 1.0, which would silently report an AUD figure as
dollars. Verified: null counts follow `max(curr_ann nulls, FX_rate nulls)` in
every book, and all 45 USD books have the column exactly equal to
`price_vol_curr_ann`.

`verify_books` now asserts the identity (17 checks, was 16). Fault-injected both
realistic ways it breaks — the FX leg dropped, and conversion with the wrong
currency's rate — and the check caught both.

---

## 2026-08-28 — The same 63 worksheets were built twice per run. 176s saved.

`contract_cycles.rule_scores` called `fc.worksheet()` directly — the *uncached*
entry point — rebuilding all 63 worksheets in ~186s and throwing them away.
`verify_holds` then rebuilt the same 63 minutes later, and stage 2 read them a
third time. Full pipeline **467s → 291s**, all four verification reports green.

```
STAGE 2b  roll scores      0 hit, 63 built  (186s)   <- paid once
verify_holds              63 hit,  0 built   (1s)    <- was 188s
stage 2                   63 hit,  0 rebuilt
```

### Caching it was not enough on its own, and that is the interesting part

`_fingerprint` hashed **the whole of contract_cycles.csv**, and `contract_cycles`
writes that file *twice*: once before scoring, then again afterwards with
`Mean_Auto_Best_V`, `Mean_Forced_Best_V`, `Roll_Rule` and `Unique_Roll` filled
in. So every worksheet cached during the scoring pass was keyed on the first
version and invalidated by the second. Caching alone would have changed nothing.

**`worksheet()` reads exactly two fields out of that file**, both filtered to one
instrument: `has_notice` via `gate()`, and `Dead_contracts` via `dead_months()`.
Nothing else reaches it — `Roll_Rule` included, because the worksheet computes
*every* rule's columns and the rule only decides which of them a book later
reads. The fingerprint now hashes those two values **through front_contract's own
accessors**, so it cannot drift from what the worksheet actually depends on.

Narrowing a cache key is exactly the change that serves stale data silently, so
it was tested in both directions:

```
PASS  ZC Mean_Auto_Best_V             -> key SURVIVES
PASS  ZC Mean_Forced_Best_V           -> key SURVIVES
PASS  ZC Roll_Rule                    -> key SURVIVES
PASS  ZC Unique_Roll                  -> key SURVIVES
PASS  ZC last_date                    -> key SURVIVES
PASS  ZC has_notice                   -> key invalidates
PASS  ZC Dead_contracts               -> key invalidates
PASS  CL Dead_contracts (other inst)  -> key SURVIVES
```

**If a new read of contract_cycles.csv is ever added to the worksheet path, it
must be added to the fingerprint too.** That is the one way this key can go
stale, and a stale worksheet is silent.

Stage 1 loads `trading_book` for the cache. The direction looks wrong and is
deliberate: it is a dependency on the *shared cache*, not on stage 2's work,
there is no import cycle, and `Update.py` already loads it the same way. The
alternative was a third copy of the fingerprinting logic, which is how two caches
end up disagreeing about what is current.

### Found while fixing it: the intermediate write can publish a truncated table

`contract_cycles.csv` was found on disk with **7 columns instead of 12** — no
`Roll_Rule`, no scores, no `last_date`. That is the intermediate write, published
and then never completed, and it is worse than not writing at all:
`trading_book.rules()` reads `Roll_Rule` out of that file, so stage 2 has no
rules to build against — yet the file parses cleanly and every row is present, so
nothing looks damaged.

The write cannot simply be moved later: the worksheets built during scoring read
`Dead_contracts` out of this very file, so it has to be current before scoring
starts. Instead the previous contents are snapshotted and restored if
`rule_scores` raises. **A stale complete table beats a fresh truncated one** —
the panel refresh is idempotent and the next run rewrites it properly.

Fault-injected to confirm: 12 columns → 7 after the intermediate write → back to
12, byte-identical, with the exception still propagating.

---

## 2026-08-28 — The pipeline looked hung after `PANEL EDGE`. Four causes, in three files.

Reported as: the console stops dead after the `PANEL EDGE` banner and nothing
happens. It was not hung — it was working in complete silence, which is
indistinguishable from the outside.

**`PANEL EDGE` is printed by three different places**, which is why the first
fix landed on the wrong one:

```
1_Roll/contract_cycles.py:1301          stage 1
1_Roll/Front_Contract/front_contract.py standalone runs
2_Engine/trading_book.py:1991           stage 2
```

`panel_as_of()` does not print it — its *callers* do. Timed line-by-line over a
full piped run, the banner the user was watching was **stage 1's**, and the
silence after it was 175 seconds long. The two worst offenders were not in this
file at all:

| where | silent for | cause |
|---|---|---|
| `contract_cycles.rule_scores` | ~175 s | rebuilds all 63 worksheets, **uncached**, printing only on warnings |
| `Update.verify_holds` | 188 s cold | progress bar guarded by `if tty`, so nothing off a terminal |
| `contract_cycles.build` | ~vendor-bound | 63 vendor round-trips, silent unless warning |
| `trading_book` worker loop | 75 s | below |

All four now print. The stage-1 loops emit one flushed line per instrument; the
`verify_holds` loop switches to a time-based line off a tty (5 s floor — a fixed
every-Nth was wrong at both ends, since that loop is 188 s cold and 2 s warm) and
**announces the instrument before working on it**, because the build is one
blocking call and CL alone takes ~13 s, so a message printed on completion leaves
the gap unattributed.

`_tick`'s off-tty interval also dropped from 30 s to 10 s: measured over a full
467-second run, its two 30-second NDU-wait gaps were the longest silences left in
the entire pipeline.

### The stage-2 half of it

**Cause 1: `results = list(ex.map(_one, tasks))`.** That blocks until all 63
instruments finish and only then prints 63 rows at once. So nothing reached the
console for the whole build, and — worse — `Update.py`'s progress bar counts
exactly those rows, so the bar had nothing to count until the stage it measures
was already over. Fixed by draining the iterator as results arrive.

`.map` yields in *submission* order, which keeps the table alphabetical. That
costs nothing at `jobs=2`: at most one instrument can finish ahead of its turn.
`as_completed` would stream marginally sooner and scramble the ordering, which
is a bad trade for a table someone reads.

**Cause 2: the child was block-buffering.** Python buffers stdout in ~8 KB
chunks whenever it is not writing to a terminal, and a stage's stdout here is
*never* a terminal — the tty path hands it a pipe so the bar can read it, the
redirected path hands it PyCharm's console. So even after cause 1, output would
have arrived in bursts. `run()` now inserts `-u` itself rather than trusting call
sites, so a stage added later cannot reintroduce it.

### Measured, cold, with output piped (the PyCharm case)

```
                          progress lines   longest silence   median gap
before                                 0             75 s            —
streaming only                        67             11.1 s        0.3 s
streaming + heartbeat                 70              5.1 s        0.3 s
```

The 11.1s residue was in-order yielding waiting on whichever instrument is
slowest — CL is 623,358 rows — while its neighbours sat finished behind it. A
heartbeat closes it: `wait([f], timeout=5)` blocks on the *next* result
specifically rather than on any result, so ticking costs no spinning and the
ordering survives. It fires exactly where expected:

```
27.9    ... building, 12 of 63 done, 20s elapsed
31.3  CL      forced_roll  ...
```

**The tick line must not look like a result row.** `Update.py` drives its bar off
`^\S+\s+\S*roll\S*\s`, so anything starting with whitespace passes through
without inflating the count. Verified against the real regex: the three row
shapes all match, and the tick, banner, header and FX lines all correctly do not.

Also filled in the `sec` column, which had been in the header since the file was
written and never populated, and added a `done  k/63` counter. `sec` is elapsed
for the *stage*, not per instrument — the workers overlap, so a per-instrument
time would not sum to the total and would invite exactly that arithmetic.

---

## 2026-08-28 — `FX_rate`: the instrument's own currency, already resolved

Every book now carries `FX_rate`, the local currency → USD rate for that
instrument, so nothing downstream has to know the currency map or which rate
file to open. Multiply a local-currency amount by it to get USD.

The currency comes from `instrument_mapping.csv`, the rate is `Derived_Rate`
from `FX/<CCY>.csv`, and the split across the book is:

```
USD 45   EUR 7   CAD 3   GBP 3   JPY 2   AUD 2   HKD 1
```

**The FX build had to move to before the write loop.** It needs `grid` — the
session union `XS_trend` is scored on — so it cannot run earlier; and every book
takes a column off the back of it, so it cannot run later. It sits between the
two.

`currency_of()` **aborts rather than defaulting to USD** on an unmapped
instrument, or one whose currency has no rate file. Defaulting to the base
currency is the worst available failure: silent, plausible-looking, and it means
an instrument's P&L is simply never converted. A CGB would be counted at 1.00
instead of 0.72 — no null anywhere to notice, every number still readable.

Verified on all 63: each book's `FX_rate` matches its own currency's file
exactly, and **no other currency file fits** it (checked against all seven, so a
wrong-file join could not pass unnoticed). Last-session notionals come out at
realistic contract values — CGB $84,909, FDAX9 $769,316, NIY $207,675, HSI
$163,590, SJB $79,232.

**Null on two books, and correctly so.** YAP4's history starts before the AUD
future does (983 sessions), as does YXT4's (375). There is no rate to carry, and
filling them with 1.0 or with the earliest known rate would invent an exchange
rate for a date nobody quoted one.

---

## 2026-08-28 — A bare `pl.read_csv` on a book returns 17 numeric columns as Strings

Found while attaching `FX_rate`; **it predates that column and affects all 63
books.** Polars infers csv dtypes from a bounded prefix (100 rows by default),
and every column with a warm-up period is blank for longer than that:

```
column                                            leading nulls
-Skew, daily_vol_abs, price_vol_curr_ann, XS_trend          256
Carry_sign, TS_trend_sign_UNCAPPED                          255
VoV, VoV_mean_ann                                           339
Skew_sign, XS_trend_sign_UNCAPPED, Trend_sign               511
VoV_sign, Sign_raw, fdm_norm                                594
FDM_MASTER, FDM_MASTER_smooth, SIGNAL                       849
```

The reader sees only blanks, settles on `String`, and arithmetic downstream
either raises or — far worse — a comparison silently succeeds and compares text.
**`SIGNAL`, the final output of the whole pipeline, is one of them.**

`FX_rate` joins the list on exactly two books, YAP4 (983) and YXT4 (375).

**Fixed on the read side with `load_book(inst)`**, which builds the schema from
the header rather than sniffing content — the failure is content-dependent, so
any fix that reads content to decide inherits the bug. A column is text if it is
named in `BOOK_TEXT_COLS` or ends in `_hold` (which `--keep-rule-name`
produces); everything else is `Float64`.

**Not fixable within csv.** A null is an empty field and dtype is the reader's
inference; the only write-side "fixes" available inside the format are to trim
real history until the blanks fit the window, or to write `NaN` instead of blank
— which would make `null_count()` report 0 and `drop_nulls()` keep the rows,
hiding exactly the condition the null exists to record.

The same trap, from the same cause, is documented for the rate files in
`FX/FX_Journal.md`; `load_fx(ccy)` is its counterpart.

**Settled by the dual-write below.**

---

## 2026-08-28 — Dual-write: csv to read, parquet to compute on

Every output is now written twice — `<name>.csv` and `<name>.parquet`, for all 63
books and all 7 rate files. Verified holding **identical frames** in every case.

```
size   csv 333.4 MB   parquet 115.3 MB   (2.9x smaller)
read   csv   0.53 s   parquet   0.22 s   (2.4x faster)
```

**The parquet is not an optimisation, it is the correct artifact.** It carries
dtypes in the file, so there is no inference to get wrong and the String problem
above cannot occur — a naive `pl.read_parquet` on YAP4 returns `SIGNAL`,
`FX_rate` and `Carry` as `Float64` with their 849 / 983 / interior nulls intact
as nulls. The csv stays because a person should be able to open a book.

### Why not trim, given it was on the table

Two reasons, and the second is the one that decided it.

1. **It costs real history.** Trimming to the first fully-populated row (`SIGNAL`,
   the slowest estimator at 849) removes **53,487 rows of 549,477 — 9.7%, about
   3.4 years per instrument.** 6E would start 2002-05-24 instead of 1999-01-04.

2. **It does not fix the bug, and makes the remainder worse.** The nulls are not
   all leading:

```
Carry, Carry_hold_O, Carry_hold_C   5,724 INTERIOR nulls across 19 books
                                    (worst: ZT, 2,175 of 9,098)
XS_trend                               83 INTERIOR nulls across 4 books
```

   No prefix trim reaches those. And those four columns read back as `Float64`
   today only because their first 100 rows happen to be populated — luck, not
   structure. Trimming would remove the columns that currently *announce* the
   problem while leaving the ones where it is latent, so the next occurrence
   would be silent, on `Carry`, with nothing to have warned anyone.

### The mtime guard

`_prefer_parquet` takes the parquet only if it is **not older** than its csv, and
the write order is **csv first, parquet second**. A run that dies between the two
writes therefore leaves the csv winning, rather than a pair whose stale half is
the preferred one. Verified by backdating `ZT.parquet`: the loader falls back to
the csv and still types `SIGNAL` as `Float64`, because the fallback names its own
schema rather than sniffing.

Both artifacts are already gitignored — `Trading_book/` as a directory, and
`*.parquet` globally since line 12.

---

## 2026-08-28 — W switched from 1,280 to 256

**Eq (3.21) (3.24) (3.25) (3.26): W switched from 1280 to 256. Less useless
burn_in, more accurate convergence towards 10.**

### The measurement

W is charged TWICE -- once normalising each alpha (3.21), again normalising the
aggregate (3.25) -- and the warm-ups stack. `FDM_MASTER` cannot begin until the
second window has filled on top of the first:

| | W = 1,280 | W = 256 |
|---|---|---|
| `FDM_MASTER` first session | 2,897 | **849** |
| coverage | 67.5% | **90.3%** |
| mean \|FDM_MASTER\| | 9.73 | **9.93** |
| cap rate | 11.2% | 10.3% |
| instruments producing a forecast | 59 | 63 |

849 = 339 (VoV chain) + 255 + 255, identical on every instrument. The 2,048
sessions saved are pure warm-up: nothing is computed differently, it simply
starts sooner.

Convergence toward Phi = 10 is closer at 256 (9.93 vs 9.73), and the cap binds
LESS often (10.3% vs 11.2%), not more. That was the opposite of what was
expected -- a shorter window was assumed to be noisier -- and the reason is that
a 256-session mean tracks the signal's own magnitude more closely, so the scalar
overshoots less.

### Why 1,280 cost three instruments, and why that is not the paper's 62

At W = 1,280 the stacked burn-in eliminated BTC (2,183 sessions), SO3 (2,030)
and SR3 (2,087) as well as ETH -- 59 tradable. At 256, none are lost.

That 59 is a NORMALISATION ARTEFACT and must not be confused with the paper's
universe of 62, which comes from a different rule entirely: an instrument is not
traded until it has five years of existence. ETH begins 2021-02-23 and crossed
1,280 sessions on **2026-03-27**, so at any backtest cut before that date it is
the single exclusion:

    cut 2025-06-30   62 instruments   short: ETH
    cut 2026-01-01   62 instruments   short: ETH
    cut 2026-06-30   63 instruments   short: none

This panel runs to 2026-08-24, five months past ETH qualifying, which is why the
book now produces signals for all 63. The 5-year rule is a UNIVERSE decision and
is deliberately not applied here; whoever consumes the book applies it and trades
62. `production/s183` encodes it as `oos_start = 5 * TRADING_DAYS_YEAR`.

### Paper vs code

The paper states W = 1,280 in both 3.21 and 3.25. `production/s183/config.py:24`
runs `SCALAR_WINDOW = 256`. The paper does not address that applying the two
normalisations sequentially doubles the warm-up. 256 is now used here, matching
the code.

### Related finding: the FDM round-trip mostly cancels

Verified on CL to 2.24e-14 on uncapped rows:

    f_master / f_direct  =  F_t * mean_W(|f_raw|) / mean_W(|f_raw| * F)  ~=  F_t / <F>_W

If FDM_t were constant over the window this is exactly 1 -- the multiply (3.24)
and the re-normalisation (3.25) cancel. `fdm_raw` has median 1.737 and range
1.00-2.00, but what survives into `FDM_MASTER` is `F_t / <F>_W`, median 0.990,
range 0.78-1.33. Correlation between `FDM_MASTER` and simply normalising
`Sign_raw` once is 0.9966.

So after 3.25, FDM contributes TIMING, not leverage: the level is cancelled by
construction and only the deviation from its own trailing average survives. The
under-engagement correction (mean |Sign_raw| 5.27 -> ~10) is delivered by the
normalisation itself, which targets Phi regardless of its input.

Removing FDM would NOT recover the burn-in -- measured, `normalise(Sign_raw)`
also starts at session 2,897 under W = 1,280. The stack comes from normalising
twice, not from the multiplier.

---

## 2026-08-28 — Eq (3.29): denominator changed from sigma_target to the sizer's own vol

**This is a deliberate departure from the published equation.** `Sig_g_vol` no
longer reproduces the paper's backtest. Recorded here in full because it changes
behaviour, not just presentation.

### What the paper and the frozen code do

`flagship_S183/ig_strategy_183.py:1066`, the implementation that produced the
paper's results:

    realised_vol = daily_ret.rolling(vol_scale_lookback).std() * np.sqrt(TRADING_DAYS_YEAR)
    vol_ratio    = (realised_vol / VOL_TARGET).fillna(1.0)...

So raw 64-day annualised vol over 0.20. Paper text, frozen backtest and
`production/s183` all agree. The first implementation here matched them.

### Why it does not do what it was designed to do

The sizer ALREADY normalises every instrument to the 20% budget, dividing by its
own vol estimate. So `raw_vol / 0.20` does not measure "has this instrument
exceeded its risk budget" -- sizing has handled that -- it measures "is this
market more volatile than 20%", which is a static property.

Measured across the 63 instruments, `raw_vol / 0.20` ranges **0.01x to 3.5x**,
fixed by the instrument. The consequence, from the run of 2026-08-28:

    sit at/near the 0.5 floor (median gate <= 0.55)   15 instruments
    sit at/near 1.0           (median gate >= 0.95)   31 instruments
    genuinely modulating in between                   17 instruments

ETH and BTC sat at the floor on **100.0%** of sessions, gate range 0.001-0.002 --
the value never moved. LEU9, SO3, SR3, YXT4, FGBS9 sat at 1.000 with range 0.000.
CL was at the floor 56.1% of the time. A "maximum reduction of 50%" described as
protection against extremity was crypto's permanent state.

### The frozen code states the intent, and it is not what the input delivers

`ig_strategy_183.py:545` derives the steepness on an explicit assumption:

    "vol_ratio = realised_vol / VOL_TARGET is a dimensionless scale variable
     fluctuating around 1. We want the sigmoid to exhibit regime-change
     behaviour: nearly flat (>= 88% open) for vol inside a +/- 20% band around
     the trigger, nearly saturated (< 12% open) beyond that band."

k = 2/VOL_TARGET = 10 is correct GIVEN a ratio near 1. The ratio is not near 1;
it is near 1 only for markets whose long-run vol happens to be 20%.

### The change

Denominator is now the sizer's own estimate -- `daily_vol_abs` (Eq 3.19's
blended 70/30, price units) divided by the close and annualised, exposed as
`sizer_vol_ann`. The ratio is exposed as `vol_ratio`.

What makes this work is that the two estimates differ in SPEED, not in kind: the
sizer's blend carries a 2,560-day leg and is slow by design; the gate's is 64
days. Their ratio measures "has vol risen above what the sizer assumed" -- a
real, time-varying overshoot signal.

Measured, `fast / sizer` is centred on 1 for EVERY instrument -- median 0.960,
range **0.726 (LEU9) to 1.059 (BTC)** -- against 0.01x-3.5x for `raw / 0.20`.
The sigmoid finally sees the input its calibration assumes.

On CL: `vol_ratio` median 0.990 (5th 0.712, 95th 1.367); time at the floor falls
from **56.1% to 3.0%**; median gate 0.502 -> 0.763.

### What this costs

`Sig_g_vol` no longer matches the published numbers. Anyone reproducing the
paper's results must use `raw_vol / SIGMA_TARGET`. The old form is one line:
replace the `vol_ratio` expression with `g_vol_64_ann / SIGMA_TARGET`.

Not yet re-run against a backtest -- the change is justified by the calibration
argument above, NOT by measured performance.

---

## 2026-08-28 — Eq (3.29): the gate is now smoothed with 3.27's recursion

Adopted. `Sig_g_vol` is the smoothed gate; there is no separate raw column.

### This is the second half of the denominator change, not a separate decision

The two go together. Under the published form (`raw_vol / 0.20`) the gate was
near-static -- 15 instruments sat at the floor with a range of 0.001-0.002, and
ETH and BTC never moved at all -- so it contributed essentially no turnover and
smoothing would have been pointless. Making the denominator relative is what
turned it into a moving quantity, and therefore into a second trading signal.

Adopting one without the other would be incoherent.

### The asymmetry it fixes

Position ~ forecast x gate, so BOTH factors drive turnover. 3.27 damps the
forecast by 38.4% and the gate was then multiplied in UNSMOOTHED, moving 2.21% a
day against the forecast's 4.1%. Smoothing one factor of a product and not the
other is an omission, not a trade-off.

### The alpha sweep

     alpha  half-life   gate |d%|   vs raw   mean |lag|
     1.000          -       2.21%       0%       0.0000
     0.500       1.0d       1.36%     -39%       0.0099
     0.250       2.4d       0.95%     -57%       0.0208
     0.125       5.2d       0.67%     -70%       0.0345

alpha = 0.5 is 3.27's own smoother, so it inherits that justification -- memory
matched to the rebalance rate -- and needs none of its own. The slower alphas buy
more at 2x and 3.5x the lag and would each be a new free parameter to defend.

### Measured effect

    gate daily |d%|       2.21%  ->  1.36%   (-39%)
    position daily |d%|   5.11%  ->  4.63%   (-9.6%)
    median gate           0.799  ->  0.798
    mean |lag|            0.0099            (gate ranges 0.5 .. 1.0)

The level is untouched -- only the noise. Note the position figure is ~10%, NOT
39%: the forecast dominates and the two factors are largely independent, so they
combine closer to quadrature than additively. The gate-level statistic overstates
what reaches the book.

### What it costs

The gate is now a day slower to cut exposure when volatility genuinely spikes.
Whether that lag costs more than the turnover it saves is a cost-aware backtest
question and has NOT been measured. Turnover is separately addressed downstream,
so this change stands on the symmetry argument -- both factors of a product
smoothed the same way -- rather than on transaction costs.

Reversion is one line: drop the `ewm_mean` and alias `vol_gate(...)` directly.

---

## 2026-08-28 — Eq (3.28)-(3.30) removed from the trading book

`g_vol_64_ann`, `sizer_vol_ann`, `Sig_g_vol` and `Sig_g_dd` are deleted, with
the helpers and constants behind them (216 lines). The risk layer belongs
downstream, where positions are known. This entry records what was learned
building it, so the next attempt starts from the finding rather than the
equation.

### The drawdown gate cut winning positions 74.7% of the time

3.30 keys on the instrument's 64-day PRICE return. It is direction-blind: if the
position is short and the price falls, the position MADE money and the gate cuts
it anyway.

Measured across the book -- 58,439 sessions where the gate was meaningfully
closing (DD < -10%):

    position was SHORT and PROFITING on   43,633   (74.7%)

    worst offenders (% of firings that cut a winner)
      VX  98.7%  (3,198)    ZW  92.9%  (2,534)    6J  90.5%  (337)
      NG  89.9%  (3,555)    ZC  85.4%  (2,451)    ZL  85.0%  (2,138)

Above 50% because a trend system is disproportionately SHORT in falling markets,
so the gate's trigger and the position's sign are correlated -- in the wrong
direction. VX at 98.7% is structural: the strategy is usually short VIX futures,
so a big VIX drop is its best outcome, and the gate reads it as a drawdown.

The paper's own words are "the latent drawdown of the last 64 days OF THE
STRATEGY on the instrument". That is P&L, not price. The equation measures
price. The frozen code (`close.pct_change(64).shift(1)`, ig_strategy_183.py:1068)
implements the equation, so paper text and code diverge -- and the text is right.

### Which is why it cannot live here

A P&L drawdown needs the position, the position needs the forecast, and the
gates multiply the forecast. That is resolvable -- lag the position by a day and
it is causal -- but it makes the gate a portfolio-level object, not a
per-instrument column computable from one market's price history. The vol gate
could live here; the drawdown gate cannot, and splitting one risk layer across
two files would be worse than moving both.

### Also learned, and still true wherever they are rebuilt

**3.29's denominator.** `raw_vol / 0.20` is a static instrument property, not a
regime signal: it ranges 0.01x to 3.5x across the book, leaving 15 instruments
pinned at the 0.5 floor (ETH and BTC on 100.0% of sessions, range 0.001) and 31
pinned open. The frozen code's own comment derives the steepness k = 2/tau on
the assumption that "vol_ratio ... is a dimensionless scale variable fluctuating
around 1", which this input is not. Fast vol over the SIZER's own estimate is:
median 0.960, range 0.726 to 1.059 across all 63.

**Both gates need smoothing.** Position ~ forecast x gate, so both factors drive
turnover. 3.27 damps the forecast by 38.4%; an unsmoothed gate then multiplied
back in at 2.21%/day against the forecast's 4.1%. Smoothing at alpha = 0.5 cut
gate turnover 39% and position turnover 9.6%, at a mean lag of 0.0099.

**3.30's return convention.** `close.pct_change(64)` on the Panama series
inverts sign wherever the adjusted close was negative 64 sessions back -- 14
instruments, HO on 88.9% of its bars. The adjusted-difference-over-raw-prior
form is sign-safe and is what daily_ret, xs_return and the VoV overlay use.

### What the book keeps

Everything 3.29 and 3.30 would need is still on the frame: `daily_ret`,
`daily_vol_abs` (the sizer's estimate), `Continuous_C` and `close` for the
sign-safe return, and `FDM_MASTER_smooth` for the position sign. Nothing has to
be recomputed to rebuild the gates downstream.
---

## 2026-08-29 — `%IRX`: the risk-free rate, and the discount-to-yield trap

`IRX/` is stage 2's third output, alongside the books and the FX rates. It is
built here rather than in `3_Portfolio` for the same reason the FX rates are:
this is the file that talks to the vendor, and a rate is panel data whatever
consumes it. `portfolio.py` reads `IRX.parquet` and does no arithmetic on it
beyond applying the accrual.

`%IRX` is the 13-week US T-bill from the vendor's Economic database, reached
through the same accented-hostname patch as everything else — `platform.node()`
lands in an HTTP header that has to be ASCII, and without the patch the vendor
call fails with a bare `ValueError`.

### The column that gets it wrong is the one that looks right

**`%IRX` is quoted as a bank discount, annualised, in percent.** Three
conventions stacked on one number, and only the third is obvious. `4.52` is not
0.0452 and it is not a yield:

```
irx_pct        the vendor's number, 4.52          mean  4.187   min -0.105   max 17.140
irx_bey_pct    bond-equivalent yield              mean  4.323   min -0.106   max 18.165
rf_cal_day     one calendar day of money          mean  1.1843e-04
rf_accrual_next  rf_cal_day x calendar days to the next session
```

with `y365 = 365d / (360 - d·n)` and `rf_cal_day = d / (360 - d·n)`, `n = 91`.
The `360 - d·n` denominator is the *price* per unit of face: a discount
instrument is bought below par and matures at par, so the return is measured
against what was paid, not against par. Dividing the quoted 4.52 by 100 and then
by 365 — the obvious reading — understates the rate by the discount basis and by
the 360/365 day count at once, and produces a number that is wrong by about 1.5%
of itself. Nothing about it looks wrong.

`rf_cal_day` is a division by the day count, not a compounding root, because
once the discount basis is undone the quantity already *is* the price of one day
of money.

**Accrual is on calendar days**, so `cal_days_to_next` is 3 across a weekend and
up to 5 across a holiday weekend (mean 1.411). A T-bill does not stop paying
because Chicago is shut. The accrual column is computed here and applied by
stage 3; the accounting convention — 100% of NAV, credited at t+1 — is stage 3's
decision and is recorded in `Portfolio_Journal.md`.

**Negative rates are left signed.** 11 sessions carry a negative bill yield
(min -0.105%). That is a fact about 2015 and 2020, not an error to floor.

Coverage is 12,553 sessions, 1978-03-07 to 2026-08-28 — the full panel grid,
which is what `verify_stages` asserts.

### It failed as "FX build failed" for one run

`build_irx` reached `_src_hash`, which had never existed in this file — it was
carried over in my head from the deleted `fx_master.py`. Worse than the
NameError: IRX was inside the FX try/except, so the traceback was reported as an
FX failure and the first place I looked was the FX code, which was fine. IRX now
has its own guard. **A shared except block makes two stages one stage** for
every purpose except the one you wrote it for.

---

## 2026-08-29 — Where 3.29 ended up: computed here, applied to the contract size

The two entries above settle what `s_g_vol` should be; this records where it
landed once stage 3 existed to consume it.

`s_g_vol` is computed in this file — it is a per-market column derived from one
market's own history, which is the definition of what belongs in a book — and
written to every book. Constants: `GVOL_SPAN 64`, `GVOL_STEEPNESS 10.0`,
`GVOL_TRIGGER 1.0`, `GVOL_FLOOR 0.50`. `s_g_dd` stays out of the book, for the
reason the 2026-08-28 entry gives: it needs the position.

**Stage 3 multiplies the contract size by it, not the signal**, and carries it
as a column in every Positions file. So `SIGNAL` means the same thing on a gated
day as on an open one — which matters here, because `SIGNAL` is what the IDM
compares across instruments and what every diagnostic reads. A gate folded into
the forecast would have changed the meaning of a column this file publishes.

Measured effect is in `Portfolio_Journal.md` and is not what the equation is
usually sold as doing: gross Sharpe is flat across the ablation, and what the
gate buys is volatility and drawdown, not edge.

### The book also publishes the cost and currency inputs

`cost_rt_of()` and `currency_of()` expose `total_avg_cost_rt_LocalCurrency` and
the instrument's currency from `instrument_mapping.csv`, cached in `_COST_RT`
and `_CURRENCY`. Stage 3 prices commission through them and converts with the
book's own `FX_rate` column. Putting the lookup here keeps one reader of the
mapping file rather than two, and means a stage-3 cost is denominated by exactly
the rate that sized the position.

---

## 2026-08-29 — 3.20's direction was decided by float rounding on 951 bars

Found while auditing something else: rebuilding the books as of 2026-06-30
instead of 2026-08-28 changed `SIGNAL` on **59 of 63 instruments**, across three
decades, by up to 8.64. The traded window was untouched, which is the only
reason it had never surfaced.

### The chain, traced column by column

```
column            rows differing   max abs   first
Continuous_C          11,764/11,764  1.28e+01  1979-10-29   <- anchor moved: EXPECTED
close                          0     0         -
daily_ret                      0     1.03e-15  -
daily_vol_abs                  0     1.69e-14  -
VoV_smooth                     0     8.24e-17  -
VoV_mean_ann                   0     2.39e-17  -
VoV                            2     1.07e+00  1989-01-19   <- diverges HERE
VoV_sign                       2     3.79e+01  1989-01-19
Sign_raw                       2     9.48e+00  1989-01-19
```

Both VoV legs agree to 1e-17 and `VoV` still comes out **exactly negated** --
-0.466603 against +0.466603. Not a numerical residual: a sign flip.

### 3.20's direction factor, and the knife edge it sits on

`vov_signal` takes `sign(C_t - C_t-64)`. That is the *correct* construction and
deliberately not a ratio -- the docstring already explains that `pct_change` on
a back-adjusted series inverts wherever the lagged price is negative, which is
14 instruments here. The difference form is anchor-invariant **in exact
arithmetic**.

It is not in float64. Subtracting two Panama levels of equal magnitude cancels
to rounding: **1.4e-14 at a bund price of 117, 4.5e-13 on the Euro Stoxx at
2,670**. On a bar where the market genuinely returned to the same level after 64
sessions, the sign is then decided by the last bit.

```
BRN 1991-06-24   -1.421e-14  ->   0.000e+00     - -> +
BRN 1993-10-07    7.105e-15  ->  -7.105e-15     + -> -
GC  1995-12-28   -2.274e-13  ->   0.000e+00     - -> +
GF  1989-01-19   -2.842e-14  ->   0.000e+00     - -> +
GF  1992-06-10    0.000e+00  ->  -2.842e-14     + -> -
HG  2000-07-05    0.000e+00  ->  -2.842e-14     + -> -
LE  1984-04-18    0.000e+00  ->  -2.842e-14     + -> -
```

**951 of 542,849 bars (0.175%) sit on this edge.** Eight flipped on one anchor
move.

### The amplifier is 3.27's smoothing, NOT the pooled FDM -- measured

The first reading of this was wrong and is worth recording as an error of
PARTITION rather than of measurement.  Seeing `SIGNAL` differ on 59 of 63 books
with a max of 8.64, I attributed the coupling to the pooled FDM: it is estimated
across the whole universe per session, so one market's flip does move the
multiplier for every other.  It does -- but by nothing that matters.

Splitting the instruments by whether the fix changed THEIR OWN `VoV_sign`:

```
channel                                  insts   max dSIGNAL   rows |d|>0.01
own knife-edge bars corrected               28       11.7579   11,016/261,155
pooled FDM only (own VoV_sign untouched)    35        0.0126        89/285,726
```

**The pooled-FDM channel moves SIGNAL by at most 0.0126 on a +/-20 scale --
0.06% of full scale.**  On a 35-contract line at SIGNAL 10 that is 0.04
contracts: deleted by truncation to whole contracts long before the 3.36 buffer,
an 80x wider band, would look at it.  It cannot move a position.

The real amplifier is 3.27's smoothing on each instrument's OWN series: 86
corrected bars across 28 instruments produce 11,016 rows moving more than 0.01,
because an EWMA carries a one-bar correction forward for years.  That is a
smoother working, not a defect.

*(The second attempt at this split was also wrong, and for a subtler reason:
`{BRN, GC, GF, HG, LE}` is the set flipped by the ANCHOR MOVE, while the FIX
corrects every knife-edge bar whose float residual was negative -- 86 bars on 28
instruments.  KC sat in that set and moved 11.76 while filed as "unaffected".
Two partitions, two wrong answers, before the numbers were split on the right
question.)*

So the pooled FDM needs no change.  It behaves as a pooled estimator should:
every observation influences the shared estimate, in proportion, and the
proportion is 0.06%.

### It reached live signals, not just history

The anchor re-anchors at the present on EVERY panel refresh, so the rounding
changes daily. **47 knife-edge bars since 2024, 14 in 2026.** `FGBM9 2025-11-05`
already carries `diff = +1.421e-14` against a float spacing of 1.4e-14 at that
price -- its direction is already a coin toss. The 2026 window surviving the
2026-06-30 anchor move was luck.

### The fix

`|diff| <= 1e-9 * price` snaps to zero before the sign is taken, preserving the
documented "exact zero counts as positive".

**This is not the epsilon band the docstring rejects.** That objection is to a
band wide enough to reclassify genuine small moves, and it is right: ticks here
run 1e-4 to 1e-2. A relative 1e-9 sits five orders of magnitude below a tick and
above the 1e-13 cancellation floor -- it can only ever remove rounding. Smoke
tested: +/-1.4e-14 stays +1, a -0.01 tick still flips to -1.

Anchor-invariance re-tested on a full rebuild against an as-of-2026-06-30
rebuild, both `--no-fx` so neither needed the vendor:

```
                    BEFORE                      AFTER
SIGNAL              59 insts, max 8.640e+00     0 rows, max 3.706e-12
VoV_sign            8 bars,   max 3.792e+01     0 rows, max 2.310e-12
Sign_raw            8 bars,   max 9.480e+00     0 rows, max 1.139e-12
FDM_MASTER_smooth   59 insts, max 8.640e+00     0 rows, max 3.706e-12
```

`Continuous_C` levels still differ on 252,620 rows -- the anchor genuinely
moved -- while its DIFFERENCES agree exactly, which is the invariant that was
supposed to hold all along.

### What it changed

```
                1990-01-02 .. 2026-08-28        2026-01-02 .. 2026-08-28
              before      after                before      after
gross ret     14.30%     14.32%                10.77%     10.68%
gross SR       1.219      1.220                 1.311      1.302
cost/yr        1.93%      1.92%                 1.48%      1.49%
NET ret       12.38%     12.40%                 9.30%      9.19%
NET SR         1.054      1.056                 1.132      1.122
equity      $21.41 B   $21.57 B              108,762,471  108,686,886
CAGR          15.52%     15.54%                13.28%     13.28%
```

Small either way, and the direction differs between windows, which is what a
coin-toss defect should look like once the coin stops being tossed. **Every
historical figure in these journals was computed on signals that were partly
rounding-determined**; the ones above supersede them.
