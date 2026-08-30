# Portfolio Journal

*Opened 2026-08-28.*

The durable record behind `portfolio.py` — stage 3, which turns forecasts into
contracts. Same convention as the other journals: every claim names the number
that produced it.

---

## 2026-08-28 — Stage 3 exists because a position is not an instrument property

Equations 3.32 and 3.33 implemented. `python portfolio.py` takes ~4s over 12,552
sessions × 63 instruments.

Two of 3.32's inputs do not exist for a single market: `w_i` needs the count of
active instruments, `IDM_t` needs the correlation across them. And `E_t`
compounds, so position → P&L → NAV → next position is a **sequential loop**,
where everything in `2_Engine` is vectorised column math. That is the whole
argument for a separate stage.

`price_vol_USD_ann` from the book *is* σ$ᵢ,ₜ — that column was added for this.

---

## 2026-08-28 — Fixed notional checked, compounding kept as the default

`--fixed-nav` sizes off a constant base instead of compounding equity. Added to
test whether the liquidity problem below is caused by compounding — it is — but
**compounding remains the default**, matching the thesis.

```
                          compounding      fixed $100M
equity            100M -> 1,285,175M    100M -> 1,114M
ann ret                       20.73%           20.69%
ann vol                       16.89%           16.84%
Sharpe                          1.23             1.23
max DD                         27.3%            17.9%
CAGR                          21.29%            5.04%
positions truncated to 0         168            2,880
```

**The risk-adjusted result is identical** — Sharpe 1.23 and ~16.9% vol either
way. The strategy's quality is not an artifact of compounding; only its scale
is.

`ann_ret` AND `cagr` ARE DIFFERENT NUMBERS AND BOTH ARE NOW REPORTED. Under
fixed notional 20.69% is return on a *constant* capital base and is **not** a
growth rate — the equity grows 11.1x over 45 years, a 5.04% CAGR. Under
compounding the two nearly coincide (20.73% / 21.29%), which is precisely why
quoting one for the other would go unnoticed.

Truncation costs more under fixed notional — 2,880 positions floored to zero
against 168 — because $100M in the 2020s cannot afford a contract in the
expensive markets (BTC, ETH, FDAX9). Same effect as the low-NAV case, biting at
the end of the history rather than the start.

### It does resolve the liquidity problem

Position as a share of the held contract's open interest:

```
year     median %OI    p90 %OI    p99 %OI   % lines >100%
1981          4.78%     15.77%     29.39%           0.0%
1990          1.39%     10.87%     44.77%           0.1%
2000          0.75%      4.05%     14.99%           0.0%
2010          0.05%      0.36%      1.93%           0.0%
2026          0.02%      0.14%      1.78%           0.0%

whole history: median 0.11% of OI,  0.02% of lines >100%
```

against **median 52% and 37% of lines over 100%** under compounding. The
constraint inverts: the tightest years become the early 1980s, when the markets
were small and the constant NAV is proportionally largest, and it eases
monotonically. The worst names are the same ones the thesis annex flags — RS,
SJB, GF, PA, PL, DX — but at 1–2% of open interest rather than 300–600%.

### Bug this surfaced: max drawdown was measured on the sizing base

`stats()` computed drawdown from `NAV`, which under fixed notional is constant
by construction — so it reported a **0.0% max drawdown** on a strategy that
plainly has them. Under compounding `NAV` and equity are the same series, which
is exactly why it hid.

Fixed to measure on `equity_USD`, and `Portfolio.csv` now carries both columns:
`NAV` is the sizing base, `equity_USD` is the money. `verify_portfolio` had the
same flaw and now reconciles P&L against equity rather than the sizing base, so
the check passes in both modes for the right reason instead of passing in one
mode by coincidence.

---

## 2026-08-28 — IDM correlation span: it is noise, on every dimension

Tested across a **16× range**, after the holiday fix below:

```
IDM span   mean     sd  |d/day|  at cap  turnover  ann vol  ann ret  Sharpe  max DD
span 64   2.932  0.811   0.0917    5.7%    334.3x   17.12%   21.24%    1.24   27.9%
span 128  2.918  0.793   0.0769    3.5%    327.2x   17.03%   21.12%    1.24   28.0%
span 256  2.905  0.779   0.0690    1.8%    323.4x   16.95%   20.94%    1.23   27.5%
span 512  2.890  0.768   0.0642    1.3%    321.2x   16.89%   20.73%    1.23   27.3%
span 1024 2.876  0.758   0.0613    0.8%    320.8x   16.87%   20.57%    1.22   27.4%
```

Sharpe moves 1.22 → 1.24 and vol 16.87% → 17.12% across a sixteenfold change in
horizon. **That is noise.** Every column is monotonic, which says the direction
is real, but the magnitude is not worth an argument.

Turnover was the one place a difference was expected — a jumpier IDM ought to
churn positions — and it barely moves either (334x → 321x, 4%). The reason is
structural: **IDM is a single scalar multiplying every position at once.** A
wobble in it rescales the whole book proportionally, and truncation to whole
contracts absorbs most small rescalings without changing any contract count.

**`IDM_CORR_SPAN = 512`, matching `FDM_CORR_SPAN`.** With nothing to choose on
merit, the tiebreak is that the pipeline's two diversification multipliers now
estimate their correlations over the same horizon — one fewer arbitrary number
to explain. It is also the better half of the range: the cap binds on 1.3% of
sessions against 5.7% at span 64. `IDM_MIN_PERIODS = 256` matches
`FDM_MIN_PERIODS` for the same reason: a full trading year of overlap before a
pair is trusted, independent of the decay applied to it.

*(256 was adopted first, on the grounds that it matched the one-year convention
of `XS_LOOKBACK`, `SKEW_WINDOW` and `TRADING_DAYS_YEAR`. Superseded — matching
the other multiplier is the stronger consistency argument, and the measurement
says the choice is free either way.)*

**`FDM_CORR_SPAN` stays at 512**, and shortening *it* to 256 is the one direction
that is NOT free. Measured: mean FDM +2.31%, 61.3% of sessions moving >1%, and
time pinned at the analytic 2.0 cap rising from **27.7% to 39.2%** — the
multiplier saturating, and so carrying less information. The asymmetry is worth
keeping in mind: the IDM is a scalar on a 63-market correlation, which is a
well-estimated object at any of these horizons, whereas the FDM is a 4×4 whose
noise the cap converts directly into lost information.

Shipped configuration:

```
IDM   mean 2.890   sd 0.768   at cap 1.3%
      ann ret 20.73%   ann vol 16.89%   Sharpe 1.23   max DD 27.3%
      turnover median 164x/yr    positions truncated to zero: 168
```

---

## 2026-08-28 — The measured IDM, and why the 4.0 cap is a rail

```
mean pairwise correlation : 0.078
1/sqrt(w'Cw)              : 3.28   (last 512 sessions, 63 instruments)
sqrt(N) if decorrelated   : 7.94
```

`1/sqrt(w'Cw)` runs from 1 (everything perfectly correlated) to √N (everything
independent). The paper's 4.0 is √16 — Carver's figure for 16 decorrelated
instruments. On this universe it is a **safety rail, not a binding constraint**:
63 genuinely diversified but not independent markets land at 3.28, and the cap
touches only 1.8% of sessions. An earlier guess that it would bind constantly
was wrong.

---

## 2026-08-28 — Bug found by the verification: the book was liquidated every holiday

`verify_portfolio` failed on "at least one active instrument per session",
`min 0`. Chasing it found a genuine defect, not a bad check.

The 21 offending sessions are **all US holidays** — Thanksgiving, Presidents'
Day, Memorial Day, July 4th, Labor Day, Christmas, Good Friday — where the grid
(the union of 63 calendars) carries a date on which only 1–5 foreign markets
have a bar, none of them yet tradable.

But the underlying flaw was much wider. `act` is false wherever an instrument has
no bar, so its sized position came out **zero** — meaning every market was
liquidated on every day it was shut and bought back the next session, in a
market that could not be traded in either direction. Markets keep different
calendars, so this fired somewhere almost every day.

Fixed by carrying the position through a closed market. The P&L is unaffected: a
forward-filled price has a zero difference on exactly those sessions.

```
                    turnover   ann ret   Sharpe   contract count changes
before (liquidate)    416.2x    18.95%     1.15         99.9% of sessions
after  (carry)        323.4x    20.94%     1.23         96.9% of sessions
```

The spurious round trips were costing **1.99pp of annual return**.

The check that failed was then replaced with the one that would have caught it
directly — *the book never goes flat once it has started* — because `n_active`
reaching zero on a holiday is correct and asserting otherwise asserts that
markets do not close.

---

## 2026-08-28 — Turnover is the open problem, and buffering is the answer

Even after the fix, a contract count changes on **96.9% of sessions** and
turnover runs at a median 165x NAV per year. Priced with the panel's own
`total_avg_cost_rt_LocalCurrency`:

```
median annual cost : 3.18% of NAV
gross annual return: 20.94%
```

So roughly **a sixth of the return goes to execution**, taking Sharpe from ~1.23
to ~0.96. This is not a defect in the sizer — it is what an unbuffered
continuous position rule does, and precisely the reason Carver's buffering
exists. With ~35 contracts in a typical line, a 3% move in the raw size crosses
an integer boundary, so truncation alone does almost nothing to damp trading.

**Buffering is the next thing to build**, and it should be measured against
these numbers.

---

## 2026-08-29 — Gates 3.29 and 3.30: they buy risk reduction, not Sharpe

Both gates are now live. `s_g_vol` is computed in the book (`trading_book.py`,
where it is a per-market column of one market's own history) and consumed here;
`s_g_dd` is computed here, because a drawdown on the strategy's P&L needs the
position. That split is the one `Trading_Book_Journal.md` argued for.

**`s_g_vol` multiplies the contract size, not the signal.** The distinction is
not cosmetic. A gate on the forecast changes what the strategy *believes*; a
gate on the size changes what it *risks* while leaving the belief intact. Only
the second is a risk control, and only the second leaves `SIGNAL` meaning the
same thing on a gated day as on an open one — which matters because `SIGNAL` is
compared across instruments in the IDM and read directly in the diagnostics.
`s_g_vol` is also carried as a column in every Positions file, so a small
position on a wild day is self-explaining rather than a number to reverse out.

Shipped constants — `GVOL_SPAN 64`, `GVOL_STEEPNESS 10.0`, `GVOL_TRIGGER 1.0`,
`GVOL_FLOOR 0.50`; `GDD_LOOKBACK 64`, `GDD_THRESHOLD -0.10` (= -tau/2),
`GDD_STEEPNESS 10.0` (= 2/tau), `GDD_FLOOR 0.50`.

`GVOL_TRIGGER = 1.0` makes the vol gate one-sided: it only ever cuts, never
adds. A gate that opens above 1.0 would be a leverage rule wearing a risk
control's name, and the book has an IDM for that already.

### The ablation, which does not say what a gate is supposed to say

```
                     gross ret   gross vol   gross SR   max DD    cost/yr   net SR
both gates (shipped)    14.30%      11.74%      1.219    17.6%      1.93%    1.054
no s_g_vol              18.92%      15.41%      1.228    23.3%      2.41%    1.071
no s_g_dd               16.28%      13.15%      1.238    18.7%      2.20%    1.071
neither                 21.62%      17.26%      1.253    24.3%      2.75%    1.094
```

*(Recomputed 2026-08-29 on the corrected commission model below. The gross
columns are unchanged; the cost and net columns moved, and the conclusion did
not.)*

**Gross Sharpe is flat to slightly negative across the whole ablation** — 1.253
ungated against 1.219 gated. The gates take return and volatility down in almost
exact proportion, which is what a well-behaved scaling rule does and is *not*
what the equations are usually sold as doing. On this book they do not find a
bad regime and step out of it; they de-lever indiscriminately.

What they do buy is the risk profile: **volatility 17.26% → 11.74%** and **max
drawdown 24.3% → 17.6%**, at a cost of 0.040 net Sharpe. That is a real and
defensible trade — a third of the drawdown for four basis points of Sharpe —
but it should be stated as the trade it is, not as an edge. They also cut
turnover, dropping cost from 2.75% to 1.93% of NAV per year.

The honest reading is that both gates are *risk budget* instruments, and if the
target were 20% realised volatility rather than 20% nominal tau, running ungated
at a lower tau would land in a similar place. That comparison has not been run.

---

## 2026-08-29 — Eq 3.36, the buffer: implemented, swept, and 0.10 is not the optimum

The no-trade buffer is applied to `N_contracts` — a band of `±b` around the
target position, inside which the existing position is held rather than
adjusted. Executed size is target-or-hold, never a partial move, and
`verify_portfolio` asserts both the band and that property.

This is the answer to the turnover problem the 2026-08-28 entry opened. Swept on
clean turnover, over the full 1990+ window:

```
    b     gross ret    gross vol   gross SR    cost/yr    NET SR
  0.00       14.39%       11.76%      1.223      2.14%     1.042
  0.10       14.30%       11.74%      1.219      1.93%     1.054   <- shipped
  0.20       14.25%       11.72%      1.215      1.79%     1.063
  0.25       14.25%       11.72%      1.215      1.73%     1.068   <- peak
  0.30       14.18%       11.73%      1.209      1.68%     1.065
  0.40       13.98%       11.80%      1.185      1.60%     1.049
  0.50       13.98%       11.90%      1.175      1.53%     1.046
  0.60       14.04%       12.15%      1.155      1.48%     1.034
```

*(Reswept 2026-08-29 on the corrected commission model. Costs are uniformly
higher, so every net figure drops ~0.055 — and the shape is identical.)*

Two things are visible and neither is subtle. **Gross return is nearly flat to
b = 0.30** — a wide band costs almost nothing in signal fidelity, because the
positions it declines to make are the small ones. And **cost falls monotonically
throughout**, from 2.14% to 1.48% of NAV per year. Net Sharpe is therefore the
difference of a flat line and a falling one: it rises to a plateau at
**b = 0.20–0.30 (1.063–1.068)** and then breaks when the band gets wide enough
to hold genuinely stale positions.

**`BUFFER = 0.10` is kept**, which is the paper's value and sits on the rising
part rather than the top. The gap to the b = 0.25 peak is 0.014 Sharpe — and it was 0.015 before the
commission fix, so the decision is not sensitive to the thing that just moved
every net number on the page. That is
inside anything one would call significant on 9,522 sessions, and moving the
constant to the argmax of a sweep run on the whole history is in-sample
optimisation dressed as tuning. If it moves later it should move on an
out-of-sample argument, not this table.

*(A previous sweep put the optimum at b = 0.50. That was measured before the
`n_active` fix below, and it was reading the thin-session artifact: turnover was
being manufactured on holidays, so wider bands looked better than they were.
Superseded.)*

---

## 2026-08-29 — Costs are deducted from NAV, and `cost_lag_USD` is why the reconciliation closes

Commission is priced per instrument per day from the panel's own
`total_avg_cost_rt_LocalCurrency`, converted with that instrument's `FX_rate`,
and aggregated to a portfolio series. Both levels carry `cost_USD` and
`net_pnl_USD`; the portfolio also carries `gross_ret` and `net_ret`.

`ret` was renamed **`gross_ret`**, which is the point of the exercise. A column
called `ret` sitting next to a column called `cost` invites exactly one question
and answers it wrong half the time.

**Costs are deducted from NAV, not reported alongside it.** The alternative —
tracking a gross equity curve and a cost tally separately — makes the sizing
base wrong, since 3.32 is linear in `E_t` and a strategy that has paid $2.6B in
commission is not entitled to size as though it has not. Deducting compounds the
drag properly, which is the only treatment that survives 37 years.

### The shift, and why it was needed

Reconciling `equity[t] = equity[t-1] + net_pnl + interest` failed by exactly one
session, repeatedly, because the cost of a trade decided at close *t* is not
borne until that trade executes. `cost_USD` is the cost of the decision;
**`cost_lag_USD` is that same series shifted one session forward**, and it is the
one that enters P&L. Both are written. `net_pnl == pnl - cost_lag` at both
levels, and `verify_portfolio` checks the shift itself
(`cost_lag_USD is cost_USD shifted one session`, max diff 0.0e+00) rather than
only the sum — so the two can never drift back together silently.

Carrying both columns is deliberate. The unlagged series answers "what did
today's decision cost", the lagged one answers "what came out of the account
today", and collapsing them to one column loses whichever question you did not
have in mind when you wrote it.

Shipped: **1.93% of NAV per year, $3.4B total**, taking Sharpe 1.219 → 1.054.
*(1.30% / $2.6B until the roll under-billing was fixed on 2026-08-29,
below.)*

---

## 2026-08-29 — Interest on cash: 100% of NAV, credited as an overnight rate at t+1

`%IRX` (13-week T-bill) is built by `trading_book.py` into `IRX/` and read here.
The build is documented in `Trading_Book_Journal.md`; what belongs in this file
is the accounting convention, because it is a choice and not a derivation.

- **The whole NAV earns.** Futures are margined, so the cash is not consumed by
  the positions; the account earns on its full balance. Not a modelling
  simplification — it is what the broker does.
- **Credited at t+1, as an overnight rate.** The rate observed at close *t* is
  applied to the balance and paid the next session. Columns:
  `rf_accrual_next` (what was computed tonight) and `rf_accrual_applied` (what
  landed today), `interest_USD` for the money.
- **Accrued on calendar days**, so a weekend earns three days and a long holiday
  weekend four. The T-bill does not stop paying because Chicago is shut.
- **`total_ret` is `net_ret` plus the interest contribution**, and the three are
  checked against each other (`total_ret - net_ret == interest / NAV[t-1]`, max
  diff 5.5e-18).

Over 1990+: **$5.3B credited**, and 11 sessions at a negative bill rate — which
is left signed rather than floored, because a negative bill yield is a fact
about 2015 and 2020 and not an error to clean up.

The daily conversion was corrected during the build. The bill is quoted as a
**discount**, not a yield, so `rf_cal_day = d / (360 - d·n)` with `n = 91`, and
the annual-to-daily step is a division by the day count rather than a
compounding root — the rate is already the price of one day of money once the
discount basis is undone.

---

## 2026-08-29 — The window: 1990-01-01, and five years before a market is tradable

Two rules, both confined to `portfolio.py` — no other stage knows about them.

- **`START_DATE = "1990-01-01"`.** The paper's window.
- **`MIN_SESSIONS = 256 * 5`.** An instrument is not tradable until it has 5
  years of its own sessions behind it.

**The burn-in and every computation continue on an untradable instrument.** It
is excluded from `tradable`, so it takes no position and contributes no P&L, but
its vol estimate, its forecasts and its correlations against the rest of the
book keep running the whole time. The alternative — starting the machinery when
the instrument becomes tradable — would give every new market a cold EWMA on its
first tradable day, which is precisely the day the rule exists to avoid.

The distinction that took the longest to get right is between *tradable*,
*begun* and *alive*, and they are three different things:

```
tradable      5 years of own sessions have elapsed
began         this instrument has ever been sizeable (finite SIGNAL, sigma > 0, finite fx)
alive         on or before this instrument's last bar in the panel
in_universe   all three
```

`in_universe` is what the risk budget is divided by. `began` is a cumulative
maximum, never a per-day test — an instrument that has started does not un-start
because today's signal is null.

---

## 2026-08-29 — Three bugs the checks found, and all three were silent

None of these would have raised anything. Recorded in full because the common
factor is worth more than the individual fixes: **each was a well-formed number
in the right column**, and each was found by a check that re-derived the value
rather than inspecting it.

### `n_active` counted today's attendance, not the universe

`w_i` divides the risk budget by the number of active instruments. It was
counting instruments with a bar *today*, so on a session when only a handful of
foreign markets traded, the budget per instrument jumped from **1.94% to 4.38%**
and turnover from **0.37x to 3.00x** — on **482 sessions**. The book was
concentrating itself onto whichever markets happened to be open.

Fixed to weight over `in_universe`, which does not blink on a holiday. **Costs
fell 37%.** This is also what invalidated the first buffer sweep.

### `notional_USD` was priced off the Panama close

Negative on **39,585 rows**. Panama back-adjustment is anchored at the present,
so differences are valid and *levels are not*: 14 books carry a negative
`Continuous_C`, and CL bottoms at **-29.11**. Notional is a level, so it must use
the raw close. Fixed, and `cost and notional never negative` is now a check —
the cheapest possible assertion for a quantity that has no negative branch.

### Interest accrued on the idle grid

The panel starts in 1978 and the traded window starts in 1990, so a run starting
in 2026 was crediting interest on an untouched balance for the intervening
decades and then **sizing off $801M instead of $100M**. The equity curve looked
plausible the whole way. Fixed with a `started_t[t]` gate: nothing accrues before
the portfolio has begun.

---

## 2026-08-29 — Reconciliation against the paper: A.29 leverage and the post-2010 Sharpe

### Gross notional over NAV

```
gross/NAV     mean 6.90x    median 5.26x    max 26.24x       (paper A.29: 8.62x)
```

By class, mean leverage contributed and share of the total:

```
class          n    mean lev      max    share
Bond          12        2.99    11.75    43.4%
STIR           3        1.73    15.99    25.0%
FX             9        0.85     2.94    12.4%
Ags           13        0.62     1.90     9.0%
Equity        11        0.30     1.81     4.4%
Metals         5        0.24     1.15     3.5%
OilGas         6        0.14     0.55     2.1%
Carbon         1        0.01     0.06     0.1%
Vol            1        0.00     0.03     0.1%
Crypto         2        0.00     0.05     0.0%
```

We now sit **20% BELOW the paper**, having started ~31% above it. The gates
account for the whole swing: ungated the book runs at 17.26% volatility against
11.74% gated, and leverage scales with it. The remaining gap is not a defect —
it is the same trade the ablation above describes, seen from the notional side.

**Leverage is a rates phenomenon**, 68.4% of it in Bonds and STIR. That is
arithmetic, not a position: leverage per unit of risk budget is `1 / annualised
return volatility`, and a Eurodollar contract has an annualised return vol near
0.5%. Nothing is wrong when a STIR line shows 16x notional; something would be
wrong if it did not.

*(An earlier reading put STIR at mean 2.09x and 42.4% of the book, well above
the paper's 32.5%, and traced the tail to LEU9 in the ZIRP years — 0.131% return
vol producing 793x leverage on 2021-12-24 with both gates open. Those figures
predate the gates and the `n_active` fix and are superseded by the table above;
STIR is now 1.73x and 25.0%. The 1% volatility floor proposed to cap that tail
was postponed and has not been revisited.)*

### Post-2010

```
                  ann ret    ann vol    Sharpe      n
1990-2009 net      14.23%     12.65%     1.125   5,188
post-2010 net      10.15%     10.54%     0.963   4,334
full 1990+ net     12.38%     11.74%     1.054   9,522
```

**Post-2010 net Sharpe 0.963, against the paper's 0.991** — now slightly BELOW
it, where before the commission fix it read 1.009 and slightly above. The
comparison flipped sides on a bookkeeping correction, which is worth holding on
to: the margin was never wide enough to carry an argument.

The decay from the first half is real — 1.125 to 0.963 — and it is still **not a
cost story**: cost drag *falls* across the two halves, 2.12% to 1.69% of NAV per
year.

Per-class gross Sharpe, contribution to the portfolio's return:

```
class        SR 90-09    SR 10+    ratio
Ags             0.568     0.232     0.41
FX              0.634     0.339     0.53
OilGas          0.554     0.318     0.57
Metals          0.595     0.414     0.70
Equity          0.305     0.248     0.81
Vol             0.362     0.329     0.91
Bond            0.460     0.923     2.01
Carbon              -     0.189        -   } no position in the
Crypto              -    -0.150        -   } first half
STIR                -     0.249        -   }
```

**Four classes roughly halve — Ags, FX, OilGas, Metals — and Bonds double.**
That second half was missed in an earlier pass and it changes the reading: this
is not uniform decay, it is the commodity and currency trends thinning while the
2010s bond trend more than compensates in its own class. The portfolio-level
number falls anyway because the halving classes are 24 of 63 markets.

Not a STIR story either — STIR has no first-half position to decay from. It is
trend-following being less good after 2010 outside fixed income, which is the
finding the literature already has.

---

## 2026-08-29 — Diagnostic audit: 22 checks, ~2M values, zero code bugs

After bugs at nearly every step, the pipeline was audited end to end with random
datapoint sampling rather than more assertions: panel → book agreement, Panama
invariants, EWMAC, both gates, 3.32, the buffer, P&L through 14,793 rolls, cost,
aggregation (relative error 3.4e-16), signs, orphan rows, hold-vs-rule, interest,
the FX mapping and IRX.

**Nothing was found.** That is the report.

Worth recording against my own future confidence: **three of the audit's
"failures" were mine, not the code's.** `s_g_dd` was declared broken three
separate times — once from computing the 64-session window on a filtered file,
once from an off-by-one in the cumulative product, once from reading `w` and
`IDM` out of the written file instead of the live grid. The code was correct
every time. The lesson is specific and cheap to apply: **re-derive from the same
object the code used, not from the artifact it wrote**, because the artifact has
already been filtered, rounded and joined.

---

## 2026-08-29 — Commission: a roll is two executions, not a change of size

The cost model billed `|N[t] - N[t-1]| . (cost_rt/2) . FX`. Correct while the
position stays in one contract — 100 to 130 trades 30 lots. **Wrong the moment
the contract changes**, because a roll closes `|N[t-1]|` of the expiring month
and opens `|N[t]|` of the next, and those are different instruments. A September
short cannot be netted against a December short, so `|dN|` was the difference of
two numbers the market never netted. The one-way quantity is both legs.

The degenerate case is the common one: **3,924 of 9,377 roll events (41.8%) have
`|dN| == 0` exactly** — the new month sized like the old — so a full two-leg roll
was charged *nothing at all*. Not an underestimate. A zero.

```
                              billed        true      short
whole history            $2.624 B     $3.707 B   $1.083 B   +41.3%
```

Found by stage 4, which had to price every leg because an order ledger has no
choice: `Bookkeeping_Journal.md` has the discovery.

### The fix

```python
traded = np.where(rolled[t], np.abs(prev) + np.abs(N[t]), np.abs(N[t] - prev))
tr     = traded * cost_rt * fx[t]
```

`rolled` is precomputed from the symbol grid, **forward-filled first**. That
detail is the whole risk in this change: `sym` is null wherever a market had no
bar, so comparing adjacent *rows* reads a holiday as two rolls, one into the null
and one out. It is the same union-grid trap that hid 578 rolls in stage 4, and
walking into it here would have *over*-billed instead of under-billing — the
error that flatters nothing and so gets noticed, but an error all the same.

### What it costs

```
                    before      after
cost/yr              1.30%      1.93%
net ann ret         13.00%     12.38%
net Sharpe           1.108      1.054
equity             $27.00 B   $21.41 B
CAGR                16.24%     15.52%
truncated to zero      302        313
gross ret/vol/SR    unchanged  14.30% / 11.74% / 1.219
```

**Gross is untouched, and that is a check rather than a coincidence.** Sizing is
linear in `E_t` and P&L is linear in the position, so `pnl/NAV` is scale
invariant; a uniformly poorer account holds proportionally fewer contracts and
earns the same *rate*. Only integer truncation breaks the invariance, which is
why 11 more positions round to zero and nothing else in the gross column moves.

**Post-2010 net Sharpe goes 1.009 to 0.963** and so crosses from just above the
paper's 0.991 to just below it. The honest reading is not that the strategy got
worse — it is that a 0.018 margin was never enough to carry the claim, and a
bookkeeping correction was always going to be able to flip it.

### Verified against an implementation that shares no code

The decisive test is not that the number went up. Stage 4 derives leg quantities
from the position and symbol path; stage 3 computes them inside the sizing loop.
Different code, different data structures, same claim. Before the fix they
disagreed by $1.083B on rolls and matched to the cent off them. After:

```
roll commission   ledger $0.999B   stage 3 $0.999B   $0M apart
```

`verify_bookkeeping` now asserts that equality rather than reporting it, so the
correction cannot be quietly lost. (The roll total falls from $1.210B to $0.999B
because the book is poorer and therefore smaller — the same scale invariance.)

### What did not change

The **buffer sweep was rerun** and the shape is identical: peak still at
b = 0.25, plateau 0.20–0.30, every net figure down ~0.055. The gap from the
shipped b = 0.10 to the optimum is 0.014, against 0.015 before. A conclusion that
survives a 48% increase in the cost it is trading against is a conclusion worth
keeping.

The **gate ablation** likewise: gross Sharpe still flat across it, the gates
still buying volatility and drawdown rather than edge.

---

## 2026-08-29 — A run parameter that is not in the output is a trap

`Portfolio.csv` now carries **`started`**, the boolean stage 3 already computed
to gate interest accrual. One column, and it exists because of how it was found.

A 2026-01-02 run reconciled 9/10: the interest tie broke by **97%**, $102.3M
recomputed against $2.6M booked. Nothing was wrong. `reconcile.py` read
`START_DATE` from the module — 1990-01-01 — while the run had been given
`--start-date 2026-01-02`, so it accrued thirty-six years of bill yield on an
idle $100M and reported a correct column as a catastrophe.

**The tempting fix is to infer the window from the data, and it is the wrong
one.** `--start-date` changes what every other column *means*: before it the
book holds nothing, trades nothing and earns nothing. An artifact that does not
say which window it describes forces every consumer to guess, and they will
guess differently. Stage 4's ledger already inherits the window silently;
`reconcile.py` was simply the first reader to guess out loud.

So the run records its own gate, and the reconciliation reads it. The arithmetic
being checked — balance x rate — stays independently recomputed from IRX. Only
the *parameter* comes from the artifact, which is the one thing an artifact is
authoritative about.

*(Related, found in the same run: `verify_bookkeeping` reported "12,552 accruing
sessions" when 170 had accrued. It was counting sessions on which the bill
quoted a rate, not sessions the book was credited. The check was right and its
own description was 70x too flattering.)*

---

## 2026-08-29 — Two mislabels in the terminal report, both stating uncomputed numbers

Neither changes a stored column; both were the report claiming things it had not
worked out. Caught by reading the output, which is the only way this class of
error is ever caught.

**The NET line printed the GROSS volatility.** `net_sharpe` divided net return by
`ann` — the standard deviation of `gross_ret` — and the printed `vol` on the NET
line was that same gross figure, under a label saying it was the net one.

```
gross vol  8.2188%      net vol  8.2098%       0.11% apart
net Sharpe   1.1310  (printed)     1.1323  (as labelled)
```

**0.0012 of Sharpe is exactly why it survived.** A figure wrong by a rounding
error is never caught by looking at it; it is caught by reading the code that
produced it, or not at all. Now computed from `net_ret` and stored as
`net_ann_vol`.

**`max DD` sat on the gross line and has never been a gross drawdown.** It is
measured on `equity_USD`, which is after commission *and* after interest. It now
has its own line saying so, because it belongs to neither of the two above:

```
  gross    ret  10.77%   vol   8.22%   Sharpe 1.311
  costs          1.48% of NAV per year
  NET      ret   9.30%   vol   8.21%   Sharpe 1.132
  max DD          5.6% on equity (after costs, after interest)
```

The `--compare-spans` table is untouched and remains gross throughout, which is
consistent within itself — it compares spans, not cost treatments.

---

## 2026-08-29 — The Sharpe was already excess of IRX. The report never said so.

Raised as a critical bug: the Sharpe does not subtract the risk-free rate.
Measured, it does — but the report gave a reader no way to know that, and the
investigation was entirely reasonable given what was on screen.

```
series                               ann ret   ann vol   Sharpe
gross_ret (before costs)             10.77%     8.22%    1.311
net_ret  (after costs)  <- REPORTED    9.30%     8.21%    1.132
total_ret (net + interest)           12.92%     8.21%    1.572
total_ret - rf  (TEXTBOOK EXCESS)     9.27%     8.21%    1.129
```

`net_ret` is `(pnl - cost_lag)/base` and contains **no interest** — the cash leg
lives in `total_ret`. So the textbook construction, `total_ret - rf` day by day,
returns the same number: **1.129 against 1.132**. Subtracting IRX from `net_ret`
would subtract it twice and report ~0.68.

The 0.0037 residual is not noise either: it is `rf x (cost/NAV)`, the interest
not earned because commission left the account before the accrual. 0.0002 over
1990+, 0.0037 on a 2026 start — larger on the short window because the rate is
higher and the sample shorter.

### The real defect was the label, and it has been fixed

```
  NET      ret   9.30%   vol   8.21%   Sharpe 1.132   (excess of IRX)
  interest       3.62% of NAV per year, earned on cash and NOT in NET above
```

A pipeline that goes to the trouble of modelling the bill rate invites the
assumption that the cash leg is *in* the headline number. It is not, and now the
line says so and quantifies what was left out.

`verify_portfolio` asserts the property rather than the prose:
**Sharpe(net_ret) == Sharpe(total_ret - rf)**, tolerance scaled to the interest
effect so it cannot pass vacuously on a zero-rate window. Fault-injected both
ways, since a check that has never fired proves nothing:

```
net_ret wrongly INCLUDES interest    net 1.5724 vs 1.1286   gap 0.4438   FAIL
IRX subtracted TWICE                 net 0.6879 vs 1.1286   gap 0.4407   FAIL
```

*(The first attempt at that injection reported both as passing. It edited
`Portfolio.parquet` while `verify_portfolio` reads `Portfolio.csv` — the test was
broken, not the check. Worth recording: a fault injection that fails to fire is
evidence about the injection before it is evidence about the code.)*

### The number is still not worth quoting

1.132 over 171 sessions has a standard error of about **1.57** — it is 0.72
standard errors from zero, statistically indistinguishable from 0 and equally
from 2.5. No accounting treatment fixes that; only more data does. The 1990 run's
1.054 over 9,522 sessions carries an SE near 0.19 and is the figure to cite.

---

## 2026-08-29 — Open execution: the backtest now says what the ledger says

Stage 4 has always claimed an order fills at the OPEN of t+1. Stage 3 gave
`N[t]` the full close-to-close move `C[t+1] - C[t]`. Both defensible, not the
same convention, and the pipeline held both — so a live fill could never be
compared against the model without conflating slippage, convention and bug.

```
pnl[t+1] = N_prev . (O[t+1] - C[t])     the gap, on the position held overnight
         + N[t]   . (C[t+1] - O[t+1])   the session, on the one that just filled
```

`gap + day == C[t+1] - C[t]`, so this REPARTITIONS the same move rather than
adding to it. What it removes is the overnight gap being credited to a position
that had not been established.

### What it was worth, and it is not what I assumed

I expected a trend book to be systematically flattered by capturing the
overnight gap in its own direction. **Measured, it is a wash** -- and the sign
flips by decade:

```
1990-1999  +0.012%     2000-2009  +0.167%
2010-2019  -0.189%     2020-2026  -0.292%     whole history -0.056% of NAV/yr
```

So the reason for the change is not P&L. It is that a journal cannot be
reconciled against a model that describes a different execution.

### Headline effect

```
                 1990-01-02 .. 2026-08-28          2026-01-02 .. 2026-08-28
                close-to-close    open exec       close-to-close    open exec
gross ret            14.32%        14.33%             10.68%        11.11%
gross SR              1.220         1.219              1.302         1.354
cost/yr               1.92%         1.93%              1.49%         1.49%
NET ret              12.40%        12.40%              9.19%         9.62%
NET SR                1.056         1.055              1.122         1.174
max DD               17.6%         18.1%               5.6%          5.6%
equity            $21.57 B      $21.60 B         108,686,886   108,996,447
```

Nil over 37 years, +0.05 of Sharpe on 171 sessions -- which is sampling, not
signal.

### Five corrections on the way, four of them mine, three the same trap

Recorded in full because the pattern is the point.

**1. The estimate was wrong before the implementation was.** A static script put
the effect at +0.026 Sharpe; it did not forward-fill O and C, so every no-bar
row scored zero and the effect was understated.

**2. `_pnl_i` silently reproduced the OLD formula.** Shifting `N[:-1]` once for
both legs gives `N[j]` twice, so the instrument frames stayed close-to-close
while the portfolio loop moved. Reconciliation ties A and B caught it -- the
instrument sum stopped equalling the portfolio.

**3. `keep` was too narrow.** Session u's P&L now depends on `N[u-2]`, so a
session with a null signal and a flat position today can still have earned money
on a position held two sessions ago. Those rows were being dropped from
Positions.

**4. `N[u-2]` IS NOT "TWO SESSIONS BACK".** On the union grid a holiday row sits
between and N is carried across it, so `N[u-2]` returns the target decided at
the previous own session rather than the one before it. The overnight holder is
the target decided at that market's OWN previous session, now tracked
explicitly as `gapN`. Same union-grid trap as the 578 hidden rolls in stage 4
and the Panama knife edge in stage 2. **Third encounter, and the first two were
not enough to prevent the third.**

**5. The two legs did not cancel on a shut market.** O and C are forward-filled
independently, so a no-bar row repeats the last real session's open and close:
gap and day come out non-zero and equal-and-opposite. Under close-to-close that
was invisible -- `diff(ffill(C))` is 0 there and ONE position multiplied the
whole thing. Split across TWO positions they no longer cancel, and the
instrument booked phantom P&L on a day its market never opened. Both legs are
now zeroed on no-bar rows.

### Three structural changes it forced

- **Positions publishes `pnl_gap_USD` and `pnl_day_USD`.** On a roll the two
  legs belong to DIFFERENT delivery months -- the gap to the expiring one still
  held overnight. Stage 4, its verifier and the reconciliation all need that
  split; publishing it here gives it one definition instead of three
  re-derivations.
- **Stage 4's realised-P&L walk moved onto the FILL timeline.** A ROLL_OUT
  decided at k now realises at k+1, after the expiring month has earned its last
  overnight leg. Realising at the decision closed the contract before its final
  P&L arrived, and the per-contract attribution never tied.
- **`keep` now retains every session the market traded.** Three consumers depend
  on an own-session lag, and a lag can only be read off a file whose consecutive
  bar rows are consecutive sessions.

### Reswept

```
gate ablation           gross SR   max DD   cost/yr   net SR
shipped                    1.219    18.1%     1.93%    1.055
no s_g_vol                 1.228    24.0%     2.41%    1.071
no s_g_dd                  1.231    19.0%     2.20%    1.064
neither                    1.243    25.0%     2.75%    1.084

buffer      b=0.00  0.10   0.20   0.25   0.30   0.40
net SR       1.044  1.055  1.069  1.071  1.076  1.068
cost/yr      2.14%  1.93%  1.79%  1.73%  1.68%  1.60%
```

Both conclusions survive: the gates still buy volatility and drawdown rather
than edge, and the buffer plateau is still 0.20-0.30 with b=0.10 on the rising
part. The optimum has drifted from 0.25 to 0.30 and the gap from the shipped
value is now 0.021, against 0.014 before -- still not grounds to retune a paper
constant on an in-sample sweep, but closer to being one.

---

## 2026-08-29 — The interest base was one session out of date, and the check agreed with it

`interest[t+1]` accrued on `EQ` *after* `cost_m[t]` had been deducted. The
comment defending it read "the base is post-cost because the commission has
already left the account" — sound reasoning applied to the wrong session.
`cost_m[t]` is the cost DECIDED at *t*, and it fills at *t+1*'s **open**: the far
end of the very window the credit accrues over.

2026-01-05 makes it plain. The book was seeded at 100,000,000 on 01-02, held
nothing and traded nothing across a three-day weekend:

```
                        before           after
interest_base_USD  99,917,719.09   100,000,000.00
interest_USD           29,682.53        29,706.97
```

The 82,280.91 missing from the base is the commission on the book's FIRST fills,
which happen at Monday's open. The account was charged interest-forgone on money
it demonstrably still held every hour of the window.

One line: read the base before `EQ -= _c`. `interest_base_USD` now equals
`opening_equity_USD` on every accruing session in the history, which is what
"the interest received on day n depends on the NAV of day n-1" was always
supposed to mean. `bookkeeping.py` recovers the base by inverting
`interest / rate`, so it needed nothing but a corrected docstring.

### The part worth more than the money

The re-run failed reconciliation tie F — which recomputes interest from
IRX x balance — because F read `(eq[k] - cost[k])`, **the same false idea stage
3 held**. F had been passing at 1e-6 for as long as both sides were wrong
together. A tie only catches a mistake the two sides do not share, which is the
case for recomputing from primary sources rather than from the other side's
intermediate. Written up at F in `reconcile.py`.

### Effect

```
final equity   108,996,447.02 -> 108,996,607.08    +160.06
2026 interest, after the fix:      2,608,934.83
```

Sharpe 1.1737, vol 8.19%, max DD -5.60%, arithmetic 9.62%, geometric 13.77% --
every ratio unchanged at displayed precision. 171 sessions is not long enough
for $160 to move one. It is long enough for the number to be wrong.

---

## 2026-08-29 — Per-instrument commission loses a charge when the market is shut

Found while building the attribution page, which needs commission per session.
`sum(cost_lag_USD)` over the 63 Positions files equals the statement's
`commission_USD` on 170 of 171 sessions in 2026, and is short by $127.50 on
2026-01-19.

It is 6N. It went flat on 01-16; that order fills at 6N's OWN next open, which is
01-20, because the 19th is a US holiday 6N does not trade. `cost_lag_USD` shifts
by one row of the UNION GRID, so on 01-20 it reads the 19th's zero instead of the
16th's 127.50 and the charge vanishes. The statement books it on the 19th — a
session 6N has no bar for — so NAV paid it and only the per-instrument
attribution is short.

The union-grid trap again, in the fifth place it has appeared. $127.50 against
$1,061,404 of 2026 commission is 0.012% and moves nothing, but the shape is the
one that hid 578 rolls and lost $1.74B elsewhere: a lag taken on the panel rather
than on the instrument's own sessions.

**Not fixed.** The publisher takes the commission total from the statement, the
series NAV is built on, so the site reconciles exactly and cannot inherit this.
The fix belongs in `cost_lag_USD` — the same own-session lag `gapN` already uses
— and is left open deliberately rather than done in passing while an unrelated
page was being wired.

---

## 2026-08-29 — `cost_ann` had two definitions; stage 3 now owns the only one

Stage 3 computed `mean(gross_ret - net_ret) x 256`. Stage 6 computed
`mean(cost[t] / NAV[t]) x 256`. The same quantity over two different bases:

```
NAV[t-1]  (stage 3)   1.493846%
NAV[t]    (stage 6)   1.492243%
```

0.0016pp apart, identical at the two decimals the site prints, which is why it
sat there unnoticed. That is the third time in this pipeline that one quantity
computed twice has diverged quietly — after reconciliation tie F, which agreed
with stage 3 for as long as both encoded the same wrong interest base, and the
summary box, which compared an arithmetic rate against a geometric one on a
different basis. The pattern is not that the arithmetic is hard. It is that a
second implementation is a second thing to keep true.

`NAV[t-1]` is the right base, and not by preference: it is what `gross_ret` and
`net_ret` already divide by, so the decomposition closes exactly,

```
mean(gross_ret) x 256   11.10991294%
minus cost_ann           1.49384567%
                       = 9.61606727%
net_ann_ret              9.61606727%      residual 1.4e-17
```

by linearity. Under `NAV[t]` it stops closing, and a reconciliation the reader
can do by eye stops working for a reason nothing on the page explains.

Lifted into `cost_ann()` beside `tb_days()`, called by `stats()` and by
`publish.py`, which loads stage 3 for it. **No check was added and none is
wanted here** — a check that the two agree would be a check that the same
function equals itself. Removing the second implementation is strictly better
than testing it.

Published figure moves 1.49% → 1.49%. Nothing else moves; the pipeline is
17/17, 16/16, 15/15, 24/24, 28/28, 7/7, 11/11.

---

## 2026-08-29 — Four markets flat at once is normal. Which four is the finding.

Asked whether holding no position in several instruments simultaneously is
suspicious for a continuous system. The count is not; the composition is.

```
flat instruments per session, 2026 (63 markets, 171 sessions)
  0 flat   29 sessions        4 flat   27 sessions
  1 flat   25 sessions        5 flat    8 sessions
  2 flat   52 sessions        6 flat    1 session
  3 flat   28 sessions        7 flat    1 session
                              mean 2.19 of 63
```

So the four on 2026-08-28 sit in the middle of the distribution. All four were
`tradable` and `sized`; none was gated out. THREE DIFFERENT MECHANISMS produce a
flat, and they mean different things:

```
SI   N_raw -0.049   SIGNAL -0.120    no forecast — correct to be flat
ZC   N_raw +0.135   SIGNAL +0.013    no forecast — correct to be flat
PA   N_raw +0.554   SIGNAL +0.488    asked for half a contract, truncated
ZL   N_raw -0.927   SIGNAL -0.170    asked for most of one, truncated
```

and a fourth, elsewhere: `ETH` is flat on 57 of 169 sessions with `tradable`
False on every one of them — the five-year rule, working as designed, and not
truncation at all. It contributes 0 truncated asks.

### Silver never reaches one contract

Of 374 flat cells in 2026, **174 (46.5%) were an ask of half a contract or more
lost to rounding toward zero**. It is not spread evenly:

```
        flat/sessions        of those, >=0.5-contract asks
  SI    104/167  62.3%                63
  ETH    57/169  33.7%                 0   (not tradable, not truncation)
  PA     32/169  18.9%                22
  NIY    15/170   8.8%                 9
```

Silver is absent from the book **62% of the time**, and on those days `|N_raw|`
has median 0.600 and **maximum 0.987** — it never once crosses 1. When it is
held the position is a median of 2 contracts. Silver lives on the granularity
floor: one contract is too large a risk unit for the weight 3.32 gives it at
$100M, so a weak forecast that should be a small position becomes no position.

**This is the documented truncation rule doing exactly what it says** — rounding
toward zero, never taking more risk than the formula asked for — and the
direction is the safe one. But the journal's existing note that only 166
positions truncate across the whole history understates what it does to
individual markets: measured per instrument rather than per book, it removes one
market from the portfolio for most of the year.

Not fixed, and not obviously fixable without changing the rule: rounding SI's
0.6 up to 1 would take 66% more risk than 3.32 asked for in that market. The
honest options are a larger NAV, a micro contract where one exists, or accepting
that the book is 62 markets plus silver-when-it-matters. Recorded so the
diversification claim is read with it.

---

## 2026-08-29 — The cost assumptions, audited and revised upward

`total_avg_cost_rt_LocalCurrency` is not a fee schedule. Decomposed, it is a
two-term model, and the formula held for 55 of 63 contracts:

```
cost_rt = one full tick value + 5 units of local currency
          \_ spread: one tick   \_ fees: commission + exchange + clearing
             crossed round trip
```

Which term dominates matters, because only one of them is checkable against
published rates:

```
2026 commission $1,061,404
  tick term (spread)        $629,742   59.3%
  residual (fees + broker)  $431,662   40.7%
```

### What was wrong, in two tiers

**Tier 1 — the fee term sat below any real schedule.** The residual is a flat
FIVE LOCAL UNITS, so its dollar value moves with FX: US$5.00 but A$5 = $3.58,
C$5 = $3.59, and ¥500 = $3.12. Against published rates a competitive broker on
a US venue pays roughly $1.18-1.45 exchange and clearing, $0.02 NFA and ~$0.85
commission per side — about $4.10-4.70 round trip. MX, ASX, SGX and HKEX are not
cheaper than CME, so five contracts were below any defensible floor. Arithmetic,
not judgement.

**Tier 2 — the spread term assumed one tick on books that do not quote one tick
wide.** One tick round trip is the correct assumption for a market order in a
one-tick market, which is true of ES, NQ, ZN, GC, CL and 6E. It is not true of
lean hogs, live cattle, palladium, the CME crypto contracts, gasoil, VIX back
months or the TSX 60. Eleven contracts were sitting at 1.1-1.5 ticks.

```
tier 1   YXT4  7.50 -> 9.50    CGB 15 -> 17    SJB 1500 -> 1800
         YAP4  30 -> 32        HSI 80 -> 89
tier 2   VX  55 -> 80     HE 15 -> 25     LE 15 -> 25    PA 25 -> 35
         EMD 25 -> 35     ETH 30 -> 55    BTC 35 -> 60   GAS 30 -> 55
         KC  23.75 -> 42.50    NIY 3500 -> 6000    SXF 25 -> 47
```

### The direction of the error is the point

**Every one of the sixteen moves UP.** Nothing was revised down. Where the
estimate is uncertain the model now takes the expensive side, deliberately and
on the record.

**A correction to how "generous" was judged.** The first pass ranked contracts
by cost expressed in TICKS and called RS (6.5), DX (6.0) and GF (2.4) generous
on that basis. Ticks are the wrong unit for a cross-contract comparison, because
the tick itself varies by two orders of magnitude across this book, and the
ranking says more about tick size than about cost. Decomposed properly:

```
       cost = tick + fee        fee in USD   verdict
DX      6.00 = 1.00 + 5.00         $5.00     STANDARD, not generous. It looked
                                             like 6 ticks only because the
                                             Dollar Index tick is $1.00.
RS     13.00 = 2.00 + 11.00        $7.91     modestly generous, on the fee term
GF     30.00 = 12.50 + 17.50      $17.50     2.4 ticks all-in. Feeder cattle is
                                             among the thinnest CME contracts,
                                             where 2-3 ticks is the plausible
                                             band -- so this is at the BOTTOM of
                                             realistic, not above it.
```

So only RS is clearly above what the evidence supports, by about $3 of fee on a
contract carrying 1.2% of commission.

**GF was then revised too, making seventeen.** It belonged in tier 2 on the
standard applied to lean hogs and live cattle: both were raised to 2.5 ticks and
feeder cattle is thinner than either, so leaving it at 2.4 while raising them was
an inconsistency a reader would find. `30.00 -> 42.50`, three ticks plus the fee
floor. It carries 0.30% of commission and the change is $1,425 — immaterial to
every published figure at two decimals, which is precisely why there was no
argument for leaving it.

The lesson is the unit, not the numbers: **compare cost against (spread x tick +
fee), never against tick count.**

That is the whole claim being made here, and it is worth stating precisely
because a larger one is tempting and would be false. **It is not "our costs are
overstated."** That would need the spread term to be an upper bound, which it is
not for the illiquid tail, and market impact to be zero, which it is not at a
book that breaches open interest under compounding. The defensible claim is
narrower: *the inputs we cannot measure are set on the conservative side, and
every revision has moved them further that way.*

### What it costs, measured after the rerun

The projection was $1,125,628; the pipeline produced $1,125,459. The small gap
is real and worth naming: raising costs lowers equity, equity is the sizing
base, so the book carries marginally fewer contracts and pays marginally less
commission than a static rescaling predicts. Costs feed back into position size.

```
                        before      16 revisions    +GF (17)
commission        $1,061,403.71  $1,125,459.12  $1,126,894.12
cost drag             1.49%/yr       1.58%/yr       1.59%/yr
net asset value $108,996,607.08 $108,948,974.85 $108,947,731.15
ann. arithmetic          9.62%          9.55%          9.55%
ann. geometric           9.72%          9.65%          9.65%
volatility               8.19%          8.20%          8.20%
Sharpe                  1.1737         1.1653         1.1651
max drawdown            -5.60%         -5.60%         -5.60%
```

Six percent more commission for 0.10pp of annual drag and 0.009 of Sharpe. The
revision changes no conclusion, which is itself the finding: the thin-contract
assumptions are not load-bearing, so being generous with them costs almost
nothing and removes the argument entirely. Pipeline green throughout:
17/17, 16/16, 15/15, 24/24, 28/28, 7/7, 11/11.

### What would actually settle it

Quoted bid-ask by contract, which this project does not have. Tier 1 is defended
by published fee schedules. **Tier 2 is judgement** — an informed guess at which
books quote wider than a tick and by how much — and the paper should say so in
those words rather than implying a measurement. An assumption stated as an
assumption is not a weakness; one implied to be a measurement is.

---

## 2026-08-30 — Commission waits for the market to open

`cost_lag_USD` was `cost_USD` shifted one row of the UNION GRID. A cost decided
at t fills at that instrument's OWN next open, and the two coincide only when
t+1 is a session for that instrument. When it is not:

- NAV paid the charge on a day the market was shut, and
- the per-instrument column lost it outright, because `keep` drops the row it
  landed on.

6N's $127.50 on 2026-01-19 was the case that exposed it. Fifth appearance of the
union-grid trap, and the same own-session fix `gapN` already uses for positions:
a decided cost is held in `_cost_pending` until `has_bar` says the instrument
trades, then charged in full.

```
6N        cost decided   charged
01-16          127.50     232.50
01-19        (no session -- nothing charged, nothing lost)
01-20           67.50     127.50   <- arrives at 6N's own next open
```

```
per-instrument cost vs statement, 2026   before 170/171 sessions   after 171/171
2026 total, per-instrument               1,061,276.21              1,126,894.12
2026 total, statement                    1,061,403.71              1,126,894.12
```

Equity moved $4. The point was never the money.

### The check that asserted the bug

`verify_portfolio` held "cost_lag_USD is cost_USD shifted one session" — which
was the defect, written down as a requirement. It failed on the fix, correctly,
and had to be replaced rather than relaxed. What survives is stronger:
**commission charged never precedes commission decided**, and **every dollar
decided is charged or still pending**, where the pending remainder is exactly
the last session's decision, which fills after the window ends. Those hold under
any lag convention; the old one only held under the wrong one.

Stage 3 is 25/25. Reconciliation ties C, D and E unchanged at 1e-16.

---

## Conventions and decisions

- **`NAV_0 = 100_000_000`**, the paper's figure, overridable with `--nav`. Not
  cosmetic: sizing is linear in `E_t`, and truncation deletes an instrument
  outright when its allocation is under one contract. At $1M that silently
  removes **34 of 63 markets** — systematically the large-notional ones (index
  futures, metals, crypto), leaving a rates-and-softs book that still reports
  itself as diversified. Every instrument clears one contract by about $25M. At
  $100M only 166 positions in the whole history truncate to zero.
- **Rounding is toward zero, always.** `floor(-2.7) = -3` would *increase* a
  short; `trunc` gives `-2`. A rounding rule must never take more risk than the
  formula asked for. Asserted in `verify_portfolio`.
- **`sized` separates a decision from a carry.** On a session where a market was
  shut, 3.32 was not evaluated and `N_raw` is null rather than 0 — 0 would claim
  the formula asked for a flat position when it was never asked.
- **The correlation matrix is never materialised.** 3.33 needs only `w'Cw`, a
  scalar; the `(T,63,63)` array would cost 399 MB for it. Pairs are accumulated
  into a running sum and discarded, and only the `i<j` half is walked since the
  diagonal contributes exactly N.
- **`_recursive_sum` now runs through `scipy.signal.lfilter`** — the identical
  recursion in C, verified **bit-identical** (max abs diff 0.0 over 12,552
  values, and all 63 books unchanged after the swap). 19x faster batched, which
  is what makes 2,016 pairs affordable: 16.5s → 0.8s.
- **`s_g_vol` multiplies the contract size; `s_g_dd` and the buffer act on
  `N_contracts`.** None of the three touches `SIGNAL`, so a forecast means the
  same thing on a gated day as on an open one.
- **P&L is on OPEN EXECUTION**: the overnight gap belongs to the position held
  before the fill, the session after it to the one that just filled. The gap
  holder is the target decided at that market's OWN previous session, never
  `N[t-1]` on the union grid.
- **The reported Sharpe is EXCESS OF IRX.** `net_ret` carries no interest, so
  it is already an excess return; the cash leg is in `total_ret`. Subtracting
  IRX from it again is the easy mistake and is now blocked by a check.
- **Cost inputs are set on the conservative side where they cannot be
  measured, and every revision has moved them up.** `cost_rt` is one tick
  (spread) plus a fee floor of about $5 round trip in every currency. See
  the 2026-08-29 audit: 17 contracts revised upward, none downward.
- **Judge a cost against `spread x tick + fee`, never against tick count.**
  Tick values span two orders of magnitude across this book, so a tick-multiple
  ranking measures tick size, not cost. It is what made DX look generous when it
  is exactly standard.
- **Commission charges both legs of a roll.** `|N[t-1]| + |N[t]|` on a session
  where the held contract changed, `|N[t] - N[t-1]|` otherwise, one-way in both
  branches. The roll test is on forward-filled symbols, never adjacent rows.
- **`cost_ann` is defined once, in `portfolio.py`, and every other stage
  calls it.** `mean(gross_ret - net_ret) x 256`, so the base is `NAV[t-1]`
  like every other rate and `gross - cost == net` closes exactly.
- **`cost_lag_USD` is what was CHARGED, not a shift of what was decided.**
  A cost is held until the instrument's own next open and charged there,
  so nothing is billed on a day a market was shut and nothing is lost.
- **`cost_lag_USD`, not `cost_USD`, enters P&L.** The cost of a decision made at
  close *t* is borne when the trade executes. Both columns are written.
- **Interest is credited at t+1 on 100% of NAV, accrued on calendar days.**
  The base is last night's CLOSING equity with nothing subtracted. The
  session's own commission leaves at the next open, at the far end of the
  window the credit accrues over; deducting it charges the book for cash it
  still held all night.
- **The window is 1990-01-01 with a five-year tradability rule, and both live
  only in `portfolio.py`.** Burn-in and every computation continue on an
  untradable instrument; only the position is withheld.
- **An undefined correlation between two active markets counts as 1.0**, the
  conservative direction — it lowers IDM and undersizes, where 0 would claim
  diversification that has not been measured. Counted every run, and it has
  never fired: **0 undefined pairs** across the whole history.

## Not yet modelled

- **Contract granularity removes individual markets, not just odd positions.**
  See the 2026-08-29 entry: silver is flat 62% of 2026 because `|N_raw|` never
  reaches 1. The whole-book count (166 truncations in the history) hides it.
- **The 1% volatility floor** proposed to cap STIR leverage in the ZIRP years.
  Postponed by decision, not by oversight; the case for it weakened once the
  gates cut STIR to 1.73x and 25.0% of the book.
- **Carver's early-roll turnover trick** — worth 0.325%/yr, about 16% of
  execution cost. The literal rule does not apply: 99.8% of the cases here are
  *growth* of an existing position rather than opening from flat, so it needs a
  conditional early roll or a suppression rule, not the rule as written.
- **Slippage and market impact.** Only commission is modelled. Cost is priced
  from `total_avg_cost_rt_LocalCurrency`, a round-trip commission estimate; the
  1.93%/yr figure is therefore a floor on true execution cost, not an estimate
  of it.
- **A realised-volatility target.** The gates hit a nominal tau, and the
  ablation shows they de-lever without improving gross Sharpe. Running ungated
  at a lower tau might reach the same risk profile more cheaply. Not measured.
