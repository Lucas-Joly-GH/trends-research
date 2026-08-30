# Publication Journal

*Opened 2026-08-29.*

The durable record behind `publish.py` and `docs/` — stage 6, the only stage that
puts anything in front of a reader. Same convention as the other journals: every
claim names the number that produced it.

---

## 2026-08-29 — The carve-out, and why prices are published on purpose

`trends-research` is public and the whole pipeline is built on the rule that
vendor data never reaches it. This is the one deliberate exception, and it is
narrow: **2026 sessions only, open and close only, non-commercial, and the data
provider is never named anywhere on the site.**

The reason to publish prices at all is that the alternative is worse. A results
table reading `SELL 37 6B` cannot be checked by anybody, and an unverifiable
results page is worth less than no page. The reader has to be able to recompute a
session by hand:

```
N x (close - open) x pointsize x FX  =  the P&L we claim
```

which needs the quantity, both prices, and the contract spec. All three are
either ours or exchange-published, and `instrument_mapping.csv` is in the repo so
the third is one click away.

### The guard is a whitelist and it aborts

`_guard()` runs on the **finished rows**, not on the frame — a column renamed
between the check and the writer would slip past a check made upstream. Anything
not named in `EXECUTED_COLS`, `PENDING_COLS`, `GIVEN_COLS`, `OUTSTANDING_COLS`,
`DAILY_COLS`, `PNL_COLS`, `BOOK_COLS`, `INDEX_COLS` or `META_KEYS` stops the
build.

The direction is the point. A blacklist ("drop `SIGNAL`") fails open the day
someone adds a column upstream, and this pipeline adds columns constantly —
`pnl_gap_USD` and `pnl_day_USD` appeared the same afternoon open execution
landed. A whitelist fails closed: a new column is simply not published until
someone comes here and names it.

`_dates_guard()` is an **abort, not a filter**. A filter would quietly publish a
truncated series if someone moved `WINDOW_START`; an abort makes them come here
and mean it.

Never published, though it is all right there in the frames: the Panama-adjusted
series (`Continuous_C`/`Continuous_O`), the risk estimate
(`price_vol_USD_ann`), the forecast (`SIGNAL`), the gates, `IDM`, `w_i`, and
everything else in the journal's provenance block. Those are the model, not the
record, and the model is what the paper is for.

---

## 2026-08-29 — Three static pages, no build step, and the bug my preview pane hid

`docs/index.html`, `docs/journal.html`, `docs/pnl.html` and a shared `app.js`.
No framework, no bundler, no dependency. GitHub Pages serves `/docs` on `main`
and the whole site is 2,031 KB of JSON plus four hand-written files.

Deliberately not a charting library: the series is 171 points, the page is meant
to read as a printed report, and a dependency that renders one polyline is a
dependency that can break the page.

### Three things that were wrong for reasons worth recording

**Locale is pinned, not inherited.** A French browser renders `1.35300` as
`1,35300` and `$108,996,447` as `108 996 447` — correct locally, ambiguous to
everyone else, and a comma decimal separator on a price table turns a checkable
number into an argument. `L = "en-US"`, everywhere.

**Alignment belongs to the column, not to its position.** It was set by
`:first-child`/`:nth-child(2)`, which suited Instrument and Contract and left
Side and Why right-aligned like numbers. Positional rules also cannot be right
for both tables at once — Pending has six columns and Executed nine, in
different orders. `l: true` on the column definition; everything else is numeric
and stays right.

**`margin:0` with a `max-width` pins the page to the left edge.** On a 1920px
window the content sat in the left half and the rest was empty. It survived a
long time because I was checking in an 863px preview pane, which the body filled
exactly — the defect was invisible at the only width I looked at. It took the
user's screenshots to settle it, after I had said the page was fine. **Screenshot
evidence from a viewport that is not the reader's viewport is not evidence.**
Every layout claim in this file since is a DOM measurement at a named width.

---

## 2026-08-29 — Using the width: measured charts, and why the tables are paired

The shell was a hard 1060px, which centred on a 1920px monitor leaves ~430px dead
on each side. Now `max-width:min(1560px, 95vw)` — the cap holds the line length
on a wide screen and yields to the viewport below it, so no breakpoint is needed.
Prose does not stretch with it: `p` still caps at 74ch and the left column is
clamped, so the gain goes to the charts and the tables.

**Chart width is measured, not assumed.** A fixed 960-unit viewBox is scaled to
fit whatever box it lands in and the axis text scales with it: in a 630px column
the 10px labels would have rendered at 6.6px. `lineChart` now takes `W` from the
element's own pixel width, so one viewBox unit is one CSS pixel and 10 means 10
wherever the chart sits. That makes `W` true for exactly one width, hence the
debounced `resize` redraw.

**Height tracks width** for the same reason: a constant height is one aspect
ratio pretending to be a design. The 170px drawdown panel reads fine at 630px and
becomes a 6.4:1 sliver at 1090px. `{r, min, max}` — the min reproduces the old
height, the ratio takes over above it, the max stops a wide monitor producing two
tall panels.

```
viewport   layout            panes   executed table   NAV chart
1920       split + paired    763     fits 763         1091 x 295
1440       split + paired    667     fits 667          953 x 260
1350       split, stacked    —       capped 1150      804 x 260
 700       stacked           —       fits 656         656 x 260
```

### The NAV chart takes the leftover column height

The prose and the summary ran 79px longer than the two charts beside them at
1920px. A taller chart was the obvious fix and a fixed ratio was the obvious way
to get one — and it would have been right at exactly one width. The slack is not
a constant: it GROWS as the viewport narrows, because a narrower left column
wraps to more lines while the charts beside it get shorter.

```
viewport   left col   NAV height   gap after
1920         713         374          0
1440         781         457          0
 700       stacked       260 (base)   n/a
```

So the chart asks how much room is left instead. Measuring is not circular here:
the columns are independent flex items under `align-items:flex-start`, so growing
the right one cannot change the height of the left. One pass, no loop.

Two things it has to get right. It **re-measures from the un-donated height**,
or each resize compounds the last — 1920 → 1440 → 1920 returns to 374, checked.
And it runs only when `.split` is actually `flex`; stacked, there is nothing to
fill and the chart keeps its base height.

The first version overshot to 414 and left the right column 40px too TALL,
because it measured before the two `figcaption`s were filled — they are the
thick end of 40px of that column. The summary and the charts also arrive on
different fetches, so whichever lands second now does the balancing.

### The tables: one wrong fix before the right one

Six and nine *short* columns stretched over 1560px grow every gap rather than
getting more readable — `Contract` ended up with a 296px column to hold `6B`.

First attempt made one column absorb the surplus (`grow` on Why) so the rest
would hug their content. It anchors both edges and **just relocates the hole**:
Why became 1230px wide and left a 1000px void mid-table. Reverted.

The right answer is that the tables do not need 1560px at all, and the width is
better spent putting them side by side. Pending and Executed pair above 1440px
(1500px on the journal, where the date picker takes 173px off the top first), and
a 1150px cap holds them at the same density in the band between. On the journal
page the pairing earns its keep twice: given and executed for one session is the
comparison that page exists to make.

---

## 2026-08-29 — The attribution page, and a balance sheet that has to add up

`Daily P&L`: every instrument's gross for a session, then Total Commissions,
Interests, and the three totals, beside a walk from opening equity to closing
equity.

It reconciles to the published NAV rather than approximating it:

```
sum(instrument pnl_USD) == statement gross_pnl_USD      to 7e-10
NAV(positions)          == closing_equity(statement)    to 6e-08
```

`build_pnl()` **aborts** on any session where the instrument rows do not sum to
the statement's gross. A balance sheet that does not add up is worthless, so it
is checked rather than trusted.

### Commission comes from the statement, deliberately

The per-instrument `cost_lag_USD` series drops a charge when an instrument's own
next session is not the union grid's next row — $127.50 on 2026-01-19, written up
in `Portfolio_Journal.md`. NAV paid it; only the attribution is short. Taking the
total from the series NAV is built on means the page reconciles exactly and
cannot inherit the defect.

### Cents, not whole dollars

The first pass rounded each line to the dollar and the sheet printed
`499,064 - 4,453 = 494,610`. Visibly wrong arithmetic, on the one page that
exists to be checked.

```
displayed at        instrument column     the walk
whole dollars       up to $5 out          a clean $1 break
two decimals        up to $0.03 out       up to $0.02 out
```

At the cent every check a reader would run ties exactly. The footer states the
rounding rather than hoping nobody adds the column up.

### Three kinds of zero, and only one of them is a holiday

A `0.00` was doing three jobs. Across the 2026 window, 782 zero rows:

```
312  39.9%  market shut — no session that day
254  32.5%  flat — no position to make anything
216  27.6%  held a position, market open, still zero
              86  price never moved
              69  gap and day legs cancelled exactly
              61  the seed session, 2026-01-02
```

Each instrument row now carries `session`, taken from that instrument's **own**
book dates — the union grid carries a row for it either way, which is exactly the
trap. A shut market renders `—`; `0.00` is reserved for a market that traded. The
heading counts markets that traded: `2026-01-19 — 20 markets traded, 39 shut`,
and the 39 are precisely the US-listed set.

The cancellation case is the one no reader could infer: on 2026-05-20 a 6C
position lost $1,470 on the overnight gap and made the same $1,470 back by the
close. Zero there means round trip, not quiet session. It is in the footer.

Only 12 instrument-days are absent from the sheet entirely in all of 2026, and
the rule is clean: the market was shut in all 12 and the book held a position in
none. That is the whole of the 59-to-63 row-count variation.

---

## 2026-08-29 — `Realised P&L $`, after two worse names

The column was `Realised $`, briefly `PnL $`, now `Realised P&L $`.

`PnL $` read as the position's move for the session, which it is not: the number
is proportional crystallisation, so a trade that adds to a position banks
nothing, one that cuts it by half banks half of what the position had accrued,
and a close banks the rest. An untouched winner shows 0 all year. `Realised` is
the word that carries that.

Rejected: `Crystallised $`, precisely the mechanism and the term `bookkeeping.py`
uses, but desk jargon — a reader who knows it does not need the header, one who
does not is no better off. `Closed P&L $` is simply wrong: a RESIZE that only
cuts a position also books a fraction here.

Verified against the ledger: `realised == 0` if and only if the trade did not
reduce the position. 2,275 trades that grew or opened a position, 2,275 zeros;
2,030 that reduced, zero zeros. No exceptions either way.

---

## 2026-08-29 — Two rates side by side on different bases, which reads as a lie

The summary showed `Return, annualised (arithmetic) 9.62%` above
`Return, annualised (geometric) 13.77%`, and a note saying interest "is not
counted in the returns above". The note was true of one line out of three.

```
                                shown     interest in it?
Total return, 171 sessions      9.00%     YES  (from the equity curve)
Return, annualised (arithmetic) 9.62%     no   (mean(net_ret) x 256)
Return, annualised (geometric) 13.77%     YES  (equity growth rate)
```

`net_ret` excludes interest by construction — `interest[t]` enters exactly one
series, `total_ret`, and never the numerator of `net_ret`. `net_cagr` was the
EQUITY curve's growth rate, and equity accrues interest every session. So the two
annualised lines were never comparable, and placed adjacent they invited the
reading that compounding accounts for a 4.15pp gap.

It does not. Of that gap, **3.62pp was the cash leg present in one number and
absent from the other**. Like for like, compounding `net_ret` itself:

```
arithmetic, trading only    9.62%
geometric,  trading only    9.72%      the honest gap: 0.11pp
geometric,  equity curve   13.77%      what was shown
```

`cagr_trading` compounds the same series the arithmetic mean and the Sharpe use.
`net_cagr` stays in the payload — the equity growth rate is a real number a
reader may want — but it is no longer the one displayed beside a trading-only
rate.

The window return keeps its interest, because 9.00% is what the money did, and is
now split into the two sources beneath it:

```
Total return, 171 sessions   9.00%
   from trading              6.39%     6,387,672.26
   from interest on cash     2.61%     2,608,934.83
```

Dollars over the opening balance rather than compounded returns, so the two parts
add to the total exactly on screen — compounded parts do not.

### The commission line stated a rate as if it were a fee

Same box, same day: `Commission 1.49% of capital a year`. `cost_ann` is
`mean(cost[t] / NAV[t]) x 256` — a daily mean annualised the same way every
other rate here is, so it means *at this trading pace, commission costs about
1.49% of the account per year*. Three things were wrong with saying it that way.
"Capital" names no base (it is NAV, which moved, averaging 107,865,403 over the
window). A percentage "of capital a year" reads as a fee levied on the account
rather than a by-product of turnover. And nothing said it was **already
deducted**, which it is — from every figure above it.

What was actually paid is a different number, and the one a reader wants:

```
gross trading P&L    7,449,075.97     7.4491%
commission          -1,061,403.71    -1.0614%
net trading P&L      6,387,672.26     6.3877%   the "from trading" line
```

1.06% not 1.49%, because the window is 0.668 of a year. Both are now published
— `cost_window_USD` beside `cost_ann` — so the rate cannot be read as an
amount.

Found by being asked what the return with interest is. The figure had been on the
page for a day and I had described the arithmetic-geometric gap as a
part-year compounding effect twice without checking that the two lines were
measuring the same thing.

---

## 2026-08-29 — The cache version, stamped by the publisher

GitHub Pages serves static assets with a ten-minute cache, and neither `app.js`
nor any of the JSON ever changes name. So a reader who visited earlier can get
new HTML against a stale script — a page that looks updated and behaves like the
old one. It is not hypothetical: it happened locally on the NAV-height change,
where new markup ran against a cached `app.js` that knew nothing about donated
chart height, and the symptom read as a layout bug rather than a caching one.

`publish.py` now stamps `<script src="app.js?v=HASH">` into all three pages, and
`app.js` reads the value back off its own `document.currentScript.src` and
exposes `bust()`. Every data fetch goes through it, so one stamp on one tag
versions the script and all six JSON endpoints together. A page opened straight
off disk has no stamp and `bust()` degrades to the plain URL.

### The stamp is content, not a counter, and excludes `generated_at`

`sha256(app.js + latest.json)[:8]`, with `meta.generated_at` removed first.
Nobody has to remember to bump it, and it moves when EITHER the script or the
data changes — which matters, because the dangerous staleness is new data
against an old script, not either alone.

Dropping `generated_at` is the part worth recording. It is wall-clock, so
leaving it in would change the stamp on every run, rewrite three unchanged HTML
files, and re-fetch everything for readers whose cached copy was already
correct. **A cache version that always changes carries as much information as
one that never does.** Verified:

```
publish, publish again      7bc6a052 -> 7bc6a052   "already current in all pages"
edit app.js                          -> a7f8ffde   3 pages rewritten
revert app.js                        -> 7bc6a052   back to the same value
```

Republishing identical data is now a byte-level no-op.

### `_pages_guard()` refuses to publish a page it cannot stamp

Same reasoning as the column whitelist: a page whose tag has been edited into a
shape the stamper does not recognise would keep serving whatever version a
reader had cached, and fail as a layout bug somewhere else entirely. Checked
before anything is written, so `--check` catches it too.

---

## 2026-08-29 — "Generated" told the reader about my process, not the numbers

The footer said `Generated 2026-08-29 19:02:35 UTC` — `datetime.now()` at
publish time. Re-running the publisher on week-old data would have refreshed
that line and told a reader the figures were current when nothing about them
was. It measured when a script ran, which is a fact about my process; the reader
wants to know when the NUMBERS were last rebuilt.

`Update.py` now stamps `Live/.pipeline_run.json` at the end of every run with
`completed_at`, the `failures` count and whether it was a full run. `publish.py`
reads it into `meta.updated_at`, and the line reads **Updated**.

```
updated_at    2026-08-29T19:46:42+00:00     pipeline finished
generated_at  2026-08-29T19:47:03+00:00     publisher ran, 21s later
```

Both are kept in the payload because they are different facts and the gap
between them is exactly the thing the old label hid. Only `updated_at` is shown.

### The stamp is written on failure too, and stage 6 refuses it

Recording only successful runs looked tidier and is worse: a failed run has
already overwritten the parquet files, so the previous success's timestamp would
stand over data it never produced. So the stamp is written either way with the
count, and `run_stamp()` aborts on two conditions — a run that ended with
verification failures, and a partial run where a `--no-*` flag skipped a stage.
Published figures come from every stage, so they have to be rebuilt by every
stage. Both tested.

`updated_at` is deliberately INSIDE the cache-version hash where `generated_at`
is not: it moves only when the data is genuinely rebuilt, which is exactly when
a reader's cached copy should be replaced.

---

## 2026-08-29 — Hover readout on both charts, still no library

Reading a value off a 171-point line by eye is guesswork — the axis offers four
gridlines and the reader wants a session. `pointermove` on the SVG snaps to the
nearest index and draws a dashed crosshair, a dot on the series, and one line of
text: `2026-06-19   $112,143,262`.

Still no dependency. The chart was already inline SVG, so this is a `<g class="hv">`
the handler refills and roughly thirty lines. A charting library would have been
a larger change than the feature.

### Three things it has to get right

**The axis formatter is not the hover formatter.** `$115.6m` is right for four
gridlines and useless for a readout, so `lineChart` takes a separate `fmtExact`
— `$108,996,607` on NAV, two decimals on the drawdown. Reporting the axis
rounding as if it were the value would defeat the point of hovering.

**State lives on the element, not in a closure.** Every redraw rebuilds the
geometry, and a listener holding a stale `x()` would keep reporting confidently
after a resize while being wrong by however much the width changed — silent, and
the kind of thing nobody checks twice. `svg._hv` is replaced on each draw and the
handler reads it fresh. Bound once, via `svg._hvBound`, so redraws do not stack
listeners.

**Client pixels are mapped through the CURRENT box.** The viewBox is 1:1 with
CSS pixels only until the browser scales it in the window between a resize and
the debounced redraw. Reading `clientX` raw drifts in exactly that window.

Verified against `history.json` rather than by eye: probes at sessions 0, 40,
85, 100, 120, 165 and 170 all reproduce the published date and equity to the
dollar, at three viewport widths and across a resize in both directions.

The label flips to the inside half at the midpoint so it never runs off the
panel, and a white stroke under the glyphs (`paint-order: stroke`) keeps it
legible where it crosses the series or a gridline — no filled box, which would
have been the first bit of chrome on an otherwise flat page.

---

## 2026-08-29 — Ink and oxblood: one hue, and it has a job

The page was deliberately monochrome. Adding colour risked undoing the reason it
was: this site's whole argument is *here are the numbers, check them*, and a
coloured section header or a branded accent bar reads as marketing, which
quietly undercuts that. So the rule is that **colour carries information or it
does not appear**.

```
--page  #fdfdfc   barely off-white; pure #fff is the unstyled tell
--ink   #14213d   near-black navy: printed, not rendered
--muted #5a6472   notes, captions, footers
--faint #9aa1ad   a zero that is genuinely nothing
--neg   #9b2226   oxblood. The only hue that means anything
--rule  #c9ccd4
```

Exactly one hue means something, and it means "below zero". **Positives stay ink
on purpose**: colouring both makes a trading terminal, colouring only the
exceptions makes a report. And `--neg` is always redundant with a minus sign
already on screen, so it speeds up scanning without ever carrying the fact by
itself — nothing is lost to colour blindness.

### The palette is defined once

Three pages with three copies of six hex values is the duplication this pipeline
has now been bitten by three times (tie F, the summary basis, `cost_ann`). So
`site.css` holds the tokens and the chart classes, the pages carry no hex at
all, and `publish.py` stamps and hashes it exactly as it does `app.js` — a
palette change busts the cache like a script change.

Chart colour had to move with it: `stroke="var(--ink)"` is not valid in an SVG
presentation attribute, so `lineChart` emits classes and `site.css` resolves
them. The chart can no longer drift from the page it sits on.

### Negatives are detected from the rendered text

`isNeg()` tests the string the cell is about to show, not the number behind it.
A column added later gets the colour with no wiring, and a formatter that rounds
-0.4 to "0.00" stays honest — coloured if and only if a minus is visible. U+2014,
the em dash marking a shut market, deliberately does not match.

### The specificity trap, again

`.neg` is (0,1,0) and lost to `.picker td:nth-child(2)` at (0,2,1): the class
landed on all 77 losing sessions and the colour did not. Same failure as `td.l`
against `table.data td` earlier in this build. Fixed with a scoped
`.picker td.neg`, placed AFTER the column rule so it wins and BEFORE `tr.on` so
the selected row keeps its reversed text — with equal specificity, order is the
whole argument. `td.zero` at (0,1,1) still outranks `.neg`, which is deliberate:
a figure that rounds to nothing reads as nothing, not as a loss.

---

## 2026-08-29 — Dark ground, and green earns its place per column

The white page was tiring to read, so the site is dark. Two things keep that
from looking cheap, and neither is the hue:

**Not pure black, not pure white.** `--page` is `#0f141c`, a dark desaturated
navy, and `--ink` is `#d6dbe4`. Maximum contrast is what actually tires the eye
on a dark ground — white on black halates, and it is the single tell that
separates a considered dark theme from an inverted one.

**Structural rules are not drawn in text colour.** Every `1px solid var(--ink)`
became `var(--rule-strong)` (`#46526a`). A near-white 2px rule across a dark
page is a glare bar, not a divider. That was 22 declarations across the three
pages.

### Green is opt-in; red is not

Red marks any negative anywhere — a minus is always worth catching. Green
required a rule, because the obvious version is wrong: a column that can only be
positive learns nothing from being green, and colouring Qty, Fill open and
Commission is exactly how a report becomes a trading terminal.

So `pos` lands only where a column declares `signed: true`, meaning both signs
genuinely occur in it. What that produced, unprompted, on 2026-08-28:

```
P&L sheet          33 green   29 red    0 uncoloured
NAV walk           opening and closing equity PLAIN -- they are balances,
                   not movements; gross, commission and interest coloured
Journal executed    1 green    9 red   20 uncoloured
                   the 20 are zeros: OPEN and ROLL_IN realise nothing, and
                   nothing is not a gain
Date picker        94 green   77 red    -- 171 sessions scannable at a glance
Summary            5 green (the return rows), 1 red (drawdown),
                   3 plain: net asset value is a balance, volatility a
                   magnitude, Sharpe a ratio. None of the three is a gain.
Qty / prices        0 coloured
```

The last two lines are the rule working. Had green been global, the summary
would have read as five green numbers plus a green account balance, and every
price in two tables would have been green — which says nothing, and says it
loudly.

`--pos` (`#4cb782`) is blue-shifted rather than a pure green so it stays
separable from `--neg` under deuteranopia, and both remain redundant with a sign
already on screen.

### Both schemes, and the light one is not an inversion

`prefers-color-scheme` decides; dark is the default for a reader who has
expressed no preference. The structure is that every colour is defined once as
`--d-*` or `--l-*` and the blocks below do nothing but point the working tokens
at one set — so a scheme is a mapping, never a parallel stylesheet that drifts.
A role that reuses another role's colour aliases it (`--l-rule-strong:
var(--l-ink)`) rather than repeating the literal, which states the decision
"strong rules are drawn in ink" instead of hiding it in a duplicated hex. 23
distinct colours, 23 occurrences, and zero hex left in any page.

**The sign hues are not shared between schemes.** `--pos` #4cb782 is washed out
on paper and `--neg` #f0666d reads as candy there, so light uses #176b45 and
#9b2226. A palette does not invert; the two grounds need different colours to
hold the same contrast. Measured against each ground:

```
                dark      light
body text      13.29      15.69
positive        7.39       6.40
negative        6.00       7.78
muted           6.18
```

All above the 4.5:1 floor, most above 7:1. And `color-scheme` is declared in
both, which is what makes the scrollbars and the canvas the browser paints
behind the page follow the theme — without it a light reader gets a dark
scrollbar.

## 2026-08-29 — The dark mode is a terminal, and that is why it finally worked

Three attempts at "professional dark" went generic (navy-slate, the default
ground of every framework), then warm-and-tiring, then generic again. The
failure was not taste. **The target was an adjective**, and an adjective cannot
be checked against anything. "Bloomberg terminal" can, and it landed first try.

Two properties carry almost the whole look:

```
ground   #04070a   near-black
accent   #ffa028   amber, on every LABEL -- headings, column heads, nav,
                   and the NAV series line
```

Numbers stay ink. Amber marks the things that NAME data rather than being data,
because a label colour competing with the sign colours would leave nothing
winning.

**It was three.** A monospaced face with uppercase headings and .04em tracking
came with the reference and went again on sight -- the type should read the same
in both schemes. Worth recording what that proved: the look survived losing it,
so the colour was doing all the work and the type was doing none. Both schemes
now render in the browser's own serif, sentence case, normal tracking.

Once both schemes wanted the same type, `--face`, `--caps` and `--track` became
twelve lines saying "serif, none, 0" twice over, so they were deleted rather
than defined identically -- and nothing replaced them, because the browser
default IS what "the same as light" means. `--accent` stayed: it is the one
thing that genuinely differs.

### The eye-strain lesson survives the style change

A terminal look does not require terminal contrast. Ink sits at 10.59:1, not the
14.21:1 that was tiring two grounds ago, on a ground that is nearer black than
either of them. The signature is the amber, not the glare.

```
             text   accent   pos    neg
terminal    10.59    9.91    8.90   6.67
light       15.69      --    6.40   7.78
```

### It is one token, not a second stylesheet

`--accent` joins the existing mapping: amber in dark, `var(--l-ink)` in light,
which is inert. Measured after the change, light is unchanged — ink navy
headings, no amber anywhere — and dark now matches it on face, case and
tracking while keeping the ground and the amber. No dark-only selector exists.

That is what has made four grounds affordable. Each attempt was one block of
`--d-*` values; no page, no chart code and no JavaScript was touched by any of
them. The typographic half of a theme rides the same mechanism as the colour
half because `text-transform` and `font-family` take custom properties as
happily as `color` does.

Monospace was wider per character and put the nine-column table at risk; with
the face reverted that risk is gone, and both pages were re-checked as fitting
with no horizontal scroll.

---

### Superseded: three dark grounds, and the one that mattered was not a colour

Navy-slate first: generic, because it is the default of nearly every framework
and dev tool. Warm charcoal next: distinctive, and tiring to read.

The tiring part was not the hue. `--d-ink` was at **14.21:1** against the
ground, close to the maximum obtainable, and comfortable reading on a dark page
sits nearer 10-12:1. Glare above that is what the eye actually feels, and it
would have been just as present on any of the three grounds.

**That is the trap in "just make it darker".** Deepening the ground while the
text stays put RAISES contrast and makes the complaint worse. The two have to
move together, and the ink has to move further:

```
              navy      warm      now
page        #0f141c   #191614   #0e1013
ink         #d6dbe4   #e9e4da   #bfc4cc
text ratio    13.29     14.21     10.87
muted          6.18      6.24      5.24
pos            7.39      7.27      7.02
neg            6.00      5.51      5.84
```

The sign colours went desaturated with it — a saturated hue on a near-black
ground glows, and glow is the same complaint as glare in a smaller place — and
the rules sit close to the ground so structure is findable when looked for and
invisible when read past.

The light scheme was not touched through any of the three: 15.69 / 6.40 / 7.78
before and after. Re-grounding the dark half is one block of `--d-*` values, no
selector, no page and no chart code.

### Superseded: the warm ground, kept for the reasoning

It was `#0f141c`, a dark navy-slate, and it read as generic because it is: navy
slate is the default ground of nearly every framework and dev tool, so it is the
most-seen dark there is. A warm cast is rare, costs nothing, and suits a page
whose identity is a printed report rather than an app.

```
            was (navy)   now (warm charcoal)
page          #0f141c      #191614
ink           #d6dbe4      #e9e4da
pos           #4cb782      #7fb08a   sage, not a bright green
neg           #f0666d      #e2686a
```

`--d-pos` went sage on purpose: a saturated green on a warm ground fights it,
where a desaturated one sits in the same light as the paper. Contrast held
through the swap and was measured, not assumed:

```
             text   pos    neg   muted
warm dark   14.21   7.27   5.51   6.24
light       15.69   6.40   7.78
```

The light scheme was not touched. That is the payoff of the two-palette
structure — re-grounding the dark half was one block of `--d-*` values, no
selector, no page and no chart code went near it.

### Light is the default, and dropping the media query is what made that true

A results page for a thesis should open as a document, so a visitor gets light
and dark is opt-in. The choice persists, so anyone who prefers the terminal sets
it once.

**Flipping which scheme `:root` held would have changed nothing observable.** A
media query for the other scheme covers the same two cases either way — system
dark gets dark, system light gets light — whichever one is written as the
default. For "light by default" to mean anything to a reader whose system says
dark, `prefers-color-scheme` had to stop deciding altogether. That is the actual
edit; the `:root` swap is bookkeeping.

It also collapsed the structure. Three mapping blocks became two:

```
was                                  now
:root                    dark        :root                  light
@media (light) :not([dt]) light      :root[data-theme=dark]  dark
:root[data-theme=light]  light
```

`app.js` had to move with it: `activeTheme()`'s fallback is the constant
`"light"` rather than a media query, and the system-preference listener is gone.
The stylesheet and the button have to agree on what "no choice yet" means, or
the button offers to switch to the theme already on screen.

Verified with the system reporting DARK and nothing stored: the page opens
light, the toggle offers "Dark", one click gives the terminal, and the choice
survives a reload and a page change.

### The toggle, and the two things that make one work

A control in the nav on all three pages — not just the Overview, because the
Journal and P&L pages are directly linkable and a reader landing on one would
otherwise have no way to change it. The choice is stored and overrides the
system preference; clearing it goes back to following the system.

**The stored choice is applied by an inline script in `<head>`, before the
stylesheet.** Applying it from `app.js` at the foot of the body renders one
frame in the system theme and then swaps — the flash every hand-rolled toggle
ships with. It is four lines, in a try/catch because `localStorage` throws
outright in some privacy modes.

**`:not([data-theme])` on the media query is what makes the choice stick.**
Without it, a reader on a light system who picks dark has the media query
fighting the attribute, and which wins comes down to specificity nobody should
have to reason about. With it, an explicit choice simply takes the media query
out of play. There is deliberately no `[data-theme="dark"]` block — `:root` is
already the dark mapping, and a second copy is one more thing to keep in step.

**Nothing is redrawn when the theme changes, including the charts.** That is the
payoff from having moved SVG colour onto classes: the drawing code emits
`class="series"` and the stylesheet decides what it means, so a switch is a
repaint the browser does by itself. Had the colours stayed in presentation
attributes, every chart would have needed rebuilding on every toggle.

The label names the DESTINATION with a `title` saying so in words, because a
bare "Light" reads equally well as the current state or the target. Verified:
one click, one change (no double-binding), choice survives a reload and carries
across pages, and the light scheme's own sign hues resolve (`#176b45`,
`#9b2226`) rather than the dark ones.

### One thing that is not a bug

Measuring the column balance immediately after a scripted viewport change reads
-174px. It settles to 0, and dispatching a real `resize` gives 0. The reading is
a race between the emulation and the debounced redraw, not the page — recorded
because I chased it twice.

---

## 2026-08-29 — The Mapping page, and the column that named the provider

`instrument_mapping.csv`, published as a page. It is the last thing needed to
make the site self-checking: `qty x (close - open) x pointsize x FX` is the
arithmetic a reader is invited to do, the quantity and both prices are on the
Journal page, and until now the point value was a click away in the repository.
63 contracts, grouped into the ten asset classes, banded so the first question
anyone asks of the file — what is in this book — is answered by the shape of it:

```
Ags 13   Bond 12   Equity 11   FX 9   OilGas 6   Metals 5
STIR 3   Crypto 2  Carbon 1    Vol 1
13 exchanges, 7 currencies
```

### The page states the cost methodology, including what it does not claim

The reason to publish the specs is that a reader can check the arithmetic; the
reason to publish the METHOD beneath them is that the cost column is the one
number on this site that is an assumption rather than a record, and a reader who
cannot tell the difference will assume the worse of the two.

So the footer says it plainly: the round-trip cost is one tick plus a fee floor
of about $5 equivalent; published US rates are $4.10-4.70, so the floor sits
above them in every currency; sixteen contracts were revised on 2026-08-29 and
**all sixteen moved up, none moved down**; the fee floor is defended by rate
cards while the spread widening is judgement and is labelled as judgement.

And it states the limit rather than letting the reader infer a stronger claim:
this is not an assertion that the costs are overstated, because the spread term
is not an upper bound on the illiquid tail and market impact is not modelled at
all. An assumption presented as an assumption is not a weakness. One implied to
be a measurement is.

### The first column was `norgate_code`

The carve-out is non-commercial use with **the data provider never named on the
site**, and a column header is a name. It is published as `instrument`, which is
what every other page already calls it.

A rename is exactly the kind of thing that survives one edit and is undone by
the next, so it is checked rather than remembered: `FORBIDDEN` holds the words
that must not reach a published byte, and `build_mapping()` searches the
finished payload for them before anything is written. Tested by putting the
provider back into a row — the guard catches it. `grep -ril norgate docs/`
returns nothing.

That is the same shape as `_guard` and `_dates_guard`: the rule is enforced by
the build, not by whoever edits next.

### What is published, and why the cost column belongs

Eight fields, all of them either exchange specification or our own assumption:
point value, tick size, currency, exchange, asset class, description, and the
round-trip cost. The cost is a model input, but it is one the site already
states the consequences of — 1.49% a year, $1,061,404 over the window — and a
reader who cannot see the per-contract assumption cannot check either figure.
The model proper (`SIGNAL`, the gates, `IDM`, `w_i`) stays unpublished as
before.

---

## 2026-08-30 — Stage 6: the publisher runs inside the pipeline

`publish.py` ran by hand, which worked only because someone remembered. The
failure it invites is the quiet one: run the pipeline, forget to publish, and
the site serves yesterday's numbers looking exactly as current as ever. The
`Updated` line makes that VISIBLE; it does not make it not happen.

So it is stage 6 of `Update.py`, on by default, with `--no-publish` to skip.
Pushing stays manual — this writes files, it does not deploy.

### Ordering was the part that needed thought

The run stamp had to move ABOVE the summary block, because stage 6 reads it:
`publish.py` takes the site's "Updated" line from `.pipeline_run.json` and
refuses a run that failed or was partial. Written after stage 6, as it was, the
site would have carried the PREVIOUS run's timestamp over THIS run's numbers —
fresh figures under a stale date, which is worse than either alone.

### Skipped, not failed

`publish.py` would refuse a bad run anyway; its `run_stamp()` aborts. But
letting it abort INSIDE the pipeline would end the run on a traceback about the
website when the real news is the verification above it. So `Update.py` checks
first and skips with a reason. The site keeping its last verified numbers is the
correct outcome, and worth saying out loud rather than exiting quietly.

Four states, all tested:

```
full clean run          publishes, summary reports `site -> docs/data`
--no-publish            skipped, silently
--no-reconcile          skipped: "this was a partial pipeline"
--no-verify             skipped: "verification was skipped"
```

### The fourth one was a hole this found

`--no-verify` leaves `failures` at zero **because nothing ran**, which is the
one way a bad run can look like a clean one to a check that only counts
failures. Wiring the stage is what surfaced it: the condition `if failures:` is
correct and insufficient.

The stamp now records `verified` separately from `failures`, so stage 6 can tell
"passed" from "never asked", and the same guard exists on both sides —
`Update.py` skips, and `publish.py` aborts independently. That second one is not
redundant: `publish.py` is also run by hand, and by hand is exactly when somebody
reaches for `--no-verify`.

Stage labels renumbered 0/6 through 6/6 throughout. A label that says 5/5 while
six stages run is a small lie that costs nothing to leave and nothing to fix.

---

## 2026-08-30 — `verify_publish`: the only report that reads `docs/` back

Every guard in `publish.py` runs BEFORE the write. The whitelist, the window
guard, the page guard, the provider guard and the per-session reconciliation all
inspect rows in memory; once the files are on disk the only thing that stage
does is total their byte counts. A truncated or half-written file — interrupted
run, full disk, encoding fault — passes every one of them and lands on the site
looking fine.

So stage 6 now has a report, in the same shape as the other nine, running after
the stage rather than inside it. **It is `verify_stages` for the publication
layer**: the question is the same one, do the artifacts agree with each other,
or is one of them internally perfect and describing a different run.

```
[OK] every top-level file present and parses          5/5
[OK] every journal session file exists and parses     170/170
[OK] every attribution session file exists and parses 171/171
[OK] latest.json agrees with Portfolio.parquet        2026-08-28  171  108,947,731
[OK] history.json ends on the portfolio's last session
[OK] every attribution sheet sums to its own book     171 sessions
[OK] all pages carry the same, current cache stamp    4 pages
[OK] the data provider is named in no published file  352 files scanned
8/8, 0.6s
```

### Six deliberate breakages, because an untested check is not a check

This file already records tie F passing at 1e-6 while both sides were wrong. A
report that has never failed has never been shown to be capable of failing, so
each check was made to fail on purpose:

```
truncated a day file mid-write        -> caught, named the date
nudged latest.json equity by $1       -> caught
removed 5 rows from an attribution    -> caught, named the date
left one page unstamped               -> caught, "UNSTAMPED"
edited app.js after publishing        -> caught, stamp stale
put the provider back into a page     -> caught, named the file
```

All six, and the state restored to 0/8 afterwards.

An unplanned result: nudging `latest.json` also tripped the CACHE STAMP check,
because the stamp hashes that file. Check 7 is therefore a tamper detector on
the published headline as well as a staleness check on the assets — the
published equity cannot be edited by hand without the stamp disagreeing.

### It borrows publish.py's definitions rather than copying them

`build_stamp`, `PAGES`, `TAGS` and `FORBIDDEN` are imported from the module
under test. Re-deriving the stamp here would give two implementations of one
rule, which is the exact shape that has cost this pipeline three quiet
divergences. The cost is that the check trusts publish.py's definition of the
stamp — it verifies the stamp was WRITTEN correctly, not that the scheme is
right — and that is the correct division: the scheme is a decision, the write is
a fact.

### What it deliberately does not check

**The live site.** The push is manual and meant to be, so `docs/` on disk is
SUPPOSED to run ahead of the deployed page; a check against the URL would fail
every time someone published and had not yet pushed. This verifies the artifact,
not the deployment. The gap between the two is now the only unverified step in
the chain, and it is a human typing `git push`.

The run stamp is rewritten after this report, because the failure count changes
after the first write and a stamp claiming zero would let the next manual
publish proceed off a run this one had just failed.

---

## 2026-08-30 — `6_Publish` became `5_Publish`, and the stage numbers with it

The directory was numbered to match `Update.py`'s stage labels, where NDU
occupied 0 and 5 as bookends and publish was pushed to 6. But the data stages
are the directories -- `1_Roll`, `2_Engine`, `3_Portfolio`, `4_Bookkeeping` --
and the next one is 5.

Renaming alone would have left `STAGE 6/6 publish.py` pointing at a directory
called `5_Publish`, so the labels moved too. **NDU is not a pipeline stage**: it
is an external application started and stopped around the stages that need it,
and numbering it was the only reason publish could not be 5. Unnumbered, every
remaining label equals its directory, which is the sole reason to number them.

```
was                          now
STAGE 0/6  NDU (start)       NDU (start)
STAGE 1/6  contract_cycles   STAGE 1/5   -> 1_Roll
STAGE 2/6  trading_book      STAGE 2/5   -> 2_Engine
STAGE 3/6  portfolio         STAGE 3/5   -> 3_Portfolio
STAGE 4/6  bookkeeping       STAGE 4/5   -> 4_Bookkeeping
STAGE 5/6  NDU (close)       NDU (close)
STAGE 6/6  publish           STAGE 5/5   -> 5_Publish
```

`git mv`, so the history of both tracked files follows. Three text references
(`.gitignore`, `Live/README.md`, the `PUBLISH` constant), and two stale counts
in the module docstring corrected while there: it claimed EIGHT verification
reports when `verify_publish` had made it nine.

### A non-blocking stage was claiming it had stopped the run

Found in the same pass. `run()` prints one failure message, and stage 4b is
caught by its caller and tolerated -- so a journal conflict printed
`[ABORT] ... Later stages NOT run.` while every later stage ran. An untrue line
in a log is worse than no line: it is what somebody reads when they are working
out why a run went wrong. `run(blocking=False)` now prints
`[FAILED] ... Non-blocking: the run continues.` Both branches unit-tested.

**And I was reading my own output too narrowly.** Every "pipeline green" in this
session came from grepping `passed|FAIL`, and that abort line contains neither
word. Stage 4b had been failing since the cost audit and I reported the run
clean several times. The filter that hides a failure is worth more scrutiny than
the failure.

---

## 2026-08-30 — The terminal, restored in full, on actual black

*Partly superseded the same day: the amber accent and the monospaced face
were both dropped — see "Four accents, and the one that won has no hue"
below. The ground, the neutral greys and the contrast reasoning stand.*

The brief moved back: dark is a Bloomberg terminal again, and the ground is
`#000000` rather than the `#04070a` it had been. That previous value was a deep
blue-black, and the blue cast is what made it read as a theme OF a terminal
rather than the thing itself.

**The greys went neutral with the ground.** Leaving a blue-tinted panel and blue
rules against pure black would have read as an unfinished conversion, not a
decision. `#0d0d0d`, `#bcbcbc`, `#8a8a8a`, `#3a3a3a`, `#1e1e1e` -- no cast at
all.

### Pure black was safe here only because the ink was already right

Earlier in this file: a ground was deepened while the text stayed put, contrast
went UP, and the page got harder to read. The pairing that actually hurts is
black against WHITE. This ink is `#bcbcbc`, so:

```
              ground     ink    text  accent   pos    neg
blue-black   #04070a  #b8bcc2  10.59   9.91   8.90   6.67
pure black   #000000  #bcbcbc  11.06  10.31   9.26   6.94
```

Full black cost nothing, because the other half of the pair had been fixed two
grounds ago. Had the ink still been the `#e9e4da` of the warm attempt, this
change would have taken it to about 15:1 and undone that whole repair.

### The type is back, and only now does it deserve a token

`--face`, `--caps` and `--track` were deleted when both schemes were meant to
read alike -- twelve lines saying "serif, none, 0" twice over. They return
because the schemes genuinely differ again, which is the ONLY condition under
which a scheme token earns its place. The comment beside them now says so, and
says that the day they agree again they should be deleted rather than defined
identically. That is the rule, not the current answer to it.

Monospace is wider per character, so the nine-column table was re-checked rather
than assumed: Pending, Executed, the attribution sheet, the reconciliation walk
and the 63-row mapping all fit, no horizontal scroll on any page.

Light is untouched by all of it -- serif, ink navy, sentence case, no accent.
Measured after the change, not asserted.

### One duplicate the invariant caught

`#ffa028` ended up declared twice, as `--d-accent` and as `--d-sel-fg`. The file
says a role reusing another role's colour aliases it rather than repeating the
literal, so `--d-sel-fg:var(--d-accent)` -- which states the decision, "selection
is drawn in the accent", instead of a number that can drift. 23 declared
colours, 23 distinct.

Two more matches were prose: the comment quoted `#bcbcbc` and `#04070a` while
explaining them. Rewritten to describe rather than quote, so the greppable
invariant stays greppable. A hex in a comment is a second copy that no compiler
will ever check.

---

## 2026-08-30 — Four accents, and the one that won has no hue

Settled: pure black ground, grey body text, a brighter grey for labels, and red
and green for signs. No accent colour at all.

Getting there took amber, cyan and a warm white, with a phosphor green ruled out
before it was tried. The deciding evidence was measured rather than argued.
Every candidate on `#000`, with dichromat simulation, against the two colours
that already MEAN something on this page:

```
                 contrast   separation from neg / pos
                 on black    normal   deuteranopia  protanopia
amber              10.31     86/216       17/152      22/123
cyan               11.62    290/133      185/89      162/97
sky blue            8.31    245/155      196/92      180/103
phosphor green     14.50    244/88        98/49       57/53
violet              8.73    162/200      130/108     130/98
warm white         16.83    197/218      139/172     143/159
        for scale: neg vs pos itself is 228 / 139 / 102
```

Two results worth keeping. **Amber sits 17 from the negative red under
deuteranopia** — they nearly converge, where the two sign colours are 139 apart.
**Phosphor green collides with the positive green at 49**, which is why the most
obviously "terminal" choice was never a candidate.

### The winner is the one that stopped competing

A warm white is not an accent, and that is the point: it leaves **exactly two
hues on the page, and both of them mean something**. Nothing has to compete with
red and green for attention. It also happens to measure best on the axis that
matters, 139/172, because a neutral has no hue to converge with.

`#e6e6e6` is 16.83:1, which would be glare on body text and is not on this.
These are headings and column heads — a few words at a glance. Glare is a
property of a paragraph read for a minute, not of a word looked at.

### The type was tried twice and removed twice

Monospace with uppercase headings came in with "the full Bloomberg look" and
went out again both times. The second time was my error: the brief was the
ground and the accent, and I changed the face uninvited along with them.

`--face`, `--caps` and `--track` are deleted, not defined identically in both
schemes. That is the rule this file already states, and it now has a second
demonstration: a scheme token is defensible only while the schemes disagree.
`--accent` is the only one left that does.

### What the whole sequence actually was

Navy-slate, warm charcoal, deep neutral, terminal-on-blue-black, pure black;
amber, cyan, warm white. Read as a list it looks like thrashing. Read by what
each round REMOVED it is not: a blue cast, then excess contrast, then a
monospaced face, then a hue. **The design converged by subtraction**, and the
final state is the one with the fewest decisions in it.

---

## 2026-08-30 — CI tested a tree that no longer exists; it now tests the one that does

The push went red. Both jobs failed in 16 seconds, which is the signature of a
collection error rather than a real failure, and the cause was not the change
that shipped with it: commit `9246c62` removed 66 files under `production/`,
deliberately — the S183 tree that `Live/` supersedes — and CI ran four scripts
from `production/tests/`. Nothing it tested was still there.

### What could still be tested on a clean checkout

The pipeline artifacts are gitignored, so nothing in `Live/` can be exercised
here: no panel, no `Portfolio.parquet`, no ledger. But `docs/data` IS tracked,
346 files of it, along with `instrument_mapping.csv`.

That is the honest successor to the `verify_track` that was deleted, and the
deleted one's own comment said why: *"no panel, no engine, no trust in either.
This is the check an outside reader can reproduce, which is the whole point of
publishing the ledgers."* The subject changed; the argument did not.

### `verify_track.py` — 13 checks, standard library only

**No dependencies at all**, deliberately. The site's claim is that a reader can
check it; a verifier that needs `pip install` weakens that claim, and one that
needs the vendor panel destroys it. CI installs nothing, which also makes it
fast enough to be worth having.

```
[OK] every top-level file present and parses
[OK] every journal session file exists and parses      170/170
[OK] every attribution session file exists and parses  171/171
[OK] every attribution sheet sums to its own book      worst 0.035
[OK] opening + gross - commission + interest == closing  worst 1.5e-08
[OK] the equity chain joins: close[t] == open[t+1]     170 joins, worst 0.0
[OK] history covers exactly the published sessions
[OK] history.json agrees with every session's book     worst 5.0e-03
[OK] latest.json is the last row of history.json
[OK] specs positive, and cost_rt exceeds one tick      63 contracts
[OK] mapping.json covers the whole instrument file
[OK] all pages load the same asset version
[OK] the data provider is named in no published file
```

**The chain check is the one a fabricated track fails.** Individually plausible
sessions can each close perfectly and still not join up; insisting that one
session's closing equity IS the next one's opening makes the equity curve a
single object rather than 171 unrelated statements. It holds here to 0.00
exactly across all 170 joins, because the sessions are generated from one
recursion rather than assembled.

`cost_rt > one tick` is the published cost methodology stated as a test: the
model is one tick plus a fee floor, so a contract can never be priced at or
below a single tick. A future edit that broke that would be caught by CI rather
than by a reader.

### Break-tested, as everything in this file now is

```
a session's walk no longer closes    -> caught, and by TWO checks
the equity chain snapped by $1,000   -> caught, named the join
history disagrees with a book by $7  -> caught, named the field
a contract priced below one tick     -> caught, named the contract
pages loading different assets       -> caught
```

Two of those trip more than one check, which is what a suite with overlapping
claims should do.

### It does not replace `verify_publish`

They ask different questions. `verify_publish` asks whether the site agrees
with the PIPELINE and needs the parquet to answer, so it can only run locally.
This one asks whether the site agrees with ITSELF, and can run anywhere. **A
stale publish passes this and fails that.** A corrupted publish fails both.
Keeping the pair is the point, not a redundancy to tidy away later.

---

## 2026-08-30 — Three site defects: a noisy deploy, an unchecked contract, a silent stale page

**The deploy committed on every run.** Every publish rewrites `updated_at`,
`generated_at` and the cache stamp, so `docs/` is ALWAYS dirty and five runs in
a day gave five commits differing in a timestamp. Same distinction the journal
draws between decision and context: those fields record the act, not the
content. The deploy now diffs `docs/` and commits only when a line changes that
is not one of them, reverting the working tree otherwise.

The argument that settled it: **an older `updated_at` on unchanged numbers is
not stale, it is accurate.** It says when those numbers were produced, which is
what a reader wants to know.

**Nothing checked that the pages and the payload agree on key names.** Rename a
key in `publish.py` and not in `app.js` and every suite still passes: the JSON
is valid, the whitelist is satisfied, the arithmetic reconciles, and the page
renders `undefined`. Structure and arithmetic are both fine; only the contract
between the two files is broken.

`verify_track` now reads the property accesses the pages make off short-named
objects (`r.instrument`, `b.gross_pnl_USD`) and requires each to be a key
`publish.py` declares. Two renames were simulated and both were caught.

The vocabulary is taken from what `publish.py` PROMISES, not from what today's
data contains — the first attempt used the JSON alone and flagged
`carried_sessions`, a real declared column that vanishes from the payload
whenever no order is outstanding. Parsed with `ast` rather than imported,
because importing would drag in polars and this suite is dependency-free on
purpose.

**A stale page looked exactly like a fresh one.** `Updated` tells a reader when
the pipeline last ran, but only if they know what it should say; a date that
stopped moving looks like a date. `staleNote()` says it in words on all three
dated pages, judged against the reader's own clock:

```
last session 2026-08-28, read on   2026-08-30  silent   (Sunday)
                                   2026-08-31  silent   (Monday, before the run)
                                   2026-09-01  "4 days ago. A session may be missing."
                                   2026-09-12  "15 days ago. This page has stopped updating."
```

Three days is what a weekend costs. The wording states the fact — the newest
session shown, and how old it is — rather than making a claim about the
pipeline, because a reader's clock can be wrong.

---

## Conventions and decisions

- **The whitelist is the point of this file.** Every output is filtered through
  an explicit column list and the build fails if anything outside it appears.
  Adding a column to the site is a deliberate edit here, never a side effect
  upstream.
- **The window guard aborts, it does not filter.** Nothing dated before
  2026-01-02 is published, at any time, for any reason.
- **The provider is never named on the site.** Not in the pages, not in the
  JSON, not in a comment that ships. Enforced by `FORBIDDEN`, which the
  build searches the finished payload against — the source file's first
  column is `norgate_code` and it is renamed to `instrument` on the way out.
- **Totals come from the series NAV is built on.** Where an attribution and the
  book disagree, the page shows the book's number and the disagreement gets
  written up in the stage that owns it.
- **Two rates shown next to each other are on the same basis, or they are not
  shown next to each other.** Trading-only and interest-inclusive figures are
  both legitimate; adjacent and unlabelled they make a difference in basis
  look like a difference in result.
- **A chart's axis formatter and its hover formatter are different things.**
  The axis rounds to stay legible; the readout is the exact figure, because
  exactness is the only reason to hover.
- **Money on the attribution page is stated to the cent.** Everywhere else,
  whole dollars — no reader is checking whether a Sharpe of 1.17 adds up.
- **Layout claims are DOM measurements at a named viewport width**, never
  screenshots from the preview pane. See the 2026-08-29 entry for what that cost.
- **Per-date files, not one big file.** A journal page that answered "what
  happened on this date" from a single archive would ship every order to every
  visitor who wants one day. 170 day files and 171 attribution files, a couple of
  kilobytes each.
- **The only hues on the page are the two that mean something.** `--neg`
  for below zero, `--pos` for above it, both redundant with a sign already
  on screen. No coloured headings, no accent bars.
- **Red is global, green is opt-in.** A negative is worth catching
  anywhere; a positive is only worth colouring in a column where the sign
  varies. Balances, magnitudes and ratios are never gains.
- **Do not quote a hex value in a comment.** It is a second copy nothing
  checks, and it breaks the one-declaration-per-colour invariant from the
  outside. Describe the colour instead.
- **The palette lives in `site.css` and nowhere else.** No page carries a
  hex value; SVG colour is set by class, not by attribute.
- **Two schemes, chosen by `prefers-color-scheme`, dark by default, with a
  toggle that overrides it.** Each colour is defined once and a scheme only
  re-points the working tokens. The sign hues differ between schemes
  because contrast does. A stored choice is applied in `<head>` so the
  page never paints in the wrong scheme.
- **The cache version is content-derived and stamped by the publisher.**
  One value on the script tag versions `app.js` and every JSON fetch; it
  changes when the script or the data changes and at no other time.
- **The site reports when the DATA was rebuilt, not when it was published.**
  `Update.py` owns that timestamp; stage 6 reads it and will not publish
  off a failed or partial run.
- **The publisher runs inside the pipeline, and refuses a run it should
  not publish.** Stage 6 of `Update.py`, on by default. Skipped on a
  failed, partial or unverified run, so the site keeps its last verified
  numbers rather than gaining doubtful ones.
- **The published track is auditable from the published files alone, with
  no dependencies.** `verify_track.py` is what CI runs and what a reader
  can run; if a check there ever needs a package or the panel, it belongs
  in `verify_publish` instead.
- **Stage 6 has a report, and it reads the written bytes.** Every guard in
  `publish.py` is pre-write; `verify_publish` is the only thing that opens
  `docs/` afterwards. It borrows that module's own definitions rather than
  restating them, and all eight checks have been made to fail on purpose.
- **`push` is manual.** The publisher writes `docs/data/`; nothing in the
  pipeline sends anything anywhere.

## Not yet built

