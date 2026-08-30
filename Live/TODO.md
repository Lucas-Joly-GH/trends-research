# TODO — notice-period lead time

## The task

Add a column giving **how far ahead of expiry the first notice period starts**,
i.e. the gap between `first_notice` and `last_trade` for a contract.

This is what makes the delivery constraint schedulable. Today the roll rule
holds out a flat 5 business days before first notice for every market, which is
a guess applied uniformly. Knowing that ZC gives ~14 days of notice lead while
another market gives 2 lets the margin be set per instrument instead.

## You do not need the vendor for this

Both dates are already exported. `Data/Contract-Metadata/contracts.csv` in the
**private** repo (`LJOLY_Memoire_INSEEC_Msc2`) has one row per contract:

    symbol, instrument, year, month_code, month, first_notice, last_trade

15,231 rows, both dates ISO `YYYY-MM-DD` or empty. So the per-contract gap is a
subtraction, and the per-instrument column is an aggregate of it. Only reach for
`norgatedata` if you need something these files do not carry.

## Where everything is

| what | where |
|---|---|
| contract dates (both) | `LJOLY_Memoire_INSEEC_Msc2/Data/Contract-Metadata/contracts.csv` |
| per-market metadata | `.../Contract-Metadata/instruments.csv` |
| the cycle table to extend | `trends-research/Live/1_Roll/contract_cycles.csv` |
| the script that writes it | `trends-research/Live/1_Roll/contract_cycles.py` |
| raw price data | `LJOLY_Memoire_INSEEC_Msc2/Data/Paper-trading/Contracts/<INST>/` |
| python | `LJOLY_Memoire_INSEEC_Msc2/.venv/Scripts/python.exe` |

Dependencies are **norgatedata, polars, numpy, scipy**. No pandas anywhere.

**Data goes in the private repo, code goes in `trends-research`.** Never commit
vendor data to `trends-research` — it is public.

## Decisions you have to make, and why they are not obvious

**1. Which table does the column belong in?** The gap is a property of a
*contract*, not a market — it varies within an instrument across months and
years. `contracts.csv` can carry the exact per-contract gap. `contract_cycles.csv`
is one row per instrument, so it can only carry an aggregate. Probably both:
exact in the first, median plus spread in the second. Report the spread — if it
is wide, a single per-instrument margin is the wrong abstraction and that is
worth knowing before anyone builds on it.

**2. Calendar days or business days?** The roll margin is currently expressed in
business days (`np.busday_count`). Matching that makes the two directly
comparable. Whichever you choose, put it in the column name — `notice_lead_bd`
beats `notice_lead`.

**3. What do you write for the 59% of contracts with no `first_notice`?** Empty
is not the same as zero, and getting this wrong makes cash-settled markets look
like they give zero days of warning, which would be read as maximum delivery
risk when in fact there is none. Keep them null and let the consumer decide.

## Known traps

* **Only 6,265 of 15,231 contracts (41.1%) carry a first notice date**, and that
  is correct rather than missing — cash-settled markets have no notice period.
  See `CASH_SETTLED` in `contract_cycles.py` for the 22 that are cash-settled.

* **36 contracts have no `last_trade`** — mostly far-dated listings the exchange
  has not scheduled yet (ZW 5, SI 3, GC/PL/HG 2 each). Handle the gap; do not
  assume the field is always populated.

* **Three deliverable markets have NO notice period at all: DX, GAS, EUA.**
  Verified against ICE documentation on 2026-08-26 — all three go straight from
  last trade to delivery. They are `is_deliverable = true` with an empty
  `first_notice`, which will look like a data error and is not. Whatever the
  column means for them, it is not zero and it is not null-because-missing —
  they are the case where the delivery obligation attaches with no warning, so
  arguably they need the *largest* margin, not the smallest.
  There may still be a notice-*like* deadline under another name (a delivery
  intention, nomination or allocation window) in the ICE delivery-procedure
  documents rather than the product specs. Nobody has checked. If you do, the
  documents are linked in the `CASH_SETTLED` comment block.

* **The NDU feed is intermittent.** `norgatedata` needs the Norgate Data Updater
  running locally. It failed twice on 2026-08-26 and succeeded on the identical
  command a minute later. Retry before concluding anything is broken, and do not
  copy `production/s183/norgate.py::is_available` — it swallows the exception and
  makes an outage indistinguishable from a bad install.

* **Session symbols and contract symbols are not interchangeable.**
  `futures_market_name` and `session_type` take a session symbol (`ES`);
  `tick_size`, `margin` and `lowest_ever_tick_size` need a contract
  (`ES-2026Z`) and return "not found" for a session, writing empty columns
  while the run still exits 0.

## What a first pass already shows

I ran the subtraction to confirm the data supports it. Business days between
`first_notice` and `last_trade`, over the 6,265 contracts carrying both:

    median 18    min -10    max 26

Three things fall out of that, and they shape the design:

**HO is negative — first notice lands AFTER last trade.** All 8 negative
contracts are heating oil, median **-10** business days, range -10 to -7. HO is
also the one instrument with partial coverage (6 notice dates out of 585). So
either the vendor's field means something different for HO, or it is wrong for
HO. Do not let a negative silently flow into a margin calculation, and do not
"fix" it by clamping at zero without first working out which of the two it is.

**The spread within an instrument is wide.** ZC ranges 9 to 16 business days,
CC 12 to 25, ZF 12 to 23. So a single per-instrument number is an approximation,
not a fact — which answers decision 1 above: carry the exact value per contract
and treat the per-instrument aggregate as a summary, not the source of truth.

**The groups are real and useful.** Grains and softs cluster near 10 business
days, US bonds near 15, metals near 20. That is the argument for a per-market
margin rather than the flat 5 currently used everywhere.

## Check your work

Whatever you compute, sanity-check a few by hand against the exchange calendar.
`ZC-2026Z` is `first_notice 2026-11-30`, `last_trade 2026-12-14`. If your column
does not say roughly two weeks for that row, the arithmetic is wrong.

Also confirm no cash-settled instrument acquires a non-null lead — `verify()` in
`contract_cycles.py` already enforces the related invariant (nothing marked
cash-settled may carry a notice date) and runs on every build.
