# Live

Clean-slate rebuild of the S183 daily paper-trading engine. Nothing here is
carried over from `production/` by default — that tree stays where it is as
prior art, and anything moved across should be moved deliberately, not
inherited.

## Data lives in the private repo, never here

`trends-research` is public. **Vendor data must not be committed to it.** The
panel is in `LJOLY_Memoire_INSEEC_Msc2` (private), which already holds it:

    Data/Paper-trading/Contracts/<INST>/<INST>-YYYYM.csv   raw delivery months
    Data/Paper-trading/Contracts/<INST>/notice_dates.csv   first-notice calendar
    Data/Cash-Yield/%IRX.csv                               collateral rate

### The second exception: what the public site publishes

`docs/` serves a results page for the **2026 run only**, and it deliberately
carries prices -- the open it filled at and the close it decided on -- because a
table of "SELL 37 6B" with no prices cannot be checked by anyone. An
unverifiable results page is worth less than none, so the site publishes enough
to let a reader recompute a day by hand:

    N x (close - open) x pointsize x FX  =  the P&L we claim

THE CARVE-OUT IS NARROW AND IS THE WHOLE POLICY:

  * 2026 sessions only.  Nothing before 2026-01-02 is ever published.
  * Open and close only.  No Panama-adjusted series, no volatility estimate,
    no SIGNAL, no IDM, no forecast provenance.
  * Non-commercial.  A thesis and a personal record, no product.

The reasoning, so the next reader can judge it rather than inherit it:
CURRENT-YEAR SETTLEMENT PRICES FOR LIQUID LISTED FUTURES ARE PUBLISHED BY THE
EXCHANGES THEMSELVES and carried by many free sources.  That is not the
proprietary asset.  The proprietary asset is the cleaned, back-adjusted
1978-2026 panel with its roll logic, and that never leaves the private repo.

`Live/5_Publish/publish.py` enforces this with a WHITELIST -- a hardcoded list
of the columns that may leave -- and fails the build if the output ever contains
a column outside it.  A blacklist fails open the day someone adds a field; a
whitelist fails closed.

The universe list is the exception, and it lives HERE, at
`Live/instrument_mapping.csv`. It carries no vendor price data -- 63 rows of
contract specs and our own cost assumptions -- so it is safe in a public repo,
and it needed a correction the private copy does not have:

* `Général/instrument_mapping.csv` in the private repo describes GAS as Dutch
  TTF Natural Gas (EUR, ICE Endex, pointsize 730, tick 0.005). Norgate's `GAS`
  code returns **ICE Low Sulphur Gasoil** (USD/tonne, 100 t/lot, tick 0.25),
  confirmed by the vendor's own name, by prices of 111-838 per tonne, by
  sessions back to 1988 when the TTF hub did not exist, and by an expiry rule
  that matches Gasoil and misses TTF by a fortnight. Under the old row a lot
  priced at 700 valued at EUR 511,000 instead of USD 70,000 -- 7.3x oversized,
  wrong currency.

* Five stale `tick_size` values: SO3 and YXT4 0.005 -> 0.0025, DX 0.005 ->
  0.001, PA 0.5 -> 0.1, 6J 0.0005 -> 0.00005. Each was checked against the
  actual price grid in `Contracts/<INST>/`, not just against the vendor field:
  DX prints gaps of 0.001/0.002/0.003 and PA of 0.1/0.2/0.3, none of which are
  multiples of the old values. 6J then lands exactly on 6E and 6S -- same tick,
  pointsize, tick value and cost -- which is what a stale tick looks like once
  it is corrected. tick_size now agrees with the vendor on all 63.

* Three `total_avg_cost_rt_LocalCurrency` values, following from those ticks: SO3 17.5 ->
  11.25, YXT4 10.0 -> 7.5, DX 10.0 -> 6.0. The column reads as one tick plus a
  five-unit commission in the row's own currency, exactly, on 43 of the 63 rows
  in the original -- and those three were among them, computed against the tick
  that was wrong. Recomputing restores the relationship; the count is now 44 of
  63, the extra being GAS.

  6J and PA were left alone, because their costs never derived from a tick.
  6J's 17.5 is what 6E and 6S carry at the identical tick value, so it was
  always right and only the tick was stale. PA's 25.0 sits above gold's 15.0 at
  the same tick value, which reads as a deliberate illiquidity premium rather
  than an arithmetic slip. Neither moves.

That copy is prior art. Do not read it, and do not sync back to it.

State as of 2026-08-26, commit `62019f1f9`: 63 instruments, 15,971 contract
files, 6,265 contracts carrying a first-notice date, all current to
**2026-08-24**. A full re-download that day produced a zero diff, so this is
the vendor's current view and not a stale copy.

**Only those three, plus the mapping above, are source of truth.**
`Data/Paper-trading/PanamaMethod/` and `InstrumentStats/` are DERIVED —
continuous chains and DailyReturn built by the
previous engine's roll rule. Regenerate them; do not inherit them.

## Constraints

Dependencies are **polars, numpy, norgatedata** (plus scipy, kept for the
engine). No pandas — not in the engine, not in the tests, not "just for this
bit". `pandas` WILL be installed regardless, because `norgatedata` declares it
as a hard dependency, so `import pandas` succeeds and pip cannot be told
otherwise. The rule is about our code and nothing enforces it: it is a review
rule, not an install-time one.

`pyarrow` is NOT required. Polars reads and writes parquet natively, which is
what the worksheet cache uses.

### The environment

**TWO venvs exist on this machine and BOTH work.** Neither is canonical:

    LJOLY_Memoire_INSEEC_Msc2/.venv    python 3.12.10   polars 1.44.0
    trends-research/.venv              python 3.11.0    polars 1.44.1  <- PyCharm

They are interoperable, which was verified rather than assumed: the parquet
worksheet cache written by either reads identically in the other (95,533 x 25,
both directions). So it does not matter which you use, only that whichever one
runs the pipeline has the packages.

Nothing has to be activated. `Update.py` runs the stages with `sys.executable`
by default, so the interpreter that starts it is the one that finishes it, and
`--python <path>` overrides that. **`preflight()` checks that interpreter before
any stage runs** and aborts in about a second with the exact missing package,
rather than failing four minutes into stage 1 with an ImportError under a
traceback.

Rebuild a venv from `Live/requirements.txt`:

    py -3.12 -m venv .venv
    .venv/Scripts/python.exe -m pip install -r Live/requirements.txt

### DO NOT `source .venv/Scripts/activate` IN GIT BASH ON THIS MACHINE

It corrupts `PATH` and takes `git` itself with it — the prompt collapses to
`()` and every command afterwards fails with `command not found`. The activate
script only converts its Windows path via `cygpath` under a condition that does
not fire here, so it prepends `C:\Users\...` verbatim, the drive-letter colon
splits `PATH`, and everything after it is lost.

Prepend the Scripts directory instead:

    export PATH="/c/Users/33698/PycharmProjects/LJOLY_Memoire_INSEEC_Msc2/.venv/Scripts:$PATH"

To repair a shell that has already been broken by it, either open a new one or:

    export PATH="/mingw64/bin:/usr/bin:/bin:/usr/local/bin:$PATH"

### The machine name breaks the vendor's status probe

This machine is called `Napoléon`, with an accent. `norgatehelper.py:44` puts
`platform.node()` straight into an HTTP header; headers must be ASCII, so NDU
answers 400 with an empty body, and norgatedata's import-time probe logs the
response body — printing `ERROR: Norgate Data: ` with nothing after it on every
import. Reproduced directly:

    no Client header        -> 200 OK
    Client "Napoleon"       -> 200 OK
    Client platform.node()  -> 400, empty body

It is harmless: every real call afterwards strips headers to ASCII. Only the
probe that runs before our code gets control fails. `Update.py` and
`contract_cycles.py` each patch `platform.node()` before importing norgatedata,
which silences it — two copies, because `platform.node()` does not read
`COMPUTERNAME` on Windows and a subprocess cannot inherit the fix.

**Renaming the machine to `Napoleon` would retire this whole section.**

## Vendor facts worth not rediscovering


These cost real time to establish.

* **NDU is intermittent.** `norgatedata` needs the Norgate Data Updater running
  locally (Windows-only). It failed twice on 2026-08-26 and succeeded on the
  identical command a minute later. Build a retry in. Do not copy
  `norgate.is_available()` from `production/` — it swallows the exception and
  makes a transient outage look exactly like a broken install.
* **Non-ASCII outbound headers break the API** with a bare `ValueError`. The
  session headers must be stripped to ASCII before the first call; see
  `production/s183/norgate.py::ensure_ascii_headers`.
* **The vendor returns float32.** Widening straight to float64 carries the
  float32 noise into every downstream number. Round to 7 significant figures to
  recover the intended decimals.
* **Use `PaddingType.NONE`.** A padded bar misreports which sessions actually
  traded, which then corrupts any quorum or liquidity rule built on top.
* **`&<CODE>_CCB` carries a `Delivery Month` column** (YYYYMM). That is the only
  way to compare a roll rule against the vendor's on the DECISION rather than on
  returns — back-adjusted levels have different offsets and cannot be compared.
  Four instruments have no vendor chain: FGBL9, FGBM9, FGBS9, YXT4.
* **Notice dates are absent for 40 of 63 instruments, and that is correct** —
  they are cash-settled, or terminate before delivery attaches (FX, CL, NG, RB).
  Fall back to last trade date for those; do not treat the gap as missing data.

## Prior art

`production/` has a working engine, its verification suite, and a written
record of what went wrong in it. Two findings there are worth reading before
rebuilding the equivalent pieces:

* `production/README.md` — the roll policy (three constraints: most liquid,
  never deliverable, actually trading) and why the market classifier, calendar
  rule and monotonic ratchet that preceded it were all unnecessary.
* `production/TODO.md` — the panel defects, and the three tests that assert
  properties of the retired panel.

`production/strategy_fix.py` records where the thesis PDF's equations
(3.11-3.13, 3.15, 3.21/3.25, 3.27, 3.33) disagree with the shipped code, and
measures the difference: over 37 years it is +0.02 of Sharpe with a worse
Calmar. The equations need correcting in the document; no published figure
moves.
