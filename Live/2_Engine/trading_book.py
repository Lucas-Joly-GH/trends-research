"""
Trading book: each instrument's worksheet, cut down to what a position needs.

Stage 2 of the pipeline.  `contract_cycles.py` refreshes the panel and decides
each market's roll rule; `front_contract.py` turns that into a worksheet with
every candidate month and every rule's answer side by side.  Most of that is
working-out.  This keeps what a book actually reads:

    date, symbol, open, close, <the one hold column this instrument is ruled on>

plus the derived series a position is decided and sized on -- the Panama
continuous prices, the three EWMACs and their mean, the adjusted return, and
the blended daily volatility:

    Continuous_O, Continuous_C, 16-64, 32-128, 64-256, TS_trend,
    daily_ret, daily_vol_abs, daily_vol, price_vol_curr_ann, VoV_inner, VoV_smooth, VoV_mean_ann, VoV, carry_hold,
    TS_trend_sign_UNCAPPED, XS_trend_sign_UNCAPPED, Trend_sign,
    Carry_sign, Skew_sign, VoV_sign, Sign_raw, fdm_raw, fdm_norm, FDM_MASTER, FDM_MASTER_smooth, SIGNAL, Carry_hold_O, Carry_hold_C,
    Carry, Carry_State, -Skew, XS_trend, FX_rate, price_vol_USD_ann

`price_vol_USD_ann` COMPLETES EQUATION 3.35: `price_vol_curr_ann x FX_rate`.
The former is 3.35 with the FX leg left off, so it carries whatever currency the
contract quotes in -- euros for FDAX9, yen for SJB -- and is not comparable
across markets.  This puts the whole book on one scale.

`FX_rate` IS THE INSTRUMENT'S OWN LOCAL CURRENCY -> USD, already resolved, so
nothing downstream has to know the currency map or which rate file to open.
Multiply a local-currency amount by it to get USD.  The currency comes from
instrument_mapping.csv (45 USD, 7 EUR, 3 GBP, 3 CAD, 2 JPY, 2 AUD, 1 HKD) and
the rate is `Derived_Rate` from `FX/<CCY>.csv`.  It is null only where a book
predates its currency's future -- YAP4 for 983 sessions, YXT4 for 375, both AUD
-- because there is no rate to carry and inventing one would be worse.

EVERY OUTPUT IS WRITTEN TWICE: `<name>.csv` to be read by a person, and
`<name>.parquet` to be read by a program.  They hold identical frames -- verified
equal on all 63 books and all 7 rate files -- and the parquet is 2.9x smaller
(333 MB -> 115 MB) and 2.4x faster to load.

THE PARQUET IS NOT AN OPTIMISATION, IT IS THE CORRECT ARTIFACT.  Csv has no way
to record that a column is a float, so polars infers dtypes from a bounded
prefix -- and every warm-up column here is blank for longer than that prefix:
SIGNAL opens with 849 nulls, VoV_sign 594, Skew_sign 511, XS_trend 256.  A naive
`pl.read_csv` therefore hands back 17 numeric columns as Strings in all 63
books, SIGNAL among them, and a comparison against one silently compares text.
Parquet carries dtypes in the file, so the question never arises and nulls stay
nulls.

Trimming history was considered and rejected: it would cost 53,487 rows (9.7%,
~3.4 years per instrument) and STILL not fix it, because Carry, Carry_hold_O,
Carry_hold_C and XS_trend hold 5,807 INTERIOR nulls that no prefix trim reaches.
See Trading_Book_Journal.md.

`load_book(inst)` and `load_fx(ccy)` read the parquet when it is current and fall
back to the csv WITH ITS SCHEMA NAMED, so both paths are correct.

WHICH hold column is not a choice made here.  It follows `Roll_Rule` in
contract_cycles.csv, which is where a market's rule is recorded and argued for:

    auto_roll                                  -> auto_roll_hold
    forced_roll                                -> forced_roll_hold
    +1_auto_roll                               -> +1_auto_roll_hold
    RS_forced_roll / LT_forced_roll /
    CS_forced_roll                             -> confirm_forced_roll_hold

The three confirmation variants share ONE column.  They were separate columns
under three names until the sheets were merged, and they were always the same
algorithm; the rule VALUES stay distinct because they record why each market was
cleared, which the column name never did.  See Front_Contract/Roll_Journal.md.

THE COLUMN IS NAMED `hold` IN EVERY FILE.  It would otherwise carry the rule's
own name, which differs per instrument, and every downstream reader would have
to address the last column by position rather than by name -- a fragile way to
read a file, and one that breaks silently the first time a column is inserted.
The provenance is not lost by renaming: `Roll_Rule` in contract_cycles.csv
records which rule each market runs, and argues for it.  `--keep-rule-name`
restores the per-instrument name.

ONE ROW PER SESSION: the contract actually held, and nothing else.  The
worksheet carries every listed month, with the session-level hold repeated down
all of them -- 6,768,788 rows across the book.  Keeping only the row whose own
symbol IS the hold gives 549,288, a 12x cut, and each row then carries that
contract's own open and close, which is what a position is priced on.
`--all-rows` returns the unfiltered worksheet for debugging.

THE HOLD COLUMN SURVIVES THE FILTER even though it now equals `symbol` on every
row.  It is kept so the file states plainly what it is -- a held-contract
series, not an arbitrary slice of a worksheet -- and so that `--all-rows`
produces the same schema, where the two columns genuinely differ.

A SESSION WHOSE HOLD IS BLANK PRODUCES NO ROW, and that must not pass quietly --
it is a hole in the position series, not an empty day.  Every run compares rows
written against sessions in the worksheet and names any instrument that lost
one.

IT ALSO WRITES THE FX RATES, one file per rate, to `FX/<CCY>.csv`:

    date, Derived_Rate, NDU_Rate, YF_Rate, NDU_diff_bp, YF_diff_bp, Status

They are built here rather than by a script of their own for two reasons. The
rates come from the SAME CACHED WORKSHEETS the books do -- the derivation needs
every listed month and its maturity, which is what a worksheet is -- so a
separate script would re-read and re-cache the panel to reproduce something this
one already holds.  And they are emitted on the BOOK'S OWN SESSION UNION, the
same grid XS_trend is scored on, so every FX file lines up row-for-row with every
book and a plain join on date is correct.  Only a caller inside this run knows
that grid.  See the FX section above `main()` and FX/FX_Journal.md.

    python trading_book.py                     every instrument, full history
    python trading_book.py --instrument ZC
    python trading_book.py --all-rows          every listed month, unfiltered
    python trading_book.py --keep-rule-name    hold column keeps the rule name
    python trading_book.py --start 2020-01-01
    python trading_book.py --no-fx             skip the FX files
    python trading_book.py --no-fx-checks      FX from the local panel only
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
import warnings
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
FC = HERE.parent / "1_Roll" / "Front_Contract" / "front_contract.py"
CYCLES = HERE.parent / "1_Roll" / "contract_cycles.csv"
MAPPING = HERE.parent / "instrument_mapping.csv"
BOOK = HERE / "Trading_book"

# Roll_Rule -> the worksheet column that rule is read out of.
HOLD_FOR = {
    "auto_roll": "auto_roll_hold",
    "forced_roll": "forced_roll_hold",
    "+1_auto_roll": "+1_auto_roll_hold",
    "RS_forced_roll": "confirm_forced_roll_hold",
    "LT_forced_roll": "confirm_forced_roll_hold",
    "CS_forced_roll": "confirm_forced_roll_hold",
}
KEEP = ["date", "symbol", "open", "close"]
CONT = ["Continuous_O", "Continuous_C"]


CACHE = HERE / ".cache" / "worksheets"


def _fingerprint(fc, inst: str, start: str, end: str, as_of) -> str:
    """Everything a cached worksheet depends on, in one hash.

    CONSERVATIVE ON PURPOSE.  A cache that misses a change serves stale prices
    and nothing looks wrong -- the worst failure this pipeline could have -- so
    the key covers every input rather than the ones that seem likely to move:

      front_contract.py source  the rule logic itself.  Edit a gate, change a
                                constant, fix a tie-break: everything rebuilds.
      contract_cycles.csv       ONLY the two fields worksheet() reads -- see
                                below.  This one is deliberately NOT the whole
                                file, and the reason matters.
      contracts.csv             the gate dates.  Shared, so a metadata refresh
                                correctly invalidates every instrument.
      the instrument's bars     name, size and mtime of every contract file.
      start / end / as_of       a narrow window must never satisfy a full run.

    Size+mtime rather than content for the bars: hashing 356 MB per run would
    cost more than it saves, and the panel is only ever written by our own
    append -- which always changes size -- or restored by git, which changes
    mtime.  Same size AND same mtime AND different content does not occur here.
    `--no-cache` is the escape hatch if it ever does.

    WHY contract_cycles.csv IS NOT HASHED WHOLE, having been for a long time.
    `worksheet()` reads exactly two things out of that file, both filtered to
    this one instrument: `has_notice`, via `gate()`, and `Dead_contracts`, via
    `dead_months()`.  Nothing else in it reaches the worksheet -- `Roll_Rule`
    included, because the worksheet computes EVERY rule's columns and the rule
    only decides which of them a book later reads.

    Hashing the whole file therefore invalidated on writes that could not change
    the result, and one of those writes is unavoidable: contract_cycles.py
    writes the table, runs `rule_scores` over all 63 worksheets, then writes it
    AGAIN with Mean_Auto_Best_V, Mean_Forced_Best_V, Roll_Rule and Unique_Roll
    filled in.  Every worksheet built during that scoring pass was keyed on the
    first version and thrown away by the second, so the same 63 worksheets were
    rebuilt by verify_holds minutes later -- about three minutes of duplicated
    work per run, for a change to four columns none of them read.

    THE TWO VALUES ARE HASHED THROUGH front_contract's OWN ACCESSORS, not by
    re-reading the columns here.  `fc.gate()` and `fc.dead_months()` are the
    exact calls the worksheet makes, so this cannot drift from what it actually
    depends on.  IF A NEW READ OF contract_cycles.csv IS EVER ADDED TO THE
    WORKSHEET PATH, IT MUST BE ADDED HERE TOO -- that is the one way this key
    can go stale, and a stale worksheet is silent.
    """
    h = hashlib.sha256()
    h.update(FC.read_bytes())
    h.update(f"gate={fc.gate(inst)}|"
             f"dead={','.join(sorted(fc.dead_months(inst)))}|".encode())
    h.update(f"{start}|{end}|{as_of}".encode())
    for f in (fc.NOTICE,):
        st = f.stat()
        h.update(f"{f.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    d = fc.CONTRACTS / inst
    for f in sorted(d.glob(f"{inst}-[0-9][0-9][0-9][0-9][A-Z].csv")):
        st = f.stat()
        h.update(f"{f.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def cached_worksheet(fc, inst: str, start: str, end: str, as_of,
                     use_cache: bool = True):
    """The worksheet, from cache when its inputs are unchanged.

    THE WORKSHEET IS ~98% OF THE COST and does not change when a formula does.
    Building it walks every session in sequence -- B_V_3 streaks and every
    ratchet carry state, so it cannot be vectorised -- at ~3s for ZC and ~8s for
    CL, against 0.05s and 0.21s for everything downstream of it.  Iterating on
    an EWMAC or a return definition should not pay that 287s again.

    Returns (frame, hit).
    """
    if not use_cache:
        return fc.worksheet(inst, start, end, as_of=as_of), False
    key = _fingerprint(fc, inst, start, end, as_of)
    pq, kf = CACHE / f"{inst}.parquet", CACHE / f"{inst}.key"
    if pq.is_file() and kf.is_file() and kf.read_text().strip() == key:
        return pl.read_parquet(pq), True
    w = fc.worksheet(inst, start, end, as_of=as_of)
    CACHE.mkdir(parents=True, exist_ok=True)
    w.write_parquet(pq)
    kf.write_text(key)
    return w, False


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rules() -> dict[str, str]:
    """{instrument: Roll_Rule}, only where a rule is set.

    An instrument with no rule is SKIPPED and named, not guessed at.  A book
    that quietly invents a rule for a market nobody cleared is worse than one
    that is short a market and says so.
    """
    t = pl.read_csv(CYCLES, infer_schema_length=0)
    out = {}
    for r in t.iter_rows(named=True):
        rule = (r.get("Roll_Rule") or "").strip()
        if rule:
            out[r["instrument"]] = rule
    return out


EWMAC_PAIRS = [(16, 64), (32, 128), (64, 256)]

# Equation 3.19 (body.tex:2455).  Carver's blended estimator.
VOL_EWM_SPAN = 32
VOL_EWM_MIN = 32
VOL_ROLL_WINDOW = 2560
VOL_ROLL_MIN = 256
VOL_W_SHORT = 0.70
VOL_W_LONG = 0.30

# Equation 3.17 -- sigma_inner, the inner leg of vol-of-vol.  A plain rolling
# standard deviation over a trading month; NOT the blended estimator above.
VOL_INNER_WINDOW = 21
VOL_INNER_MIN = 10

# Equation 3.29, the volatility gate.  Every one of these is DERIVED from the
# 20% risk budget rather than fitted, which is the point -- the gate adds no
# free parameter to the strategy.
#
#   GVOL_STEEPNESS   The sigmoid must be flat inside a normal vol regime and
#                    saturated outside it.  A standard logistic crosses 0.88 at
#                    z = +2 and 0.12 at z = -2, so putting the edge of a
#                    +/-20% band at z = 2 gives k . 0.20 = 2, k = 10.  It is
#                    2/VOL_TARGET, not a tuned 10.
#   GVOL_TRIGGER     1.0: the gate opens at parity, where recent vol equals the
#                    vol the position was sized on.
#   GVOL_FLOOR       0.50, the deepest cut the gate can make.  Bounded on
#                    purpose: a gate that could shut the book entirely would let
#                    the strategy score well merely by being dormant.
#   GVOL_SPAN        64, one trading quarter -- long enough to be an estimate,
#                    short enough to be "lately".
GVOL_SPAN = 64
GVOL_STEEPNESS = 10.0
GVOL_TRIGGER = 1.0
GVOL_FLOOR = 0.50

# Equation 3.17 read literally: the FULL set {r_i,tau} for tau = t-21 .. t, so
# min_samples equals the window rather than the thesis code's 10.
VOV_INNER_WINDOW = 21

# Equation 3.18 -- the outer leg, three trading months.  Read the same way:
# min_samples equals the window, so every value rests on a full 64 inner
# observations.  The thesis code uses 21 here (`vov_outer_min_periods`).
VOV_OUTER_WINDOW = 64

# Equation 3.19 -- the one-year average of VoV.  Thesis: `vov_avg_window`.
VOV_AVG_WINDOW = 256

# Equation 3.20 -- the direction overlay's lookback.
VOV_DIR_LOOKBACK = 64

# Equation 3.21 -- signal normalisation.  Phi is the target absolute value a
# signal converges to; W the window the average is taken over; the cap is
# Carver's, quoted in the same paragraph.
SIGNAL_PHI = 10.0
# W: the paper states 1,280 in both 3.21 and 3.25; production/s183 runs 256
# (`SCALAR_WINDOW`).  The choice is not cosmetic -- W is charged TWICE, once
# normalising each alpha and again normalising the aggregate, and the warm-ups
# stack: at 1,280 the master forecast starts at session 2,897 and only 59 of
# the 63 instruments ever produce one, against the paper's stated 62.
SIGNAL_W = 256
SIGNAL_CAP = 20.0

# Equation 3.23 -- forecast diversification multiplier.  Span and min_periods
# are the thesis's (`fdm_corr_span`, `fdm_min_periods`); the 1.0 floor and 2.0
# cap are the paper's, 2.0 being the analytic maximum for four equally
# weighted uncorrelated signals: 1/sqrt(4 x 0.25^2) = 2.
FDM_CORR_SPAN = 512
FDM_MIN_PERIODS = 256
FDM_FLOOR = 1.0
FDM_CAP = 2.0
FDM_VAR_FLOOR = 0.01

# Equation 3.13.  The z-score window matches the SLOW EWMAC pair (64, 256), so
# an instrument the time-series component calls trending is judged against the
# others over the same horizon.
XS_LOOKBACK = 256
XS_MIN_INSTS = 3

# Equation 3.15.  Trading days per year, over a CALENDAR-day gap -- the
# paper's own mixed convention, stated explicitly in the text.
TRADING_DAYS_YEAR = 256

# Equation 3.16.
SKEW_WINDOW = 256
SKEW_MIN = SKEW_WINDOW  # a FULL window; see the note in skew()


def ewmac(close: pl.Series) -> dict:
    """EWMAC(f, s) for each configured pair.  {"16-64": Series, ...}.

    Equation 3.10, verbatim:

        EWMA_N(P_t) = a * P_t + (1 - a) * EWMA_N(P_t-1),      a = 2 / (N + 1)

    and 3.11:

        EWMAC(f, s) = EWMA_f(P_t) - EWMA_s(P_t)

    `adjust=False` IS THE WHOLE POINT.  polars offers two EWMA conventions and
    only this one is the recursion above, seeded at P_0.  `adjust=True` computes
    the bias-corrected weighted mean instead, which differs sharply while the
    window is filling -- on a 7-point check with N=4 it was off by 0.225 where
    adjust=False matched the hand-rolled recursion to 1.4e-14.  Getting this
    wrong would shift every signal in the book and nothing would look broken.

    COMPUTED ON Continuous_C AND NOTHING ELSE.  The raw close jumps at every
    roll, and an EWMAC of it would read those jumps as trend -- the back-adjusted
    series exists precisely so that a moving average sees price moves rather
    than contract changes.

    ONE ROW PER SESSION IS REQUIRED.  The worksheet repeats each session across
    every listed month; running this over that frame would feed the same price
    in N times per day and compress the effective span by a factor of N.  The
    caller passes the held-only series.

    TS_trend (3.12) is the arithmetic mean of the three, and is returned in the
    same dict -- one pass over the price, one place the speeds are defined.

    THE FIRST YEARS OF THE SLOW PAIRS ARE SEEDED, NOT MEASURED.  Both EWMAs
    start at P_0, so EWMAC starts at exactly 0 and converges as the seed decays
    with weight (1 - a)^t.  That decay is slower than it looks: a 256-span EWMA
    still carries 13.5% of its seed after 256 sessions, and does not reach 2%
    until ~512.  Its seed half-life is 88.7 sessions.

    Measured on CL, adjust=False against adjust=True for EWMAC(64,256), the
    worst gap between the two conventions is 0.4324 in year one against a mean
    |EWMAC| of 0.3661 -- the convention matters more than the signal does --
    then 0.1087 in year two, 0.0193 in years three to four, and 0.0007
    thereafter.  Treat the first two years of the 64-256 pair as burn-in, not
    the first one.

    The values are left in rather than nulled: where the warm-up ends is a
    modelling choice for whoever fits on this, and silently blanking rows would
    hide that the choice was ever made.  A backtest should discard its own
    burn-in.
    """
    out = {}
    for f, s in EWMAC_PAIRS:
        fast = close.ewm_mean(alpha=2.0 / (f + 1), adjust=False)
        slow = close.ewm_mean(alpha=2.0 / (s + 1), adjust=False)
        out[f"{f}-{s}"] = fast - slow

    # Equation 3.12: the mean of the three speeds.
    #
    #     f_trend^TS = (1/3) [ EWMAC(16,64) + EWMAC(32,128) + EWMAC(64,256) ]
    #
    # READ AS GROUPING BRACKETS, NOT ABSOLUTE VALUE, and the distinction is the
    # whole signal: |.| would make TS_trend non-negative on every session, so a
    # trend system built on it could only ever be long and would hold through
    # every downtrend in the book.  A trend forecast carries its direction in
    # its sign.  If the thesis really does mean |.|, this is the one line to
    # change -- and the consequence above is what to expect.
    total = None
    for name in (f"{f}-{s}" for f, s in EWMAC_PAIRS):
        total = out[name] if total is None else total + out[name]
    out["TS_trend"] = total / len(EWMAC_PAIRS)
    return out


def _blended_std(x: pl.Series) -> pl.Series:
    """0.70 * EWM_32 + 0.30 * Rolling_2560 -- Eq 3.19's estimator, on any input.

    Used by `daily_vol_abs` only.  `daily_vol` is Eq 3.17, a plain rolling
    std, and deliberately does NOT share this estimator -- see that function.

    The conventions:
    `adjust=True` and `bias=False` on the short leg (the thesis implementation's
    pandas default, debiased by sum_w^2 / (sum_w^2 - sum_w2)), ddof=1 on the
    long one.  See `daily_vol_abs` for why the blend is 70/30 and why the first
    256 rows are null.
    """
    short = x.ewm_std(span=VOL_EWM_SPAN, adjust=True, bias=False,
                      min_samples=VOL_EWM_MIN, ignore_nulls=False)
    long = x.rolling_std(VOL_ROLL_WINDOW, min_samples=VOL_ROLL_MIN, ddof=1)
    return VOL_W_SHORT * short + VOL_W_LONG * long


def daily_vol_abs(cont_close: pl.Series) -> pl.Series:
    """Blended daily standard deviation, IN PRICE UNITS, verbatim:

        sigma_i,t = 0.70 * EWM_32(std(dP_i))_t + 0.30 * Rolling_2560(std(dP_i))_t

    THE `_abs` IS THE UNITS, AND IT IS LOAD-BEARING.  This is an ABSOLUTE
    volatility -- ticks, dollars, whatever the contract is quoted in -- not the
    volatility of `daily_ret`.  The two are not interchangeable and nothing
    would flag confusing them, so the name carries the distinction rather than
    leaving it to a docstring nobody re-reads.  A return volatility would be
    dimensionless and comparable across markets; this one is not, and must not
    be read as a percentage.

    ON PRICE CHANGES, NOT RETURNS, and that is the whole character of the
    number.  It is in the contract's own price units -- ticks, not percent --
    because it exists to size a position, and contracts are bought in price
    units.  A return volatility would be dimensionless and comparable across
    markets, which is a different and also useful quantity; it is not this one.
    The units matter downstream: the EWMAC columns are price differences too,
    so `EWMAC / daily_vol_abs` cancels correctly.  Dividing them by a return vol
    would produce a number with no meaning and no symptom.

    dP IS DIFFERENCED FROM THE ADJUSTED SERIES, matching the thesis, which
    takes it from the Panama close.  The raw close jumps at every roll and its
    differences would read those jumps as volatility -- one fake outlier per
    roll, ~200 of them on ZC, biasing the estimate high exactly at the dates
    where a position is being changed.

    WHY BLEND AT ALL, rather than take the 32-day leg on its own.  Volatility
    clusters, so a short window bottoms out just as risk is about to arrive:
    after a quiet stretch EWM_32 returns a small sigma, and anything sizing off
    it takes its largest position immediately before the break.  The 2,560-day
    leg is ten years of the market's own history; it barely moves, so it acts
    as a floor made of evidence rather than of an arbitrary constant.  70/30
    tracks the current regime while never letting the estimate collapse.

    `adjust=True` HERE, AGAINST `adjust=False` IN `ewmac` ABOVE.  Both live in
    this file on purpose and neither is a slip.  Eq 3.10 defines the EWMA of
    PRICE as the seeded recursion, so ewmac uses adjust=False.  The volatility
    legs follow the thesis implementation, which uses the pandas default: the
    weighted mean, debiased by sum_w^2 / (sum_w^2 - sum_w2) at bias=False.

    ON THIS COLUMN THE CHOICE IS UNOBSERVABLE, and that is worth knowing before
    anyone "fixes" the inconsistency.  The two conventions differ only by the
    seed weight (1 - a)^t, and nothing is emitted here until bar 256 because
    the long leg needs min_samples=256 -- by which point a span-32 EWM has
    (1 - 0.0606)^256, i.e. 0.0000%, of its seed left.  Measured across CL, ES,
    ZC, SR3, VX and BTC the largest gap between adjust=True and adjust=False is
    1.2e-04 on BTC and ~1e-08 elsewhere: float arithmetic, not modelling.
    Contrast ewmac, where the same switch moves year-one values by more than
    the signal's own magnitude.

    THE FIRST 256 SESSIONS ARE NULL, and deliberately so.  The long leg needs
    min_samples=256 before it will report anything, and 0.7*x + 0.3*null is
    null.  That is the estimator refusing to state a risk it cannot support
    yet, not a gap to be filled -- a backtest should discard its own burn-in,
    the same way it must for the slow EWMAC pairs.
    """
    return _blended_std(cont_close.diff())


def s_g_vol(cont_close: pl.Series, vol_abs: pl.Series) -> pl.Series:
    """Equation 3.29 -- the volatility gate.

        g_vol(i,t) = 1 - (1 - 0.50) . sigma( 10.0 . ( sigmahat / sigmatarget - 1 ) )

    WHAT THE RATIO ACTUALLY COMPARES.  Both legs are `price_vol_USD_ann` -- one
    contract's annualised USD volatility -- differing only in the estimator
    underneath:

        numerator    the SAME quantity computed exclusively as a 64-day EWMA:
                     a fast read of how volatile this market has been LATELY.
        denominator  `price_vol_USD_ann` as the book already carries it, on the
                     blended 70/30 estimator of Eq 3.19 -- the deliberately
                     stable number the position was SIZED with.

    So the ratio asks one question: has this instrument's recent volatility run
    above the budget the position was built on?  Above 1, the gate closes and
    exposure is cut, which is a tail-risk brake rather than a forecast.

    THE USD AND ANNUALISATION FACTORS CANCEL, so this takes price-unit vols.
    Both legs are (vol_abs . pointsize . sqrt(256) . FX); everything but the
    vol_abs divides out exactly, leaving `fast / daily_vol_abs`.  That is not
    just tidier -- it is what lets the gate be computed HERE, per instrument, in
    a worker that has no FX_rate yet, instead of waiting for the panel.

    THE FAST LEG MATCHES THE SLOW LEG'S CONVENTIONS, deliberately: the same dP
    from the Panama close, the same `adjust=True, bias=False`.  A ratio of two
    estimators is only meaningful if the difference between them is the window
    and nothing else.

    1.0 WHERE THE RATIO CANNOT BE COMPUTED, meaning no reduction.  The
    denominator is null for the first 256 sessions (its long leg needs them),
    so a gate would otherwise be inventing a haircut out of a warm-up.  Note
    this diverges from the thesis code, which fills the missing RATIO with 1.0
    and therefore applies `1 - 0.5 . sigma(0)` = 0.75 -- a permanent 25% cut
    across every instrument's first year, which reads as an artifact of the
    fill rather than an intended brake.

    BOUNDED [0.50, 1.00] BY CONSTRUCTION, since sigma() is in (0,1): the most
    it can ever do is halve the exposure, however extreme the volatility.  That
    bound is the paper's, and its reason is overfitting -- a gate that can shut
    the book entirely would let the strategy look good simply by being absent.
    """
    fast = cont_close.diff().ewm_std(span=GVOL_SPAN, adjust=True, bias=False,
                                     min_samples=GVOL_SPAN, ignore_nulls=False)
    ratio = fast / vol_abs
    z = GVOL_STEEPNESS * (ratio - GVOL_TRIGGER)
    gate = 1.0 - (1.0 - GVOL_FLOOR) * (1.0 / (1.0 + (-z).exp()))
    return gate.fill_null(1.0)


def daily_vol(daily_ret: pl.Series,
              window: int = VOL_INNER_WINDOW,
              min_samples: int = VOL_INNER_MIN) -> pl.Series:
    """Equation 3.17 -- sigma_inner, a plain rolling std of the daily returns.

        sigma_inner(i,t) = std({r_i,tau} for tau = t-21 .. t)

    A PLAIN STANDARD DEVIATION, DELIBERATELY NOT THE 70/30 BLEND that
    `daily_vol_abs` uses.  The two columns answer different questions and are
    NOT the same estimator in different units:

      daily_vol_abs   Eq 3.19, blended, in price units.  A SIZING input, so it
                      is built to be stable -- the 2,560-day leg exists to stop
                      the estimate collapsing in a calm spell and over-levering
                      the book.
      daily_vol       Eq 3.17, plain, dimensionless.  A MEASUREMENT, and the
                      inner leg of vol-of-vol.  It must be free to move, because
                      the whole point of the outer std is to measure how much
                      this one moves.  Blending it would damp the very signal
                      3.18 is about to extract, and vol-of-vol would read as
                      quieter than it is.

    ONE TRADING MONTH, 21 sessions, with min_samples 10 -- the thesis
    convention (`vov_inner_std_w`, `vov_inner_mp`).  Short on purpose: this is
    meant to track the current regime closely enough that its own variability
    is informative.

    ddof=1, the sample standard deviation, matching the thesis kernel.

    ON RETURNS, per the paper.  Note the shipped thesis code takes the inner
    std of dP instead and calls the result `daily_vol` as well; same name, same
    windows, different input, so the two are not comparable number for number.

    ONE CAUTION ON READING IT.  An interest-rate future is quoted as 100 minus
    a rate, so its PRICE barely moves in percentage terms and its return vol
    comes out near zero -- SR3 and LEU9 will look like the calmest things in
    the book.  That is the quoting convention, not a statement that they are
    safe: their economic risk lives in the rate, not in the percentage move of
    the price.
    """
    return daily_ret.rolling_std(window, min_samples=min_samples, ddof=1)


def skew(daily_ret: pl.Series, window: int = SKEW_WINDOW,
         min_samples: int = SKEW_MIN) -> pl.Series:
    """Equation 3.16 -- rolling skewness of daily returns, sign inverted.

        f_skew(i,t) = - Skew({r_i,tau} for tau = t-256 .. t)

    THE MINUS SIGN IS THE STRATEGY, not a convention -- which is why the output
    column is called `-Skew` and not `Skew`.  An asset with strongly
    negative skew loses often and small, then occasionally very large -- it
    behaves like a sold insurance policy, and investors demand a premium to
    hold it.  Inverting the sign turns "this has been ugly" into a BUY, so the
    signal collects that premium: long the unattractive, short the attractive.
    Drop the minus and the strategy runs exactly backwards, paying the premium
    instead of earning it, and nothing in the output would look wrong.

    ON `daily_ret`, THE RETURN, NOT ON PRICE CHANGES.  The paper says returns
    and returns is what this uses.  Note the shipped thesis code takes skew of
    dP instead; skewness is invariant to a CONSTANT rescaling, but dividing by
    a price that itself moves is not constant, so the two are genuinely
    different numbers, not two spellings of one.

    BIAS-CORRECTED G1, matching the thesis kernel and pandas' own `.skew()`:

        G1 = n / ((n-1)(n-2)) * sum(((x - xbar) / s)^3),   s with ddof=1

    polars defaults to `bias=True`, the uncorrected third moment, so this is
    passed explicitly.  The two converge for large n but differ by ~2% at
    n = 128, which is exactly where this signal starts reporting.

    A FULL WINDOW IS REQUIRED, WHICH DIVERGES FROM THE THESIS KERNEL, and the
    divergence is worth stating exactly because it is deliberate.

    The thesis uses min_periods = window // 2 and a NaN-aware estimator, so it
    reports from session 128 on whatever finite points the window holds.  polars
    blanks any window containing a null instead.  `daily_ret` always has one --
    its first session has no previous close to divide by -- so under polars the
    signal cannot start until that null has scrolled out, at session 256.

    Measured, the consequence is exactly 128 rows of extra burn-in per
    instrument and nothing else: verified across all 63 files, `daily_ret` has
    NO interior nulls, so from session 256 onward both estimators see the same
    256 finite returns and agree to 5e-16.  Passing 128 here would not restore
    the thesis behaviour anyway -- polars would still blank those rows -- it
    would only look as though it did.  Requiring the full window is also the
    literal reading of 3.16, which is written over the SET of 256 returns.
    """
    return -daily_ret.rolling_skew(window, bias=False, min_samples=min_samples)


def vov_inner(daily_ret: pl.Series,
              window: int = VOV_INNER_WINDOW) -> pl.Series:
    """Equation 3.17, read literally -- sigma_inner over a FULL 21 sessions.

        sigma_inner(i,t) = std({r_i,tau} for tau = t-21 .. t)

    THE ONLY DIFFERENCE FROM `daily_vol` IS min_samples, and it shows up
    nowhere but the burn-in edge.  `daily_vol` follows the thesis code, which
    reports once 10 of the 21 observations exist (`vov_inner_mp`); this reports
    only once all 21 do, which is what the equation's set notation actually
    says.  Measured on CL that is 11 rows -- the two columns are otherwise
    bitwise identical, max difference 0.000e+00 across 10,762 values.

    KEPT AS A SEPARATE COLUMN RATHER THAN FOLDED INTO `daily_vol` because the
    two are the inputs to different things: `daily_vol` is the general-purpose
    return volatility, and this is specifically the inner leg 3.18's outer
    standard deviation is taken OF.  Building the outer leg on a series whose
    first rows rest on as few as 10 observations would let the noisiest
    estimates in the sample drive the regime signal at the start of every
    instrument's history.

    ddof=1, on returns, no blending, no annualisation -- as in `daily_vol`, and
    for the same reasons.
    """
    return daily_ret.rolling_std(window, min_samples=window, ddof=1)


def vov_smooth(inner: pl.Series,
               window: int = VOV_OUTER_WINDOW) -> pl.Series:
    """Equation 3.18 -- the standard deviation of sigma_inner over 64 sessions.

        VoV(i,t) = std({sigma_inner(i,tau)} for tau = t-64 .. t)

    A STANDARD DEVIATION OF A STANDARD DEVIATION, which is the whole idea: the
    inner leg says how volatile the instrument is right now, and this says how
    much THAT has been moving.  A market can be persistently wild and score low
    here, and a normally quiet one that has just started lurching scores high.
    So it reads a REGIME, not a direction -- the paper is explicit that it "is
    not a proper conviction".

    NOTE THE PAPER CALLS THIS SMOOTHING.  It is not: smoothing a series would
    damp its variation, and taking a standard deviation MEASURES that variation
    instead.  The name is kept because it is the paper's, but nothing here
    averages or filters sigma_inner.

    BUILT ON `VoV_inner`, NOT ON `daily_vol`.  The two are bitwise identical
    wherever both exist, but daily_vol also reports rows resting on as few as
    10 returns.  Feeding those in would let the least reliable estimates in the
    sample set the regime at the start of every instrument's history, which is
    exactly where a spurious signal is hardest to notice.

    A FULL WINDOW, matching the strict reading used for 3.17: 3.18 is written
    over the SET of 64 inner values, so all 64 must exist.  The thesis code
    instead uses min_periods = 21, which on CL starts the series at session 41
    instead of 84 and yields 43 extra rows.  Where both report they agree to
    2.8e-17 -- float noise from polars taking a different summation path, not a
    difference in the estimate.  So the choice buys nothing but the burn-in
    edge, and costs nothing but 43 rows an instrument.

    Because sigma_inner itself needs 21 sessions, the first value here lands at
    session 84 -- 21 + 64 - 1 -- and that is a genuine floor, not a choice:
    nothing before it is measurable from a full window of full windows.

    ddof=1, as everywhere else in this file.
    """
    return inner.rolling_std(window, min_samples=window, ddof=1)


def vov_mean_ann(smooth: pl.Series,
            window: int = VOV_AVG_WINDOW) -> pl.Series:
    """Equation 3.19 -- the one-year mean of VoV.

        VoV_bar(i,t) = (1/256) * sum(VoV(i,tau) for tau = t-256 .. t)

    THE OPERATION IS THE MEAN; THE ANNUALISATION IS A NO-OP.  3.19 divides a
    sum of 256 terms by 256, which is an average -- annualising a standard
    deviation would be multiplying by sqrt(256), and that is not written here.
    It does not matter, and the reason is worth recording: 3.20 uses this only
    as the denominator of VoV / VoV_bar, a ratio of two quantities in the same
    units, so any constant factor cancels exactly.  Verified on CL: annualising
    BOTH legs leaves 3.20's term bitwise identical on all 10,433 rows.

    THE ONE WAY IT COULD BITE is applying sqrt(256) to only one leg.  Scaling
    just the denominator moves the median term from +0.1189 to +0.9449 and
    flips the sign on 43.1% of sessions -- the strategy would run inverted on
    nearly half the book.  Hence this function returns the plain mean and
    leaves any scaling to the ratio, where it provably cannot matter.

    WHAT IT IS ACTUALLY FOR is the denominator of the next step.  On its own a
    VoV level says nothing comparable across markets -- SR3's vol-of-vol and
    BTC's differ by orders of magnitude for reasons that have nothing to do
    with regime.  Dividing today's VoV by this one-year average turns it into
    "how unsettled is this market RIGHT NOW versus its own normal", which is
    comparable, and which is what the thesis then does
    (`vov / vov_avg - 1`).

    A FULL WINDOW, as for 3.17 and 3.18.  The thesis uses min_periods = 64
    here.  Because VoV_smooth itself only begins at session 84, this column
    starts at 84 + 256 - 1 = 339 -- a year of three-month windows of
    one-month windows, which is simply what the equation asks for.

    Null, not zero, before then: a zero average would make the ratio in the
    next step infinite rather than undefined.
    """
    return smooth.rolling_mean(window, min_samples=window)


def vov_signal(smooth: pl.Series, mean_ann: pl.Series,
               cont_close: pl.Series,
               lookback: int = VOV_DIR_LOOKBACK) -> pl.Series:
    """Equation 3.20 -- the finished vol-of-vol forecast.

        f_vov(i,t) = sign(pct_change_64(P)) x ( -( VoV / VoV_bar - 1 ) )

    TWO PARTS, AND ONLY ONE OF THEM IS A CONVICTION.  The right factor is the
    size: how unsettled this market is versus its own normal, INVERTED, so that
    a calm regime scores positive and a turbulent one negative.  That is the
    brake -- high vol-of-vol predicts poor risk-adjusted returns (Baltussen,
    van Bekkum & van der Grient 2018), so the signal pulls exposure in.  The
    left factor is only a direction: it decides which way the brake is applied,
    and carries no strength of its own.

    THE SIGN IS TAKEN ON THE PRICE DIFFERENCE, NOT ON A RATIO, and that matters
    far more here than it looks.  A literal `pct_change_64` on the Panama close
    is P_t / P_t-64 - 1, whose sign INVERTS whenever P_t-64 is negative -- and
    back-adjusted prices go negative often: 14 instruments on this panel, HO on
    88.9% of its bars, GAS 87.6%, CC 71.8%.  On those the overlay would point
    the wrong way for most of their history, and nothing downstream would look
    wrong.  Taking the sign of (C_t - C_t-64) is identical wherever the ratio
    is valid and correct everywhere else, and is the same choice already made
    for `daily_ret` and `xs_return`.

    EXACT ZERO COUNTS AS POSITIVE, matching the thesis (`pct >= 0`).  A true
    sign() would return 0 and silently delete the forecast on those bars; here
    that is 760 of 545,256 (0.14%).  This is knife-edge by construction -- a
    one-tick difference flips the overlay -- and it is left that way on
    purpose.  An epsilon band would reclassify every genuinely small move as
    flat, which diverges further, not less.

    Null wherever either VoV leg is null, so the column starts at session 339
    with `VoV_mean_ann`.
    """
    ratio = -((smooth / mean_ann) - 1.0)
    prev = cont_close.shift(lookback)
    diff = (cont_close - prev).to_numpy().astype(np.float64)
    # THE NOISE FLOOR, and it is not the epsilon band the docstring rejects.
    #
    # `diff` subtracts two Panama levels of the same magnitude, so exact
    # cancellation leaves float64 rounding of order ulp(P) -- 1.4e-14 at a bund
    # price of 117, 4.5e-13 on the Euro Stoxx at 2,670.  On a bar where the
    # market genuinely returned to the same level after 64 sessions, THE SIGN IS
    # THEN DECIDED BY ROUNDING.  And the Panama anchor moves on EVERY panel
    # refresh, so the rounding changes daily: rebuilding the book as of
    # 2026-06-30 instead of 2026-08-28 flipped 8 such bars across 5 instruments,
    # and through the POOLED FDM -- which coupled every instrument to those 8 --
    # perturbed SIGNAL on 59 of 63 books across three decades.
    #
    # 951 of 542,849 bars (0.175%) sit on this edge, 47 of them since 2024 and
    # 14 in 2026, so it reaches live signals rather than only history.
    #
    # A RELATIVE 1e-9 IS FIVE ORDERS OF MAGNITUDE BELOW A TICK (ticks here run
    # 1e-4 to 1e-2), so it cannot reclassify a genuine move -- which is exactly
    # the objection the docstring raises against an epsilon band, and it does
    # not apply at this scale.  What it removes is only the rounding.
    scale = np.maximum(np.abs(cont_close.to_numpy().astype(np.float64)),
                       np.abs(prev.to_numpy().astype(np.float64)))
    diff = np.where(np.abs(diff) <= 1e-9 * np.maximum(scale, 1.0), 0.0, diff)
    direction = np.where(np.isnan(diff), np.nan, np.where(diff >= 0, 1.0, -1.0))
    return pl.Series(direction).fill_nan(None) * ratio


def normalise_signal(f: pl.Series, phi: float = SIGNAL_PHI,
                     window: int = SIGNAL_W,
                     cap: float | None = SIGNAL_CAP) -> pl.Series:
    """Equation 3.21 -- put a raw signal on the +/-10 conviction scale.

        S_t = Phi / ( (1/W) * sum(|f_tau|) for tau = t-W .. t ),  Phi=10, W=1,280

    and the column returned is f_t * S_t, clipped to +/-`cap`.

    WHAT IT IS FOR.  The raw alphas are in incompatible units -- Carry is a
    percentage rate, -Skew is a dimensionless third moment, TS_trend is in the
    contract's own price units.  Their magnitudes differ by orders of
    magnitude, so any weighted blend of them is really weighted by scale rather
    than by the stated weights.  Dividing each by its own long-run mean
    absolute value removes the units entirely: afterwards every signal averages
    +/-10 regardless of what it measures, and "0.25 each" or "0.5 each" means
    what it says.

    IT SCALES, IT DOES NOT RE-RANK.  S_t is one positive number per session,
    shared by the whole series, so the sign is untouched and the ordering
    within an instrument is untouched.  Only the units change.

    THE CAP IS NOT COSMETIC.  +/-20 is two normal convictions; beyond that the
    paper's own reading is that the signal has stopped being informative and is
    "taking incoherent positions".  Clipping bounds the position a single
    outlier can open.  Note it makes the operation NON-LINEAR, so normalising
    then blending is NOT the same as blending then normalising -- see
    `trend_sign`.

    GAPS ARE FILLED WITH ZERO, THE WINDOW IS THEN STRICT, and that pairing is
    the whole design.  3.21 is written as a clean sum over W terms, which
    silently assumes no gaps; real signals have them.  Two ways out, and they
    are not equivalent:

      tolerate  average over whatever exists and require some minimum count.
                Works, but invents a threshold the paper never states -- and
                the instrument count turns on it: 63 tradable at 640, 62 at
                1,100, 60 at 1,280.  Choosing it to reach a known answer would
                be fitting, not deriving.
      fill      set interior gaps to 0 and keep the window strict.  No free
                parameter, and it is what the thesis implementation does
                (`assemble.py` maps the XS forecast back with
                `_from_grid(..., 0.0)`).

    This takes the second.  Measured across the book the two barely differ in
    distribution -- mean |signal| 9.07 against 9.09, cap rate 9.9% against
    10.0% -- but only the fill has no knob in it.

    ZERO MEANS "NO VIEW", AND ONLY AFTER THE SIGNAL EXISTS.  Leading rows
    before an instrument's first value are left null, not zeroed: those are
    "not computable yet", which is a different statement from "flat today".
    Zeroing them would drag the mean absolute value down and inflate every
    scalar that saw them.

    MISSING MEANS NaN *OR* NULL.  `cross_sectional_z` builds its output from a
    numpy array, so XS_trend's gaps once arrived as NaN, where `is_not_null()`
    reports True and `fill_null` does nothing -- an earlier version of this
    function ignored every gap it existed to handle, and TS_trend, which has
    no gaps, went on working and hid it.  np.isfinite catches both.

    W = 1,280 is five trading years, a large burn-in, and is the paper's own
    figure.  It is what excludes ETH: 1,383 sessions is not enough history to
    normalise, which is why the paper trades 62 of the 63 instruments here.

    `cap=None` SCALES WITHOUT CLIPPING, for a signal that is not yet a
    forecast.  The +/-20 bound belongs to a FINISHED alpha; applying it to a
    half that is about to be averaged truncates twice, and the second truncation
    can only ever pull the result in.  See `trend_sign`.
    """
    # MISSING MEANS NaN *OR* NULL, and conflating the two is a live hazard in
    # this file: `cross_sectional_z` builds its output from a numpy array, so
    # XS_trend's gaps arrive as NaN, where `is_not_null()` reports True and
    # `fill_null` does nothing.  A "null-tolerant" version written against
    # nulls alone silently ignored every gap it existed to handle -- and TS
    # trend, which has no gaps, went on working and hid it.  np.isfinite is
    # the one test that catches both.
    x = f.to_numpy().astype(np.float64)
    fin = np.isfinite(x)
    if not fin.any():
        return pl.Series(np.full(x.size, np.nan)).fill_nan(None)
    filled = x.copy()
    first = int(np.argmax(fin))
    filled[first:] = np.where(fin[first:], filled[first:], 0.0)
    present = np.isfinite(filled)
    num = pl.Series(np.where(present, np.abs(filled), 0.0)).rolling_sum(
        window, min_samples=1).to_numpy()
    den = pl.Series(present.astype(np.float64)).rolling_sum(
        window, min_samples=1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_abs = np.where(den >= window, num / np.maximum(den, 1.0), np.nan)
        out = filled * (phi / mean_abs)
        if cap is not None:
            out = np.clip(out, -cap, cap)
    return pl.Series(np.where(np.isfinite(out), out, np.nan)).fill_nan(None)


def trend_sign(ts_sign: pl.Series, xs_sign: pl.Series,
               w_ts: float = 0.5, w_xs: float = 0.5) -> pl.Series:
    """Equation 3.14, over the two halves AFTER each has been normalised.

        f_trend = 0.5 * TS_trend_sign + 0.5 * XS_trend_sign

    THE ORDER IS THE POINT, AND IT IS NOT THE PAPER'S.  3.12 leaves f^TS in
    price units and 3.13 leaves f^XS as a z-score, so blending them as written
    weights by price scale rather than by 0.5: measured on this panel the TS
    share runs from 0.4% on 6M to 99.9% on BTC, with 35 of 63 instruments
    taking over 90% of their trend signal from one side alone.  Normalising
    each half first puts both on the +/-10 scale, after which 0.5/0.5 means
    half each.

    3.21 CANNOT FIX THIS FROM WHERE THE PAPER PUTS IT.  Section V normalises
    the four finished alphas, which is after 3.14 -- and a scalar applied to a
    finished blend rescales it while leaving the mix untouched.  The thesis
    IMPLEMENTATION does normalise both sides first (`assemble.py`, Stage
    B.1/B.2); the equations simply do not say so.

    THE CAP IS APPLIED ONCE, HERE, AND THE INPUTS ARE UNCAPPED.  Capping each
    half first and then the blend truncates twice, and the second truncation
    can only pull inward: with both halves clipped the finished trend hit the
    bound on 4.8% of sessions against 9.5-13.9% for carry, skew and VoV, and
    averaged 8.69 where they averaged 9.2-9.5.  Trend was quieter than its
    peers for a purely mechanical reason, so 3.22's equal 25% weight was not
    delivering equal conviction.  The +/-20 bound belongs to a finished alpha;
    a half that is about to be averaged is not one.

    Note this differs from the thesis implementation, which caps the EWMAC legs
    and the XS forecast before the sub-blend and then again after.

    Null where either half is null: a trend view needs both, and a session
    with no cross-section is not the same as one where the cross-section says
    zero.
    """
    return (w_ts * ts_sign + w_xs * xs_sign).clip(-SIGNAL_CAP, SIGNAL_CAP)


def _recursive_sum(v: np.ndarray, decay: float) -> np.ndarray:
    """s_t = v_t + decay * s_{t-1}, the accumulator every EWM statistic needs.

    Runs along axis 0, so a (T,) series and a (T, K) block of K series both
    work; the block form is what makes an N-instrument correlation affordable.

    `lfilter` IS THE SAME RECURSION, NOT AN APPROXIMATION OF IT.  A one-pole IIR
    filter with b=[1], a=[1, -decay] expands to exactly s_t = v_t + decay
    s_{t-1} with s_-1 = 0, which is the loop this used to be.  Checked rather
    than assumed: over 12,552 random values the two agree to a max absolute
    difference of 0.0 -- bit-identical, not merely close -- so no number
    anywhere in this file moves because of it.

    THE REASON TO CARE IS THE PORTFOLIO STAGE.  The loop cost 2.05 ms per
    series, which is nothing for the FDM's 10 pairs but 16.5 s for the 2,016
    pairs an IDM over 63 instruments needs.  Batched through lfilter the same
    work is 0.8 s.
    """
    from scipy.signal import lfilter
    return lfilter([1.0], [1.0, -decay], np.asarray(v, dtype=np.float64), axis=0)


def ewm_corr_4(X: np.ndarray, span: float, min_periods: int) -> np.ndarray:
    """(T,4,4) exponentially weighted correlation matrices.

    polars has corr, rolling_corr, ewm_mean and ewm_std but NO exponentially
    weighted covariance, so the primitive is built here from the same four
    decaying accumulators an EWM variance needs, extended to cross-products.

    pandas' convention throughout -- adjust=True with the bias correction
    sum_w^2 / (sum_w^2 - sum_w2) -- because that is what the thesis computes
    (`kernels.ewm_corr_pairwise`), and this is differential-tested against it.

    PAIRWISE, NOT JOINT, NaN HANDLING: each pair accumulates only over bars
    where BOTH of its series are observed, as pandas does.  A gap in one signal
    therefore degrades only the pairs that touch it, rather than blanking the
    whole matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    T, K = X.shape
    alpha = 2.0 / (span + 1.0)
    decay = 1.0 - alpha
    mask = np.isfinite(X)
    vals = np.where(mask, X, 0.0)
    mf = mask.astype(np.float64)

    cov = np.full((T, K, K), np.nan)
    for i in range(K):
        for j in range(i, K):
            both = mf[:, i] * mf[:, j]
            xi, xj = vals[:, i] * both, vals[:, j] * both
            sw = _recursive_sum(both, decay)
            sw2 = _recursive_sum(both, decay * decay)
            sx = _recursive_sum(xi, decay)
            sy = _recursive_sum(xj, decay)
            sxy = _recursive_sum(xi * xj / np.where(both > 0, 1.0, 1.0), decay)
            with np.errstate(invalid="ignore", divide="ignore"):
                mx, my = sx / sw, sy / sw
                c = sxy / sw - mx * my
                denom = sw * sw - sw2
                c = np.where(denom > 0, c * (sw * sw) / denom, np.nan)
            n_obs = np.cumsum(both)
            c = np.where(n_obs >= max(min_periods, 1), c, np.nan)
            cov[:, i, j] = c
            cov[:, j, i] = c

    d = np.sqrt(np.clip(np.einsum("tii->ti", cov), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / (d[:, :, None] * d[:, None, :])
    corr = np.clip(corr, -1.0, 1.0)
    idx = np.arange(K)
    corr[:, idx, idx] = np.where(np.isfinite(corr[:, idx, idx]), 1.0, np.nan)
    return corr


def pooled_fdm(mean_signals: np.ndarray) -> np.ndarray:
    """Equation 3.23 -- FDM_t = min(2, max(1, 1/sqrt(w' R_t w))).

    POOLED OVER THE UNIVERSE, NOT PER INSTRUMENT, and the paper says why: a
    4x4 correlation estimated on one market is a far noisier object than one
    estimated on 62, and fitting a multiplier per instrument is where
    overfitting would enter.  So R_t is built from the CROSS-SECTIONAL MEAN of
    each signal -- one series per alpha, averaged across instruments each
    session -- and every instrument shares the resulting FDM on that date.

    WHAT IT CORRECTS.  Four signals averaged at 1/4 each produce an aggregate
    weaker than any of them whenever they disagree: measured on this book,
    mean |Sign_raw| is 5.24 against ~9.2 for its inputs.  That is not a flaw in
    the signals, it is arithmetic -- and it would leave the strategy
    systematically under-engaged.  1/sqrt(w'Rw) is exactly the factor that
    undoes it.

    THE 2.0 CAP IS ANALYTIC, NOT CHOSEN.  Four equally weighted uncorrelated
    signals give 1/sqrt(4 x 0.25^2) = 2 exactly, so the cap is the theoretical
    maximum rather than a tuned bound: it can only bind on a matrix that is
    NEGATIVELY correlated overall, which is not diversification but a
    cancelling pair.

    IDENTITY 1.0 WHERE IT CANNOT BE COMPUTED -- before min_periods, on any bar
    whose matrix is not fully populated, or where w'Rw falls below a small
    floor (there 1/sqrt would explode).  Returning 1.0 means "no correction",
    which is the safe direction: it under-engages rather than levering up on a
    number it does not have.
    """
    T = mean_signals.shape[0]
    fdm = np.ones(T, dtype=np.float64)
    corr = ewm_corr_4(mean_signals, FDM_CORR_SPAN, FDM_MIN_PERIODS)
    w = np.full(4, 0.25)
    ok = ~np.isnan(corr).any(axis=(1, 2))
    var_blend = np.einsum("i,tij,j->t", w, np.clip(corr, -1.0, 1.0), w)
    good = ok & (var_blend > FDM_VAR_FLOOR)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.clip(1.0 / np.sqrt(var_blend), FDM_FLOOR, FDM_CAP)
    fdm[good] = vals[good]
    return fdm


_POINTSIZE: dict[str, float] | None = None


def pointsize_of(inst: str) -> float:
    """The contract multiplier, from instrument_mapping.csv.

    Cached per process: 63 workers each read a 4 KB file once rather than once
    per instrument.

    ABORTS ON A MISSING INSTRUMENT rather than defaulting to 1.0.  A pointsize
    of 1.0 is a plausible-looking number that would silently misstate contract
    volatility by whatever the true multiplier is -- 100,000 on 6A, 1,000 on
    CL -- and nothing downstream could detect it.
    """
    global _POINTSIZE
    if _POINTSIZE is None:
        t = pl.read_csv(MAPPING, infer_schema_length=0)
        _POINTSIZE = {r["norgate_code"]: float(r["pointsize"])
                      for r in t.iter_rows(named=True)
                      if (r.get("pointsize") or "").strip()}
    if inst not in _POINTSIZE:
        raise SystemExit(f"[ABORT] {inst}: no pointsize in {MAPPING.name}")
    return _POINTSIZE[inst]


_COST_RT = None


def cost_rt_of(inst: str) -> float:
    """Average ROUND-TRIP cost of one contract, in the instrument's own currency.

    `total_avg_cost_rt_LocalCurrency` from instrument_mapping.csv: commission
    plus the spread crossed, for a full in-and-out.  A ONE-WAY trade is half of
    it, which is the form stage 3 actually needs -- `|dN|` counts one-way
    contracts changed, not round trips.

    IN LOCAL CURRENCY, so it must be multiplied by FX_rate to reach USD, exactly
    like the price is.  Forgetting that understates SJB's cost by 159x and
    overstates nothing, which is the quiet direction.

    IT IS A FIXED COST PER CONTRACT AND THEREFORE MODELS NO MARKET IMPACT.  That
    is the whole limitation of the number: commission and half-spread scale with
    contracts, impact scales with size against available liquidity, and only the
    first is here.  At the position sizes this book reaches -- multiples of open
    interest, see Portfolio_Journal -- impact would dominate and is absent.  Any
    cost figure derived from this is a FLOOR, valid at small size.

    ABORTS ON A MISSING INSTRUMENT rather than defaulting to zero, for the usual
    reason: a zero cost is a plausible-looking number that silently flatters
    every net figure downstream.
    """
    global _COST_RT
    if _COST_RT is None:
        t = pl.read_csv(MAPPING, infer_schema_length=0)
        _COST_RT = {r["norgate_code"]: float(r["total_avg_cost_rt_LocalCurrency"])
                    for r in t.iter_rows(named=True)
                    if (r.get("total_avg_cost_rt_LocalCurrency") or "").strip()}
    if inst not in _COST_RT:
        raise SystemExit(f"[ABORT] {inst}: no total_avg_cost_rt_LocalCurrency "
                         f"in {MAPPING.name}")
    return _COST_RT[inst]


_CURRENCY = None


def currency_of(inst: str) -> str:
    """The instrument's LOCAL currency, from instrument_mapping.csv.

    This is the currency the contract's price and pointsize are quoted in -- what
    a position actually makes and loses before any conversion.  On this panel:
    45 USD, 7 EUR, 3 GBP, 3 CAD, 2 JPY, 2 AUD, 1 HKD.

    ABORTS ON A MISSING INSTRUMENT, and on one whose currency has no rate file,
    rather than falling back to USD.  Defaulting to the base currency is the
    worst possible failure here: it is silent, it looks right, and it means an
    instrument's P&L is simply never converted -- a CGB would be counted at 1.00
    instead of 0.72, understating nothing and overstating everything by ~39%,
    with no null anywhere to notice.  Better to stop and be told which
    instrument.
    """
    global _CURRENCY
    if _CURRENCY is None:
        t = pl.read_csv(MAPPING, infer_schema_length=0)
        _CURRENCY = {r["norgate_code"]: (r.get("currency") or "").strip().upper()
                     for r in t.iter_rows(named=True)
                     if (r.get("currency") or "").strip()}
    if inst not in _CURRENCY:
        raise SystemExit(f"[ABORT] {inst}: no currency in {MAPPING.name}")
    c = _CURRENCY[inst]
    if c not in FX_CCY:
        raise SystemExit(f"[ABORT] {inst}: currency {c} has no rate -- add it to "
                         f"FX_CCY in {Path(__file__).name}")
    return c


def price_vol_curr_ann(vol_abs: pl.Series, pointsize: float) -> pl.Series:
    """Equation 3.35 WITHOUT the FX leg -- annualised contract vol, in the
    contract's own quotation currency.

        sigma_hat x pointsize x sqrt(256)

    THREE UNIT CHANGES, ONE AT A TIME.  3.35 does the whole conversion in a
    single line -- price units to dollars, per-point to per-contract, daily to
    annual.  Splitting the FX leg off makes the intermediate meaningful on its
    own: this is what ONE CONTRACT of this instrument is worth in annual risk,
    expressed in the currency it trades in.  For a USD-quoted market that IS
    the dollar figure; for FDAX9 it is euros, for SJB yen.

    NOT COMPARABLE ACROSS CURRENCIES until the FX leg is applied.  The
    `currency` column of instrument_mapping.csv says which unit each number is
    in, and mixing them would be an error the values themselves cannot reveal.

    sigma_hat is Eq 3.34, which is `daily_vol_abs` -- the blended 70/30
    estimator in price units.  That is the one the sizer needs; the return-space
    `daily_vol` would be dimensionless and cancel the pointsize entirely.
    """
    return vol_abs * pointsize * np.sqrt(TRADING_DAYS_YEAR)


def xs_return(cont_close: pl.Series, raw_close: pl.Series,
              lookback: int = XS_LOOKBACK) -> pl.Series:
    """r^256 -- the 256-session return, in the same convention as `daily_ret`.

        r_i,t^256 = (PanamaP_t - PanamaP_t-256) / P_raw_t-256

    THE SAME MIXED PAIR as daily_ret, for the same reason and at 256 sessions
    instead of one: the numerator from the adjusted series, which is the only
    clean measure of P&L across the rolls in between, and the denominator from
    the raw close, which is the capital that was actually at risk.

    THE DENOMINATOR MUST NOT BE THE ADJUSTED PRICE.  Over 256 sessions the
    accumulated offset is large, so dividing by it deflates the return by
    however much history has stacked up -- and worse, the adjusted close can be
    NEGATIVE on a market whose cumulative adjustment exceeds its price, which
    silently INVERTS the sign.  The raw close is a traded contract price and is
    strictly positive, so the sign is always the true direction.

    Null for the first 256 sessions: there is no price 256 back to measure from.
    """
    return (cont_close - cont_close.shift(lookback)) / raw_close.shift(lookback)


def cross_sectional_z(panel: dict[str, pl.Series], dates: pl.Series,
                      min_insts: int = XS_MIN_INSTS) -> dict:
    """Equation 3.13, one z-score per instrument per session.

        f_trend^XS(i,t) = (r_i,t^256 - rbar_t^256) / sigma_t^256

    NEITHER rbar NOR sigma CARRIES AN i.  Both are statistics ACROSS
    INSTRUMENTS on session t -- the mean and the standard deviation of that
    day's cross-section.  This is the one column in the book that cannot be
    computed from a single instrument, and the reason main() has to gather every
    instrument before any file can be finished.

    ON THE UNION OF ALL SESSION DATES, not on any one instrument's calendar.
    Markets keep different holidays; scoring an instrument against a different
    day's cross-section would be a look-ahead on some dates and a look-behind on
    others.  A date an instrument does not trade simply leaves it out of that
    day's statistics.

    A ROW IS BLANKED ENTIRELY when fewer than `min_insts` instruments have a
    value, or when the spread is exactly zero.  Three is the floor because a
    z-score over two points is always +/-0.7071 whatever the prices did -- it
    carries the sign and no information, and would look like a real forecast.
    Zero spread means every instrument moved identically, so the z-score is 0/0;
    a null says "no cross-section today", which is the truth.

    ddof=1, matching the thesis (`assemble._cross_sectional_z`).
    """
    insts = sorted(panel)
    M = np.column_stack([panel[i].to_numpy().astype(float) for i in insts])
    finite = np.isfinite(M)
    n = finite.sum(axis=1)

    mean = np.full(M.shape[0], np.nan)
    std = np.full(M.shape[0], np.nan)
    usable = n >= max(min_insts, 2)
    if usable.any():
        Mm = np.where(finite, M, np.nan)
        with np.errstate(invalid="ignore"):
            mean[usable] = np.nanmean(Mm[usable], axis=1)
            std[usable] = np.nanstd(Mm[usable], axis=1, ddof=1)
    good = usable & np.isfinite(std) & (std != 0.0)
    z = np.full(M.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        z[good] = (M[good] - mean[good, None]) / std[good, None]

    # NULLS, NOT NaN.  polars treats them differently -- is_not_null() is True
    # for a NaN -- and mixing the two has already caused one silent failure in
    # normalise_signal.  Emit the same "missing" every other column uses.
    return ({i: pl.Series(z[:, k]).fill_nan(None) for k, i in enumerate(insts)},
            dates, int(good.sum()), int(M.shape[0]))


def panama(w: pl.DataFrame, col: str) -> tuple[dict, list]:
    """Panama back-adjustment offsets, one per session.  {date: offset}.

    ANCHORED AT THE PRESENT and walked backwards.  The most recently held
    contract is the truth -- it is what a position is worth today -- so it takes
    a zero offset, and every earlier session is shifted by the accumulated gaps
    of the rolls that came after it.  The alternative, anchoring at the start,
    leaves today's series at some historical level and makes the newest number
    in the file the one you can trust least.

    THE GAP IS MEASURED BETWEEN THE TWO CONTRACTS ON THE SAME DAY:

        gap = close_new(T) - close_old(T)

    both quoted on the roll date T.  NOT close_new(T) - close_old(T-1), which
    would absorb the genuine overnight move into the adjustment and delete a
    real day of P&L.  With the same-day gap the roll-day return comes out as

        Continuous_C(T) - Continuous_C(T-1) = close_old(T) - close_old(T-1)

    -- the return of the contract actually held into the roll, which is the
    convention a backtest assumes.  Verified: the outgoing contract is still
    quoted on the roll date in 609 of 609 rolls across all four rule types, so
    this needs no fallback; a roll where it is absent is reported, not guessed.

    DIFFERENCES, NOT RATIOS.  An interest-rate future is 100 minus a rate, so a
    ratio return on SR3 or LEU9 is meaningless, and a ratio-adjusted series goes
    negative on any market whose cumulative adjustment exceeds its price.

    ONE OFFSET PER DATE, applied to open and close alike.  Deriving a separate
    open-gap would break `Continuous_C - Continuous_O == close - open`, quietly
    corrupting every intraday move in the file.  The close gap is the standard
    and the one that keeps the identity.

    Returns the offsets and any roll it could not measure.
    """
    # ROLL-DRIVEN, not session-driven.  The offsets only change at a roll, and
    # ZC has 197 of those against 12,206 sessions -- so walking every session and
    # materialising its rows as dicts did ~60x more work than the answer needs.
    # Two columnar extractions and a dict of closes replace that; the loop below
    # touches only the sessions where the hold actually changes.
    sess = (w.select(["date", col])
             .unique(subset=["date"], keep="first")
             .sort("date"))
    order = [str(x) for x in sess.get_column("date").to_list()]
    holds = [h or None for h in sess.get_column(col).to_list()]

    # (date, symbol) -> close, built once.  This is what the per-session dict
    # rebuild was really providing, and it is needed only at roll dates.
    close_of = {}
    for d, sym, c in zip(w.get_column("date").cast(pl.Utf8).to_list(),
                         w.get_column("symbol").to_list(),
                         w.get_column("close").to_list()):
        close_of[(d, sym)] = c

    adj = {d: 0.0 for d in order}
    unmeasured = []
    run = 0.0
    for i in range(len(order) - 1, 0, -1):
        cur, prev = order[i], order[i - 1]
        h_cur, h_prev = holds[i], holds[i - 1]
        if h_cur and h_prev and h_cur != h_prev:
            c_new = close_of.get((cur, h_cur))
            c_old = close_of.get((cur, h_prev))
            if c_new is not None and c_old is not None:
                run += float(c_new) - float(c_old)
            else:
                # The outgoing contract has no bar on the roll date, so the gap
                # is unmeasurable.  Carrying the offset unchanged leaves the
                # jump in the series rather than inventing a number for it --
                # and the caller is told, because a silent one is a fake return.
                unmeasured.append((cur, h_prev, h_cur))
        adj[prev] = run
    return adj, unmeasured


def carry_contract(w: pl.DataFrame, col: str) -> tuple[pl.DataFrame, int]:
    """The contract one step further out than the one held, per session.

    The held contract is the near leg of a carry pair; this is the far leg --
    the next delivery the market lists after it.  Together they give the shape
    of the curve at the point a position actually sits on it.

    ORDERED ON THE EXPIRY DATE, NOT ON THE SYMBOL.  The symbol encodes a
    delivery month (CL-2026V is October 2026), so sorting it alphabetically
    would order 2026V after 2026X and put the wrong contract next.  The expiry
    is an ISO date, so ordering it is ordering time.

    WHICH EXPIRY COLUMN DEPENDS ON THE MARKET, and the worksheet carries only
    the one that market is gated on: `last_trade` for the 41 cash-settled and
    financial markets, `first_notice` for the 22 deliverable ones (grains,
    metals, livestock, bonds).  Never both, never neither.  Either serves here,
    because the two run in the same order across delivery months -- a contract
    that gives notice earlier also expires earlier -- and only the ORDER is
    used.  Reading whichever is present avoids duplicating that gate decision,
    which belongs to front_contract.py.

    "NEXT" MEANS NEXT AMONG THE MONTHS LISTED THAT SESSION, which is not the
    same as the next calendar month.  Most markets skip months -- and which
    ones are listed changes over the life of the panel -- so stepping a fixed
    number of months would fabricate contracts that were never quoted.  Taking
    the smallest expiry strictly greater than the hold's asks the sheet what
    actually existed on the day.

    A contract already past its last trade sorts BEFORE the hold and so cannot
    be chosen: the strict inequality does that work, with no need to read
    `is_passed`.

    NULL WHEN THE HOLD IS THE FURTHEST-DATED CONTRACT LISTED.  There is no next
    leg, and the count is returned so the caller can report it rather than let
    an empty carry column pass unnoticed.

    `Carry_hold_O` / `Carry_hold_C` ARE THAT CONTRACT'S OWN RAW QUOTES, and
    deliberately NOT back-adjusted.  Carry compares two contracts priced on the
    SAME session, so there is nothing to adjust for: the Panama offset exists to
    join different contracts across TIME, and this comparison never crosses a
    roll.  Worse, the offset on this frame belongs to the near leg's chain --
    applying it to the far leg would shift a price by an amount computed from a
    different contract's roll history, and (F1 - F2) / F1 would silently stop
    being the curve's slope.  Raw against raw is the only pair that is a spread.
    """
    exp = next((c for c in ("last_trade", "first_notice") if c in w.columns), None)
    if exp is None:
        raise SystemExit("[ABORT] worksheet has neither 'last_trade' nor "
                         "'first_notice'; cannot order contracts by expiry")
    base = (w.select(["date", "symbol", exp, col])
             .drop_nulls(exp)
             .filter(pl.col(exp) != ""))
    held = (base.filter(pl.col("symbol") == pl.col(col))
                .select(["date", pl.col(exp).alias("_hold_exp")])
                .unique(subset=["date"], keep="first"))
    nxt = (base.join(held, on="date", how="inner")
               .filter(pl.col(exp) > pl.col("_hold_exp"))
               .group_by("date")
               .agg(pl.col("symbol").sort_by([exp, "symbol"]).first()
                      .alias("carry_hold"),
                    pl.col(exp).sort_by([exp, "symbol"]).first().alias("_far_exp"),
                    pl.col("_hold_exp").first()))

    # dT -- Eq 3.15's denominator, in CALENDAR days, as the paper states.  It is
    # the gap between the two contracts' own expiry dates, not between quote
    # dates: it is what turns a spread into an annualised rate, by saying how
    # much of a year the curve is being traversed over.  Measured on whichever
    # gate column this market carries; consecutive deliveries sit the same
    # distance apart under either, and only the DIFFERENCE is used.
    nxt = nxt.with_columns(
        (pl.col("_far_exp").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
         - pl.col("_hold_exp").str.strptime(pl.Date, "%Y-%m-%d", strict=False))
        .dt.total_days().cast(pl.Float64).alias("_dT")).drop("_far_exp", "_hold_exp")

    # The far leg's own quotes, looked up on the same session.
    prices = (w.select(["date",
                        pl.col("symbol").alias("carry_hold"),
                        pl.col("open").cast(pl.Float64).alias("Carry_hold_O"),
                        pl.col("close").cast(pl.Float64).alias("Carry_hold_C")])
               .unique(subset=["date", "carry_hold"], keep="first"))
    nxt = nxt.join(prices, on=["date", "carry_hold"], how="left")

    n_sessions = held.height
    return nxt, n_sessions - nxt.height


def book_one(fc, inst: str, rule: str, start: str, end: str, *,
             as_of: str | None = "auto", held_only: bool = True,
             uniform: bool = True, use_cache: bool = True):
    col = HOLD_FOR.get(rule)
    if col is None:
        raise SystemExit(f"[ABORT] {inst}: Roll_Rule {rule!r} has no hold "
                         f"column in HOLD_FOR")
    w, hit = cached_worksheet(fc, inst, start, end, as_of, use_cache=use_cache)
    if col not in w.columns:
        raise SystemExit(f"[ABORT] {inst}: worksheet has no column {col!r} "
                         f"-- has front_contract.py been renamed under it?")
    n_sessions = w.get_column("date").n_unique()

    # Panama offsets come from the FULL worksheet, before the held-only filter:
    # the gap at a roll needs the OUTGOING contract's close on the roll date,
    # and that row is exactly what the filter is about to drop.
    adj, unmeasured = panama(w, col)
    # Same reason as panama above: the far leg is one of the rows the held-only
    # filter is about to drop, so it has to be identified from the full sheet.
    carry, n_no_next = carry_contract(w, col)
    out = w.select(KEEP + [col]).with_columns(
        pl.col("date").cast(pl.Utf8).replace_strict(adj, default=0.0)
          .alias("_adj"))
    out = out.with_columns(
        (pl.col("open").cast(pl.Float64) + pl.col("_adj")).alias("Continuous_O"),
        (pl.col("close").cast(pl.Float64) + pl.col("_adj")).alias("Continuous_C"),
    ).drop("_adj")
    # The held-only series is built either way: EWMAC needs one row per session,
    # and running it over the repeated worksheet rows would feed the same price
    # in N times a day and compress every span by a factor of N.
    held = out.filter(pl.col("symbol") == pl.col(col)).sort("date")
    sig = pl.DataFrame({"date": held.get_column("date")})
    for name, series in ewmac(held.get_column("Continuous_C")).items():
        sig = sig.with_columns(series.alias(name))

    # daily_ret -- the adjusted (Panama) return.  A MIXED PAIR, deliberately:
    #
    #     (PanamaP_n - PanamaP_n-1) / P_n-1
    #
    # numerator from the ADJUSTED series, denominator from the RAW close.
    #
    # The adjusted LEVEL is arbitrary: it is the raw price plus whatever offset
    # history has accumulated, and it moves again on every future roll.  Divide
    # by it and every return is deflated by however much has been stacked on
    # top -- on ES in October 1997 that turns a true -0.4092% into -0.2434%, a
    # 40% understatement, and it gets worse the further back you look.
    #
    # The adjusted DIFFERENCE, on the other hand, is the only clean measure of
    # P&L across a roll: on a roll session it is the held contract's own move,
    # with the contract change removed.  So take the difference from the
    # adjusted series and the base from the raw close, which is the capital
    # actually at risk.
    #
    # The result is invariant to the anchor.  Our Panama level for ES on
    # 1997-10-14 is 1908.65 where the source paper's is 1643.65 -- different
    # anchor dates -- and both give -0.4092%.  That is why daily_ret is stable
    # across runs while Continuous_C is not.
    #
    # First session is null: there is no previous close to divide by, and a
    # zero there would be a fabricated flat day.
    sig = sig.with_columns(
        ((held.get_column("Continuous_C")
          - held.get_column("Continuous_C").shift(1))
         / held.get_column("close").shift(1)).alias("daily_ret"))

    # Equation 3.16 -- on the return just computed, one row per session.
    # Named "-Skew", with the minus in the column name, because the inversion
    # IS the signal and a column called "Skew" would be read as the opposite of
    # what it holds.  A reader who never opens this file still cannot mistake
    # the sign.
    sig = sig.with_columns(skew(sig.get_column("daily_ret")).alias("-Skew"))

    # Equation 3.19 -- the sizer's volatility.  Built from the same held-only
    # series as the EWMACs and for the same reason: the worksheet repeats each
    # session across every listed month, and differencing that would produce a
    # column of zeros between months of the same day and one real move per day,
    # which is not the price change of anything.
    sig = sig.with_columns(
        daily_vol_abs(held.get_column("Continuous_C")).alias("daily_vol_abs"))
    # Equation 3.29 -- the volatility gate, computed here because the USD and
    # annualisation factors cancel out of its ratio and it therefore needs
    # nothing the panel has not already given this worker.  NOT YET APPLIED to
    # SIGNAL: 3.31 multiplies the smoothed forecast by BOTH gates, and 3.30 is
    # not built.  The column stands on its own until it is.
    sig = sig.with_columns(
        s_g_vol(held.get_column("Continuous_C"),
                sig.get_column("daily_vol_abs")).alias("s_g_vol"))
    sig = sig.with_columns(
        daily_vol(sig.get_column("daily_ret")).alias("daily_vol"))

    # Equation 3.35, first step: price units -> contract units, annualised.
    # The FX leg is deliberately not applied here; see price_vol_curr_ann.
    sig = sig.with_columns(
        price_vol_curr_ann(sig.get_column("daily_vol_abs"),
                           pointsize_of(inst)).alias("price_vol_curr_ann"))


    # Equation 3.17 literally -- the inner leg 3.18 will be built on.
    sig = sig.with_columns(
        vov_inner(sig.get_column("daily_ret")).alias("VoV_inner"))
    # Equation 3.18 -- the outer std, taken of the inner leg just built.
    sig = sig.with_columns(
        vov_smooth(sig.get_column("VoV_inner")).alias("VoV_smooth"))
    # Equation 3.19 -- the one-year mean of the outer leg.
    sig = sig.with_columns(
        vov_mean_ann(sig.get_column("VoV_smooth")).alias("VoV_mean_ann"))
    # Equation 3.20 -- size from the VoV ratio, direction from the 64-day move.
    sig = sig.with_columns(
        vov_signal(sig.get_column("VoV_smooth"),
                   sig.get_column("VoV_mean_ann"),
                   held.get_column("Continuous_C")).alias("VoV"))

    # Equation 3.21 -- every alpha onto the same +/-10 conviction scale, so the
    # 25/25/25/25 aggregation downstream weights signals rather than units.
    # Trend normalises its two halves BEFORE blending; see `trend_sign`.

    # Equation 3.13's NUMERATOR INPUT only.  The z-score itself needs every
    # other instrument, so it cannot be finished here; main() gathers this
    # column across the book and fills XS_trend in afterwards.  It is carried
    # on the frame rather than returned alongside it so that it survives the
    # --all-rows join unchanged, like every other per-session series.
    sig = sig.with_columns(
        xs_return(held.get_column("Continuous_C"),
                  held.get_column("close").cast(pl.Float64)).alias("r256"))

    sig = sig.join(carry, on="date", how="left")

    # Equation 3.15 -- carry as an annualised implied return.
    #
    #     f_carry(i,t) = (F1 - F2) / F1 x 256 / dT
    #
    # F1 is the held contract's own raw close and F2 the next contract's, both
    # quoted on session t.  RAW ON BOTH SIDES, as the paper requires: Panama
    # prices would "distort the structural spread of the curve", and the offset
    # on this frame belongs to the near leg's chain anyway.
    #
    # THE SIGN IS THE POSITION.  Backwardation (F1 > F2) gives a positive carry
    # and argues for being long; contango gives a negative one and argues for
    # being short.  It is read as the return earned by simply holding while the
    # curve rolls down to spot, IF the spot price does not move -- which is the
    # assumption the whole signal rests on, and the reason it is a forecast
    # rather than a measurement.
    #
    # 256 OVER dT ANNUALISES A SPREAD THAT SPANS dT CALENDAR DAYS.  Mixing 256
    # trading days over a calendar-day gap is the paper's own convention, not a
    # slip: 3.15 says calendar days explicitly.
    #
    # Null rather than a number when it cannot be computed: no far leg, no
    # quote on either side, a non-positive F1 (which would inflate or invert
    # the ratio), or a non-positive dT.  A fabricated carry is worse than a gap,
    # because nothing downstream could tell it was invented.
    f1 = held.get_column("close").cast(pl.Float64)
    sig = sig.with_columns(f1.alias("_F1"))
    sig = sig.with_columns(
        pl.when((pl.col("_F1") > 0) & pl.col("Carry_hold_C").is_not_null()
                & (pl.col("_dT") > 0))
          .then((pl.col("_F1") - pl.col("Carry_hold_C")) / pl.col("_F1")
                * (TRADING_DAYS_YEAR / pl.col("_dT")))
          .otherwise(None).alias("Carry")).drop("_F1", "_dT")

    # Carry_State -- the curve's shape as a letter, for reading by eye.
    #
    #     B  backwardation, F1 > F2, Carry > 0, argues long
    #     C  contango,      F1 < F2, Carry < 0, argues short
    #     F  flat,          F1 = F2, Carry = 0, argues nothing
    #
    # DERIVED FROM `Carry`'S OWN SIGN, not from a second comparison of the two
    # legs.  A debugging column that can disagree with the thing it is there to
    # debug is worse than no column: re-deriving it from F1 and F2 would let a
    # future change to Eq 3.15 leave the two silently inconsistent, and the
    # label would be believed over the number.
    #
    # F IS NOT A ROUNDING CASE.  The two contracts settle at exactly the same
    # price on 2.7% of sessions -- 14,666 of them, across 61 instruments, and
    # 42% of YXT4's own -- because the tick is coarse relative to the spread on
    # short-dated rates.  Folding those into B or C would depend entirely on
    # whether the test was written > or >=, and would misreport a genuinely
    # flat curve as a directional signal on one instrument in three.
    sig = sig.with_columns(
        pl.when(pl.col("Carry").is_null()).then(None)
          .when(pl.col("Carry") > 0).then(pl.lit("B"))
          .when(pl.col("Carry") < 0).then(pl.lit("C"))
          .otherwise(pl.lit("F")).alias("Carry_State"))

    # Equation 3.21 -- each alpha onto the same +/-10 conviction scale, so the
    # 25/25/25/25 aggregation downstream weights signals rather than units.
    # Placed here because Carry is the last of the three to exist; Trend_sign
    # is NOT built here at all -- it needs XS_trend, which is cross-sectional
    # and cannot exist until every instrument has been built, so the parent
    # adds it in the same pass that fills XS_trend.
    sig = sig.with_columns(
        normalise_signal(sig.get_column("Carry")).alias("Carry_sign"),
        normalise_signal(sig.get_column("-Skew")).alias("Skew_sign"),
        normalise_signal(sig.get_column("VoV")).alias("VoV_sign"),
    )

    if held_only:
        # The hold is session-level, so it repeats down every contract row of a
        # session.  Keeping the row whose own symbol IS the hold collapses that
        # to one row per session, carrying that contract's own open and close.
        # A session whose hold is blank matches nothing and DISAPPEARS -- the
        # caller compares against n_sessions to catch exactly that.
        out = held
    # --all-rows joins the same per-session signal back onto every listed month,
    # so the column means the same thing in both shapes.
    out = out.join(sig, on="date", how="left")
    if uniform:
        out = out.rename({col: "hold"})
    return out, n_sessions, unmeasured, hit, n_no_next


def _one(args_tuple):
    """Build one instrument in a worker process.  Module-level so it pickles.

    as_of IS PASSED IN, NOT RESOLVED HERE.  Resolving it walks all 15,231
    contract files and costs ~4.6s; letting each of 63 workers repeat that would
    add more than the parallelism saves.  The parent resolves it once and hands
    down the answer -- which also guarantees every instrument is squared off at
    the SAME date, where a worker computing its own could disagree if the panel
    changed under a long run.
    """
    (inst, rule, start, end, as_of, held_only, uniform, out_dir,
     use_cache) = args_tuple
    fc = _load(FC, "fc")
    try:
        d, n_avail, unmeasured, hit, n_no_next = book_one(
            fc, inst, rule, start, end, as_of=as_of, held_only=held_only,
            uniform=uniform, use_cache=use_cache)
    except SystemExit as exc:
        return inst, None, 0, 0, [], str(exc), False, None, 0
    # NOT WRITTEN HERE ANY MORE.  XS_trend (3.13) is a cross-sectional z-score,
    # so no instrument's file is complete until every instrument has been built.
    # The frame goes back to the parent, which fills that column and writes.
    return (inst, d.height, d.get_column("date").n_unique(), n_avail,
            unmeasured, None, hit, d, n_no_next)


# ===========================================================================
#  FX -- one USD conversion rate per currency, written one file per rate.
# ===========================================================================
#
# The book is priced in local currency.  A CGB moves in CAD, a NIY in JPY, and a
# position's risk only means something once it is in the account's currency -- so
# every non-USD instrument needs a rate, on every session it trades, for the
# whole history it is backtested over.
#
# ONE FILE PER RATE, `FX/<CCY>.csv`, mirroring `Trading_book/<INST>.csv`:
#
#     date, Derived_Rate, NDU_Rate, YF_Rate, NDU_diff_bp, YF_diff_bp, Status
#
# THE BOOK USES `Derived_Rate` AND NOTHING ELSE.  The other two columns are not
# alternatives a caller picks between at run time -- they are there to be
# disagreed with.  A conversion rate is the one input in this pipeline that is
# both silently wrong-able and catastrophic when wrong: a rate off by a factor
# of 100 does not raise, it sizes a position 100x wrong, and every downstream
# number stays plausible.  Two independent sources beside the one in use turn
# that from an invisible failure into a printed one.
#
# EVERY RATE IS CCY -> USD.  Multiply a local-currency amount by it to get USD.
# That direction is fixed everywhere here, checks included, so an inversion error
# surfaces as a ~100% disagreement rather than as a plausible number.
#
# --- where Derived_Rate comes from -----------------------------------------
#
# NDU HAS NO SPOT FX.  The Forex Spot database holds exactly one symbol, $USDX,
# which is the dollar index and not a rate.  So spot has to be derived.  Covered
# interest parity prices a currency future off spot and the two countries' rates:
#
#     F(T) = S * exp((r_USD - r_foreign) * T)
#
# so ln F is linear in T and the intercept at T = 0 is ln S.  Two quoted
# contracts give the line:
#
#     slope = (ln f2 - ln f1) / (T2 - T1)
#     S     = exp(ln f1 - slope * T1)
#
# THE TWO NEAREST CONTRACTS, NOT A FIT THROUGH ALL OF THEM.  FX futures are
# quarterly and the front two carry nearly all the volume; deferred months are
# frequently settlement-only, with zero volume and an unchanged price for weeks
# (6E-2026U opens in 2021 at 1.2587 flat).  Fitting through those would let a
# stale quote set the slope.
#
# T1 = 0 IS ALLOWED AND IS NOT A DEGENERACY: on the front's last trade date the
# formula collapses to S = f1, which is right -- at expiry the future is spot.
# T2 > T1 is required, since the slope divides by the difference.
#
# FILTERING FOR LIVE QUOTES CHANGED NOTHING MEASURABLE.  Requiring volume > 0 or
# open interest > 0 on the two nearest contracts left the last session identical
# to six decimal places in all five currencies, because the nearest two ARE the
# liquid two.  Not applied: a branch and a fallback path for no gain.
#
# --- why not the vendor's continuous ---------------------------------------
#
# `&6E` is the UNADJUSTED continuous and is a real candidate -- it carries true
# price levels and the vendor owns the roll.  It was measured against yfinance on
# the same ~29,000-observation inner join and lost:
#
#     source            signed mean        mean abs      median      p95
#     Derived         -0.009% .. -0.029%     0.316%      0.202%    0.977%
#     &6E (NDU)       -0.168% .. +0.315%     0.397%      0.284%    1.132%
#
# THE SIGNED MEANS ARE THE REASON, NOT THE AVERAGES -- they differ by only 0.08pp
# on absolute error.  The derivation's bias is within +/-0.03% of zero in every
# currency; the continuous's is not, because a front-contract price carries the
# basis and the basis has a sign.  A persistent offset does not average out over
# a backtest; scatter of the same size does.  It is kept as NDU_Rate, where being
# a near-copy of a different construction is exactly what makes it a useful check.
#
# `&6E_CCB`, THE BACK-ADJUSTED CONTINUOUS, IS DISQUALIFIED OUTRIGHT: its 1999
# EURUSD value is 1.4543, a rate that never existed.  Panama preserves
# differences, not levels, and a conversion rate is purely a level.  Named here
# because _CCB is the symbol used elsewhere in this repo, so reaching for it is
# the natural mistake.

FX_DIR = HERE / "FX"
FX_CACHE = CACHE.parent / "fx"
HKD_PEG = 7.80


def _src_hash() -> str:
    """This module's source, hashed.  Part of every vendor-cache key.

    The FX and IRX caches are keyed on the PANEL EDGE, which answers "has new
    data arrived" but not "has the code that shapes it changed".  Edit a scale
    factor, a symbol or a conversion and the old result would keep being served
    until the next session happened to appear.  One hash of one file per run is
    cheap insurance against that.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

# `inst` drives the derivation, `ndu`/`yf` the two checks, `scale` divides the
# quoted price, `fixed` replaces the derivation for currencies with no future.
FX_CCY: dict[str, dict] = {
    # USD.csv IS SUPPOSED TO LOOK EMPTY.  Derived_Rate is 1.0 on all 12,552 rows
    # and every other column is null, because THE PORTFOLIO IS DENOMINATED IN
    # USD: the conversion is the identity, so there is nothing to derive and no
    # source that could disagree with it.  `fixed=1.0` is what marks the base
    # currency.  It is written out rather than left absent because 45 of the 63
    # instruments are USD and generic caller code would otherwise special-case
    # the commonest currency in the book -- a missing file and a rate of 1.0 are
    # different things.  See FX/FX_Journal.md.
    "USD": dict(inst=None, scale=1.0,   ndu=None,  yf=None,       inv=False,
                fixed=1.0),
    "EUR": dict(inst="6E", scale=1.0,   ndu="&6E", yf="EURUSD=X", inv=False,
                fixed=None),
    "GBP": dict(inst="6B", scale=1.0,   ndu="&6B", yf="GBPUSD=X", inv=False,
                fixed=None),
    "CAD": dict(inst="6C", scale=1.0,   ndu="&6C", yf="CADUSD=X", inv=False,
                fixed=None),
    "AUD": dict(inst="6A", scale=1.0,   ndu="&6A", yf="AUDUSD=X", inv=False,
                fixed=None),
    # 6J QUOTES USD PER 100 JPY AND NOTHING IN THE METADATA SAYS SO.  Last close
    # 0.6287 -> rate 0.006287 -> 159.1 JPY/USD.  instrument_mapping.csv cannot
    # help: 6J's pointsize is identical to EUR's and CHF's.  This is exactly the
    # failure the check columns exist for, and they catch it -- an unscaled JPY
    # shows a 9,900% disagreement against BOTH of them.
    "JPY": dict(inst="6J", scale=100.0, ndu="&6J", yf="JPYUSD=X", inv=False,
                fixed=None),
    # HKD HAS NO INSTRUMENT ANYWHERE -- not in the panel, not in NDU -- and is
    # held at the peg the HKMA has defended in a 7.75-7.85 band since 2005.
    # Against USDHKD=X the peg is worth +0.178% signed, 0.378% mean absolute,
    # 2.053% worst: the same order as the derivation error on currencies that do
    # have a future, which is what makes it good enough.  HSI is the only HKD
    # instrument in the book.  yfinance carries only the USD-base pair here, so
    # `inv` reciprocates it; the majors are read direct (CADUSD=X, JPYUSD=X)
    # because one less arithmetic step is one less thing to get backwards.
    "HKD": dict(inst=None, scale=1.0,   ndu=None,  yf="USDHKD=X", inv=True,
                fixed=1.0 / HKD_PEG),
}

# Disagreement thresholds, in basis points of relative difference.  SET FROM THE
# MEASURED DISTRIBUTION, not picked for roundness: derived vs yfinance runs
# median 20bp, p95 98bp, max 1,746bp over ~29,000 observations.  So WATCH sits at
# about the 95th percentile -- it fires on the tail the derivation genuinely has
# -- and ALERT is well outside anything the method produces in normal operation.
# An ALERT means look at it, not that the rate is unusable.
FX_WATCH_BP = 100.0
FX_ALERT_BP = 300.0


def _prefer_parquet(base: Path) -> Path:
    """`base.parquet` if it is present AND NOT OLDER than `base.csv`, else the csv.

    EVERY OUTPUT HERE IS WRITTEN TWICE -- csv to be read by a person, parquet to
    be read by a program.  The parquet is the machine artifact because dtypes
    travel inside the file: there is no inference to get wrong, so the whole
    class of bug where a long run of leading nulls turns a Float64 column into a
    String simply cannot occur.  It is also 2.9x smaller and 2.4x faster to read.

    THE MTIME GUARD IS THE POINT OF THIS FUNCTION.  Both files are written in the
    same loop, so they normally agree -- but a run that dies between the two
    writes, or an older build that wrote only csv, would leave a parquet that is
    silently out of date.  Reading that as truth is exactly the failure this
    codebase spends its cache fingerprints avoiding.  Older parquet loses to the
    csv, and the csv path still knows its own schema, so the fallback is correct
    rather than merely available.
    """
    pq, csv = base.with_suffix(".parquet"), base.with_suffix(".csv")
    if pq.is_file() and (not csv.is_file()
                         or pq.stat().st_mtime_ns >= csv.stat().st_mtime_ns):
        return pq
    return csv


FX_SCHEMA = {"date": pl.Utf8, "Derived_Rate": pl.Float64, "NDU_Rate": pl.Float64,
             "YF_Rate": pl.Float64, "NDU_diff_bp": pl.Float64,
             "YF_diff_bp": pl.Float64, "Status": pl.Utf8}


def load_fx(ccy: str, d: Path | None = None) -> pl.DataFrame:
    """Read FX/<CCY>.csv with the dtypes named rather than guessed.

    `Derived_Rate` survives a bare `pl.read_csv` -- the files are trimmed so it
    is populated from the first row -- so a caller who only wants the rate does
    not need this.  THE CHECK COLUMNS DO NOT SURVIVE ONE: yfinance history starts
    in 2003 while these files start in 1979, so `YF_Rate` opens with thousands of
    blanks, polars infers dtype from a bounded prefix and settles on String.

    THE TRAP IS DATA-DEPENDENT, WHICH IS WHY IT IS A FUNCTION RATHER THAN A NOTE.
    It would quietly fix itself if the panel started later, and return the first
    time an earlier contract is added, so a caller who got away with read_csv
    once cannot conclude they will again.

    Prefers the parquet, where dtypes travel in the file and the question does
    not arise; the schema below is what makes the csv fallback equally correct.
    """
    p = _prefer_parquet((d or FX_DIR) / ccy)
    return (pl.read_parquet(p) if p.suffix == ".parquet"
            else pl.read_csv(p, schema=FX_SCHEMA))


# The only non-numeric columns a book carries.  `carry_hold` and any *_hold are
# contract symbols; `Carry_State` is a label.  Everything else is a Float64.
BOOK_TEXT_COLS = {"date", "symbol", "hold", "carry_hold", "Carry_State"}


def load_book(inst: str, d: Path | None = None) -> pl.DataFrame:
    """Read Trading_book/<INST>.csv with dtypes named rather than guessed.

    A BARE pl.read_csv ON A BOOK HANDS BACK NUMERIC COLUMNS AS STRINGS, and this
    is not an edge case -- it affects 17 columns in all 63 books today, `SIGNAL`
    among them.  Polars infers csv dtypes from a bounded prefix (100 rows by
    default), and every column with a warm-up is blank for longer than that:
    daily_vol_abs and XS_trend open with 256 nulls, Skew_sign with 511, VoV_sign
    with 594, FDM_MASTER and SIGNAL with 849.  The reader sees only blanks,
    settles on String, and arithmetic downstream either raises or -- far worse --
    a comparison silently succeeds and compares text.

    FX_rate joins that list on exactly two books, YAP4 and YXT4, whose histories
    begin before the AUD future did.

    Prefers the parquet, where dtypes travel in the file and none of the above
    can happen.  The csv fallback is what the rest of this docstring is about.

    THE SCHEMA IS BUILT FROM THE HEADER, NOT SNIFFED, which is the whole point:
    the failure is content-dependent, so any fix that reads content to decide
    inherits the bug.  A column is text if it is named in BOOK_TEXT_COLS or ends
    in `_hold` (which --keep-rule-name produces); everything else is Float64.
    """
    p = _prefer_parquet((d or BOOK) / inst)
    if p.suffix == ".parquet":
        return pl.read_parquet(p)
    with open(p, "r", encoding="utf-8") as fh:
        cols = fh.readline().rstrip("\r\n").split(",")
    return pl.read_csv(p, schema={
        c: (pl.Utf8 if c in BOOK_TEXT_COLS or c.endswith("_hold") else pl.Float64)
        for c in cols})


def fx_derived(w: pl.DataFrame, scale: float) -> pl.DataFrame:
    """(date, Derived_Rate) from one FX instrument's worksheet.

    THE WORKSHEET IS THE INPUT, NOT THE CONTRACT CSV.  It already carries every
    listed month per session with `till_last_trade_cd` -- the maturity in
    calendar days -- so the two nearest contracts and their T fall straight out
    of a sort, and no date arithmetic or second read of the panel is needed.  It
    is also already as_of aligned and already cached, so these rates land on
    exactly the sessions the books land on, which a separate read of the panel
    could not guarantee.

    A CONTRACT WITH NO `last_trade` IS DROPPED, NOT GUESSED.  Exactly one month
    per market lacks one -- the 2031M at the far end of the board, 52 rows of
    86,775 for 6E -- and it has never been within the nearest two of anything.
    Substituting a month-end would put a fabricated T into a formula that
    divides by it.
    """
    d = (w.select(["date", "close", "till_last_trade_cd"])
          .drop_nulls()
          .filter((pl.col("close") > 0) & (pl.col("till_last_trade_cd") >= 0))
          .sort(["date", "till_last_trade_cd"])
          .group_by("date", maintain_order=True)
          .agg([pl.col("close").head(2).alias("f"),
                pl.col("till_last_trade_cd").head(2).alias("t")])
          .filter(pl.col("f").list.len() == 2))
    if not d.height:
        return pl.DataFrame({"date": [], "Derived_Rate": []},
                            schema={"date": pl.Utf8, "Derived_Rate": pl.Float64})
    f1 = pl.col("f").list.get(0) / scale
    f2 = pl.col("f").list.get(1) / scale
    t1 = pl.col("t").list.get(0).cast(pl.Float64) / 365.25
    t2 = pl.col("t").list.get(1).cast(pl.Float64) / 365.25
    slope = (f2.log() - f1.log()) / (t2 - t1)
    return (d.with_columns(
                pl.when(t2 > t1)
                  .then((f1.log() - slope * t1).exp())
                  .otherwise(None).alias("Derived_Rate"))
             .select(["date", "Derived_Rate"]))


def _fx_cached(name: str, key: str, build):
    """Small keyed cache for the two network sources.  (frame, hit).

    THE CHECKS ARE KEYED ON THE PANEL EDGE, not on a file, because there is no
    local file to fingerprint.  The edge stands in for "a new session exists", so
    they refetch exactly when new data has arrived and a same-day rerun is free.
    A BUILD THAT RAISES IS NOT CACHED, so a transient outage cannot become a
    sticky empty column.
    """
    pq, kf = FX_CACHE / f"{name}.parquet", FX_CACHE / f"{name}.key"
    if pq.is_file() and kf.is_file() and kf.read_text().strip() == key:
        return pl.read_parquet(pq), True
    df = build()
    if df.height:
        FX_CACHE.mkdir(parents=True, exist_ok=True)
        df.write_parquet(pq)
        kf.write_text(key)
    return df, False


def fx_ndu(symbol: str, scale: float, col: str) -> pl.DataFrame:
    """(date, col) from the vendor's UNADJUSTED continuous.

    THE ACCENTED-HOSTNAME PATCH MUST PRECEDE THE IMPORT.  norgatehelper puts
    platform.node() straight into an HTTP header, this machine is named with an
    accent, headers must be ASCII, and the vendor answers 400 to its own
    import-time probe.  Harmless but noisy.  Same patch as Update.py and
    contract_cycles.py, and it is a third copy for the same reason those are two:
    platform.node() does not read COMPUTERNAME on Windows, so a subprocess cannot
    inherit the fix.
    """
    import platform
    n = platform.node()
    if not n.isascii():
        platform.node = lambda _v=n.encode("ascii", "ignore").decode(): _v
    import norgatedata as nd

    r = nd.price_timeseries(symbol, timeseriesformat="numpy-recarray",
                            datetimeformat="datetime64ns",
                            padding_setting=nd.PaddingType.NONE)
    if r is None or not len(r):
        return pl.DataFrame({"date": [], col: []},
                            schema={"date": pl.Utf8, col: pl.Float64})
    ds = np.datetime_as_string(np.asarray(r["Date"]).astype("datetime64[D]"),
                               unit="D")
    cl = np.asarray(r["Close"], dtype=float) / scale
    return (pl.DataFrame({"date": [str(x) for x in ds], col: cl})
              .filter(pl.col(col).is_finite() & (pl.col(col) > 0)))


def fx_yf(ticker: str, invert: bool, col: str) -> pl.DataFrame:
    """(date, col) from yfinance, already in CCY -> USD."""
    import yfinance as yf
    h = yf.Ticker(ticker).history(period="max", interval="1d")
    if h is None or not len(h):
        return pl.DataFrame({"date": [], col: []},
                            schema={"date": pl.Utf8, col: pl.Float64})
    v = np.asarray(h["Close"], dtype=float)
    df = pl.DataFrame({"date": [str(i)[:10] for i in h.index], col: v})
    df = df.filter(pl.col(col).is_finite() & (pl.col(col) > 0))
    return df.with_columns((1.0 / pl.col(col)).alias(col)) if invert else df


IRX_DIR = HERE / "IRX"
IRX_SYMBOL = "%IRX"
# 13-week US T-bill.  Quoted ANNUALISED IN PERCENT -- 4.52 means 4.52% -- so it
# divides by 100 before it is a rate at all.
IRX_SCALE = 100.0
# Actual days on a 13-week bill.  It appears in the discount-to-yield
# conversion, where 360 - d.n is the price per unit of face.
IRX_BILL_DAYS = 91


def irx_series(col: str = "irx_pct") -> pl.DataFrame:
    """(date, irx_pct) from the vendor's Economic database.

    THE SAME ACCENTED-HOSTNAME PATCH as everywhere else, and for the same
    reason: platform.node() reaches an HTTP header that must be ASCII.
    """
    import platform
    n = platform.node()
    if not n.isascii():
        platform.node = lambda _v=n.encode("ascii", "ignore").decode(): _v
    import norgatedata as nd

    r = nd.price_timeseries(IRX_SYMBOL, timeseriesformat="numpy-recarray",
                            datetimeformat="datetime64ns",
                            padding_setting=nd.PaddingType.NONE)
    if r is None or not len(r):
        return pl.DataFrame({"date": [], col: []},
                            schema={"date": pl.Utf8, col: pl.Float64})
    ds = np.datetime_as_string(np.asarray(r["Date"]).astype("datetime64[D]"),
                               unit="D")
    c = np.asarray(r["Close"], dtype=float)
    return (pl.DataFrame({"date": [str(x) for x in ds], col: c})
              .filter(pl.col(col).is_finite()))


def build_irx(grid: pl.Series | None, out_dir: Path,
              use_cache: bool = True) -> pl.DataFrame | None:
    """Write IRX/IRX.csv -- the risk-free rate, on the book's session grid.

        date, irx_pct, irx_bey_pct, rf_cal_day, cal_days_to_next,
        rf_accrual_next

    ONE SERIES, NOT AN INSTRUMENT, which is why it gets its own directory rather
    than a column in 63 books or a row in contract_cycles.csv.  It has no
    delivery month, no roll, no pointsize and no currency exposure; the only
    thing it shares with the panel is a calendar.

    BUILT IN STAGE 2 THOUGH IT IS USED IN STAGE 3.  Stage 3 reads only local
    files and needs no network, which is what lets a portfolio be rebuilt at a
    different tau or NAV offline in seconds; putting a vendor download there
    would spend that for nothing.  Stage 2 already talks to the vendor for the
    FX checks and already owns the caching and dual-write, so this is the same
    move `2_Engine/FX/` was.

    TWO CONVERSIONS, AND THEY ARE SEPARATE ERRORS IF EITHER IS SKIPPED.

    1. DISCOUNT -> YIELD.  %IRX is quoted on FACE and on ACT/360, so it
       understates the return twice: the gain is divided by face rather than by
       the price actually paid, and annualised over 360 days rather than 365.

           P     = 1 - d.n/360              price per $1 of face
           y365  = 365d / (360 - d.n)       bond-equivalent yield

       At d = 3.678%, n = 91 that is 3.764% -- the quote understates the true
       return by 2.34% OF ITSELF, and by more at higher rates.

    2. CALENDAR DAYS, NOT TRADING DAYS.  Cash accrues over a three-day weekend;
       it does not care that the exchange was shut.  So the rate is per CALENDAR
       day and the accrual is multiplied by the gap to the next session.

    Both collapse into one expression:

        rf_cal_day = d / (360 - d.n)

    which accumulated over 365 days returns exactly y365 -- the check that it is
    the right one.

    THE PREVIOUS FORM, `d / 256` PER TRADING DAY, WAS WRONG TWICE AND THE ERRORS
    HID EACH OTHER.  The per-day figure was 1.393x too large and applied on 256
    days instead of 365, so the annual total came out at 3.678% -- which merely
    reproduced the input, because dividing by 256 and multiplying by 256 is a
    tautology.  It looked self-consistent while carrying the whole
    discount-vs-yield error, and it short-changed every weekend.

    NEGATIVE VALUES ARE KEPT.  Eleven sessions carry a negative bill rate --
    minimum -0.105 -- and they are real: flights to quality in which investors
    paid to hold Treasuries.  Clipping them at zero would erase a genuine market
    state to make a series look tidy.

    COVERAGE IS COMPLETE: the vendor's history starts 1960-01-04 against a panel
    beginning 1978-03-07, so there is no head to backfill and no gap policy to
    invent.  A forward fill still applies WITHIN the range, because the bill
    market keeps its own holidays and a rate that did not print has not changed.
    """
    if grid is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    key = f"{_src_hash()}|{IRX_SYMBOL}|{grid[-1] if len(grid) else 'none'}"
    try:
        raw, hit = (_fx_cached("irx", key, irx_series) if use_cache
                    else (irx_series(), False))
    except Exception as exc:
        print(f"  [SKIP] IRX ({IRX_SYMBOL}): {type(exc).__name__}: {exc}")
        return None
    if not raw.height:
        print(f"  [SKIP] IRX ({IRX_SYMBOL}): vendor returned nothing")
        return None

    d = pl.col("irx_pct") / IRX_SCALE                      # decimal discount
    denom = 360.0 - d * IRX_BILL_DAYS
    df = (pl.DataFrame({"date": grid.to_list()}).sort("date")
            .join_asof(raw.sort("date"), on="date", strategy="backward")
            .with_columns([
                # Discount -> bond-equivalent yield, stored so the conversion is
                # auditable rather than buried inside the daily figure.
                (365.0 * d / denom * IRX_SCALE).alias("irx_bey_pct"),
                # THE PER-CALENDAR-DAY RATE.  d / (360 - d.n) accumulated over
                # 365 days returns exactly the BEY above, which is the check
                # that it is the right expression.
                (d / denom).alias("rf_cal_day"),
                # Calendar days to the NEXT session: 1 midweek, 3 over a
                # weekend, more over a holiday.  Cash does not care that the
                # exchange was shut.
                (pl.col("date").str.to_date().shift(-1)
                 - pl.col("date").str.to_date()).dt.total_days()
                .alias("cal_days_to_next")]))
    df = df.with_columns(
        # THE OVERNIGHT FACTOR, indexed at the session whose close it starts
        # from: the fraction of the cash balance earned between this session and
        # the next, and CREDITED AT THE NEXT.  Stage 3 multiplies the post-cost
        # balance by this and books the result one row later, which is how an
        # overnight sweep actually pays.  Null on the final row -- there is no
        # next session to be paid at.
        (pl.col("rf_cal_day") * pl.col("cal_days_to_next"))
        .alias("rf_accrual_next"))
    df.write_csv(out_dir / "IRX.csv")
    df.write_parquet(out_dir / "IRX.parquet")
    v = df.get_column("irx_pct")
    print(f"\n  IRX -> {out_dir}    {df.height:,} sessions"
          f"   {'cache hit' if hit else 'fetched'}")
    print(f"    {IRX_SYMBOL}  {raw.get_column('date').min()} .. "
          f"{raw.get_column('date').max()}   {raw.height:,} vendor bars")
    print(f"    on grid: mean {v.mean():.3f}%   min {v.min():.3f}%   "
          f"max {v.max():.3f}%   last {v[-1]:.3f}%")
    b = df.get_column("irx_bey_pct")
    print(f"    BEY    : mean {b.mean():.3f}%   last {b[-1]:.3f}%"
          f"   (+{b[-1] - v[-1]:.3f}pp over the quoted discount)")
    print(f"    rf_cal_day last {df.get_column('rf_cal_day')[-1]:.10f}"
          f"   x 365 = {df.get_column('rf_cal_day')[-1] * 365:.3%}")
    cd = df.get_column("cal_days_to_next").drop_nulls()
    print(f"    calendar gap to next session: mean {cd.mean():.2f}d"
          f"   max {cd.max()}d   (1 midweek, 3 over a weekend)")
    return df


def build_fx(fc, as_of, start: str, end: str, grid: pl.Series | None,
             out_dir: Path, use_cache: bool = True,
             checks: bool = True) -> dict[str, pl.DataFrame]:
    """Write FX/<CCY>.csv, one file per rate.  Returns the frames.

    THE EMIT GRID IS THE BOOK'S OWN SESSION UNION when it is available, so every
    FX file lines up row-for-row with the books and a plain join on date is
    correct.  Without it the alignment traps are nasty and silent: markets keep
    different holidays, so an equality join against a currency's native sessions
    leaves 98-224 nulls per instrument on this panel, and USD -- a constant
    defined on every date that ever existed -- would still be missing 319 ZC
    sessions that predate the first FX contract in 1979.

    RATES ARE CARRIED OVER A GAP WITH A BACKWARD AS-OF, which is what a desk does
    with a rate over a holiday.  Rows before a currency's first observation stay
    null rather than being back-filled; Status says NO_DERIVED and the caller
    decides.  A null Derived_Rate is never repaired from the check columns -- a
    column named Derived_Rate that is sometimes not derived is a worse object
    than a null.

    THE DIFFS ARE COMPUTED ON EXACT DATE MATCHES, BEFORE THE CARRY.  Comparing an
    as-of'd stale value against a fresh one would manufacture disagreements out
    of holidays and put them in the same column that is supposed to reveal real
    ones.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    edge = as_of or "none"
    frames: dict[str, pl.DataFrame] = {}
    rows = []
    for ccy, m in FX_CCY.items():
        # -- Derived, from this currency's own worksheet -----------------
        if m["inst"] is not None:
            w, _ = cached_worksheet(fc, m["inst"], start, end, as_of,
                                    use_cache=use_cache)
            nat = fx_derived(w, m["scale"])
        else:
            base = grid if grid is not None else pl.Series("date", [])
            nat = pl.DataFrame({"date": base.to_list()}).with_columns(
                pl.lit(m["fixed"], dtype=pl.Float64).alias("Derived_Rate"))
        nat = nat.sort("date")

        # -- the two checks, exact-matched onto those sessions -----------
        for col, key, fn in (
                ("NDU_Rate", m["ndu"],
                 (lambda mm=m: fx_ndu(mm["ndu"], mm["scale"], "NDU_Rate"))),
                ("YF_Rate", m["yf"],
                 (lambda mm=m: fx_yf(mm["yf"], mm["inv"], "YF_Rate")))):
            got = None
            if checks and key:
                try:
                    got, _hit = _fx_cached(f"{col[:3].lower()}_{ccy}",
                                           f"{key}|{m['scale']}|{m['inv']}|{edge}",
                                           fn) if use_cache else (fn(), False)
                except Exception as exc:
                    print(f"  [SKIP] {ccy} {col} ({key}): "
                          f"{type(exc).__name__}: {exc}")
                    got = None
            nat = (nat.join(got.sort("date"), on="date", how="left")
                   if got is not None and got.height
                   else nat.with_columns(pl.lit(None, dtype=pl.Float64).alias(col)))

        nat = nat.with_columns([
            ((pl.col("Derived_Rate") / pl.col("NDU_Rate") - 1.0) * 1e4)
                .alias("NDU_diff_bp"),
            ((pl.col("Derived_Rate") / pl.col("YF_Rate") - 1.0) * 1e4)
                .alias("YF_diff_bp")])
        worst = pl.max_horizontal(pl.col("NDU_diff_bp").abs(),
                                  pl.col("YF_diff_bp").abs())
        nat = nat.with_columns(
            pl.when(pl.col("Derived_Rate").is_null()).then(pl.lit("NO_DERIVED"))
              .when(worst.is_null()).then(pl.lit("UNCHECKED"))
              .when(worst >= FX_ALERT_BP).then(pl.lit("ALERT"))
              .when(worst >= FX_WATCH_BP).then(pl.lit("WATCH"))
              .otherwise(pl.lit("OK")).alias("Status"))

        # -- carry onto the book's grid ---------------------------------
        if grid is not None and m["inst"] is not None:
            nat = (pl.DataFrame({"date": grid.to_list()}).sort("date")
                     .join_asof(nat, on="date", strategy="backward"))
            nat = nat.with_columns(
                pl.col("Status").fill_null(pl.lit("NO_DERIVED")))

        nat = nat.select(["date", "Derived_Rate", "NDU_Rate", "YF_Rate",
                          "NDU_diff_bp", "YF_diff_bp", "Status"])

        # THE FILE STARTS WHERE THE RATE STARTS.  The book's grid reaches back
        # ~433 sessions before the first FX contract ever traded, and those rows
        # would carry a null Derived_Rate -- no information, and actively
        # harmful: polars infers csv dtypes from a bounded prefix, so a run of
        # leading blanks longer than that window makes `pl.read_csv` hand back
        # Derived_Rate as a **String**, and arithmetic on it raises (or worse, a
        # comparison quietly does something string-y).  Trimming them means the
        # one column the book actually reads is numeric from row 1 under even a
        # naive read.  Interior nulls cannot occur -- the as-of carry fills every
        # session after the first -- so this only ever removes a leading block.
        #
        # NDU_Rate and YF_Rate can still trip that inference, since yfinance
        # starts in 2003 and the file starts in 1979.  That is what `load_fx`
        # is for; it is not worth truncating real history to flatter a default.
        if m["inst"] is not None:
            nat = nat.filter(
                pl.col("date") >= pl.lit(
                    nat.filter(pl.col("Derived_Rate").is_not_null())
                       .get_column("date").min() or ""))
        # CSV FIRST, PARQUET SECOND, and the order is load-bearing: `_prefer_parquet`
        # rejects a parquet older than its csv, so writing them this way round
        # means a run interrupted between the two leaves the csv winning rather
        # than a half-updated pair where the stale file is the one preferred.
        nat.write_csv(out_dir / f"{ccy}.csv")
        nat.write_parquet(out_dir / f"{ccy}.parquet")
        frames[ccy] = nat

        nb = nat.get_column("NDU_diff_bp").drop_nulls().abs()
        yb = nat.get_column("YF_diff_bp").drop_nulls().abs()
        both = pl.concat([nb, yb])
        rows.append((ccy, nat.height,
                     nat.get_column("Derived_Rate").drop_nulls().len(),
                     len(nb), len(yb),
                     both.mean() if len(both) else None,
                     both.quantile(0.95) if len(both) else None,
                     both.max() if len(both) else None,
                     nat.filter(pl.col("Status") == "WATCH").height,
                     nat.filter(pl.col("Status") == "ALERT").height,
                     nat.get_column("Derived_Rate").drop_nulls().tail(1)))

    hdr = (f"\n{'ccy':<6}{'rows':>9}{'derived':>9}{'NDU':>8}{'yf':>8}"
           f"{'mean bp':>10}{'p95 bp':>9}{'max bp':>10}{'WATCH':>7}{'ALERT':>7}"
           f"{'last rate':>14}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    dash = lambda v, w, p=1: (f"{'-':>{w}}" if v is None or v != v
                              else f"{v:>{w},.{p}f}")
    for (c, n, nd_, a, b, mu, p95, mx, wt, al, last) in rows:
        print(f"{c:<6}{n:>9,}{nd_:>9,}{a:>8,}{b:>8,}{dash(mu, 10)}"
              f"{dash(p95, 9)}{dash(mx, 10)}{wt:>7,}{al:>7,}"
              f"{(last[0] if len(last) else float('nan')):>14.6f}")
    print("-" * (len(hdr) - 1))
    n_alert = sum(r[9] for r in rows)
    print(f"  FX -> {out_dir}    {len(frames)} rates, one file each")
    print(f"  the book reads Derived_Rate; NDU_Rate and YF_Rate are checks")
    if n_alert:
        print(f"  [WATCH] {n_alert:,} rows disagree by >= {FX_ALERT_BP:.0f}bp "
              f"-- inspect with Status == 'ALERT'")
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default=None,
                    help="one instrument; default is every ruled instrument")
    ap.add_argument("--start", default="1900-01-01")
    ap.add_argument("--end", default="2100-01-01")
    ap.add_argument("--out", default=str(BOOK))
    ap.add_argument("--all-rows", action="store_true",
                    help="every listed month, not just the held contract")
    ap.add_argument("--keep-rule-name", action="store_true",
                    help="hold column keeps the rule's own name per instrument")
    ap.add_argument("--as-of", default="auto", dest="as_of",
                    help="square the panel off at this date; 'none' to disable")
    ap.add_argument("--no-cache", action="store_true",
                    help="rebuild every worksheet, ignoring the cache")
    ap.add_argument("--clear-cache", action="store_true",
                    help="delete the worksheet cache and exit")
    ap.add_argument("--no-fx", action="store_true",
                    help="skip the FX rate files entirely")
    ap.add_argument("--no-fx-checks", action="store_true",
                    help="FX from the local panel only -- no vendor, no network")
    ap.add_argument("--jobs", type=int, default=2,
                    help="worker processes. DEFAULT 2. Bound by MEMORY, not "
                         "cores -- see the note in main()")
    args = ap.parse_args()

    if args.clear_cache:
        import shutil
        if CACHE.is_dir():
            n = len(list(CACHE.glob("*.parquet")))
            shutil.rmtree(CACHE)
            print(f"cache cleared: {n} worksheet(s) removed from {CACHE}")
        else:
            print(f"no cache at {CACHE}")
        return 0

    fc = _load(FC, "fc")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    as_of = None if args.as_of == "none" else args.as_of
    if as_of == "auto":
        as_of, edge = fc.panel_as_of()
        print("PANEL EDGE")
        fc.report_edge(as_of, edge)
        print("")

    rule_of = rules()
    if args.instrument:
        if args.instrument not in rule_of:
            raise SystemExit(f"[ABORT] {args.instrument} has no Roll_Rule in "
                             f"{CYCLES.name}")
        rule_of = {args.instrument: rule_of[args.instrument]}

    print(f"trading book -> {out_dir}    as_of {as_of or 'none'}")
    # SAY WHAT IS ABOUT TO HAPPEN BEFORE IT TAKES A MINUTE.  The line below is
    # the last thing printed before the workers start, and on a cold cache the
    # first result is a few seconds away -- so it names the work, the worker
    # count and whether the cache is warm, which is what decides between "a few
    # seconds" and "a few minutes".
    n_cached = len(list(CACHE.glob("*.parquet"))) if CACHE.is_dir() else 0
    print(f"  building {len(rule_of)} instrument(s) on {max(1, args.jobs)} "
          f"worker(s); worksheet cache holds {n_cached}"
          + ("  (cold -- first rows in a few seconds, full run ~80s+)"
             if n_cached < len(rule_of) else "  (warm)"),
          flush=True)
    hdr = (f"\n{'inst':<8}{'rule':<18}{'hold column':<26}{'rows':>10}"
           f"{'sessions':>10}{'sec':>6}{'done':>9}")
    print(hdr, flush=True)
    print("-" * (len(hdr) - 1), flush=True)

    tot_rows = tot_sess = 0
    written, holes, unmeasured_all = [], [], []
    # PARALLELISM IS MEMORY-BOUND HERE, NOT CPU-BOUND.  TWO, ALWAYS, unless
    # somebody deliberately says otherwise.
    #
    # Sizing this by os.cpu_count() took a 24-core / 16 GB machine down.  Each
    # worker holds a whole instrument -- CL is 623,358 rows, and the per-session
    # loop materialises them as Python dicts, on the order of a gigabyte before
    # polars' own buffers.  Twenty-four of those do not fit in 16 GB.  Polars
    # also starts a thread pool sized to the core count INSIDE every worker, so
    # 24 workers asked for roughly 576 threads as well.
    #
    # Two fits with room to spare on 16 GB, and it is a FIXED number rather than
    # one derived from cores or free memory on purpose: a default that varies
    # with the machine is a default nobody can reason about, and the run that
    # breaks is the one on the day memory happened to be tight.  The worksheet
    # cache means a warm rebuild is seconds anyway, so there is very little left
    # for more workers to buy -- 4 was tried on a machine with 5.1 GB free and
    # would not have fitted.
    #
    # Raise it only deliberately, having looked at free memory: budget ~1.5 GB
    # per worker.
    jobs = max(1, args.jobs)
    if jobs > 1:
        # Cap polars threads per worker so N workers do not each claim the whole
        # machine.  Must be set before polars is imported in the child.
        os.environ.setdefault("POLARS_MAX_THREADS", "2")
    tasks = [(inst, rule, args.start, args.end, as_of, not args.all_rows,
              not args.keep_rule_name, str(out_dir), not args.no_cache)
             for inst, rule in sorted(rule_of.items())]

    # Instruments are fully independent: separate inputs, separate output file,
    # no shared state.  So this parallelises cleanly, and it is where the time
    # is -- the per-session loop in front_contract is sequential by nature
    # (B_V_3 streaks and every ratchet carry state across sessions) and cannot
    # be vectorised without changing what it computes.
    # RESULTS ARE CONSUMED AS THEY ARRIVE, NOT COLLECTED FIRST.  This was
    # `list(ex.map(...))`, which blocks until all 63 instruments are done and
    # only then prints 63 rows at once.  Nothing reached the console for the
    # whole build -- 78s warm, minutes cold -- so a run looked hung immediately
    # after the PANEL EDGE banner, and Update.py's progress bar, which counts
    # exactly these rows, had nothing to count until the stage was already over.
    #
    # `.map` yields IN SUBMISSION ORDER, which keeps the table alphabetical.
    # That costs nothing here: with jobs=2 at most one instrument can finish
    # ahead of its turn, so a row is never held back by more than one
    # instrument's work.  as_completed would stream marginally sooner and
    # scramble the ordering, which is a bad trade for a table someone reads.
    #
    # THE POOL MUST STAY OPEN WHILE THE ITERATOR IS DRAINED, hence the context
    # manager around the loop rather than around the map call.
    #
    # A HEARTBEAT ON TOP OF THE STREAMING, because streaming alone still leaves
    # gaps.  Measured cold with 2 workers: 67 progress lines over 75s, median
    # gap 0.3s -- but a worst gap of 11.1s, because in-order yielding waits on
    # whichever instrument is slowest (CL is 623,358 rows) while its neighbours
    # sit finished behind it.  Eleven seconds of nothing is short of alarming
    # but still long enough to wonder.  `wait([f], timeout=...)` blocks on the
    # NEXT result specifically rather than on any result, so ticking costs no
    # spinning and the table stays alphabetical.
    #
    # THE TICK LINE MUST NOT LOOK LIKE A RESULT ROW.  Update.py drives its
    # progress bar off `^\S+\s+\S*roll\S*\s`, so anything starting with
    # whitespace is safely ignored by it and simply passes through.
    from contextlib import nullcontext
    HEARTBEAT = 5.0

    def _serial():
        for t in tasks:
            yield _one(t)

    def _parallel(pool):
        from concurrent.futures import wait as _fwait
        futs = [pool.submit(_one, t) for t in tasks]
        for i, f in enumerate(futs):
            while not f.done():
                _fwait([f], timeout=HEARTBEAT)
                if not f.done():
                    print(f"  ... building, {i} of {len(futs)} done, "
                          f"{time.time() - t_build:.0f}s elapsed", flush=True)
            yield f.result()

    hits = 0
    frames = {}
    no_next = []
    t_build = time.time()
    n_task = len(tasks)
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        _pool = ProcessPoolExecutor(max_workers=jobs)
        ctx, stream = _pool, _parallel(_pool)
    else:
        ctx, stream = nullcontext(), _serial()
    with ctx:
        for k, (inst, h, n_sess, n_avail, unm, err, hit, frame,
                nnn) in enumerate(stream, 1):
            if err:
                print(f"{inst:<8}{err}", flush=True)
                continue
            rule = rule_of[inst]
            # `sec` is elapsed for the STAGE, not for this instrument: the
            # workers overlap, so a per-instrument time would not sum to the
            # total and would invite exactly that arithmetic.  Read as a clock,
            # it also answers the question the column is really there for --
            # how long has this been going, and is it still moving.
            print(f"{inst:<8}{rule:<18}{HOLD_FOR[rule]:<26}{h:>10,}"
                  f"{n_sess:>10,}{time.time() - t_build:>6.0f}"
                  f"{k:>5}/{n_task}", flush=True)
            if unm:
                unmeasured_all.append((inst, unm))
            # One row per session is the contract: fewer means the rule held
            # NOTHING on those sessions, which is a hole in the position series.
            if not args.all_rows and n_sess < n_avail:
                holes.append((inst, n_avail - n_sess, n_avail))
            tot_rows += h
            tot_sess += n_sess
            hits += int(hit)
            frames[inst] = frame
            if nnn:
                no_next.append((inst, nnn, n_sess))
            written.append(inst)

    print("-" * (len(hdr) - 1))

    # ---- Equation 3.13, the one column that needs the whole book ----------
    #
    # Every other column is a function of one instrument, which is why the
    # workers can run independently.  The XS z-score is not: it scores each
    # instrument against the SAME SESSION'S cross-section, so it can only be
    # computed once every instrument is in hand.  That is why the workers hand
    # their frames back rather than writing them.
    xs_rows = xs_total = 0
    grid = None
    if frames:
        # The union of every instrument's sessions.  Markets keep different
        # holidays, so no single calendar covers the book; aligning on one
        # instrument's would score the others against the wrong day.
        grid = pl.Series("date", sorted(
            set().union(*(set(f.get_column("date").to_list())
                          for f in frames.values()))))
        panel = {}
        for inst, f in frames.items():
            # One value per session.  Under --all-rows the frame repeats each
            # session across every listed month; r256 is identical down them,
            # so take the first and the cross-section is unchanged.
            per = (f.select(["date", "r256"])
                    .unique(subset=["date"], keep="first"))
            aligned = (pl.DataFrame({"date": grid})
                       .join(per, on="date", how="left"))
            panel[inst] = aligned.get_column("r256")
        z, _, xs_rows, xs_total = cross_sectional_z(panel, grid)
        for inst in frames:
            zf = pl.DataFrame({"date": grid, "XS_trend": z[inst]})
            f = frames[inst].join(zf, on="date", how="left").drop("r256")
            # Eq 3.14 + 3.21, on ONE ROW PER SESSION.  Under --all-rows the
            # frame repeats each session across every listed month, and the
            # 1,280-session rolling window inside normalise_signal would then
            # span a fraction of the history it should.  Collapse, compute,
            # join back -- the same treatment every other per-session series
            # gets.
            per = (f.select(["date", "TS_trend", "XS_trend"])
                    .unique(subset=["date"], keep="first").sort("date"))
            per = per.with_columns(
                # cap=None: these are HALVES of one alpha, not alphas.  The
                # cap is applied once, to the blend.  The names carry the
                # _UNCAPPED suffix because every other *_sign column in this
                # file is bounded to +/-20 and these two are not -- they reach
                # 187 and 121 on this panel.  Anything that assumes the +/-20
                # contract must read Trend_sign, not these.
                normalise_signal(per.get_column("TS_trend"),
                                 cap=None).alias("TS_trend_sign_UNCAPPED"),
                normalise_signal(per.get_column("XS_trend"),
                                 cap=None).alias("XS_trend_sign_UNCAPPED"))
            per = per.with_columns(
                trend_sign(per.get_column("TS_trend_sign_UNCAPPED"),
                           per.get_column("XS_trend_sign_UNCAPPED")).alias("Trend_sign"))
            f = f.join(
                per.select(["date", "TS_trend_sign_UNCAPPED", "XS_trend_sign_UNCAPPED", "Trend_sign"]),
                on="date", how="left")

            # Equation 3.22 -- the four alphas at 1/N.
            #
            #   f_raw = 0.25 trend + 0.25 carry + 0.25 skew + 0.25 vov
            #
            # EQUAL WEIGHTS ONLY MEAN SOMETHING BECAUSE OF 3.21.  Before
            # normalisation these four were a percentage roll yield, a third
            # moment, a regime ratio and a price-unit trend; 0.25 each would
            # have allocated by unit, not by conviction.  After it they all
            # average close to Phi = 10, so the weights allocate capital.
            #
            # NO CAP, AND NONE NEEDED: each input is already bounded to +/-20,
            # so their equal-weighted mean cannot leave that range.  Clipping
            # would be a no-op that implied otherwise.
            #
            # NULL WHERE ANY ALPHA IS NULL.  Treating a missing signal as 0
            # keeps more rows but silently rebalances the blend -- an
            # instrument with no carry view would put 25% of its weight on "no
            # opinion" rather than the other three sharing it.  A null says the
            # aggregate is not computable and leaves that choice to the caller.
            # NOTE the thesis code does zero-fill (`carry_alpha` returns zeros
            # when carry is unavailable; skew and vov are _zerofill'd).
            frames[inst] = f.with_columns(
                (0.25 * pl.col("Trend_sign") + 0.25 * pl.col("Carry_sign")
                 + 0.25 * pl.col("Skew_sign") + 0.25 * pl.col("VoV_sign"))
                .alias("Sign_raw"))

        # ---- Equation 3.23, pooled over the universe -----------------------
        #
        # One FDM per SESSION, shared by every instrument, because the paper
        # estimates R_t on the whole book rather than per market -- a 4x4
        # correlation from one instrument is a far noisier object than one from
        # 62, and per-instrument fitting is where overfitting would enter.
        #
        # Built from the CROSS-SECTIONAL MEAN of each alpha: one series per
        # signal, averaged across instruments each session.  That is the same
        # union grid the XS z-score uses, and for the same reason -- markets
        # keep different holidays, so a single instrument's calendar would
        # silently drop the others from that day's average.
        gd = grid.to_list()
        pos = {d: k for k, d in enumerate(gd)}
        panels = {c: np.full((len(gd), len(frames)), np.nan)
                  for c in ("Trend_sign", "Carry_sign", "Skew_sign", "VoV_sign")}
        for col, inst in enumerate(sorted(frames)):
            fr = (frames[inst].select(["date"] + list(panels))
                  .unique(subset=["date"], keep="first"))
            rows = [pos.get(d) for d in fr.get_column("date").to_list()]
            for c in panels:
                vals = fr.get_column(c).to_numpy().astype(float)
                for r, v in zip(rows, vals):
                    if r is not None:
                        panels[c][r, col] = v
        # "Mean of empty slice" is EXPECTED here and is not suppressed by
        # np.errstate -- it is a RuntimeWarning from numpy's warnings
        # machinery, not a floating-point error state.  Early union-grid rows
        # have no instrument with a signal yet, so the row-wise nanmean is over
        # nothing; NaN is the right answer and pooled_fdm holds those bars at
        # FDM = 1.0.  Filtered by MESSAGE rather than blanket-silencing
        # RuntimeWarning, so anything else numpy wants to say still gets
        # through -- a nightly run nobody reads is worse than one that warns.
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.filterwarnings("ignore", message="Mean of empty slice",
                                    category=RuntimeWarning)
            means = np.column_stack([np.nanmean(panels[c], axis=1)
                                     for c in ("Trend_sign", "Carry_sign",
                                               "Skew_sign", "VoV_sign")])
        fdm = pooled_fdm(means)
        fdm_f = pl.DataFrame({"date": grid, "fdm_raw": fdm})
        for inst in frames:
            frames[inst] = frames[inst].join(fdm_f, on="date", how="left")

        # ---- Equation 3.24 -------------------------------------------------
        #
        #     f_fdm(i,t) = f_raw(i,t) x FDM_t
        #
        # The multiplier undoes the mechanical weakening of averaging four
        # decorrelated signals: mean |Sign_raw| is 5.57 against ~9.2 for each
        # of its inputs, and after this it is 9.60 -- back in line with the
        # alphas it was built from.
        #
        # THIS COLUMN IS NOT BOUNDED TO +/-20.  Sign_raw is, and FDM reaches
        # 2.0, so the product spans +/-40.  That is deliberate at this step and
        # is exactly why the paper calls it leverage and re-normalises it next
        # (3.25); nothing should size off this column directly.
        for inst in frames:
            frames[inst] = frames[inst].with_columns(
                (pl.col("Sign_raw") * pl.col("fdm_raw")).alias("fdm_norm"))

        # ---- Equations 3.25 / 3.26 -- the master forecast -------------------
        #
        #   S2_t     = 10 / ( (1/W) sum |f_fdm(i,tau)| ),   W = 1,280
        #   f_master = clip( f_fdm x S2_t, -20, +20 )
        #
        # THE SECOND NORMALISATION, and it is the same operation as 3.21 --
        # same Phi, same window, same cap -- applied to a different input.  So
        # it reuses `normalise_signal` rather than restating it.
        #
        # WHY IT IS NEEDED, in the paper's words: without it "the magnitude of
        # the signal would drift and 10 would no longer be a normal position".
        # 3.24 multiplied by a number that moves between 1.0 and 2.0 over time,
        # so the +/-10 convention 3.21 established no longer holds -- measured
        # here, mean |fdm_norm| is 9.05 but 6.7% of rows sit outside +/-20
        # entirely.  This restores the scale.
        #
        # PER INSTRUMENT, NOT POOLED.  The sum in 3.25 carries the i index, so
        # each market is normalised against its OWN history -- unlike FDM_t
        # above, which is deliberately estimated on the whole universe.  The
        # two are different objects despite sitting one line apart.
        for inst in frames:
            frames[inst] = frames[inst].with_columns(
                normalise_signal(
                    frames[inst].get_column("fdm_norm")).alias("FDM_MASTER"))

        # ---- Equation 3.27 -- signal smoothing ------------------------------
        #
        #   f_smooth,t = a * f_t + (1-a) * f_smooth,t-1,   a = 1 - e^(-ln2/1) = 0.5
        #
        # A HALF-LIFE OF ONE DAY, matched to the rebalance rate: the portfolio
        # trades daily, so a memory longer than that would have the book chasing
        # a signal it has already acted on, and a shorter one would not damp
        # anything.  Half of today, half of everything before.
        #
        # `adjust=False` IS THE EQUATION.  3.27 is written as the recursion, and
        # only adjust=False computes that; adjust=True would compute a
        # normalised weighted average instead, which differs while the window
        # fills.  Same choice, same reason, as `ewmac` above -- and the opposite
        # of the volatility legs, which follow the thesis implementation's
        # pandas default.  All three conventions coexist in this file on purpose.
        #
        # NO CAP NEEDED.  A convex combination of values bounded to +/-20 is
        # itself bounded to +/-20, so the range of FDM_MASTER survives untouched.
        for inst in frames:
            frames[inst] = frames[inst].with_columns(
                pl.col("FDM_MASTER")
                  .ewm_mean(alpha=0.5, adjust=False, min_samples=1,
                            ignore_nulls=False)
                  .alias("FDM_MASTER_smooth"))

        # ---- Equation 3.31 -- the final signal -----------------------------
        #
        #   f_final = clip( f_smooth * g_vol * g_dd, -20, +20 )
        #
        # WITHOUT THE GATES.  g_vol (3.29) and g_dd (3.30) are not computed in
        # this file -- see Trading_Book_Journal.md, 2026-08-28.  A drawdown gate
        # measured on the STRATEGY'S P&L needs the position, which needs the
        # forecast, so it is a portfolio-level object rather than something
        # derivable from one instrument's price history.  Both gates are applied
        # downstream, where the position is known.
        #
        # THE CLIP CANNOT BIND, either here or in the full equation, and it is
        # worth saying so rather than implying a bound is being enforced.
        # FDM_MASTER is clipped to +/-20 at 3.26; 3.27 smooths it, and a convex
        # combination of values in [-20, 20] stays in [-20, 20].  Both gates lie
        # in (0.5, 1.0], so multiplying by them only ever SHRINKS the magnitude.
        # The column exists to name the end of the chain, not to enforce a
        # range that is already guaranteed.
        for inst in frames:
            frames[inst] = frames[inst].with_columns(
                pl.col("FDM_MASTER_smooth").clip(-SIGNAL_CAP, SIGNAL_CAP)
                  .alias("SIGNAL"))
        n_active = int((fdm > FDM_FLOOR).sum())

    # ---- FX, one file per rate, then one column per book -----------------
    #
    # BUILT AFTER THE FRAMES BUT BEFORE THEY ARE WRITTEN.  It needs `grid` -- the
    # session union XS_trend is scored on -- so it cannot run earlier; and every
    # book carries an `FX_rate` column off the back of it, so it cannot run
    # later.  Under --instrument grid is that one instrument's sessions, which is
    # narrower than the book but consistent with what was actually built.
    fx_frames = {}
    if not args.no_fx:
        try:
            fx_frames = build_fx(fc, as_of, args.start, args.end, grid, FX_DIR,
                                 use_cache=not args.no_cache,
                                 checks=not args.no_fx_checks)

        except Exception as exc:
            # A FAILED FX BUILD MUST NOT DISCARD THE BOOKS.  They are complete
            # apart from one column, and losing a 12-minute cold run over a
            # network blip would be the wrong trade.  The column is dropped, the
            # warning is loud, and the rate files keep their previous contents.
            print(f"\n  [WARN] FX build failed: {type(exc).__name__}: {exc}")
            print("         Books will be written WITHOUT FX_rate. "
                  "Rerun, or use --no-fx to skip deliberately.")

    # IRX rides with the FX build -- same stage, same vendor session, same grid
    # -- but under its OWN guard.  Sharing the FX try/except reported an IRX
    # NameError as "FX build failed", which sent the next reader to the wrong
    # function entirely.
    if not (args.no_fx or args.no_fx_checks):
        try:
            build_irx(grid, IRX_DIR, use_cache=not args.no_cache)
        except Exception as exc:
            print(f"\n  [WARN] IRX build failed: "
                  f"{type(exc).__name__}: {exc}")
            print("         Books and FX are unaffected; stage 3 has no rate.")

    # ---- FX_rate: the instrument's own currency, on its own sessions ------
    #
    # ONE COLUMN, ALREADY RESOLVED, so nothing downstream has to know the
    # currency map or which file to open.  The rate is CCY -> USD: multiply a
    # local-currency amount by it to get USD.
    #
    # A PLAIN JOIN ON DATE IS CORRECT HERE and the alignment is not accidental --
    # the rate files are emitted on this same `grid`, which is exactly why they
    # are built inside this run.  Under --all-rows the frame repeats each session
    # across every listed month and the join broadcasts down them, which is right:
    # one rate per session, whichever month the row describes.
    #
    # A NULL IS POSSIBLE AND IS HONEST.  YAP4's book starts 1983-02-16 while the
    # AUD future starts 1987-01-13, so its first 983 sessions have no rate that
    # ever existed to carry.  Filling those with 1.0, or with the earliest known
    # rate, would invent an exchange rate for a date nobody quoted one.
    fx_missing = []
    if fx_frames:
        for inst in list(frames):
            ccy = currency_of(inst)
            r = (fx_frames[ccy].select(["date", pl.col("Derived_Rate")
                                                  .alias("FX_rate")]))
            frames[inst] = frames[inst].join(r, on="date", how="left")
            # ---- Equation 3.35, completed ------------------------------
            #
            #     price_vol_USD_ann = price_vol_curr_ann x FX_rate
            #
            # `price_vol_curr_ann` is 3.35 with the FX leg deliberately left
            # off -- sigma_hat x pointsize x sqrt(256), in whatever currency the
            # contract quotes in.  That intermediate is meaningful on its own
            # but is NOT COMPARABLE ACROSS MARKETS: FDAX9's figure is euros,
            # SJB's is yen, ZC's is already dollars.  This multiplies the last
            # unit change through, so one contract's annual risk is on one scale
            # for the whole book and a position sizer can finally compare them.
            #
            # IT HAS TO HAPPEN HERE, not in book_one with the rest of the vol
            # chain, because FX_rate does not exist until the rates are built --
            # and they are built from the session union of every book, which is
            # only known once the workers have finished.
            #
            # NULL WHERE THE RATE IS NULL, and that is the honest answer: YAP4
            # and YXT4 open before the AUD future existed, so there is no rate
            # to convert with.  Polars propagates the null through the product
            # rather than inventing a 1.0, which would silently report an AUD
            # figure as though it were dollars.
            frames[inst] = frames[inst].with_columns(
                (pl.col("price_vol_curr_ann") * pl.col("FX_rate"))
                .alias("price_vol_USD_ann"))
            n = frames[inst].get_column("FX_rate").null_count()
            if n:
                fx_missing.append((inst, ccy, n, frames[inst].height))

    # DUAL-WRITE.  The csv is for reading by eye; the parquet is what a program
    # should read, because dtypes travel inside it and the leading-null inference
    # trap documented on `load_book` cannot occur.  It is also 2.9x smaller
    # (333 MB -> 115 MB across the book) and 2.4x faster to load.
    #
    # CSV FIRST -- see the same note in build_fx.  `_prefer_parquet` refuses a
    # parquet older than its csv, so this order degrades safely if a run dies
    # mid-loop.
    for inst, f in frames.items():
        f.write_csv(out_dir / f"{inst}.csv")
        f.write_parquet(out_dir / f"{inst}.parquet")

    if fx_frames:
        by_ccy = {}
        for inst in frames:
            by_ccy[currency_of(inst)] = by_ccy.get(currency_of(inst), 0) + 1
        print(f"\n  FX_rate attached to {len(frames)} book(s): "
              + "  ".join(f"{c} {n}" for c, n in sorted(by_ccy.items(),
                                                        key=lambda kv: -kv[1])))
        if fx_missing:
            print(f"  [NOTE] FX_rate is null on some sessions -- the book "
                  f"predates the currency's future, which has no rate to carry:")
            for inst, ccy, n, tot in sorted(fx_missing,
                                            key=lambda x: -x[2]):
                print(f"     {inst:<10}{ccy}  {n:,} of {tot:,} sessions")

    print("")
    print(f"{len(written)} instruments, {tot_rows:,} rows, "
          f"{tot_sess:,} instrument-sessions")
    if frames:
        pct = 100.0 * xs_rows / xs_total if xs_total else 0.0
        print(f"  XS_trend: {xs_rows:,} of {xs_total:,} union sessions scored "
              f"({pct:.1f}%)")
        if len(frames) < XS_MIN_INSTS:
            # A z-score needs a cross-section.  With --instrument there is none,
            # and the column is entirely null -- say so, because a silently
            # empty forecast column is worse than an absent one.
            print(f"  [WARN] only {len(frames)} instrument(s) in this run, so "
                  f"there is no cross-section to score against and XS_trend is "
                  f"ENTIRELY NULL. Eq 3.13 needs >= {XS_MIN_INSTS}; run the "
                  f"full book for a usable column.")
    print(f"  worksheet cache: {hits} hit, {len(written)-hits} rebuilt")
    if no_next:
        n = sum(k for _, k, _ in no_next)
        print("")
        print(f"  [WARN] carry_hold is NULL on {n:,} session(s) across "
              f"{len(no_next)} instrument(s): the held contract was the "
              f"furthest-dated month listed, so there is no next leg:")
        for inst, k, tot in sorted(no_next, key=lambda r: -r[1])[:8]:
            print(f"     {inst:<8}{k:>7,} of {tot:,} sessions")
    if holes:
        print("")
        print(f"  [WARN] {len(holes)} instrument(s) HOLD NOTHING on some "
              f"sessions -- those sessions have no row at all:")
        for inst, n, tot in holes:
            print(f"     {inst:<8}{n:>7,} of {tot:,} sessions missing")
    elif not args.all_rows:
        print("  every session has a held contract; no holes")
    if unmeasured_all:
        n = sum(len(u) for _, u in unmeasured_all)
        print("")
        print(f"  [WARN] {n} roll(s) across {len(unmeasured_all)} instrument(s) "
              f"had no quote for the outgoing contract on the roll date, so the "
              f"gap could not be measured and the jump stays in the series:")
        for inst, u in unmeasured_all[:5]:
            for dt, o, nw in u[:3]:
                print(f"     {inst:<8}{dt}  {o} -> {nw}")
    # Only meaningful on a full run: with --instrument the other 62 are absent
    # because they were not asked for, not because they lack a rule.
    if not args.instrument:
        missing = sorted(set(pl.read_csv(CYCLES, infer_schema_length=0)
                             .get_column("instrument").to_list()) - set(written))
        if missing:
            print(f"NOT written ({len(missing)}), no Roll_Rule: "
                  f"{', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
