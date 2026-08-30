"""
Delivery-cycle table: one row per instrument, one column per contracts-per-year.

Norgate exposes no delivery-cycle field, so the cycle is counted from the month
codes present in `futures_market_session_contracts(inst)`.

`per_year` is simply len(codes) and is written as such.  There used to be twelve
one-hot columns "1".."12" marking the same count; they were removed on
2026-08-28 after checking the identity held on all 63 rows with no exceptions,
and nothing read them.  `codes` carries the month letters themselves -- two
markets can share a count and not a cycle (CC is HKNUZ, CT is HKNVZ, both five),
which is why the letters are kept and the count is not stored twice.  Rows are
indexed on instrument.

`Mean_Auto_Best_V` and `Mean_Forced_Best_V` are measured for EVERY instrument,
off whichever of the three worksheets models it: has_notice markets count down
to first notice, the other deliverable ones to last trade, and the cash-settled
ones to last trade as a settlement date.  The first column scores auto_roll, the
calendar rule; the second scores forced_roll_hold, the four-branch ladder with
the ratchet.  Reading them side by side is the point -- the gap between them is
what a market gains from the ladder, and it is what put ZC on forced_roll (.7715
against .9611) and left GC on auto_roll (.9858, nothing to gain).

EVERY SCORE IN THIS FILE IS SESSION-WEIGHTED, and that is not a formatting
detail.  Auto_Best_V is one fact per session, copied onto every contract row of
that session, so a mean over ROWS counts it once per listed month -- and the
number of months listed is not part of what is being measured.  It would be
harmless if contract counts were stable.  They are not: they grow, so weighting
by them weights by era.  CL lists 12 contracts in the 1980s and 129 today, and
the 1980s are the only decade its calendar rule fails (.6166); row-weighting
gives that decade 3.1% of the weight from 14.6% of the sessions and lifts the
score from .9281 to .9786.  Read row-weighted figures as answering "on a random
(session, contract) pair" -- a question no position is ever taken in.

`has_notice` is true where the vendor carries a first notice date on EVERY
contract of the counted year -- the flag that says a delivery-driven roll can be
scheduled for this market from vendor data alone.  For 2025 it is all-or-nothing
(22 true, 41 false; no instrument is partly covered), so a partial result means
something has changed and is warned about rather than quietly reduced to false.
It is NOT the complement of `is_deliverable`: 19 markets deliver physically and
still carry no notice date, because delivery attaches at termination rather than
before it -- the eight FX, the four NYMEX energies, SB, DX, GAS, EUA and the
three Eurex bonds.  For those, last trade is the binding date.

`Unique_Roll` is true where Roll_Rule is neither auto_roll nor forced_roll --
the RS_/LT_/CS_forced_roll confirmation variants.  Derived from Roll_Rule alone.
Blank, not false, everywhere else, including the three markets with no rule.

`Roll_Rule` names the roll a market has been CLEARED for, not the roll it is
being given, and it is set from CURATED SETS ONLY -- it does not read
Mean_Auto_Best_V.  A market earns its rule by surviving the timeseries audit in
Front_Contract/, and the score sits beside it as information.  auto_roll for the three
AUTO_ROLL sets: the calendar rule already sits on the volume leader often
enough that nothing more elaborate is warranted.  LT_forced_roll for LT_FORCED_ROLL, the last-trade twin of that variant, and
CS_forced_roll for CS_FORCED_ROLL, the cash-settled one.
forced_roll for the curated FORCED_ROLL set, markets whose
forced_roll_hold series is a valid schedule over full history -- no blanks, no
steps backward, no contracts jumped.  RS_forced_roll for RS_FORCED_ROLL, which
reads the confirmation variant instead; see the comment on that set for why it
is cleared on different evidence.  They are tested in that order, and the first
that matches wins.
Anything else is BLANK -- undecided, not "something else" -- because the rule
those markets need has not been settled.
Blank also covers every instrument with no score at all, which is every market
where has_notice is false: no score, nothing to clear.
Derived from Mean_Auto_Best_V and written in the same pass, so --no-scores
leaves both columns off rather than writing one against a stale other.

One contract per listed month is the invariant that says the year filter is
clean; it is checked and warned about rather than carried as a column.

The count is taken from ONE expiry year (default 2025), not from the whole
listing history.  Cycles change: counting every contract ever listed gives ZC
seven months because J and X appear in 1970s listings, when CME corn trades five
today.  A single recent, fully-expired year is the live cycle.

    python Live/contract_cycles.py                     # writes contract_cycles.csv
    python Live/contract_cycles.py --year 2024         # a different year
    python Live/contract_cycles.py --all-sessions      # every vendor session
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import platform
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# STRIP THE MACHINE NAME TO ASCII BEFORE norgatedata IS EVER IMPORTED.
#
# This machine is called "Napoleon" with an acute accent.  norgatehelper.py
# line 44 puts platform.node() straight into an HTTP header, headers must be
# ASCII, and NDU answers 400 with an empty body.  norgatedata's import-time
# probe logs the response BODY -- so the empty body prints as
# "ERROR: Norgate Data: " with nothing after it, once per import.
#
# Harmless, because every call this file makes afterwards goes through
# ensure_ascii-style header stripping.  Confusing, because it looks like a real
# failure at the top of every run.  Isolated and reproducible:
#     no Client header / Client "Napoleon" -> 200 OK
#     Client platform.node()               -> 400, empty body
#
# Must precede the `import norgatedata` inside main() and refresh_metadata --
# the probe runs at import, before any of our header handling.  The same patch
# is at the top of ../Update.py for its own process; platform.node() does not
# read COMPUTERNAME on Windows, so each interpreter needs its own copy.
# Renaming the machine would retire both.
# ---------------------------------------------------------------------------
_NODE = platform.node()
if not _NODE.isascii():
    platform.node = lambda _n=_NODE.encode("ascii", "ignore").decode(): _n

import polars as pl

# Cash-settled markets.  Norgate exposes no settlement-type field -- `subtype1`
# gives only an asset class, and that does not decide it (Energy holds both
# physically-delivered CL and cash-settled Brent).  So this is CURATED from
# exchange contract specs, kept as the cash list because it is the shorter one
# and the exceptions are what need justifying:
#
#   GF, HE      feeder cattle and lean hogs settle to a CME index, unlike LE
#   BRN         ICE Brent settles to the Brent Index; NYMEX WTI does not
#   SJB, YXT4   SGX mini-JGB and ASX bond futures are cash-settled, unlike
#               every other bond in the universe
#
# Checked against the vendor: a first notice date only exists where physical
# delivery does, so nothing carrying one may appear here.  `verify()` enforces
# that and is run on every build.  That covers 23 of the 41 deliverable markets.
#
# Three deliverable markets carry NO notice date and were confirmed against
# exchange documentation on 2026-08-26, because the vendor cannot settle them
# either way and each looks cash-settled at a glance:
#
#   DX    ICE USDX -- physically delivered: the long delivers the six-currency
#         basket through CLS and receives USD.  Last trade is the 2nd business
#         day before the 3rd Wednesday.
#   GAS   ICE Endex Dutch TTF -- physical delivery by transfer of title at the
#         TTF Virtual Trading Point (GTS), spread hourly across the delivery
#         month.  Trading ceases 2 UK business days before it begins.
#   EUA   ICE EUA -- deliverable: 1,000 allowances per lot transferred through
#         the Union Registry.
#
# All three deliver with no notice period, so last trade IS the binding date
# and the roll margin is the only thing standing between a held position and a
# delivery obligation.
CASH_SETTLED = {
    "BRN", "BTC", "ETH", "EMD", "ES", "FDAX9", "FESX9", "GF", "HE", "HSI",
    "LEU9", "LFT9", "NIY", "NQ", "RTY", "SJB", "SO3", "SR3", "SXF", "VX",
    "YAP4", "YXT4",
}

# Delivery months measured as held-but-dead, and the ones that carry the market.
# Curated per instrument, because it is a property of the market that no vendor
# field reports.  ZC measured over 12,253 sessions, 1978-2026: holding F gives a
# median 0.8% of session volume and X gives 0.4%, against 28.8-62.4% for
# H/K/N/U/Z.  F and X were listed around 2000-2002 and barely traded, so a
# purely calendar-driven roll walks into them -- 123 sessions on ZC alone.
# Empty means NOT YET MEASURED, not "none".
# A median-share test cannot judge a STIR: LEU9, SO3 and SR3 trade a deep
# strip of quarterlies simultaneously, so no month holds much of a session's
# volume and EVERY month reads as dead -- which would empty the instrument.
# Where the method is not competent, the months stay ACTIVE.
#   LEU9 IS NOW TRIMMED ANYWAY, on a different test that IS competent here.
#   Median share asks how big a month is; that question has no good answer in a
#   strip.  Ask instead which months are in the CYCLE at all -- has this month
#   ever led the book on volume -- and the strip answers cleanly:
#       H 907   M 1,045   U 866   Z 1,719   sessions as volume leader
#       F 0   G 0   J 0   K 0   N 0   Q 0   V 0   X 0   -- never, not once
#   over 4,537 sessions.  HMUZ carry 98.70% of all LEU9 volume across 95
#   contracts; the eight serials carry 1.30% across 148.  Median share 0.0004
#   -0.0007 against 0.0230-0.0263, a 30-60x gap, and a zero-volume rate of
#   34-41% against 13-14%.  These are ECB serial months, listed and quoted but
#   not traded, and a calendar roll walks straight into them -- which is the
#   whole of why auto_roll scored .0547 here.
#   SO3 and SR3 need no entry: both are listed HMUZ already.
#   This does NOT reopen the median-share method for strips.  It records that
#   a leadership test survives where a magnitude test does not.
#
# MEASURED AND CLEAN, which an empty string cannot say on its own -- absent an
# entry below, "" means NOT YET MEASURED, and the two look identical from the
# CSV.  Anything listed here has been run and came back with nothing dead:
#   RS  (2026-08-27, 11,451 sessions 1980-2026)
#       F 49.7  H 49.3  K 48.5  M 49.9  N 50.1  X 62.9  U 30.5  Q 15.4
#       Weakest is Q at 15.4%, and Q stopped being listed in 2001.  A genuinely
#       dead month reads 0.4-0.8% on this test -- ZC F and X -- so Q is an order
#       of magnitude clear of the line, and U at 30.5% is above ZC's U, which is
#       live.  Nothing to trim.  Recorded because RS scores worst in the book on
#       the roll rules (f_r_h_Best_V .8455) and a polluted candidate set was the
#       first suspect; it is not the cause, and this saves the next person
#       re-running it.  RS's cycle DID change: M was the second-largest month
#       until 1996-06-19 and was replaced by K and N that year, so pre-1996 RS
#       is effectively a different contract.  That is a spec change, not a dead
#       month, and this column is the wrong place to record it.
#
# The share is measured over the sessions a contract is the FRONT month -- the
# one a calendar roll would sit on -- not over every session it is listed.  The
# latter dilutes every month toward zero, since a contract spends most of its
# life years out, and would read the whole book as dead.
DEAD_CONTRACTS = {
    "CT": "V",
    "EUA": "FGHJKMNQUVX",
    "GC": "FHKNUVX",
    "HE": "K",
    "HG": "FGJMQVX",
    "LE": "KNUX",
    "LEU9": "FGJKNQVX",
    "PL": "GHKMQUXZ",
    "SB": "FU",
    "SI": "FGJMQVX",
    "YAP4": "FGJKNQVX",
    "ZC": "FX",
}
ACTIVE_CONTRACTS = {
    "6A": "HMUZ",
    "6B": "HMUZ",
    "6C": "HMUZ",
    "6E": "HMUZ",
    "6J": "HMUZ",
    "6M": "HMUZ",
    "6N": "HMUZ",
    "6S": "HMUZ",
    "BRN": "FGHJKMNQUVXZ",
    "BTC": "FGHJKMNQUVXZ",
    "CC": "HKNUZ",
    "CGB": "HMUZ",
    "CL": "FGHJKMNQUVXZ",
    "CT": "HKNZ",
    "DX": "HMUZ",
    "EMD": "HMUZ",
    "ES": "HMUZ",
    "ETH": "FGHJKMNQUVXZ",
    "EUA": "Z",
    "FDAX9": "HMUZ",
    "FESX9": "HMUZ",
    "FGBL9": "HMUZ",
    "FGBM9": "HMUZ",
    "FGBS9": "HMUZ",
    "GAS": "FGHJKMNQUVXZ",
    "GC": "GJMQZ",
    "GF": "FHJKQUVX",
    "HE": "GJMNQVZ",
    "HG": "HKNUZ",
    "HO": "FGHJKMNQUVXZ",
    "HSI": "FGHJKMNQUVXZ",
    "KC": "HKNUZ",
    "LE": "GJMQVZ",
    "LEU9": "HMUZ",
    "LFT9": "HMUZ",
    "LLG": "HMUZ",
    "NG": "FGHJKMNQUVXZ",
    "NIY": "HMUZ",
    "NQ": "HMUZ",
    "PA": "HMUZ",
    "PL": "FJNV",
    "RB": "FGHJKMNQUVXZ",
    "RS": "FHKMNQUX",
    "RTY": "HMUZ",
    "SB": "HKNV",
    "SI": "HKNUZ",
    "SJB": "HMUZ",
    "SO3": "HMUZ",
    "SR3": "HMUZ",
    "SXF": "HMUZ",
    "UB": "HMUZ",
    "VX": "FGHJKMNQUVXZ",
    "YAP4": "HMUZ",
    "YXT4": "HMUZ",
    "ZB": "HMUZ",
    "ZC": "HKNUZ",
    "ZF": "HMUZ",
    "ZL": "FHKNQUVZ",
    "ZM": "FHKNQUVZ",
    "ZN": "HMUZ",
    "ZS": "FHKNQUX",
    "ZT": "HMUZ",
    "ZW": "HKNUZ",
}

MONTH_ORDER = "FGHJKMNQUVXZ"
ROLL_RULE_MIN = 0.95    # THE BAR THE DECISIONS WERE MADE AGAINST, and no
                        # longer wired into anything: Roll_Rule is now set
                        # from the curated sets alone.  Kept because every
                        # set below cites it, and an instrument recorded as
                        # clearing or missing .95 should have the number it
                        # was measured against in the same file.
                        # Mean_Auto_Best_V at or above this cleared a market for
                        # auto_roll.  The 22 scored instruments fall either side
                        # of it with a wide gap and nothing near the line -- HG
                        # .9689 above, KC .9271 below -- so the split does not
                        # turn on the threshold's exact value.  It DOES turn on
                        # the weighting for one instrument in the other half of
                        # the book: CL reads .9786 row-weighted and .9281
                        # session-weighted.  Session is the measure used here;
                        # see the module docstring for why.
# Curated, like DEAD_CONTRACTS above and for the same reason: it records a
# MEASUREMENT, and the measurement is not a threshold on any column in this
# table.  A market earns forced_roll by its forced_roll_hold series being a
# valid schedule over full history -- no blank session, no step back to a nearer
# expiry, and no jump over a listed contract.  Checked in Front_Contract/, where
# front_contract.py builds the series; re-run it there before adding to this
# set.  It gates on till_notice_cd for these markets.
# CC, KC and LE cleared all three over 11,555 / 11,762 / 11,898 sessions, each
# one unbroken run.  They also score BETTER on the volume leader than auto_roll
# does -- f_r_h_Best_V .9444 / .9807 / .9432 against Auto_Best_V .9080 / .9294
# / .8064 -- so the rule is not being bought at the cost of accuracy here.
# The seven other markets Roll_Rule leaves blank are continuous too, but jump a
# listed contract at the crop-year roll: ZL 51 times, ZM 49, ZS 49, ZC 41, ZW
# 19, RS 16, CT 2.  Continuous is not the same as stepwise, and they stay blank
# until that is settled.
# ZC added on the same three continuity checks plus a full timeseries audit:
# 198 contracts over 12,206 sessions 1978-2026, every one held in a single
# contiguous block, never a contract that was unlisted, past notice, inside the
# notice window, or without volume.  f_r_h_Best_V .9722.  It carries 41 +2 jumps
# at the crop-year roll, accepted deliberately -- the skipped U month is liquid
# but open interest has already moved to Z well before the roll.
# NOTE for anyone building a price series off ZC: the N->Z roll crosses the
# old-crop/new-crop spread, and the gap is large -- median 2.9% across all ZC
# rolls but -28.1% on 1996-05-23, -17.3% on 2013-05-28, -15.5% on 2021-06-21.
# Real, not an error, and it must be back-adjusted or it books as return.
# Session-weighted f_r_h_Best_V over full history: ZS .9814, KC .9720, ZC .9611,
# ZW .9584 clear the bar outright.  All with a clean timeseries: no contract
# ever returned to, no hold on anything unlisted, past notice, or inside the
# notice window, and roll counts matching the effective cycle.  ZS is the
# cleanest series in the book -- minimum hold 7 sessions, not one hold under 5.
# ZL, ZM and ZS trade a shorter cycle than they list, skipping the Q/U/V
# transition months the way ZC skips U.
#
# FIVE MEMBERS DO NOT CLEAR .95 AND ARE HERE ON OTHER EVIDENCE.  Stated plainly
# so the set is not mistaken for a uniform standard:
#   ZL .9479 and ZM .9370 -- cleared on row-weighting at .9653 and .9567 before
#     that measure was corrected.  They keep their place on the RS argument
#     rather than the score: leader-or-not punishes an oscillation between two
#     adjacent liquid contracts, and what they actually hold is the top two by
#     volume on 98.5% and 98.6% of sessions, median 49.6% and 48.2% of session
#     volume, fourth-best or worse on 0.1% and 0.0%.
#   CC .9384 and LE .9368 -- never cleared on accuracy at all.  They were the
#     first two admitted, on CONTINUITY alone: zero blanks, zero steps
#     backward, zero contracts jumped over full history.
#   CT .9419 full history -- the window exemption below, which survives the
#     switch to session weighting: .9546 for 1990+, .9651 for 2000+, .9539 for
#     2010+.
#
# CT IS AN EXCEPTION AND THE WINDOW MATTERS.  It scores .9419 on full history --
# under the bar -- and clears only from 1990 on: .9546 for 1990+, .9651 for
# 2000+, .9539 for 2010+.  (Session-weighted throughout; the row-weighted
# figures this exemption was first written on were .9473/.9558/.9633/.9538, and
# the conclusion is unchanged.)  The whole shortfall is the 1980s, the same thin,
# badly recorded era that gives CT its three zero-volume sessions (1998 aside,
# two of which recorded no volume in ANY contract).  Cleared on the 1990+ window
# deliberately, and every other member of this set clears on full history.  Do
# not quietly extend that exemption to another instrument without saying so.
#
# RS stays out of THIS set: .8244 full history session-weighted.  It has its
# own rule below, on the confirmation variant.
# THE LAST-TRADE HALF.  Same rule, scored off till_last_trade_cd.  Session
# weighted f_r_h_Best_V, full history unless a window is stated:
#   HO .9671 and RB .9590 -- ZERO hard failures of any kind, top two by volume
#     on 99.7% and 97.8% of sessions.  Both are transformed by the forced
#     ladder: auto_roll scores .7532 and .7197 on the same series.
#   GAS .9443 full history, cleared on the 2000+ WINDOW at .9654 (.9753 for
#     2010+, .9717 for 2015+).  Early history is the whole shortfall, as with
#     CT -- see the note on excluded early history below.
# The energy split is not the obvious one: crude and natural gas take auto_roll
# comfortably while the refined products and European gas need this.
# THE CASH-SETTLED HALF, scored off Front_Contract/front_contract.py (last-trade gate).
# Nothing is delivered in these markets, so the gates are liquidity gates rather
# than protection -- see that file's header.  Session-weighted f_r_h_Best_V:
#   ES, NQ, RTY, EMD, SJB all 1.0000; SXF .9994; YXT4 .9962; YAP4 .9927;
#   BRN .9913.  Zero hard failures and 100% top-two by volume throughout,
#   except three single sessions where the vendor published no open interest at
#   all while volume was healthy -- YAP4 1985-05-10, and YAP4 and YXT4 both on
#   2026-07-03, one gap hitting two instruments.  Bad bars, not rule failures.
#   GF .9021 does NOT clear .95 and is here on the RS precedent: top two by
#   volume on 98.3% of sessions, zero hard failures.  Cash-settled feeder
#   cattle leadership oscillates between adjacent liquid months, which
#   leader-or-not scores as total failure and a trader would not notice.
# MOVED OFF auto_roll ONCE Mean_Forced_Best_V EXISTED.  These four were cleared
# on auto_roll and passed every hard check on it -- a working rule, but not the
# best available, which nothing was comparing them against at the time.  The
# ladder beats the calendar rule on all four with the same roll shape:
#   CL .9281 -> .9815, 514 rolls against 515, minimum hold 6 sessions either
#     way, zero skips, top two 99.5% -> 100%, median gap 0.91% -> 0.84% and the
#     28.49% maximum unchanged.  CL had already been re-argued twice -- cleared
#     on pl.mean() .9786, found to be .9281 session-weighted, then kept on the
#     rank distribution.  None of that compared it to forced_roll_hold.
#   6B .9787 -> .9997 with an identical series, 6C .9711 -> .9944, 6J .9756 ->
#     .9946 (minimum hold 52 -> 11 sessions, a fortnight, not churn).
FORCED_ROLL = {"6B", "6C", "6J", "BRN", "CC", "CL", "CT", "EMD", "ES", "GAS",
               "GF", "HO", "KC", "LE", "NQ", "RB", "RTY", "SJB", "SXF",
               "YAP4", "YXT4", "ZC", "ZL", "ZM", "ZS", "ZW"}

# Reads RS_forced_roll_hold rather than forced_roll_hold: the same four branches
# and the same ratchet, but it will not move until the SAME contract has been
# the answer two sessions running, and it abandons that wait once the incumbent
# reaches the notice window.
# RS needs it because the ratchet turns a one-session mistake into a permanent
# one, and canola supplies the mistakes.  On 2025-03-10 a single session of
# volume touched RS-2025N while open interest stayed on K; the hold jumped, and
# coming back was a backward step, so it sat wrong for 21 sessions.  Confirming
# the move cuts the worst such run from 81 sessions to 20 and the count of runs
# over 8 sessions from 59 to 11, taking accuracy .8244 -> .8915.
#
# IT DOES NOT MEET THE .95 BAR AND IS CLEARED ANYWAY, on different evidence.
# f_r_h_Best_V is leader-or-not, and RS oscillates between two adjacent liquid
# contracts, which that test scores as total failure.  What it actually holds
# over 11,451 sessions: rank 1 by volume 88.4%, rank 2 10.7%, rank 3 0.9%, never
# worse.  Median 54.5% of session volume and 50.8% of open interest; at the 10th
# percentile still 38.4% and 1,201 lots.  Two sessions in 11,451 hold under 5%
# of session volume, and none holds zero.  Every one of the 1,243 "misses" is
# the second or third most traded contract, median 1,958 lots.  A trader would
# not notice most of them.
# Integrity is clean: no blanks, no steps backward, nothing held past notice or
# inside the notice window, no hold shorter than 5 sessions.  223 contracts,
# 222 rolls, 4.72 a year, median hold 43 sessions.
# Price gap at roll reaches 29.0%, on a par with ZC -- back-adjust.
# RS's cycle changed in 1996 (M replaced by K and N), which is why the months it
# rolls into are uneven: F 46, H 46, X 46, K 30, N 28, M 16, U 11.
RS_FORCED_ROLL = {"RS"}

# The last-trade half of the book: is_deliverable with has_notice FALSE, scored
# off Front_Contract/front_contract.py, where till_last_trade_cd stands in for
# till_notice_cd throughout.  These markets have no notice period at all
# -- the vendor returns None for first_notice on every contract they list --
# so last trade is the only date between a held position and delivery.
#
# EARLY HISTORY IS EXCLUDED WHERE THE VENDOR DATA IS NOT USABLE, and that is a
# data judgement, not a tuning knob.  The stored Mean_Auto_Best_V is always the
# FULL-history figure; the window below says what the instrument was actually
# cleared on, the same convention CT uses.
#   FGBL9, FGBM9, FGBS9 -- 2005 EXCLUDED, cleared on 2006+.  Their 17, 35 and 45
#     zero-open-interest holds fall entirely in 2005, on sessions carrying a
#     single contract with no open interest published at all.  inception() fired
#     correctly -- FGBS9-2005Z printed 3,000 lots -- so the market was trading;
#     the vendor simply had not started reporting the other field.  From 2006 all
#     three score a perfect 1.0000 with zero failures of any kind across 5,252
#     sessions.  205 sessions of unusable history were the entire problem, and no
#     other rule fixes it: forced_roll and the confirmation variant score LOWER
#     on these three and FGBS9 gets worse, 100 and 86 failures against 45.
#   GAS -- pre-2000 excluded, see FORCED_ROLL.
#
# Eleven instruments that came through the timeseries audit with nothing at all:
# no blank session, no hold on a contract that was unlisted, past last trade,
# inside the last-trade window, or missing a close, no step backward, no
# contract returned to, no jump over a listed contract.  Session-weighted
# Auto_Best_V:
#   6N 1.0000  6E .9954  6A .9882  6M .9870  6B .9787  6S .9766  DX .9766
#   NG .9757   6J .9756  6C .9711  CL .9281
# Ten of them now reach the ROLL_RULE_MIN threshold on their own, auto_best_v()
# having been extended to score this sheet too.  The set is kept because it
# records WHICH instruments were audited, and because CL no longer clears the
# threshold -- see below.
# The eight FX and the currency-like DX roll one contract at a time on a fixed
# quarterly cadence: 100% of sessions in the top two by volume, median hold 63
# sessions, worst price gap under 2.5%.
#
# NG IS IN THE SET BUT IS NOT THAT ANIMAL.  Monthly contracts on a deep curve:
# median share of session volume 40.6% against 97-100% for the rest, worst price
# gap 31.6% against under 2.5%, and holds down to 1 session against 40+.  It
# passes every hard check and the ladder does not beat it (.9757 auto against
# .9694 forced), so it stays.  CL was its twin here and has moved to
# forced_roll; see that set.
#
# HELD BACK despite scoring above .95, each for a stated reason:
#   FGBL9 .9969, FGBM9 .9978, FGBS9 .9967 -- 17, 35 and 45 sessions holding a
#     contract with zero open interest, every one of them in 2005 on sessions
#     carrying a single contract.  inception() fired correctly (FGBS9-2005Z
#     printed 3,000 lots) but the vendor published no open interest for those
#     first weeks.  A data-coverage gap at the start of the history, not a rule
#     defect, and cheap to clear by trimming the affected span.
#   EUA .9918 -- 3 sessions, one each December 2023, 2024 and 2025, holding
#     EUA-YYYYZ at zero volume AND zero open interest while the rest of the
#     curve carried 319k-455k lots.  EUA lists December only, so the front
#     contract dies before its scheduled last trade every single year, and
#     auto_roll reads the calendar alone and cannot see it.  This one is a real
#     limitation of the rule rather than a data artifact -- a volume- or
#     interest-aware rule would catch it.
#
# NOT SCORED HIGH ENOUGH TO CONSIDER: SB .8841, HO .7760, RB .7109, GAS .4892.
# The split within energy is not the obvious one -- crude and natural gas clear
# comfortably while the refined products and European gas do not.
# has_notice markets the calendar rule handles on its own.  Session-weighted
# Auto_Best_V, every one audited clean over full history -- no blanks, no steps
# backward, nothing held past notice or inside the notice window:
#   UB 1.0000  ZN 1.0000  LLG .9999  CGB .9997  ZF .9997  ZT .9985  PL .9925
#   ZB .9899   SI .9877   GC .9858   PA .9792   HG .9689
HN_AUTO_ROLL = {"CGB", "GC", "HG", "LLG", "PA", "PL",
                "SI", "UB", "ZB", "ZF", "ZN", "ZT"}

# Reads LT_forced_roll_hold: forced_roll_hold's four branches and ratchet, but
# it will not move until the same contract has been the answer
# ROLL_CONFIRM_SESSIONS running, and it abandons that wait once the incumbent
# reaches the last-trade window.  The last-trade twin of RS_FORCED_ROLL, and it
# earns its place the same way -- by beating every alternative on markets whose
# leadership oscillates between adjacent liquid months.
#   SB .9555, against .9452 for plain forced_roll and .8829 for auto_roll.  It
#     also has the better shape: minimum hold 7 sessions against 3, one skip
#     against three, and no hold under 5 sessions at all.
#   EUA .9377, and this one is chosen for INTEGRITY rather than accuracy.
#     auto_roll scores .9916 but holds an untradeable contract on 3 sessions --
#     one each December 2023, 2024 and 2025, holding EUA-YYYYZ at zero volume
#     AND zero open interest while the rest of the curve carried 319k-455k lots.
#     EUA lists December only, so its front contract dies before its scheduled
#     last trade every year and auto_roll, reading the calendar alone, cannot
#     see it.  Plain forced_roll collapses to .7000; this holds nothing
#     unlisted, past last trade, inside the window, or without a close, on any
#     of 5,134 sessions, and sits in the top two by volume on 99.2% of them.
#     5.4 points of accuracy for a series with no untradeable holds is the
#     trade being made, and it is a deliberate one.
#   6S .9913, and it is here rather than on forced_roll for SHAPE, not score.
#     Plain forced_roll gets .9931 -- eighteen ten-thousandths more -- by
#     dropping the minimum hold from 52 sessions to ONE and taking a short hold
#     with it.  A one-session position is two round trips of spread for a
#     rounding difference.  This keeps the 52-session minimum and no short
#     holds.  auto_roll, which it came off, scored .9766.
LT_FORCED_ROLL = {"6S", "EUA", "SB"}

# Cash-settled markets the calendar rule already handles, session-weighted
# Auto_Best_V, zero hard failures and 100% top-two by volume for all seven:
#   BTC 1.0000  ETH 1.0000  HSI 1.0000  LFT9 1.0000  NIY 1.0000
#   FDAX9 .9983  FESX9 .9978
# A curated set for the same reason as LT_AUTO_ROLL: auto_best_v() does not
# score this sheet, so Mean_Auto_Best_V is null for all 22 cash-settled markets
# and none of them can reach the ROLL_RULE_MIN threshold.  Extending it to a
# third sheet is a live option; the reason it has not been done is that a
# leader-based score is MEANINGLESS for the three STIRs below, and a column that
# is right for 19 rows and nonsense for 3 is worse than a column that is empty.
CS_AUTO_ROLL = {"BTC", "ETH", "FDAX9", "FESX9", "HSI", "LFT9", "NIY"}

# Reads CS_forced_roll_hold, the cash-settled twin of RS_ and LT_FORCED_ROLL:
# forced_roll_hold's four branches and ratchet, but it will not move until the
# same contract has been the answer ROLL_CONFIRM_SESSIONS running.
#   HE .9360, top two by volume on 99.8% of sessions, zero hard failures.
#   VX .9030, top two on 97.7%, zero hard failures.
# Neither clears .95 and both are here on the RS precedent.  VX is worth a note:
# a volatility future has its own term structure and the front month is not
# where the position naturally sits, so a front-contract leader score is a poor
# fit for it even by the standards of this exercise.  Cleared on the rank
# distribution, not on the score.
CS_FORCED_ROLL = {"HE", "VX"}

# The one STIR the dead-month trim rescued: LEU9.  Cleared on the TIMESERIES
# audit and NOT on the score -- Mean_Auto_Best_V is .2872, and a leader score
# means nothing in a strip, for the reasons in the STIR note below.
# Over 4,537 sessions, HMUZ only, auto_roll against forced_roll:
#     held nothing / held an expired month     0 / 0    vs   0 / 0
#     STALE -- held a zero-volume month              0  vs        0
#     backward rolls                                 0  vs        0
#     median lots in the held month             80,830  vs   97,117
#     median volume share                        0.108  vs    0.124
#     maturity IQR                               46 d   vs    118 d
# Both are tradeable here, so this is decided on EXPOSURE.  46 days is the
# irreducible sawtooth of holding one quarterly as it ages; forced_roll wanders
# 118 and the series stops tracking a single point on the curve.  auto_roll
# also has the worse worst-case roll gap (1.140 vs 0.625) on one roll; the
# medians are 0.040 and 0.050 and that outlier did not decide it.
# WHY THIS WORKS FOR EURIBOR AND NOT FOR SOFR/SONIA: LEU9 settles to a
# forward-looking fixing ON its expiry date, so its front month stays uncertain,
# and stays traded, to the end.  SO3 and SR3 settle to daily rates COMPOUNDED IN
# ARREARS over the contract's own reference quarter, so by the time one is the
# front month its price is largely already determined and the volume has left.
# That is a fact about the contracts, not about the rule.
# Only reachable because DEAD_CONTRACTS now trims LEU9 to HMUZ: on the full
# 12-month listing auto_roll scored .0547, because a calendar roll walks into
# ECB serial months that have never once led the book.
STRIP_AUTO_ROLL = {"LEU9"}

# Reads +1_auto_roll_hold, a series that did not exist when the note below was
# written: auto_roll's decision taken one contract further out.  Same calendar
# ordering, same AUTO_ROLL_CD window, same blindness to volume -- the front
# month is simply never held.
#   SO3, 1,418 sessions from 2021-01-12, against auto_roll on the same window:
#       median lots held       6,046  ->  45,198
#       sessions under 1,000     24%  ->      0%
#       stale (zero volume)     1.1%  ->    0.1%
#       volume share            .017  ->    .113
#       rolls / backward       22 / 0 ->  22 / 0     unchanged
#       maturity IQR              46  ->      46     unchanged
#   From 2022 the WORST session of every year is 6,007 lots and there is not one
#   thin or stale session in 1,000+.  The four exceptions are Jan-Jun 2021, all
#   546-949 lots, and all more than four months from expiry -- they are the
#   young market, not the rule.  The single zero is 2026-08-24, the last day of
#   the panel.
#   BOTH LEGS of all 22 rolls are executable, which is the test that matters
#   since the outgoing is sold and the incoming bought on one date: worst leg
#   3,627 lots in 2021, and from 2022 the worst is 23,625.
#   ROLLING EARLIER BUYS NOTHING, and this was checked rather than assumed:
#   sweeping the trigger over 5/15/30/45 leaves the thin rate flat at 0.3-0.4%
#   and makes the worst roll leg WORSE (3,627 -> 1,486 at 30).  The held month
#   never comes within 90 days of expiry against a death zone starting near 40,
#   so there is nothing for an early roll to fix.
#   Unique_Roll is true here, correctly: this is not auto_roll or forced_roll.
#   SR3 measures the same way and slightly better.  1,411 sessions from
#   2021-01-12: median lots 36,806 -> 296,905, sessions under 1,000 9% -> 0%,
#   stale .8% -> 0%, share .017 -> .113, rolls 22/0 unchanged, IQR 46 unchanged.
#   From 2022 the worst session of any year is 21,839 lots.  Worst roll leg
#   2,497 in 2021, and 166,597 from 2022.
#   TIMESERIES AUDIT, both instruments, 2021-01-12 on, Panama on DIFFERENCES:
#   continuous, no gaps.  Daily change median 1.0bp, p95 6.5-7.0bp.  Flat days
#   fall from 45.7% to 19.4% (SO3) and 35.8% to 15.7% (SR3), and the flat days
#   that remain are real: median volume 25,389 and 53,694 lots, none at zero.
#   They cluster where the RATE was pinned -- 48% and 44% in 2021 at 0.1% Bank
#   Rate, 8-11% through the hiking cycle -- so they are an unchanged close, not
#   a dead contract.  Roll gaps are 22bp against a 1bp daily move, but that is
#   the forward slope and the adjustment absorbs it: on a roll date the series
#   return reduces exactly to old_close(t) - old_close(t-1).  Both series track
#   real events -- SO3's largest moves are the 2022-09-23/26 mini-budget and
#   gilt crisis, SR3's is 2023-03-13 +86.5bp on 2.79M lots, the Monday after
#   SVB.  Total back-adjustment is -2.565 (SO3) and -0.050 (SR3); neither drifts.
STIR_PLUS1 = {"SO3", "SR3"}

# WHY NEITHER auto_roll NOR forced_roll RULES THE STIRs.  Kept because it is
# the argument the set above rests on, and because it is the reasoning that
# would otherwise be rediscovered.  Written when SO3 and SR3 were unruled.
# auto_roll scores .0227 and .0206 and
# lands in the top two contracts by volume on 3.5% and 3.1% of sessions; the
# forced ladder lifts them only to .3764 and .3287.  The DEAD_CONTRACTS note above
# already explains why, for the dead-month test, and it applies here in full: a
# STIR trades a deep strip of quarterlies at once, so no month holds much of a
# session's volume and there is no front contract in the sense every rule in
# these three sheets assumes.
# NOTE WHAT DOES NOT RESCUE THEM.  The rank-distribution argument that carried
# RS, GF, HE and VX fails here too -- top-two is 45-58% even on the best rule,
# against 97%+ for every instrument cleared that way.  So this is not a metric
# quibble that a better score would settle: the CONCEPT does not apply.  Fitting
# any of these rules to a STIR would produce a number, and the number would be
# untrustworthy.  Left null until someone models a strip properly.
# THAT IS WHAT +1_auto_roll_hold DID, and the set above is the result.
# WHAT THE TIMESERIES AUDIT ADDS, for whoever picks this up.  Between the two
# rules forced_roll is not close: auto_roll holds a ZERO-VOLUME month on 207 of
# 2,030 SO3 sessions (10.2%) and 142 of 2,087 SR3 sessions (6.8%), which are not
# merely illiquid prints but stale ones -- a fake zero return, then a fake jump
# when the contract next trades.  forced_roll cuts that to 9 and 12, and lifts
# the median held month from 1,513 to 30,807 lots (SO3) and 11,342 to 158,143
# (SR3).  So forced_roll is the only TRADEABLE of the two.  It is still not
# written here, because it carries a maturity IQR of 238 and 168 days against
# auto_roll's 46: the exposure wanders one to two and a half quarterly steps, so
# the series does not track a single point on the curve.  A pinned-offset rule
# on the HMUZ list holds 73-77% of the achievable volume share with a 46-day
# IQR and no backward rolls, and is the shape of the answer.  It was thought to
# need a new rule class; it did not.  Holding one contract further out is
# auto_roll's own logic with the index shifted, which is all +1_auto_roll_hold is.
# Only position 2 and beyond would need the rule to step twice.

LT_AUTO_ROLL = {"6A", "6E", "6M", "6N", "DX", "FGBL9", "FGBM9", "FGBS9", "NG"}

SYM = re.compile(r"^(?P<root>.+)-(?P<year>\d{4})(?P<code>[FGHJKMNQUVXZ])$")
MAPPING_NAME = "instrument_mapping.csv"


def find_mapping() -> Path:
    """The universe CSV, searched for upward from this file.

    Live/ owns this file.  The private repo has its own copy, kept as prior art,
    and the two are KNOWN to disagree: that copy's GAS row describes Dutch TTF
    (EUR, ICE Endex, pointsize 730) while norgate's GAS code returns Low Sulphur
    Gasoil (USD/tonne, tick 0.25).  Reading the private copy would silently
    restore that error, so the default points here and nowhere else.

    Searched for rather than reached by a counted `parents[N]`: the count was
    silently wrong the moment this script moved into Live/1_Roll/, and produced a
    default naming a file that does not exist rather than an error anyone reads.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / MAPPING_NAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{MAPPING_NAME} not found in any parent of {here} -- pass --mapping")


def wait_for_ndu(nd, tries: int = 4) -> bool:
    """`nd.status()` is the call worth retrying -- the data calls are not.

    Read from norgatedata 1.0.77: checkstatus() asks ONCE, with 1-second connect
    and read timeouts, and swallows any failure into a bare False.  Every data
    call goes through get_api_data with maxretries=10 and 5/15-second timeouts.
    So a False from status() is far weaker evidence than a failed data call, and
    is exactly the transient the README records -- twice failing, then fine a
    minute later.  Retry it here; do not retry the data calls after it.
    """
    for attempt in range(tries):
        if nd.status():
            return True
        if attempt < tries - 1:
            time.sleep(2 * (attempt + 1))
    return False


def first_notice(nd, symbol: str) -> str | None:
    """The vendor's notice date for one contract, or None if it has none.

    None is authoritative, not ambiguous: norgatedata returns None for an empty
    response, for its 9999-12-31 sentinel, for an empty string and for any date
    it cannot parse, and raises ValueError for a symbol it does not know.  There
    is no "not found" string here to be mistaken for a date, so a falsy result
    really does mean this contract has no notice period.

    No retry loop: get_api_data already makes 10 attempts.  What it does not do
    is fail loudly -- once those are spent it calls a bare `sys.exit()`, which
    raises SystemExit.  SystemExit does not inherit from Exception, so no
    `except Exception` anywhere in this file can see it, and its exit code is
    None, which the shell reports as 0.  Left alone, an NDU dropout mid-build
    ends the run looking like a success that merely wrote nothing.
    """
    try:
        return nd.first_notice_date(symbol, datetimeformat="iso") or None
    except SystemExit as e:
        raise RuntimeError(
            f"norgatedata called sys.exit({e.code!r}) while fetching {symbol}: "
            f"10 attempts failed, so NDU stopped answering mid-build"
        ) from e


def build(instruments: list[str], year: int) -> tuple[pl.DataFrame, dict]:
    import norgatedata as nd

    rows = []
    notices: dict[str, dict[str, str | None]] = {}
    # Silent until it warned about something, which meant a clean run showed
    # nothing at all while it made 63 vendor round-trips.  One line per
    # instrument, on the same reasoning as rule_scores below.
    t0 = time.time()
    print(f"\nSTAGE 2  cycle table -- reading {len(instruments)} listings "
          f"from the vendor", flush=True)
    for n, inst in enumerate(instruments, 1):
        print(f"  [{n:>3}/{len(instruments)}] {inst:<8}"
              f"{time.time() - t0:>6.0f}s", flush=True)
        try:
            contracts = nd.futures_market_session_contracts(inst)
        except Exception:
            continue
        codes = set()
        n_contracts = 0
        for c in contracts:
            m = SYM.match(str(c))
            if not m:
                continue
            if int(m.group("year")) != year:
                continue
            codes.add(m.group("code"))
            n_contracts += 1
        if not codes:
            continue
        present = "".join(k for k in MONTH_ORDER if k in codes)
        if n_contracts != len(present):
            print(f"  [warn] {inst}: {n_contracts} contracts for "
                  f"{len(present)} distinct months in {year}")
        notices[inst] = {c: first_notice(nd, f"{inst}-{year}{c}") for c in present}
        n_notice = sum(v is not None for v in notices[inst].values())
        if 0 < n_notice < len(present):
            print(f"  [warn] {inst}: notice dates on {n_notice} of {len(present)} "
                  f"{year} contracts -- has_notice reduces that to false")
        row = {"instrument": inst, "codes": present, "per_year": len(present),
               "Dead_contracts": DEAD_CONTRACTS.get(inst, ""),
               "Active_contracts": ACTIVE_CONTRACTS.get(inst, ""),
               "is_deliverable": inst not in CASH_SETTLED,
                 "has_notice": n_notice == len(present)}
        rows.append(row)

    # Dead_contracts and Active_contracts were populated in `row` above but
    # missing from this list, so every run silently dropped them from the CSV.
    # Mean_Auto_Best_V is appended later, in main -- it needs the file on disk.
    cols = ["instrument", "codes", "Dead_contracts", "Active_contracts",
            "per_year", "is_deliverable", "has_notice"]
    return pl.DataFrame(rows).select(cols).sort("instrument"), notices


# ----------------------------------------------------------------------------
# STAGE 1: PANEL REFRESH
#
# contract_cycles.py runs FIRST in the pipeline, so it is what brings the raw
# bars up to date before front_contract.py decides anything from them.  Nothing
# below is carried over from production/ -- that tree stays where it is.
#
# APPEND-ONLY, AND THAT IS A RULE NOT AN OPTIMISATION.  A row already on disk is
# never rewritten.  Where the vendor disagrees with a bar we already hold, the
# disagreement is REPORTED and the file is left alone: a panel that silently
# restates its own history cannot be audited, and every published number rests
# on it.  Acting on a conflict is a deliberate decision, taken having read it.
#
# `PaddingType.NONE` throughout, per Live/README.md.  It CAN omit real bars --
# SR3-2024H lost 2024-05-16 and 05-17 that way -- but a sweep of all 63
# instruments, 15,231 contracts and 7,140,933 rows against ALLMARKETDAYS found
# that to be the only case in the whole panel.  The alternative misreports which
# sessions actually traded, which corrupts every liquidity rule downstream.
# Keep NONE, and re-run the sweep periodically instead.
# ----------------------------------------------------------------------------

_F32_SIG = 7
GAPS = "gaps"
BAR_COLS = ["Open", "High", "Low", "Close", "Volume", "Open Interest"]


def _widen(a, sig: int = _F32_SIG):
    """float32 -> float64 at the vendor's TRUE precision.

    The vendor returns float32.  Widening straight to float64 carries float32
    noise into every downstream number -- 484.75 arrives as 484.7500061... --
    so round to 7 significant figures to recover the decimals actually meant.
    """
    import numpy as np
    x = np.asarray(a, dtype=np.float64)
    out = x.copy()
    m = np.isfinite(x) & (x != 0.0)
    if m.any():
        mag = np.floor(np.log10(np.abs(x[m])))
        f = 10.0 ** (sig - 1 - mag)
        out[m] = np.round(x[m] * f) / f
    return out


def fetch_bars(nd, symbol: str, tries: int = 3):
    """One contract's OHLCV and open interest, or None if the vendor has none.

    Retries because NDU is intermittent: it failed twice on 2026-08-26 and
    succeeded on the identical command a minute later.  A bare SystemExit is
    re-raised as a RuntimeError -- norgatedata calls sys.exit() once its own 10
    attempts are spent, SystemExit is invisible to `except Exception`, and its
    code is None, which the shell reports as 0.  Left alone an NDU dropout ends
    the run looking like a success that merely wrote nothing.
    """
    import numpy as np
    rec = None
    for i in range(tries):
        try:
            rec = nd.price_timeseries(
                symbol, timeseriesformat="numpy-recarray",
                datetimeformat="datetime64ns",
                padding_setting=nd.PaddingType.NONE)
            break
        except SystemExit as e:
            raise RuntimeError(
                f"norgatedata called sys.exit({e.code!r}) fetching {symbol}: "
                f"10 attempts failed, so NDU stopped answering mid-run") from e
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2.0)
    if rec is None or len(rec) == 0:
        return None
    d = np.asarray(rec["Date"]).astype("datetime64[D]")
    ymd = [s.replace("-", "") for s in np.datetime_as_string(d, unit="D")]
    cols = {c: _widen(rec[c]) for c in BAR_COLS if c in rec.dtype.names}
    return ymd, cols


def append_bars(path: Path, fetched, *, dry: bool = False) -> dict:
    """Append the vendor rows dated AFTER what the file already holds.

    Returns counts, plus any CONFLICT: a date present on both sides whose values
    disagree.  Conflicts are reported and NOT applied -- see the stage note.
    """
    ymd, cols = fetched
    rep = {"added": 0, "gaps": 0, "conflicts": [], "created": False}
    cur = None
    have = {}
    if path.exists():
        cur = pl.read_csv(path, infer_schema_length=0)
        have = {r["Date"]: r for r in cur.iter_rows(named=True)}
    else:
        rep["created"] = True

    # ANY vendor date the file lacks, wherever it falls -- not merely dates
    # after the last one on disk.  The first version tested `d > last`, which
    # can extend the tail but cannot fill a HOLE, and a hole is exactly what
    # this has to repair: SR3-2024H was missing 2024-05-16 and 05-17 in the
    # middle of its history.  Deleting session 2026-08-24 from the whole panel
    # exposed it -- YAP4 and YXT4 trade a session ahead and already held
    # 2026-08-25, so their last date was still newer than the gap and 10 bars
    # were never restored.  A tail-only append leaves such a hole forever.
    last = max(have) if have else ""
    new_rows = []
    for i, d in enumerate(ymd):
        if d in have:
            for c in BAR_COLS:
                if c not in cols:
                    continue
                prev = have[d].get(c)
                if prev in (None, ""):
                    continue
                if abs(float(prev) - float(cols[c][i])) > 1e-6:
                    rep["conflicts"].append(
                        (d, c, float(prev), float(cols[c][i])))
        else:
            new_rows.append({"Date": d,
                             **{c: float(cols[c][i]) for c in BAR_COLS
                                if c in cols}})
            if d < last:
                rep["gaps"] += 1
    rep["added"] = len(new_rows)
    if new_rows and not dry:
        add = pl.DataFrame(new_rows)
        if cur is not None:
            cur = cur.with_columns([pl.col(c).cast(pl.Float64, strict=False)
                                    for c in cur.columns if c != "Date"])
            out = pl.concat([cur, add.select(cur.columns)], how="vertical")
        else:
            out = add
        out.sort("Date").write_csv(path)
    return rep


def refresh_panel(nd, fc, instruments: list[str], *, full: bool = False,
                  dry: bool = False) -> dict:
    """Bring every contract file up to date.  Stage 1 of the pipeline.

    A contract whose file already reaches its SCHEDULED last trading day is
    skipped: an expired month cannot gain a bar, and re-reading 15,231 of them
    every night is pure cost.  The test is SELF-CORRECTING -- a file short of
    its last trade keeps being fetched until it is not -- so a session missed
    during an outage repairs itself on the next run rather than needing a full
    rebuild.  `--refresh-full` overrides it.

    Last trade is read from the metadata already on disk rather than asked of
    the vendor per contract, which would be 15,231 extra calls to answer a
    question the file beside it already answers.
    """
    meta = {}
    if fc.NOTICE.exists():
        t = pl.read_csv(fc.NOTICE, infer_schema_length=0)
        meta = {r["symbol"]: (r["last_trade"] or "")
                for r in t.iter_rows(named=True)}

    tot = {"fetched": 0, "skipped": 0, "added": 0, "gaps": 0, "created": 0,
           "empty": 0, "conflicts": 0, "failed": 0}
    detail = {}
    hdr = (f"{'inst':<8}{'listed':>8}{'skip':>7}{'fetch':>7}{'new bars':>10}"
           f"{'new files':>11}{'conflict':>10}{'sec':>7}")
    print(f"\nSTAGE 1  panel refresh -> {fc.CONTRACTS}"
          f"{'   [DRY RUN]' if dry else ''}\n\n{hdr}\n" + "-" * len(hdr))

    for inst in instruments:
        t0 = time.time()
        d = fc.CONTRACTS / inst
        d.mkdir(parents=True, exist_ok=True)
        try:
            syms = [str(c) for c in nd.futures_market_session_contracts(inst)]
        except Exception as exc:
            print(f"{inst:<8}  LISTING FAILED  {type(exc).__name__}: {exc}")
            tot["failed"] += 1
            continue
        r = {k: 0 for k in ("skipped", "fetched", "added", "gaps",
                            "created", "empty", "conflicts")}
        for sym in sorted(syms):
            path = d / f"{sym}.csv"
            lt = meta.get(sym) or ""
            if not full and path.exists() and lt:
                cur = pl.read_csv(path, infer_schema_length=0)
                if cur.height:
                    last = cur.get_column("Date").to_list()[-1]
                    if last >= lt.replace("-", ""):
                        r["skipped"] += 1
                        continue
            try:
                got = fetch_bars(nd, sym)
            except Exception as exc:
                print(f"{inst:<8}  {sym} FETCH FAILED "
                      f"{type(exc).__name__}: {exc}")
                tot["failed"] += 1
                continue
            r["fetched"] += 1
            if got is None:
                r["empty"] += 1
                continue
            rep = append_bars(path, got, dry=dry)
            r["added"] += rep["added"]
            r["gaps"] += rep["gaps"]
            r["created"] += int(rep["created"])
            r["conflicts"] += len(rep["conflicts"])
            if rep["conflicts"]:
                detail.setdefault(inst, []).extend(
                    (sym,) + tuple(c) for c in rep["conflicts"][:3])
        for k in r:
            tot[k] += r[k]
        flag = "  <<<" if r["conflicts"] else ""
        print(f"{inst:<8}{len(syms):>8}{r['skipped']:>7}{r['fetched']:>7}"
              f"{r['added']:>10}{r['created']:>11}{r['conflicts']:>10}"
              f"{time.time() - t0:>7.0f}{flag}")
        sys.stdout.flush()

    print("-" * len(hdr))
    print("")
    print(f"{tot['added']:,} new bars, {tot['created']} new contract files, "
          f"{tot['fetched']:,} fetched, {tot['skipped']:,} skipped as settled")
    if tot["gaps"]:
        n_gap = tot[GAPS]
        print(f"  {n_gap:,} of those filled a HOLE -- a session the panel "
              f"was missing BEHIND its own last bar")
    if tot["empty"]:
        print(f"  {tot['empty']} contracts the vendor returned nothing for")
    if tot["failed"]:
        print(f"  {tot['failed']} failure(s)")
    if tot["conflicts"]:
        print(f"\n  {tot['conflicts']} CONFLICT(S) -- the vendor disagrees with "
              f"bars already on disk.  NOT applied.  Sample:")
        for inst, rows in list(detail.items())[:5]:
            for sym, dt, col, old, new in rows[:3]:
                print(f"     {inst:<6}{sym:<12}{dt}  {col:<14}"
                      f"disk {old:>14,.4f}   vendor {new:>14,.4f}")
    return tot


def refresh_metadata(nd, fc, instruments: list[str], *, dry: bool = False,
                     full: bool = False) -> dict:
    """Update Contract-Metadata/contracts.csv: first notice and last trade.

    MERGES, NEVER REPLACES.  Rows for instruments outside this run are carried
    forward untouched.  The first version rebuilt the file from whatever
    `instruments` held, so a single-instrument run silently deleted the other
    62 -- 14,975 rows, restored from git.  A subset run must be a subset write.

    A contract is only re-queried when its dates can still change:

      * no last_trade on record          -- nothing to decide immutability on
      * last_trade today or later        -- still listed, schedule can move
      * absent from the metadata         -- newly listed
      * --refresh-full                   -- picks up a vendor restatement

    Everything else is carried forward unread.  Once a contract has stopped
    trading its two dates are settled history, and asking 12,000 expired months
    the same question every night is the kind of cost that makes a pipeline too
    slow to run.

    WHAT IS NOT SKIPPED: a live contract with both dates already populated is
    still re-queried, because BOTH FIELDS ARRIVE LATE.  The exchange lists a
    delivery month and schedules it weeks afterwards -- 17 contracts listed
    between 2026-06-18 and 2026-07-29 carried neither date while their
    neighbours carried both, and GC-2028Z, listed December 2022, has had its
    dates all along.  That is a lag, not a gap, and a refresh that skipped
    populated rows would never pick the schedule up when it finally lands.

    `last_quoted_date` is forward-looking for a live contract: ZC-2026Z returns
    2026-12-14, its scheduled final session, not the last date it has traded.
    """
    have: dict[str, dict] = {}
    if fc.NOTICE.exists():
        t = pl.read_csv(fc.NOTICE, infer_schema_length=0)
        have = {r["symbol"]: r for r in t.iter_rows(named=True)}

    today = _dt.date.today().isoformat()
    in_run = set(instruments)
    # Carry forward every instrument this run does not cover.
    rows = [r for r in have.values() if r.get("instrument") not in in_run]
    carried_other = len(rows)

    queried = carried = added = changed = 0
    print(f"\nSTAGE 1b  contract metadata -> {fc.NOTICE}"
          f"{'   [DRY RUN]' if dry else ''}")
    for inst in instruments:
        try:
            syms = [str(c) for c in nd.futures_market_session_contracts(inst)]
        except Exception as exc:
            print(f"  [warn] {inst}: listing failed, its rows kept as they were"
                  f" -- {type(exc).__name__}: {exc}")
            rows.extend(r for r in have.values() if r.get("instrument") == inst)
            continue
        for sym in sorted(syms):
            m = SYM.match(sym)
            if not m:
                continue
            old = have.get(sym)
            lt_old = (old or {}).get("last_trade") or ""
            if old is not None and not full and lt_old and lt_old < today:
                rows.append(old)          # settled: dates are history
                carried += 1
                continue

            fn = first_notice(nd, sym) or ""
            try:
                lt = nd.last_quoted_date(sym, datetimeformat="iso") or ""
            except SystemExit as e:
                raise RuntimeError(
                    f"norgatedata called sys.exit({e.code!r}) on {sym}") from e
            queried += 1
            fn, lt = (str(fn)[:10] if fn else ""), (str(lt)[:10] if lt else "")
            if old is None:
                added += 1
            elif (old.get("first_notice") or "") != fn \
                    or (old.get("last_trade") or "") != lt:
                changed += 1
                print(f"     {sym:<14}{'first_notice':<14}"
                      f"{(old.get('first_notice') or '-'):>12} -> {fn or '-'}"
                      f"   {'last_trade':<11}"
                      f"{(old.get('last_trade') or '-'):>12} -> {lt or '-'}")
            rows.append({"symbol": sym, "instrument": inst,
                         "year": int(m.group("year")),
                         "month_code": m.group("code"),
                         "month": MONTH_ORDER.index(m.group("code")) + 1,
                         "first_notice": fn, "last_trade": lt})

    print(f"  {len(rows):,} contracts on file   {queried:,} queried   "
          f"{carried:,} settled and carried   {carried_other:,} outside this run")
    print(f"  {added} newly listed   {changed} with a changed date")
    if not dry and rows:
        (pl.DataFrame(rows)
           .select(["symbol", "instrument", "year", "month_code", "month",
                    "first_notice", "last_trade"])
           .with_columns(pl.col("year").cast(pl.Int64),
                         pl.col("month").cast(pl.Int64))
           .sort("symbol")
           .write_csv(fc.NOTICE))
    return {"rows": len(rows), "queried": queried, "carried": carried,
            "added": added, "changed": changed}

def _roll_rules() -> dict[str, str]:
    """One instrument -> one rule, built from the curated sets.

    Asserts on overlap.  With eight sets feeding five rule names, four of them
    meaning auto_roll, a market landing in two sets would otherwise be resolved
    by whichever branch happened to be written first -- which is exactly the
    kind of silent precedence this replaced.
    """
    out: dict[str, str] = {}
    for names, rule in ((HN_AUTO_ROLL, "auto_roll"),
                        (LT_AUTO_ROLL, "auto_roll"),
                        (CS_AUTO_ROLL, "auto_roll"),
                        (STRIP_AUTO_ROLL, "auto_roll"),
                        (FORCED_ROLL, "forced_roll"),
                        (STIR_PLUS1, "+1_auto_roll"),
                        (RS_FORCED_ROLL, "RS_forced_roll"),
                        (LT_FORCED_ROLL, "LT_forced_roll"),
                        (CS_FORCED_ROLL, "CS_forced_roll")):
        for i in names:
            assert i not in out, f"{i} is in two rule sets: {out[i]} and {rule}"
            out[i] = rule
    return out


# One sheet since 2026-08-27.  It replaced has_notice_/not_has_notice_/
# not_deliverable_front_contract.py, which were ~85% identical code and had no
# behavioural divergence between them -- see Front_Contract/Roll_Journal.md section 1.
# `front_contract.gate()` picks first_notice or last_trade per instrument, so
# the three-way dispatch this file used to carry is gone.
FC = Path(__file__).resolve().parent / "Front_Contract" / "front_contract.py"
# Stage 2's module, loaded here for ONE thing: its worksheet cache.
#
# THE DIRECTION LOOKS WRONG AND IS DELIBERATE.  Stage 1 reaching into stage 2 is
# not a dependency on stage 2's work -- it is a dependency on the cache both
# stages share, which happens to live there because that is where it was first
# needed.  There is no import cycle (trading_book imports front_contract, never
# this file), and Update.py already loads it the same way for HOLD_FOR and
# cached_worksheet.  The alternative was a third copy of the fingerprinting
# logic, which is how two caches end up disagreeing about what is current.
BOOK = Path(__file__).resolve().parents[1] / "2_Engine" / "trading_book.py"


def _load(path: Path, name: str):
    """Import a worksheet module off disk by path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rule_scores(df: pl.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """Per instrument, two scores: how often each rule sits on the volume leader.

    Returns (auto, forced).  `Mean_Auto_Best_V` is the mean of the worksheet's
    `Auto_Best_V` -- the calendar rule.  `Mean_Forced_Best_V` is the mean of
    `f_r_h_Best_V` -- forced_roll_hold, the four-branch ladder with the ratchet.
    One boolean per session each.

    THE FIVE CONFIRMATION-VARIANT MARKETS ARE SCORED BY NEITHER.  RS, SB, EUA,
    HE and VX run RS_/LT_/CS_forced_roll_hold, which adds a two-session
    confirmation on top of the ladder, and it scores higher than the plain
    version on all five -- RS .8915 against .8244 is the widest.  Their columns
    here are the two BASELINES, not the rule they were cleared on; the figure
    that cleared them is in the comment on their set.

    SESSION-weighted, not row-weighted.  Auto_Best_V describes a session but is
    repeated on every contract row of it, so a plain mean over rows weights each
    session by how many months happened to be listed that day.  The two differ
    by up to a point (ZC 0.769 row vs 0.759 session).

    ONE SHEET, ONE COLUMN, TWO GATES.  Every market is scored off
    front_contract.py.  It counts down to first_notice for has_notice markets
    and to last_trade for the rest -- the vendor returns None for first_notice
    on every contract those markets list, so there is nothing else to gate on.
    The question asked is identical -- did auto_roll sit on the volume leader --
    and the answers are comparable.  What differs is the date the calendar rule
    counts down to, and that difference is a property of the market, not of the
    measurement.
    For cash-settled markets last trade is a SETTLEMENT date rather than a
    delivery boundary, so their gate is a liquidity gate rather than protection
    from an obligation.  Same arithmetic, different thing at stake.
    This was three separate sheets until 2026-08-27.  They were ~85% identical
    and had NO behavioural divergence -- the one filter that differed was
    checked over 1,421,027 candidate sets and binds zero times.  See
    Front_Contract/Roll_Journal.md section 1.

    THIS COLUMN IS INFORMATION, NOT A DECISION.  Roll_Rule is set from the
    curated sets alone and does not read this number.  They were coupled while
    the has_notice half was being worked through, and the coupling broke as soon
    as a market could earn a rule its Auto_Best_V would not have given it -- ES,
    NQ, RTY, EMD, SJB, SXF, YAP4 and YXT4 all score above .95 on auto_roll and
    are all on forced_roll, because the forced ladder scores higher still.
    Read a low figure here as "the calendar rule alone does not fit this
    market", nothing more.  For the three STIRs -- LEU9 .0547, SO3 .0227,
    SR3 .0206 -- read it as "the question does not apply": they trade a deep
    strip at once and have no front contract to lead.
    """
    fc = _load(FC, "fc")
    todo = [(i, fc) for i in df.get_column("instrument").to_list()]

    # THIS IS THE LONGEST SILENCE IN THE PIPELINE AND IT USED TO PRINT NOTHING.
    # It rebuilds every instrument's FULL worksheet, and unlike stage 2 it calls
    # `worksheet()` directly rather than the cached wrapper, so none of it is
    # free.  Sitting immediately after the PANEL EDGE banner, that made a
    # working run indistinguishable from a hung one for minutes -- and it was
    # reported as exactly that.
    #
    # ONE FLUSHED LINE PER INSTRUMENT, NOT A REDRAWN BAR.  This runs as a
    # SUBPROCESS of Update.py whose stdout is a pipe, never a terminal, so
    # carriage returns would collapse into one unreadable line.  Update.py
    # passes `-u`, so a flushed line here reaches the console immediately.
    # THROUGH THE SHARED CACHE, NOT `worksheet()` DIRECTLY.  This loop used to
    # call the uncached entry point, so it rebuilt all 63 worksheets from
    # scratch -- ~175s -- and then threw them away: verify_holds rebuilt the
    # same 63 minutes later, and stage 2 read them a third time.  Writing into
    # the cache here means the later two stages hit it instead, which is the
    # difference between paying that cost once per run and paying it twice.
    #
    # THE ARGUMENTS MUST MATCH THE OTHER CALLERS EXACTLY or the key differs and
    # nothing hits.  verify_holds and trading_book both use the full window and
    # the RESOLVED as_of, so this resolves it too rather than passing the
    # literal "auto" that `worksheet()` defaults to -- the fingerprint hashes
    # that argument verbatim, and "auto" and "20260827" are different strings
    # for the same panel.  `panel_as_of()` is memoised per process, and main()
    # has already refreshed it, so this is free.
    tb = _load(BOOK, "tb_cache")
    as_of, _edge = fc.panel_as_of()

    t0 = time.time()
    print(f"\nSTAGE 2b  roll scores -- {len(todo)} worksheets "
          f"(cached: shared with stage 2)", flush=True)

    auto: dict[str, float] = {}
    forced: dict[str, float] = {}
    n_hit = 0
    for n, (inst, mod) in enumerate(todo, 1):
        print(f"  [{n:>3}/{len(todo)}] {inst:<8}{time.time() - t0:>6.0f}s",
              flush=True)
        try:
            w, hit = tb.cached_worksheet(mod, inst, "1900-01-01", "2100-01-01",
                                         as_of)
            n_hit += int(hit)
        except Exception as exc:
            print(f"  [warn] {inst}: scores not computed -- "
                  f"{type(exc).__name__}: {exc}")
            continue
        if not w.height:
            print(f"  [warn] {inst}: empty worksheet, scores left null")
            continue
        # ONE traversal for both.  worksheet() is the expensive call -- full
        # history for 63 instruments -- and it already computes every column;
        # scoring the two metrics in separate passes would double the cost to
        # read a different field off the same frame.
        sm = mod.session_means(w).row(0, named=True)
        auto[inst] = round(sm["Auto_Best_V"], 4)
        forced[inst] = round(sm["f_r_h_Best_V"], 4)
    print(f"  worksheet cache: {n_hit} hit, {len(todo) - n_hit} built  "
          f"({time.time() - t0:.0f}s)", flush=True)
    return auto, forced


def verify(rows: list[dict], notices: dict, year: int) -> list[str]:
    """A market the vendor gives a notice date for cannot be cash-settled.

    Reads what `build` already fetched rather than asking the vendor a second
    time: a re-fetch doubles the calls and, if NDU drops out between the two
    passes, can disagree with the has_notice column it is meant to corroborate.
    """
    bad = []
    for r in rows:
        if r["is_deliverable"]:
            continue
        for code, v in notices.get(r["instrument"], {}).items():
            if v:
                bad.append(f"{r['instrument']} marked cash-settled but "
                           f"{r['instrument']}-{year}{code} has a notice date {str(v)[:10]}")
                break
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default=None,
                    help=f"universe CSV; defaults to the {MAPPING_NAME} above this file")
    ap.add_argument("--year", type=int, default=2025,
                    help="count the cycle from this expiry year alone")
    ap.add_argument("--all-sessions", action="store_true",
                    help="every vendor session symbol, not just our universe")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent
                                         / "contract_cycles.csv"))
    ap.add_argument("--no-scores", action="store_true",
                    help="skip the Mean_Auto_Best_V pass (it rebuilds every worksheet)")
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip stage 1 and read the panel as it stands on disk")
    ap.add_argument("--refresh-full", action="store_true",
                    help="stage 1 refetches settled contracts too, not just live ones")
    ap.add_argument("--refresh-only", action="store_true",
                    help="run stage 1 and stop, leaving contract_cycles.csv alone")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage 1 reports what it would append and writes nothing")
    args = ap.parse_args()

    import norgatedata as nd
    from norgatedata import norgatehelper as H
    for name in ("User-Agent", "X-Requested-With"):
        v = H.session.headers.get(name)
        if v is not None and not str(v).isascii():
            H.session.headers[name] = str(v).encode("ascii", "ignore").decode()
    if not wait_for_ndu(nd):
        print("[ABORT] NDU is not answering after 4 attempts")
        return 2

    if args.all_sessions:
        instruments = sorted(nd.futures_market_session_symbols())
    else:
        mapping = Path(args.mapping) if args.mapping else find_mapping()
        with open(mapping, encoding="utf-8") as f:
            instruments = [r["norgate_code"] for r in csv.DictReader(f)]

    # ---- STAGE 1: bring the panel up to date --------------------------------
    # First in the pipeline, and first in this function, because everything
    # after it reads the files it writes: `build` asks the vendor for the cycle,
    # and `rule_scores` runs front_contract.py over the bars refreshed here.
    # front_contract owns the panel paths, so they are read off it rather than
    # recomputed -- two definitions of where the data lives is one too many.
    fc = _load(FC, "fc")
    if not args.no_refresh:
        refresh_metadata(nd, fc, instruments, dry=args.dry_run,
                         full=args.refresh_full)
        refresh_panel(nd, fc, instruments,
                      full=args.refresh_full, dry=args.dry_run)
        # NB: --refresh-only returns AFTER the edge report below, not here.
        # An earlier version stopped at this point and silently skipped the one
        # report the operator running a refresh most wants to see.
    elif args.refresh_only:
        print("[ABORT] --refresh-only with --no-refresh does nothing")
        return 2

    # ---- the ragged edge -----------------------------------------------------
    # The panel does not end on the same date for every market: ASX closes
    # earliest in the global day, so YAP4 and YXT4 carry a date the other 61 do
    # not have yet whenever the vendor's update lands between the two closes.
    # Computed on EVERY run, not only when stage 1 refreshed, because the
    # alignment the signals use is a fact about the panel and not about whether
    # this run happened to write to it.  refresh=True busts front_contract's
    # per-process cache, which stage 1 has just invalidated by adding bars.
    print("")
    print("PANEL EDGE")
    as_of, edge = fc.panel_as_of(refresh=True)
    lagging = fc.report_edge(as_of, edge)
    if lagging:
        print(f"    [WARN] {len(lagging)} instrument(s) look stale rather than "
              f"merely ragged -- a feed that has stopped reads the same as one "
              f"that is early")
    if args.refresh_only:
        print("")
        print("--refresh-only: stopping before the cycle table")
        return 0

    # ---- STAGE 2: the cycle table -------------------------------------------
    df, notices = build(instruments, args.year)

    problems = verify(df.to_dicts(), notices, args.year)
    for msg in problems:
        print(f"  [FAIL] {msg}")
    if problems:
        print(f"\n{len(problems)} settlement classification(s) contradict the vendor")
        return 1

    out = Path(args.out)
    # Written BEFORE scoring on purpose.  The worksheet reads its dead-month
    # list out of this very file, so it has to see the current one -- scoring
    # against the previous run's dead months would silently mis-trim.
    #
    # THAT INTERMEDIATE WRITE IS A LIVE HAZARD AND IS NOW UNDONE ON FAILURE.
    # It publishes a version of the table with NO Roll_Rule, Mean_Auto_Best_V,
    # Mean_Forced_Best_V or Unique_Roll -- seven columns where there should be
    # twelve -- and the scoring pass that fills them in takes ~3 minutes.
    # Anything that goes wrong in that window (and it was observed happening)
    # leaves the file published in that state, which is worse than not having
    # written it at all: `trading_book.rules()` reads Roll_Rule out of this
    # file, so the next run of stage 2 has no rules to build against, and
    # nothing about the file looks damaged -- it parses cleanly and every row
    # is there.
    #
    # So snapshot first and put it back if the scoring section raises.  A stale
    # complete table beats a fresh truncated one every time: the panel refresh
    # is idempotent and the next run rewrites it properly.
    _prev = out.read_bytes() if out.is_file() else None
    df.write_csv(out)

    if not args.no_scores:
        # ONLY `rule_scores` IS GUARDED because it is the only thing here that
        # can realistically fail: it rebuilds 63 worksheets over ~3 minutes,
        # touching the vendor panel, the cache and front_contract's rule logic.
        # Everything after it is polars column arithmetic over a frame already
        # in hand.
        try:
            scores, forced = rule_scores(df)
        except BaseException:
            if _prev is not None:
                out.write_bytes(_prev)
                print(f"\n  [RESTORED] scoring failed -- {out.name} put back to "
                      f"its previous COMPLETE contents rather than left "
                      f"truncated. Re-run to refresh it.")
            raise
        df = df.with_columns(
            pl.col("instrument").replace_strict(scores, default=None)
              .cast(pl.Float64).alias("Mean_Auto_Best_V"),
            pl.col("instrument").replace_strict(forced, default=None)
              .cast(pl.Float64).alias("Mean_Forced_Best_V"))
        # Roll_Rule, straight off the score just written.  A separate
        # with_columns because Mean_Auto_Best_V is not visible inside the one
        # that creates it.  Null propagates through the comparison -- an
        # unscored instrument gives null >= threshold = null, which `when` sends
        # to the otherwise branch -- so the markets with no score come out blank
        # without needing a test of their own.
        # auto_roll is tested FIRST and wins outright: a market the calendar
        # rule already handles has no reason to carry a more elaborate one.  No
        # instrument is currently in both -- CC .9082, KC .9271 and LE .8178 are
        # all well under ROLL_RULE_MIN -- but the order is stated rather than
        # left to chance, so adding one to FORCED_ROLL later cannot silently
        # demote it.  Everything else stays null: undecided, not "none".
        df = df.with_columns(
            # Straight from the curated sets, reading nothing else.  Roll_Rule and
            # Mean_Auto_Best_V are INDEPENDENT: a market earns its rule from the
            # timeseries audit, and the score is reported beside it as
            # information.  They were coupled while the has_notice half was
            # worked through, and the coupling broke as soon as an instrument
            # could earn a rule its score would not have given it.
            pl.col("instrument").replace_strict(_roll_rules(), default=None)
              .alias("Roll_Rule"))
        # Unique_Roll: true where the rule is one of the confirmation variants,
        # blank otherwise -- including blank, not false, for the three markets
        # with no rule at all.  Derived from Roll_Rule and nothing else, so it
        # cannot disagree with it.
        df = df.with_columns(
            pl.when(pl.col("Roll_Rule").is_not_null()
                    & ~pl.col("Roll_Rule").is_in(["auto_roll", "forced_roll"]))
              .then(pl.lit(True))
              .otherwise(None)
              .alias("Unique_Roll"))
        head = ["instrument", "last_date", "codes", "Dead_contracts",
                "Active_contracts", "per_year", "is_deliverable", "has_notice",
                "Mean_Auto_Best_V", "Mean_Forced_Best_V", "Roll_Rule",
                "Unique_Roll"]
        # last_date: the newest bar this instrument holds.  SECOND column, so a
        # mismatch is the first thing visible in the file rather than something
        # found by scrolling.  Anything not equal to as_of is either the ordinary
        # ragged edge (ahead) or a feed that has stopped (behind).
        df = df.with_columns(
            pl.col("instrument").replace_strict(edge, default=None)
              .alias("last_date"))
        df = df.select(head + [c for c in df.columns if c not in head])
        df.write_csv(out)
        if scores:
            avg = sum(scores.values()) / len(scores)
            print("")
            favg = sum(forced.values()) / len(forced) if forced else 0.0
            print(f"Mean_Auto_Best_V written for {len(scores)} instruments"
                  f"  (mean {avg:.4f})")
            print(f"Mean_Forced_Best_V written for {len(forced)} instruments"
                  f"  (mean {favg:.4f})")
            # Derived from the curated sets, NOT hardcoded: a hardcoded list
            # silently omitted +1_auto_roll when it was added, and the counts
            # summed to 61 of 63 while the CSV itself was correct.
            for rule in sorted(set(_roll_rules().values())):
                n = df.filter(pl.col("Roll_Rule") == rule).height
                print(f"Roll_Rule {rule:12s} {n:3d} of {len(scores)}")
            n = df.filter(pl.col("Mean_Auto_Best_V").is_not_null()
                          & pl.col("Roll_Rule").is_null()).height
            print(f"Roll_Rule {'undecided':12s} {n:3d} of {len(scores)}")
        else:
            print("")
            print("Mean_Auto_Best_V: nothing scored")

    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=200):
        print(df)
    print(f"\n{df.height} instruments -> {out}")
    n_yes = int(df["has_notice"].sum())
    print(f"\nnotice date on every {args.year} contract: {n_yes} instruments; "
          f"{df.height - n_yes} without "
          f"({df.filter(pl.col('is_deliverable') & ~pl.col('has_notice')).height} "
          f"of those deliverable)")
    counts = df.group_by("per_year").len().sort("per_year")
    print("\ncontracts per year, how many instruments:")
    for r in counts.iter_rows(named=True):
        print(f"  {r['per_year']:>2} per year   {r['len']:>3} instruments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
