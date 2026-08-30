# Roll Journal

*Opened 2026-08-27. Last updated 2026-08-27 (merge complete).*

The durable record behind the front-contract worksheets: what was tried, what
failed, and the specific sessions that forced each decision.

It exists because the three sheets are being merged into one `front_contract.py`
and the comments are the most valuable thing in them -- roughly 970 of their
2,057 lines are prose, and a merge is the operation most likely to flatten it.
Anything here has already cost someone the time to discover. Read this before
deciding a rule "looks unnecessary".

Convention: every claim names the instrument and session that produced it. A
finding with no case attached is a guess someone wrote down confidently.

---

## 1. Why there were three sheets, and why one is enough

    has_notice_front_contract.py       22 markets, is_deliverable AND has_notice
    not_has_notice_front_contract.py   19 markets, is_deliverable, no notice
    not_deliverable_front_contract.py  22 markets, cash-settled

The split looks like a trichotomy and is not. Measured on the code with comments
stripped, the three are ~211/214/224 lines and **~85% identical**. The real axes:

* **Date source is BINARY**, not ternary: `has_notice ? first_notice : last_trade`.
  Groups B and C already share it, which is why those two differ by only 18
  code lines.
* **The variant hold column is ONE algorithm under three names.** After
  normalising the `rs_`/`lt_`/`cs_` prefix, the blocks are identical -- the
  remaining diff is statement order and one temp variable.
* Everything else is data (blacklists), one extra column, or cosmetics.

**The one genuine logic divergence does not bind.** `has_notice` carries an extra
`till > FORCED_ROLL_MIN_CD` test in `held2` that the other two omit. Its comment
claims it cannot bind. Checked over **1,421,027 candidate sets: it binds 0
times, on 0 instruments.** So the three sheets have no behavioural divergence
and a merge can reproduce all three outputs exactly. Keep the defensive filter
anyway, for the reason its own comment gives: a filter that states its
precondition does not quietly change meaning when someone edits the line above.

---

## 2. Negative results -- things that were tried and did NOT work

The most expensive knowledge here. Each of these looked right.

**Vendor comparison in calendar months.** Made every quarterly market look three
months wrong. Must be measured in CYCLE STEPS.

**Asserting agreement with the vendor.** Failed for being right: SR3 carries
6.9x the vendor's volume share. Assert liquidity, not agreement.

**Persistence to fix RS canola.** Did not work -- RS's deferred OI lead lasts
weeks, not days. Reopened 2,610 gaps. The generalisation came from four days of
data.

**Dominance override without a direction constraint.** ZB rolled 1 -> 74
backward, ZN 1 -> 57. Fixed only by the forward-only constraint.

**A `volume > 0` filter on auto_roll's candidates.** Inherited from `live`.
Emptied the column on quiet sessions -- CGB on Christmas Eve 1999, holding
29,027 open interest and 63 days from notice. Removing it closed all 7 gaps AND
40 backward rolls. auto_roll reads NO volume, NO open interest, NO is_passed.
A passed contract has `till <= 0`, which the `<= AUTO_ROLL_CD` rule already skips.

**Gating Best_Oi on volume.** Made a missing field disqualify every real
contract -- CT 1998-01-16. Best_Oi tests `open_interest > 0`; Best_Vol tests
`volume > 0`. The asymmetry is deliberate: open interest is what survives a
session the vendor recorded no volume for, which is the one occasion a fallback
is worth having.

**Leaving Best_Oi ungated on the notice window.** Defensible while the column
was merely descriptive; stopped being so once forced_roll_hold fell back on it.
ZM 1990-06-25 put the hold four days from delivery.

**Fixed offsets down the strip, ranked within the `volume > 0` set.** Manufactured
turnover -- a contract printing nothing shifts every position behind it. LEU9
read 548-1,046 rolls. Rank on LISTED contracts and it is 32, the true quarterly
cycle. Measurement artifact, not market behaviour.

**Reading a 40-day "cliff" as a threshold.** The 5.8x step between the 30-39 and
40-49 day buckets is real decay, but sweeping the roll offset shows improvement
is MONOTONIC with no knee. Nothing distinguishes 40 from 50 or 70.

---

## 3. Incidents that decided a rule

| case | what it forced |
|---|---|
| CGB 1999-12-24 | auto_roll must not read volume (29,027 OI, 63 days out, column empty) |
| CT 1998-01-16 | Best_Oi must not require volume |
| ZM 1990-06-25 | Best_Oi needs the notice-window gate; hold landed 4 days from delivery |
| RS 2025-03-10 | one session's volume print + ratchet = 21 sessions in the wrong contract; became the confirmation variant |
| SI 1980F / 1978-03-07 | inception must be measured on SELECTABLE contracts; F is dead in silver, so 37 sessions had no auto_roll |
| CT 1978 | a single contract 522 days from notice printing 5-10 lots -- inception exists because coverage is not liquidity |
| SR3-2024H 2024-05-16/17 | `PaddingType.NONE` can omit real bars (16,230 and 6,407 lots). Reimported. Swept all 15,231 contracts: this was the ONLY case |
| HO 2007-09-03 | US Labor Day, NYMEX closed, one phantom bar -> the only backward roll in group B |
| LFT9 2025-05 | 17 sessions of missing front-month rows |

**`Forced_roll_V` needs its "not equal to auto_roll" gate.** Without it the column
was auto_roll verbatim on all 6,314 populated sessions across the 22 in-scope
instruments, 2005-2015. Both rules select the nearest contract clear of the
window -- one by expiry order, one by notice date -- and those orders do not
diverge in these markets.

**`+2_Forced_Roll_V` exists for the crop-year jump.** The first two columns are
distance-limited to the nearest and second-nearest. Grains roll old crop to new
crop in one jump of two or three months. Every blank `Test_Hold` before this
column was added had the leader sitting +2 or further, and **95% were that jump**.

**The ratchet.** Without it forced_roll_hold reversed **1,605 times** across the
ten instruments Roll_Rule left blank -- 224 in LE alone, which has 2 blank
sessions. Both branches chase the volume leader and follow it back when it
flickers. Compare on TODAY's `till`: each contract's number falls by a day
overnight, so comparing across sessions scores an ordinary hold as a reversal.

**Why the last fallback is Best_Vol.** The newest session in any file carries
volume and no open interest, because exchanges publish OI the next morning.
Historically one row at the tail; on live data it is always today.

---

## 4. The STIRs -- LEU9, SO3, SR3

The longest single investigation. Conclusions in order of how much they cost.

**A leader-based score is meaningless in a strip.** The volume LEADER itself
carries only 14-20% of a session (LEU9 .140 median, SO3 .153, SR3 .163). "Is the
leader" is a binary against a leader that does not dominate. Judge on SHARE OF
THE ACHIEVABLE CEILING, and on the timeseries.

**A median-share dead-month test cannot judge a strip, but a LEADERSHIP test
can.** Median share asks how big a month is -- no good answer in a strip. Ask
which months are in the CYCLE at all:

    LEU9   H 907   M 1,045   U 866   Z 1,719   sessions as volume leader
           F 0  G 0  J 0  K 0  N 0  Q 0  V 0  X 0   -- never, not once / 4,537

HMUZ carry **98.70%** of LEU9 volume across 95 contracts; the eight ECB serial
months carry 1.30% across 148. Trimming LEU9 to HMUZ moved Mean_Auto_Best_V
.0547 -> .2872 on its own.

**Euribor and SOFR/SONIA are structurally different, and it is not a data
artifact.** LEU9 settles to a forward-looking fixing ON expiry, so its front
month stays uncertain and traded to the end. SO3 and SR3 settle to daily rates
COMPOUNDED IN ARREARS over the contract's own reference quarter -- by the time
one is the front month its price is largely already determined and the volume has
gone. That is why LEU9 takes auto_roll and the other two cannot.

**Almost every STIR pathology is a young-market artifact.** SONIA and SOFR
futures launched 2018 and were not real markets until 2021. SO3 quarterly
breadth -- how many listed months actually trade -- plateaus at 21-24 from
**2021-01-12** and never falls back; before that it is 9-13 months on a few
thousand lots. 205 of SO3's 207 stale sessions and ALL 142 of SR3's fall before
Q4 2021.

    THE INCEPTION RULE IS TOO WEAK FOR THESE. INCEPTION_VOLUME = 1000 admitted
    SO3 from 2018-08 on a 5,103-lot book with 9.8 months trading. That is 39-41%
    of each panel. The rule is global, so any instrument whose vendor history
    predates its liquidity has the same soft edge. NOT YET SWEPT.

**Norgate's own continuous series is not a reference implementation.** `&SO3_CCB`
uses HMUZ, rolls on the LAST TRADING DAY (26 of 30 rolls at 0 days, 4 at 1), and
agrees with our auto_roll 94-96% / forced_roll 0-5%. On most roll days the
contract it rolls out of traded **0 lots**. Copying it imports the defect.

**`+1_auto_roll_hold` is auto_roll with the index shifted, not a new rule class.**
An earlier note in contract_cycles.py concluded a pinned-offset rule was "a
build, not a set entry". True for position 2 and beyond, which needs the rule to
step twice. False for position 1: `AUTO_ROLL_CD >= 91` makes the front always
inside the window, so auto_roll always takes cal[1], and it saturates there
(91/95/100/120 give identical output).

**Rolling earlier buys nothing, and this was checked rather than assumed.**
Sweeping the trigger over 5/15/30/45 leaves the thin rate flat at 0.3-0.4% and
makes the worst roll leg WORSE (3,627 -> 1,486 at N=30). The held month never
comes within 90 days of expiry against a death zone starting near 40.

**Timeseries audit, both instruments, from 2021-01-12, Panama on DIFFERENCES**
(an interest-rate future is 100 minus a rate; a ratio return is meaningless):

* Continuous, no gaps. Daily change median 1.0bp, p95 6.5-7.0bp.
* Flat days fall 45.7% -> 19.4% (SO3) and 35.8% -> 15.7% (SR3), and the ones that
  remain are REAL: median volume 25,389 and 53,694 lots, none at zero. They
  cluster where the RATE was pinned -- ~44-48% in 2021 at 0.1% Bank Rate, 8-11%
  through the hiking cycle. An unchanged close, not a dead contract.
* Roll gaps are ~22bp against a 1bp daily move, but that is the forward slope and
  the adjustment absorbs it: on a roll date the series return reduces exactly to
  `old_close(t) - old_close(t-1)`.
* Both track real events -- SO3's largest moves are the 2022-09-23/26 mini-budget
  and gilt crisis; SR3's is 2023-03-13, **+86.5bp on 2.79M lots**, the Monday
  after SVB.
* Total back-adjustment -2.565 (SO3) and -0.050 (SR3). Neither drifts.

---

## 5. Data defects in the panel

**HO 2007-09-03.** US Labor Day, NYMEX closed, yet the panel carries a single
bar for HO-2008G (563 lots, 12,662 OI, five months out) and nothing else. The
sessions either side carry 36 contracts. One row in a session forces any
nearest-expiry rule to name it -- the only backward roll in the 19-instrument
group. Deleted rather than tolerated.

**LFT9 2025-05.** Sessions carrying only 2025Z and 2026H, both at zero, while
LFT9-2025M -- the front month, ~62,000 lots, 18 days from last trade -- has NO
ROW AT ALL. The FTSE 100 future did not stop for a fortnight; the bars are
missing.

    STALE COMMENT, FIX IN THE MERGE. The comment above BAD_SESSIONS says
    "2025-05-16 .. 05-30: ten consecutive sessions". The set actually holds
    17 entries from 2025-05-06 to 2025-05-30. The data was corrected after the
    first version was built off a truncated listing; the comment was not.
    Verified 2026-08-27.

**187 zero-volume sessions across 19 instruments** were identified. Only LFT9 was
fixed, by explicit instruction. The rest are still in the panel.

**`PaddingType.NONE` was cleared.** It CAN omit real bars -- SR3-2024H proved it.
Swept all 63 instruments, 15,231 contracts, 7,140,933 rows against
`ALLMARKETDAYS`: **0 dropped bars, 0 restatements, 0 fetch failures.** SR3-2024H
was the only case in the panel. The detector was positive-controlled first (4
deliberately deleted rows, all 4 found). Keep using NONE.

---

## 6. Constants

    BV3_SESSIONS = 3            consecutive sessions to arm B_V_3
    AUTO_ROLL_CD = 5            step off the front this close to the gate date
    BEST_VOL_MIN_CD = 5         a contract this close cannot win Best_Vol/Best_Oi
    FORCED_ROLL_MIN_CD = 5      Forced_roll_V will not park you this close
    ROLL_CONFIRM_SESSIONS = 2   the variant hold waits for a repeat
    INCEPTION_VOLUME = 1000     panel starts at the first session a SELECTABLE
                                contract prints this

`AUTO_ROLL_CD` is currently module-level and shared by every instrument.
`+1_auto_roll` needs it per-instrument, or its own equivalent -- that is the one
real structural change the STIR work implies.

**The confirmation variant's cost is a late roll.** Every genuine move lands one
session after the signal -- a real position held one day too long, every roll,
forever. The wait is ABANDONED once the incumbent reaches the window or stops
being listed; without that escape the delay pushed the hold INTO the notice
window on 6 RS sessions. A one-day delay costs nothing mid-contract and
everything at the end of one.

---

## 7. Conventions that are easy to break

* **Blank is not false.** A passed contract gets a BLANK flag -- out of the
  running, not losing it. `means` reads blank as null and drops it from the
  denominator. Forced_Best_V and +2_Forced_Best_V are blank where their rule was
  not asked; Auto_Best_V and Test_Best_V are NEVER blank, because those rules are
  asked every session and a session they cannot answer is a LOSS.
* **Session-level columns are not blanked per row.** auto_roll and its scores
  describe the SESSION, so the same answer appears on every row including passed
  ones. The blanking convention belongs to the per-contract flags.
* **Warm-up is load-bearing.** B_V_3 and every hold series carry state across
  sessions. `worksheet` iterates from the start of contract history and only
  emits inside the window. Slice the input instead and the first rows read false
  when they should read true.
* **Dead months are dropped from the DATA, not just from selection.** Keeping
  them made the sheet unreadable where most of the listing is dead: EUA trades
  only December, so a 9-session window carried 178 rows of which 9 mattered.
* **Session symbols and contract symbols are not interchangeable.**
  `futures_market_name` and `session_type` take `ES`; `tick_size`, `margin` and
  `lowest_ever_tick_size` need `ES-2026Z` and return "not found" for a session,
  writing empty columns while the run still exits 0.
* **`_private()` walks up parents.** A counted `parents[N]` was silently wrong
  the moment a script moved directory, and produced a default naming a file that
  does not exist rather than an error anyone reads.

---

## 8. The merge, as executed (2026-08-27)

DONE. `front_contract.py`, 689 lines, replaces the three sheets' 2,057.

**Acceptance: 63 of 63 instruments identical, 0 failed.** Every column the old
sheets produced is present and byte-identical in the merged one, compared per
column BY NAME over 6,768,796 rows. `contract_cycles.py` re-run end to end
reproduces `contract_cycles.csv` byte for byte.

What changed on purpose, and nothing else did:

* One column order. The confirmation variant now sits after `f_r_h_Best_V`;
  the notice sheet had it before.
* `RS_`/`LT_`/`CS_forced_roll_hold` -> **`confirm_forced_roll_hold`**, one
  column. The `Roll_Rule` VALUES in contract_cycles.csv deliberately keep the
  old three names: those record WHY each market was cleared, which the column
  name never did.
* All 63 instruments now carry `+1_auto_roll_hold`; 41 of them gained it.
  Additive, so no existing column moved.
* Group date labels kept per group -- `first_notice`/`till_notice_cd` against
  `last_trade`/`till_last_trade_cd`. `gate()` picks the pair.
* `contract_cycles.py`: `HN`/`LT`/`CS` and three `_load` calls collapse to one
  `FC`; the three-way `todo` build collapses to one list.

TWO BUGS THE MERGE SURFACED, both in the surrounding work rather than the sheets:

* **The acceptance test was wrong before the code was.** The first run reported
  27 failures. `a != b` yields NULL where both sides are null, and filling that
  with True counted two absences as a difference. The failing counts summed to
  exactly 584 -- the number of rows carrying no gate date -- which is what gave
  it away. Use `eq_missing`, never `!=` plus `fill_null(True)`.
* **The Roll_Rule count printout was hardcoded** and silently omitted
  `+1_auto_roll`, so it summed to 61 of 63 while the CSV itself was correct.
  Now derived from `_roll_rules()`, so it cannot go stale again.

Also fixed in passing: the stale LFT9 comment (section 5) now says seventeen
sessions from 2025-05-06, which is what the data holds.

---

## 9. Original merge checklist

1. Capture goldens: all 63 instruments, full history, from the CURRENT three
   scripts. Nothing modified yet.
2. Write `front_contract.py`.
3. Diff against goldens. **Pass = byte-identical**, modulo deliberate renames.
   Any diff is a merge bug, not a design question -- section 1 established there
   is no behavioural divergence to reconcile.
4. Repoint `contract_cycles.py` (`HN`/`LT`/`CS` + three `_load` calls collapse to
   one; the three-way `todo` build collapses to one list). Re-run and require
   `Mean_Auto_Best_V`, `Mean_Forced_Best_V`, `Roll_Rule` unchanged for all 63.
5. Merge the prose deliberately, not by concatenation.
6. Delete the three originals.

Steps 1-4 are reversible; the originals stay on disk until 6.

**Settled naming.** Group date labels stay as they are -- `first_notice` /
`till_notice_cd` for the notice group, `last_trade` / `till_last_trade_cd` for
the others. A generic name would hide which gate applied, which is information a
reader of the worksheet wants. The variant hold column needs one name in place
of RS_/LT_/CS_; the `Roll_Rule` VALUES in contract_cycles.csv should NOT follow
it, because those three names record why each market was cleared.

**Done 2026-08-27:** `SO3_+1_hold` -> `+1_auto_roll_hold` (worksheet),
`SO3_+1` -> `+1_auto_roll` (contract_cycles). It is computed for all 22
instruments in the cash-settled sheet and consumed by SO3 and SR3.
