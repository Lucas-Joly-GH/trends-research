"""
Portfolio: signals into contracts.  Stage 3 of the pipeline.

Stage 2 leaves each instrument with a forecast and a risk figure.  This turns
those into a number of contracts to hold, which is the first object in the
pipeline that is about the PORTFOLIO rather than about an instrument.

Equation 3.32:

    N(i,t) = f_final(i,t)/10  x  ( E_t . tau . w_i . IDM_t ) / sigma$(i,t)

    f_final   SIGNAL from the trading book, capped +/-20 (eq 3.31)
    E_t       NAV at t -- COMPOUNDING, so this stage is sequential
    tau       0.20, the RISK BUDGET: annualised vol the PORTFOLIO targets.
              The per-instrument share is tau . w_i . IDM_t, which on 63
              markets is 0.92% of NAV, not 20% -- see the note on TAU.
    w_i       1/N over the instruments ACTIVE at t
    IDM_t     eq 3.33, below
    sigma$    price_vol_USD_ann from the book -- eq 3.35, complete

and equation 3.33:

    IDM_t = min( 4.0, 1 / sqrt( w' C_t w ) )

WHY THIS IS A SEPARATE STAGE AND NOT MORE COLUMNS IN THE BOOK.  Two of 3.32's
inputs do not exist for a single instrument: `w_i` needs the count of active
markets, and `IDM_t` needs the correlation matrix across them.  And `E_t`
compounds, so position -> P&L -> NAV -> next position is a sequential loop,
where everything in 2_Engine is vectorised column math over a fixed history.
The two deferred gates (3.29, 3.30) belong here too, for the reason the trading
book's journal already gives: a drawdown gate measured on the strategy's P&L
needs the position, which needs the forecast.

    python portfolio.py
    python portfolio.py --nav 250e6         capital
    python portfolio.py --tau 0.15          risk budget (portfolio target vol)
    python portfolio.py --compare-spans 256,512
    python portfolio.py --no-write          statistics only
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
BOOK_PY = HERE.parent / "2_Engine" / "trading_book.py"
POS = HERE / "Positions"
PORTFOLIO = HERE / "Portfolio.csv"
IRX_FILE = BOOK_PY.parent / "IRX" / "IRX.parquet"

# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------
# NAV_0 is the paper's 100M.  A DEFAULT, NOT A CONSTANT OF NATURE -- `--nav`
# overrides it, because the sizing is linear in E_t and the whole book scales
# with it.  It matters more than it looks: flooring to whole contracts deletes
# an instrument entirely when its allocation is under one contract, and at 1M
# that silently removes 34 of 63 markets -- systematically the large-notional
# ones (index futures, metals, crypto), leaving a rates-and-softs book that
# still reports itself as diversified.  Every instrument clears one contract by
# about 25M.  At 100M the dearest (BTC) affords ~7.7 contracts and a typical
# 30k-sigma$ market ~35, so truncation costs on the order of 1%.
NAV_0 = 100_000_000.0

# TAU IS THE RISK BUDGET -- 0.20, annualised volatility -- and `--tau` overrides
# it exactly as `--nav` overrides the capital.  The two are the only free
# parameters of 3.32; everything else in it is measured.
#
# WHERE THE PER-INSTRUMENT BUDGET ACTUALLY COMES FROM.  In 3.32 tau multiplies
# E_t, the WHOLE account, and w_i = 1/N then divides that across the active
# markets.  So the amount of risk instrument i is allowed is not tau, it is
#
#     E_t . tau . w_i . IDM_t        (annualised USD volatility)
#
# and dividing by sigma$(i,t) -- one contract's annualised USD volatility -- is
# what turns a risk budget into a contract count.  At the current 63 markets and
# IDM 2.89 that is 0.20 x (1/63) x 2.89 = 0.92% of NAV per instrument, not 20%.
#
# THE TWO READINGS ARE NOT INTERCHANGEABLE, and the IDM is what settles it.
# Summing the per-instrument budgets under the correlation matrix gives
#
#     portfolio vol = E . tau . IDM . sqrt(w'Cw) = E . tau
#
# because IDM is DEFINED as 1/sqrt(w'Cw).  Cancelling exactly like that is the
# whole purpose of 3.33: tau is the volatility the PORTFOLIO targets, and the
# multiplier exists precisely so the parts add up to it.  Were tau instead a
# per-instrument target of 20%, the book would carry roughly 63 times the
# intended risk and the IDM term would have nothing to cancel.
#
# The realised series agrees: annualised portfolio volatility comes out at
# 16.89% against tau = 20%, the shortfall being truncation to whole contracts,
# the IDM cap, and forecast dilution -- all of which only ever reduce it.
TAU = 0.20

# Equation 3.33.  The cap is sqrt(16) -- Carver's figure for 16 perfectly
# decorrelated instruments -- and it is a rail rather than a binding
# constraint here: measured over the last 512 sessions this universe runs a
# mean pairwise correlation of 0.078, giving 1/sqrt(w'Cw) = 3.28 against
# sqrt(63) = 7.94 if it were truly independent.
IDM_CAP = 4.0
# 512 TO MATCH `FDM_CORR_SPAN`, and the tiebreak is deliberate rather than lazy.
# The span was swept over a SIXTEENFOLD range and changes essentially nothing:
# Sharpe 1.22 to 1.24, annual vol 16.87% to 17.12%, turnover 321x to 334x,
# monotonic throughout but far inside the noise.  Turnover was where a
# difference was expected -- a jumpier multiplier ought to churn positions --
# and it moved 4%, because IDM is a SINGLE SCALAR applied to the whole book at
# once, so a wobble rescales every line proportionally and truncation to whole
# contracts absorbs most small rescalings without changing any contract count.
#
# With nothing to choose on merit, the two diversification multipliers in this
# pipeline now estimate their correlations over the same horizon, which is one
# fewer arbitrary number to explain.  It is also the better half of the range
# here: the cap binds on 1.3% of sessions against 5.7% at span 64.
IDM_CORR_SPAN = 512
# 256 matches FDM_MIN_PERIODS -- a full trading year of overlap before a pair
# is trusted, independent of the decay applied to it.
IDM_MIN_PERIODS = 256
# w'Cw below this is treated as unusable rather than inverted: 1/sqrt would
# explode.  Same guard, same reason, as FDM_VAR_FLOOR in the trading book.
IDM_VAR_FLOOR = 0.01

# Equation 3.30, the drawdown gate.  Like 3.29's, every constant is DERIVED
# from the risk budget: the threshold is -tau/2 (half the annual budget spent
# in one quarter) and the slope is 2/tau, so changing tau recalibrates both.
GDD_LOOKBACK = 64
GDD_THRESHOLD = -0.10          # = -TAU / 2
GDD_STEEPNESS = 10.0           # = 2 / TAU
GDD_FLOOR = 0.50

# Equation 3.36, the no-trade buffer.  Carver's 10%, kept as-is: the paper
# calls it arbitrary and adopts it unchanged rather than fitting one, which
# is the honest treatment of a number that cannot be derived.
BUFFER = 0.10

# TRADABILITY IS NOT COMPUTABILITY, and keeping the two apart is the whole
# reason these live here rather than upstream.
#
# Every instrument keeps computing over its FULL history in stages 1-2: the
# EWMACs, the volatility legs, skew, vol-of-vol, the pooled FDM and the
# cross-sectional z-score all need that burn-in, and XS_trend in particular is
# estimated ACROSS instruments -- dropping a young market from the books would
# change what the mature ones are scored against.  So nothing upstream knows
# about either rule.  They gate one thing only: whether a position may be taken.
#
# START_DATE  the first session the book may hold anything.  Matched with >=,
#             so it lands on the first actual session on or after it rather than
#             on a date that may not be a session at all.
# MIN_SESSIONS an instrument's OWN session count before it may be traded -- five
#             years at 256 sessions.  It binds well after the signal warm-up
#             (849 sessions), so it is the operative constraint, not a
#             restatement of one.  No instrument is excluded permanently: LEU9
#             clears it 2013-10-31, BTC 2023-01-18, ETH 2026-03-26.
START_DATE = "2026-01-02"
MIN_SESSIONS = 256 * 5

# f_final/10.  Phi from eq 3.21 -- the scale SIGNAL is normalised to, so this
# ratio is 1.0 for an average-strength forecast rather than an arbitrary tenth.
SIGNAL_PHI = 10.0


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ewm_corr_sum(X: np.ndarray, active: np.ndarray, span: float,
                 min_periods: int, chunk: int = 128):
    """(w'Cw, n_active, n_undefined_pairs) per session, from EWM correlations.

    THE MATRIX IS NEVER MATERIALISED.  3.33 needs only w'C_t w, and with
    w = 1/N on the active set that is

        w'Cw = ( 1/N^2 ) . sum over active i,j of C(i,j,t)

    a scalar per session.  Holding the (T,N,N) array to get it would cost 399 MB
    at 63 instruments for a number that is one float wide, so pairs are
    accumulated into a running sum instead and discarded.  The diagonal
    contributes exactly N (each C(i,i) = 1), so only the i<j pairs are walked:

        w'Cw = ( N + 2 . sum over i<j of C(i,j,t) ) / N^2

    THE MATH IS ewm_corr_4's, DELIBERATELY, so the two agree where they overlap:
    pandas' adjust=True convention with the sum_w^2/(sum_w^2 - sum_w2) bias
    correction, and PAIRWISE rather than joint NaN handling -- each pair
    accumulates only over bars where both series are observed.  That last part
    is not cosmetic on this panel, where inception dates span 1979 to 2010; a
    joint rule would blank the whole matrix on any bar missing one market.

    BATCHED THROUGH lfilter.  The accumulators run over a (T, chunk) block
    rather than one series at a time, which is the difference between 16.5 s and
    0.8 s for 2,016 pairs.  `chunk` bounds the working set: 128 pairs is ~77 MB.

    AN UNDEFINED PAIR BETWEEN TWO ACTIVE MARKETS COUNTS AS 1.0, not as 0.  It
    can only happen while a pair's overlap is shorter than min_periods, and the
    choice is the conservative one: 1.0 says "assume these move together",
    which lowers IDM and undersizes.  Treating it as 0 would claim
    diversification that has not been measured and lever up on it.  The count is
    returned so that a policy which never fires can be seen never to fire.
    """
    from scipy.signal import lfilter

    T, K = X.shape
    alpha = 2.0 / (span + 1.0)
    decay = 1.0 - alpha
    mask = np.isfinite(X)
    vals = np.where(mask, X, 0.0)
    mf = mask.astype(np.float64)

    def acc(a, d):
        return lfilter([1.0], [1.0, -d], a, axis=0)

    # ---- diagonal first: each series' own EWM variance ------------------
    sw = acc(mf, decay)
    sw2 = acc(mf, decay * decay)
    sx = acc(vals * mf, decay)
    sxx = acc(vals * vals * mf, decay)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = sx / sw
        var = sxx / sw - m * m
        den = sw * sw - sw2
        var = np.where(den > 0, var * (sw * sw) / den, np.nan)
    n_obs = np.cumsum(mf, axis=0)
    var = np.where(n_obs >= max(min_periods, 1), var, np.nan)
    sd = np.sqrt(np.clip(var, 0.0, None))
    # An instrument whose own variance is not estimable cannot be sized, so it
    # is not active regardless of what the book says about its forecast.
    ok = np.isfinite(sd) & (sd > 0)
    act = active & ok

    n_act = act.sum(axis=1).astype(np.float64)
    pair_sum = np.zeros(T, dtype=np.float64)
    n_undef = np.zeros(T, dtype=np.float64)

    iu, ju = np.triu_indices(K, k=1)
    for s in range(0, len(iu), chunk):
        i = iu[s:s + chunk]
        j = ju[s:s + chunk]
        both = mf[:, i] * mf[:, j]
        xi = vals[:, i] * both
        xj = vals[:, j] * both
        b_sw = acc(both, decay)
        b_sw2 = acc(both, decay * decay)
        b_sx = acc(xi, decay)
        b_sy = acc(xj, decay)
        b_sxy = acc(xi * xj, decay)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = b_sx / b_sw
            my = b_sy / b_sw
            cov = b_sxy / b_sw - mx * my
            den = b_sw * b_sw - b_sw2
            cov = np.where(den > 0, cov * (b_sw * b_sw) / den, np.nan)
            corr = cov / (sd[:, i] * sd[:, j])
        corr = np.where(np.cumsum(both, axis=0) >= max(min_periods, 1),
                        corr, np.nan)
        corr = np.clip(corr, -1.0, 1.0)
        live = act[:, i] & act[:, j]
        bad = live & ~np.isfinite(corr)
        n_undef += bad.sum(axis=1)
        pair_sum += np.where(live, np.where(np.isfinite(corr), corr, 1.0),
                             0.0).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        wcw = (n_act + 2.0 * pair_sum) / (n_act * n_act)
    wcw = np.where(n_act > 0, wcw, np.nan)
    return wcw, n_act, n_undef, act


def idm_from(wcw: np.ndarray) -> np.ndarray:
    """Equation 3.33.  1.0 where it cannot be computed.

    1.0 IS "NO DIVERSIFICATION CREDIT", which is the safe direction: it sizes
    the book as though every position were the same bet.  The alternative --
    carrying the last usable value forward -- would apply a multiplier measured
    on a different universe to the one actually held.
    """
    out = np.ones_like(wcw)
    good = np.isfinite(wcw) & (wcw > IDM_VAR_FLOOR)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[good] = np.minimum(IDM_CAP, 1.0 / np.sqrt(wcw[good]))
    return out


def panel(tb, insts: list[str]):
    """Every series stage 3 needs, on one grid.  (dates, dict of (T,K) arrays)."""
    cols = ["date", "symbol", "SIGNAL", "price_vol_USD_ann", "daily_ret",
            "Continuous_C", "Continuous_O", "close", "FX_rate", "s_g_vol"]
    per = {i: tb.load_book(i).select(cols) for i in insts}
    dates = sorted(set().union(*(set(d.get_column("date").to_list())
                                 for d in per.values())))
    pos = {d: k for k, d in enumerate(dates)}
    T, K = len(dates), len(insts)
    out = {c: np.full((T, K), np.nan) for c in
           ("SIGNAL", "sigma", "ret", "contc", "conto", "raw", "fx", "gvol")}
    sym = np.full((T, K), None, dtype=object)
    src = {"SIGNAL": "SIGNAL", "sigma": "price_vol_USD_ann", "ret": "daily_ret",
           "contc": "Continuous_C", "conto": "Continuous_O", "raw": "close",
           "fx": "FX_rate", "gvol": "s_g_vol"}
    for k, i in enumerate(insts):
        d = per[i]
        idx = np.array([pos[x] for x in d.get_column("date").to_list()])
        for name, col in src.items():
            v = d.get_column(col).to_numpy().astype(np.float64)
            out[name][idx, k] = v
        for r, s in zip(idx, d.get_column("symbol").to_list()):
            sym[r, k] = s
    return dates, out, sym


def _ffill(a: np.ndarray) -> np.ndarray:
    """Carry the last observation forward down axis 0.

    ONLY FOR THE PRICE USED IN P&L.  A market shut for a holiday has not moved;
    forward-filling makes its diff 0 that day and lets the next session carry
    the whole move, which is what actually happened.  Applied to a forecast or a
    risk figure it would invent information, so nothing else here uses it.
    """
    m = np.isfinite(a)
    idx = np.where(m, np.arange(a.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return a[idx, np.arange(a.shape[1])[None, :]]


def simulate(dates, P, sym, insts, tb, nav0: float, span: int,
             tau: float = TAU, compound: bool = True,
             use_gvol: bool = True, use_gdd: bool = True,
             use_buffer: bool = True, buffer: float = BUFFER,
             start_date: str = START_DATE):
    """The sequential pass.  Returns (per-instrument frames, portfolio frame)."""
    T, K = P["SIGNAL"].shape
    ps = np.array([tb.pointsize_of(i) for i in insts], dtype=np.float64)

    # ACTIVE = the book gives both a forecast and a risk figure.  sigma$ must be
    # strictly positive: it divides.
    # AN INSTRUMENT'S OWN SESSION COUNT, not calendar time: the cumulative
    # number of bars it has printed by t.  A market that trades thinly still
    # has to live through 1,280 of its own sessions.
    seen = np.cumsum(np.isfinite(P["contc"]), axis=0)
    old_enough = seen >= MIN_SESSIONS
    after_start = np.array([d >= start_date for d in dates])[:, None]
    tradable = old_enough & after_start

    # TWO DIFFERENT SETS, AND CONFLATING THEM WAS A BUG.
    #
    #   in_universe  the instrument is PART OF THE BOOK at t: past its 5-year
    #                mark, past START_DATE, has begun and has not ended.  It
    #                does NOT blink with exchange calendars.
    #   sizeable     in_universe AND the market printed a bar today, so a new
    #                position can actually be calculated.
    #
    # `w_i = 1/N` and the IDM correlation set take in_universe.  Only the
    # decision "may I re-size this line today" takes sizeable.
    #
    # WHY IT MATTERS.  Weighting by today's attendance means that on a session
    # where most markets are shut -- New Year's Eve, a US holiday while Europe
    # trades -- the handful that are open inherit the whole risk budget.
    # Measured before the fix: the per-instrument budget jumped from 1.94% of
    # NAV to 4.38% on such sessions, the open markets were sized 2.3x too
    # large, and the position reverted the next day.  Turnover on those
    # sessions ran 3.00x of NAV against a 0.37x median, and 2.88x on the day
    # after -- an 8x spike in and another out, on 482 sessions, 3.8% of the
    # traded history.  The IDM wobbled with it, 3.50 -> 2.41 -> 3.09 -> 3.49
    # across four consecutive sessions.
    #
    # The portfolio still HOLDS the shut lines -- they are carried, correctly --
    # so their risk is still on.  Handing their budget to whoever happened to
    # open double-counts it.
    sizeable_today = (np.isfinite(P["SIGNAL"]) & np.isfinite(P["sigma"])
                      & (P["sigma"] > 0) & np.isfinite(P["fx"]))
    # "has begun": ever been sizeable.  "has not ended": at or before the last
    # bar the panel carries for it.  Between those two, a closed market stays in
    # the universe.
    began = np.cumsum(sizeable_today, axis=0) >= 1
    has_bar = np.isfinite(P["contc"])
    last_bar = np.where(has_bar.any(axis=0), has_bar.shape[0] - 1
                        - np.argmax(has_bar[::-1], axis=0), -1)
    alive = np.arange(len(dates))[:, None] <= last_bar[None, :]
    in_universe = tradable & began & alive
    started_t = after_start[:, 0]

    # The correlation set and the weights are the UNIVERSE, not today's
    # attendance; `uni` comes back with the estimability screen applied.
    wcw, n_act, n_undef, uni = ewm_corr_sum(P["ret"], in_universe, span,
                                            IDM_MIN_PERIODS)
    act = uni & sizeable_today
    idm = idm_from(wcw)

    # Everything except E_t is knowable up front, so the sequential loop below
    # only has to multiply and accumulate.
    # Equation 3.29 -- the volatility gate, applied to the CONTRACT SIZE rather
    # than folded into the forecast.
    #
    # ARITHMETICALLY THE TWO ARE THE SAME THING HERE, and that was measured
    # rather than assumed: over 495,990 rows, clip(f_smooth . g) and
    # clip(f_smooth) . g differ by 0.000e+00, because the +/-20 clip cannot bind
    # -- |f_smooth| tops out at exactly 20 and g <= 1 only shrinks it.
    #
    # SO THE CHOICE IS ABOUT MEANING, AND THERE ARE THREE REASONS FOR THE SIZE.
    # The gate is a risk statement, not a forecast: it does not change what the
    # strategy thinks the market will do, only how much of that view to carry.
    # Keeping it out of SIGNAL preserves attribution -- "was the forecast
    # right?" stays answerable separately from "did the brake help?".  And
    # decisively, 3.30's drawdown gate is measured on the STRATEGY'S P&L, so it
    # can only exist at this stage; applying both brakes where the position is
    # formed keeps SIGNAL a clean per-instrument forecast in the book instead of
    # forcing it downstream to meet its second multiplier.
    #
    # THE EQUIVALENCE IS NOT PERMANENT.  It holds only while the clip cannot
    # bind.  If 3.26 or 3.27 ever let |f_smooth| exceed 20, the paper's ordering
    # would let a gated-down forecast slip under a cap that would otherwise have
    # caught it, and the two would diverge.
    #
    # A MISSING GATE COUNTS AS 1.0, i.e. no reduction -- it is only missing on a
    # session the instrument did not trade, where `act` zeroes k regardless.
    gvol = (np.where(np.isfinite(P["gvol"]), P["gvol"], 1.0)
            if use_gvol else np.ones_like(P["gvol"]))
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(n_act > 0, 1.0 / n_act, 0.0)[:, None]
        k = (np.where(act, P["SIGNAL"], 0.0) / SIGNAL_PHI) * (
            tau * w * idm[:, None]) / np.where(P["sigma"] > 0, P["sigma"], np.nan)
        k = k * gvol
    k = np.where(np.isfinite(k) & act, k, 0.0)

    # P&L uses the PANAMA-ADJUSTED close: its differences are the true price
    # move with the roll gap already removed, which is the whole reason that
    # series exists.  Raw closes would book every roll as a profit or a loss.
    # ---- OPEN EXECUTION: the session splits in two ------------------------
    #
    # An order decided at the close of t fills at the OPEN of t+1, which is what
    # stage 4's `execute_at` says and what a live desk actually does.  So the
    # session from t to t+1 is held by two different positions:
    #
    #     gap[t+1] = O[t+1] - C[t]     overnight, still on the OLD position
    #     day[t+1] = C[t+1] - O[t+1]   after the fill, on the NEW one
    #
    #     pnl[t+1] = N[t-1].gap[t+1] + N[t].day[t+1]
    #
    # and gap + day == C[t+1] - C[t], so this is a REPARTITION of the same move,
    # not an addition to it.  What it removes is the overnight gap being credited
    # to a position that had not been established yet -- worth `dN . gap`, which
    # over 1990+ is -0.056% of NAV per year and +0.0026 of Sharpe.  Negligible,
    # and NOT the reason for the change: the reason is that the backtest and the
    # order ledger have to describe the same convention before a live fill can be
    # compared against either.  A trend book capturing the overnight gap it never
    # held would flatter itself; measured here it does not, and the sign flips by
    # decade (+0.012%, +0.167%, -0.189%, -0.292%), which is what a wash looks like.
    _C = _ffill(P["contc"])
    _O = _ffill(P["conto"])
    dP = np.diff(_C, axis=0, prepend=np.nan)
    _bar = np.array([[x is not None for x in row] for row in sym], dtype=bool)
    gapP = np.full_like(_C, np.nan)
    gapP[1:] = _O[1:] - _C[:-1]
    dayP = _C - _O
    # A SHUT MARKET CONTRIBUTES EXACTLY ZERO, and the two legs have to be
    # zeroed SEPARATELY rather than left to cancel.  O and C are forward-filled
    # independently, so on a no-bar row they repeat the last real session's
    # open and close: gap and day come out non-zero and equal-and-opposite.
    # Under close-to-close that was invisible -- `diff(ffill(C))` is 0 there and
    # one position multiplied the whole thing.  Split across two positions they
    # no longer cancel, and the instrument books a phantom P&L on a day its
    # market never opened.
    gapP = np.where(_bar, gapP, 0.0)
    dayP = np.where(_bar, dayP, 0.0)
    fx = _ffill(P["fx"])

    # ---- WHICH SESSIONS ARE ROLLS -----------------------------------------
    #
    # Needed by the commission model below, and it has to be computed on each
    # instrument's OWN calendar.  `sym` is null wherever a market had no bar, so
    # comparing adjacent ROWS reads a holiday as two rolls -- one into the null
    # and one out of it.  Stage 4 made exactly this mistake and it hid 578 rolls
    # before anyone noticed, because every symptom was a well-formed number.
    # Forward-filling first makes the comparison one between successive
    # SESSIONS, which is what a roll is defined on.
    symf = np.empty_like(sym)
    _last = np.array([None] * K, dtype=object)
    for _t in range(T):
        _row = sym[_t]
        _last = np.where(np.equal(_row, None), _last, _row)
        symf[_t] = _last
    rolled = np.zeros((T, K), dtype=bool)
    rolled[1:] = (np.not_equal(symf[1:], symf[:-1])
                  & ~np.equal(symf[1:], None) & ~np.equal(symf[:-1], None))

    # WHO HOLDS THE OVERNIGHT GAP, on each instrument's OWN calendar.
    #
    # The order decided at a market's session p fills at the open of its NEXT
    # OWN session q -- so from close(p) to open(q) the book still holds what was
    # decided at p-1.  On the union grid p-1 is NOT `u-2`: a holiday row sits in
    # between and `N` is carried across it, so `N[u-2]` returns the target
    # decided at p and the gap is credited to a position that had already been
    # replaced.  Same union-grid trap that hid 578 rolls in stage 4 and that
    # stage 2's roll detection has to forward-fill around; it is the reason this
    # array exists rather than a `N[t-1]`.
    has_bar = _bar
    gapN = np.zeros((T, K))
    _own_last = np.zeros(K)
    _own_prev = np.zeros(K)
    # COMMISSION WAITS FOR THE MARKET TO OPEN.  A cost decided at t fills at
    # that instrument's OWN next open, which is not t+1 when t+1 is a holiday
    # for it.  `_cost_pending` holds a decided-but-unfilled charge until the
    # instrument actually trades; `cost_chg[t]` is what was charged AT t.
    _cost_pending = np.zeros(K)
    cost_chg = np.zeros((T, K))

    nav = np.full(T, np.nan)
    equity = np.full(T, np.nan)
    pnl = np.zeros(T)
    N = np.zeros((T, K))
    # Equation 3.30 -- the drawdown gate.  Per instrument, per session.
    #
    #     g_dd(i,t) = 1 - (1 - 0.50) . sigma( 10.0 . ( -0.10 - DD64(i,t) ) )
    #
    # DD64 IS THE STRATEGY'S P&L ON THE INSTRUMENT, NOT THE INSTRUMENT'S PRICE.
    # The paper says "the latent drawdown of the last 64 days OF THE STRATEGY ON
    # THE INSTRUMENT", and the design is to back off where the strategy is being
    # beaten -- an adverse regime for our approach, not merely a market that
    # fell.  Those differ by the position's sign: short a market that drops is
    # our best case, and a price-return proxy would cut exposure precisely
    # there.  NOTE the thesis code uses `close.pct_change(64)`, which is that
    # price-return proxy; this deliberately does not.
    #
    # THE DENOMINATOR IS E.w.IDM, WHICH IS WHAT MAKES -0.10 MEAN SOMETHING.  The
    # position is sized so its annualised USD volatility is E.tau.w.IDM;
    # dividing the 64-day P&L by E.w.IDM therefore expresses it in units where
    # the instrument's own annual vol is exactly tau = 0.20, so one quarter's
    # 1-sigma move is tau/2 = 10% -- which IS the trigger.  That is the paper's
    # own rationale ("consumes more than half of its risk budget allocated
    # annually over 3 months") made arithmetic.  Measured on this book the
    # resulting DD64 has sd 9.46%, against the 10% the reasoning predicts;
    # dropping IDM from the denominator gives 29.5% and the trigger would fire
    # on 29% of sessions instead of 7.7%.
    #
    # LAGGED ONE SESSION, per the paper's "shifted by one day to avoid
    # data-snooping".  P&L through t is in fact known when sizing at t, so this
    # is conservatism rather than a correctness fix -- but it is the paper's,
    # and it costs one day of responsiveness.
    #
    # THIS CANNOT BE PRECOMPUTED like g_vol: it reads the running P&L, which
    # depends on the positions, which depend on this gate.  Causal, not
    # circular -- every term is from t-1 or earlier -- but it does mean the
    # multiplication has to happen inside the loop.
    cum = np.zeros((T + 1, K))
    gdd = np.ones((T, K))
    tgt = np.zeros((T, K))
    cost_rt = np.array([tb.cost_rt_of(i) for i in insts],
                       dtype=np.float64) / 2.0
    cost_m = np.zeros((T, K))
    # OVERNIGHT CASH RATE, aligned to this run's grid.  Absent file ->
    # zeros, i.e. no accrual, which is the safe direction: it understates
    # the equity rather than inventing interest that was never earned.
    rf_next = np.zeros(T)
    if IRX_FILE.is_file():
        _ix = pl.read_parquet(IRX_FILE)
        _m = dict(zip(_ix.get_column('date').to_list(),
                      _ix.get_column('rf_accrual_next').to_list()))
        rf_next = np.array([(_m.get(d) or 0.0) for d in dates],
                           dtype=np.float64)
    interest = np.zeros(T)
    E = nav0
    EQ = nav0
    prev = np.zeros(K)
    for t in range(T):
        nav[t] = E
        # `nav` is the SIZING BASE, `equity` is the money.  Under
        # compounding they are the same series; under --fixed-nav the base
        # is pinned and only equity moves.  Both are now NET of costs.
        equity[t] = EQ
        if use_gdd and t >= 2:
            lo = max(0, t - 1 - GDD_LOOKBACK)
            # Zero until a full window exists, matching the reference's
            # fillna(0.0): DD = 0 is "no drawdown", which still carries the
            # standing 0.8655 haircut the standard sigmoid applies at parity.
            span = (cum[t - 1] - cum[lo]) if (t - 1 - lo) == GDD_LOOKBACK else 0.0
            den = E * w[t, 0] * idm[t]
            dd = np.divide(span, den, out=np.zeros(K),
                           where=np.isfinite(den) & (den > 0))
            gdd[t] = 1.0 - (1.0 - GDD_FLOOR) / (
                1.0 + np.exp(-GDD_STEEPNESS * (GDD_THRESHOLD - dd)))
        if E > 0:
            # TRUNCATE TOWARD ZERO, always.  floor() would turn -2.7 into -3 and
            # INCREASE a short; trunc gives -2.  A rounding rule must never take
            # more risk than the formula asked for.
            want = np.trunc(E * k[t] * gdd[t])
            # Equation 3.36 -- the no-trade buffer.
            #
            #   N_exec = N*   if |N* - N_prev| > b.|N_prev|,   b = 0.10
            #          = N_prev  otherwise
            #
            # APPLIED TO N_contracts, AFTER TRUNCATION, which is the only place
            # it can go: the buffer's whole purpose is to leave the EXECUTED
            # position alone, and both branches are already whole contracts, so
            # the result needs no further rounding and cannot drift off an
            # integer.
            #
            # THE BAND IS PROPORTIONAL TO THE POSITION HELD, not to the target,
            # so it scales with the line and vanishes at zero.  That makes the
            # two boundary cases come out right without special-casing:
            # OPENING from flat always executes (b.|0| = 0, so any non-zero
            # target clears it), and CLOSING always executes (|0 - N_prev| =
            # |N_prev| > 0.1|N_prev|).  A buffer that could trap you in a
            # position, or stop you taking one, would be a different and much
            # worse object.
            tgt[t] = want
            if use_buffer:
                want = np.where(np.abs(want - prev) > buffer * np.abs(prev),
                                want, prev)
            # A SHUT MARKET HOLDS WHAT IT HELD.  `act` is false wherever an
            # instrument has no bar, and simply taking `want` there would set the
            # position to zero -- i.e. liquidate the whole line on every holiday
            # and buy it back the next session, in a market that was not open to
            # trade in either direction.  Markets keep different calendars, so
            # this fires somewhere almost every day: on US holidays the entire
            # book went flat and was re-established, and the spurious round trips
            # dominated turnover.  Carry the position instead; the P&L is
            # unaffected because a forward-filled price has a zero difference on
            # exactly those sessions.
            N[t] = np.where(act[t], want, prev)
            # ---- commission, charged where it is actually incurred --------
            #
            #   contracts(i,t) = |N(i,t-1)| + |N(i,t)|     on a roll
            #                    |N(i,t) - N(i,t-1)|       otherwise
            #   cost(i,t)      = contracts . (cost_rt(i) / 2) . FX(i,t)
            #
            # A ROLL IS TWO EXECUTIONS, NOT A CHANGE OF SIZE.  It closes the
            # expiring month and opens the next, and those are different
            # instruments: a September short cannot be netted against a December
            # short, so `|dN|` is the difference of two numbers the market never
            # netted.  The true one-way quantity is both legs.
            #
            # Billing `|dN|` here understated commission by $1.083B over the
            # history, 41.3%, and the average is the mild part -- on 3,924 of
            # 9,377 roll events (41.8%) the new month is sized exactly like the
            # old, `|dN|` is 0, and a full two-leg roll was charged NOTHING.
            # Off a roll the two formulas agree to the cent across 172,760
            # sessions, so this branch is the whole of the correction.  Found by
            # stage 4, which prices every leg because an order ledger has to.
            #
            # ONE-WAY, HENCE THE HALF: `cost_rt` is a full in-and-out, and both
            # branches count contracts changed in one direction, so charging the
            # round trip would bill every position twice over its life.
            # CONVERTED TO USD by the same FX_rate the price uses -- the mapping
            # quotes cost in the contract's own currency, and omitting that
            # understates SJB by 159x while leaving the USD majority right, an
            # error that survives a glance at the total.
            #
            # COMPUTED INSIDE THE LOOP because the cost is now DEDUCTED FROM
            # NAV, and a trade has to be paid for out of equity known before it
            # is made.  Vectorising it afterwards was fine while costs were only
            # reported; it is not, once they compound.
            traded = np.where(rolled[t], np.abs(prev) + np.abs(N[t]),
                              np.abs(N[t] - prev))
            tr = traded * cost_rt * fx[t]
            cost_m[t] = np.where(np.isfinite(tr), tr, 0.0)
            prev = N[t]
        # PAY FOR THE TRADE, THEN EARN THE MOVE.  That ordering is not a
        # convention, it is the only causal one: `nav[t]` above is the equity the
        # position was sized with, and the cost of reaching that position leaves
        # the account before the market pays anything.
        # Roll the own-session lags forward now that N[t] is final: `_own_last`
        # becomes the target that fills at the next open (the day-leg holder,
        # which equals N[t] because N is carried on no-bar rows) and
        # `_own_prev` the one it replaces (the gap-leg holder).
        _b = has_bar[t]
        _own_prev = np.where(_b, _own_last, _own_prev)
        _own_last = np.where(_b, N[t], _own_last)
        if t + 1 < T:
            gapN[t + 1] = _own_prev
        # THE INTEREST BASE IS READ BEFORE THE COST IS PAID.  `cost_m[t]` is
        # decided at t and fills at t+1's OPEN, so it leaves the account at the
        # FAR END of the very window this interest accrues over.  Deducting it
        # first charged the book for cash it demonstrably still held all night:
        # on 2026-01-05 the account had sat flat on 100,000,000 across a 3-day
        # weekend, having never traded and never paid a commission in its life,
        # and was credited interest on 99,917,719.09 -- short by a commission
        # first incurred at Monday's open.  Post-cost is the right base for a
        # window ENDING one session later than the one being credited; the
        # credited window ends here.
        _base = E if compound else EQ
        # THE UNION GRID IS NOT A CALENDAR, for costs any more than for
        # positions.  This used to charge `cost_m[t].sum()` -- every cost
        # decided at t, billed into the step that produces equity[t+1] -- which
        # is right only when t+1 is that instrument's own next session.  When it
        # is not, the money left the account on a day the market was shut, and
        # the per-instrument column lost the charge outright because `keep`
        # drops the row it landed on: 6N's $127.50 on 2026-01-19 vanished from
        # the attribution while NAV still paid it.  Fifth appearance of this
        # trap; same own-session fix `gapN` uses for the position.
        _cost_pending = _cost_pending + cost_m[t]
        if t + 1 < T:
            _fill = np.where(has_bar[t + 1], _cost_pending, 0.0)
            cost_chg[t + 1] = _fill
            _cost_pending = _cost_pending - _fill
            _c = float(_fill.sum())
        else:
            _c = 0.0
        EQ -= _c
        if compound:
            E -= _c
        # OVERNIGHT INTEREST ON 100% OF THE CASH BALANCE, credited at the
        # NEXT session.  Futures need only margin, so the whole balance
        # sits in bills; the base is last night's CLOSING equity -- every
        # commission settled up to and including t is already inside it, and so
        # is t's own P&L, which is what "the NAV of day n-1" means.
        # `rf_accrual_next[t]` already carries the calendar gap,
        # so a weekend earns three days without special-casing here.
        # NO ACCRUAL BEFORE THE BOOK OPENS.  The grid starts in 1978 whatever
        # START_DATE says -- it has to, so the frame aligns with the books, FX
        # and IRX -- but capital that has not been committed yet must not sit
        # there earning the bill rate.  Left ungated, a run starting 2026-01-02
        # arrived at that date with 801M instead of the 100M requested, having
        # compounded $701M of interest on an idle account, and then sized every
        # position off the wrong base.  The 1990 default hid it because the
        # ungated head was only twelve sessions long.
        if t + 1 < T and started_t[t]:
            interest[t + 1] = _base * rf_next[t]
        if t + 1 < T:
            g = ((gapN[t + 1] * gapP[t + 1] + N[t] * dayP[t + 1])
                 * ps * fx[t + 1])
            gi = np.where(np.isfinite(g), g, 0.0)
            cum[t + 1] = cum[t] + gi
            p = float(np.nansum(gi))
            pnl[t + 1] = p
            # COMPOUNDING IS A MODELLING CHOICE, NOT A DETAIL.  With it, sizing
            # is linear in an E_t that grows 12,850x over the history, so the
            # 2020s carry ~13,000x the 1980s' contracts and dominate every
            # aggregate; positions reach multiples of the whole open interest,
            # which the thesis annex A.29/A.30 documents and caps.  Fixed
            # notional sizes every era off the same base, so a Sharpe computed
            # across 45 years is an average over comparable years rather than a
            # statement about the last five.
            EQ += p + interest[t + 1]
            E = (E + p + interest[t + 1]) if compound else nav0
    cost_t = cost_m.sum(axis=1)
    cost_chg_t = cost_chg.sum(axis=1)
    # THE SHIFT IS THE WHOLE POINT OF THIS PAIR OF COLUMNS.
    #
    #   cost_USD[t]      what the trade made at t cost -- the per-session cost,
    #                    charged where it is actually incurred.
    #   cost_lag_USD[t]  the same series shifted one session: the cost that
    #                    established the position which earned pnl_USD[t].
    #
    # A trade is paid for at t-1 and earns its move into t, so the two are not
    # the same number on the same row.  Netting against the UNSHIFTED cost gives
    # an intuitive daily figure that does NOT step the equity, and the gap --
    # exactly the first and last session's cost -- is small enough to read as
    # float noise and be waved through.  Netting against the shifted one gives
    #
    #     equity[t] = equity[t-1] + net_pnl_USD[t]
    #
    # exactly, which is a statement anyone can check on two adjacent rows.  Both
    # columns are kept: `cost_USD` answers "what did we spend today", and
    # `cost_lag_USD` is the one the arithmetic uses.
    # `cost_lag_USD` IS NOW WHAT WAS CHARGED, not a shift of what was decided.
    # On the 98% of sessions where every instrument trades the two are the same
    # series; they part company exactly where a market was shut, which is the
    # case the shift got wrong.
    cost_lag = cost_chg_t
    net_pnl = pnl - cost_lag

    gross_ret = np.zeros(T)
    net_ret = np.zeros(T)
    total_ret = np.zeros(T)
    ok = np.isfinite(nav) & (np.abs(nav) > 0)
    base = np.where(ok[:-1], nav[:-1], 1.0)
    gross_ret[1:] = np.where(ok[:-1], pnl[1:] / base, 0.0)
    net_ret[1:] = np.where(ok[:-1], net_pnl[1:] / base, 0.0)
    # net_ret is the STRATEGY's return and is already an excess return --
    # nothing in it earns the bill rate.  total_ret adds the cash leg, so
    # total_ret - net_ret is exactly the interest.  Sharpe on net_ret is
    # therefore the paper's 'SR excess of IRX' without subtracting anything
    # twice, which is the easy mistake here.
    total_ret[1:] = np.where(ok[:-1],
                             (net_pnl[1:] + interest[1:]) / base, 0.0)
    ret = gross_ret

    # A position the formula wanted but truncation removed.  Counted rather than
    # assumed harmless: it is the mechanism by which a too-small NAV silently
    # drops the expensive half of the universe.
    wanted = np.abs(nav[:, None] * k) >= 1e-12
    floored = int(np.sum(wanted & (N == 0) & act))

    # NOTIONAL AND TURNOVER PRICE OFF THE RAW CLOSE, NOT THE PANAMA SERIES.
    #
    # Panama is anchored at the present, so accumulated roll gaps drive the
    # ADJUSTED close negative in early history -- 14 of the 63 books touch or
    # cross zero, CL bottoming at -29.11 against a raw low of 10.42.  Multiplying
    # |N| by that produced NEGATIVE notionals on 39,585 instrument-sessions and
    # a negative mean Gross/NAV for the whole OilGas class, which is what
    # exposed it.
    #
    # The distinction is not cosmetic: the P&L is RIGHT to use the Panama series,
    # because its DIFFERENCES are the true price move with the roll removed.  But
    # notional is a LEVEL, and a back-adjusted level is not a price anyone could
    # transact at.  Differences from Panama, levels from raw.
    px = _ffill(P["raw"])
    notional = np.abs(N) * px * ps * fx
    # TURNOVER IS THE COST THE BACKTEST CANNOT SEE.  No commission or slippage
    # is modelled here, so two configurations can post the same Sharpe while one
    # trades far more to get it.  Contracts actually changed, priced at the
    # session they were changed on, is the honest denominator for that.
    dN = np.abs(np.diff(N, axis=0, prepend=0.0))
    traded = np.nansum(np.where(np.isfinite(dN * px * ps * fx),
                                dN * px * ps * fx, 0.0), axis=1)
    def _pnl_i(kk: int) -> np.ndarray:
        """One instrument's P&L for sessions 1..T-1, on the open-execution split.

        THE SAME TWO LEGS AS THE PORTFOLIO LOOP, per instrument: the overnight
        gap belongs to N[t-1] because the order had not filled yet, and the rest
        of the session to N[t].  Written once here so the per-instrument frames
        cannot drift from the aggregate -- the reconciliation ties them together
        and would catch it, but a shared expression is cheaper than a check.
        """
        # v[j] is session j+1.  The gap leg needs N[j-1] (two lags from j+1),
        # the day leg N[j].  Writing the first as `N[:-1]` shifted once gives
        # N[j] for BOTH and silently reproduces the old close-to-close formula
        # -- which is what the first draft of this did, leaving the instrument
        # frames on one convention and the portfolio loop on the other.
        n_day = N[:-1, kk]                                  # N[j]
        n_gap = gapN[1:, kk]                                # own-session lag
        z = lambda x: np.where(np.isfinite(x), x, 0.0)
        g = z(n_gap * gapP[1:, kk] * ps[kk] * fx[1:, kk])
        d = z(n_day * dayP[1:, kk] * ps[kk] * fx[1:, kk])
        return g + d, g, d

    frames = {}
    for kk, i in enumerate(insts):
        # ROWS THAT CARRY P&L MUST BE KEPT, and under open execution that is a
        # wider set than it was.  Session u's P&L now depends on N[u-2] as well
        # as N[u-1] -- the gap leg belongs to the position held before the fill
        # -- so a session with a null SIGNAL and a flat position today can still
        # have earned money on a position held two sessions ago.  Dropping those
        # rows leaves the per-instrument frames short of the portfolio total,
        # which is exactly what reconciliation ties A and B caught.
        _tot, _gp, _dy = _pnl_i(kk)
        _pv = np.concatenate([[0.0], _tot])
        # EVERY SESSION THIS MARKET TRADED IS KEPT, and that is now a contract
        # with the consumers rather than a convenience.  Open execution made
        # three downstream objects depend on an OWN-SESSION lag -- stage 4's
        # realised P&L, its verifier, and reconciliation tie A -- and a lag can
        # only be read off a file whose consecutive bar rows are consecutive
        # sessions.  Dropping a flat, signal-less bar row silently turns `k-2`
        # into "two rows back", which is the union-grid trap one level down.
        keep = (has_bar[:, kk] | np.isfinite(P["SIGNAL"][:, kk])
                | (N[:, kk] != 0) | (_pv != 0.0))
        frames[i] = pl.DataFrame({
            "date": [dates[t] for t in range(T) if keep[t]],
            "symbol": [sym[t, kk] for t in range(T) if keep[t]],
            "SIGNAL": P["SIGNAL"][keep, kk],
            "price_vol_USD_ann": P["sigma"][keep, kk],
            # Carried through from the book so the position file shows every
            # term of 3.32 side by side: N_raw is reproducible from this row
            # alone, gate included, without opening the trading book.
            "s_g_vol": P["gvol"][keep, kk],
            "s_g_dd": gdd[keep, kk],
            "w_i": np.where(act[keep, kk], w[keep, 0], np.nan),
            "IDM": idm[keep],
            "NAV": nav[keep],
            # `sized` SEPARATES A DECISION FROM A CARRY.  On a session where
            # this market was shut, 3.32 was not evaluated and the position is
            # yesterday's; N_raw is null there rather than 0, because 0 would
            # claim the formula asked for a flat position when it was never
            # asked at all.  Anything reconciling N_contracts against N_raw --
            # the rounding check in Update.py, for one -- must read this first.
            "tradable": in_universe[keep, kk],
            "sized": act[keep, kk],
            # MUST INCLUDE g_dd.  It is applied inside the loop rather than
            # folded into `k`, so writing `nav * k` here would emit a number
            # that is not the position before truncation -- and would do so
            # silently, since g_dd < 1 keeps |N| <= |N_raw| and the rounding
            # check would still pass, for the wrong reason.
            "N_raw": np.where(act[:, kk], nav * k[:, kk] * gdd[:, kk],
                              np.nan)[keep],
            # N_raw -> N_target -> N_contracts is the full chain:
            # 3.32 with both gates, truncated toward zero, then 3.36.
            "N_target": tgt[keep, kk],
            "N_contracts": N[keep, kk],
            "notional_USD": notional[keep, kk],
            "pnl_USD": _pv[keep],
            # The two legs of the open-execution split.  gap belongs to the
            # position held BEFORE the fill, day to the one held after; on a
            # roll they are different delivery months.
            "pnl_gap_USD": np.concatenate([[0.0], _gp])[keep],
            "pnl_day_USD": np.concatenate([[0.0], _dy])[keep],
            "cost_USD": cost_m[keep, kk],
            "cost_lag_USD": cost_chg[keep, kk],
            "net_pnl_USD": (_pv - cost_chg[:, kk])[keep],
            "cum_cost_USD": np.cumsum(cost_m[:, kk])[keep],
        })
    port = pl.DataFrame({
        "date": dates,
        # THE RUN'S START, RECORDED IN THE RUN'S OUTPUT.  `--start-date` changes
        # what every other column means -- before it the book holds nothing and
        # earns no interest -- and a parameter that changes the meaning of an
        # artifact without appearing in it is a trap for everything downstream.
        # The reconciliation walked straight into it: reading the module default
        # while the run used 2026-01-02, it accrued 36 years of bill yield on an
        # idle $100M and reported a 97% break in a column that was correct.
        "started": started_t,
        "n_active": n_act,
        "wCw": wcw,
        "IDM": idm,
        "n_undefined_pairs": n_undef,
        "NAV": nav,
        "equity_USD": equity,
        "pnl_USD": pnl,
        "cost_USD": cost_t,
        "cost_lag_USD": cost_lag,
        "net_pnl_USD": net_pnl,
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        # Same shift discipline as cost_lag_USD.  `rf_accrual_next[t]` is the
        # rate STARTING at t; `rf_accrual_applied[t]` is the one that actually
        # produced this row's interest, i.e. the previous session's.  Without
        # it a reader comparing interest against the rate on the same row sees
        # sign disagreements that are not there -- which is exactly what the
        # first version of the verification did.
        "rf_accrual_next": rf_next,
        "rf_accrual_applied": np.concatenate([[0.0], rf_next[:-1]]),
        "interest_USD": interest,
        "total_ret": total_ret,
        "gross_notional_USD": np.nansum(np.where(np.isfinite(notional),
                                                 notional, 0.0), axis=1),
        "n_positions": (N != 0).sum(axis=1).astype(np.float64),
        "traded_notional_USD": traded,
    })
    return frames, port, floored


def stats(port: pl.DataFrame, label: str) -> dict:
    r = port.get_column("gross_ret").to_numpy()
    nr = (port.get_column("net_ret").to_numpy()
          if "net_ret" in port.columns else r)
    nav = port.get_column("NAV").to_numpy()
    idm = port.get_column("IDM").to_numpy()
    # MEASURED OVER THE TRADED PERIOD, NOT THE WHOLE GRID.  The frame spans
    # 1978 so that it aligns with the books, FX and IRX -- the cross-stage
    # checks assert exactly that -- but nothing is held before START_DATE, and
    # twelve years of zero returns would drag every statistic toward zero while
    # looking like a longer, calmer sample.  The window opens at the first
    # session that actually carries a position.
    npos = (port.get_column("n_positions").to_numpy()
            if "n_positions" in port.columns else np.ones(len(r)))
    held = np.flatnonzero(npos > 0)
    t0 = int(held[0]) if len(held) else 1
    live = np.isfinite(r) & (np.arange(len(r)) >= max(t0, 1))
    rr = r[live]
    ann = float(np.std(rr, ddof=0) * math.sqrt(tb_days()))
    mu = float(np.mean(rr) * tb_days())
    # DRAWDOWN IS MEASURED ON EQUITY, NOT ON THE SIZING BASE.  Under fixed
    # notional `NAV` is constant by construction, so a peak-to-trough on it is
    # identically zero -- which is what this reported before, a 0.0% max
    # drawdown on a strategy that plainly has them.  Equity is the money and is
    # the only series a drawdown means anything on; under compounding the two
    # coincide, which is exactly why the bug hid.
    eq = (port.get_column("equity_USD").to_numpy()
          if "equity_USD" in port.columns else nav)
    dd = eq / np.maximum.accumulate(np.where(np.isfinite(eq), eq, -np.inf))
    tr = port.get_column("traded_notional_USD").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        turn = np.where(np.isfinite(nav) & (nav > 0), tr / nav, np.nan)
    n_yr = live.sum() / tb_days()
    e0 = eq[max(t0 - 1, 0)]
    cagr = (((eq[-1] / e0) ** (1.0 / n_yr) - 1.0)
            if n_yr > 0 and e0 > 0 else float("nan"))
    return {
        "label": label,
        "sessions": int(live.sum()),
        "t0": t0,
        "NAV_end": float(eq[-1]),
        # `ann_ret` is the MEAN PERIODIC RETURN annualised, on whatever base was
        # used for sizing.  Under compounding that base is equity, so it is a
        # growth rate; under fixed notional it is return on a constant capital
        # base and is NOT a growth rate -- `cagr` is, and the two diverge widely
        # there (20.7% against 5.5%).  Both are reported so neither is mistaken
        # for the other.
        "ann_ret": mu,
        "net_ann_ret": float(np.mean(nr[live]) * tb_days()),
        # NET VOL IS MEASURED ON NET RETURNS.  It used to reuse `ann` -- the
        # GROSS volatility -- under a label that said "vol" on the NET line, so
        # the report stated a number it had not computed and the net Sharpe
        # divided by the wrong denominator.  The gap is small (8.2188% against
        # 8.2098%, 0.0012 of Sharpe) which is precisely why it survived: a
        # figure that is wrong by a rounding error is never caught by reading it.
        "net_ann_vol": float(np.std(nr[live], ddof=0) * math.sqrt(tb_days())),
        "net_sharpe": (float(np.mean(nr[live]) * tb_days())
                       / float(np.std(nr[live], ddof=0) * math.sqrt(tb_days())))
                      if float(np.std(nr[live], ddof=0)) else float("nan"),
        "cost_ann": cost_ann(r, nr, live),
        # The cash leg, reported so the NET line's "excess of IRX" is legible
        # rather than something a reader has to take on trust.
        "rf_ann": (float(np.mean(
            (port.get_column("total_ret").to_numpy() - nr)[live]) * tb_days())
            if "total_ret" in port.columns else float("nan")),
        "cagr": cagr,
        "ann_vol": ann,
        "sharpe": mu / ann if ann else float("nan"),
        "max_dd": float(1.0 - np.nanmin(dd[np.isfinite(dd)])) if len(dd) else float("nan"),
        # SCOPED TO THE TRADED WINDOW, like every other statistic here.  These
        # four used to average over the WHOLE grid, which for a late start is
        # mostly sessions the book never traded, where IDM sits at its 1.0
        # identity and turnover is zero.  A run beginning 2026-01-02 reported an
        # IDM mean of 1.029 while the multiplier it actually sized with was
        # 3.125 -- and 1.029 reads as a cold correlation estimator, which it is
        # NOT: `ewm_corr_sum` accumulates over the entire return matrix whatever
        # the start date, so the traded-window IDM is bit-identical to the
        # 1990-start run's on the same sessions.  The number was a reporting
        # artifact, and it cost an argument about a warm-up that never happened.
        "idm_mean": float(np.nanmean(idm[live])),
        "idm_sd": float(np.nanstd(idm[live])),
        "idm_dod": float(np.nanmean(np.abs(np.diff(idm[live])))),
        "idm_at_cap": float(np.mean(np.isclose(idm[live], IDM_CAP))),
        "turnover": float(np.nanmean(turn[live]) * tb_days()),
    }


def tb_days() -> int:
    return 256


def cost_ann(gross_ret, net_ret, live=None) -> float:
    """Commission as an annualised fraction of the sizing base.

    THE ONE DEFINITION. It lived here and, separately, in `publish.py`, which
    divided by `NAV[t]` where this divides by `NAV[t-1]`. 1.4938% against
    1.4922% -- invisible at the two decimals the site prints, which is exactly
    why it sat there. Two implementations of one formula is the shape that has
    already cost this pipeline twice: reconciliation tie F agreed with stage 3
    for as long as both encoded the same wrong interest base, and the summary
    box compared an arithmetic rate against a geometric one on a different
    basis. Stage 6 now calls this rather than restating it.

    `NAV[t-1]`, NOT `NAV[t]`, because that is the base `gross_ret` and `net_ret`
    already use, and it is what makes the decomposition close:

        mean(gross_ret) x 256  -  cost_ann  ==  net_ann_ret

    exactly, by linearity. Divide by `NAV[t]` and it stops closing -- a
    reconciliation a reader can do by eye stops working for a reason that is
    not visible on the page.
    """
    g = np.asarray(gross_ret, dtype=float)
    n = np.asarray(net_ret, dtype=float)
    if live is not None:
        g, n = g[live], n[live]
    return float(np.mean(g - n) * tb_days())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", type=float, default=NAV_0)
    ap.add_argument("--tau", type=float, default=TAU,
                    help="risk budget: annualised volatility the "
                         "PORTFOLIO targets (default 0.20)")
    ap.add_argument("--idm-span", type=int, default=IDM_CORR_SPAN)
    ap.add_argument("--compare-spans", default=None,
                    help="comma-separated spans to score against each other")
    ap.add_argument("--no-gvol", action="store_true",
                    help="ablation: disable the 3.29 volatility gate")
    ap.add_argument("--no-gdd", action="store_true",
                    help="ablation: disable the 3.30 drawdown gate")
    ap.add_argument("--no-buffer", action="store_true",
                    help="ablation: disable the 3.36 no-trade buffer")
    ap.add_argument("--start-date", default=START_DATE,
                    help="first session the book may hold anything "
                         "(default 1990-01-01)")
    ap.add_argument("--buffer", type=float, default=BUFFER,
                    help="3.36 band width b (default 0.10)")
    ap.add_argument("--fixed-nav", action="store_true",
                    help="size off a constant NAV -- no compounding")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    tb = _load(BOOK_PY, "tb")
    insts = sorted(p.stem for p in tb.BOOK.glob("*.csv"))
    if not insts:
        raise SystemExit(f"[ABORT] no books in {tb.BOOK}; run stage 2 first")
    t0 = time.time()
    print(f"portfolio -> {HERE}    NAV {args.nav:,.0f}   tau {args.tau:.0%}"
          f"   {'FIXED notional' if args.fixed_nav else 'compounding'}"
          f"   from {args.start_date}")
    print(f"  loading {len(insts)} books ...", flush=True)
    dates, P, sym = panel(tb, insts)
    print(f"  panel {len(dates):,} sessions x {len(insts)} instruments "
          f"({time.time() - t0:.0f}s)", flush=True)

    if args.compare_spans:
        spans = [int(x) for x in args.compare_spans.split(",")]
        rows = []
        for s in spans:
            t1 = time.time()
            _f, port, fl = simulate(dates, P, sym, insts, tb, args.nav, s,
                                    args.tau, not args.fixed_nav,
                                    not args.no_gvol, not args.no_gdd,
                                    not args.no_buffer, args.buffer,
                                    args.start_date)
            st = stats(port, f"span {s}")
            st["floored"] = fl
            st["secs"] = time.time() - t1
            rows.append(st)
            print(f"  span {s:<5} done in {st['secs']:.0f}s", flush=True)
        hdr = (f"\n{'IDM span':<10}{'mean':>8}{'sd':>8}{'|d/day|':>9}"
               f"{'at cap':>8}{'turnover':>10}{'ann vol':>9}{'ann ret':>9}"
               f"{'Sharpe':>8}{'max DD':>8}")
        print(hdr)
        print("-" * (len(hdr) - 1))
        for st in rows:
            print(f"{st['label']:<10}{st['idm_mean']:>8.3f}{st['idm_sd']:>8.3f}"
                  f"{st['idm_dod']:>9.4f}{st['idm_at_cap']:>8.1%}"
                  f"{st['turnover']:>9.1f}x{st['ann_vol']:>9.2%}"
                  f"{st['ann_ret']:>9.2%}{st['sharpe']:>8.2f}"
                  f"{st['max_dd']:>8.1%}")
        return 0

    frames, port, floored = simulate(dates, P, sym, insts, tb, args.nav,
                                     args.idm_span, args.tau,
                                     not args.fixed_nav, not args.no_gvol,
                                     not args.no_gdd, not args.no_buffer,
                                     args.buffer, args.start_date)
    st = stats(port, "run")
    print(f"\n  IDM      mean {st['idm_mean']:.3f}   sd {st['idm_sd']:.3f}   "
          f"at cap {st['idm_at_cap']:.1%}")
    eq = port.get_column("equity_USD")[-1]
    d0 = port.get_column("date")[st["t0"]]
    print(f"  traded   {d0} .. {port.get_column('date')[-1]}"
          f"   {st['sessions']:,} sessions"
          f"   ({st['sessions']/256:.1f} years)")
    print(f"  equity   {args.nav:,.0f} -> {eq:,.0f}"
          f"   ({eq/args.nav:,.1f}x)")
    print(f"  gross    ret {st['ann_ret']:>7.2%}   vol {st['ann_vol']:>7.2%}"
          f"   Sharpe {st['sharpe']:.3f}")
    print(f"  costs        {st['cost_ann']:>7.2%} of NAV per year")
    # "EXCESS OF IRX" IS SAID OUT LOUD because the alternative is that a reader
    # assumes the cash leg is included -- which is the natural assumption when a
    # pipeline goes to the trouble of modelling the bill rate -- and then
    # subtracts IRX again to "correct" it.  net_ret is trading P&L after
    # commission and contains no interest, so it IS the excess return; the cash
    # leg is in total_ret.  Checked, not just claimed: `verify_portfolio`
    # asserts Sharpe(net_ret) == Sharpe(total_ret - rf).
    print(f"  NET      ret {st['net_ann_ret']:>7.2%}   vol {st['net_ann_vol']:>7.2%}"
          f"   Sharpe {st['net_sharpe']:.3f}   (excess of IRX)")
    print(f"  interest     {st['rf_ann']:>7.2%} of NAV per year, earned on cash "
          f"and NOT in NET above")
    # MAX DD GETS ITS OWN LINE because it belongs to neither of the two above:
    # it is measured on `equity_USD`, which is after commission AND after
    # interest.  Printed on the gross line it read as a gross drawdown, which
    # it has never been.
    print(f"  max DD       {st['max_dd']:>7.1%} on equity "
          f"(after costs, after interest)")
    print(f"  CAGR     {st['cagr']:>7.2%}     (equity growth rate; differs from "
          f"ann ret under fixed notional)")
    und = float(port.get_column("n_undefined_pairs").sum())
    print(f"  undefined correlation pairs among active markets: {und:,.0f}")
    print(f"  positions truncated to zero: {floored:,}")

    if not args.no_write:
        POS.mkdir(parents=True, exist_ok=True)
        for i, f in frames.items():
            f.write_csv(POS / f"{i}.csv")
            f.write_parquet(POS / f"{i}.parquet")
        port.write_csv(PORTFOLIO)
        port.write_parquet(PORTFOLIO.with_suffix(".parquet"))
        print(f"\n  wrote {len(frames)} position files -> {POS}")
        print(f"  wrote {PORTFOLIO.name}   {port.height:,} sessions")
    print(f"  {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
