"""
Strategy 183 -- Capacity Analysis
=================================
Estimates the AUM breakpoint at which market-impact costs erode the
strategy's alpha by 50 bps of Sharpe Ratio.

Methodology:
  1. For each instrument, compute the front-month Average Daily Volume (ADV)
     over the last 5 years of available contract data.
  2. From the checkpoint, reconstruct the average absolute position per
     instrument (in contracts) at the backtest's starting capital ($100M).
  3. Scale positions linearly for various AUM levels and estimate market
     impact via the square-root model:
         impact_bps = k * sqrt(participation_rate)
     where k ~ 10 bps (conservative for liquid futures).
  4. Find the capacity breakpoint where net SR drops 50 bps from gross SR.
"""

import sys
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# -- Paths ----------------------------------------------------------------
IG_DIR = Path(__file__).resolve().parent.parent  # IG_Backtest/
sys.path.insert(0, str(IG_DIR))

from ig_shared_config import (
    get_project_paths, load_mapping,
    STARTING_CAPITAL, VOL_TARGET, FORECAST_TARGET, ANNUALISE_DAILY,
    idm_from_n_active,
)

CHECKPOINT_PATH = IG_DIR / "Strategy_183" / "Strategy_183_IG_VoV_Quad_Sharpened_checkpoint.pkl"
CONTRACTS_DIR   = IG_DIR.parent.parent / "Data" / "Contracts"
PANAMA_DIR      = IG_DIR.parent.parent / "Data" / "PanamaMethod"

# -- FX rates: currency -> USD multiplier ----------------------------------
# For USD-denominated instruments, fx = 1.0.
# For non-USD, we use approximate recent spot rates (capacity analysis is
# order-of-magnitude; precise daily FX alignment is unnecessary).
CURRENCY_TO_USD = {
    "USD": 1.0,
    "CAD": 0.74,
    "EUR": 1.08,
    "GBP": 1.27,
    "JPY": 0.0067,
    "AUD": 0.66,
    "CHF": 1.13,
    "HKD": 0.128,
    "NZD": 0.61,
    "MXN": 0.058,
    "SEK": 0.096,
}

# -- Load checkpoint -------------------------------------------------------
with open(CHECKPOINT_PATH, "rb") as f:
    cp = pickle.load(f)

GROSS_SR    = cp["sr"]
ANN_VOL     = cp["ann_vol"]
ANN_RET     = cp["ann_ret"]
N_YEARS     = cp["n_years"]
INSTRUMENTS = list(cp["per_inst_pnl"].columns)

paths   = get_project_paths(Path("Strategy_183"))
mapping = load_mapping(paths["mapping"])

print(f"Gross Sharpe Ratio : {GROSS_SR:.3f}")
print(f"Annualised Vol     : {ANN_VOL:.3%}")
print(f"Instruments        : {len(INSTRUMENTS)}")
print(f"Starting Capital   : ${STARTING_CAPITAL:,.0f}")
print()

# =========================================================================
# STEP 1: Front-month ADV per instrument (last 5 years)
# =========================================================================
print("=" * 70)
print("STEP 1: Computing front-month ADV per instrument (last 5 years)")
print("=" * 70)

cutoff_date = pd.Timestamp("2021-01-01")


def compute_front_month_adv(instrument):
    """Front-month = max-volume contract each day.  Return mean ADV."""
    inst_dir = CONTRACTS_DIR / instrument
    if not inst_dir.exists():
        return np.nan

    frames = []
    for csv_file in inst_dir.glob(f"{instrument}-*.csv"):
        try:
            df = pd.read_csv(csv_file, parse_dates=["Date"], date_format="%Y%m%d")
            df = df[["Date", "Volume"]].copy()
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return np.nan

    all_data = pd.concat(frames, ignore_index=True)
    all_data = all_data[all_data["Date"] >= cutoff_date]
    if all_data.empty:
        return np.nan

    front_vol = all_data.groupby("Date")["Volume"].max()
    front_vol = front_vol[front_vol > 0]
    return front_vol.mean() if len(front_vol) > 0 else np.nan


adv_dict = {}
for inst in INSTRUMENTS:
    adv_dict[inst] = compute_front_month_adv(inst)
    adv = adv_dict[inst]
    tag = f"ADV = {adv:>12,.0f}" if not np.isnan(adv) else "ADV = N/A"
    print(f"  {inst:<10s}  {tag}")

print()

# =========================================================================
# STEP 2: Average position per instrument at base capital ($100M)
# =========================================================================
print("=" * 70)
print("STEP 2: Estimating average position size & contract values")
print("=" * 70)

n_active = len(INSTRUMENTS)
idm      = idm_from_n_active(n_active)
print(f"  n_active = {n_active}, IDM = {idm:.2f}")
print(f"  Vol target = {VOL_TARGET:.0%}, Forecast target = {FORECAST_TARGET:.0f}")


def load_panama(instrument):
    """Load panama-adjusted continuous close for an instrument."""
    for suffix in ["_continuous.csv", "_continuous_panama.parquet"]:
        p = PANAMA_DIR / f"{instrument}{suffix}"
        if p.exists():
            if str(p).endswith(".parquet"):
                return pd.read_parquet(p)
            else:
                df = pd.read_csv(p, parse_dates=["Date"])
                return df.set_index("Date")
    return None


def fx_rate(instrument):
    """Return USD-per-local-currency for an instrument."""
    ccy = mapping.loc[instrument, "currency"] if instrument in mapping.index else "USD"
    return CURRENCY_TO_USD.get(ccy, 1.0)


avg_pos_base    = {}    # average absolute position in contracts at $100M
contract_val_usd = {}   # average contract notional value in USD

for inst in INSTRUMENTS:
    df = load_panama(inst)
    if df is None:
        avg_pos_base[inst] = np.nan
        contract_val_usd[inst] = np.nan
        continue

    close = df["Close"].dropna()
    recent = close[close.index >= cutoff_date]
    if len(recent) < 100:
        avg_pos_base[inst] = np.nan
        contract_val_usd[inst] = np.nan
        continue

    ps  = mapping.loc[inst, "pointsize"] if inst in mapping.index else 1.0
    fx  = fx_rate(inst)

    # Contract notional value in USD
    avg_price = recent.mean()
    cv_usd = abs(avg_price) * ps * fx
    contract_val_usd[inst] = cv_usd

    # Annualised volatility of one contract in USD
    daily_vol = recent.diff().dropna().std()
    ann_vol_contract = daily_vol * ps * ANNUALISE_DAILY * fx

    if ann_vol_contract < 1.0:
        avg_pos_base[inst] = np.nan
        continue

    # Vol-targeting: position = risk_budget / ann_vol_per_contract
    # Average |forecast| ~ forecast_target, so forecast/target ~ 1
    risk_budget = STARTING_CAPITAL * VOL_TARGET * (1.0 / n_active) * idm
    avg_pos_base[inst] = risk_budget / ann_vol_contract

risk_budget_per_inst = STARTING_CAPITAL * VOL_TARGET * (1.0 / n_active) * idm
print(f"  Risk budget per instrument: ${risk_budget_per_inst:,.0f}")
valid_insts = [i for i in INSTRUMENTS if not np.isnan(avg_pos_base[i])]
print(f"  Valid instruments: {len(valid_insts)} / {len(INSTRUMENTS)}")
print()

# =========================================================================
# STEP 3: Market impact across AUM levels
# =========================================================================
print("=" * 70)
print("STEP 3: Market impact analysis across AUM levels")
print("=" * 70)

AUM_LEVELS   = [100_000_000, 250_000_000, 500_000_000, 1_000_000_000, 2_000_000_000, 5_000_000_000]
K_IMPACT     = 10.0     # bps, conservative for liquid futures
TRADING_DAYS = 256

# Daily turnover fraction.
# trades_yr counts trade *events* (days where position changed), NOT the
# fraction of position turned over.  For a Carver-style buffered CTA:
#   - Trade events per instrument per day: ~0.30
#   - Average position change per event:   ~10-15% of position
#   - Effective daily turnover:            ~3-5%
#
# Cross-check: total commission = 2.23% of avg equity/yr.
# With avg cost_rt ~ $98 and avg contract value ~$200K, cost/notional ~ 0.05%.
# Turnover = 2.23% / 0.05% = ~45x notional.  But that includes 62 instruments,
# so per-instrument annual turnover = 45/62 ~ 0.7x, daily = 0.7/256 = 0.003.
# That seems too LOW because it uses raw cost_rt which is dominated by cheap
# instruments.
#
# Most robust estimate: use the standard CTA daily turnover assumption of ~5%
# (one-way), consistent with Pedersen (2015) and AQR research on managed futures.
# This corresponds to ~12-13x annual one-way turnover.
DAILY_TURNOVER_FRAC = 0.05
print(f"  Daily turnover fraction: {DAILY_TURNOVER_FRAC:.0%} of position (standard CTA assumption)")
print()


def compute_impact_for_aum(aum):
    """Return (details_list, total_annual_impact_usd) for a given AUM."""
    scale = aum / STARTING_CAPITAL
    details = []
    total_cost = 0.0

    for inst in INSTRUMENTS:
        pos_base = avg_pos_base.get(inst, np.nan)
        adv      = adv_dict.get(inst, np.nan)
        cv       = contract_val_usd.get(inst, np.nan)

        if any(np.isnan(x) for x in [pos_base, adv, cv]) or adv < 1 or cv < 1:
            continue

        pos_scaled = pos_base * scale

        # Daily contracts traded = position * daily turnover fraction
        daily_traded = pos_scaled * DAILY_TURNOVER_FRAC

        # Participation rate
        participation = daily_traded / adv

        # Square-root impact model
        impact_bps = K_IMPACT * np.sqrt(participation)

        # Annual impact cost for this instrument
        annual_traded = daily_traded * TRADING_DAYS
        cost_per_contract = (impact_bps / 10_000) * cv
        annual_cost = annual_traded * cost_per_contract
        total_cost += annual_cost

        details.append({
            "instrument":        inst,
            "pos_contracts":     pos_scaled,
            "adv":               adv,
            "participation_pct": participation * 100,
            "impact_bps":        impact_bps,
            "annual_cost":       annual_cost,
        })

    return details, total_cost


results = []
for aum in AUM_LEVELS:
    details, total_cost = compute_impact_for_aum(aum)
    drag_pct  = total_cost / aum * 100 if aum > 0 else 0
    drag_sr   = (total_cost / aum) / ANN_VOL if ANN_VOL > 0 else 0
    net_sr    = GROSS_SR - drag_sr

    results.append({
        "aum":                  aum,
        "avg_pos":              np.mean([d["pos_contracts"] for d in details]) if details else 0,
        "avg_participation_pct": np.mean([d["participation_pct"] for d in details]) if details else 0,
        "max_participation_pct": max(d["participation_pct"] for d in details) if details else 0,
        "avg_impact_bps":       np.mean([d["impact_bps"] for d in details]) if details else 0,
        "max_impact_bps":       max(d["impact_bps"] for d in details) if details else 0,
        "annual_impact_cost":   total_cost,
        "impact_drag_pct":      drag_pct,
        "net_sr":               net_sr,
        "details":              details,
    })

# =========================================================================
# STEP 4: Capacity breakpoint (net SR drops by 0.50 from gross)
# =========================================================================
print("=" * 70)
print("STEP 4: Capacity breakpoint (net SR drops by 0.50 from gross)")
print("=" * 70)

target_sr = GROSS_SR - 0.50

aum_grid = np.linspace(50_000_000, 10_000_000_000, 1000)
sr_grid  = np.empty(len(aum_grid))

for j, aum in enumerate(aum_grid):
    _, total_cost = compute_impact_for_aum(aum)
    sr_grid[j] = GROSS_SR - (total_cost / aum) / ANN_VOL

breakpoint_aum = None
for j in range(len(sr_grid) - 1):
    if sr_grid[j] >= target_sr > sr_grid[j + 1]:
        frac = (target_sr - sr_grid[j]) / (sr_grid[j + 1] - sr_grid[j])
        breakpoint_aum = aum_grid[j] + frac * (aum_grid[j + 1] - aum_grid[j])
        break

if breakpoint_aum is None:
    if sr_grid[-1] >= target_sr:
        breakpoint_aum = float("inf")
        print(f"  Net SR stays above {target_sr:.3f} even at $10B AUM")
    else:
        breakpoint_aum = aum_grid[0]
        print(f"  Net SR already below {target_sr:.3f} at ${aum_grid[0]/1e6:.0f}M")
else:
    print(f"  Capacity breakpoint: ${breakpoint_aum/1e9:.2f}B")
    print(f"  (Net SR drops from {GROSS_SR:.3f} to {target_sr:.3f})")

# =========================================================================
# Output tables
# =========================================================================
print()
print("=" * 94)
print("  CAPACITY ANALYSIS -- Strategy 183 (IG VoV Quad Sharpened)")
print("=" * 94)
print(f"  Gross Sharpe Ratio: {GROSS_SR:.3f}  |  Ann. Vol: {ANN_VOL:.2%}  |  k (impact): {K_IMPACT:.0f} bps")
print(f"  Daily turnover: {DAILY_TURNOVER_FRAC:.0%} of position  |  {len(INSTRUMENTS)} instruments")
print(f"  Model: impact_bps = {K_IMPACT:.0f} * sqrt(daily_contracts_traded / ADV)")
print("-" * 94)

header = (f"{'AUM':>12s}  {'Avg Pos':>10s}  {'Avg %ADV':>10s}  {'Max %ADV':>10s}  "
          f"{'Avg Impact':>12s}  {'Ann. Cost':>14s}  {'Drag':>8s}  {'Net SR':>8s}")
print(header)
print("-" * 94)

for r in results:
    aum_str     = f"${r['aum']/1e6:,.0f}M"
    pos_str     = f"{r['avg_pos']:,.1f}"
    avg_adv_str = f"{r['avg_participation_pct']:.3f}%"
    max_adv_str = f"{r['max_participation_pct']:.3f}%"
    impact_str  = f"{r['avg_impact_bps']:.2f} bps"
    cost_str    = f"${r['annual_impact_cost']/1e6:,.1f}M"
    drag_str    = f"{r['impact_drag_pct']:.2f}%"
    sr_str      = f"{r['net_sr']:.3f}"
    print(f"{aum_str:>12s}  {pos_str:>10s}  {avg_adv_str:>10s}  {max_adv_str:>10s}  "
          f"{impact_str:>12s}  {cost_str:>14s}  {drag_str:>8s}  {sr_str:>8s}")

print("-" * 94)

if breakpoint_aum == float("inf"):
    bp_str = "> $10.0B"
else:
    bp_str = f"${breakpoint_aum/1e9:.2f}B"
print(f"  Capacity breakpoint (SR -0.50): {bp_str}")
print()

# -- Top-10 constrained instruments at $1B --------------------------------
print("=" * 70)
print("  TOP 10 CAPACITY-CONSTRAINED INSTRUMENTS (at $1B AUM)")
print("=" * 70)

r_1b = [r for r in results if r["aum"] == 1_000_000_000]
if r_1b:
    details = sorted(r_1b[0]["details"],
                     key=lambda x: x["participation_pct"], reverse=True)
    print(f"{'Instrument':>12s}  {'Position':>10s}  {'ADV':>12s}  "
          f"{'% of ADV':>10s}  {'Impact':>10s}  {'Ann. Cost':>12s}")
    print("-" * 70)
    for d in details[:10]:
        print(f"{d['instrument']:>12s}  {d['pos_contracts']:>10,.1f}  "
              f"{d['adv']:>12,.0f}  {d['participation_pct']:>9.3f}%  "
              f"{d['impact_bps']:>8.2f} bps  ${d['annual_cost']/1e3:>9,.0f}K")
    print("-" * 70)

# -- Per-instrument: AUM at which participation hits 1% ADV ----------------
print()
print("=" * 70)
print("  PER-INSTRUMENT CAPACITY (AUM at 1% ADV participation)")
print("=" * 70)
print(f"{'Instrument':>12s}  {'Base Pos':>10s}  {'ADV':>12s}  {'AUM @ 1%':>14s}")
print("-" * 54)

# At base capital ($100M), participation = pos_base * DAILY_TURNOVER_FRAC / ADV
# We want: pos_base * scale * DAILY_TURNOVER_FRAC / ADV = 0.01
# scale = 0.01 * ADV / (pos_base * DAILY_TURNOVER_FRAC)
# AUM @ 1% = scale * STARTING_CAPITAL
capacity_1pct = {}
for inst in INSTRUMENTS:
    pos = avg_pos_base.get(inst, np.nan)
    adv = adv_dict.get(inst, np.nan)
    if np.isnan(pos) or np.isnan(adv) or pos < 0.01:
        continue
    scale_1pct = 0.01 * adv / (pos * DAILY_TURNOVER_FRAC)
    aum_1pct = scale_1pct * STARTING_CAPITAL
    capacity_1pct[inst] = aum_1pct

for inst, aum_1pct in sorted(capacity_1pct.items(), key=lambda x: x[1]):
    pos = avg_pos_base[inst]
    adv = adv_dict[inst]
    if aum_1pct < 20_000_000_000:  # only show those under $20B
        print(f"{inst:>12s}  {pos:>10,.1f}  {adv:>12,.0f}  ${aum_1pct/1e6:>10,.0f}M")

print("-" * 54)

print()
print("Notes:")
print("  - ADV = front-month average daily volume (last 5 yrs, max-volume contract per day)")
print("  - Impact model: square-root with k=10 bps (conservative for liquid futures)")
print("  - Participation rate = (position * daily_turnover) / ADV")
print("  - Contract values converted to USD using approximate spot FX rates")
print("  - Net SR = Gross SR - (annual impact cost / AUM) / ann_vol")
print("  - Capacity breakpoint = AUM where net SR = gross SR - 0.50")
