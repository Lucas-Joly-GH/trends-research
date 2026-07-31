"""
Theoretical Optimal Alpha Blend Optimizer
==========================================
Uses the single-alpha correlation matrix + individual SRs to find
the weight vector that maximises the theoretical portfolio Sharpe Ratio.

Methods:
  1. Unconstrained analytical (Markowitz): w* ∝ Σ⁻¹ μ
  2. Long-only constrained (scipy.optimize)
  3. Subset search: brute-force best K-alpha combos (K=2..8)
  4. Constrained with trend >= 1/N (mandate)
  5. Compare against S157 actual weights

Assumes all alphas are scaled to unit volatility (SR = mean / vol = mean
when vol=1), so the covariance matrix Σ = correlation matrix.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────
corr_df = pd.read_csv(r"C:\Users\l.joly\LJOLY_Memoire_INSEEC_Msc2\Part 3\IG_Backtest\Single_Alpha\single_alpha_correlation_matrix.csv", index_col=0)
summary = pd.read_csv(r"C:\Users\l.joly\LJOLY_Memoire_INSEEC_Msc2\Part 3\IG_Backtest\Single_Alpha\single_alpha_summary.csv")

alphas = list(corr_df.columns)
n = len(alphas)
sr_map = dict(zip(summary["Alpha"], summary["SR_Full"]))
mu = np.array([sr_map[a] for a in alphas])          # expected SR vector
Sigma = corr_df.values.copy()                        # correlation = covariance (unit vol)

print("=" * 80)
print("THEORETICAL OPTIMAL ALPHA BLEND OPTIMIZER")
print("=" * 80)
print(f"\nAlphas ({n}):")
for i, a in enumerate(alphas):
    print(f"  {i+1:2d}. {a:25s}  SR = {mu[i]:+.4f}  Avg_Corr = {corr_df.iloc[i].drop(a).mean():.4f}")

# ── Helper: portfolio SR given weights ─────────────────────────────────
def portfolio_sr(w, mu, Sigma):
    port_mean = w @ mu
    port_var  = w @ Sigma @ w
    if port_var <= 0:
        return 0.0
    return port_mean / np.sqrt(port_var)

# ══════════════════════════════════════════════════════════════════════
# 1. UNCONSTRAINED ANALYTICAL (MARKOWITZ)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("1. UNCONSTRAINED MARKOWITZ  w* = inv(Sigma) * mu")
print("=" * 80)

try:
    Sigma_inv = np.linalg.inv(Sigma)
    w_raw = Sigma_inv @ mu
    w_mk = w_raw / np.sum(np.abs(w_raw))   # normalise to sum(|w|)=1
    sr_mk = portfolio_sr(w_mk, mu, Sigma)
    print(f"\nTheoretical Portfolio SR = {sr_mk:.4f}\n")
    for i, a in enumerate(alphas):
        print(f"  {a:25s}  w = {w_mk[i]:+.4f}")
except np.linalg.LinAlgError:
    print("  Σ is singular – skipping unconstrained solution")
    w_mk = None

# ══════════════════════════════════════════════════════════════════════
# 2. LONG-ONLY CONSTRAINED OPTIMISATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("2. LONG-ONLY CONSTRAINED (w_i >= 0, sum = 1)")
print("=" * 80)

def neg_sr(w):
    port_mean = w @ mu
    port_var  = w @ Sigma @ w
    if port_var <= 1e-12:
        return 0.0
    return -port_mean / np.sqrt(port_var)

constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
bounds = [(0, 1)] * n
x0 = np.ones(n) / n

res_lo = minimize(neg_sr, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                  options={"maxiter": 2000, "ftol": 1e-14})
w_lo = res_lo.x
sr_lo = -res_lo.fun

print(f"\nTheoretical Portfolio SR = {sr_lo:.4f}\n")
print("  Non-zero weights:")
for i, a in enumerate(alphas):
    if w_lo[i] > 0.005:
        print(f"    {a:25s}  w = {w_lo[i]:.4f}  ({w_lo[i]*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════
# 3. BEST SUBSET SEARCH (brute-force K = 2..8)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("3. BEST SUBSET SEARCH (K = 2..8 alphas, long-only)")
print("=" * 80)

best_per_k = {}
for k in range(2, 9):
    best_sr_k = -999
    best_combo_k = None
    best_w_k = None
    for combo in combinations(range(n), k):
        idx = list(combo)
        mu_sub = mu[idx]
        # Skip if all negative
        if np.all(mu_sub <= 0):
            continue
        Sigma_sub = Sigma[np.ix_(idx, idx)]

        def neg_sr_sub(w):
            pm = w @ mu_sub
            pv = w @ Sigma_sub @ w
            if pv <= 1e-12:
                return 0.0
            return -pm / np.sqrt(pv)

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bds = [(0, 1)] * k
        x0_sub = np.ones(k) / k

        r = minimize(neg_sr_sub, x0_sub, method="SLSQP", bounds=bds, constraints=cons,
                     options={"maxiter": 1000, "ftol": 1e-14})
        sr_sub = -r.fun
        if sr_sub > best_sr_k:
            best_sr_k = sr_sub
            best_combo_k = idx
            best_w_k = r.x

    best_per_k[k] = (best_sr_k, best_combo_k, best_w_k)
    names = [alphas[i] for i in best_combo_k]
    print(f"\n  K={k}  SR = {best_sr_k:.4f}")
    for j, idx in enumerate(best_combo_k):
        if best_w_k[j] > 0.005:
            print(f"    {alphas[idx]:25s}  w = {best_w_k[j]:.4f}  ({best_w_k[j]*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════
# 4. MANDATE-CONSTRAINED: trend >= 1/N, long-only
#    "Trend" = any of EWMAC_*, XS_Momentum, Network_Momentum, Breakout, Acceleration
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("4. MANDATE-CONSTRAINED (trend >= 1/N = 7.7%, long-only)")
print("=" * 80)

trend_alphas = {"EWMAC_Ensemble", "EWMAC_Fast", "EWMAC_Medium", "EWMAC_Slow",
                "XS_Momentum", "Network_Momentum", "Breakout", "Acceleration"}
trend_idx = [i for i, a in enumerate(alphas) if a in trend_alphas]

def neg_sr_mandate(w):
    port_mean = w @ mu
    port_var  = w @ Sigma @ w
    if port_var <= 1e-12:
        return 0.0
    return -port_mean / np.sqrt(port_var)

constraints_m = [
    {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    {"type": "ineq", "fun": lambda w: np.sum(w[trend_idx]) - 1.0/n},  # trend >= 1/N
]
bounds_m = [(0, 1)] * n

res_m = minimize(neg_sr_mandate, x0, method="SLSQP", bounds=bounds_m, constraints=constraints_m,
                 options={"maxiter": 2000, "ftol": 1e-14})
w_m = res_m.x
sr_m = -res_m.fun

print(f"\nTheoretical Portfolio SR = {sr_m:.4f}\n")
print("  Non-zero weights:")
trend_total = 0
for i, a in enumerate(alphas):
    if w_m[i] > 0.005:
        tag = " [TREND]" if a in trend_alphas else ""
        print(f"    {a:25s}  w = {w_m[i]:.4f}  ({w_m[i]*100:.1f}%){tag}")
        if a in trend_alphas:
            trend_total += w_m[i]
print(f"\n  Total trend weight: {trend_total*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════
# 5. COMPARE WITH S157 ACTUAL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("5. S157 ACTUAL vs THEORETICAL OPTIMAL")
print("=" * 80)

# S157: Trend 33.3% (1/3 TS-EWMAC + 1/3 XS-momentum + 1/3 NMM), Carry 33.3%, Skew 33.3%
# Map to single-alpha universe:
#   EWMAC_Ensemble ≈ TS-EWMAC component → 33.3% * 33.3% = 11.1%
#   XS_Momentum → 33.3% * 33.3% = 11.1%
#   Network_Momentum → 33.3% * 33.3% = 11.1%
#   Carry → 33.3%
#   Skew → 33.3%
w_s157 = np.zeros(n)
for i, a in enumerate(alphas):
    if a == "EWMAC_Ensemble":
        w_s157[i] = 1/3 * 1/3   # 11.1%
    elif a == "XS_Momentum":
        w_s157[i] = 1/3 * 1/3
    elif a == "Network_Momentum":
        w_s157[i] = 1/3 * 1/3
    elif a == "Carry":
        w_s157[i] = 1/3
    elif a == "Skew":
        w_s157[i] = 1/3

sr_s157 = portfolio_sr(w_s157, mu, Sigma)
print(f"\n  S157 Theoretical SR = {sr_s157:.4f}")
print(f"  Best Long-Only SR   = {sr_lo:.4f}  (improvement: {(sr_lo/sr_s157 - 1)*100:+.1f}%)")
print(f"  Mandate-Constr SR   = {sr_m:.4f}  (improvement: {(sr_m/sr_s157 - 1)*100:+.1f}%)")

print("\n  S157 weight decomposition:")
for i, a in enumerate(alphas):
    if w_s157[i] > 0.005:
        print(f"    {a:25s}  w = {w_s157[i]:.4f}  ({w_s157[i]*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════
# 6. SENSITIVITY: WHAT IF WE ADD SKEW_XS OR CARRY_MOMENTUM?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("6. S157 + ORTHOGONAL SATELLITE TEST")
print("=" * 80)

# Test adding each unused orthogonal alpha at 5%, 10%, 15% satellite weight
unused_ortho = ["Skew_XS", "Carry_Momentum", "Value"]
for sat_name in unused_ortho:
    sat_idx = alphas.index(sat_name)
    print(f"\n  Adding {sat_name} (SR={mu[sat_idx]:+.4f}, Avg_Corr={corr_df.iloc[sat_idx].drop(sat_name).mean():.4f}):")
    for sat_pct in [0.05, 0.10, 0.15]:
        w_test = w_s157.copy() * (1 - sat_pct)
        w_test[sat_idx] += sat_pct
        sr_test = portfolio_sr(w_test, mu, Sigma)
        delta = (sr_test / sr_s157 - 1) * 100
        print(f"    {sat_pct*100:.0f}% satellite → SR = {sr_test:.4f}  ({delta:+.1f}%)")

# ══════════════════════════════════════════════════════════════════════
# 7. GRID SEARCH: BEST 5-ALPHA BLEND (the S157 architecture space)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("7. GRID SEARCH: OPTIMAL WEIGHTS FOR S157's 5 ALPHAS")
print("=" * 80)

# Focus on EWMAC_Ensemble, XS_Momentum, Network_Momentum, Carry, Skew
core5 = ["EWMAC_Ensemble", "XS_Momentum", "Network_Momentum", "Carry", "Skew"]
core5_idx = [alphas.index(a) for a in core5]
mu5 = mu[core5_idx]
Sigma5 = Sigma[np.ix_(core5_idx, core5_idx)]

def neg_sr5(w):
    pm = w @ mu5
    pv = w @ Sigma5 @ w
    if pv <= 1e-12:
        return 0.0
    return -pm / np.sqrt(pv)

# 7a. Unconstrained within S157's 5 alphas
cons5 = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
bds5 = [(0, 1)] * 5
x05 = np.ones(5) / 5

res5 = minimize(neg_sr5, x05, method="SLSQP", bounds=bds5, constraints=cons5,
                options={"maxiter": 2000, "ftol": 1e-14})
w5 = res5.x
sr5 = -res5.fun

print(f"\n  7a. Optimal 5-alpha blend (long-only):")
print(f"      SR = {sr5:.4f}  (vs S157 = {sr_s157:.4f}, delta = {(sr5/sr_s157-1)*100:+.1f}%)\n")
for j, a in enumerate(core5):
    print(f"      {a:25s}  w = {w5[j]:.4f}  ({w5[j]*100:.1f}%)")

# 7b. With trend >= 33% constraint (mandate)
trend5_idx = [0, 1, 2]  # EWMAC, XS, NMM are trend
cons5m = [
    {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    {"type": "ineq", "fun": lambda w: w[0] + w[1] + w[2] - 0.333},
]
res5m = minimize(neg_sr5, x05, method="SLSQP", bounds=bds5, constraints=cons5m,
                 options={"maxiter": 2000, "ftol": 1e-14})
w5m = res5m.x
sr5m = -res5m.fun

print(f"\n  7b. Optimal 5-alpha blend (trend >= 33.3%):")
print(f"      SR = {sr5m:.4f}  (vs S157 = {sr_s157:.4f}, delta = {(sr5m/sr_s157-1)*100:+.1f}%)\n")
for j, a in enumerate(core5):
    tag = " [TREND]" if j in trend5_idx else ""
    print(f"      {a:25s}  w = {w5m[j]:.4f}  ({w5m[j]*100:.1f}%){tag}")
print(f"      Trend total: {sum(w5m[j] for j in trend5_idx)*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════
# 8. EXHAUSTIVE: BEST N-ALPHA COMBO FROM ALL 13 (long-only, N=3..7)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("8. SUMMARY TABLE: BEST CONFIGURATIONS")
print("=" * 80)

results = []
results.append(("S157 Actual", sr_s157, "EWMAC(11.1%) + XS(11.1%) + NMM(11.1%) + Carry(33.3%) + Skew(33.3%)"))
results.append(("5-Alpha Optimal (unconstrained)", sr5,
                " + ".join(f"{core5[j]}({w5[j]*100:.0f}%)" for j in range(5) if w5[j] > 0.005)))
results.append(("5-Alpha Optimal (trend>=33%)", sr5m,
                " + ".join(f"{core5[j]}({w5m[j]*100:.0f}%)" for j in range(5) if w5m[j] > 0.005)))
results.append(("Full 13-Alpha Long-Only", sr_lo,
                " + ".join(f"{alphas[i]}({w_lo[i]*100:.0f}%)" for i in range(n) if w_lo[i] > 0.005)))
results.append(("Full 13-Alpha Mandate", sr_m,
                " + ".join(f"{alphas[i]}({w_m[i]*100:.0f}%)" for i in range(n) if w_m[i] > 0.005)))

for k in sorted(best_per_k.keys()):
    sr_k, idx_k, w_k = best_per_k[k]
    desc = " + ".join(f"{alphas[idx_k[j]]}({w_k[j]*100:.0f}%)" for j in range(len(idx_k)) if w_k[j] > 0.005)
    results.append((f"Best {k}-Alpha Subset", sr_k, desc))

# Sort by SR
results.sort(key=lambda x: x[1], reverse=True)

print(f"\n  {'Configuration':<38s} {'SR':>7s}  {'vs S157':>8s}  Composition")
print("  " + "-" * 120)
for name, sr_val, desc in results:
    delta = (sr_val / sr_s157 - 1) * 100 if sr_s157 > 0 else 0
    print(f"  {name:<38s} {sr_val:7.4f}  {delta:+7.1f}%  {desc}")

# ══════════════════════════════════════════════════════════════════════
# 9. KEY TAKEAWAYS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("9. KEY TAKEAWAYS")
print("=" * 80)

# Find the best practical config (5-alpha with mandate)
print(f"""
  • S157 theoretical SR = {sr_s157:.4f}
  • Best 5-alpha (S157 universe, trend>=33%) = {sr5m:.4f} ({(sr5m/sr_s157-1)*100:+.1f}%)
  • Best long-only from all 13 = {sr_lo:.4f} ({(sr_lo/sr_s157-1)*100:+.1f}%)

  → S157's equal-weight architecture is {'close to optimal' if abs(sr5m/sr_s157-1) < 0.05 else 'suboptimal — reweighting could help'}
  → {'Value should be excluded (negative SR)' if mu[alphas.index('Value')] < 0 else ''}
  → Most diversification benefit comes from: Skew, Carry, Skew_XS (lowest avg correlations)
""")

print("=" * 80)
print("DONE")
print("=" * 80)
