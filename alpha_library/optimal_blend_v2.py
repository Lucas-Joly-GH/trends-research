"""
Optimal Blend v2 -- With all 30 alphas (13 original + 17 new)
==============================================================
Focus on: positive-SR alphas only, then MVO to find optimal weights.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# Load data
corr_df = pd.read_csv(r"C:\Users\l.joly\LJOLY_Memoire_INSEEC_Msc2\Part 3\IG_Backtest\Single_Alpha\single_alpha_correlation_matrix.csv", index_col=0)
summary = pd.read_csv(r"C:\Users\l.joly\LJOLY_Memoire_INSEEC_Msc2\Part 3\IG_Backtest\Single_Alpha\single_alpha_summary.csv")

# Filter: positive SR only
pos_mask = summary["SR_Full"] > 0
summary_pos = summary[pos_mask].copy()
alphas_pos = list(summary_pos["Alpha"])
n = len(alphas_pos)

mu = np.array([summary_pos.loc[summary_pos["Alpha"]==a, "SR_Full"].values[0] for a in alphas_pos])
Sigma = corr_df.loc[alphas_pos, alphas_pos].values.copy()

print("=" * 90)
print("OPTIMAL BLEND v2 -- 30 ALPHAS (positive SR only)")
print("=" * 90)
print(f"\nPositive-SR alphas ({n}):")
for i, a in enumerate(alphas_pos):
    ac = corr_df.loc[a].drop(a).mean()
    print(f"  {i+1:2d}. {a:30s}  SR={mu[i]:.3f}  AvgCorr={ac:.3f}")

def portfolio_sr(w, mu, Sigma):
    pm = w @ mu
    pv = w @ Sigma @ w
    return pm / np.sqrt(pv) if pv > 1e-12 else 0.0

def neg_sr(w):
    return -portfolio_sr(w, mu, Sigma)

# ── 1. FULL LONG-ONLY OPTIMISATION ──
print("\n" + "=" * 90)
print("1. LONG-ONLY OPTIMISATION (all positive-SR alphas)")
print("=" * 90)
cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
bds = [(0, 1)] * n
x0 = np.ones(n) / n

res = minimize(neg_sr, x0, method="SLSQP", bounds=bds, constraints=cons,
               options={"maxiter": 5000, "ftol": 1e-14})
w_opt = res.x
sr_opt = -res.fun

print(f"\nTheoretical Portfolio SR = {sr_opt:.4f}\n")
print(f"  {'Alpha':30s} {'Weight':>8s} {'SR':>7s} {'AvgCorr':>8s}")
print(f"  {'-'*30} {'-'*8} {'-'*7} {'-'*8}")
for i, a in enumerate(alphas_pos):
    if w_opt[i] > 0.005:
        ac = corr_df.loc[a].drop(a).mean()
        print(f"  {a:30s} {w_opt[i]:7.1%} {mu[i]:7.3f} {ac:8.3f}")

# ── 2. S157 ACTUAL ──
print("\n" + "=" * 90)
print("2. S157 ACTUAL vs OPTIMAL")
print("=" * 90)
w_s157 = np.zeros(n)
for i, a in enumerate(alphas_pos):
    if a == "EWMAC_Ensemble": w_s157[i] = 1/3 * 1/3
    elif a == "XS_Momentum":  w_s157[i] = 1/3 * 1/3
    elif a == "Network_Momentum": w_s157[i] = 1/3 * 1/3
    elif a == "Carry":        w_s157[i] = 1/3
    elif a == "Skew":         w_s157[i] = 1/3
sr_s157 = portfolio_sr(w_s157, mu, Sigma)

print(f"\n  S157 Theoretical SR     = {sr_s157:.4f}")
print(f"  Full Optimal SR         = {sr_opt:.4f}  ({(sr_opt/sr_s157-1)*100:+.1f}%)")

# ── 3. BEST 5-ALPHA, 6-ALPHA, 7-ALPHA COMBOS ──
from itertools import combinations

print("\n" + "=" * 90)
print("3. BEST K-ALPHA SUBSETS (K=4..8)")
print("=" * 90)

for k in range(4, 9):
    best_sr = -999
    best_combo = None
    best_w = None
    for combo in combinations(range(n), k):
        idx = list(combo)
        mu_sub = mu[idx]
        Sigma_sub = Sigma[np.ix_(idx, idx)]

        def neg_sr_sub(w):
            pm = w @ mu_sub
            pv = w @ Sigma_sub @ w
            return -pm / np.sqrt(pv) if pv > 1e-12 else 0.0

        c = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        b = [(0, 1)] * k
        x = np.ones(k) / k
        r = minimize(neg_sr_sub, x, method="SLSQP", bounds=b, constraints=c,
                     options={"maxiter": 2000, "ftol": 1e-14})
        if -r.fun > best_sr:
            best_sr = -r.fun
            best_combo = idx
            best_w = r.x

    print(f"\n  K={k}  SR={best_sr:.4f}  ({(best_sr/sr_s157-1)*100:+.1f}% vs S157)")
    for j, idx in enumerate(best_combo):
        if best_w[j] > 0.005:
            print(f"    {alphas_pos[idx]:30s}  w={best_w[j]:.1%}")

# ── 4. NEW ALPHA CONTRIBUTION TEST ──
print("\n" + "=" * 90)
print("4. MARGINAL VALUE OF EACH NEW ALPHA (added to S157 at 10%)")
print("=" * 90)

new_alphas = ["Normalized_Momentum", "Relative_Carry", "Vol_of_Vol", "Anchored_Momentum",
              "Vol_Regime", "Realized_Semivariance", "Skew_XS", "Carry_Momentum"]

print(f"\n  {'Alpha':30s} {'SR@10%':>8s} {'Delta':>8s} {'AvgCorr':>8s}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
for sat_name in new_alphas:
    if sat_name not in alphas_pos:
        continue
    sat_idx = alphas_pos.index(sat_name)
    w_test = w_s157.copy() * 0.90
    w_test[sat_idx] += 0.10
    sr_test = portfolio_sr(w_test, mu, Sigma)
    delta = (sr_test / sr_s157 - 1) * 100
    ac = corr_df.loc[sat_name].drop(sat_name).mean()
    print(f"  {sat_name:30s} {sr_test:8.4f} {delta:+7.1f}% {ac:8.3f}")

# ── 5. OPTIMAL S157 + BEST NEW ALPHAS ──
print("\n" + "=" * 90)
print("5. OPTIMAL BLEND: S157 CORE + NEW ALPHAS")
print("=" * 90)

# Core S157 + best new candidates
core_plus = ["EWMAC_Ensemble", "XS_Momentum", "Network_Momentum", "Carry", "Skew",
             "Normalized_Momentum", "Relative_Carry", "Vol_of_Vol", "EWMAC_Slow",
             "Skew_XS", "Carry_Momentum", "Vol_Regime"]
cp_idx = [alphas_pos.index(a) for a in core_plus if a in alphas_pos]
cp_names = [alphas_pos[i] for i in cp_idx]
mu_cp = mu[cp_idx]
Sigma_cp = Sigma[np.ix_(cp_idx, cp_idx)]

def neg_sr_cp(w):
    pm = w @ mu_cp
    pv = w @ Sigma_cp @ w
    return -pm / np.sqrt(pv) if pv > 1e-12 else 0.0

# Trend constraint: EWMAC + XS + NMM + NormMom >= 33%
trend_names = {"EWMAC_Ensemble", "XS_Momentum", "Network_Momentum", "Normalized_Momentum", "EWMAC_Slow"}
trend_cp_idx = [i for i, a in enumerate(cp_names) if a in trend_names]

cons_cp = [
    {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    {"type": "ineq", "fun": lambda w: sum(w[i] for i in trend_cp_idx) - 0.333},
]
bds_cp = [(0, 1)] * len(cp_idx)
x0_cp = np.ones(len(cp_idx)) / len(cp_idx)

res_cp = minimize(neg_sr_cp, x0_cp, method="SLSQP", bounds=bds_cp, constraints=cons_cp,
                  options={"maxiter": 5000, "ftol": 1e-14})
w_cp = res_cp.x
sr_cp = -res_cp.fun

print(f"\n  SR = {sr_cp:.4f}  ({(sr_cp/sr_s157-1)*100:+.1f}% vs S157)")
print(f"\n  {'Alpha':30s} {'Weight':>8s}")
print(f"  {'-'*30} {'-'*8}")
trend_tot = 0
for j, name in enumerate(cp_names):
    if w_cp[j] > 0.005:
        tag = " [TREND]" if name in trend_names else ""
        print(f"  {name:30s} {w_cp[j]:7.1%}{tag}")
        if name in trend_names:
            trend_tot += w_cp[j]
print(f"\n  Trend total: {trend_tot:.1%}")

# ── 6. FINAL SUMMARY TABLE ──
print("\n" + "=" * 90)
print("6. FINAL RANKING")
print("=" * 90)

configs = [
    ("S157 Actual (equal-weight Trinity)", sr_s157),
    ("Full Optimal (all positive-SR)", sr_opt),
    ("Core+New Optimal (trend>=33%)", sr_cp),
]

configs.sort(key=lambda x: x[1], reverse=True)
print(f"\n  {'Configuration':45s} {'SR':>7s} {'vs S157':>8s}")
print(f"  {'-'*45} {'-'*7} {'-'*8}")
for name, sr in configs:
    delta = (sr / sr_s157 - 1) * 100
    print(f"  {name:45s} {sr:7.4f} {delta:+7.1f}%")

# ── 7. KEY TAKEAWAYS ──
print("\n" + "=" * 90)
print("7. KEY TAKEAWAYS")
print("=" * 90)
print(f"""
  NEW ALPHAS WORTH ADDING TO STRATEGY:

  1. Normalized Momentum (SR=0.555): Carver's normmom variant.
     -> BUT 0.967 corr with EWMAC_Ensemble! Almost identical signal.
     -> Verdict: REDUNDANT with existing EWMAC.

  2. Relative Carry (SR=0.363): Cross-sectional carry z-score.
     -> 0.628 corr with Carry (high but not redundant).
     -> Low avg corr (0.060). POTENTIALLY USEFUL.

  3. Vol-of-Vol (SR=0.237): Volatility stability signal.
     -> Avg corr = -0.012 (extremely orthogonal!).
     -> SR decent at 0.237. POTENTIALLY USEFUL.

  4. Vol Regime (SR=0.176): Vol mean-reversion.
     -> Avg corr = -0.007 (extremely orthogonal!).
     -> But 0.575 corr with Vol_of_Vol - pick one.
     -> Moderate SR. MARGINAL.

  5. Anchored Momentum (SR=0.203): Multi-horizon trend.
     -> 0.836 corr with Normalized_Momentum, 0.818 with EWMAC.
     -> Verdict: REDUNDANT.

  6. Realized Semivariance (SR=0.108): Up/down vol ratio.
     -> 0.875 corr with EWMAC_Fast! It's just picking up trend.
     -> Verdict: REDUNDANT.
""")

print("  BOTTOM LINE:")
print("  -> Vol_of_Vol (SR=0.237, avg_corr=-0.012) is the BEST new addition.")
print("  -> Relative Carry (SR=0.363, avg_corr=0.060) is the 2nd best.")
print("  -> Everything else is either redundant with EWMAC or has negative SR.")
print("=" * 90)
