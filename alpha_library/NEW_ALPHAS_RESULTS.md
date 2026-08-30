# New Single-Alpha Sleeves — Results

Six new single-alpha signals plus one bonus (COT). Universe: same 62 futures used by the existing 31 alphas. Backtest engine: `run_compounded_portfolio` from `ig_shared_config`. Sharpe is excess-of-T-bill, computed from the saved checkpoint `pkl` file (same methodology as `single_alpha_summary.csv`).

## 1. Headline metrics

| Alpha | SR_Full | SR_Post2010 | CAGR | Max_DD | Avg_Corr (vs all 31 existing) |
|---|---:|---:|---:|---:|---:|
| **Vol_Managed_Mom** *(transform of EWMAC)* | **0.663** | 0.359 | 17.79% | 57.46% | 0.257 |
| **Hurst_Filter** *(transform of EWMAC)*    | **0.501** | 0.455 | 11.12% | 46.11% | 0.257 |
| EOM_Drift                                  | 0.450  | 0.354 | 11.56% | 68.53% | -0.003 |
| LeadLag (HG→6A, ES_RV→ZN, CL→6C)          | 0.183  | -0.072 | 4.45% | 56.87% | 0.066 |
| FOMC_Drift (equity-only)                   | 0.113  | -0.153 | 3.29% | 67.44% | 0.004 |
| Inflation_Tilt (BLS CPI)                   | 0.035  | 0.102 | -0.57% | 82.58% | 0.076 |
| COT_Positioning (CFTC SODA)               | -0.211 | -0.064 | -1.90% | 84.21% | -0.141 |

EWMAC_Ensemble baseline (for context): **SR_Full = 0.574**, Max_DD 53.6%.

## 2. Correlations to S183 pillars

|Alpha | corr→EWMAC | corr→Carry | corr→Skew | corr→VoV | max\|corr\| |
|---|---:|---:|---:|---:|---:|
| EOM_Drift           |  0.135 | -0.130 |  0.200 |  0.003 | **0.200** |
| FOMC_Drift          |  0.039 | -0.015 |  0.127 | -0.020 | **0.127** |
| LeadLag             |  0.117 | -0.000 | -0.087 | -0.058 | **0.117** |
| Inflation_Tilt      |  0.259 |  0.089 |  0.087 | -0.106 | **0.259** |
| Hurst_Filter        |  **0.935** |  0.220 |  0.222 | -0.117 | **0.935** |
| Vol_Managed_Mom     |  **0.939** |  0.226 |  0.270 | -0.024 | **0.939** |
| COT_Positioning     | -0.432 | -0.074 | -0.040 |  0.072 | **0.432** |

The two transformations (Hurst gate, Moreira-Muir vol scaling) sit on top of EWMAC, so 0.93+ correlation to EWMAC_Ensemble is expected — they are *replacements*, not additions. Their value would only be real if they materially improve risk-adjusted return over the EWMAC baseline.

## 3. Recommendation

**Acceptance criteria:** SR_Full > 0.20, max\|corr to existing pillars\| < 0.30, SR_Post2010 not negative.

| Alpha | SR>0.20 | maxCorr<0.30 | SR_2010+>=0 | Add to S183? |
|---|:--:|:--:|:--:|:--:|
| EOM_Drift           | YES (0.45) | YES (0.20) | YES (0.35)  | **YES** |
| FOMC_Drift          | NO  (0.11) | YES        | NO          | no |
| LeadLag             | NO  (0.18) | YES        | NO          | no |
| Inflation_Tilt      | NO  (0.04) | YES (0.26) | YES (0.10)  | no |
| Hurst_Filter        | YES (0.50) | NO (0.94)  | YES         | no — replacement only |
| Vol_Managed_Mom     | YES (0.66) | NO (0.94)  | YES         | no — replacement only |
| COT_Positioning     | NO  (neg)  | NO (0.43)  | NO          | no |

**Single recommendation: add EOM_Drift to S183.** It clears every gate cleanly: SR_Full 0.45 (better than 22 of the 31 existing alphas), max correlation to any S183 pillar 0.20, post-2010 SR still 0.35 (not a pre-2010 ghost). Avg correlation to the full alpha library is essentially zero (-0.003) — about as orthogonal as anything we own, comparable to Skew (Avg_Corr 0.02) and Vol_of_Vol (-0.01). Calendar-only signal, ~1,600 trades/yr — costs are real but covered.

**Separate note on the two trend transforms.** Vol_Managed_Mom (SR 0.66) and Hurst_Filter (SR 0.50) are both interesting alternatives to the current EWMAC_Ensemble (SR 0.57). Hurst_Filter pays for its (modest) SR loss with a much lower max drawdown (46% vs 54%); Vol_Managed_Mom raises SR but also raises Max_DD slightly. These should be evaluated as *replacements* for EWMAC_Ensemble inside S183, not as additional sleeves. Recommend: backtest S183 with EWMAC_Ensemble swapped for Vol_Managed_Mom in a follow-up — if portfolio-level SR improves and Max_DD does not blow out, swap is worthwhile.

## 4. Failure log

- **FRED CSV download blocked.** Direct `urllib`/`curl` to `fredgraph.csv` returned `ConnectionResetError` from this network. Worked around by switching `Inflation_Tilt` to BLS public API (`api.bls.gov/publicAPI/v2/timeseries/data/CUUR0000SA0`) in 10-year chunks — equivalent CPI-U All-Items series, cached to `_cpi_cache.csv`. The `_load_cpi()` function in the script still tries FRED first; the BLS path is hand-cached upstream.
- **COT signal underperformed.** CFTC SODA endpoint worked end-to-end (43 contracts mapped, weekly z-score, 1-day release lag), but the contrarian "fade speculator crowding" signal posted SR -0.21. Possibilities: (a) trend-following is the dominant regime in the S183 universe so fading momentum loses; (b) the Disaggregated reports only start in 2006 — pre-2006 coverage is sparse and may bias the rolling z-score early on. Could be re-tried as a momentum (rather than fade) sign or restricted post-2010, but as specified the signal is rejected.

## Files written

Scripts: `ig_single_eom_drift.py`, `ig_single_fomc_drift.py`, `ig_single_leadlag.py`, `ig_single_inflation_tilt.py`, `ig_single_hurst_filter.py`, `ig_single_vol_managed_mom.py`, `ig_single_cot_positioning.py`.

Per-alpha output (each alpha): `Single_Alpha_<NAME>_portfolio_returns.csv`, `Single_Alpha_<NAME>_checkpoint.pkl`, `Single_Alpha_<NAME>_NAV_Curve.png`, `Single_Alpha_<NAME>_Tearsheet.png`.

Data caches: `_cpi_cache.csv` (BLS CUUR0000SA0, 1985-2026), `_cot_cache/*.parquet` (43 CFTC series).
