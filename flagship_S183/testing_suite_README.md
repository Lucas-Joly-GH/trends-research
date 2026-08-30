# Strategy 183 — Testing Suite README

`ig_testing_suite_183.py` is a single-file, self-contained statistical testing artifact for Strategy 183. It is adapted from `ig_testing_suite_182.py` (which includes the post-bug-fix engine patches: subperiod dSRs, sneak-under detection, and GROUP 9/10 combination interaction tests) with the baseline defaults rotated to S183's four architectural deltas: `use_conviction=False`, `w_ts=1.0/w_xs=0.0`, `sigmoid_steepness=10.0`, `smooth_span=3`. Pooled 4×4 FDM, VoV direction overlay, and the 3-speed TS-EWMAC ensemble are retained unchanged.

The suite answers six orthogonal questions:

1. **Local sensitivity** — is MASTER on a flat plateau? *(Datamining sweep, 103 variants, 18 axes)*
2. **Component importance** — which components are load-bearing? *(Ablation matrix, 69 variants)*
3. **Statistical significance** — is the observed SR distinguishable from null models? *(Analysis battery, T1–T22)*
4. **Factor robustness** — does alpha survive known risk premia? *(T14, T15, AQR kitchen-sink)*
5. **Parsimony & arbitrariness** — can MASTER be simplified further, and do arbitrary parameters produce similar performance? *(Parsimony battery, T23–T28)*
6. **Execution & cross-instrument robustness** — does the strategy survive live-execution latency, instrument removal, signal noise, and leverage changes? Is the reported SR sourced from trading alpha or cash yield? *(Robustness battery, T29–T34)*

## 1. Quick start

```bash
cd "Part 3/IG_Backtest"

# Regenerate the S183 MASTER checkpoint (multi-hour run; do this first).
python Strategy_183/_run_s183_master.py

# Full suite: DM + Ablation + Analysis + Parsimony + Robustness (default).
python Strategy_183/ig_testing_suite_183.py

# Individual modules:
python Strategy_183/ig_testing_suite_183.py --dm
python Strategy_183/ig_testing_suite_183.py --ablation
python Strategy_183/ig_testing_suite_183.py --analysis
python Strategy_183/ig_testing_suite_183.py --parsimony
python Strategy_183/ig_testing_suite_183.py --robustness

# Resume an interrupted run (skip checkpoints that already exist):
python Strategy_183/ig_testing_suite_183.py --all --skip-existing

# Rebuild CSVs and JSON reports from existing checkpoints only (fast):
python Strategy_183/ig_testing_suite_183.py --metrics-only

# Narrow Monte Carlo sample count for the parsimony battery:
python Strategy_183/ig_testing_suite_183.py --parsimony --mc-N 30

# Parallel run (recommended: leave 1-2 cores free):
python Strategy_183/ig_testing_suite_183.py --all --workers 6
```

Output lands under `Strategy_183/TestingSuite/`. The `Analysis/` subdirectory contains the summary CSVs and JSON reports that you will actually read.

## 2. Dependencies

Inherited from `ig_shared_config.py` and `shared_config.py`:

- Python ≥ 3.9
- `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`
- The Carver project's `shared_config.py` (located one directory above `IG_Backtest/`)
- `%IRX.csv` risk-free rate data under `Data/Cash-Yield/`
- Instrument Panama continuous data under `Data/PanamaMethod/`
- Per-contract stats under `Data/InstrumentStats/`
- Mapping file `Général/instrument_mapping.csv`

Optional (for factor regressions T14, T15):

- `fh_factors.csv`, `ff5_factors.csv`, `tsmom_factor.csv`, etc. under `Strategy_183/TestingSuite/ExtData/` or `Strategy_180/TestingSuite/ExtData/` (the loader falls back to the S180 tree so factor CSVs do not need to be duplicated).

## 3. CLI reference

```
python ig_testing_suite_183.py [FLAGS]

Module selection (use any combination; --all is the default):
  --dm              Run the 103-variant datamining sweep
  --ablation        Run the 69-variant ablation matrix
  --analysis        Run the T1-T22 statistical battery against existing checkpoints
  --parsimony       Run the T23-T28 parsimony + random-parameter battery
  --robustness      Run the T29-T34 extended robustness battery
  --all             Run DM + Ablation + Analysis + Parsimony + Robustness
                    (default when no flag given)

Execution modes:
  --skip-existing   Resume: if a variant's checkpoint file already exists, reuse it
  --metrics-only    Rebuild summary CSVs + analysis from existing checkpoints.
                    Skips DM/Ablation/Parsimony (which would need fresh backtests).
  --variants V1 V2  Restrict DM/Ablation runs to the named variant IDs only.
                    Useful for debugging a single variant.

Monte Carlo and bootstrap controls:
  --seed N          Bootstrap seed for reproducibility (default 20260405).
                    Also used as the base seed for T24/T25 random draws.
  --boot-B N        Bootstrap resample count for LW / Romano-Wolf / SPA tests
                    (default 5000).
  --block N         Bootstrap block length (default 21).
  --mc-N N          Monte Carlo sample count for T24 (random-parameter) and T25
                    (random-simplex). Default 100. Each sample is a full backtest,
                    so runtime scales linearly.

Parallelisation:
  --workers N       Number of parallel worker processes for the DM sweep,
                    Ablation matrix, and T24/T25 MC loops (default 1 = serial).
                    Recommended: leave 1-2 cores free, e.g. --workers 6 on an
                    8-core machine.  The Phase-0 library and raw-signal objects
                    are built once in the main process and sent to each worker
                    at pool startup; expect ~5-15 s of startup overhead before
                    the progress bar begins.  The Analysis battery (T1-T22) and
                    T23/T26/T27/T28 parsimony tests are single-threaded and are
                    not affected by --workers.
```

## 4. Modules

The suite is composed of six logical modules. `--all` runs them in the order below; `--dm`, `--ablation`, `--analysis`, `--parsimony`, `--robustness` let you run each individually.

### 4.1 MASTER run

Not part of the suite file itself. Run `python Strategy_183/_run_s183_master.py` to regenerate the authoritative S182 checkpoint at `Strategy_183/Strategy_183_IG_VoV_Quad_Parsimonious_Sharpened_checkpoint.pkl`. This is the reference point every other module compares against.

Expected metrics (directly measured S183 pipeline): `SR_full = 0.986`, `CAGR = 16.05%`, `vol = 13.26%`, `MDD = 23.16%`, `Calmar = 0.693`, `post-2010 SR = 0.971`.

### 4.2 Datamining sweep — `run_datamining`

A 103-variant one-at-a-time parameter sweep along 18 axes (A–R). Each variant perturbs exactly one parameter from the S183 MASTER anchor, allowing a per-axis plateau test. The Phase-0 library (`dm_precompute_library`) precomputes all per-instrument forecast signals once and shares them across all 103 variants for efficiency.

**Pre-flight coverage check.** `dm_build_variant_signals` validates every variant's requested windows against the baked lists (`DM_ALL_VOV_WINDOWS`, `DM_ALL_XS_LOOKBACKS`, `DM_ALL_SKEW_WINDOWS`, `DM_ALL_CONV_WINDOWS`, `DM_ALL_SPEED_PAIRS`) before running. A missing key raises a loud `ValueError` instead of the S180-suite behaviour of silently falling back to `pd.Series(0.0, ...)` and zeroing that alpha component. This check closed a bug where the S180 inherited lists `[21,42,63,90,126]` and `[126,252,378]` were missing S183 MASTER's 64 and 256, which had silently contaminated 41 of the 97 DM variants plus the entire parsimony battery. The same coverage check is now in `abl_build_inst_signals` as well.

**Axes re-centred from S182 to S183:**

| Axis | S183 MASTER | Sweep values |
|---|---|---|
| A (EWMAC speeds) | 3-speed 16/64, 64/256, 128/512 | 1/2/3/4-speed combinations |
| B (conviction) | ON, window=256 | OFF, + windows {128, 192, 256\*, 384, 512} |
| C (trend FDM) | 1.0 | {0.85, 0.90, 0.95, 1.00\*, 1.05, 1.10} |
| D (alpha weights) | 25/25/25/25 | alternative weight vectors |
| E (FDM cap) | **inf** (S182 is uncapped) | caps {1.20, 1.50, 1.65, 1.80, 2.00, 2.20, 2.40, 3.00, inf\*} |
| F (FDM corr span) | 512 | {256, 512, 1024} |
| G (skew window) | 256 | {128, 192, 256\*, 384, 512} |
| H (smooth span) | 5 | {0, 2, 3, 5\*, 8, 10, 15, 21} |
| I (sigmoid steepness) | **1.0** (identity) | {0.5, 1.0\*, 2.0, 3.0, 5.0, 7.0, 10.0} |
| J (overlay bundles) | tighter | {off, tight, default, loose, tighter\*} |
| K (TS/XS split) | 50/50 | {100/0, 75/25, 50/50\*, 25/75, 0/100} |
| L (XS lookback) | **256** | {126, 192, 252, 256\*, 378, 512} |
| M (VoV window) | **64** | {21, 42, 63, 64\*, 90, 126} |
| N (VoV direction) | **ON** | OFF |
| O (VoV weight) | 25% | {0, 10, 15, 20, 25\*, 30, 40}% |
| P (FDM pooling mode) | pooled | {per-inst, pooled\*, shrink, clip} |
| Q (FDM corr span) | 512 | {128, 256, 512\*, 1024, 2048} |
| R (FDM min periods) | 256 | {128, 256\*, 384} |

Asterisks (\*) mark the S183 MASTER value within each axis.

### 4.3 Ablation matrix — `run_ablation`

A 69-variant attribution matrix testing **component removal** and **directed reversions** from the S183 baseline. Unlike DM, ablations can change multiple parameters at once and can enable/disable whole components (drop VoV, drop overlays, drop smoothing).

**Reversion variants re-centred for S182:**

| Variant | What it does |
|---|---|
| `R_TFDM110` | Revert TREND_FDM 1.0 → 1.10 (tests S182 delta 1) |
| `R_S183_SMOOTH5` | S183 + smooth_span 3 → 5 (revert to S182) |
| `R_ADDCAP180` | Impose FDM_CAP = 1.80 on S182's uncapped MASTER |
| `R_ADDCAP200` | Impose FDM_CAP = 2.00 on the S183 master |
| `R_S183_STEEP1` | S183 + sigmoid steepness 10 -> 1 (revert to S182) |
| `R_S183_STEEP20` | S183 + sigmoid steepness 10 → 20 (push further, probe plateau vs cliff) |
| `R_SHIFTED` | Switch to zero-drag shifted sigmoid |
| `R_S183_ADD_CONV` | S183 + conviction ramp ON (revert `use_conviction=False->True`) |
| `R_NO_VOVDIR` | Drop the VoV direction overlay (sign(pct_change(63))); load-bearing cliff-check |
| `R_S183_FULL_REVERT` | Revert ALL FOUR S183 deltas -> S182 (conviction ON + XS 50/50 + steepness 1.0 + smooth_span 5) |
| `R_S183_FULL_REVERT` | Revert all four S183 deltas → S182 (directly measures the combined interaction) |

### 4.4 Analysis battery — `run_analysis` (T1–T22)

Runs against existing DM + Ablation checkpoints; no new backtests needed.

| Test | What it computes | Output |
|---|---|---|
| T1 | Politis-White optimal block length for MASTER | `extra_bootstrap_report.json` |
| T2 | Block-length sensitivity grid | `extra_bootstrap_report.json` |
| T3 | Romano-Wolf stepdown over the 143-variant pool (FWER=5%) | `extra_bootstrap_report.json` |
| T4 | Hansen SPA test | `extra_bootstrap_report.json` |
| T5 | One-sample block-bootstrap Sharpe CI | `extra_bootstrap_report.json` |
| T6 | Block-bootstrap max-drawdown CI | `extra_bootstrap_report.json` |
| T7 | Distributional sanity (Jarque-Bera, skew, kurt, Ljung-Box, ARCH-LM) | `thesis_battery_report.json` |
| T8 | Lo (2002) AR-adjusted and Mertens (2002) fat-tail Sharpe SE | `thesis_battery_report.json` |
| T9 | Minimum Track Record Length (Bailey-Lopez de Prado) | `thesis_battery_report.json` |
| T10 | Harvey-Liu multi-testing haircut (BHY, Bonferroni, Holm) | `thesis_battery_report.json` |
| T11 | Probability of Backtest Overfitting (CSCV) | `thesis_battery_report.json` |
| T12 | Model Confidence Set (Hansen-Lunde-Nason) | `thesis_battery_report.json` |
| T13 | Crisis-window conditional performance | `thesis_battery_report.json` |
| T14 | Henriksson-Merton + Treynor-Mazuy market timing | `thesis_battery_report.json` |
| T15 | Factor regressions (FH 5-PTFS, FF5, FF5+UMD, AQR kitchen sink) | `thesis_battery_report.json` |
| T16 | Post-2015 temporal holdout | `thesis_battery_report.json` |
| T17 | White-noise shuffle test (100 permutations of returns) | `thesis_battery_report.json` |
| T18 | Break-even cost multiplier | `thesis_battery_report.json` |
| T19 | Rolling Sharpe diagnostics (5-year window) | `thesis_battery_report.json` |
| T20 | CTA benchmark regression (DBMF proxy) | `thesis_battery_report.json` |
| T21 | Look-ahead audit (lag-1 return autocorrelation) | `thesis_battery_report.json` |
| T22 | Vol-scaled transaction costs (adversarial) | `thesis_battery_report.json` |

Also produces: `datamining_summary.csv`, `ablation_summary.csv`, `dm_significance.csv`, `ablation_significance.csv`, `master_stability.csv`, `master_by_asset_class.csv`, `deflated_sharpe.csv`, stability scatter PNG, parameter heatmap PNG, and a terminal `FINAL VERDICT` block.

### 4.5 Parsimony & random-parameter battery — `run_parsimony_battery` (T23–T28)

New in S182. Adds six tests that address two questions the T1–T22 battery does not cover:

- *Can MASTER be simplified further?* → T23, T26, T27
- *Do arbitrary parameters produce similar performance?* → T24, T25, T28

| Test | Backtests | What it answers |
|---|---:|---|
| **T23** Progressive parsimony | 8 | Cumulatively drops components in a pre-registered order (conviction → VoV direction → smoothing → overlays → fastest speed → slowest speed → XS momentum → VoV alpha). Reports the minimum-spec variant with SR ≥ 0.80 × MASTER. |
| **T24** Random-parameter MC | **N** (default 100) | Draws uniformly from 7 continuous axes (trend_fdm, vol_trigger, vol_dampen, dd_threshold, dd_scale, sigmoid_steepness [log-uniform], fdm_cap) and 6 discrete axes (smooth_span, xs_lookback, vov_window, skew_window, conviction_window, use_shifted_sigmoid). Reports MASTER's percentile rank in the empirical SR distribution. |
| **T25** Random simplex MC | **N** (default 100) | Draws alpha weights from Dirichlet(1,1,1,1) on the 4-simplex. Reports equal-weight MASTER's rank in the random-simplex SR distribution. |
| **T26** Identity strategy | 1 | Single variant with **1** EWMAC speed (64/256), all lookbacks unified to 256, all multipliers at identity. Tests whether the multi-speed ensemble is load-bearing. |
| **T27** Coarse rounding | 1 (or 0) | Rounds every numeric parameter to a coarse grid (lookbacks → nearest power of 2, scale factors → 0.1, thresholds → 0.02) and reruns. Lists which parameters actually moved. If nothing moved, MASTER has no fitted digits. |
| **T28** Parameter scramble | 15 | Adversarial mirror of T24 — 15 deliberately-broken configs (swapped lookbacks, positive DD threshold, binary cliff sigmoid, negative alpha weights, etc.). Expected: median SR ≈ 0, no scramble beats MASTER. |

### 4.6 Extended robustness battery — `run_extended_robustness` (T29–T34)

Six tests filling gaps in the T1–T22 analysis battery and the T23–T28 parsimony battery. Each test addresses a specific defensive claim the thesis needs: execution-lag survival, cross-instrument generalisation, signal-space robustness, leverage invariance, cash-yield attribution, and decade stability.

| Test | Backtests | What it answers |
|---|---:|---|
| **T29** T+1 execution lag | 1 | *Does the reported SR survive a one-day forecast delay?* Reruns MASTER with `forecast.shift(1).fillna(0)`. Defends the S180 report's live-execution claim (which quoted −0.4% SR impact but did not actually run the test in the suite). |
| **T30** Instrument leave-one-out jackknife | **~62** | *How much of MASTER depends on any single instrument?* Drops each instrument in turn and rerun. Reports jackknife mean/std/min/max, identifies the single most-disruptive and largest-boost drops, and counts how many drops still deliver ≥ 0.8 × MASTER. The strongest cross-instrument parsimony evidence available. |
| **T31** Signal noise injection | 4 | *Does MASTER degrade gracefully under Gaussian corruption of the master forecast?* Reruns at σ_noise ∈ {0.10, 0.25, 0.50, 1.00} × σ_forecast per instrument. Complements T24 (parameter-space MC) by testing signal-space robustness. A fitted strategy collapses under noise; a robust one degrades gracefully. |
| **T32** Leverage scaling | 3 | *Is SR invariant to forecast-magnitude scaling?* Scales forecasts by k ∈ {0.8, 1.0, 1.2}. In the cap-unconstrained regime, SR should be invariant (positions scale linearly → returns scale linearly → SR = mean/vol is scale-invariant). Deviation flags either `FORECAST_CAP` biting or an implicit absolute-magnitude dependence. Confirms the S181 linear-scaling claim. |
| **T33** IRX attribution | 0 | *What fraction of the reported SR comes from cash yield vs trading alpha?* Analytical decomposition: `sr_gross − sr_excess_of_irx`. Shows whether the reported SR would survive a rate-cut regime. No new backtest. |
| **T34** Calendar-decade SRs | 0 | *How does MASTER perform in each calendar decade (1990s / 2000s / 2010s / 2020s)?* Analytical cut on the MASTER `daily_ret` series. Complements `master_stability.csv`'s 5-year buckets with clean decade boundaries. No new backtest. |

## 5. Output files

### Directory tree

```
Strategy_183/
├── _run_s183_master.py
├── ig_strategy_183.py
├── ig_testing_suite_183.py
├── testing_suite_README.md            ← this file
├── Strategy_183_IG_VoV_Quad_Parsimonious_Sharpened_checkpoint.pkl  (after master run)
└── TestingSuite/
    ├── Datamining/
    │   ├── DM183_ANCHOR_checkpoint.pkl
    │   ├── DM183_ANCHOR_portfolio_returns.csv
    │   ├── DM183_A_2spd_16+64_checkpoint.pkl
    │   ├── DM183_A_2spd_16+64_portfolio_returns.csv
    │   └── ... (103 variants × 2 files)
    ├── Ablation/
    │   ├── S183_Ablation_MASTER_checkpoint.pkl
    │   ├── S183_Ablation_MASTER_portfolio_returns.csv
    │   ├── S183_Ablation_R_S183_ADD_CONV_checkpoint.pkl
    │   └── ... (69 variants × 2 files)
    ├── Parsimony/
    │   ├── PAR_T23_STEP01_drop_conviction_checkpoint.pkl
    │   ├── PAR_T24_MC0000_checkpoint.pkl
    │   └── ... (~230 checkpoint files for default N=100)
    ├── Robustness/
    │   ├── RB_T29_T_PLUS_1_LAG_checkpoint.pkl
    │   ├── RB_T30_DROP_<INSTR>_checkpoint.pkl  (one per dropped instrument)
    │   ├── RB_T31_NOISE_0.10_checkpoint.pkl
    │   ├── RB_T32_LEV_080_checkpoint.pkl
    │   └── ... (~70 checkpoint files)
    ├── Analysis/
    │   ├── datamining_summary.csv
    │   ├── ablation_summary.csv
    │   ├── dm_significance.csv
    │   ├── ablation_significance.csv
    │   ├── master_stability.csv
    │   ├── master_by_asset_class.csv
    │   ├── deflated_sharpe.csv
    │   ├── extra_bootstrap_report.json
    │   ├── thesis_battery_report.json
    │   ├── datamining_stability_scatter.png
    │   ├── datamining_heatmaps.png
    │   ├── t23_progressive_parsimony.csv
    │   ├── t24_random_parameter_mc.csv
    │   ├── t24_random_parameter_mc_summary.json
    │   ├── t25_random_simplex_mc.csv
    │   ├── t25_random_simplex_summary.json
    │   ├── t26_identity_strategy.csv
    │   ├── t27_coarse_rounding.csv
    │   ├── t28_parameter_scramble.csv
    │   ├── t28_parameter_scramble_summary.json
    │   ├── t29_t_plus_1_lag.csv
    │   ├── t30_instrument_leave_one_out.csv
    │   ├── t30_instrument_loo_summary.json
    │   ├── t31_signal_noise_injection.csv
    │   ├── t32_leverage_scaling.csv
    │   ├── t33_irx_attribution.csv
    │   └── t34_decade_sharpe.csv
    ├── ExtData/                         (optional, for T14/T15 factor data)
    │   ├── fh_factors.csv
    │   ├── ff5_factors.csv
    │   └── tsmom_factor.csv
    └── suite_failures.log               (append-only error log)
```

### Key summary CSVs

- **`datamining_summary.csv`** — one row per DM variant with `variant_id, dim, label, n_days, sr_full, sr_10, sr_15, sr_20, cagr, max_dd, calmar, ann_vol, rho_vs_master, dsr_vs_master, stat_*`. Sort by `dsr_vs_master` to find the most/least disruptive perturbations on each axis.
- **`ablation_summary.csv`** — same schema, one row per ablation variant. Sort by `stat_jkm_p` to find individually significant components.
- **`master_stability.csv`** — subperiod SRs (PRE-2000, 2000-2004, 2005-2009, 2010-2014, 2015-2019, 2020-NOW, FULL, OOS_POST2005) with HAC SEs, t-stats, CAGR/MDD, LW 95% CI.
- **`deflated_sharpe.csv`** — pool-wide Deflated Sharpe Ratio (Bailey-Lopez de Prado) + Harvey-Liu BHY haircut.
- **`t23_progressive_parsimony.csv`** — one row per cumulative drop step with `step, cumulative_drops, sr_full, sr_10, cagr, ann_vol, max_dd, note`. The final row is the minimum-spec variant.
- **`t24_random_parameter_mc.csv`** — one row per random draw with full metrics and the sampled cfg as a JSON string in `cfg_note`. Paired with `t24_random_parameter_mc_summary.json` which contains `master_sr_full`, `pct_rank_master`, `random_sr_median`, quantiles, etc.
- **`t28_parameter_scramble.csv`** — one row per scramble with a human-readable `note` describing what was broken. Paired with `t28_parameter_scramble_summary.json`.
- **`t29_t_plus_1_lag.csv`** — single row: `master_sr_full, lagged_sr_full, dsr, pct_sr_retained, master_p10_sr, lagged_p10_sr, note`. Target: `pct_sr_retained > 0.95`.
- **`t30_instrument_leave_one_out.csv`** — one row per dropped instrument with `dropped_instrument, sr_full, sr_10, cagr, max_dd, ann_vol, dsr, cfg_note`. Paired with `t30_instrument_loo_summary.json` containing the jackknife distribution statistics and the most-disruptive / most-boost instruments.
- **`t31_signal_noise_injection.csv`** — one row per noise level (baseline + 4 injected) with `noise_level, sr_full, sr_10, dsr, note`. Sort by `noise_level` to read off the SR-vs-noise curve.
- **`t32_leverage_scaling.csv`** — three rows (k ∈ {0.8, 1.0, 1.2}) with `leverage_k, sr_full, sr_10, cagr, ann_vol, max_dd, dsr_vs_master`. Target: SR range across k near zero (cap-unconstrained invariance).
- **`t33_irx_attribution.csv`** — single row: `sr_gross, sr_excess_of_irx, delta_sr_from_irx, annual_irx_rate, pct_sr_from_cash, note`.
- **`t34_decade_sharpe.csv`** — one row per decade (1990s / 2000s / 2010s / 2020s) with `decade, n_days, sr_full, cagr, ann_vol, max_dd, note`.

## 6. Reproducing the thesis headline numbers

To reproduce the S183 headline metrics (equal to the ablation target COMBO_NOCONV_NOXS_STEEP10_SMOOTH3 up to ~1 pp pipeline drift) (`SR_full = 0.986`, `CAGR = 16.05%`, `vol = 13.26%`, `MDD = 23.16%`, `post-2010 SR = 0.971`):

```bash
# 1. Generate the authoritative MASTER checkpoint.
python Strategy_183/_run_s183_master.py
# -> Strategy_183/Strategy_183_IG_VoV_Quad_Parsimonious_Sharpened_checkpoint.pkl

# 2. Run the ABL MASTER (which should reproduce the same numbers).
python Strategy_183/ig_testing_suite_183.py --ablation --variants MASTER
# -> Strategy_183/TestingSuite/Ablation/S183_Ablation_MASTER_checkpoint.pkl

# 3. Run the analysis battery to populate the CSVs.
python Strategy_183/ig_testing_suite_183.py --analysis
# -> Strategy_183/TestingSuite/Analysis/master_stability.csv
#    Expected 'FULL' row: SR_ann ≈ 0.946, CAGR ≈ 13.11%, vol ≈ 10.22%
```

The ABL MASTER and the top-level master checkpoint should produce bit-identical numbers because `ig_strategy_183.py` and `abl_build_inst_signals(..., MASTER_CFG)` use the same underlying `dm_build_variant_signals` pipeline (with the critical `combined_scalar = rolling_forecast_scalar(fdm_scaled)` step that the S180 suite was missing).

## 7. Runtime expectations

All timings are rough estimates on a single-machine baseline of ~1 minute per full 36-year backtest (varies with instrument count, C++ acceleration, and IO).

### Serial (default, `--workers 1`)

| Module | Backtests | Estimated runtime |
|---|---:|---:|
| Master run | 1 | ~3–5 min |
| DM sweep | 101 | ~1.5–2 h |
| Ablation matrix | 51 | ~50 min |
| Analysis (T1–T22) | 0 (reads checkpoints) | ~5–15 min |
| **Parsimony battery (default N=100)** | **225** | **~3.75 h** |
| - T23 | 8 | ~8 min |
| - T24 | 100 | ~100 min |
| - T25 | 100 | ~100 min |
| - T26 | 1 | ~1 min |
| - T27 | 1 | ~1 min |
| - T28 | 15 | ~15 min |
| **Robustness battery (T29–T34)** | **~70** | **~50 min** |
| - T29 | 1 | ~1 min |
| - T30 (leave-one-out) | ~62 | ~45 min |
| - T31 | 4 | ~4 min |
| - T32 | 3 | ~3 min |
| - T33 | 0 | instant |
| - T34 | 0 | instant |
| **`--all` total (default N=100)** | **~448** | **~7.5 h** |

### Parallel (`--workers 6`, typical 8-core machine)

| Module | Serial | Parallel | Speedup |
|---|---:|---:|---:|
| DM sweep (103 variants) | ~1.5–2 h | ~20–25 min | ~4–5× |
| Ablation matrix (69 variants) | ~50 min | ~10–12 min | ~4–5× |
| T24 MC (N=100) | ~100 min | ~18–22 min | ~4–5× |
| T25 simplex (N=100) | ~100 min | ~18–22 min | ~4–5× |
| T30 instrument leave-one-out (62) | ~45 min | ~10 min | ~4–5× (when parallelised) |
| Analysis (T1–T22) | ~5–15 min | ~5–15 min | 1× (not parallelised) |
| T23 / T26 / T27 / T28 / T29 / T31 / T32 / T33 / T34 | ~30 min | ~30 min | 1× (not parallelised) |
| **`--all` total (default N=100)** | **~7.5 h** | **~1h 50min** | **~4×** |

Startup overhead (pickling the Phase-0 library to each worker process) is ~5–15 s and is paid once per module.

For a faster first pass, use `--mc-N 30` (~90 min serial; ~25 min with `--workers 6`). For thesis-grade publication-quality parsimony inference, use `--mc-N 500` (~17 h serial; ~4 h with `--workers 6`, overnight job).

The robustness battery by itself is the cheapest of the three heavy modules (~50 min serial, ~12 min parallel) so `--robustness` on its own is the fastest way to get a sanity check on a just-rebuilt MASTER.

## 8. Interpreting key outputs

### The Analysis module

- **`ablation_summary.csv` → `stat_jkm_p`** — any variant with `stat_jkm_p < 0.05` is a statistically-distinguishable departure from MASTER. For an S182 robustness claim, you want the *reversion* variants (`R_S183_ADD_CONV`, `R_S183_ADD_XS`, etc.) to have **significant** p-values showing the reverts materially hurt (i.e. S183 deltas are genuinely load-bearing improvements) (the deltas are cost-free) AND the structural components (overlays, VoV, trend) to have **significant** p-values (removing them is disruptive).
- **`datamining_summary.csv` → `sr_full` across each axis** — the axis is "flat" if the SR range across the sweep is ≤ 0.10. Flat = robust. Steep = fitted.
- **`deflated_sharpe.csv`** — the BHY column is the honest headline. Expected: `BHY SR ≈ 0.74`.
- **`master_stability.csv`** — every subperiod should have `SR > 0`. The weakest subperiod (typically 2005–2009) is expected at SR ≈ 0.4 with t-stat ≈ 0.8.

### The Parsimony module

- **T23 → minimum-spec step** — if the minimum-spec variant with SR ≥ 0.80 × MASTER is at step 4 or later, the architecture tolerates substantial simplification. If it's at step 1 or 2, the strategy is tightly coupled to its full component set.
- **T24 → `pct_rank_master`** — **this is the single most important number in the parsimony battery**. The interpretation:
  - `pct_rank ∈ [0.50, 0.70]` → random parameters work about as well as MASTER. This is the strongest possible defence of the parameter choices: they're indistinguishable from any other reasonable draw, i.e. not tuned.
  - `pct_rank ∈ [0.25, 0.50]` → random parameters typically work *better* than MASTER. This is possible and is not bad news — it means the default is conservative.
  - `pct_rank ∈ [0.90, 1.00]` → MASTER is at the very top of the random distribution. **This is a warning sign**: the reported SR depends on the specific parameter values.
- **T25 → percentile of equal-weight** — same interpretation, but for the alpha-weight choice specifically.
- **T26 / T27 → `dsr`** — if |dSR| < 0.05, the simplification is free. If dSR < −0.10, the simplification is costly.
- **T28 → `n_above_master`** — must be 0 (no scramble should beat MASTER). If any do, investigate.

### The Robustness module

- **T29 → `pct_sr_retained`** — the fraction of MASTER's SR that survives a 1-day forecast lag. Target: **≥ 0.95** (S182 retained ~0.98 of its SR under T+1 lag). If T29 shows `pct_sr_retained < 0.90`, the reported SR has a material latency component and the strategy is not cleanly deployable under realistic execution. If `pct_sr_retained > 0.98`, latency is a non-issue and you can quote S182 as "T+1 robust" with direct evidence.
- **T30 → jackknife distribution + `max_disruption_dsr`** — **the strongest cross-instrument parsimony evidence available**. Interpretation:
  - `jackknife_sr_std < 0.02` and `max_disruption_dsr > −0.05` → every instrument contributes equally; MASTER does not depend on any single instrument. Strong result.
  - `max_disruption_dsr < −0.10` → one specific instrument is driving ≥ 10% of MASTER's SR. The instrument is named in `max_disruption_instrument`. Worth investigating as a concentration-risk flag.
  - `n_above_0_8_master ≥ 50/62` → even dropping the worst instrument leaves ≥ 80% of MASTER intact. Strong.
- **T31 → SR vs noise curve** — read off `sr_full` at each `noise_level`. Expected shape: graceful monotone decay from SR_master at `noise=0` to near-zero at `noise=1.0*sigma_fc`. Key checkpoints:
  - `noise = 0.10`: should retain ≥ 95% of SR (trivial perturbation).
  - `noise = 0.25`: should retain ≥ 80% (moderate noise).
  - `noise = 0.50`: should retain ≥ 50%.
  - `noise = 1.0`: should retain ≥ 20% (at 1× sigma the signal is dominated by noise; any retention above zero is real alpha, not coincidence).
  - A sharp cliff (e.g. SR collapsing from 0.90 to 0.10 at `noise = 0.25`) means the signal is fragile — a small perturbation destroys it.
- **T32 → `sr_range` across `leverage_k`** — should be approximately zero. Concretely, `|sr(k=0.8) − sr(k=1.2)| < 0.02` confirms cap-unconstrained invariance. If the range is > 0.05, either the cap is binding in the k=1.2 run (check `max_forecast` in the checkpoint) or there's an implicit absolute-magnitude dependence somewhere.
- **T33 → `pct_sr_from_cash`** — what fraction of the reported SR comes from the T-bill cash yield on collateral vs pure trading alpha. Expected: ~10–25% for a 20-year sample where IRX averaged ~2–4%. **If `pct_sr_from_cash > 0.50`, the reported SR is mostly cash yield and the strategy is vulnerable to a rate-cut regime.** Note: S183 quotes excess-of-IRX SR = 0.986; the gross (cash-included) SR will be higher by roughly `ann_irx / ann_vol`.
- **T34 → decade-by-decade SR table** — every decade should have `sr_full > 0`. A negative decade is a major flag. Expected ranges for S182: 1990s ~0.85, 2000s ~0.70, 2010s ~1.10, 2020s ~0.70, driven by the relative strength of trend regimes in each decade. The 2010s should be strongest (Euro-crisis + China-deval + 2022 inflation all produce strong TF-alpha).

### The terminal "FINAL VERDICT" block

Prints at the end of `run_analysis`. Expected output on a healthy run:

```
  None. S183 MASTER survives the full DM multi-testing bar.
```

followed by the per-axis neighbour-gap analysis and the bootstrap parameters. If any DM variant beats MASTER after the Romano-Wolf FWER correction, its variant ID and dSR will be printed there.

## 9. Troubleshooting

### "No MASTER checkpoint available. Run --ablation first."

The analysis / parsimony modules need a MASTER checkpoint. Either:
```bash
python Strategy_183/ig_testing_suite_183.py --ablation          # creates ABL MASTER
# or
python Strategy_183/_run_s183_master.py                          # creates top-level MASTER
```
Then retry `--analysis` / `--parsimony`.

### "[THESIS] skipped -- MASTER checkpoint missing"

Same cause. T7–T22 need MASTER to be present.

### T14/T15 factor regressions are all marked "skipped"

The suite could not find the external factor CSVs. Expected locations:
- `Strategy_183/TestingSuite/ExtData/<stem>.csv`
- `Strategy_180/TestingSuite/ExtData/<stem>.csv` (fallback)
- `Strategy_183/ExtData/<stem>.csv`
- `IG_Backtest/ExtData/<stem>.csv`

Download the Fama-French, UMD, AQR TSMOM and Fung-Hsieh PTFS factor files yourself and place them in one of the above locations.

### Parsimony battery is too slow

Use `--mc-N 30` for a ~3x speedup, or `--workers N` to parallelise T24/T25 across cores. If you only want T23, T26, T27, T28 (the cheap tests, no Monte Carlo), temporarily comment out the T24 and T25 calls in `run_parsimony_battery` at the bottom of `ig_testing_suite_183.py`.

For a cheap sanity check that does NOT require the parsimony battery, run the extended robustness battery (`--robustness`) on its own. It's ~50 min serial or ~12 min with `--workers 6` and produces T29 (T+1 lag), T30 (leave-one-out), T31 (noise), T32 (leverage), T33 (IRX attribution), T34 (decade SRs) — a good first-line cross-check on the MASTER checkpoint before committing to the full parsimony run.

### Parallel run hangs or workers crash silently

All failures are written to `suite_failures.log`. Common causes with `--workers > 1`:

- **Windows "freeze_support" error** — only occurs if the script is run without the `if __name__ == "__main__"` guard. This guard is already present at the bottom of `ig_testing_suite_183.py`; do not remove it.
- **Memory pressure** — each worker process receives a full copy of the Phase-0 library (~500 MB for a 70-instrument universe). On a machine with <8 GB RAM, reduce `--workers` or use `--workers 1`.
- **Disk contention** — many workers writing checkpoint files simultaneously can saturate an HDD. Use an SSD or reduce `--workers`.
- **Worker OOM-killed** — the OS killed a worker due to memory. Check system RAM and reduce `--workers`.

Progress bars show completed variants, not elapsed time. If a bar stalls for >5 min, check `suite_failures.log`.

### Out-of-disk on Parsimony/

Each parsimony variant writes a `PAR_*_checkpoint.pkl` and a `PAR_*_portfolio_returns.csv`. At default N=100 that's ~500 files totalling a few hundred MB. To clean up:
```bash
rm Strategy_183/TestingSuite/Parsimony/PAR_T24_MC*
rm Strategy_183/TestingSuite/Parsimony/PAR_T25_SMX*
```
The summary CSVs and JSON files under `Analysis/` are the actual inference outputs; the per-variant checkpoints are only needed if you want to re-analyse individual draws.

### "ValueError: variant ...: cfg requests windows not in the pre-baked library"

Raised by the pre-flight coverage check in `dm_build_variant_signals` (or `abl_build_inst_signals`). A variant is requesting a window (VoV, skew, XS lookback, or conviction) that `dm_precompute_library` did not pre-bake. This is intentional — it surfaces library mismatches loudly instead of silently zeroing the missing alpha component (which was the S180-inherited bug that contaminated the first S182 suite run).

To fix, add the missing value to the corresponding `DM_ALL_*` list in `ig_testing_suite_183.py` (around line 1380) and rerun `run_datamining`. The required value is printed in the error message. Example:

```
ValueError: [dm_build_variant_signals] variant 'T24_MC0042': cfg requests
  windows not in the pre-baked library. Errors:
  - vov_window=200 not in DM_ALL_VOV_WINDOWS=[5, 16, 21, 42, 63, 64, 90, 126, 2048]
```

Add `200` to `DM_ALL_VOV_WINDOWS` and rerun. Same pattern for `DM_ALL_XS_LOOKBACKS`, `DM_ALL_SKEW_WINDOWS`, `DM_ALL_CONV_WINDOWS`, `DM_ALL_SPEED_PAIRS`.

### Robustness battery T30 takes much longer than the other tests

T30 (instrument leave-one-out) runs N backtests where N = number of instruments (~62). It is the single most expensive test in the whole suite after T24/T25 and it is not yet parallelised. Options:

1. Run it with `--robustness --workers 6` on a multi-core machine. (T30 itself is serial internally, but running it alongside other tests won't block them.)
2. Skip T30 temporarily by running `--robustness` with the other tests only. (No CLI flag for this yet; comment out the T30 line in `run_extended_robustness` if you need to skip it for a fast iteration.)
3. Reduce the universe to a smaller subset before running — but this changes what "MASTER" means, so only do this for isolated debugging.

### "suite_failures.log" is growing

Per-variant exceptions are appended to this log rather than halting the whole suite. Tail it for the failure messages and tracebacks:
```bash
tail -n 200 Strategy_183/TestingSuite/suite_failures.log
```
Common causes: a particular random-sample variant produces degenerate signals (all-zero forecasts); IRX or Panama data missing for a specific date range; instrument mapping out of sync with the Panama file directory.

### "Using top-level S180 checkpoint ..."

The analysis module's fallback found a top-level checkpoint when looking for MASTER. This is fine: the fallback path checks `Strategy_183/<STRATEGY_NAME_MASTER>_checkpoint.pkl` which is `Strategy_183_IG_VoV_Quad_Parsimonious_Sharpened_checkpoint.pkl`. If the fallback text prints, confirm the file path is under `Strategy_183/` and matches the S183 master name.

## 10. Extending the suite

To add a new ablation variant, edit `ABL_VARIANTS` in `ig_testing_suite_183.py` (~line 1930). Each entry is a call to `_abl_cfg(label=..., **overrides)`, where the overrides are the cfg keys that differ from the S183 MASTER baseline.

To add a new DM axis, edit `_build_dm_variants()` (~line 1387). Append a new block following the `V[f"<AXIS>_<value>"] = dict(..., label=..., dim="<AXIS>: <name>")` pattern. Then, if your axis references a parameter that isn't yet in the Phase-0 library, extend `dm_precompute_library` to precompute any new per-instrument signals.

To add a new parsimony test, append a `t**N**_*` function alongside T28 and add a `try/except` call in `run_parsimony_battery`. Use `_run_parsimony_variant(library, cfg, vid, paths)` to evaluate any cfg you build. Follow the T24 template for MC tests and the T26 template for single-variant tests.

To add a new robustness test, append a `t**N**_*` function alongside T34 and add a `try/except` call in `run_extended_robustness`. Use `_run_robustness_variant(library, cfg, vid, paths, postprocess=...)` if your test needs to mutate the per-instrument forecasts (T29/T31/T32 do this via a `postprocess` callable that takes `inst_signals` and returns a modified dict). For analytical tests that only need the MASTER checkpoint and no new backtests (T33/T34 style), just read `master_ck["daily_ret"]` directly and write to `out_dir / "tNN_*.csv"`.

**Important:** any new test that requests a window (VoV, skew, XS lookback, conviction, or EWMAC speed pair) must first verify that value is in the corresponding `DM_ALL_*` list near line 1380. The pre-flight coverage check in `dm_build_variant_signals` will raise a `ValueError` otherwise — see the troubleshooting entry on coverage-check errors.

## 11. Files in this directory

| File | Purpose |
|---|---|
| `_run_s183_master.py` | Thin runner that calls `run_strategy()` from `ig_strategy_183.py` |
| `ig_strategy_183.py` | The S182 strategy itself: byte-exact replica of the Apr-7 16:36 build with an honest docstring |
| `ig_testing_suite_183.py` | Single-file testing suite (this README documents its usage) |
| `Illustrations/S182_Strategy_Explanation.tex` | Plain-language LaTeX explanation of S182's mechanics |
| `testing_suite_README.md` | This file |
| `Strategy_183_IG_VoV_Quad_Parsimonious_Sharpened_checkpoint.pkl` | Generated by `_run_s183_master.py` |
| `TestingSuite/` | All suite outputs (see §5 above) |

## 12. References

The testing suite implements statistical methods from:

- Bailey, D. & Lopez de Prado, M. (2014). *The deflated Sharpe ratio.* Journal of Portfolio Management 40(5), 94–107.
- Harvey, C., Liu, Y. & Zhu, H. (2016). *…and the cross-section of expected returns.* Review of Financial Studies 29(1), 5–68.
- Hansen, P. (2005). *A test for superior predictive ability.* JBES 23(4), 365–380.
- Hansen, P., Lunde, A. & Nason, J. (2011). *The Model Confidence Set.* Econometrica 79(2), 453–497.
- Jobson, J. & Korkie, B. (1981). *Performance hypothesis testing with the Sharpe and Treynor measures.* Journal of Finance 36(4), 889–908.
- Ledoit, O. & Wolf, M. (2008). *Robust performance hypothesis testing with the Sharpe ratio.* Journal of Empirical Finance 15(5), 850–859.
- Lo, A. (2002). *The statistics of Sharpe ratios.* Financial Analysts Journal 58(4), 36–52.
- Memmel, C. (2003). *Performance hypothesis testing with the Sharpe ratio.* Finance Letters 1, 21–23.
- Mertens, E. (2002). *Comments on variance of the IID estimator in Lo (2002).* Working paper.
- Newey, W. & West, K. (1987). *A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix.* Econometrica 55(3), 703–708.
- Politis, D. & Romano, J. (1994). *The stationary bootstrap.* JASA 89(428), 1303–1313.
- Romano, J. & Wolf, M. (2005). *Stepwise multiple testing as formalized data snooping.* Econometrica 73(4), 1237–1282.

See `testing_suite_README.md` in `Strategy_182/` for the original post-bug-fix README that this file was derived from, and `S180_FULL_REPORT.md` (under `Strategy_180/`) for the full narrative interpretation of the statistical battery.
