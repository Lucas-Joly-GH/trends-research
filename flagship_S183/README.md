# Flagship — Strategy S183 (`IG VoV Quad Sharpened`)

The promoted thesis strategy: a four-alpha equal-weight composite with **zero backtest-fitted
parameters**. Raw Sharpe 0.97, BHY-corrected 0.78 at K = 118 audited trials.

| File | Role |
|---|---|
| `ig_strategy_183.py` | The strategy: signal construction, pooled 4×4 FDM, sigmoid overlays. |
| `ig_testing_suite_183.py` | Native testing suite (ablation, datamining, statistical battery). |
| `S183_FULL_REPORT.md` | The complete write-up (also as PDF in [`../thesis/`](../thesis/S183_Final_Report.pdf)). |
| `figures/` | Equity & drawdown, return distribution, rolling Sharpe, tail-risk. |
| `robustness_runs/` | Half-life sweep, liquidity-weighting, Vol-of-Vol direction sweep, signal-shuffle null test, capacity analysis. |

Every parameter is either a closed-form derivation or a literature citation. See
§4.4 of the report for the full parameter-sourcing audit.
