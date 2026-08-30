# S183 WGAN Robustness Analysis

## Real Backtest Baseline (S183, 1990-2026)
- **Sharpe Ratio:** 0.9725
- **CAGR:** 15.49%
- **Max Drawdown:** 21.59%
- **Calmar:** 0.7175

## Synthetic Path Distribution (1000 paths x 504 days)

> **Note:** Synthetic P&L is gross (pre-cost). Real SR=0.93 includes commissions
> (~$2.5B cumulative). Synthetic SRs are therefore upward-biased vs. real.

### Sharpe Ratio Distribution
- Mean: 0.1441  |  Median: 0.2532
- Std:  0.6686
- 5th pct: -0.9236  |  95th pct: 1.0871
- Paths with SR > real baseline (0.97): **8.3%**
- Paths with SR > 0.5: **38.9%**
- Paths with SR > 0:   **58.4%**

### CAGR Distribution
- Mean: 170.47%  |  Median: 126.62%
- 5th-95th pct range: [0.00%, 509.08%]

### Max Drawdown Distribution
- Mean: 364.77%  |  Median: 241.51%
- 5th-95th pct range: [176.03%, 967.84%]

### Calmar Distribution
- Mean: 0.8005  |  Median: 0.4245

## WGAN Validation
- **QQ MSE (mean over instruments):** 3468.95374  (threshold: 0.05)
- **Correlation Frobenius norm (normalised):** 0.0010  (threshold: 0.01)
- **ACF SSD (mean):** 0.0278  (threshold: 0.1)
- **PCA top-3 eigenvalue max relative error:** 0.1448  (threshold: 0.2)
  - Real eigvals: ['10.84', '6.63', '4.66']  (35.7% variance explained)
  - Synth eigvals: ['12.38', '7.59', '4.67']  (39.8% variance explained)

### Validation Warnings
- QQ MSE (3468.9537) exceeds threshold 0.05.
- Kurtosis ratio pass rate (0.0%) below threshold 80%. Synthetic tails may be too thin (mode collapse symptom).

## Interpretation

The distribution of SR across 1000 synthetic paths quantifies how much of
S183's historical SR is structural edge vs. path-dependent luck.

- Real SR (0.97) exceeds 8.3% of synthetic paths.
  p-value (one-sided, SR above real): 0.917

## Configuration
- Architecture: CWGAN | full 62-instrument, no copula
- Instruments: 62
- WGAN epochs: 3000  |  Batch: 128
- Noise dim: 256  |  Window: 20
- Paths: 1000  |  Path length: 504 days