# S183 WGAN-GP Robustness Testing

Synthetic market generation pipeline for assessing the robustness of Strategy 183
(62-instrument futures portfolio, 1990-2026). A Conditional WGAN-GP learns the
joint distribution of daily price differences across all instruments, generates
1000 synthetic 2-year paths, and backtests the strategy on each to quantify how
much of the historical Sharpe ratio is structural edge vs. path-dependent luck.

---

## v1 Results (Broken) -- April 2025

The initial Colab run produced catastrophically incorrect results. All validation
metrics failed, and the robustness test output was not interpretable.

### v1 Result Summary

| Metric                  | Real Baseline | Synthetic (v1) | Status   |
|-------------------------|---------------|----------------|----------|
| Sharpe Ratio            | 0.93          | 5.91 (mean)    | BROKEN   |
| CAGR                    | 12.43%        | 0.26%          | BROKEN   |
| Max Drawdown            | 20.35%        | 0.04%          | BROKEN   |
| Annualized Volatility   | ~13%          | ~0.05%         | BROKEN   |
| Calmar Ratio            | 0.61          | 9.10 (mean)    | BROKEN   |

### v1 Validation Metrics

| Metric                          | Value     | Threshold | Status              |
|---------------------------------|-----------|-----------|---------------------|
| QQ MSE (mean over instruments)  | 4111.33   | 0.05      | FAIL (5 orders off) |
| Correlation Frobenius (norm)    | 0.0085    | 0.01      | PASS (barely)       |
| ACF SSD (returns mean)          | 0.0588    | 0.05      | FAIL                |
| PCA top-3 max relative error    | 1.85      | 0.20      | FAIL (9x threshold) |
| Kurtosis pass rate              | 17.7%     | 80%       | FAIL                |

### v1 Diagnosis: Why It Failed

The synthetic paths had near-zero volatility and near-zero returns, producing
absurdly high Sharpe ratios (tiny positive returns / tiny denominator). This was
caused by five interacting defects:

#### 1. NaN-to-Zero Data Poisoning (Critical)

`data_loader.py` filled pre-launch instrument dates with 0.0 across the full
9,341-date training range. Instruments that launched after 1990 (many launched
2000-2015) had thousands of artificial zero returns. The preprocessor treated
all 9,341 values per instrument as real observations, distorting rank
distributions: ~60% of values were zeros for late-launching instruments,
compressing the real return distribution into a narrow band of normal scores.

The WGAN learned that "most returns are near zero" and faithfully reproduced
near-constant outputs -- a correct response to corrupted training data.

#### 2. BatchNorm Mode Collapse in Generator

The convolutional generator used `nn.BatchNorm1d` in all conv blocks. BatchNorm
computes mean/variance across the entire batch, then normalizes each sample
toward the same statistics. This destroys inter-sample diversity -- the
fundamental requirement for a GAN generator. Combined with defect #1, this
caused the generator to collapse to a single near-zero mode.

#### 3. Critic Too Weak

Both the generator and critic used identical channel configurations
`[256, 256, 128, 64]`. WGAN-GP requires the critic to be substantially more
powerful than the generator (the critic must accurately estimate Wasserstein
distance for the GP penalty to work). An underpowered critic cannot distinguish
subtle distributional differences, so the generator converges to any mode that
satisfies the weak critic.

#### 4. No Anti-Mode-Collapse Mechanism

The generator loss was purely adversarial: `loss_g = -critic(fake_r, window).mean()`.
No diversity incentive, no moment matching, no feature matching. For a 62-dimensional
output space, pure adversarial training is highly susceptible to mode collapse.

#### 5. Configuration Bugs

- Local `config.py` had `CRITIC_WARMUP_EPOCHS = 0` (disabled warmup entirely)
- Colab config used `MEMORY = 20` instead of `WINDOW_SIZE = 20` (undefined variable)
- `VAL_QQ_MSE_WARN = 1.0` in config vs. 0.05 in the summary (inconsistent threshold)
- Cosine annealing with only 2000 epochs killed the learning rate by epoch ~1500

---

## v2 Fixes Applied

### Fix 1: Validity Mask for Training Data

**Files:** `data_loader.py`, `preprocessor.py`, `run_pipeline.py`

Added `compute_validity_mask()` which returns a `(T, D)` boolean array identifying
where each instrument has real (non-artificial) data. For each instrument, the mask
is `True` from its first non-zero observation onward.

The `ReturnPreprocessor` now accepts this mask in `fit()` and `transform()`:
- Ranks and normal scores are computed per-instrument using only valid rows
- `sorted_originals` is stored as a list of per-instrument arrays (ragged, different
  lengths) rather than a single `(T, D)` array
- `inverse_transform()` uses per-instrument quantile grids

This ensures that an instrument with 4,000 real observations uses only those
4,000 values for its rank/quantile statistics, not 9,341 values padded with
5,341 artificial zeros.

### Fix 2: InstanceNorm Replaces BatchNorm

**File:** `cwgan_model.py`

Replaced `nn.BatchNorm1d` with `nn.InstanceNorm1d(affine=True)` in the
convolutional generator. InstanceNorm normalizes per-sample per-channel
(statistics computed over the W=20 time dimension for each batch element
independently), preserving inter-sample diversity.

### Fix 3: Stronger Critic (Separate Channels)

**File:** `config.py`, `run_pipeline.py`

Introduced separate channel configs:
- Generator: `CWGAN_GEN_CHANNELS = [256, 256, 128, 64]`
- Critic: `CWGAN_CRITIC_CHANNELS = [512, 512, 256, 128]` (2x wider)

The critic now has substantially more parameters than the generator, enabling
it to accurately estimate Wasserstein distance for 62-dimensional outputs.

### Fix 4: Moment Matching Regularization

**File:** `trainer.py`

Added a moment matching penalty to the generator loss:

```
loss_g = loss_g_adversarial + lambda * (mean_penalty + std_penalty)
```

Where:
- `mean_penalty = MSE(fake_batch_mean, real_batch_mean)` per instrument
- `std_penalty = MSE(fake_batch_std, real_batch_std)` per instrument
- `lambda = MOMENT_PENALTY_LAMBDA = 10.0`

This prevents the generator from collapsing to a single mode with wrong mean
or variance. A fresh real batch is sampled for each moment comparison to avoid
gradient leakage from the critic's computation graph.

### Fix 5: Configuration Fixes

**File:** `config.py`, `colab_run.ipynb`

- `MODEL_TYPE = "cwgan"` (was `"dense"`)
- `N_EPOCHS = 3000` (was 5000 dense / 2000 colab)
- `CRITIC_WARMUP_EPOCHS = 500` (was 0)
- `CRITIC_ITERS_WARMUP = 10` (was 5)
- `VAL_QQ_MSE_WARN = 0.05` (was 1.0)
- Fixed `MEMORY = 20` bug in Colab config (now `WINDOW_SIZE = 20`)

---

## Architecture

### Convolutional Generator (CWGAN)

```
Input:
  z      (B, 256)       noise vector
  window (B, 20, 62)    last 20 days of scaled returns

Pipeline:
  z -> Linear -> reshape -> (B, 62, 20)     noise feature map
  window -> permute      -> (B, 62, 20)     memory
  concat along channels  -> (B, 124, 20)

  Conv1d blocks (channels=[256, 256, 128, 64]):
    Conv1d(in, out, k=3, pad=1) -> InstanceNorm1d -> LeakyReLU(0.2)

  Global average pool    -> (B, 64)
  Linear(64, 62) -> Tanh -> (B, 62)         output in [-1, 1]
```

### Convolutional Critic (CWGAN)

```
Input:
  r      (B, 62)        real or fake return vector
  window (B, 20, 62)    conditioning context

Pipeline:
  r -> unsqueeze         -> (B, 62, 1)
  window -> permute      -> (B, 62, 20)
  concat along time      -> (B, 62, 21)

  Conv1d blocks (channels=[512, 512, 256, 128]):
    SpectralNorm(Conv1d(in, out, k=3, pad=1)) -> LeakyReLU(0.2)

  Global average pool    -> (B, 128)
  SpectralNorm(Linear(128, 1))               unbounded score
```

### Preprocessing Pipeline

```
Raw price diffs -> [validity mask: skip pre-launch zeros]
                -> Rank (per instrument, valid rows only)
                -> Probit (norm.ppf(rank / (n+1)))
                -> MinMax scale to [-1, 1]

Inverse:
  [-1, 1] -> MinMax inverse -> norm.cdf -> empirical quantile interpolation
```

---

## How to Run

### Local (CPU)

```bash
cd "Part 3"
python -m IG_Backtest.Strategy_183.GAN_Robustness.run_pipeline --retrain --epochs 3000
```

Note: CPU training for 3000 epochs with 62 instruments takes ~8-12 hours.

### Google Colab (GPU, recommended)

1. Upload `PanamaMethod/` and `GAN_Robustness/` folders to Google Drive root
2. Open `colab_run.ipynb` in Colab
3. Select GPU runtime (Runtime -> Change runtime type -> T4 GPU)
4. Run all cells

Estimated GPU time: ~2-3 hours for 3000 epochs.

### CLI Options

```
--retrain       Re-train the WGAN even if wgan_model.pt exists
--n-paths N     Number of synthetic paths (default: 1000)
--epochs N      Training epochs (default: 3000)
--no-gpu        Force CPU even if CUDA is available
```

---

## Validation Thresholds

After training, the pipeline runs statistical validation comparing pooled
synthetic returns against real training data:

| Metric                          | Threshold | Description                           |
|---------------------------------|-----------|---------------------------------------|
| QQ MSE (mean)                   | < 0.05    | Quantile-quantile fit per instrument  |
| Correlation Frobenius (norm)    | < 0.01    | Correlation structure preservation    |
| ACF SSD (returns mean)          | < 0.10    | Autocorrelation structure             |
| PCA top-3 max relative error    | < 0.20    | Factor structure preservation         |
| Kurtosis pass rate              | > 80%     | Tail behavior (kurtosis ratio 0.5-2x) |

All thresholds are informational warnings, not hard gates.

---

## Output Files

| File                          | Description                                    |
|-------------------------------|------------------------------------------------|
| `gan_backtest_results.json`   | Real vs synthetic performance distributions    |
| `gan_validation_report.json`  | Full validation metrics per instrument         |
| `gan_robustness_summary.md`   | Human-readable summary                         |
| `wgan_model.pt`               | Trained generator + critic state dicts         |
| `preprocessor.pkl`            | Fitted preprocessor (for inference)            |
| `synthetic_paths/path_*.parquet` | Individual synthetic path data (62 instruments) |

---

## Key Design Decisions

1. **Price differences, not log-returns:** Panama back-adjusted futures prices can
   be negative (e.g., ZN has 1,342 negative closes), making log-returns undefined.

2. **Full 62-instrument WGAN (no copula):** The WGAN generates all 62 dimensions
   directly. A copula-based approach (WGAN on subset + Gaussian copula for the rest)
   is available by setting `CORE_INSTRUMENTS` to a list in `config.py`.

3. **Gross P&L comparison:** Synthetic backtests are gross (pre-cost). The real
   S183 SR=0.93 includes ~$2.5B cumulative commissions over 36 years. Synthetic
   SRs are therefore upward-biased relative to a net-of-cost comparison.

4. **Exposure anchoring:** Each synthetic backtest uses exposures (positions) from
   the last 504 days of the real backtest, representing the most recent regime.
