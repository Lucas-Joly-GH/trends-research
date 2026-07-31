"""
config.py — All constants, paths, and hyperparameters for the WGAN robustness pipeline.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parents[4]  # auto-detect from config.py location
PANAMA_DIR     = PROJECT_ROOT / "Data" / "PanamaMethod"
CHECKPOINT_PKL = (
    PROJECT_ROOT / "Part 3" / "IG_Backtest" / "Strategy_183"
    / "Strategy_183_IG_VoV_Quad_Sharpened_checkpoint.pkl"
)
MAPPING_CSV    = PROJECT_ROOT / "Général" / "instrument_mapping.csv"
OUT_DIR        = (
    PROJECT_ROOT / "Part 3" / "IG_Backtest" / "Strategy_183" / "GAN_Robustness"
)
SYNTH_DIR      = OUT_DIR / "synthetic_paths"

# ---------------------------------------------------------------------------
# Core instruments.
#
# Set to None to use ALL instruments from the checkpoint (62 total).
# In that mode the Gaussian copula is skipped entirely: the WGAN generates
# all 62 dimensions directly, using the full checkpoint date range (9,341 rows)
# with NaN -> 0 fills for pre-launch instruments.
#
# Set to a list to use a specific subset + copula for the remainder.
# ---------------------------------------------------------------------------
CORE_INSTRUMENTS = None   # None = all 62, no copula

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
WINDOW_SIZE = 20        # rolling conditioning context (days)

# ---------------------------------------------------------------------------
# WGAN-GP hyperparameters — scaled for D=62 output dimensions
# (following Amundi 2020 Section 3.3.4, scaled up for higher dimensionality)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Model architecture selection
# ---------------------------------------------------------------------------
# "cwgan" (Approach B) — 1D convolutional architecture; better captures
#   temporal patterns (autocorrelation, vol clustering); recommended for D=62.
#   ~6-8h on CPU for 2000 epochs.
#
# "dense" (Approach A) — fully-connected MLP with LR schedule and more critic
#   steps; slower convergence for high D.  Use --epochs 5000.
#   ~12h on CPU for 5000 epochs.
# ---------------------------------------------------------------------------
MODEL_TYPE = "cwgan"    # "dense" or "cwgan"

# ---------------------------------------------------------------------------
# Shared WGAN-GP hyperparameters
# ---------------------------------------------------------------------------
NOISE_DIM           = 256                       # ~4x D=62
LEAKY_SLOPE         = 0.2                       # 0.2 is standard for conv networks
GRAD_PENALTY_LAMBDA = 10.0
N_CRITIC_STEPS      = 5                         # critic updates per generator step (default)
LR                  = 1e-4
BETA1               = 0.5
BETA2               = 0.9
N_EPOCHS            = 3000                      # CWGAN with cosine annealing
BATCH_SIZE          = 128                      # 256 OOM'd on 8GB RTX 4060 Laptop with attention critic; 128 is safe
TRAIN_JITTER_STD    = 0.0                       # was 0.01; disabled -- extra jitter blurs the target distribution

# LR schedule (Approach A: cosine annealing; Approach B: None = constant)
LR_SCHEDULE          = "cosine"   # None or "cosine"
LR_MIN               = 1e-6       # minimum LR at end of cosine schedule

# Warmup critic iterations (extra critic steps in early epochs helps critic
# lead the generator, especially important for high-dimensional outputs)
CRITIC_ITERS_WARMUP  = 5          # was 10; reduced for 8GB VRAM (less memory churn during warmup)
CRITIC_WARMUP_EPOCHS = 500        # number of warmup epochs

# Dense-MLP specific (used only when MODEL_TYPE="dense")
GENERATOR_LAYERS    = [1024, 512, 256, 128]     # hidden dims; output layer D appended
CRITIC_LAYERS       = [512, 256, 128]           # hidden dims; output = 1
SPECTRAL_NORM       = True                      # apply spectral norm to critic layers

# Conv-WGAN specific (used only when MODEL_TYPE="cwgan")
CWGAN_GEN_CHANNELS    = [256, 256, 128, 64]      # generator channel progression
CWGAN_CRITIC_CHANNELS = [512, 512, 256, 128]     # critic channels (wider for stronger critic)
CWGAN_CHANNELS        = CWGAN_GEN_CHANNELS       # backward-compat alias
CWGAN_KERNEL_SIZE     = 3                        # conv kernel size
CWGAN_SPECTRAL_NORM   = True                     # spectral norm on all conv+linear layers

# Moment matching regularization (anti-mode-collapse)
# Rebalanced (v3) after observing "generator outputs unconditional mean"
# collapse at lambdas 10/5/2 -- with synthetic SR median 3.20 vs real 0.97.
# New lambdas are gentle nudges that let the Wasserstein critic drive training.
MOMENT_PENALTY_LAMBDA = 1.0                      # weight for mean+std matching penalty (was 10.0)
COV_PENALTY_LAMBDA    = 0.5                      # weight for cross-instrument covariance matching (was 5.0)
KURT_PENALTY_LAMBDA   = 0.2                      # weight for per-instrument log-kurtosis matching (was 0.1)
SORT_MATCH_LAMBDA     = 1.0                      # weight for per-instrument sorted-samples (1D W2) matching.
                                                 # Directly targets marginal QQ mismatch that moment/kurt can't fix.

# Checkpoint saving
CHECKPOINT_EVERY      = 250                      # save checkpoint every N epochs (+ always on final)

# ---------------------------------------------------------------------------
# Path generation
# ---------------------------------------------------------------------------
N_PATHS         = 1000
PATH_LENGTH     = 504        # ~2 trading years
GEN_SEED_BASE   = 1000       # path i uses seed GEN_SEED_BASE + i
TRAIN_SEED      = 42

# ---------------------------------------------------------------------------
# Simplified backtest
# ---------------------------------------------------------------------------
EXPOSURE_MIN_RET = 1e-6      # |hist_price_diff| below this -> exposure = 0
EXPOSURE_CLIP    = 1e10      # absolute exposure clip (USD / price-point)
INITIAL_EQUITY   = 1e8       # $100M, matches checkpoint equity[0]

# ---------------------------------------------------------------------------
# Validation thresholds (informational — not hard gates)
# ---------------------------------------------------------------------------
VAL_FROBENIUS_WARN    = 0.01   # warn if normalized Frobenius norm exceeds this
VAL_QQ_MSE_WARN       = 0.05   # warn if mean QQ MSE per instrument exceeds this
VAL_ACF_SSD_WARN      = 0.10   # warn if mean ACF SSD exceeds this
VAL_PCA_REL_ERR_WARN  = 0.20   # warn if any top-3 eigenvalue relative error exceeds this
VAL_KURTOSIS_MIN_FRAC = 0.80   # fraction of instruments that must pass kurtosis ratio check
