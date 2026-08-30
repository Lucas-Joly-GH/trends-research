"""
run_pipeline.py — Top-level orchestrator for the WGAN robustness pipeline.

Usage:
  python run_pipeline.py [--retrain] [--n-paths N] [--epochs N] [--no-gpu]

Modes (controlled by config.CORE_INSTRUMENTS):
  None (default) — Full 62-instrument WGAN.  The WGAN generates all 62
    dimensions directly.  No copula.  Training uses all checkpoint dates
    (~9,341 rows) with NaN->0 fills for pre-launch instruments.
  list — Subset WGAN + Gaussian copula for the remaining instruments.

Stages:
  0. Install PyTorch if missing (CPU-only wheel, idempotent).
  1. Create output directories (clear stale paths when --retrain).
  2. Load data: checkpoint + instrument returns.
  3. Preprocessing: fit ReturnPreprocessor, scale training data.
  4. WGAN training (skipped if wgan_model.pt exists and --retrain not passed).
  5. Copula fitting (skipped when all instruments are core).
  6. Generate N_PATHS synthetic paths; save each as parquet.
  7. Simplified backtest: derive exposures + run N_PATHS synthetic backtests.
  8. Validation: QQ, correlation, ACF, PCA on pooled paths.
  9. Save gan_backtest_results.json + gan_robustness_summary.md.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Stage 0: Ensure PyTorch is installed (before any torch import)
# ---------------------------------------------------------------------------
import importlib.util
import subprocess
import sys


def _ensure_torch() -> None:
    if importlib.util.find_spec("torch") is None:
        print("[setup] PyTorch not found. Installing CPU-only wheel...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "torch",
            "--index-url", "https://download.pytorch.org/whl/cpu",
            "--quiet",
        ])
        print("[setup] PyTorch installed successfully.")


_ensure_torch()

# ---------------------------------------------------------------------------
# Standard imports (after torch is available)
# ---------------------------------------------------------------------------
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# cuDNN autotune disabled to conserve VRAM on 8GB laptop GPUs.
# Autotune picks the fastest conv kernel, often at the cost of a larger
# workspace allocation. On an RTX 4060 Laptop (8GB) with the attention-
# augmented critic + gradient penalty, the autotune workspace pushed total
# VRAM over the limit and caused system crashes. Re-enable if training
# on a GPU with >=12GB VRAM.
torch.backends.cudnn.benchmark = False

# Local package imports — support both `python -m` (relative) and direct execution
try:
    from . import config as cfg
    from .data_loader import (
        load_checkpoint,
        load_panama_returns,
        load_all_instrument_returns,
        compute_validity_mask,
        find_common_dates,
        build_return_matrix,
    )
    from .preprocessor import ReturnPreprocessor
    from .wgan_model import (
        ConditionalGenerator, Critic, count_parameters as count_params_dense,
    )
    from .cwgan_model import (
        ConvGenerator, ConvCritic, count_parameters as count_params_conv,
    )
    from .trainer import train_wgan
    from .generator_rollout import generate_all_paths
    from .copula import fit_gaussian_copula, simulate_non_core_returns, combine_paths
    from .backtest import compute_exposures, run_all_backtests
    from .validator import run_full_validation
except ImportError:
    _pkg_dir = str(Path(__file__).resolve().parent)
    if _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)
    import config as cfg  # type: ignore[no-redef]
    from data_loader import (  # type: ignore[no-redef]
        load_checkpoint, load_panama_returns, load_all_instrument_returns,
        compute_validity_mask, find_common_dates, build_return_matrix,
    )
    from preprocessor import ReturnPreprocessor  # type: ignore[no-redef]
    from wgan_model import ConditionalGenerator, Critic, count_parameters as count_params_dense  # type: ignore[no-redef]
    from cwgan_model import ConvGenerator, ConvCritic, count_parameters as count_params_conv  # type: ignore[no-redef]
    from trainer import train_wgan  # type: ignore[no-redef]
    from generator_rollout import generate_all_paths  # type: ignore[no-redef]
    from copula import fit_gaussian_copula, simulate_non_core_returns, combine_paths  # type: ignore[no-redef]
    from backtest import compute_exposures, run_all_backtests  # type: ignore[no-redef]
    from validator import run_full_validation  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _distribution_stats(values: list[float]) -> dict:
    arr = np.array(values)
    return {
        "mean":  float(np.mean(arr)),
        "std":   float(np.std(arr, ddof=1)),
        "min":   float(np.min(arr)),
        "p5":    float(np.percentile(arr, 5)),
        "p25":   float(np.percentile(arr, 25)),
        "p50":   float(np.percentile(arr, 50)),
        "p75":   float(np.percentile(arr, 75)),
        "p95":   float(np.percentile(arr, 95)),
        "max":   float(np.max(arr)),
    }


def _write_summary(
    checkpoint: dict,
    backtest_df: pd.DataFrame,
    val_report: dict,
    config: object,
    out_dir: Path,
    core_instruments: list[str],
) -> None:
    real_sr   = checkpoint["sr"]
    real_cagr = checkpoint["cagr"]
    real_mdd  = checkpoint["max_dd"]
    real_cal  = checkpoint["calmar"]

    synth_sr       = backtest_df["sr"].tolist()
    pct_above_real = float((np.array(synth_sr) > real_sr).mean())
    pct_above_zero = float((np.array(synth_sr) > 0).mean())
    pct_above_05   = float((np.array(synth_sr) > 0.5).mean())

    n_paths = len(backtest_df)
    pca     = val_report.get("pca_comparison", {})

    lines = [
        "# S183 WGAN Robustness Analysis",
        "",
        "## Real Backtest Baseline (S183, 1990-2026)",
        f"- **Sharpe Ratio:** {real_sr:.4f}",
        f"- **CAGR:** {real_cagr:.2%}",
        f"- **Max Drawdown:** {real_mdd:.2%}",
        f"- **Calmar:** {real_cal:.4f}",
        "",
        f"## Synthetic Path Distribution ({n_paths} paths x {config.PATH_LENGTH} days)",
        "",
        "> **Note:** Synthetic P&L is gross (pre-cost). Real SR=0.93 includes commissions",
        "> (~$2.5B cumulative). Synthetic SRs are therefore upward-biased vs. real.",
        "",
        "### Sharpe Ratio Distribution",
        f"- Mean: {backtest_df['sr'].mean():.4f}  |  Median: {backtest_df['sr'].median():.4f}",
        f"- Std:  {backtest_df['sr'].std():.4f}",
        f"- 5th pct: {backtest_df['sr'].quantile(0.05):.4f}  |  "
        f"95th pct: {backtest_df['sr'].quantile(0.95):.4f}",
        f"- Paths with SR > real baseline ({real_sr:.2f}): **{pct_above_real:.1%}**",
        f"- Paths with SR > 0.5: **{pct_above_05:.1%}**",
        f"- Paths with SR > 0:   **{pct_above_zero:.1%}**",
        "",
        "### CAGR Distribution",
        f"- Mean: {backtest_df['cagr'].mean():.2%}  |  Median: {backtest_df['cagr'].median():.2%}",
        f"- 5th-95th pct range: [{backtest_df['cagr'].quantile(0.05):.2%}, "
        f"{backtest_df['cagr'].quantile(0.95):.2%}]",
        "",
        "### Max Drawdown Distribution",
        f"- Mean: {backtest_df['max_dd'].mean():.2%}  |  "
        f"Median: {backtest_df['max_dd'].median():.2%}",
        f"- 5th-95th pct range: [{backtest_df['max_dd'].quantile(0.05):.2%}, "
        f"{backtest_df['max_dd'].quantile(0.95):.2%}]",
        "",
        "### Calmar Distribution",
        f"- Mean: {backtest_df['calmar'].mean():.4f}  |  "
        f"Median: {backtest_df['calmar'].median():.4f}",
        "",
        "## WGAN Validation",
        f"- **QQ MSE (mean over instruments):** "
        f"{val_report['qq_statistic']['qq_mse_mean']:.5f}  "
        f"(threshold: {config.VAL_QQ_MSE_WARN})",
        f"- **Correlation Frobenius norm (normalised):** "
        f"{val_report['correlation_distance']['frobenius_norm_normalised']:.4f}  "
        f"(threshold: {config.VAL_FROBENIUS_WARN})",
        f"- **ACF SSD (mean):** "
        f"{val_report['acf_comparison']['ssd_returns_mean']:.4f}  "
        f"(threshold: {config.VAL_ACF_SSD_WARN})",
    ]

    if pca:
        pca_threshold = getattr(config, "VAL_PCA_REL_ERR_WARN", 0.20)
        lines += [
            f"- **PCA top-3 eigenvalue max relative error:** "
            f"{pca['max_relative_error']:.4f}  (threshold: {pca_threshold})",
            f"  - Real eigvals: {[f'{v:.2f}' for v in pca['eigenvalues_real']]}  "
            f"({pca['pct_variance_real']:.1%} variance explained)",
            f"  - Synth eigvals: {[f'{v:.2f}' for v in pca['eigenvalues_synthetic']]}  "
            f"({pca['pct_variance_synthetic']:.1%} variance explained)",
        ]

    lines.append("")

    if val_report["validation_warnings"]:
        lines.append("### Validation Warnings")
        for w in val_report["validation_warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        f"The distribution of SR across {n_paths} synthetic paths quantifies how much of",
        "S183's historical SR is structural edge vs. path-dependent luck.",
        "",
        f"- Real SR ({real_sr:.2f}) exceeds {pct_above_real:.1%} of synthetic paths.",
        f"  p-value (one-sided, SR above real): {1 - pct_above_real:.3f}",
        "",
        "## Configuration",
        f"- Architecture: {getattr(config, 'MODEL_TYPE', 'dense').upper()} | "
        f"{'full 62-instrument, no copula' if len(core_instruments) == 62 else f'{len(core_instruments)}-instrument + copula'}",
        f"- Instruments: {len(core_instruments)}",
        f"- WGAN epochs: {config.N_EPOCHS}  |  Batch: {config.BATCH_SIZE}",
        f"- Noise dim: {config.NOISE_DIM}  |  Window: {config.WINDOW_SIZE}",
        f"- Paths: {config.N_PATHS}  |  Path length: {config.PATH_LENGTH} days",
    ]

    summary_path = out_dir / "gan_robustness_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[output] Summary saved -> {summary_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(
    retrain: bool = False,
    n_paths: int | None = None,
    n_epochs: int | None = None,
    no_gpu: bool = False,
    resume_from: str | None = None,
) -> None:
    t0 = time.time()

    if n_paths is not None:
        cfg.N_PATHS = n_paths
    if n_epochs is not None:
        cfg.N_EPOCHS = n_epochs

    # -----------------------------------------------------------------------
    # Stage 1: Output directories
    # -----------------------------------------------------------------------
    print("[stage 1] Creating output directories...")
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

    if retrain and cfg.SYNTH_DIR.exists():
        old_parquets = list(cfg.SYNTH_DIR.glob("path_*.parquet"))
        if old_parquets:
            print(f"  Clearing {len(old_parquets)} stale path files from previous run...")
            for f in old_parquets:
                f.unlink()

    cfg.SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    if no_gpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"[setup] Using device: {device}")

    # -----------------------------------------------------------------------
    # Stage 2: Data loading
    # -----------------------------------------------------------------------
    print("[stage 2] Loading data...")

    checkpoint       = load_checkpoint(cfg.CHECKPOINT_PKL)
    checkpoint_dates = checkpoint["dates"]
    per_inst_pnl     = checkpoint["per_inst_pnl"]   # (9341, 62)
    all_inst_cols    = list(per_inst_pnl.columns)   # canonical 62-instrument order

    print(f"  Checkpoint: {len(checkpoint_dates)} dates | {len(all_inst_cols)} instruments")
    print(f"  SR={checkpoint['sr']:.4f} | CAGR={checkpoint['cagr']:.2%} | "
          f"MaxDD={checkpoint['max_dd']:.2%} | Calmar={checkpoint['calmar']:.4f}")

    # Resolve CORE_INSTRUMENTS: None -> use all 62
    if cfg.CORE_INSTRUMENTS is None:
        core_instruments = all_inst_cols
        full_wgan_mode   = True
    else:
        core_instruments = cfg.CORE_INSTRUMENTS
        full_wgan_mode   = (set(core_instruments) == set(all_inst_cols))

    print(f"  Mode: {'full 62-instrument WGAN (no copula)' if full_wgan_mode else f'{len(core_instruments)}-instrument WGAN + copula'}")

    # Always load all-62 returns aligned to checkpoint dates (NaN->0 fill).
    # This serves as:  (a) training data in full_wgan_mode,
    #                  (b) exposure data for the backtest in all modes,
    #                  (c) copula fitting data in subset mode.
    print("  Loading all-instrument returns (62 instruments, aligned to checkpoint dates)...")
    all_inst_returns_df = load_all_instrument_returns(checkpoint, cfg.PANAMA_DIR)
    print(f"  All-instrument returns: {all_inst_returns_df.shape}")

    if full_wgan_mode:
        # Training data = full checkpoint range, 0 fills for pre-launch instruments.
        # Column order follows all_inst_cols (= core_instruments).
        training_dates = checkpoint_dates
        X_raw = all_inst_returns_df.to_numpy(dtype=np.float64)   # (9341, 62)
        # Validity mask: identifies real vs. zero-filled data per instrument
        validity_mask = compute_validity_mask(all_inst_returns_df)
        print(f"  Training rows (full checkpoint, NaN->0 fill): {X_raw.shape[0]}")
    else:
        # Subset mode: restrict to dates where all chosen core instruments are valid.
        core_returns_df = load_panama_returns(core_instruments, cfg.PANAMA_DIR)
        common_dates    = find_common_dates(core_returns_df, checkpoint_dates)
        X_raw           = build_return_matrix(core_returns_df, common_dates)
        training_dates  = common_dates
        validity_mask   = np.ones(X_raw.shape, dtype=bool)  # all valid in subset mode
        print(f"  Core instruments: {len(core_instruments)} | "
              f"Training rows: {len(common_dates)} "
              f"({common_dates[0].date()} -> {common_dates[-1].date()})")

    # -----------------------------------------------------------------------
    # Stage 3: Preprocessing
    # -----------------------------------------------------------------------
    print("[stage 3] Fitting preprocessor...")
    preprocessor_path = cfg.OUT_DIR / "preprocessor.pkl"

    preprocessor = ReturnPreprocessor()
    preprocessor.fit(X_raw, validity_mask=validity_mask)
    X_scaled = preprocessor.transform(X_raw, validity_mask=validity_mask)
    preprocessor.save(preprocessor_path)

    print(f"  Preprocessor fitted | X_scaled: shape={X_scaled.shape}  "
          f"min={X_scaled.min():.3f}  max={X_scaled.max():.3f}")

    # -----------------------------------------------------------------------
    # Stage 4: WGAN training
    # -----------------------------------------------------------------------
    model_type = getattr(cfg, "MODEL_TYPE", "dense").lower()
    model_path = cfg.OUT_DIR / "wgan_model.pt"
    D = X_raw.shape[1]  # 62 in full_wgan_mode, len(core_instruments) otherwise

    if model_type == "cwgan":
        gen_channels = getattr(cfg, "CWGAN_GEN_CHANNELS",
                               getattr(cfg, "CWGAN_CHANNELS", [256, 256, 128, 64]))
        critic_channels = getattr(cfg, "CWGAN_CRITIC_CHANNELS", gen_channels)
        kernel   = getattr(cfg, "CWGAN_KERNEL_SIZE", 3)
        sn       = getattr(cfg, "CWGAN_SPECTRAL_NORM", True)
        generator = ConvGenerator(
            noise_dim=cfg.NOISE_DIM,
            window_size=cfg.WINDOW_SIZE,
            n_instruments=D,
            channels=gen_channels,
            kernel_size=kernel,
            leaky_slope=cfg.LEAKY_SLOPE,
        ).to(device)
        critic = ConvCritic(
            n_instruments=D,
            window_size=cfg.WINDOW_SIZE,
            channels=critic_channels,
            kernel_size=kernel,
            leaky_slope=cfg.LEAKY_SLOPE,
            use_spectral_norm=sn,
        ).to(device)
        n_params_g = count_params_conv(generator)
        n_params_c = count_params_conv(critic)
    else:
        sn = getattr(cfg, "SPECTRAL_NORM", False)
        generator = ConditionalGenerator(
            noise_dim=cfg.NOISE_DIM,
            window_size=cfg.WINDOW_SIZE,
            n_instruments=D,
            hidden_dims=cfg.GENERATOR_LAYERS,
            leaky_slope=cfg.LEAKY_SLOPE,
            use_spectral_norm=False,   # generator doesn't use SN
        ).to(device)
        critic = Critic(
            n_instruments=D,
            window_size=cfg.WINDOW_SIZE,
            hidden_dims=cfg.CRITIC_LAYERS,
            leaky_slope=cfg.LEAKY_SLOPE,
            use_spectral_norm=sn,
        ).to(device)
        n_params_g = count_params_dense(generator)
        n_params_c = count_params_dense(critic)

    print(f"[stage 4] {model_type.upper()} | D={D} | "
          f"Generator: {n_params_g:,} params | Critic: {n_params_c:,} params")

    # Auto-resume: if no finished model and no --retrain, find latest checkpoint
    if not resume_from and not retrain and not model_path.exists():
        ckpt_dir = cfg.OUT_DIR / "checkpoints"
        ckpt_files = sorted(ckpt_dir.glob("wgan_checkpoint_ep*.pt")) if ckpt_dir.exists() else []
        if ckpt_files:
            resume_from = str(ckpt_files[-1])
            print(f"  Auto-resuming from latest checkpoint: {ckpt_files[-1].name}")

    if model_path.exists() and not retrain and not resume_from:
        print(f"  Loading existing model from {model_path} (use --retrain to re-train)")
        saved = torch.load(model_path, map_location=device, weights_only=False)
        generator.load_state_dict(saved["generator_state"])
        critic.load_state_dict(saved["critic_state"])
        generator.eval()
        critic.eval()
    else:
        if resume_from:
            print(f"  --resume from {resume_from}")
        elif retrain and model_path.exists():
            print("  --retrain flag set, re-training from scratch...")
        print(f"  Training for {cfg.N_EPOCHS} epochs | {X_scaled.shape[0]} sequences "
              f"| ~{max(1, X_scaled.shape[0] // cfg.BATCH_SIZE)} batches/epoch...")
        t_train = time.time()
        history = train_wgan(
            generator=generator,
            critic=critic,
            X_scaled=X_scaled,
            config=cfg,
            device=device,
            out_dir=cfg.OUT_DIR,
            verbose=True,
            resume_from=resume_from,
        )
        elapsed = time.time() - t_train
        print(f"  Training complete in {elapsed:.1f}s "
              f"| Final W-est: {history['wasserstein_estimate'][-1]:+.4f}")

        torch.save(
            {
                "generator_state": generator.state_dict(),
                "critic_state":    critic.state_dict(),
                "config": {
                    "MODEL_TYPE":        model_type,
                    "NOISE_DIM":         cfg.NOISE_DIM,
                    "WINDOW_SIZE":       cfg.WINDOW_SIZE,
                    "CORE_INSTRUMENTS":  core_instruments,
                    "GENERATOR_LAYERS":  getattr(cfg, "GENERATOR_LAYERS", None),
                    "CRITIC_LAYERS":     getattr(cfg, "CRITIC_LAYERS", None),
                    "CWGAN_GEN_CHANNELS":    getattr(cfg, "CWGAN_GEN_CHANNELS", None),
                    "CWGAN_CRITIC_CHANNELS": getattr(cfg, "CWGAN_CRITIC_CHANNELS", None),
                    "MOMENT_PENALTY_LAMBDA": getattr(cfg, "MOMENT_PENALTY_LAMBDA", 0.0),
                    "N_EPOCHS":          cfg.N_EPOCHS,
                    "full_wgan_mode":    full_wgan_mode,
                },
                "history": history,
            },
            model_path,
        )
        print(f"  Model saved -> {model_path}")

    # -----------------------------------------------------------------------
    # Stage 5: Copula fitting (skipped in full_wgan_mode)
    # -----------------------------------------------------------------------
    non_core_cols  = [c for c in all_inst_cols if c not in core_instruments]
    copula_params  = None

    if full_wgan_mode:
        print("[stage 5] Copula fitting SKIPPED (full 62-instrument WGAN mode).")
    else:
        print(f"[stage 5] Fitting Gaussian copula | "
              f"core={len(core_instruments)} | non-core={len(non_core_cols)}...")
        copula_params = fit_gaussian_copula(
            all_returns_df=all_inst_returns_df,
            core_cols=core_instruments,
            non_core_cols=non_core_cols,
        )
        print(f"  Copula fitted | core={copula_params['n_core']} | "
              f"non-core={copula_params['n_noncore']}")

    # -----------------------------------------------------------------------
    # Stage 6: Path generation
    # -----------------------------------------------------------------------
    print(f"[stage 6] Generating {cfg.N_PATHS} synthetic paths ({cfg.PATH_LENGTH} steps each)...")
    t_gen = time.time()

    # WGAN rollout for core instruments (or all 62 in full mode)
    core_paths = generate_all_paths(
        generator=generator,
        preprocessor=preprocessor,
        real_returns=X_raw,
        config=cfg,
        device=device,
        verbose=True,
    )

    # Assemble full 62-dim paths
    all_full_paths: list[pd.DataFrame] = []

    for path_idx, core_path in enumerate(core_paths):
        if full_wgan_mode:
            # WGAN already produced all 62 dimensions
            full_df = pd.DataFrame(core_path, columns=all_inst_cols)
        else:
            # Expand to 62 instruments via copula
            rng = np.random.default_rng(cfg.GEN_SEED_BASE + path_idx + 10_000)
            noncore_path = simulate_non_core_returns(
                core_synth_path=core_path,
                copula_params=copula_params,
                rng=rng,
            )
            full_df = combine_paths(
                core_path=core_path,
                noncore_path=noncore_path,
                all_columns=all_inst_cols,
                core_cols=core_instruments,
                non_core_cols=non_core_cols,
            )

        all_full_paths.append(full_df)

        save_path = cfg.SYNTH_DIR / f"path_{path_idx:03d}.parquet"
        full_df.to_parquet(save_path, index=False)

        if (path_idx + 1) % 100 == 0:
            print(f"  [gen] Saved {path_idx + 1}/{cfg.N_PATHS} paths")

    print(f"  Generation complete in {time.time() - t_gen:.1f}s")

    # -----------------------------------------------------------------------
    # Stage 7: Simplified backtest
    # -----------------------------------------------------------------------
    print("[stage 7] Computing exposures and running simplified backtests...")
    t_bt = time.time()

    exposure_df = compute_exposures(
        per_inst_pnl=per_inst_pnl,
        hist_returns=all_inst_returns_df,
        min_ret_threshold=cfg.EXPOSURE_MIN_RET,
        clip_value=cfg.EXPOSURE_CLIP,
    )
    print(f"  Exposures: {exposure_df.shape} | "
          f"non-zero: {(exposure_df != 0).values.mean():.1%}")

    backtest_df = run_all_backtests(
        exposure_df=exposure_df,
        all_paths=all_full_paths,
        config=cfg,
        verbose=True,
    )
    print(f"  Backtests complete in {time.time() - t_bt:.1f}s")
    print(f"  SR: mean={backtest_df['sr'].mean():.3f}  "
          f"p50={backtest_df['sr'].median():.3f}  "
          f"p5-p95=[{backtest_df['sr'].quantile(0.05):.3f}, {backtest_df['sr'].quantile(0.95):.3f}]")

    # -----------------------------------------------------------------------
    # Stage 8: Validation
    # -----------------------------------------------------------------------
    print("[stage 8] Running statistical validation...")

    val_report = run_full_validation(
        real_returns=X_raw,
        all_synth_paths=core_paths,   # (PATH_LENGTH, D) — same D as WGAN output
        instrument_names=core_instruments,
        config=cfg,
        out_dir=cfg.OUT_DIR,
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Stage 9: Save results
    # -----------------------------------------------------------------------
    print("[stage 9] Saving results...")

    pca_report = val_report.get("pca_comparison", {})
    results = {
        "real_baseline": {
            "sr":     float(checkpoint["sr"]),
            "cagr":   float(checkpoint["cagr"]),
            "max_dd": float(checkpoint["max_dd"]),
            "calmar": float(checkpoint["calmar"]),
        },
        "note_gross_pnl": (
            "Synthetic P&L is GROSS (pre-cost). Real SR=0.93 includes commissions "
            "and cash yield (~$2.5B cumulative over 36 years). Synthetic SRs "
            "are upward-biased relative to a net-of-cost comparison."
        ),
        "mode":           f"{model_type}_{'62' if full_wgan_mode else str(len(core_instruments))}{'_no_copula' if full_wgan_mode else '_copula'}",
        "n_core_instruments": D,
        "n_paths":        cfg.N_PATHS,
        "path_length":    cfg.PATH_LENGTH,
        "synthetic": {
            "sr":      _distribution_stats(backtest_df["sr"].tolist()),
            "cagr":    _distribution_stats(backtest_df["cagr"].tolist()),
            "max_dd":  _distribution_stats(backtest_df["max_dd"].tolist()),
            "calmar":  _distribution_stats(backtest_df["calmar"].tolist()),
            "ann_vol": _distribution_stats(backtest_df["ann_vol"].tolist()),
        },
        "robustness_flags": {
            "pct_paths_sr_positive":   float((backtest_df["sr"] > 0).mean()),
            "pct_paths_sr_above_0_5":  float((backtest_df["sr"] > 0.5).mean()),
            "pct_paths_sr_above_real": float((backtest_df["sr"] > checkpoint["sr"]).mean()),
            "p_value_one_sided":       float(1.0 - (backtest_df["sr"] > checkpoint["sr"]).mean()),
        },
        "pca_validation": pca_report,
    }

    results_path = cfg.OUT_DIR / "gan_backtest_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved -> {results_path}")

    _write_summary(checkpoint, backtest_df, val_report, cfg, cfg.OUT_DIR, core_instruments)

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    elapsed_total = time.time() - t0
    print(f"\n[done] Total elapsed: {elapsed_total:.1f}s ({elapsed_total / 60:.1f} min)")
    print(f"  Outputs in: {cfg.OUT_DIR}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WGAN Market Generator for S183 Robustness Testing"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Re-train the WGAN even if wgan_model.pt already exists.",
    )
    parser.add_argument(
        "--n-paths",
        type=int,
        default=None,
        help=f"Number of synthetic paths to generate (default: {cfg.N_PATHS}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of training epochs (default: {cfg.N_EPOCHS}).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU even if a CUDA GPU is available.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a wgan_checkpoint_epXXXXX.pt file to resume training from.",
    )
    args = parser.parse_args()
    main(
        retrain=args.retrain,
        n_paths=args.n_paths,
        n_epochs=args.epochs,
        no_gpu=args.no_gpu,
        resume_from=args.resume,
    )
