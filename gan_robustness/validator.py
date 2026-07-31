"""
validator.py — Statistical validation of WGAN-generated synthetic paths.

Three metrics are computed to assess whether the WGAN has learned the
correct joint distribution of core instrument returns:

1. QQ-plot statistic
   Mean squared difference between empirical quantiles of real vs. pooled
   synthetic returns, averaged over all core instruments.

2. Correlation matrix distance
   Frobenius norm of (C_real - C_synth) normalised by D², where C are
   Pearson correlation matrices.  Also reports maximum element difference
   to catch sign reversals in critical pairs (e.g. bonds vs equities).

3. ACF comparison
   Sum of squared differences of autocorrelation at lags 1-20 for both
   raw returns (should be near zero) and squared returns (captures
   volatility clustering, also expected near zero for futures).

All metrics are computed on the POOLED distribution (all 500 paths stacked),
which gives the average distributional properties of the generator.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


# ---------------------------------------------------------------------------
# Q-Q statistic
# ---------------------------------------------------------------------------

def qq_statistic(
    real_returns: np.ndarray,   # (T, D)
    synth_returns: np.ndarray,  # (T', D) — typically pooled paths
    n_quantiles: int = 99,
) -> dict:
    """
    Per-instrument Q-Q mean squared error between real and synthetic quantiles.

    Quantile levels: np.linspace(0.01, 0.99, n_quantiles).

    Returns:
        dict with:
          qq_mse_mean          : float — mean over D instruments
          qq_mse_per_instrument: list[float] — one per instrument
    """
    D = real_returns.shape[1]
    levels = np.linspace(0.01, 0.99, n_quantiles)

    per_inst = []
    for d in range(D):
        q_real  = np.quantile(real_returns[:, d],  levels)
        q_synth = np.quantile(synth_returns[:, d], levels)
        mse = float(np.mean((q_real - q_synth) ** 2))
        per_inst.append(mse)

    return {
        "qq_mse_mean":           float(np.mean(per_inst)),
        "qq_mse_per_instrument": per_inst,
        "n_quantiles":           n_quantiles,
    }


# ---------------------------------------------------------------------------
# Correlation matrix distance
# ---------------------------------------------------------------------------

def correlation_matrix_distance(
    real_returns: np.ndarray,   # (T, D)
    synth_returns: np.ndarray,  # (T', D)
) -> dict:
    """
    Frobenius norm of (C_real - C_synth), normalised by D².

    Also reports max absolute element difference and the element-wise
    difference matrix for inspection.

    Returns:
        dict with frobenius_norm, frobenius_norm_normalised, max_element_diff,
        and diff_matrix (D×D list of lists).
    """
    D = real_returns.shape[1]

    C_real  = np.corrcoef(real_returns.T)    # (D, D)
    C_synth = np.corrcoef(synth_returns.T)   # (D, D)

    diff = C_real - C_synth
    frob = float(np.linalg.norm(diff, "fro"))
    frob_norm = frob / (D * D)
    max_diff  = float(np.max(np.abs(diff)))

    return {
        "frobenius_norm":            frob,
        "frobenius_norm_normalised": frob_norm,
        "max_element_diff":          max_diff,
        "diff_matrix":               diff.tolist(),
    }


# ---------------------------------------------------------------------------
# ACF comparison
# ---------------------------------------------------------------------------

def _acf_series(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute sample autocorrelation of x at lags 1…max_lag."""
    x = x - x.mean()
    var = np.var(x)
    if var < 1e-15:
        return np.zeros(max_lag)
    acf = np.array([
        np.mean(x[lag:] * x[:-lag]) / var if lag > 0 else 1.0
        for lag in range(1, max_lag + 1)
    ])
    return acf


def acf_comparison(
    real_returns: np.ndarray,   # (T, D)
    synth_returns: np.ndarray,  # (T', D)
    max_lag: int = 20,
) -> dict:
    """
    Sum of squared differences of ACF (lags 1-20) for returns AND squared returns.

    Squared-return ACF tests whether volatility clustering is preserved
    (important for realistic regime simulation).

    Returns:
        dict with ssd_returns_mean, ssd_sq_returns_mean, per_instrument list.
    """
    D = real_returns.shape[1]
    per_inst = []

    for d in range(D):
        acf_r_real  = _acf_series(real_returns[:, d],         max_lag)
        acf_r_synth = _acf_series(synth_returns[:, d],        max_lag)
        acf_r2_real  = _acf_series(real_returns[:, d] ** 2,   max_lag)
        acf_r2_synth = _acf_series(synth_returns[:, d] ** 2,  max_lag)

        ssd_r  = float(np.sum((acf_r_real  - acf_r_synth)  ** 2))
        ssd_r2 = float(np.sum((acf_r2_real - acf_r2_synth) ** 2))
        per_inst.append({"ssd_returns": ssd_r, "ssd_squared_returns": ssd_r2})

    ssd_r_all  = [x["ssd_returns"]         for x in per_inst]
    ssd_r2_all = [x["ssd_squared_returns"] for x in per_inst]

    return {
        "ssd_returns_mean":          float(np.mean(ssd_r_all)),
        "ssd_sq_returns_mean":       float(np.mean(ssd_r2_all)),
        "max_lag":                   max_lag,
        "per_instrument":            per_inst,
    }


# ---------------------------------------------------------------------------
# PCA eigenvalue comparison
# ---------------------------------------------------------------------------

def pca_comparison(
    real_returns: np.ndarray,   # (T, D)
    synth_returns: np.ndarray,  # (T', D)
    n_components: int = 3,
) -> dict:
    """
    Compare the top-K eigenvalues of the correlation matrix between real and
    synthetic returns.  Eigenvalues capture the principal factor structure
    (market beta, rates factor, commodity factor, etc.).

    A large relative error in the top eigenvalues means the WGAN has distorted
    the factor structure — e.g., the equity-bond anti-correlation during crises.

    Returns:
        dict with eigenvalues_real, eigenvalues_synthetic, relative_error,
        and max_relative_error.
    """
    D = real_returns.shape[1]

    C_real  = np.corrcoef(real_returns.T)    # (D, D)
    C_synth = np.corrcoef(synth_returns.T)   # (D, D)

    # eigvalsh returns eigenvalues in ascending order; reverse for descending
    eigvals_real  = np.sort(np.linalg.eigvalsh(C_real))[::-1]
    eigvals_synth = np.sort(np.linalg.eigvalsh(C_synth))[::-1]

    top_real  = eigvals_real[:n_components]
    top_synth = eigvals_synth[:n_components]

    # Relative error; guard against near-zero real eigenvalues
    rel_error = np.abs(top_real - top_synth) / np.maximum(np.abs(top_real), 1e-9)

    # Variance explained (real)
    total_variance = max(float(eigvals_real.sum()), 1e-9)
    pct_explained_real  = float(top_real.sum()  / total_variance)
    pct_explained_synth = float(top_synth.sum() / float(max(eigvals_synth.sum(), 1e-9)))

    return {
        "n_components":            n_components,
        "eigenvalues_real":        top_real.tolist(),
        "eigenvalues_synthetic":   top_synth.tolist(),
        "relative_error":          rel_error.tolist(),
        "max_relative_error":      float(np.max(rel_error)),
        "pct_variance_real":       pct_explained_real,
        "pct_variance_synthetic":  pct_explained_synth,
    }


# ---------------------------------------------------------------------------
# Marginal statistics
# ---------------------------------------------------------------------------

def marginal_stats(
    real_returns: np.ndarray,    # (T, D)
    synth_returns: np.ndarray,   # (T', D)
    instrument_names: list[str],
) -> dict:
    """
    Per-instrument comparison of first four moments.

    Returns dict with 'real' and 'synthetic' sub-dicts, each mapping
    instrument name -> {mean, std, skewness, kurtosis}.
    """
    D = real_returns.shape[1]

    def _stats(X: np.ndarray) -> dict:
        result = {}
        for d in range(D):
            col = X[:, d]
            result[instrument_names[d]] = {
                "mean":     float(np.mean(col)),
                "std":      float(np.std(col, ddof=1)),
                "skewness": float(skew(col)),
                "kurtosis": float(kurtosis(col, fisher=True)),  # excess kurtosis
            }
        return result

    return {
        "real":      _stats(real_returns),
        "synthetic": _stats(synth_returns),
    }


# ---------------------------------------------------------------------------
# Full validation report
# ---------------------------------------------------------------------------

def run_full_validation(
    real_returns: np.ndarray,            # (T, D) training data in original scale
    all_synth_paths: list[np.ndarray],   # N_PATHS × (PATH_LENGTH, D) in original scale
    instrument_names: list[str],
    config: object,
    out_dir: Path,
    verbose: bool = True,
) -> dict:
    """
    Pool all synthetic paths, compute all three validation metrics, save JSON.

    The pooled synthetic array has shape (N_PATHS × PATH_LENGTH, D).

    Returns the full report dict (also saved to out_dir / 'gan_validation_report.json').
    """
    if verbose:
        print(f"[validator] Pooling {len(all_synth_paths)} paths for validation...")

    synth_pooled = np.vstack(all_synth_paths)  # (N*T', D)

    if verbose:
        print(f"[validator] Real: {real_returns.shape} | Pooled synth: {synth_pooled.shape}")

    qq     = qq_statistic(real_returns, synth_pooled)
    corr   = correlation_matrix_distance(real_returns, synth_pooled)
    acf    = acf_comparison(real_returns, synth_pooled)
    pca    = pca_comparison(real_returns, synth_pooled, n_components=3)
    mstats = marginal_stats(real_returns, synth_pooled, instrument_names)

    # Threshold warnings
    warnings_list = []
    if corr["frobenius_norm_normalised"] > config.VAL_FROBENIUS_WARN:
        msg = (f"Frobenius norm ({corr['frobenius_norm_normalised']:.3f}) exceeds "
               f"threshold {config.VAL_FROBENIUS_WARN}. Correlation structure may be degraded.")
        warnings_list.append(msg)
        if verbose:
            print(f"  [validator] WARNING: {msg}")
    if qq["qq_mse_mean"] > config.VAL_QQ_MSE_WARN:
        msg = (f"QQ MSE ({qq['qq_mse_mean']:.4f}) exceeds threshold {config.VAL_QQ_MSE_WARN}.")
        warnings_list.append(msg)
        if verbose:
            print(f"  [validator] WARNING: {msg}")
    if acf["ssd_returns_mean"] > config.VAL_ACF_SSD_WARN:
        msg = (f"ACF SSD ({acf['ssd_returns_mean']:.4f}) exceeds threshold {config.VAL_ACF_SSD_WARN}.")
        warnings_list.append(msg)
        if verbose:
            print(f"  [validator] WARNING: {msg}")
    pca_threshold = getattr(config, "VAL_PCA_REL_ERR_WARN", 0.20)
    if pca["max_relative_error"] > pca_threshold:
        msg = (f"PCA top-3 max relative error ({pca['max_relative_error']:.3f}) exceeds "
               f"threshold {pca_threshold}. Factor structure may not be preserved.")
        warnings_list.append(msg)
        if verbose:
            print(f"  [validator] WARNING: {msg}")

    # Kurtosis ratio check (synthetic / real should be in [0.5, 2.0])
    D = real_returns.shape[1]
    kurtosis_pass = 0
    kurtosis_details: list[dict] = []
    for d in range(D):
        inst = instrument_names[d]
        k_real  = mstats["real"][inst]["kurtosis"]
        k_synth = mstats["synthetic"][inst]["kurtosis"]
        # Guard: instruments with |k_real| < 0.5 are near-Gaussian; skip ratio
        if abs(k_real) < 0.5:
            passes = True
        else:
            ratio = k_synth / k_real if k_real != 0 else float("inf")
            passes = 0.5 <= ratio <= 2.0
        if passes:
            kurtosis_pass += 1
        kurtosis_details.append({
            "instrument": inst,
            "kurtosis_real":  k_real,
            "kurtosis_synth": k_synth,
            "passes":         passes,
        })
    kurtosis_pass_frac = kurtosis_pass / D
    kurtosis_min_frac  = getattr(config, "VAL_KURTOSIS_MIN_FRAC", 0.80)
    kurtosis_summary = {
        "pass_fraction":     kurtosis_pass_frac,
        "pass_count":        kurtosis_pass,
        "total_instruments": D,
        "per_instrument":    kurtosis_details,
    }
    if kurtosis_pass_frac < kurtosis_min_frac:
        msg = (f"Kurtosis ratio pass rate ({kurtosis_pass_frac:.1%}) below "
               f"threshold {kurtosis_min_frac:.0%}. "
               f"Synthetic tails may be too thin (mode collapse symptom).")
        warnings_list.append(msg)
        if verbose:
            print(f"  [validator] WARNING: {msg}")

    report = {
        "qq_statistic":              qq,
        "correlation_distance":      {k: v for k, v in corr.items() if k != "diff_matrix"},
        "acf_comparison":            acf,
        "pca_comparison":            pca,
        "kurtosis_check":            kurtosis_summary,
        "marginal_stats":            mstats,
        "validation_warnings":       warnings_list,
        "n_paths_pooled":            len(all_synth_paths),
        "path_length":               all_synth_paths[0].shape[0] if all_synth_paths else 0,
        "n_training_rows":           real_returns.shape[0],
        "core_instruments":          instrument_names,
    }

    report_path = out_dir / "gan_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(f"[validator] Report saved -> {report_path}")
        print(f"  QQ MSE:              {qq['qq_mse_mean']:.5f}")
        print(f"  Frobenius (norm):    {corr['frobenius_norm_normalised']:.4f}")
        print(f"  ACF SSD:             {acf['ssd_returns_mean']:.4f}")
        print(f"  PCA top-3 rel error: {pca['max_relative_error']:.4f}  "
              f"eigvals_real={[f'{v:.2f}' for v in pca['eigenvalues_real']]}  "
              f"eigvals_synth={[f'{v:.2f}' for v in pca['eigenvalues_synthetic']]}")
        print(f"  Kurtosis pass rate:  {kurtosis_summary['pass_fraction']:.1%}  "
              f"({kurtosis_summary['pass_count']}/{kurtosis_summary['total_instruments']} instruments)")

    return report
