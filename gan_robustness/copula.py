"""
copula.py — Gaussian copula for simulating the 50 non-core instruments,
conditional on the 12 WGAN-generated core instrument returns.

Approach:
  1. Fit a Gaussian copula on the full historical cross-section (62 instruments).
     - Normal-score transform each column.
     - Estimate correlation matrix of Gaussianized data.
     - Partition into core-core (Σ₁₁), core-noncore (Σ₁₂),
       noncore-core (Σ₂₁), noncore-noncore (Σ₂₂).
     - Precompute conditional mean operator A = Σ₂₁ @ inv(Σ₁₁ + ε·I).
     - Precompute conditional covariance and its Cholesky factor.

  2. Simulate non-core returns for a given core path:
     For each time step t:
       z_core    = normal_score_transform_single(core_return[t])   # (12,)
       mu_cond   = A @ z_core                                       # (50,)
       z_noncore = mu_cond + L_cond @ N(0, I_50)                   # (50,)
       p         = Φ(z_noncore)                                     # -> [0,1]
       r_noncore = quantile(p, empirical_cdf_noncore)               # price-diffs

Numerical robustness:
  - Σ₁₁ regularized with +1e-6·I before solve (prevents singularity from
    correlated core instruments like ZN/FGBL9, r≈0.85).
  - Σ_cond regularized with +1e-6·I before Cholesky.
  - Non-core instruments that have no history (all zeros) are handled: their
    copula contribution is just pure noise from the marginal.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm


# ---------------------------------------------------------------------------
# Normal-score helpers (standalone — copula uses its own transform)
# ---------------------------------------------------------------------------

def _normal_scores_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute normal scores for each column of X (T × D).
    rank_i / (T + 1) -> probit.  Returns same shape as X.
    """
    T, D = X.shape
    ns = np.empty_like(X, dtype=np.float64)
    for d in range(D):
        r = rankdata(X[:, d], method="average")
        ns[:, d] = norm.ppf(r / (T + 1))
    return ns


def _normal_score_single(
    x: np.ndarray,               # (D,)
    sorted_originals: np.ndarray,  # (T, D)
) -> np.ndarray:
    """
    Map a single observation x into normal-score space via binary search
    into sorted_originals.  Returns (D,) normal scores.
    """
    T, D = sorted_originals.shape
    ns = np.empty(D, dtype=np.float64)
    for d in range(D):
        rank = np.searchsorted(sorted_originals[:, d], x[d])
        rank_clamped = np.clip(rank, 1, T - 1)
        p = rank_clamped / (T + 1)
        ns[d] = norm.ppf(p)
    return ns


def _quantile_interp(
    p: np.ndarray,               # (D,) probabilities in [0, 1]
    sorted_originals: np.ndarray,  # (T, D) sorted per column
) -> np.ndarray:
    """
    Invert empirical CDF: map probabilities p -> original values via linear
    interpolation through sorted_originals.  Returns (D,) price-differences.
    """
    T, D = sorted_originals.shape
    grid = np.linspace(0.0, 1.0, T)
    result = np.empty(D, dtype=np.float64)
    for d in range(D):
        result[d] = np.interp(p[d], grid, sorted_originals[:, d])
    return result


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_gaussian_copula(
    all_returns_df: pd.DataFrame,   # (T_full, 62) full historical price-diffs
    core_cols: list[str],           # 12 core instrument column names
    non_core_cols: list[str],       # 50 non-core instrument column names
) -> dict:
    """
    Fit the Gaussian copula and return all parameters needed for simulation.

    The fit is performed on rows where at least one core instrument is non-zero
    (i.e. we have meaningful data).  Non-core instruments that are all-zero
    (instruments not yet launched in the data window) are handled gracefully.

    Returns dict with keys:
      A                       : (n_nc, n_c)   conditional mean operator
      L_cond                  : (n_nc, n_nc)  Cholesky of conditional covariance
      sorted_core             : (T, n_c)      sorted core original values
      sorted_noncore          : (T, n_nc)     sorted non-core original values
      core_cols               : list[str]
      non_core_cols           : list[str]
      n_core                  : int
      n_noncore               : int
    """
    # Align columns to requested order, fill any missing with 0
    all_cols = core_cols + non_core_cols
    for col in all_cols:
        if col not in all_returns_df.columns:
            warnings.warn(f"copula.fit: {col} not in all_returns_df, filling with 0", stacklevel=2)
            all_returns_df = all_returns_df.copy()
            all_returns_df[col] = 0.0

    X = all_returns_df[all_cols].to_numpy(dtype=np.float64)   # (T_full, 62)
    T_full = X.shape[0]
    n_c  = len(core_cols)
    n_nc = len(non_core_cols)

    # Normal-score transform all columns jointly
    Xns = _normal_scores_matrix(X)   # (T_full, 62)

    # Correlation matrix on Gaussianized data
    corr = np.corrcoef(Xns.T)        # (62, 62)

    # Partition: first n_c columns are core
    S11 = corr[:n_c, :n_c]          # (n_c,  n_c)
    S12 = corr[:n_c, n_c:]          # (n_c,  n_nc)
    S21 = corr[n_c:, :n_c]          # (n_nc, n_c)
    S22 = corr[n_c:, n_c:]          # (n_nc, n_nc)

    # Regularise S11 for numerical stability (ZN/FGBL9 have r≈0.85)
    reg = 1e-6
    S11_reg = S11 + reg * np.eye(n_c)

    # Conditional mean operator: A = S21 @ inv(S11_reg)
    A = S21 @ np.linalg.solve(S11_reg, np.eye(n_c))   # (n_nc, n_c)

    # Conditional covariance: Σ_cond = S22 - A @ S12
    S_cond = S22 - A @ S12                              # (n_nc, n_nc)
    S_cond = (S_cond + S_cond.T) / 2.0                 # symmetrise
    S_cond += 1e-6 * np.eye(n_nc)                       # regularise

    # Cholesky decomposition
    try:
        L_cond = np.linalg.cholesky(S_cond)             # (n_nc, n_nc) lower triangular
    except np.linalg.LinAlgError:
        warnings.warn(
            "Cholesky of conditional covariance failed; adding stronger regularisation (1e-4).",
            stacklevel=2,
        )
        S_cond += 1e-4 * np.eye(n_nc)
        L_cond = np.linalg.cholesky(S_cond)

    # Store sorted originals for empirical quantile inversion
    sorted_core    = np.sort(X[:, :n_c],  axis=0)   # (T_full, n_c)
    sorted_noncore = np.sort(X[:, n_c:],  axis=0)   # (T_full, n_nc)

    return {
        "A":               A,
        "L_cond":          L_cond,
        "sorted_core":     sorted_core,
        "sorted_noncore":  sorted_noncore,
        "core_cols":       core_cols,
        "non_core_cols":   non_core_cols,
        "n_core":          n_c,
        "n_noncore":       n_nc,
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_non_core_returns(
    core_synth_path: np.ndarray,    # (path_length, n_c) in original price-diff scale
    copula_params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate non-core returns conditional on core path via Gaussian copula.

    For each time step t:
      z_core[t]    = normal_score_single(core_synth_path[t], sorted_core)  # (n_c,)
      mu_cond      = A @ z_core[t]                                          # (n_nc,)
      z_noncore[t] = mu_cond + L_cond @ eps    where eps ~ N(0, I_{n_nc})  # (n_nc,)
      p            = Φ(z_noncore[t])                                        # (n_nc,)
      r_noncore[t] = quantile(p, sorted_noncore)                            # (n_nc,)

    Returns:
        (path_length, n_nc) in original price-difference scale.
    """
    A             = copula_params["A"]             # (n_nc, n_c)
    L_cond        = copula_params["L_cond"]        # (n_nc, n_nc)
    sorted_core   = copula_params["sorted_core"]   # (T, n_c)
    sorted_noncore = copula_params["sorted_noncore"]  # (T, n_nc)
    n_nc          = copula_params["n_noncore"]

    path_length = core_synth_path.shape[0]
    result = np.empty((path_length, n_nc), dtype=np.float64)

    for t in range(path_length):
        # Step 1: normal-score the core returns
        z_core = _normal_score_single(core_synth_path[t], sorted_core)   # (n_c,)

        # Step 2: conditional mean
        mu_cond = A @ z_core                                               # (n_nc,)

        # Step 3: conditional draw
        eps      = rng.standard_normal(n_nc)
        z_noncore = mu_cond + L_cond @ eps                                 # (n_nc,)

        # Step 4: map to probability
        p = norm.cdf(z_noncore)                                            # (n_nc,)

        # Step 5: empirical quantile inversion
        result[t] = _quantile_interp(p, sorted_noncore)

    return result


# ---------------------------------------------------------------------------
# Path assembly
# ---------------------------------------------------------------------------

def combine_paths(
    core_path: np.ndarray,       # (path_length, n_c)
    noncore_path: np.ndarray,    # (path_length, n_nc)
    all_columns: list[str],      # 62 column names in exact per_inst_pnl order
    core_cols: list[str],
    non_core_cols: list[str],
) -> pd.DataFrame:
    """
    Assemble the full 62-instrument synthetic path DataFrame.

    Column order matches per_inst_pnl.columns exactly — critical for the
    exposure-based simplified backtest.
    """
    n_core   = len(core_cols)
    n_nc     = len(non_core_cols)
    path_len = core_path.shape[0]

    # Build column lookup
    core_map    = {col: i for i, col in enumerate(core_cols)}
    noncore_map = {col: i for i, col in enumerate(non_core_cols)}

    data = np.empty((path_len, len(all_columns)), dtype=np.float64)
    for j, col in enumerate(all_columns):
        if col in core_map:
            data[:, j] = core_path[:, core_map[col]]
        elif col in noncore_map:
            data[:, j] = noncore_path[:, noncore_map[col]]
        else:
            # Column in per_inst_pnl but not in either list — fill with 0
            data[:, j] = 0.0

    return pd.DataFrame(data, columns=all_columns)
