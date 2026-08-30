"""
preprocessor.py — Stateful robust preprocessor for WGAN training.

Pipeline per instrument column:
  1. Robust scaling: (x - median) / (IQR + eps), clipped to [-CLIP, +CLIP],
     then divided by CLIP to produce values in [-1, 1].

     Unlike the legacy normal-score transform (rank -> norm.ppf -> MinMax),
     this preserves the actual shape of the return distribution (fat tails,
     skew) in the scaled space, giving the generator the opportunity to learn
     non-Gaussian features.

Inverse pipeline:
  1. Multiply by CLIP, then x * (IQR + eps) + median.

v3 changes (from v2 normal-score):
  - Replaced normal-score + MinMax with robust median/IQR scaling.
  - Tail information is preserved: the generator sees actual fat tails
    and skew in [-1, 1] space instead of Gaussianised proxies.
  - Fixes: QQ MSE, kurtosis pass rate, PCA eigenvalue error.
  - Legacy ReturnPreprocessor renamed to _LegacyNormalScorePreprocessor
    for backward-compat loading of old .pkl files.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, norm


# ───────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────

_CLIP_SIGMA = 10      # clip at ±10 IQR-units (was 5). Preserves 10x more tail
                      # mass so the generator can learn extreme events (6M, 6S,
                      # BTC daily returns reach ±10-15 IQR). Trade-off: typical
                      # values compressed toward zero in scaled [-1, 1] space,
                      # but Tanh saturation is softer so the generator has more
                      # effective range for tail modelling.
_EPS        = 1e-8    # prevent division by zero for constant instruments


# ───────────────────────────────────────────────────────────────────────
# Active preprocessor (v3 robust scaler)
# ───────────────────────────────────────────────────────────────────────

class ReturnPreprocessor:
    """
    Robust median/IQR preprocessor that maps raw returns to [-1, 1].

    Attributes set after fit():
      median         : (D,) float64  — per-instrument median
      iqr            : (D,) float64  — per-instrument IQR (p75 - p25)
      n_train        : int           — total number of training rows
      n_instruments  : int           — D
    """

    def __init__(self) -> None:
        self.median: np.ndarray | None = None
        self.iqr: np.ndarray | None = None
        self.n_train: int = 0
        self.n_instruments: int = 0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        validity_mask: np.ndarray | None = None,
    ) -> "ReturnPreprocessor":
        """
        Fit on training data X of shape (T, D).

        Args:
            X              : (T, D) raw returns (price differences)
            validity_mask  : (T, D) bool — True where instrument has real data.
                             If None, all rows are treated as valid.
        """
        T, D = X.shape
        self.n_train = T
        self.n_instruments = D

        if validity_mask is None:
            validity_mask = np.ones((T, D), dtype=bool)

        medians = np.zeros(D, dtype=np.float64)
        iqrs    = np.zeros(D, dtype=np.float64)

        for d in range(D):
            valid_vals = X[validity_mask[:, d], d]
            n = len(valid_vals)

            if n < 2:
                warnings.warn(
                    f"Instrument {d}: only {n} valid observations. "
                    "Outputs for this instrument will be unreliable.",
                    stacklevel=2,
                )
                medians[d] = 0.0
                iqrs[d] = 1.0  # arbitrary non-zero
                continue

            medians[d] = np.median(valid_vals)
            q25, q75 = np.percentile(valid_vals, [25, 75])
            iqr = q75 - q25

            if iqr < _EPS:
                # Near-constant instrument — fall back to std
                s = np.std(valid_vals)
                iqr = max(s, _EPS)
                warnings.warn(
                    f"Instrument {d}: IQR ~ 0 (constant column). "
                    f"Falling back to std={s:.2e}.",
                    stacklevel=2,
                )

            iqrs[d] = iqr

        self.median = medians
        self.iqr    = iqrs
        return self

    # ------------------------------------------------------------------
    # Transform (training data)
    # ------------------------------------------------------------------

    def transform(
        self,
        X: np.ndarray,
        validity_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Transform training data X of shape (T, D) -> scaled array in [-1, 1].

        Masked rows (validity_mask == False) are filled with 0.0 (midpoint).
        """
        self._check_fitted()
        T, D = X.shape

        if validity_mask is None:
            validity_mask = np.ones((T, D), dtype=bool)

        # Vectorised robust scaling
        centered = X - self.median                     # (T, D)
        scaled = centered / (self.iqr + _EPS)          # (T, D) in IQR units
        clipped = np.clip(scaled, -_CLIP_SIGMA, _CLIP_SIGMA)
        result = clipped / _CLIP_SIGMA                 # -> [-1, 1]

        # Zero-out masked entries
        result[~validity_mask] = 0.0

        return result

    # ------------------------------------------------------------------
    # transform_single (inference / rollout)
    # ------------------------------------------------------------------

    def transform_single(self, x: np.ndarray) -> np.ndarray:
        """
        Map a single observation x of shape (D,) or (1, D) to scaled space.
        Returns array of shape (D,) in [-1, 1].
        """
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self.n_instruments:
            raise ValueError(
                f"Expected {self.n_instruments} values, got {x.shape[0]}"
            )

        centered = x - self.median
        scaled = centered / (self.iqr + _EPS)
        clipped = np.clip(scaled, -_CLIP_SIGMA, _CLIP_SIGMA)
        return clipped / _CLIP_SIGMA

    # ------------------------------------------------------------------
    # Inverse transform
    # ------------------------------------------------------------------

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse of transform / transform_single.

        X_scaled: shape (T, D) or (D,).
        Returns original-scale values (price differences).
        """
        self._check_fitted()
        scalar = X_scaled.ndim == 1
        X_scaled = np.atleast_2d(np.asarray(X_scaled, dtype=np.float64))

        result = X_scaled * _CLIP_SIGMA * (self.iqr + _EPS) + self.median

        return result[0] if scalar else result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "ReturnPreprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        # Detect old normal-score format (has sorted_originals attribute)
        if hasattr(obj, "sorted_originals"):
            raise RuntimeError(
                "Loaded preprocessor uses old normal-score format. "
                "Re-run the pipeline with --retrain to regenerate."
            )
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.median is None:
            raise RuntimeError(
                "ReturnPreprocessor has not been fitted. Call fit() first."
            )
