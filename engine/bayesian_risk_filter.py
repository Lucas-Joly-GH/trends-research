"""
bayesian_risk_filter.py -- Bayesian Latent Factor Risk Filter (van Oord 2016)
==============================================================================
Implements a time-varying Bayesian latent factor model adapted from
Jacobus Arie (Arco) van Oord's 2016 thesis "Essays on Momentum Strategies
in Finance". Translates his cross-sectional equity methodology into a
time-series CTA framework.

The filter intercepts raw price data, extracts latent market factors via
Bayesian inference (NumPyro + NUTS), and returns:
  1. Residual returns / prices (factor-hedged)
  2. A regime multiplier based on the number of active latent factors

Architecture:
  - Monthly MCMC refit (every 21 trading days) for computational efficiency
  - Daily OLS projection of factor returns between refits
  - ARD (Automatic Relevance Determination) priors to determine effective K
  - No look-ahead bias: strictly causal rolling window

Usage:
    from bayesian_risk_filter import BayesianRiskFilter

    brf = BayesianRiskFilter()
    result = brf.filter_returns(inst_data_cache, mapping)
    # result["residual_prices"]     -> {instrument: pd.Series}
    # result["residual_returns"]    -> {instrument: pd.Series}
    # result["regime_multiplier"]   -> pd.Series (leverage scale)
    # result["n_factors"]           -> pd.Series (effective factor count)
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Lazy imports for NumPyro (heavy JAX dependency)
_NUMPYRO_AVAILABLE = None


def _check_numpyro():
    global _NUMPYRO_AVAILABLE
    if _NUMPYRO_AVAILABLE is None:
        try:
            import jax          # noqa: F401
            import numpyro      # noqa: F401
            _NUMPYRO_AVAILABLE = True
        except ImportError:
            _NUMPYRO_AVAILABLE = False
    return _NUMPYRO_AVAILABLE


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Volatility-Scaling Preprocessor
# ══════════════════════════════════════════════════════════════════════════════

def vol_scale_returns(price_changes, ewm_span=32, long_span=2560, blend=0.7):
    """Normalize price changes by blended EWM/rolling volatility.

    Matches the blended_volatility approach from shared_config but returns
    the vol-scaled returns directly.

    Parameters
    ----------
    price_changes : pd.Series
        Raw daily price changes (close-to-close).
    ewm_span : int
        Short-run EWM volatility span.
    long_span : int
        Long-run rolling volatility window.
    blend : float
        Weight on short-run EWM component (1-blend on rolling).

    Returns
    -------
    vol_scaled : pd.Series
        Returns divided by blended volatility estimate.
    vol : pd.Series
        The blended volatility series itself.
    """
    short_vol = price_changes.ewm(span=ewm_span, min_periods=max(10, ewm_span // 2)).std()
    long_vol = price_changes.rolling(long_span, min_periods=min(256, long_span)).std()
    vol = blend * short_vol + (1 - blend) * long_vol
    vol = vol.replace(0.0, np.nan).ffill()
    vol_scaled = (price_changes / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return vol_scaled, vol


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: NumPyro Bayesian Latent Factor Model
# ══════════════════════════════════════════════════════════════════════════════

def _bayesian_factor_model(R, K_max):
    """NumPyro probabilistic model for Bayesian factor analysis with ARD.

    Model:
        tau_k ~ HalfCauchy(1)           for k = 1..K_max   (ARD precision)
        B[n,k] ~ Normal(0, tau_k)       loadings
        F[t,k] ~ Normal(0, 1)           factor returns
        sigma_n ~ HalfCauchy(0.5)       idiosyncratic vol
        R[t,n] ~ Normal(F[t,:] @ B[n,:].T, sigma_n)

    Small tau_k effectively kills factor k (ARD shrinkage).
    """
    import numpyro
    import numpyro.distributions as dist
    import jax.numpy as jnp

    T, N = R.shape

    # ARD priors: per-factor relevance scale
    tau = numpyro.sample("tau", dist.HalfCauchy(jnp.ones(K_max)))

    # Factor loadings: (N_assets, K_max) with ARD prior
    B = numpyro.sample("B", dist.Normal(
        jnp.zeros((N, K_max)),
        jnp.broadcast_to(tau, (N, K_max))
    ))

    # Factor returns: (T_window, K_max)
    F = numpyro.sample("F", dist.Normal(
        jnp.zeros((T, K_max)),
        jnp.ones((T, K_max))
    ))

    # Idiosyncratic volatility per asset
    sigma = numpyro.sample("sigma", dist.HalfCauchy(0.5 * jnp.ones(N)))

    # Likelihood: R = F @ B.T + noise
    mu = jnp.matmul(F, B.T)  # (T, N)
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=R)


def run_mcmc_factor_model(returns_window, K_max=8, n_samples=500,
                          n_warmup=300, seed=42):
    """Run NUTS MCMC on the Bayesian factor model.

    Parameters
    ----------
    returns_window : np.ndarray, shape (T_window, N_assets)
        Volatility-scaled, standardized returns.
    K_max : int
        Maximum number of latent factors.
    n_samples : int
        Number of posterior samples after warmup.
    n_warmup : int
        Number of NUTS warmup (adaptation) steps.
    seed : int
        JAX PRNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        "B"     : np.ndarray (N_assets, K_max) — posterior mean loadings
        "F"     : np.ndarray (T_window, K_max) — posterior mean factor returns
        "tau"   : np.ndarray (K_max,) — posterior mean ARD scales
        "sigma" : np.ndarray (N_assets,) — posterior mean idiosyncratic vol
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS

    # Force CPU to avoid GPU memory issues in batch mode
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(1)

    R_jnp = jnp.array(returns_window.astype(np.float32))

    kernel = NUTS(_bayesian_factor_model, max_tree_depth=8,
                  target_accept_prob=0.80)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), R=R_jnp, K_max=K_max)

    samples = mcmc.get_samples()

    return {
        "B":     np.array(samples["B"].mean(axis=0)),      # (N, K_max)
        "F":     np.array(samples["F"].mean(axis=0)),      # (T_win, K_max)
        "tau":   np.array(samples["tau"].mean(axis=0)),     # (K_max,)
        "sigma": np.array(samples["sigma"].mean(axis=0)),   # (N,)
    }


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Residual Return / Price Generator
# ══════════════════════════════════════════════════════════════════════════════

def project_factors_ols(R_t, B):
    """Daily OLS projection: estimate today's factor returns given fixed B.

    F_t = (B'B)^{-1} B' R_t

    Parameters
    ----------
    R_t : np.ndarray, shape (N_assets,)
        Today's cross-sectional vol-scaled returns.
    B : np.ndarray, shape (N_assets, K)
        Current factor loading matrix.

    Returns
    -------
    F_t : np.ndarray, shape (K,)
        Estimated factor returns for today.
    """
    BtB = B.T @ B
    # Regularize for numerical stability
    BtB += 1e-6 * np.eye(BtB.shape[0])
    F_t = np.linalg.solve(BtB, B.T @ R_t)
    return F_t


def compute_residual_returns(returns_t, B, F_t):
    """Strip factor contribution from returns.

    residual[n] = returns[n] - sum_k(B[n,k] * F_t[k])
    """
    return returns_t - B @ F_t


def count_effective_factors(tau, threshold=0.1):
    """Count factors with ARD scale above threshold.

    Factors with tau_k < threshold are considered 'killed' by the ARD prior.
    """
    return int(np.sum(tau > threshold))


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Regime-Aware Position Sizer
# ══════════════════════════════════════════════════════════════════════════════

def compute_regime_multiplier(n_factors_series, ewm_span=5):
    """Compute leverage multiplier based on effective factor count.

    Logic:
      - baseline K = expanding median of n_factors (at least 256 days)
      - K_t > baseline + 2 => 0.50  (fragmented market)
      - K_t > baseline + 1 => 0.75  (moderately complex)
      - K_t <= baseline    => 1.00  (unified macro trend)

    Smoothed with EWM to avoid whipsaws.

    Parameters
    ----------
    n_factors_series : pd.Series
        Effective factor count over time.
    ewm_span : int
        Smoothing span for the multiplier.

    Returns
    -------
    pd.Series : regime leverage multiplier in [0.5, 1.0]
    """
    # Expanding median (causal: only past data)
    baseline = n_factors_series.expanding(min_periods=256).median()
    # Fill early values with first valid baseline
    baseline = baseline.ffill().bfill()

    delta = n_factors_series - baseline
    raw_mult = pd.Series(1.0, index=n_factors_series.index)
    raw_mult[delta > 2] = 0.50
    raw_mult[(delta > 1) & (delta <= 2)] = 0.75

    # Smooth to avoid daily whipsaws
    smoothed = raw_mult.ewm(span=ewm_span, min_periods=1).mean()
    return smoothed.clip(0.5, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Main Class: BayesianRiskFilter
# ══════════════════════════════════════════════════════════════════════════════

class BayesianRiskFilter:
    """Bayesian latent factor risk filter for CTA momentum strategies.

    Intercepts raw price data, extracts time-varying latent factors via
    Bayesian inference, and returns factor-hedged residual prices plus a
    regime-aware leverage multiplier.

    Parameters
    ----------
    refit_every : int
        Trading days between full MCMC refits (default: 21 ~ monthly).
    warmup_days : int
        Minimum number of days before first MCMC fit and rolling window
        length for each refit (default: 512).
    n_samples : int
        NUTS posterior samples per refit (default: 500).
    n_warmup : int
        NUTS warmup steps per refit (default: 300).
    max_factors : int
        Maximum latent factors (K_max for ARD model, default: 8).
    ewm_vol_span : int
        EWM span for return volatility scaling (default: 32).
    ard_threshold : float
        Minimum tau_k to count a factor as active (default: 0.1).
    """

    def __init__(self, refit_every=21, warmup_days=512, n_samples=500,
                 n_warmup=300, max_factors=8, ewm_vol_span=32,
                 ard_threshold=0.1):
        if not _check_numpyro():
            raise ImportError(
                "NumPyro and JAX are required for BayesianRiskFilter. "
                "Install with: pip install numpyro jax jaxlib"
            )
        self.refit_every = refit_every
        self.warmup_days = warmup_days
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.max_factors = max_factors
        self.ewm_vol_span = ewm_vol_span
        self.ard_threshold = ard_threshold

    def filter_returns(self, inst_data_cache, mapping):
        """Main entry point: filter raw returns through the Bayesian factor model.

        Parameters
        ----------
        inst_data_cache : dict
            {instrument: data_dict} where data_dict has keys:
            "daily_dates", "daily_return", "close", "open", "price_changes"
        mapping : pd.DataFrame
            Instrument mapping with "asset_class" column.

        Returns
        -------
        dict with keys:
            "residual_prices"   : {instrument: pd.Series}
            "residual_returns"  : {instrument: pd.Series}
            "regime_multiplier" : pd.Series
            "n_factors"         : pd.Series
            "factor_betas"      : {instrument: np.ndarray}  (N, K_max)
        """
        from tqdm import tqdm

        # ── Step 1: Build aligned vol-scaled return matrix ──────────────
        print("[BayesianRiskFilter] Step 1: Preprocessing & vol-scaling returns...")

        instruments = [inst for inst in mapping.index if inst in inst_data_cache]
        inst_to_col = {inst: i for i, inst in enumerate(instruments)}
        N_assets = len(instruments)

        # Collect all dates across instruments
        all_dates_set = set()
        inst_vol_returns = {}
        inst_raw_returns = {}

        for inst in instruments:
            data = inst_data_cache[inst]
            pc = data["price_changes"]
            vol_ret, _ = vol_scale_returns(pc, ewm_span=self.ewm_vol_span)
            inst_vol_returns[inst] = vol_ret
            inst_raw_returns[inst] = pc
            all_dates_set.update(pc.index.tolist())

        all_dates = pd.DatetimeIndex(sorted(all_dates_set))
        T_total = len(all_dates)

        # Build (T, N) return matrix (vol-scaled)
        R_matrix = np.full((T_total, N_assets), np.nan)
        for inst, col in inst_to_col.items():
            aligned = inst_vol_returns[inst].reindex(all_dates)
            R_matrix[:, col] = aligned.values

        # Forward-fill then zero-fill NaNs for matrix completeness
        R_df = pd.DataFrame(R_matrix, index=all_dates)
        R_df = R_df.ffill().fillna(0.0)
        R_matrix = R_df.values

        print(f"  Return matrix: {T_total} days x {N_assets} assets")

        # ── Step 2 & 3: Walk-forward MCMC + residual computation ────────
        print("[BayesianRiskFilter] Step 2-3: Walk-forward Bayesian factor extraction...")

        import pickle as _pkl
        from pathlib import Path as _Path
        _STEP23_CK = _Path(__file__).resolve().parent / "Strategy_88" / "meta_checkpoint_step23.pkl"

        residual_matrix = np.copy(R_matrix)  # will overwrite with residuals
        n_factors_arr = np.full(T_total, np.nan)

        # State: current model parameters (None until first fit)
        current_B = None
        current_tau = None
        last_refit_t = -999
        refit_count = 0
        start_t = self.warmup_days

        # ── Try to resume from step 2-3 checkpoint ──
        if _STEP23_CK.exists():
            try:
                with open(str(_STEP23_CK), "rb") as _fh:
                    _ck23 = _pkl.load(_fh)
                # Restore state
                residual_matrix[:_ck23["t"]+1, :] = _ck23["residual_matrix"][:_ck23["t"]+1, :]
                n_factors_arr[:_ck23["t"]+1] = _ck23["n_factors_arr"][:_ck23["t"]+1]
                current_B    = _ck23["current_B"]
                current_tau  = _ck23["current_tau"]
                last_refit_t = _ck23["last_refit_t"]
                refit_count  = _ck23["refit_count"]
                start_t      = _ck23["t"] + 1
                print(f"  [RESUME] Step 2-3 checkpoint loaded: resuming from t={start_t}"
                      f"  ({refit_count} refits done, {T_total - start_t} days remaining)")
            except Exception as e:
                print(f"  [WARN] Failed to load step 2-3 checkpoint: {e} — starting fresh.")

        n_refits_expected = max(1, (T_total - self.warmup_days) // self.refit_every)

        # Checkpoint save interval: every 10 refits (~210 trading days)
        _SAVE_EVERY_REFITS = 10
        _last_save_refit = refit_count

        for t in tqdm(range(start_t, T_total), desc="BayesianFilter",
                      initial=start_t - self.warmup_days,
                      total=T_total - self.warmup_days):
            need_refit = (t - last_refit_t) >= self.refit_every

            # ── Full MCMC refit ──
            if need_refit:
                window_start = max(0, t - self.warmup_days)
                R_window = R_matrix[window_start:t, :]

                # Check that window has sufficient non-zero data
                valid_cols = np.sum(np.abs(R_window) > 1e-10, axis=0) > (self.warmup_days // 4)
                if valid_cols.sum() < 5:
                    # Not enough data for meaningful factor extraction
                    if current_B is not None:
                        # Use existing model
                        pass
                    else:
                        n_factors_arr[t] = 0
                        continue
                else:
                    # Subset to valid columns for MCMC
                    valid_idx = np.where(valid_cols)[0]
                    R_sub = R_window[:, valid_idx]

                    # Standardize cross-sectionally for numerical stability
                    col_std = R_sub.std(axis=0)
                    col_std[col_std < 1e-8] = 1.0
                    R_sub = R_sub / col_std

                    try:
                        result = run_mcmc_factor_model(
                            R_sub,
                            K_max=self.max_factors,
                            n_samples=self.n_samples,
                            n_warmup=self.n_warmup,
                            seed=42 + refit_count,
                        )

                        # Expand back to full N_assets (zero loadings for invalid).
                        # result["B"] has shape (len(valid_idx), K_max) — rows
                        # are in the order of valid_idx, NOT indexed by valid_idx.
                        B_full = np.zeros((N_assets, self.max_factors))
                        for out_i, orig_col in enumerate(valid_idx):
                            B_full[orig_col, :] = result["B"][out_i, :] * col_std[out_i]
                        current_B = B_full
                        current_tau = result["tau"]
                        last_refit_t = t
                        refit_count += 1

                    except Exception as e:
                        print(f"  [WARN] MCMC failed at t={t}: {e}")
                        if current_B is None:
                            n_factors_arr[t] = 0
                            continue

                # ── Periodic checkpoint save ──
                if (refit_count - _last_save_refit) >= _SAVE_EVERY_REFITS:
                    _STEP23_CK.parent.mkdir(parents=True, exist_ok=True)
                    with open(str(_STEP23_CK), "wb") as _fh:
                        _pkl.dump({
                            "t": t,
                            "residual_matrix": residual_matrix,
                            "n_factors_arr": n_factors_arr,
                            "current_B": current_B,
                            "current_tau": current_tau,
                            "last_refit_t": last_refit_t,
                            "refit_count": refit_count,
                        }, _fh)
                    _last_save_refit = refit_count
                    print(f"\n  [CHECKPOINT] Saved at t={t} ({refit_count} refits)")

            # ── Daily projection using current model ──
            if current_B is not None:
                R_t = R_matrix[t, :]
                F_t = project_factors_ols(R_t, current_B)
                residual_matrix[t, :] = compute_residual_returns(R_t, current_B, F_t)
                n_factors_arr[t] = count_effective_factors(
                    current_tau, threshold=self.ard_threshold
                )
            else:
                n_factors_arr[t] = 0

        print(f"  Completed {refit_count} MCMC refits over {T_total} days")

        # Clean up step 2-3 checkpoint (completed successfully)
        if _STEP23_CK.exists():
            _STEP23_CK.unlink()
            print("  [CHECKPOINT] Step 2-3 checkpoint removed (completed).")

        # ── Step 4: Regime multiplier ───────────────────────────────────
        print("[BayesianRiskFilter] Step 4: Computing regime multiplier...")

        n_factors = pd.Series(n_factors_arr, index=all_dates)
        n_factors = n_factors.ffill().fillna(0.0)
        regime_mult = compute_regime_multiplier(n_factors)

        # ── Pack results per instrument ─────────────────────────────────
        print("[BayesianRiskFilter] Packing per-instrument residual prices...")

        residual_prices = {}
        residual_returns = {}
        factor_betas = {}

        for inst, col in inst_to_col.items():
            data = inst_data_cache[inst]
            dates = data["daily_dates"]
            idx = pd.DatetimeIndex(dates) if not isinstance(dates, pd.DatetimeIndex) else dates

            # Get residual vol-scaled returns for this instrument
            resid_vol_ret = pd.Series(residual_matrix[:, col], index=all_dates)
            resid_vol_ret = resid_vol_ret.reindex(idx).fillna(0.0)

            # Convert residual vol-scaled returns back to price-scale residuals
            # by multiplying by the original volatility
            _, inst_vol = vol_scale_returns(data["price_changes"],
                                            ewm_span=self.ewm_vol_span)
            resid_price_changes = resid_vol_ret * inst_vol.reindex(idx).ffill().fillna(
                inst_vol.median()
            )

            # Reconstruct residual cumulative price from original starting price
            # plus cumulative residual price changes
            orig_close = data["close"]
            start_price = orig_close.iloc[0] if len(orig_close) > 0 else 1.0
            resid_cum_price = start_price + resid_price_changes.cumsum()

            residual_prices[inst] = resid_cum_price
            residual_returns[inst] = resid_price_changes

            if current_B is not None:
                factor_betas[inst] = current_B[col, :]

        return {
            "residual_prices":   residual_prices,
            "residual_returns":  residual_returns,
            "regime_multiplier": regime_mult,
            "n_factors":         n_factors,
            "factor_betas":      factor_betas,
        }
