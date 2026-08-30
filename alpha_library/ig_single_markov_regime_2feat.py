"""
Single Alpha: Markov Regime (2-feature variant) -- Posterior expected-return alpha
====================================================================================
Identical methodology to ig_single_markov_regime.py except the HMM is fit
on TWO features per day:

    feature 0 = daily return                  (signed)
    feature 1 = absolute daily return         (volatility proxy)

The intuition is that financial returns exhibit volatility clustering
(GARCH-type effects). A 2-state Gaussian HMM on returns alone tends to
collapse the volatility structure into the variance of each state. Adding
|return| as an explicit feature lets the HMM separate the canonical regime
structure {bull-calm, bear-volatile} more cleanly, which is the form most
commonly found in the regime-switching literature on equity / multi-asset
panels.

The alpha is still the posterior-weighted expected return divided by current
vol -- only the regime-detection input is enriched. This file exists so that
the "minimal HMM" defence (returns only, ig_single_markov_regime.py) can be
triangulated against a richer-features version, pre-empting the
"you weren't trying hard enough" critique.

References (in addition to those of ig_single_markov_regime.py):
- Ang, A. and G. Bekaert (2002). "International Asset Allocation With
  Regime Shifts." Review of Financial Studies 15(4). Use returns + vol
  as joint emission features in a 2-state HMM.
- Guidolin, M. and A. Timmermann (2007). "Asset allocation under
  multivariate regime switching." Journal of Economic Dynamics and
  Control 31(11). 2-4 state HMMs on returns + dispersion.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from ig_shared_config import (
    FORECAST_CAP, FORECAST_TARGET, VOL_TARGET,
    get_project_paths, load_mapping, load_fx_rates, load_instrument_data,
    align_fx, blended_volatility, cap_forecast, run_compounded_portfolio,
    rolling_forecast_scalar,
)

STRATEGY_NAME = "Single_Alpha_Markov_Regime_2Feat"

# HMM hyperparameters (identical to ig_single_markov_regime.py)
N_STATES       = 2
BURN_IN        = 750
REFIT_EVERY    = 64
HMM_LOOKBACK   = 50
EM_N_ITER      = 100
EM_TOL         = 1e-2

OOS_START = 1280


def compute_hmm_expected_return_2feat(feat_arr, vol_arr, return_col=0,
                                      burn_in=BURN_IN,
                                      refit_every=REFIT_EVERY,
                                      hmm_lookback=HMM_LOOKBACK,
                                      n_components=N_STATES,
                                      n_iter=EM_N_ITER,
                                      tol=EM_TOL):
    """
    Walk-forward HMM posterior expected return (multi-feature emission).

    Parameters
    ----------
    feat_arr     : ndarray (T, n_features) of cleaned features
    vol_arr      : ndarray (T,) of daily volatilities (proportional)
    return_col   : int, index of the "return" feature in feat_arr
                   (the expected-value forecast is computed in this dim)
    burn_in, refit_every, hmm_lookback, n_components, n_iter, tol :
                   see ig_single_markov_regime.py
    """
    T, n_feat = feat_arr.shape
    raw = np.full(T, np.nan)
    if burn_in >= T:
        return raw

    log2pi_d = n_feat * np.log(2.0 * np.pi)

    scaler = StandardScaler()
    last_fit_t = -999

    lsp = lt = None
    means_orig_ret = None  # state means in ORIGINAL units, return col only
    means_sc = cov_invs = logdets = None

    t = burn_in
    while t < T:
        need_refit = (t - last_fit_t) >= refit_every

        if need_refit:
            train_start = max(0, t - burn_in)
            train_data  = feat_arr[train_start:t]
            if not np.isfinite(train_data).all() or len(train_data) < 30:
                t += 1
                continue
            scaled_train = scaler.fit_transform(train_data)
            hmm = GaussianHMM(n_components=n_components, covariance_type="diag",
                              n_iter=n_iter, tol=tol, random_state=42,
                              init_params="stc")
            # Smart init: spread per-feature means at empirical quantiles.
            qs = np.linspace(0.25, 0.75, n_components)
            hmm.means_ = np.quantile(scaled_train, qs, axis=0).reshape(
                n_components, n_feat)
            try:
                hmm.fit(scaled_train)
                last_fit_t = t
                lsp = np.log(np.maximum(hmm.startprob_, 1e-300))
                lt  = np.log(np.maximum(hmm.transmat_,  1e-300))
                means_sc = hmm.means_.copy()
                if hmm.covars_.ndim == 2:
                    _covars = np.array([np.diag(hmm.covars_[k])
                                        for k in range(n_components)])
                else:
                    _covars = hmm.covars_.copy()
                cov_invs = np.array([np.linalg.inv(_covars[k])
                                     for k in range(n_components)])
                logdets  = np.array([np.linalg.slogdet(_covars[k])[1]
                                     for k in range(n_components)])
                # Reconstruct return-column means in original units
                means_orig_ret = (means_sc[:, return_col]
                                  * scaler.scale_[return_col]
                                  + scaler.mean_[return_col])
            except Exception:
                if lsp is None:
                    t += 1
                    continue

        if lsp is None:
            t += 1
            continue

        chunk_end = min(T, last_fit_t + refit_every)
        if chunk_end <= t:
            chunk_end = t + 1

        earliest = max(0, t - hmm_lookback + 1)
        scaled_chunk = scaler.transform(feat_arr[earliest:chunk_end])

        n_obs = scaled_chunk.shape[0]
        log_emis = np.empty((n_obs, n_components))
        for k in range(n_components):
            diff  = scaled_chunk - means_sc[k]
            mahal = np.sum(diff @ cov_invs[k] * diff, axis=1)
            log_emis[:, k] = -0.5 * (log2pi_d + logdets[k] + mahal)

        for day_t in range(t, chunk_end):
            seq_start = max(0, day_t - hmm_lookback + 1)
            i_start   = seq_start - earliest
            i_end     = day_t - earliest + 1

            la = lsp.copy() + log_emis[i_start]
            for s in range(i_start + 1, i_end):
                a0 = np.logaddexp(la[0] + lt[0, 0],
                                  la[1] + lt[1, 0]) + log_emis[s, 0]
                a1 = np.logaddexp(la[0] + lt[0, 1],
                                  la[1] + lt[1, 1]) + log_emis[s, 1]
                la[0], la[1] = a0, a1

            log_norm = np.logaddexp(la[0], la[1])
            posterior = np.exp(la - log_norm)

            exp_ret = float(np.dot(posterior, means_orig_ret))

            v_t = vol_arr[day_t] if day_t < len(vol_arr) else np.nan
            if not np.isfinite(v_t) or v_t < 1e-10:
                continue
            raw[day_t] = exp_ret / v_t

        t = chunk_end

    return raw


def run_strategy():
    paths    = get_project_paths(_HERE)
    mapping  = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])

    inst_signals = {}
    for instrument in tqdm(mapping.index, desc=STRATEGY_NAME):
        data = load_instrument_data(instrument, paths["stats"], paths["panama"])
        if data is None:
            continue
        dates = data["daily_dates"]
        if len(dates) < OOS_START + 2:
            continue
        ccy = mapping.loc[instrument, "currency"]
        fx  = align_fx(ccy, dates, fx_daily)
        if fx is None:
            continue

        close         = data["close"]
        price_changes = data["price_changes"]
        vol           = blended_volatility(price_changes)
        idx           = pd.DatetimeIndex(dates)

        daily_ret = close.pct_change().fillna(0.0).replace(
            [np.inf, -np.inf], 0.0).values
        abs_ret = np.abs(daily_ret)
        feat_arr = np.column_stack([daily_ret, abs_ret])

        vol_pct = (vol / close).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values

        raw_arr = compute_hmm_expected_return_2feat(
            feat_arr=feat_arr,
            vol_arr=vol_pct,
            return_col=0,
            burn_in=BURN_IN,
            refit_every=REFIT_EVERY,
            hmm_lookback=HMM_LOOKBACK,
            n_components=N_STATES,
        )

        raw = pd.Series(raw_arr, index=idx).fillna(0.0)
        forecast_scaled = cap_forecast(raw * rolling_forecast_scalar(raw))

        final_fc = forecast_scaled.reindex(idx).fillna(0.0)
        final_fc.iloc[:OOS_START] = 0.0

        inst_signals[instrument] = {
            "oos_date_set": set(dates[OOS_START:]),
            "forecast":     final_fc,
            "vol":          vol.reindex(idx),
            "fx":           fx.reindex(idx),
            "close":        close.reindex(idx),
            "open":         data["open"].reindex(idx),
            "pointsize":    float(mapping.loc[instrument, "pointsize"]),
            "cost_rt":      float(mapping.loc[instrument, "total_avg_cost_rt"]),
        }

    run_compounded_portfolio(inst_signals, STRATEGY_NAME, paths)


if __name__ == "__main__":
    run_strategy()
