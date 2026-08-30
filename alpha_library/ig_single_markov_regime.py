"""
Single Alpha: Markov Regime -- Posterior expected-return alpha
==================================================================
A walk-forward 2-state Gaussian HMM is fit on each instrument's daily
returns. At each time t, the forecast is the posterior-weighted
expected return:

    raw_t  =  ( sum_k  P(state_k | r_<=t) * mu_k )  /  sigma_t

where mu_k is the mean of state k under the most recent EM fit (re-fit
every REFIT_EVERY days on a rolling BURN_IN-day window) and sigma_t is
the current rolling daily volatility. The forecast then inherits S183's
standard scaling (rolling_forecast_scalar -> FORECAST_TARGET) and capping
(FORECAST_CAP).

Conceptually: when the HMM puts posterior mass on the positive-drift
regime, the strategy goes long; when it puts mass on the negative-drift
regime, it goes short. The HMM is therefore the *alpha source*, not a
filter on top of another signal (contrast with Strategy_34 which used
HMM danger probability as a multiplier on an EWMAC ensemble).

Architecture: identical to other ig_single_* strategies (single instrument
forecast -> S183's run_compounded_portfolio for vol targeting, FDM,
position sizing, T-bill cash yield). The only thing that varies is the
signal.

References:
- Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle." Econometrica 57(2).
- Ang, A. and G. Bekaert (2002). "International Asset Allocation With
  Regime Shifts." Review of Financial Studies 15(4).
- Bulla, J., S. Mergner, I. Bulla, A. Sesboue, C. Chesneau (2011).
  "Markov-switching asset allocation: Do profitable strategies exist?"
  Journal of Asset Management 12(5).
- Nystrup, P., H. Madsen, E. Lindstrom (2015). "Stylised facts of
  financial time series and hidden Markov models in continuous time."
  Quantitative Finance 15(9).
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

STRATEGY_NAME = "Single_Alpha_Markov_Regime"

# HMM hyperparameters (mirror Strategy_34's convention so the comparison
# is on signal definition only, not on training protocol)
N_STATES       = 2
BURN_IN        = 750     # min training window before first fit
REFIT_EVERY    = 64      # ~3 months between EM re-estimations
HMM_LOOKBACK   = 50      # forward-algorithm sliding window length
EM_N_ITER      = 100
EM_TOL         = 1e-2

OOS_START = 1280         # standard for single_alpha (5y warmup)


def compute_hmm_expected_return(ret_arr, vol_arr,
                                burn_in=BURN_IN,
                                refit_every=REFIT_EVERY,
                                hmm_lookback=HMM_LOOKBACK,
                                n_components=N_STATES,
                                n_iter=EM_N_ITER,
                                tol=EM_TOL):
    """
    Walk-forward HMM posterior expected return, normalised by current vol.

    For each t >= burn_in, fits a Gaussian HMM on returns up to t (refit
    every refit_every days), then computes the posterior P(state_k | r_<=t)
    via the forward algorithm on a sliding window, and returns:

        raw_t = ( sum_k posterior_k * mu_k ) / vol_t

    Parameters
    ----------
    ret_arr      : ndarray (T,) of cleaned daily returns
    vol_arr      : ndarray (T,) of daily volatilities (same units as ret)
    burn_in      : int, minimum training window before first fit
    refit_every  : int, days between HMM re-estimations
    hmm_lookback : int, sliding-window length for forward algorithm
    n_components : int, number of HMM states
    n_iter       : int, EM iteration ceiling
    tol          : float, EM convergence tolerance

    Returns
    -------
    raw : ndarray (T,) of raw forecasts (NaN for t < burn_in or before
          first successful fit)
    """
    T = len(ret_arr)
    raw = np.full(T, np.nan)
    if burn_in >= T:
        return raw

    # Single feature: returns
    feat_arr = ret_arr.reshape(-1, 1).astype(float)

    log2pi_d = np.log(2.0 * np.pi)  # n_feat = 1

    scaler = StandardScaler()
    last_fit_t = -999

    # Cached parameters (set on first successful fit)
    lsp = lt = None                       # log startprob, log transmat
    means_orig = None                     # state means in ORIGINAL units (for forecast)
    means_sc = cov_invs = logdets = None  # in SCALED units (for emission probs)

    t = burn_in
    while t < T:
        need_refit = (t - last_fit_t) >= refit_every

        # --- Refit EM ---
        if need_refit:
            train_start = max(0, t - burn_in)
            train_data  = feat_arr[train_start:t]
            # Skip if too few finite observations
            if not np.isfinite(train_data).all() or len(train_data) < 30:
                t += 1
                continue
            scaled_train = scaler.fit_transform(train_data)
            hmm = GaussianHMM(n_components=n_components, covariance_type="diag",
                              n_iter=n_iter, tol=tol, random_state=42,
                              init_params="stc")
            # Smart init: spread means at empirical 25th/75th percentiles
            # of scaled training data (avoids degenerate fits where both
            # states converge to similar means).
            qs = np.linspace(0.25, 0.75, n_components)
            hmm.means_ = np.quantile(scaled_train, qs, axis=0).reshape(
                n_components, -1)
            try:
                hmm.fit(scaled_train)
                last_fit_t = t
                lsp = np.log(np.maximum(hmm.startprob_, 1e-300))
                lt  = np.log(np.maximum(hmm.transmat_,  1e-300))
                means_sc = hmm.means_.copy()
                # covars_ shape: (n_components, n_features) for "diag",
                # (n_components, n_features, n_features) for "full".
                # Promote diag to full for uniform downstream handling.
                if hmm.covars_.ndim == 2:
                    _covars = np.array([np.diag(hmm.covars_[k])
                                        for k in range(n_components)])
                else:
                    _covars = hmm.covars_.copy()
                cov_invs = np.array([np.linalg.inv(_covars[k])
                                     for k in range(n_components)])
                logdets  = np.array([np.linalg.slogdet(_covars[k])[1]
                                     for k in range(n_components)])
                # State means in original return units
                # (StandardScaler: x_orig = x_sc * scale_ + mean_)
                means_orig = (means_sc[:, 0] * scaler.scale_[0]
                              + scaler.mean_[0])
            except Exception:
                if lsp is None:
                    t += 1
                    continue

        if lsp is None:
            t += 1
            continue

        # --- Batch predict until next refit point ---
        chunk_end = min(T, last_fit_t + refit_every)
        if chunk_end <= t:
            chunk_end = t + 1

        earliest = max(0, t - hmm_lookback + 1)
        scaled_chunk = scaler.transform(feat_arr[earliest:chunk_end])

        # Emission log-probs (vectorised)
        n_obs = scaled_chunk.shape[0]
        log_emis = np.empty((n_obs, n_components))
        for k in range(n_components):
            diff  = scaled_chunk - means_sc[k]
            mahal = np.sum(diff @ cov_invs[k] * diff, axis=1)
            log_emis[:, k] = -0.5 * (log2pi_d + logdets[k] + mahal)

        # Forward algorithm per sliding window
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
            posterior = np.exp(la - log_norm)  # shape (n_components,)

            # Posterior-weighted expected return (in original return units)
            exp_ret = float(np.dot(posterior, means_orig))

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

        # Daily returns (proportional, for HMM emission distribution)
        daily_ret = close.pct_change().fillna(0.0).replace(
            [np.inf, -np.inf], 0.0).values

        # Volatility series in same units as daily_ret (proportional)
        # vol from blended_volatility is in price units; convert to %
        # by dividing by close, then align to idx
        vol_pct = (vol / close).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values

        raw_arr = compute_hmm_expected_return(
            ret_arr=daily_ret,
            vol_arr=vol_pct,
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
