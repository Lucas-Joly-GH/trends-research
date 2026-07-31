"""
hmm_utils.py – Optimized walk-forward HMM for regime detection
==============================================================
Shared by HMM strategies (34–42). Three optimizations over naive loop:

  1. Early-stopping EM  (tol=1e-2, n_iter ceiling 100)
  2. Batched emission log-probs + inline forward algorithm
     between refits (eliminates per-day predict_proba overhead)
  3. joblib parallel helper for instrument loops
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Core: walk-forward HMM danger-probability computation
# ---------------------------------------------------------------------------

def compute_hmm_danger_probs(feat_arr, burn_in, T, refit_every, hmm_lookback,
                              n_components=2, n_iter=100, tol=1e-2):
    """
    Walk-forward HMM regime detection with batched computation.

    Parameters
    ----------
    feat_arr     : ndarray (T, n_features), pre-cleaned (no NaN/Inf)
    burn_in      : int, minimum training window before first fit
    T            : int, total number of timesteps (== feat_arr.shape[0])
    refit_every  : int, days between HMM re-estimations
    hmm_lookback : int, sliding-window length for forward algorithm
    n_components : int, number of HMM states (default 2)
    n_iter       : int, EM iteration ceiling (default 100)
    tol          : float, EM convergence tolerance (default 1e-2)

    Returns
    -------
    prob_danger : ndarray (T,), probability of danger state at each t
                  (0.0 for t < burn_in or before first successful fit)
    """
    prob_danger = np.zeros(T)
    if burn_in >= T:
        return prob_danger

    n_feat = feat_arr.shape[1]
    log2pi_d = n_feat * np.log(2.0 * np.pi)

    scaler = StandardScaler()
    last_fit_t = -999
    danger_idx = 0

    # Cached model parameters (set on first successful fit)
    lsp = lt = None                       # log startprob, log transmat
    means = cov_invs = logdets = None

    t = burn_in
    while t < T:
        need_refit = (t - last_fit_t) >= refit_every

        # ── Refit EM ──
        if need_refit:
            train_start = max(0, t - burn_in)
            scaled_train = scaler.fit_transform(feat_arr[train_start:t])
            hmm = GaussianHMM(n_components=n_components, covariance_type="full",
                              n_iter=n_iter, tol=tol, random_state=42)
            try:
                hmm.fit(scaled_train)
                danger_idx = int(np.argmax(hmm.means_[:, 0]))
                last_fit_t = t
                lsp = np.log(np.maximum(hmm.startprob_, 1e-300))
                lt  = np.log(np.maximum(hmm.transmat_,  1e-300))
                means    = hmm.means_.copy()
                _covars  = hmm.covars_.copy()
                cov_invs = np.array([np.linalg.inv(_covars[k])
                                     for k in range(n_components)])
                logdets  = np.array([np.linalg.slogdet(_covars[k])[1]
                                     for k in range(n_components)])
            except Exception:
                if lsp is None:          # never fitted → skip
                    t += 1
                    continue

        if lsp is None:                  # still no model
            t += 1
            continue

        # ── Batch predict until next refit point ──
        chunk_end = min(T, last_fit_t + refit_every)
        if chunk_end <= t:
            chunk_end = t + 1            # at least advance one day

        earliest = max(0, t - hmm_lookback + 1)
        all_scaled = scaler.transform(feat_arr[earliest:chunk_end])

        # Emission log-probs for all observations in chunk (vectorised)
        n_obs = all_scaled.shape[0]
        log_emis = np.empty((n_obs, n_components))
        for k in range(n_components):
            diff  = all_scaled - means[k]
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
            pd = np.exp(la[danger_idx] - log_norm)
            prob_danger[day_t] = pd if not np.isnan(pd) else 0.0

        t = chunk_end

    return prob_danger


# ---------------------------------------------------------------------------
# Parallel instrument loop helper
# ---------------------------------------------------------------------------

def parallel_instrument_map(func, instruments, n_jobs=-1, desc="HMM"):
    """
    Run ``func(instrument)`` in parallel over *instruments*.

    Parameters
    ----------
    func        : callable(instrument) -> dict or None
    instruments : iterable of instrument keys
    n_jobs      : int, joblib concurrency (-1 = all cores)
    desc        : str, label for progress output

    Returns
    -------
    dict  :  {instrument: result} for non-None results
    """
    insts = list(instruments)
    print(f"  [{desc}] Dispatching {len(insts)} instruments across "
          f"{n_jobs if n_jobs > 0 else 'all'} cores …")
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(func)(inst) for inst in insts
    )
    out = {}
    for inst, res in zip(insts, results):
        if res is not None:
            out[inst] = res
    print(f"  [{desc}] {len(out)}/{len(insts)} instruments completed.")
    return out
