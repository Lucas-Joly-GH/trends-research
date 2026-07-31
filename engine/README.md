# Engine — backtesting framework

The shared infrastructure every strategy runs on. Signal generation is vectorised in
NumPy; the day-by-day walk-forward (position sizing → P&L → compounding) is accelerated in
C++.

| File | Role |
|---|---|
| `ig_shared_config.py` | Core compounding engine: date alignment, forecast-to-position sizing, geometric metrics, checkpointing. |
| `fast_backtest.cpp` | C++ (pybind11) accelerator for the Phase-3 walk-forward loop. |
| `fast_backtest_bridge.py` | Python ↔ C++ bridge with a pure-Python fallback. |
| `setup_fast_backtest.py` | Build script for the C++ extension. |
| `hmm_utils.py` | Walk-forward Gaussian HMM regime detection. |
| `bayesian_risk_filter.py` | Bayesian latent-factor risk model (NumPyro / JAX). |

> Note: this is the engine layer only. Data loaders and the raw price panel live in the
> larger private repo and are excluded here.
