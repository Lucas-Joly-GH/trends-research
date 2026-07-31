# Alpha library — 45 individually-tested signals

Each `ig_single_*.py` module isolates one alpha and backtests it standalone over the full
1990–2026 window, so its contribution can be measured before it enters any composite. This
is the raw-material stage: the flagship S183 strategy blends four of these (Trend, Carry,
Skew, Vol-of-Vol) in equal weight.

- `single_alpha_summary.csv` — full ranking (Sharpe, post-2010 Sharpe, CAGR, max DD, avg correlation).
- `single_alpha_correlation_matrix.csv` / `single_alpha_correlation_heatmap.png` — cross-alpha correlation structure (the diversification case).
- `optimal_blend_optimizer.py` / `optimal_blend_v2.py` — simplex weight search (which confirmed equal-weighting is unbeaten).

Signals span trend (EWMAC family, breakout, anchored/normalised momentum), carry (structural,
relative, basis, curvature), cross-sectional and value factors, higher-moment signals
(skew, kurtosis, tail asymmetry, semivariance), microstructure (Amihud illiquidity, OI/volume,
lead-lag), regime models (Markov, HMM, Hurst) and calendar effects (seasonality, FOMC/EOM drift).
