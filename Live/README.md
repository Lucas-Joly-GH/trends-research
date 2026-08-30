# Live — data policy

This repository is public. **Vendor data is not committed to it.** The cleaned,
back-adjusted price panel and its roll logic live in a private repository and
never leave it.

The published site is a deliberate, narrow exception. `docs/` serves a results
page for the 2026 run, and it carries prices — the open each order filled at and
the close it was decided on — because a table reading "SELL 37 6B" with no prices
cannot be checked by anybody, and an unverifiable results page is worth less than
no page. The site publishes exactly enough for a reader to recompute a session by
hand:

    N x (close - open) x pointsize x FX  =  the P&L claimed

The carve-out is the whole policy:

  * **2026 sessions only.** Nothing dated before 2026-01-02 is ever published.
  * **Open and close only.** No back-adjusted series, no volatility estimate, no
    forecast, no position-sizing provenance.
  * **Non-commercial.** A thesis and a personal record, not a product.

The reasoning, so it can be judged rather than inherited: current-year settlement
prices for liquid listed futures are published by the exchanges themselves and
carried by many free sources. That is not the proprietary asset. The proprietary
asset is the cleaned 1978–2026 panel with its roll logic, and that stays private.

`5_Publish/publish.py` enforces this with a whitelist — a hardcoded list of the
columns permitted to leave — and fails the build if any output contains a column
outside it. A blacklist fails open the day someone adds a field; a whitelist
fails closed. A second guard sweeps every published payload for any value shaped
like a date before 2026-01-02 and aborts on one.

The universe list is the one exception that lives here, at
`instrument_mapping.csv`. It carries no vendor price data.
