"""
Single Alpha: FOMC Pre-Announcement Drift
==========================================
Lucca & Moench (2015), "The Pre-FOMC Announcement Drift",
Journal of Finance 70(1):329 -- equity index returns earn ~80% of
their unconditional risk premium in the 24h window preceding
scheduled FOMC announcements.

Mechanism
---------
Long equity-index futures from close on day t-1 (where t is an FOMC
meeting day) through the announcement at ~14:00 ET on day t. We
approximate this by holding a long position on day t-1 (the day BEFORE
the FOMC meeting) and the meeting day itself, then flat.

Universe
--------
Equity-index instruments only (asset_class == "Equity"). Other markets
are not the subject of the Lucca-Moench paper.

Signal
------
Pure calendar dummy on equity instruments: +1 on (t-1, t) for each FOMC
meeting; 0 otherwise. Then rolling-scalar normalised.

FOMC dates 1990-2026 hardcoded from federalreserve.gov/monetarypolicy/
fomccalendars.htm (scheduled meetings only -- no inter-meetings).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from ig_shared_config import (
    FORECAST_CAP, FORECAST_TARGET, VOL_TARGET,
    get_project_paths, load_mapping, load_fx_rates, load_instrument_data,
    align_fx, blended_volatility,
    cap_forecast, run_compounded_portfolio, rolling_forecast_scalar,
)

STRATEGY_NAME = "Single_Alpha_FOMC_Drift"
OOS_START     = 1280

# FOMC scheduled meeting dates 1990-2026 (announcement day; for 2-day
# meetings this is the second/decision day).  Source: Federal Reserve.
FOMC_DATES = [
    # 1990
    "1990-02-07","1990-03-27","1990-05-15","1990-07-03","1990-08-21","1990-10-02","1990-11-13","1990-12-18",
    # 1991
    "1991-02-06","1991-03-26","1991-05-14","1991-07-03","1991-08-20","1991-10-01","1991-11-05","1991-12-17",
    # 1992
    "1992-02-05","1992-03-31","1992-05-19","1992-06-30","1992-08-18","1992-10-06","1992-11-17","1992-12-22",
    # 1993
    "1993-02-03","1993-03-23","1993-05-18","1993-07-07","1993-08-17","1993-09-21","1993-11-16","1993-12-21",
    # 1994
    "1994-02-04","1994-03-22","1994-05-17","1994-07-06","1994-08-16","1994-09-27","1994-11-15","1994-12-20",
    # 1995
    "1995-02-01","1995-03-28","1995-05-23","1995-07-06","1995-08-22","1995-09-26","1995-11-15","1995-12-19",
    # 1996
    "1996-01-31","1996-03-26","1996-05-21","1996-07-03","1996-08-20","1996-09-24","1996-11-13","1996-12-17",
    # 1997
    "1997-02-05","1997-03-25","1997-05-20","1997-07-02","1997-08-19","1997-09-30","1997-11-12","1997-12-16",
    # 1998
    "1998-02-04","1998-03-31","1998-05-19","1998-07-01","1998-08-18","1998-09-29","1998-11-17","1998-12-22",
    # 1999
    "1999-02-03","1999-03-30","1999-05-18","1999-06-30","1999-08-24","1999-10-05","1999-11-16","1999-12-21",
    # 2000
    "2000-02-02","2000-03-21","2000-05-16","2000-06-28","2000-08-22","2000-10-03","2000-11-15","2000-12-19",
    # 2001
    "2001-01-31","2001-03-20","2001-05-15","2001-06-27","2001-08-21","2001-10-02","2001-11-06","2001-12-11",
    # 2002
    "2002-01-30","2002-03-19","2002-05-07","2002-06-26","2002-08-13","2002-09-24","2002-11-06","2002-12-10",
    # 2003
    "2003-01-29","2003-03-18","2003-05-06","2003-06-25","2003-08-12","2003-09-16","2003-10-28","2003-12-09",
    # 2004
    "2004-01-28","2004-03-16","2004-05-04","2004-06-30","2004-08-10","2004-09-21","2004-11-10","2004-12-14",
    # 2005
    "2005-02-02","2005-03-22","2005-05-03","2005-06-30","2005-08-09","2005-09-20","2005-11-01","2005-12-13",
    # 2006
    "2006-01-31","2006-03-28","2006-05-10","2006-06-29","2006-08-08","2006-09-20","2006-10-25","2006-12-12",
    # 2007
    "2007-01-31","2007-03-21","2007-05-09","2007-06-28","2007-08-07","2007-09-18","2007-10-31","2007-12-11",
    # 2008
    "2008-01-30","2008-03-18","2008-04-30","2008-06-25","2008-08-05","2008-09-16","2008-10-29","2008-12-16",
    # 2009
    "2009-01-28","2009-03-18","2009-04-29","2009-06-24","2009-08-12","2009-09-23","2009-11-04","2009-12-16",
    # 2010
    "2010-01-27","2010-03-16","2010-04-28","2010-06-23","2010-08-10","2010-09-21","2010-11-03","2010-12-14",
    # 2011
    "2011-01-26","2011-03-15","2011-04-27","2011-06-22","2011-08-09","2011-09-21","2011-11-02","2011-12-13",
    # 2012
    "2012-01-25","2012-03-13","2012-04-25","2012-06-20","2012-08-01","2012-09-13","2012-10-24","2012-12-12",
    # 2013
    "2013-01-30","2013-03-20","2013-05-01","2013-06-19","2013-07-31","2013-09-18","2013-10-30","2013-12-18",
    # 2014
    "2014-01-29","2014-03-19","2014-04-30","2014-06-18","2014-07-30","2014-09-17","2014-10-29","2014-12-17",
    # 2015
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17","2015-07-29","2015-09-17","2015-10-28","2015-12-16",
    # 2016
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27","2016-09-21","2016-11-02","2016-12-14",
    # 2017
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26","2017-09-20","2017-11-01","2017-12-13",
    # 2018
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01","2018-09-26","2018-11-08","2018-12-19",
    # 2019
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    # 2020
    "2020-01-29","2020-03-18","2020-04-29","2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    # 2021
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28","2021-09-22","2021-11-03","2021-12-15",
    # 2022
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27","2022-09-21","2022-11-02","2022-12-14",
    # 2023
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26","2023-09-20","2023-11-01","2023-12-13",
    # 2024
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31","2024-09-18","2024-11-07","2024-12-18",
    # 2025
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30","2025-09-17","2025-10-29","2025-12-10",
    # 2026 (estimated dates per published preliminary calendar)
    "2026-01-28","2026-03-18","2026-04-29","2026-06-17","2026-07-29","2026-09-16","2026-10-28","2026-12-09",
]
FOMC_TS = pd.DatetimeIndex(pd.to_datetime(FOMC_DATES))


def _build_fomc_signal(idx: pd.DatetimeIndex) -> pd.Series:
    """+1 on (t-1, t) for each FOMC meeting; 0 elsewhere."""
    sig = pd.Series(0.0, index=idx)
    pos_lookup = {ts: i for i, ts in enumerate(idx)}
    for ts in FOMC_TS:
        if ts in pos_lookup:
            i = pos_lookup[ts]
            sig.iloc[i] = 1.0
            if i - 1 >= 0:
                sig.iloc[i - 1] = 1.0
        else:
            # FOMC may be a non-trading day; find the next trading day
            j = idx.searchsorted(ts)
            if j < len(idx):
                # j is the next trading day on/after ts -- use j and j-1
                sig.iloc[j] = 1.0
                if j - 1 >= 0:
                    sig.iloc[j - 1] = 1.0
    return sig


def run_strategy():
    paths    = get_project_paths(_HERE)
    mapping  = load_mapping(paths["mapping"])
    fx_daily = load_fx_rates(paths["panama"])

    eq_universe = mapping.index[mapping["asset_class"] == "Equity"]
    print(f"FOMC drift universe ({len(eq_universe)}): {list(eq_universe)}")

    inst_signals = {}
    for instrument in tqdm(mapping.index, desc=STRATEGY_NAME):
        data = load_instrument_data(instrument, paths["stats"], paths["panama"])
        if data is None:
            continue
        dates = data["daily_dates"]
        if len(dates) < OOS_START + 5:
            continue
        ccy = mapping.loc[instrument, "currency"]
        fx  = align_fx(ccy, dates, fx_daily)
        if fx is None:
            continue

        close         = data["close"]
        price_changes = data["price_changes"]
        vol           = blended_volatility(price_changes)
        idx           = pd.DatetimeIndex(dates)

        # -- SIGNAL --
        if instrument in eq_universe:
            raw = _build_fomc_signal(idx)
            forecast_scaled = cap_forecast(raw * rolling_forecast_scalar(raw))
        else:
            forecast_scaled = pd.Series(0.0, index=idx)

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
