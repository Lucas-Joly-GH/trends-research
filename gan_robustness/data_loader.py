"""
data_loader.py — Load Panama parquet files and the S180 checkpoint.

Returns are computed as Close.diff() (price differences), NOT log-returns.
Reason: ZN has 1,342 negative Panama-adjusted close prices, making log-returns
undefined. Price differences are the physically correct representation for
Panama back-adjusted futures data.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Panama data
# ---------------------------------------------------------------------------

def load_panama_returns(
    instruments: list[str],
    panama_dir: Path,
) -> pd.DataFrame:
    """
    Load Close price differences for each instrument from Panama parquet files.

    Returns a DataFrame with DatetimeIndex and one column per instrument
    (column name = instrument code).  Instruments not found emit a warning
    and are omitted from the result.

    Price differences are used universally (not log-returns) because Panama
    back-adjustment produces negative prices on instruments like ZN.
    """
    series = {}
    for inst in instruments:
        fpath = panama_dir / f"{inst}_continuous_panama.parquet"
        if not fpath.exists():
            warnings.warn(f"Parquet file not found, skipping: {fpath}", stacklevel=2)
            continue
        df = pd.read_parquet(fpath)
        # Normalise column names — parquet may store as 'Close' or 'close'
        df.columns = [c.capitalize() for c in df.columns]
        if "Close" not in df.columns:
            warnings.warn(f"{inst}: no 'Close' column, skipping", stacklevel=2)
            continue
        close = df["Close"].sort_index()
        # Sanity check: warn on zero prices (not negative — Panama prices CAN be negative)
        n_zero = (close == 0).sum()
        if n_zero > 0:
            warnings.warn(f"{inst}: {n_zero} zero close prices found", stacklevel=2)
        price_diff = close.diff()
        series[inst] = price_diff

    if not series:
        raise ValueError("No instruments loaded — check PANAMA_DIR and instrument list.")

    result = pd.DataFrame(series)
    result.index = pd.to_datetime(result.index)
    result.index.name = "Date"
    return result


def find_common_dates(
    returns_df: pd.DataFrame,
    checkpoint_dates: pd.DatetimeIndex,
    require_all_instruments: bool = True,
) -> pd.DatetimeIndex:
    """
    Returns the intersection of checkpoint dates and the date range where ALL
    instruments have valid (non-NaN) price differences.

    If require_all_instruments=False, dates where at least one instrument has
    data are kept (NaN columns filled downstream).

    The result is sorted ascending.
    """
    if require_all_instruments:
        valid_mask = returns_df.notna().all(axis=1)
    else:
        valid_mask = returns_df.notna().any(axis=1)

    valid_dates = returns_df.index[valid_mask]
    common = valid_dates.intersection(checkpoint_dates)
    return common.sort_values()


def build_return_matrix(
    returns_df: pd.DataFrame,
    common_dates: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Returns float64 array of shape (T, D) aligned to common_dates.
    Remaining NaN values (instruments that have gaps within the window) are
    filled with 0.0 with a warning.
    """
    aligned = returns_df.reindex(common_dates)
    n_nan = aligned.isna().sum().sum()
    if n_nan > 0:
        warnings.warn(
            f"build_return_matrix: filling {n_nan} NaN cells with 0.0",
            stacklevel=2,
        )
        aligned = aligned.fillna(0.0)
    return aligned.to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(pkl_path: Path) -> dict:
    """
    Load the S180 strategy checkpoint pickle.

    Returns the dict with at minimum these keys:
      dates        — pd.DatetimeIndex (9,341 entries)
      pnl          — np.ndarray, daily portfolio PnL in USD
      equity       — np.ndarray, daily NAV starting at $100M
      daily_ret    — np.ndarray, daily geometric returns
      per_inst_pnl — pd.DataFrame (9341 × 62), daily PnL per instrument in USD
      sr, cagr, max_dd, calmar — float
    """
    with open(pkl_path, "rb") as f:
        ckpt = pickle.load(f)

    # Reconstruct per_inst_pnl from numpy arrays (Colab-compatible checkpoint)
    if "per_inst_pnl" not in ckpt and "per_inst_pnl_values" in ckpt:
        ckpt["per_inst_pnl"] = pd.DataFrame(
            ckpt["per_inst_pnl_values"],
            columns=ckpt["per_inst_pnl_columns"],
            index=pd.DatetimeIndex(ckpt["per_inst_pnl_index"]),
        )
        del ckpt["per_inst_pnl_values"]
        del ckpt["per_inst_pnl_columns"]
        del ckpt["per_inst_pnl_index"]

    required_keys = {"dates", "equity", "daily_ret", "per_inst_pnl", "sr"}
    missing = required_keys - set(ckpt.keys())
    if missing:
        raise KeyError(f"Checkpoint missing keys: {missing}")

    # Normalise dates to DatetimeIndex
    if not isinstance(ckpt["dates"], pd.DatetimeIndex):
        ckpt["dates"] = pd.DatetimeIndex(ckpt["dates"])

    return ckpt


# ---------------------------------------------------------------------------
# All-instrument returns (for copula fitting)
# ---------------------------------------------------------------------------

def load_all_instrument_returns(
    checkpoint: dict,
    panama_dir: Path,
) -> pd.DataFrame:
    """
    Load price differences for ALL instruments present in per_inst_pnl columns.
    Aligned to the checkpoint date range (uses find_common_dates per instrument,
    then assembles a full DataFrame indexed by checkpoint dates).

    Instruments not found in panama_dir are skipped with a warning; their
    columns in the result are filled with 0.0.
    """
    instruments = list(checkpoint["per_inst_pnl"].columns)
    checkpoint_dates = checkpoint["dates"]

    series = {}
    for inst in instruments:
        fpath = panama_dir / f"{inst}_continuous_panama.parquet"
        if not fpath.exists():
            warnings.warn(
                f"All-instrument load: {inst} parquet not found, using 0",
                stacklevel=2,
            )
            series[inst] = pd.Series(0.0, index=checkpoint_dates, dtype=np.float64)
            continue
        df = pd.read_parquet(fpath)
        df.columns = [c.capitalize() for c in df.columns]
        if "Close" not in df.columns:
            warnings.warn(f"All-instrument load: {inst} has no Close, using 0", stacklevel=2)
            series[inst] = pd.Series(0.0, index=checkpoint_dates, dtype=np.float64)
            continue
        close = df["Close"].sort_index()
        price_diff = close.diff()
        price_diff.index = pd.to_datetime(price_diff.index)
        # Reindex to checkpoint dates; NaN -> 0 (instrument not yet launched or data gap)
        reindexed = price_diff.reindex(checkpoint_dates).fillna(0.0)
        series[inst] = reindexed

    result = pd.DataFrame(series, index=checkpoint_dates)
    return result


# ---------------------------------------------------------------------------
# Validity mask (identifies real vs. zero-filled data per instrument)
# ---------------------------------------------------------------------------

def compute_validity_mask(returns_df: pd.DataFrame) -> np.ndarray:
    """
    Return a boolean array of shape (T, D) where True means the instrument
    has real (non-artificial) data at that date.

    For each instrument, finds the first non-zero observation and marks
    everything from that row onward as valid.  Instruments that are entirely
    zero (parquet missing or no data) get an all-False column.

    This prevents the preprocessor from treating NaN->0 fills for
    pre-launch instruments as real data points, which would distort
    rank distributions and normal-score transforms.
    """
    arr = returns_df.to_numpy(dtype=np.float64)
    T, D = arr.shape
    mask = np.zeros((T, D), dtype=bool)

    for d in range(D):
        col = arr[:, d]
        nonzero_idx = np.nonzero(col)[0]
        if len(nonzero_idx) > 0:
            first_valid = nonzero_idx[0]
            mask[first_valid:, d] = True

    valid_counts = mask.sum(axis=0)
    instruments = list(returns_df.columns)
    low_count = [(instruments[d], int(valid_counts[d]))
                 for d in range(D) if valid_counts[d] < 1000]

    print(f"  Validity mask: min valid rows = {valid_counts.min()}, "
          f"max = {valid_counts.max()}")
    if low_count:
        print(f"  Instruments with <1000 valid rows: "
              f"{', '.join(f'{name}({n})' for name, n in low_count)}")

    return mask
