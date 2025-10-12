# -*- coding: utf-8 -*-

"""
load_managed_portfolios.py — Kozak, Nagel & Santosh (2020) replication
----------------------------------------------------------------------
Streamlined loader for 50 anomaly portfolios.

Enhancements
------------
• Optional date filtering via t0/tN (consistent with load_ff25)
• trim_ratio (alias for drop_perc) retained for backward compatibility
• Identical behaviour for NaN cleaning and structure of outputs
"""

import pandas as pd

def load_managed_portfolios(
    filename,
    daily=True,
    trim_ratio=1,           # same as old drop_perc
    omit_prefixes=None,
    keeponly=None,
    t0=None,
    tN=None
):
    """
    Load 50 managed portfolios from CSV file.

    Parameters
    ----------
    filename : str or Path
        Path to CSV file.
    daily : bool
        True for daily data, False for monthly data.
    trim_ratio : float, optional
        Drop characteristics with > trim_ratio missing (default: 1, same as old drop_perc).
    omit_prefixes, keeponly : unused, kept for API compatibility.
    t0, tN : datetime, optional
        Sample start and end dates (inclusive).

    Returns
    -------
    dates : pd.Series
    re : pd.DataFrame
    mkt : pd.Series
    names : list
    """
    # Read CSV
    DATA = pd.read_csv(filename)

    # Identify and parse date column
    date_col = next((col for col in ['date', 'Date'] if col in DATA.columns), None)
    if not date_col:
        raise ValueError("No date column found in file.")

    # Flexible date parsing
    for fmt in ['%m/%d/%Y' if daily else '%m/%Y', '%Y-%m-%d', '%Y%m%d']:
        try:
            DATA['date'] = pd.to_datetime(DATA[date_col], format=fmt)
            break
        except Exception:
            continue
    else:
        DATA['date'] = pd.to_datetime(DATA[date_col])

    # === Apply date filtering if provided ===
    if t0 is not None:
        DATA = DATA[DATA['date'] >= pd.Timestamp(t0)]
    if tN is not None:
        DATA = DATA[DATA['date'] <= pd.Timestamp(tN)]

    # Market column (always rme)
    mkt_col = 'rme'

    # Portfolio columns (exclude market/date)
    portfolio_cols = [col for col in DATA.columns if col not in ['date', 'rme', 're_ew']]

    # Check completeness
    all_cols = ['date', mkt_col] + portfolio_cols
    complete_rows = DATA[all_cols].notna().all(axis=1)
    n_complete = complete_rows.sum()
    if n_complete < 0.75 * len(DATA):
        raise ValueError(
            f"More than 25% of observations need to be dropped! "
            f"({len(DATA) - n_complete} / {len(DATA)})"
        )

    # Filter complete rows
    DATA_clean = DATA.loc[complete_rows].reset_index(drop=True)

    # === Optional trimming / winsorization ===
    if 0 < trim_ratio < 1:
        # Trim columns with excessive missingness if desired (legacy meaning)
        missing_perc = DATA_clean[portfolio_cols].isna().mean()
        keep_cols = [c for c in portfolio_cols if missing_perc[c] <= trim_ratio]
        if len(keep_cols) < len(portfolio_cols):
            print(f"Dropped {len(portfolio_cols)-len(keep_cols)} portfolios with > {trim_ratio*100:.1f}% missing.")
            portfolio_cols = keep_cols

    # Return clean components
    dates = DATA_clean['date']
    re = DATA_clean[portfolio_cols]
    mkt = DATA_clean[mkt_col]
    names = portfolio_cols

    print(f"Loaded {len(dates)} observations from {dates.min().date()} to {dates.max().date()} "
          f"({len(names)} portfolios).")

    return dates, re, mkt, names
