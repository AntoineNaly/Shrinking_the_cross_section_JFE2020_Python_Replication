# -*- coding: utf-8 -*-
"""
load_managed_portfolios.py - Streamlined data loader for 50 anomaly portfolios
"""
import pandas as pd

def load_managed_portfolios(filename, daily, drop_perc=1, omit_prefixes=None, keeponly=None):
    """
    Load 50 managed portfolios from CSV file.
    
    Parameters:
    - filename: Path to CSV file
    - daily: Boolean for daily/monthly data
    - drop_perc: Drop characteristics with > drop_perc missing (default: 1)
    - omit_prefixes: List of prefixes to drop (unused, kept for compatibility)
    - keeponly: Keep only these characteristics (unused, kept for compatibility)
    
    Returns: dates, re (50 portfolios), mkt, names
    """
    
    # Read data
    DATA = pd.read_csv(filename)
    
    # Parse dates flexibly
    date_col = next((col for col in ['date', 'Date'] if col in DATA.columns), None)
    if not date_col:
        raise ValueError("No date column found")
    
    # Try multiple date formats
    for fmt in ['%m/%d/%Y' if daily else '%m/%Y', '%Y-%m-%d', '%Y%m%d']:
        try:
            DATA['date'] = pd.to_datetime(DATA[date_col], format=fmt)
            break
        except ValueError:
            continue
    else:
        DATA['date'] = pd.to_datetime(DATA[date_col])
    
    # Find market column BEFORE removing any columns
    mkt_col = 'rme'
    
    # Extract portfolio columns (50 anomalies + potentially re_ew)
    portfolio_cols = [col for col in DATA.columns if col not in ['date','rme','re_ew']]
    
    # Clean missing data using ALL relevant columns
    all_cols = ['date', mkt_col] + portfolio_cols
    complete_rows = DATA[all_cols].notna().all(axis=1)
    n_complete = complete_rows.sum()
    
    if n_complete < 0.75 * len(DATA):
        raise ValueError(f"More than 25% of observations need to be dropped! ({len(DATA) - n_complete} out of {len(DATA)})")
    
    # Filter to complete observations
    DATA_clean = DATA[complete_rows].reset_index(drop=True)
    
    # Extract final components
    dates = DATA_clean['date']
    re = DATA_clean[portfolio_cols]  # This includes re_ew if present
    mkt = DATA_clean[mkt_col]
    names = portfolio_cols

    return dates, re, mkt, names