# -*- coding: utf-8 -*-

"""
load_ff25.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_ff25(datapath, daily, t0=None, tN=None):
    """
    Load Fama-French 25 portfolios and factors.
    
    Parameters:
    - datapath: Path to data directory
    - daily: Boolean, True for daily data, False for monthly
    - t0: Start date (datetime or numeric)
    - tN: End date (datetime or numeric)
    
    Returns:
    - dates: Series of dates
    - ret: DataFrame of excess returns (portfolio returns - RF)
    - mkt: Series of market excess returns (Mkt-RF)
    - DATA: Full merged DataFrame
    - labels: List of portfolio names
    """
    # Set default parameters - consolidated parameter setup
    t0 = t0 or 0  # MATLAB uses 0 as default
    tN = tN or float('inf')
    
    # Determine file names based on frequency - consolidated file naming
    file_config = {
        True: {  # daily
            'factors': 'F-F_Research_Data_Factors_daily.csv',
            'portfolios': '25_Portfolios_5x5_Daily_average_value_weighted_returns_daily.csv'
        },
        False: {  # monthly
            'factors': 'F-F_Research_Data_Factors.csv',
            'portfolios': '25_Portfolios_5x5_average_value_weighted_returns_monthly.csv'
        }
    }
    
    files = file_config[daily]
    
    # Load and process factor data - consolidated factor loading
    DATA = _load_and_process_factors(datapath, files['factors'], t0, tN)
    
    # Load and process portfolio returns - consolidated portfolio loading
    RET = _load_and_process_portfolios(datapath, files['portfolios'])
    
    # Merge data and extract components - consolidated merging and extraction
    DATA = pd.merge(DATA, RET, on='Date', how='inner')
    dates, mkt, ret, labels = _extract_components(DATA)
    
    return dates, ret, mkt, DATA, labels


def _load_and_process_factors(datapath, filename, t0, tN):
    """Load and process factor data with date filtering - consolidated factor processing."""
    try:
        DATA = pd.read_csv(os.path.join(datapath, filename))
        
        # Parse dates - consolidated date parsing
        if 'Date' not in DATA.columns:
            raise ValueError("No Date column found in factor file")
        
        DATA['Date'] = _parse_ff_dates(DATA['Date'])
        
        # Apply date filtering - consolidated date filtering
        t0_date, tN_date = _convert_date_bounds(t0, tN)
        DATA = DATA[(DATA['Date'] >= t0_date) & (DATA['Date'] <= tN_date)]
        DATA = DATA.dropna(subset=['Date'])
        
        return DATA
        
    except Exception as e:
        raise Exception(f"Error reading factor file {filename}: {e}")


def _load_and_process_portfolios(datapath, filename):
    """Load and process portfolio returns - consolidated portfolio processing."""
    try:
        RET = pd.read_csv(os.path.join(datapath, filename))
        
        # Parse dates - consolidated date parsing
        if 'Date' not in RET.columns:
            raise ValueError("No Date column found in portfolio file")
        
        RET['Date'] = _parse_ff_dates(RET['Date'])
        
        return RET
        
    except Exception as e:
        raise Exception(f"Error reading portfolio file {filename}: {e}")


def _parse_ff_dates(date_series):
    """Parse Fama-French dates with fallback formats - consolidated date parsing."""
    # FF data typically has dates as integers (YYYYMMDD)
    try:
        return pd.to_datetime(date_series.astype(str), format='%Y%m%d')
    except (ValueError, TypeError):
        # Fallback to pandas inference
        return pd.to_datetime(date_series)


def _convert_date_bounds(t0, tN):
    """Convert date bounds to pandas timestamps - consolidated date conversion."""
    def convert_single_date(date_val):
        if isinstance(date_val, (int, float)):
            if date_val > 0 and not np.isinf(date_val):
                # Convert numeric to timestamp (assuming days since epoch)
                return pd.Timestamp('1900-01-01') + pd.Timedelta(days=date_val)
            else:
                return pd.Timestamp.min if date_val <= 0 else pd.Timestamp.max
        elif isinstance(date_val, datetime):
            return pd.Timestamp(date_val)
        else:
            return pd.Timestamp.min
    
    t0_date = convert_single_date(t0)
    
    # Handle tN separately to avoid np.isinf issues with datetime objects
    if isinstance(tN, (int, float)):
        tN_date = convert_single_date(tN) if not np.isinf(tN) else pd.Timestamp.max
    else:
        tN_date = convert_single_date(tN)
    
    return t0_date, tN_date


def _extract_components(DATA):
    """Extract dates, market returns, portfolio returns, and labels - consolidated extraction."""
    # Extract basic components
    dates = DATA['Date']
    mkt = DATA['Mkt-RF'] / 100  # Convert from percentage (already excess return)
    
    # Identify portfolio columns - consolidated portfolio identification
    portfolio_cols = _identify_portfolio_columns(DATA)
    
    # Ensure exactly 25 portfolios - consolidated portfolio validation
    portfolio_cols = _validate_portfolio_count(portfolio_cols)
    
    # Calculate excess returns: portfolio returns - risk free rate - consolidated excess return calculation
    ret = DATA[portfolio_cols] / 100 - DATA['RF'].values.reshape(-1, 1) / 100
    labels = portfolio_cols
    
    return dates, mkt, ret, labels


def _identify_portfolio_columns(DATA):
    """Identify portfolio columns using pattern matching - consolidated identification."""
    # Standard factor columns to exclude
    factor_cols = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF', 'Mom', 'RMW', 'CMA']
    
    # Look for columns matching FF 25 portfolio patterns
    portfolio_cols = []
    for col in DATA.columns:
        if col not in factor_cols:
            # Check for size-value portfolio patterns
            portfolio_patterns = ['BM', 'ME', 'SMALL', 'BIG', 'Lo', 'Hi']
            if any(pattern in col for pattern in portfolio_patterns):
                portfolio_cols.append(col)
    
    # Fallback: take remaining columns after excluding factors
    if not portfolio_cols:
        portfolio_cols = [col for col in DATA.columns if col not in factor_cols]
    
    return portfolio_cols


def _validate_portfolio_count(portfolio_cols):
    """Validate and adjust portfolio count to 25 - consolidated validation."""
    if len(portfolio_cols) != 25:
        print(f"Warning: Expected 25 portfolios, found {len(portfolio_cols)}")
        # Take the first 25 columns
        portfolio_cols = portfolio_cols[:25]
    
    return portfolio_cols