# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

def load_ff_anomalies(datapath, daily, t0, tN):
    """
    Load Fama-French anomaly factors (beyond basic 3-factor model).
    This includes momentum, short-term reversal, long-term reversal, etc.
    
    Parameters:
    - datapath: Path to data directory
    - daily: Boolean for daily/monthly frequency
    - t0: Start date
    - tN: End date
    
    Returns:
    - dates: Series of dates
    - returns: DataFrame of factor returns
    - DATA: Full DataFrame
    """
    
    # Define factor files based on frequency
    if daily:
        factor_files = {
            '5_factors': 'F-F_Research_Data_5_Factors_2x3_daily.csv',
            'momentum': 'F-F_Momentum_Factor_daily.csv',
            'st_reversal': 'F-F_ST_Reversal_Factor_daily.csv',
            'lt_reversal': 'F-F_LT_Reversal_Factor_daily.csv'
        }
    else:
        factor_files = {
            '5_factors': 'F-F_Research_Data_5_Factors_2x3.csv',
            'momentum': 'F-F_Momentum_Factor.csv', 
            'st_reversal': 'F-F_ST_Reversal_Factor.csv',
            'lt_reversal': 'F-F_LT_Reversal_Factor.csv'
        }
    
    all_factors = None
    
    # Load each factor file
    for name, filename in factor_files.items():
        filepath = os.path.join(datapath, filename)
        
        if os.path.exists(filepath):
            try:
                # Read CSV
                df = pd.read_csv(filepath)
                
                # Handle date column
                if 'Date' in df.columns:
                    try:
                        df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d')
                    except:
                        df['Date'] = pd.to_datetime(df['Date'])
                else:
                    # Find date column
                    date_cols = [col for col in df.columns if 'date' in col.lower()]
                    if date_cols:
                        df['Date'] = pd.to_datetime(df[date_cols[0]].astype(str), format='%Y%m%d')
                        df = df.drop(columns=date_cols[0])
                
                # Filter by date range
                if isinstance(t0, pd.Timestamp) or isinstance(t0, pd.datetime):
                    df = df[(df['Date'] >= t0) & (df['Date'] <= tN)]
                
                # Merge with existing factors
                if all_factors is None:
                    all_factors = df
                else:
                    # Avoid duplicate columns when merging
                    merge_cols = [col for col in df.columns if col not in all_factors.columns or col == 'Date']
                    all_factors = pd.merge(all_factors, df[merge_cols], on='Date', how='outer')
                    
            except Exception as e:
                print(f"Warning: Could not load {name} from {filename}: {e}")
    
    if all_factors is None:
        # Return empty data if no factors loaded
        print("Warning: No FF anomaly factors could be loaded")
        return pd.Series(dtype='datetime64[ns]'), pd.DataFrame(), pd.DataFrame()
    
    # Sort by date and clean
    all_factors = all_factors.sort_values('Date').dropna(subset=['Date'])
    
    # Extract components
    dates = all_factors['Date']
    
    # Get factor columns (exclude Date and RF)
    factor_cols = [col for col in all_factors.columns if col not in ['Date', 'RF']]
    returns = all_factors[factor_cols] / 100  # Convert from percentage
    
    return dates, returns, all_factors