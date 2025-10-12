# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def demarket(r, mkt, b=None):
    """
    Demarket function to compute market beta and de-market (i.e. market-adjusted) returns.
    
    Parameters:
    - r: DataFrame or 2D array of returns.
    - mkt: Series or 1D array of market returns.
    - b: Optional; market beta. If not provided, it will be computed.
    
    Returns:
    - rme: DataFrame or 2D array of de-marketed returns.
    - b: market beta.
    """
    # Handle pandas inputs
    if isinstance(r, pd.DataFrame):
        r_values = r.values
        was_df = True
        r_index = r.index
        r_columns = r.columns
    else:
        r_values = r
        was_df = False
    
    if isinstance(mkt, pd.Series):
        mkt_values = mkt.values
    else:
        mkt_values = mkt
    
    # If b (beta) is not provided, compute it
    if b is None:
        # Create design matrix: [ones, market_returns] for intercept + beta
        X = np.column_stack([np.ones(mkt_values.shape[0]), mkt_values.flatten()])
        
        # Solve normal equations: (X'X)^(-1)X'Y for each asset
        try:
            # Use lstsq for numerical stability
            coeffs = np.linalg.lstsq(X, r_values, rcond=None)[0]
            b = coeffs[1, :]  # Extract market betas (second row, skip intercept)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            coeffs = np.linalg.pinv(X) @ r_values  
            b = coeffs[1, :]
    else:
        # Use provided betas
        b = np.asarray(b)
        if b.ndim == 1:
            b = b.reshape(-1)
    
    # De-market: subtract market exposure from returns
    # rme = r - mkt * beta (broadcasting across assets)
    rme_values = r_values - np.outer(mkt_values.flatten(), b)
    
    # Return in original format
    if was_df:
        rme = pd.DataFrame(rme_values, index=r_index, columns=r_columns)
    else:
        rme = rme_values
    
    return rme, b