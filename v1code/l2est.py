# -*- coding: utf-8 -*-

import numpy as np

def l2est(X, y, params, compute_errors=False):
    """
    L2 shrinkage (ridge regression) estimation.
    
    Parameters:
    - X: Covariance matrix (N x N)
    - y: Mean returns vector (N x 1) or (N,)
    - params: Dictionary containing 'L2pen' and 'T'
    - compute_errors: Boolean, whether to compute standard errors
    
    Returns:
    - b: Coefficient estimates (N,)
    - params: Input parameters (unchanged)
    - se: Standard errors (N,) or NaN array if compute_errors=False
    """
    l = params['L2pen']
    n = X.shape[0]
    
    # Ensure y is properly shaped
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    if compute_errors:
        # Compute inverse for standard errors
        try:
            Xinv = np.linalg.inv(X + l * np.eye(n))
            b = np.dot(Xinv, y)
            se = np.sqrt(1 / params['T'] * np.diag(Xinv)) 
            # simplified formula for se, ignores covariances, assumes Var(y) ≈ X/T, ignores ridge bias. etc
            # for proper inference would need to bootstrap, or at least compute: Var(b̂ ) = (X + λI)^(-1) · Var(y) · [(X + λI)^(-1)]' = (1/T) · (X + λI)^(-1) · X · (X + λI)^(-1) 
        except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
            Xinv = np.linalg.pinv(X + l * np.eye(n))
            b = np.dot(Xinv, y)
            se = np.sqrt(1 / params['T'] * np.diag(Xinv))
    else:
        # Solve a system of linear equations instead if errors are not needed
        try:
            b = np.linalg.solve(X + l * np.eye(n), y)
        except np.linalg.LinAlgError:
            b = np.linalg.pinv(X + l * np.eye(n)) @ y
        se = np.full(n, np.nan)
    
    # Ensure b is 1D array
    b = b.flatten()
    
    return b, params, se