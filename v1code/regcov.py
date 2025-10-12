# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def regcov(r):
    """
    Compute the regularized covariance matrix of r.
    
    Parameters:
    - r: Input data matrix (T x N) - observations x variables
    
    Returns:
    - X: Regularized covariance matrix (N x N)
    """
    # Handle pandas DataFrame
    if isinstance(r, pd.DataFrame):
        r = r.values
    
    # Compute covariance matrix
    X = np.cov(r, rowvar=False)  # rowvar=False means columns are variables
    
    # Handle single variable case
    if np.ndim(X) == 0:
        X = np.array([[X]])
    
    # Covariance regularization (with flat Wishart prior)
    T, n = r.shape
    a = n / (n + T)
    X = a * np.trace(X) / n * np.eye(n) + (1 - a) * X
    
    return X