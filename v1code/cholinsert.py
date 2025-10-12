# -*- coding: utf-8 -*-

"""
cholinsert.py - Fast update of Cholesky factorization
Based on MATLAB implementation from Kozak, Nagel, and Santosh (2020)
"""

import numpy as np
from scipy.linalg import solve_triangular


def cholinsert(R, x, X, delta=0):
    """
    Fast update of Cholesky factorization of X'*X.
    
    Returns the Cholesky factorization of [X x]'*[X x] given a
    Cholesky factorization R of X'*X.
    
    Parameters
    ----------
    R : array_like
        Current upper triangular matrix to be updated from Cholesky 
        factorization R = chol(X'*X). Can be empty array.
    x : array_like
        Column vector representing the variable to be added
    X : array_like
        Data matrix containing the currently active variables (not including x)
    delta : float, optional
        Regularization parameter for ridge regression. Default is 0.
        If delta > 0, returns R = chol([X x]'*[X x] + delta*I)
        
    Returns
    -------
    R : array
        Updated Cholesky factorization with variable x added
        
    Notes
    -----
    Uses efficient rank-1 update of Cholesky factorization.
    """
    # Ensure x is a column vector
    x = np.asarray(x).flatten()
    
    # Diagonal element k in X'X (or X'X + delta*I) matrix
    diag_k = np.dot(x, x) + delta
    
    if R is None or R.size == 0:
        # Return resulting 1x1 matrix (scalar in array form)
        R = np.array([[np.sqrt(diag_k)]])
    else:
        # Ensure R is 2D
        R = np.atleast_2d(R)
        
        # Elements of column k in X'X matrix
        col_k = X.T @ x  # x'*X in MATLAB notation
        
        # Solve R'*R_k = col_k for R_k using forward substitution
        # In MATLAB: R_k = R'\col_k'
        # Since R is upper triangular, R' is lower triangular
        R_k = solve_triangular(R, col_k, trans='T', lower=False)
        
        # Find last element by exclusion
        # norm(x'x) = norm(R'*R), so R_kk^2 = x'x - R_k'*R_k
        R_kk = np.sqrt(diag_k - np.dot(R_k, R_k))
        
        # Update R by adding new column and row
        n = R.shape[0]
        R_new = np.zeros((n + 1, n + 1))
        R_new[:n, :n] = R
        R_new[:n, n] = R_k
        R_new[n, n] = R_kk
        R = R_new
    
    return R


def cholinsert_manual(R, x, X, delta=0):
    """
    Alternative implementation using manual forward substitution.
    May be useful if scipy is not available.
    """
    # Ensure x is a column vector
    x = np.asarray(x).flatten()
    
    # Diagonal element
    diag_k = np.dot(x, x) + delta
    
    if R is None or R.size == 0:
        R = np.array([[np.sqrt(diag_k)]])
    else:
        R = np.atleast_2d(R)
        n = R.shape[0]
        
        # Column k elements
        col_k = X.T @ x
        
        # Forward substitution to solve R'*R_k = col_k
        R_k = np.zeros(n)
        for i in range(n):
            sum_val = col_k[i]
            for j in range(i):
                sum_val -= R[j, i] * R_k[j]
            R_k[i] = sum_val / R[i, i]
        
        # Last diagonal element
        R_kk = np.sqrt(diag_k - np.dot(R_k, R_k))
        
        # Build updated matrix
        R_new = np.zeros((n + 1, n + 1))
        R_new[:n, :n] = R
        R_new[:n, n] = R_k
        R_new[n, n] = R_kk
        R = R_new
    
    return R