# -*- coding: utf-8 -*-

"""
choldelete.py - Fast downdate of Cholesky factorization
Based on MATLAB implementation from Kozak, Nagel, and Santosh (2020)
"""

import numpy as np


def choldelete(R, j):
    """
    Fast downdate of Cholesky factorization of X'*X.
    
    Returns the Cholesky factorization of the Gram matrix X'*X
    where the jth column of X has been removed.
    
    Parameters
    ----------
    R : array_like
        Upper triangular matrix from Cholesky factorization R = chol(X'*X)
    j : int
        Index of column to remove (0-based indexing in Python)
        
    Returns
    -------
    R : array
        Updated Cholesky factorization with jth variable removed
    """
    # Make a copy to avoid modifying the input
    R = R.copy()
    
    # Remove column j
    R = np.delete(R, j, axis=1)
    
    # Get size after deletion
    n = R.shape[1]
    
    # Apply Givens rotations
    for k in range(j, n):
        # Compute Givens rotation to zero out R[k+1, k]
        a = R[k, k]
        b = R[k+1, k]
        
        if abs(b) < 1e-14:  # Already zero, skip
            continue
            
        # Compute rotation parameters (matching MATLAB's planerot)
        if abs(b) > abs(a):
            tau = -a / b
            s = 1.0 / np.sqrt(1.0 + tau**2)
            c = s * tau
        else:
            tau = -b / a  
            c = 1.0 / np.sqrt(1.0 + tau**2)
            s = c * tau
        
        # Apply rotation to columns k through n
        for i in range(k, n):
            temp1 = R[k, i]
            temp2 = R[k+1, i]
            R[k, i] = c * temp1 - s * temp2
            R[k+1, i] = s * temp1 + c * temp2
        
        # Zero out the element we eliminated
        R[k+1, k] = 0
    
    # Remove last row
    R = R[:-1, :]
    
    return R


def choldelete_alt(R, j):
    """
    Alternative implementation using manual Givens rotation calculation.
    This may be more numerically stable in some cases.
    """
    # Make a copy to avoid modifying the input
    R = R.copy()
    
    # Remove column j
    R = np.delete(R, j, axis=1)
    
    # Get size after deletion
    n = R.shape[1]
    
    # Apply Givens rotations
    for k in range(j, n):
        # Compute Givens rotation to zero out R[k+1, k]
        a = R[k, k]
        b = R[k+1, k]
        
        if abs(b) < 1e-14:  # Already zero, skip
            continue
            
        # Compute rotation
        if abs(a) < 1e-14:
            c = 0.0
            s = 1.0
        else:
            if abs(b) > abs(a):
                tau = a / b
                s = 1.0 / np.sqrt(1.0 + tau * tau)
                c = s * tau
            else:
                tau = b / a
                c = 1.0 / np.sqrt(1.0 + tau * tau)
                s = c * tau
        
        # Apply rotation to columns k through n
        for i in range(k, n):
            temp1 = R[k, i]
            temp2 = R[k+1, i]
            R[k, i] = c * temp1 - s * temp2
            R[k+1, i] = s * temp1 + c * temp2
    
    # Remove last row
    R = R[:-1, :]
    
    return R