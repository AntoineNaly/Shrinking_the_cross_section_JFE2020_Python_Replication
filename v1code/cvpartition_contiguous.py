# -*- coding: utf-8 -*-

def cvpartition_contiguous(n, k):
    """
    Create contiguous partitions for cross-validation.
    
    We use contiguous partitioning for financial data due to their high
    persistence. Using random sampling partitioning produces highly
    correlated samples.
    
    Parameters:
    - n: Number of observations
    - k: Number of folds
    
    Returns:
    - indices: List of k index arrays for each fold
    """
    s = n // k  # Size of each fold (except possibly the last)
    indices = []
    
    for i in range(k-1):
        indices.append(list(range(s*i, s*(i+1))))
    
    # Last fold gets remaining observations
    indices.append(list(range(s*(k-1), n)))
    
    return indices