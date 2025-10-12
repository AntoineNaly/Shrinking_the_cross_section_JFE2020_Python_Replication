# -*- coding: utf-8 -*-

"""
elasticnet_sdf_HJdist.py - Elastic Net for SDF with Hansen-Jagannathan Distance
Exact Python translation of MATLAB version from Kozak, Nagel, and Santosh (2020)
"""

import numpy as np
from larsen import larsen
from typing import Tuple, Dict


def elasticnet_sdf_HJdist(X: np.ndarray, y: np.ndarray, params: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Elastic Net regularization for SDF estimation with Hansen-Jagannathan distance.
    
    This function implements elastic net (L1 + L2) regularization for estimating
    stochastic discount factor coefficients. It minimizes:
    
    (y - Xb)' W (y - Xb) + λ₂||b||₂² subject to ||b||₁ ≤ t
    
    where W is the weighting matrix (inverse covariance for HJ distance).
    
    The function uses the LARS-EN algorithm for efficient computation of the
    entire regularization path.
    """
    from parse_config import parse_config
    
    # Validate inputs and ensure proper dimensions
    if X.shape[0] != len(y):
        raise ValueError("X and y must have compatible dimensions")
    
    y = np.asarray(y).flatten()
    
    # Check for pre-computed path - early return for cached results
    if params.get('use_precomputed', False):
        return _retrieve_cached_path(params)
    
    # Setup default parameters - consolidated parameter setup
    default_params = {
        'cv_iteration': 1, 'L2pen': 0, 'stop': 0,
        'storepath': False, 'verbose': False
    }
    
    p = parse_config(params, default_params)
    
    # Validate L2 penalty
    if p['L2pen'] < 0:
        raise ValueError('L2 penalty must be non-negative')
    
    delta, cv_iter = p['L2pen'], p['cv_iteration']
    
    # Get or compute matrix transformations - consolidated cache handling
    X1, y1 = _get_or_compute_transformations(X, y, delta, cv_iter, params)
    
    # Quick exit for cache-only runs
    if params.get('cache_run', False):
        return None, params
    
    # Run LARS-EN algorithm on transformed problem
    bpath, steps = larsen(X1, y1, delta, p['stop'], None, p['storepath'], p['verbose'])
    
    # Store path for this CV iteration
    if 'bpath' not in params:
        params['bpath'] = {}
    params['bpath'][p['cv_iteration']] = bpath
    
    # Return final coefficients - handle both vector and matrix results
    b = bpath if bpath.ndim == 1 else bpath[:, -1]
    
    return b, params


def _retrieve_cached_path(params: Dict) -> Tuple[np.ndarray, Dict]:
    """Retrieve coefficients from pre-computed path."""
    cv_iter = params['cv_iteration']
    
    if 'bpath' not in params:
        raise ValueError("No precomputed path found")
    
    bpath = params['bpath'][cv_iter]
    
    # Find coefficients with desired sparsity level
    n_nonzero = np.sum(bpath != 0, axis=0)
    idx_array = np.where(n_nonzero >= abs(params['stop']))[0]
    
    # Select appropriate coefficients
    b = bpath[:, idx_array[0]] if len(idx_array) > 0 else bpath[:, -1]
    
    return b, params


def _get_or_compute_transformations(X: np.ndarray, y: np.ndarray, delta: float, 
                                   cv_iter: int, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Get cached or compute matrix transformations for elastic net."""
    # Check cache for pre-computed matrix decompositions
    cache_key = 'elasticnet_cache'
    if cache_key in params and cv_iter in params[cache_key]:
        cache = params[cache_key][cv_iter]
        return cache['X1'], cache['y1']
    
    # Compute matrix square root and its inverse
    # This transforms the problem to standard form
    Q, D, _ = np.linalg.svd(X)
    d = D
    
    # Tolerance for numerical stability
    tol = max(X.shape) * np.finfo(float).eps * np.linalg.norm(d, np.inf)
    r1 = np.sum(d > tol) + 1
    
    # Keep only well-conditioned components - consolidated matrix computation
    Q1 = Q[:, :r1-1]
    s = d[:r1-1]
    s2 = 1.0 / np.sqrt(s)
    
    # Transform variables
    X2 = Q @ np.diag(np.sqrt(d)) @ Q.T    # X^(1/2)
    X2inv = (Q1 * s2) @ Q1.T              # X^(-1/2)
    
    X1, y1 = X2, X2inv @ y
    
    # Cache for reuse - consolidated cache storage
    if cache_key not in params:
        params[cache_key] = {}
    
    params[cache_key][cv_iter] = {'X1': X1, 'y1': y1}
    
    return X1, y1
