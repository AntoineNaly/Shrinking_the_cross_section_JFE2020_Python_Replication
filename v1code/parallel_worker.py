# -*- coding: utf-8 -*-

"""
parallel_worker.py - Separate module for parallel processing functions

This module contains functions that need to be parallelized using joblib.
Keeping them in a separate module avoids serialization issues.
"""

import numpy as np
from cross_validate import cross_validate
from elasticnet_sdf_HJdist import elasticnet_sdf_HJdist


def process_L2_penalty(i, lCV_val, p_cache_in, L1range, L1rn, dd, r_train):
    """
    Process elastic net estimation for one L2 penalty value.
    
    This function handles the L1 path for a fixed L2 penalty.
    It uses the LARS-EN algorithm to efficiently compute solutions
    for all L1 penalty values.
    """
    # Deep copy and initialize parameters - consolidated initialization
    params = p_cache_in.copy()
    params.update({
        'bpath': {},
        'storepath': True,
        'L2pen': lCV_val
    })
    
    # Initialize storage - consolidated storage allocation
    storage_shape = (L1rn,)
    cv = np.full((*storage_shape, 4), np.nan)
    phis = {}
    cv_folds_i = np.full((*storage_shape, p_cache_in['kfold']), np.nan)
    
    # Process L1 grid backwards for LARS algorithm efficiency
    # This allows the LARS algorithm to reuse computations
    first_run = True
    for j in range(L1rn - 1, -1, -1):
        # Update parameters for current iteration - consolidated parameter update
        params.update({
            'stop': -L1range[j],
            'use_precomputed': not first_run
        })
        first_run = False
        
        # Run cross-validation for this L1-L2 combination
        cv_result, params, cv_folds_ = cross_validate(
            elasticnet_sdf_HJdist,
            dd.values if hasattr(dd, 'values') else dd,
            r_train,
            params
        )
        
        # Store results - consolidated result storage
        cv[j, :] = cv_result
        cv_folds_i[j, :] = cv_folds_[:, 1]  # OOS objective
        if 'cv_phi' in params:
            phis[j] = params['cv_phi'].copy()
    
    # Convert dict to list in correct order - consolidated output formatting
    phis_list = [phis.get(j, None) for j in range(L1rn)]
    
    return cv[:, 1], cv[:, 3], phis_list, cv_folds_i