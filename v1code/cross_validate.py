# -*- coding: utf-8 -*-

"""
cross_validate.py — Contiguous k-fold / sample-split / bootstrap CV for SDF estimators

Purpose
-------
Provides fast, contiguous-block cross-validation for SDF estimators (e.g., Ridge L2, Elastic-Net).
Per fold, builds TRAIN/TEST moments (mean & cov), calls the estimator on TRAIN, evaluates
the requested objective on both TRAIN and TEST, and aggregates mean + s.e. across folds.
Behavior mirrors the KNS (2020) Matlab flow while being tuple-safe for estimator outputs.

Modes
-----
  - "CV"       : k-fold with contiguous time blocks.
  - "ssplit"   : single split at a calendar date (PRE / OOS).
  - "bootstrap": out-of-bag evaluation with replacement.

I/O
---
cross_validate(estimator_func, dates, returns, params)
  -> (obj_stats, params, obj_folds)
obj_stats : [IS_mean, OOS_mean, IS_se, OOS_se]
params    : updated dict (fold caches, optional TEST returns)
obj_folds : per-fold [IS_obj, OOS_obj]

Notes
-----
• Supports a "cache_run" pass to precompute fold moments only.
• No label-based reindexing—inputs should already be aligned.
"""

from typing import Callable, Tuple, Dict, Union
import numpy as np
import pandas as pd
import warnings


def cross_validate(estimator_func: Callable, dates: np.ndarray,
                   returns: Union[pd.DataFrame, np.ndarray],
                   params: Dict) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """Entry point selecting CV mode."""
    params.update({'dd': dates, 'ret': returns, 'fun': estimator_func})
    mode = params.get('method', 'CV')
    if   mode == 'CV':        return _cv(params)
    elif mode == 'ssplit':    return _ssplit(params)
    elif mode == 'bootstrap': return _bootstrap(params)
    else: raise ValueError(f"Unknown validation method: {mode}")


def _cv(params: Dict) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """k-fold CV with contiguous blocks."""
    from cvpartition_contiguous import cvpartition_contiguous
    k, ret = params.get('kfold', 2), params['ret']
    n = len(ret) if isinstance(ret, pd.DataFrame) else ret.shape[0]
    parts = cvpartition_contiguous(n, k)
    obj = np.zeros((k, 2))
    params.setdefault('cv_cache', {}); params.setdefault('cv_MVE', {}); params.setdefault('cv_phi', {})
    # only create cv_idx_test if caller provided it previously (to preserve numerics/side-effects)
    has_idx_store = 'cv_idx_test' in params
    if has_idx_store and not isinstance(params['cv_idx_test'], dict):
        params['cv_idx_test'] = {}

    for i, test_idx in enumerate(parts):
        if has_idx_store:
            params['cv_idx_test'][i] = test_idx
        params['cv_iteration'] = i
        obj[i, :], params = _eval_fold(test_idx, params)

    means, ses = np.mean(obj, axis=0), np.std(obj, axis=0) / np.sqrt(k)
    return np.concatenate([means, ses]), params, obj


def _ssplit(params: Dict) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """Single calendar-date split (PRE / OOS)."""
    from datetime import datetime
    dd = pd.to_datetime(params['dd'])
    splitdate = params.get('splitdate', '01JAN2000')
    try:
        split_dt = datetime.strptime(splitdate, '%d%b%Y')
    except ValueError:
        try:
            split_dt = datetime.strptime(splitdate, '%Y-%m-%d')
        except ValueError:
            split_dt = datetime(2000, 1, 1)
            warnings.warn(f"Could not parse splitdate '{splitdate}', using 2000-01-01")
    test_idx = np.where(dd >= split_dt)[0].tolist()
    if len(test_idx) == 0:
        warnings.warn("No test data after split date")
    obj, params = _eval_fold(test_idx, params)
    return np.concatenate([obj, [0, 0]]), params, np.array([obj])


def _bootstrap(params: Dict) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """Out-of-bag bootstrap evaluation."""
    ret = params['ret']
    n = len(ret) if isinstance(ret, pd.DataFrame) else ret.shape[0]
    niter, min_test = params.get('niter', 100), params.get('min_test_size', 10)
    if params.get('random_seed') is not None:
        np.random.seed(params['random_seed'])
    boot = []
    for b in range(niter):
        boot_idx = np.random.choice(n, size=n, replace=True)
        oob = np.setdiff1d(np.arange(n), np.unique(boot_idx)).tolist()
        if len(oob) >= min_test:
            params['cv_iteration'] = b
            obj, params = _eval_fold(oob, params)
            if not np.any(np.isnan(obj)):
                boot.append(obj)
    if len(boot) == 0:
        return np.full(4, np.nan), params, np.array([[np.nan, np.nan]])
    boot = np.array(boot)
    stats = np.concatenate([np.mean(boot, axis=0), np.std(boot, axis=0)])
    return stats, params, boot


def _eval_fold(idx_test: list, params: Dict) -> Tuple[np.ndarray, Dict]:
    """Core evaluation for one TEST set (tuple-safe for estimator returns)."""
    from regcov import regcov
    ret, fun = params['ret'], params['fun']
    objective, it = params.get('objective', 'SSE'), params.get('cv_iteration', 0)
    n = len(ret) if isinstance(ret, pd.DataFrame) else ret.shape[0]
    if len(idx_test) == 0:
        return [np.nan, np.nan], params

    # Train/Test split
    idx_train = np.setdiff1d(np.arange(n), idx_test)
    if isinstance(ret, pd.DataFrame):
        r_tr, r_te = ret.iloc[idx_train], ret.iloc[idx_test]
    else:
        r_tr, r_te = ret[idx_train, :], ret[idx_test, :]

    # Cache fold moments
    cache = params.setdefault('cv_cache', {})
    if it not in cache:
        X, Xt = regcov(r_tr), regcov(r_te)
        y  = (np.mean(r_tr, axis=0).values if hasattr(np.mean(r_tr, axis=0), 'values')
              else np.mean(r_tr, axis=0)).reshape(-1, 1)
        yt = (np.mean(r_te, axis=0).values if hasattr(np.mean(r_te, axis=0), 'values')
              else np.mean(r_te, axis=0)).reshape(-1, 1)
        cvd = {'X': X, 'y': y, 'X_test': Xt, 'y_test': yt}
        if objective in ['GLS', 'GLSR2', 'SRexpl']:
            cvd.update({'invX': np.linalg.pinv(X), 'invX_test': np.linalg.pinv(Xt)})
        cache[it] = cvd
    cvd = cache[it]

    if params.get('cache_run', False):  # precompute-only pass
        return [np.nan, np.nan], params

    # Run estimator; accept (phi), (phi, params), or (phi, _, se)
    res = fun(cvd['X'], cvd['y'], params)
    if isinstance(res, tuple):
        phi = res[0]
        if len(res) >= 2 and isinstance(res[1], dict):
            params = res[1]
    else:
        phi = res
    phi = np.asarray(phi).reshape(-1, 1)

    # Store fold coefficients & TEST portfolio returns
    params.setdefault('cv_phi', {})[it] = phi
    port_te = (r_te.values @ phi if isinstance(r_te, pd.DataFrame) else r_te @ phi).flatten()
    params.setdefault('cv_MVE', {})[it] = port_te

    # Predictions (+ optional scaling)
    fact, fact_t = cvd['X'] @ phi, cvd['X_test'] @ phi
    if params.get('ignore_scale', False):
        b     = float(np.linalg.lstsq(fact,   cvd['y'],      rcond=None)[0])
        b_tst = float(np.linalg.lstsq(fact_t, cvd['y_test'], rcond=None)[0])
    else:
        b = b_tst = 1.0

    # Evaluate objective IS/OOS
    obj_fun = _get_objective_function(objective)
    res = [
        obj_fun(fact * b,       cvd['y'],      cvd.get('invX'),     phi, r_tr, params),
        obj_fun(fact_t * b_tst, cvd['y_test'], cvd.get('invX_test'), phi, r_te, params)
    ]
    return res, params


# ---------- Objective mapping & formulas (match prior numerics) ----------

def _get_objective_function(name: str) -> Callable:
    mapper = {
        'SSE': _obj_sse,
        'GLS': _obj_hj_distance,
        'CSR2': _obj_cross_sectional_r2,
        'GLSR2': _obj_gls_r2,
        'SRexpl': _obj_sr_explained,
        'SR': _obj_sharpe_ratio,
        'MVU': _obj_mean_variance_utility
    }
    return mapper.get(name, _obj_sse)


def _obj_cross_sectional_r2(y_hat, y, invX, phi, r, params):
    """Uncentered cross-sectional R² (as in original code)."""
    y, y_hat = y.flatten(), y_hat.flatten()
    denom = np.sum(y**2)
    return 1 - np.sum((y_hat - y) ** 2) / denom if denom > 0 else 0.0


def _obj_hj_distance(y_hat, y, invX, phi, r, params):
    """Hansen–Jagannathan distance (square-rooted, annualized)."""
    e = y_hat - y
    return np.sqrt((e.T @ invX @ e).item() * params.get('freq', 252))


def _obj_gls_r2(y_hat, y, invX, phi, r, params):
    """GLS R²."""
    e = y_hat - y
    num = (e.T @ invX @ e).item()
    den = (y.T @ invX @ y).item()
    return 1 - num / den if den > 0 else 0.0


def _obj_sr_explained(y_hat, y, invX, phi, r, params):
    """Explained squared Sharpe ratio (annualized scale)."""
    e = y_hat - y
    f = params.get('freq', 252)
    return ((y.T @ invX @ y) - (e.T @ invX @ e)).item() * f


def _obj_sharpe_ratio(y_hat, y, invX, phi, r, params):
    """Sharpe ratio of the SDF portfolio formed in TEST set."""
    ret = (r.values @ phi if isinstance(r, pd.DataFrame) else r @ phi).flatten()
    mu, sd = np.mean(ret), np.std(ret, ddof=0)
    return (np.sqrt(params.get('freq', 252)) * mu / sd) if sd > 0 else 0.0


def _obj_sse(y_hat, y, invX, phi, r, params):
    """Root mean squared pricing error (annualized)."""
    e = y_hat - y
    f = params.get('freq', 252)
    return np.sqrt(f * (e.T @ e).item() / len(y))


def _obj_mean_variance_utility(y_hat, y, invX, phi, r, params):
    """Mean-variance utility (annualized)."""
    f = params.get('freq', 252)
    if r is not None:
        ret = (r.values @ phi if isinstance(r, pd.DataFrame) else r @ phi).flatten()
        pret = 1 + ret
        if np.any(pret <= 0): return -1e9
        logret = np.log(pret)
        return f * (np.mean(logret) - 0.5 * np.var(logret, ddof=0))
    mean = (y @ phi).item()
    var  = (phi.T @ invX @ phi).item() if invX is not None else 0.0
    return f * (mean - 0.5 * var)
