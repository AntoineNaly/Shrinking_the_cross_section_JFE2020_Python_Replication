# -*- coding: utf-8 -*-

"""
regression_utils.py — Minimal, stable time-series regression helpers (OLS + HAC)

Purpose
-------
Provides small, dependency-light utilities to compute:
  • OLS regression with homoskedastic standard errors (ddof consistent with code).
  • Newey–West HAC variance with Bartlett kernel (lags = floor(4*(T/100)^(2/9)) if None).
  • Alpha helpers tailored for Table-4 style reporting:
      - `calculate_capm_alpha`: α from a zero benchmark (mean×freq).
      - `calculate_factor_alpha`: α from OLS of portfolio on one or more benchmarks, annualized.

Design / numerical notes
------------------------
  • Inputs are coerced to numpy; 1D X is auto-reshaped to (T,1).
  • Solves normal equations with `solve`, falling back to `pinv` on singularity.
  • HAC implementation mirrors a textbook sandwich estimator:
      S = Σ x_t e_t e_t x_t' + Σ_j w_j (Γ_j + Γ_j'), w_j Bartlett;
      V = (X'X)^{-1} S (X'X)^{-1}.
  • All loops/order are fixed to maintain bit-stable results where feasible.

Inputs / outputs
----------------
  • `time_series_regression(y, X, newey_west=False, lags=None)` returns dict with:
      alpha, alpha_se, alpha_tstat, beta, beta_se, r_squared, residuals, vcov.
  • `calculate_capm_alpha(returns, freq)` returns dict with alpha, se, tstat (annualized).
  • `calculate_factor_alpha(portfolio, benchmark, freq, ...)` aligns lengths, runs OLS/HAC,
     prints a one-line summary, and returns alpha, se, tstat, r_squared (annualized α, s.e.).

Intended use
------------
Keep regression logic local, transparent, and reproducible for Table-4 style
reporting without altering upstream estimation numerics.
"""

import numpy as np

def time_series_regression(y, X, newey_west=False, lags=None):
    y = np.asarray(y).ravel()
    X = np.asarray(X);  X = X.reshape(-1,1) if X.ndim==1 else X
    T = y.shape[0]
    if X.shape[0] != T: raise ValueError(f"Dimension mismatch: y={T}, X={X.shape[0]}")
    Xi = np.column_stack((np.ones(T), X)); K = Xi.shape[1]
    XtX = Xi.T @ Xi; Xty = Xi.T @ y
    try:    bhat = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError: bhat = np.linalg.pinv(XtX) @ Xty
    resid = y - Xi @ bhat
    tss = np.sum((y - y.mean())**2); rss = resid @ resid; r2 = 1 - (rss/tss) if tss>0 else 0
    if newey_west:
        if lags is None: lags = int(np.floor(4 * (T/100.0)**(2/9)))
        vcov = newey_west_vcov(Xi, resid, lags)
    else:
        sigma2 = rss / (T - K)
        vcov = sigma2 * np.linalg.pinv(XtX)
    se = np.sqrt(np.diag(vcov))
    a, a_se = bhat[0], se[0]
    return {
        'alpha': a, 'alpha_se': a_se, 'alpha_tstat': (a/a_se if a_se>0 else np.nan),
        'beta': (bhat[1:] if K>1 else np.array([])),
        'beta_se': (se[1:] if K>1 else np.array([])),
        'r_squared': r2, 'residuals': resid, 'vcov': vcov
    }

def newey_west_vcov(X, e, lags):
    """HAC (Bartlett) — identical loop/order to preserve numerics."""
    X = np.asarray(X); e = np.asarray(e).ravel()
    T,K = X.shape
    S = np.zeros((K,K))
    # contemporaneous
    for t in range(T):
        xe = (X[t:t+1,:].T * e[t]); S += xe @ xe.T
    # auto terms
    for j in range(1, lags+1):
        w = 1 - j/(lags+1)
        Gj = np.zeros((K,K))
        for t in range(j, T):
            xt  = (X[t:t+1,:].T   * e[t])
            xtj = (X[t-j:t-j+1,:].T * e[t-j])
            Gj += xt @ xtj.T
        S += w * (Gj + Gj.T)
    XTXi = np.linalg.pinv(X.T @ X)
    return XTXi @ S @ XTXi

def calculate_capm_alpha(pret, freq):
    pret = np.asarray(pret).ravel()
    a = pret.mean(); se = pret.std(ddof=1)/np.sqrt(len(pret))
    return {'alpha': a*freq, 'se': se*freq, 'tstat': (a/se if se>0 else np.nan)}

def calculate_factor_alpha(pret, bench, freq, benchmark_name="Factor", newey_west=False, lags=None):
    try:
        pret  = np.asarray(pret).ravel()
        bench = np.asarray(bench)
        n = min(len(pret), (bench.shape[0] if bench.ndim>1 else len(bench)))
        pret, bench = pret[:n], (bench[:n] if bench.ndim>1 else bench[:n])
        res = time_series_regression(pret, bench, newey_west=newey_west, lags=lags)
        a, se = res['alpha']*freq, res['alpha_se']*freq
        t = (a/se if se>0 else np.nan)
        print(f"  {benchmark_name} alpha: {a*100:,.2f}% (se: {se*100:,.2f}%)")
        print(f"    R-squared: {res['r_squared']:.3f}" + (f"\n    Newey-West lags: {lags}" if newey_west else ""))
        return {'alpha': a, 'se': se, 'tstat': t, 'r_squared': res['r_squared']}
    except Exception as e:
        print(f"  {benchmark_name} alpha calculation failed: {e}")
        return {'alpha': np.nan, 'se': np.nan, 'tstat': np.nan, 'r_squared': np.nan}
