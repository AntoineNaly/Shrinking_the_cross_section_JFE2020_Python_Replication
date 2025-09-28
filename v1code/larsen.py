# -*- coding: utf-8 -*-

"""
larsen.py — Compact LARS-EN path solver (Elastic-Net) with Cholesky updates

What this implements
--------------------
A numerically faithful variant of the LARS algorithm adapted to Elastic-Net:
  • Maintains an active set, updates coefficients along piecewise-linear paths.
  • Supports either Gram-matrix mode or Cholesky rank-one insert/delete on X.
  • Exact interpolation at an ℓ₁-norm stopping point to hit a target sparsity/penalty.

Key features
------------
  • Elastic-Net scaling: when `delta>0`, the effective ℓ₁ path is rescaled so that
    `stop` refers to the EN-scaled path (matches upstream usage).
  • Two stopping modes:
      - `stop > 0` : stop when ||b||₁ reaches `stop` (with interpolation).
      - `stop < 0` : stop after selecting `-stop` variables (cardinality stop).
  • Drop events: detects sign changes and removes variables via Cholesky downdates.
  • Storage: if `storepath=True`, returns full path `b` (p × steps); otherwise only
    the current coefficients vector is maintained.

Inputs / outputs
----------------
  larsen(X, y, delta, stop=0, Gram=None, storepath=True, verbose=False) →
      (b, steps)
    • X : (n×p) returns matrix; y : length-n target.
    • delta : EN mixing (>0 implies EN; 0 ⇒ LASSO path).
    • stop : ℓ₁ bound (>0) or cardinality (negative).
    • Gram : optional precomputed X'X; if None, uses Cholesky factor updates.
    • b : coefficient path (p×steps if storepath; else p×1 current vector).
    • steps : number of path steps computed.

Numerical behavior
------------------
  • Update order and algebra mirror the reference Matlab implementation used in KNS.
  • Fallbacks to least-squares/`pinv` only on singular systems, preserving path shape.
  • Warnings on forced exit (max steps) to flag potential degeneracy.

Intended use
------------
Used internally by dual-penalty estimation to trace EN paths efficiently while
keeping results consistent with the paper’s Matlab code.
"""

import warnings, numpy as np
from typing import Tuple, Optional, Union
from scipy.linalg import solve_triangular

def larsen(X: np.ndarray, y: np.ndarray, delta: float, stop: Union[int,float]=0,
           Gram: Optional[np.ndarray]=None, storepath: bool=True, verbose: bool=False) -> Tuple[np.ndarray,int]:
    from cholinsert import cholinsert
    from choldelete import choldelete
    n,p = X.shape; y = np.asarray(y).ravel()
    maxVars, maxSteps = (min(n,p) if delta < np.finfo(float).eps else p), 8*min(n,p)
    b = (np.zeros((p,2*p)) if storepath else np.zeros(p)); b_prev = np.zeros(p)
    mu = np.zeros(n); A = []; I = list(range(p)); R = None; useG = Gram is not None
    if delta>0 and stop>0: stop = stop/(1+delta)
    lasso, done, step = False, False, 1
    if verbose: print('Step\tAdded\tDropped\tActive')
    while (len(A)<maxVars and not done and step<maxSteps):
        r = y - mu; c = X.T @ r
        if not I: break
        ci = c[I]; j_loc = int(np.argmax(np.abs(ci))); cmax = float(np.abs(ci[j_loc])); j = I[j_loc]
        if not lasso:
            if not useG:
                Xa = X[:,A] if A else np.empty((n,0))
                R = cholinsert(R, X[:,j], Xa, delta)
            A.append(j); I.pop(j_loc)
            if verbose: print(f'{step}\t{j}\t\t{len(A)}')
        else:
            lasso = False
        if not A: continue
        if useG:
            G = Gram[np.ix_(A,A)]; Xy = X[:,A].T @ y
            try: bOLS = np.linalg.solve(G, Xy)
            except np.linalg.LinAlgError: bOLS = np.linalg.lstsq(G, Xy, rcond=None)[0]
        else:
            Xy = X[:,A].T @ y
            z = solve_triangular(R, Xy, trans='T', lower=False)
            bOLS = solve_triangular(R, z, lower=False)
        d = X[:,A] @ bOLS - mu
        gamma_tilde, dropIdx = np.inf, 0
        if len(A)>1:
            b_act = (b[A[:-1], step-1] if storepath else b[A[:-1]])
            with np.errstate(divide='ignore', invalid='ignore'):
                g_cand = b_act / (b_act - bOLS[:-1])
            mask = g_cand > 0
            if np.any(mask):
                mi = int(np.argmin(g_cand[mask]))
                gamma_tilde = float(g_cand[mask][mi])
                dropIdx = int(np.where(mask)[0][mi])
        gamma = 1.0
        if I:
            cd = X.T @ d; cdi = cd[I]
            with np.errstate(divide='ignore', invalid='ignore'):
                g1 = (c[I] - cmax) / (cdi - cmax)
                g2 = (c[I] + cmax) / (cdi + cmax)
                gv = np.concatenate([g1, g2])
            gv = gv[gv > 1e-12]
            if gv.size: gamma = float(np.min(gv))
        if gamma_tilde < gamma:
            lasso, gamma = True, gamma_tilde
        if storepath:
            if b.shape[1] <= step: b = np.hstack([b, np.zeros((p,b.shape[1]))])
            b[:,step] = b[:,step-1]; b[A,step] = b[A,step-1] + gamma*(bOLS - b[A,step-1])
        else:
            b_prev[:] = b; b[A] += gamma*(bOLS - b[A])
        mu += gamma*d; step += 1
        if stop>0:
            curr = (b[:,step-1] if storepath else b); l1 = np.sum(np.abs(curr))
            if l1 >= stop:
                prev = (b[:,step-2] if storepath else b_prev); l1p = np.sum(np.abs(prev))
                s = (stop - l1p) / (l1 - l1p)
                if storepath: b[:,step-1] = prev + s*(curr - prev)
                else:         b[:]       = prev + s*(curr - prev)
                done = True
        if lasso and dropIdx < len(A):
            dropped = A.pop(dropIdx); I.append(dropped); I.sort()
            if not useG: R = choldelete(R, dropIdx)
            if verbose: print(f'{step}\t\t{dropped}\t{len(A)}')
        if stop<0 and len(A) >= -stop: done = True
    if storepath and b.shape[1] > step-1: b = b[:, :step]
    if step >= maxSteps: warnings.warn('LARS-EN forced exit: maximum iterations reached')
    return b, step-1
