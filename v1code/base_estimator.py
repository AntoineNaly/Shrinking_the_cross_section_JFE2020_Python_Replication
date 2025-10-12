# -*- coding: utf-8 -*-

"""
base_estimator.py
=================

Purpose
-------
Lightweight scaffolding shared by L2 and L1–L2 estimators. It normalizes inputs,
sets defaults, prepares train/test splits, applies optional pre-processing
(demarket + de-vol), and builds the L2 (ridge) regularization grid in kappa space.

Inputs
------
- dates : pd.Series or np.ndarray
    Time index for the return panel. If a Series, its index is aligned to `re`/`market`.
- re : pd.DataFrame or np.ndarray (T×N)
    Panel of managed portfolio/anomaly returns.
- market : pd.Series or np.ndarray (T,)
    Market excess return used for de-marketing and scaling.
- freq : int
    Annualization frequency (252 daily, 12 monthly).
- anomalies : list[str]
    Column names (or labels) for `re`.
- parameters : dict
    Optional overrides. Important keys:
      • 'oos_test_date' (datetime): split point for a *held-out* test block used by
        the generic estimators; only the **train** side feeds cross-validation.
      • 'demarket_unconditionally' (bool, default True): remove market beta using
        betas fit on **train**, then apply to **test**.
      • 'devol_unconditionally' (bool, default True): scale every column of `re`
        to have the same std as the market (computed on the full sample).
      • 'rotate_PC' (bool, default False): (used by L2est) rotate returns to PCs.
      • plotting/export flags: 'results_export', 'show_plot', etc.

Key behavior
------------
1) Train/Test split:
   - If `dates` is a Series, the split is done by **timestamp** at `oos_test_date`.
   - Else, a boolean mask is built on positions. `r_train` is ≤ split; `r_test` is > split.
   - If no test block exists (or you don’t intend to withhold data), `r_test` is empty.

2) De-market & De-vol (if enabled):
   - De-market: regress each asset on the market using **train**, keep residuals; apply
     the **train betas** to the test side (strict out-of-sample).
   - De-vol: rescale each column to match market volatility (std computed with ddof=1).

3) L2 grid in kappa space:
   - `_setup_grid` probes coefficient stability across a rough kappa ladder, then
     builds a dense log-grid of kappas and converts to L2 penalties via `_kappa2pen`.
   - Stores helpers (`Q`, `d`, `Xinv`) for fast solvers.

Outputs/State
-------------
Methods return:
  - `_prepare_data()` → `(r_train, r_test, idx_train_mask)` and updates `self.p` with
    sample dimensions `T`, `n`.
  - `_setup_grid(X, y, T)` → `(x, l, lCV)` where `x` are kappas, `l` the L2 penalties,
    and `lCV` the CV-adjusted penalties.

Notes
-----
• `oos_test_date` here is only for the estimators’ internal holdout logic and CV; it is
  **independent** of the Table 4 “pure OOS” split, which is implemented separately.
• This module is designed to preserve the authors’ MATLAB numerics (index handling,
  ddof choices, uncentered R² conventions) to maintain exact replication.
"""

import os, numpy as np, pandas as pd

class BaseEstimator:
    def __init__(self, dates, re, market, freq, anomalies, parameters):
        self.dates, self.re, self.market, self.freq, self.anomalies = dates, re, market, freq, anomalies
        self.p = self._init_params(parameters)
        if self.p.get('results_export', False): os.makedirs('results_export', exist_ok=True)

    def _init_params(self, params):
        p = {
            'gridsize':20, 'method':'CV', 'objective':'CSR2', 'kfold':5, 'freq':self.freq,
            'rotate_PC':False, 'demarket_unconditionally':True, 'devol_unconditionally':True,
            'plot_dof':False, 'plot_coefpaths':False, 'plot_objective':False,
            'L2_max_legends':20, 'L2_sort_loc':'opt', 'L2_log_scale':True, 'L1_log_scale':True,
            'results_export':True, 'show_plot':False,
            'oos_test_date': self.dates.iloc[-1] if isinstance(self.dates, pd.Series) else self.dates[-1]
        }
        p.update(params)
        if p['objective'] in ('GLS','SSE'):   p['optfunc'], p['optfunc_np'] = min, np.argmin
        else:                                  p['optfunc'], p['optfunc_np'] = max, np.argmax
        names = {'CSR2':'Cross-sectional $R^2$','GLSR2':'Cross-sectional GLS $R^2$','GLS':'Residual $SR^2$',
                 'SRexpl':'Explained SR','SSE':'SDF RMSE','SR':'Sharpe Ratio','MVU':'Mean-variance utility'}
        p['sObjective'] = names.get(p['objective'], p['objective'])
        return p

    def _prepare_data(self):
        tT0 = pd.to_datetime(self.p['oos_test_date']) if isinstance(self.p['oos_test_date'], str) else self.p['oos_test_date']
        if isinstance(self.re, pd.DataFrame) and isinstance(self.dates, pd.Series):
            self.re, self.market, self.dates = [x.reset_index(drop=True) for x in [self.re, self.market, self.dates]]
            self.re.index = self.market.index = self.dates
        if isinstance(self.dates, pd.Series):
            idx_train = (self.dates <= tT0).values; idx_test = (self.dates > tT0).values
        else:
            idx_train = self.dates <= tT0;         idx_test = self.dates > tT0

        r0 = self._process_returns(idx_train, idx_test)
        r_train, r_test = self._split_data(r0, idx_train, idx_test)
        T,n = (r_train.shape if isinstance(r_train, pd.DataFrame) else (r_train.shape if len(r_train.shape)>1 else (len(r_train),1)))
        self.p.update({'T':T,'n':n})
        return r_train, r_test, idx_train

    def _split_data(self, r0, idx_train, idx_test):
        if isinstance(r0, pd.DataFrame): return r0[idx_train], (r0[idx_test] if idx_test.any() else pd.DataFrame())
        return r0[idx_train,:], (r0[idx_test,:] if idx_test.any() else np.array([]))

    def _process_returns(self, idx_train, idx_test):
        r0 = self.re.copy()
        if self.p['demarket_unconditionally']: r0 = self._demarket(idx_train, idx_test)
        if self.p['devol_unconditionally']:
            market_std = (self.market.std(ddof=1) if hasattr(self.market,'std') else np.std(self.market, ddof=1))
            if isinstance(r0, pd.DataFrame):
                r0 = r0.divide(r0.std(axis=0, ddof=1), axis=1).multiply(market_std)
            else:
                s = np.nanstd(r0, axis=0, ddof=1); s[s==0]=1.0; r0 = r0/s*market_std
        return r0

    def _demarket(self, idx_train, idx_test):
        from demarket import demarket
        if isinstance(self.re, pd.DataFrame):
            r_tr, r_te = self.re[idx_train], (self.re[idx_test] if idx_test.any() else pd.DataFrame())
            m_tr, m_te = self.market[idx_train], (self.market[idx_test] if idx_test.any() else pd.Series())
        else:
            r_tr, r_te = self.re[idx_train,:], (self.re[idx_test,:] if idx_test.any() else np.array([]))
            m_tr, m_te = self.market[idx_train], (self.market[idx_test] if idx_test.any() else np.array([]))
        r_tr_d, b = demarket(r_tr, m_tr)
        if len(r_te)>0:
            r_te_d, _ = demarket(r_te, m_te, b)
            return pd.concat([r_tr_d, r_te_d], axis=0) if isinstance(r_tr_d, pd.DataFrame) else np.vstack([r_tr_d, r_te_d])
        return r_tr_d

    def _setup_grid(self, X, y, T):
        from l2est import l2est as l2
        n = self.p['n']; lr = np.arange(1,22); lm = 1; z = np.zeros((n,len(lr)))
        for i,L in enumerate(lr):
            prm = self.p.copy(); prm['L2pen'] = self._kappa2pen(2**(L-lm), T, X)
            z[:,i] = l2(X, y, prm)[0].flatten()
        m = np.nanmean(np.abs(np.diff(z,axis=1))/(1+np.abs(z[:,:-1])), axis=0)
        x_rlim = lr[np.where(m>0.01)[0]] - lm
        if len(x_rlim)==0: raise ValueError("No coefficient stabilization found. Check grid/data.")
        x = np.logspace(np.log10(2**x_rlim[-1]), np.log10(0.01), self.p['gridsize'])
        l = np.array([self._kappa2pen(k, T, X) for k in x])
        lCV = l / (1 - 1/self.p['kfold'])
        return x, l, lCV

    def _kappa2pen(self, kappa, T, X):
        return self.p['freq'] * np.trace(X) / T / (kappa**2)
