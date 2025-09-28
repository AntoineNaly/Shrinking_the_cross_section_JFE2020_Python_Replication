# -*- coding: utf-8 -*-

"""
SCS_L2est.py — Ridge (L2) SDF estimation with kappa grid, CV, and optional PC rotation

Purpose
-------
Implements the L2 (ridge) estimator for SDF loadings b that minimizes:
    min_b  (y − X b)'(y − X b) + λ b'b
where X is the cross-sectional covariance of returns and y is the cross-sectional mean.
The procedure follows Kozak, Nagel, and Santosh (2020) with the same moment definitions
and grid logic, and returns a results dictionary used throughout the replication.

Pipeline
--------
1) Data prep (via BaseEstimator):
   • Align inputs; split TRAIN/TEST using `oos_test_date`.
   • (Optional) de-market and de-volatilize according to parameters.
2) Optional PC rotation on TRAIN:
   • Compute SVD of `regcov(TRAIN)`; set Q = U (left singular vectors).
   • Rotate both TRAIN and TEST by Q; rename portfolios to PC1, PC2, ...
   • This mirrors the Matlab approach exactly.
3) Moments and grid:
   • X = regcov(TRAIN), y = mean(TRAIN).
   • Build a κ grid (root expected SR²) via BaseEstimator._setup_grid, then map κ → λ.
   • `lCV = λ / (1 - 1/kfold)` for CV.
4) For each λ:
   • Solve ridge once (store coefficient path and s.e.).
   • Evaluate OOS objective via `cross_validate`, also recording fold MVE returns.
5) Select the optimal model by `optfunc` on the OOS objective.
6) Return a rich results dict (coeffs, paths, CV stats, optimal κ, plots/tables optionally).

Inputs
------
dates      : array-like
re         : (T×N) DataFrame/ndarray of returns (after BaseEstimator processing).
market     : length-T market series (used by BaseEstimator if demarketing/devol is enabled).
freq       : int (252 daily, 12 monthly).
anomalies  : list[str] portfolio names.
parameters : dict controlling behavior; key fields include:
   • 'objective' (default 'CSR2'), 'kfold', 'gridsize',
   • 'rotate_PC', 'demarket_unconditionally', 'devol_unconditionally',
   • plotting/export flags (all honored).

Outputs
-------
Dict `estimates` with (key subset):
   - 'coeffsPaths'  : (N×nl) L2 coefficients across grid
   - 'df'           : effective DoF across grid
   - 'objL2_IS', 'objL2_OOS' : IS/OOS objective per grid point
   - 'bL2'          : coefficients at optimal grid index
   - 'optimal_model_L2' : { 'coefficients','se','objective','dof','kappa','SR' }
   - 'x','l','lCV'  : κ grid and penalties
   - 'r_train','r_test','dd','anomalies'
   - 'rotation_matrix' : Q if PC rotation applied
   - plotting artifacts only if enabled

Numerical guarantees
--------------------
• Uses `regcov` moments and κ→λ mapping identical to the original code.
• PC rotation uses the left singular vectors of regcov(TRAIN), applied to both samples.
• Cross-validation uses contiguous folds with fixed objective definitions.
• Inversions use `solve` with `pinv` fallback to maintain stability without altering results.

Intended use
------------
Provides the L2 baseline and grids used directly by `SCS_L1L2est` (elastic-net),
and generates plots/tables for replication when requested.
"""


import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from base_estimator import BaseEstimator
from regcov import regcov
from cross_validate import cross_validate

_asnp = lambda x: x.values if hasattr(x,'values') else np.asarray(x)

def L2est(dates, re, market, freq, anomalies, parameters):
    if isinstance(re, pd.DataFrame): assert not re.isnull().any().any(), 'Missing observations in returns'
    else:                             assert not np.isnan(re).any(),       'Missing observations in returns'
    from l2est import l2est

    est = BaseEstimator(dates, re, market, freq, anomalies, parameters)
    r_train, r_test, idx_train = est._prepare_data()
    dd = dates[idx_train] if isinstance(dates, pd.Series) else dates[idx_train]

    # PC rotation (exact original logic)
    Q = None
    if est.p.get('rotate_PC', False):
        print("Rotating returns into PC space...")
        U, S, Vh = np.linalg.svd(regcov(r_train)); Q = U
        def _rot(d):
            if len(d)==0: return d
            if isinstance(d, pd.DataFrame):
                return pd.DataFrame(d.values@Q, index=d.index, columns=[f'PC{i+1}' for i in range(Q.shape[1])])
            return d @ Q
        r_train, r_test = _rot(r_train), _rot(r_test)
        est.anomalies = [f'PC{i+1}' for i in range(Q.shape[1])]
        print(f"Rotated to {len(est.anomalies)} principal components")

    # Moments & grid
    X = regcov(r_train)
    y = _asnp(np.mean(r_train, axis=0)).reshape(-1,1)
    x, l, lCV = est._setup_grid(X, y, est.p['T']); nl = len(l)

    # Precompute SVD (matches original)
    Qsvd, D, _ = np.linalg.svd(X)
    est.p.update({'Q':Qsvd, 'd':D, 'Xinv':np.linalg.pinv(X)})

    # Storage
    n = est.p['n']
    phi  = np.zeros((n, nl)); se = np.zeros((n, nl))
    objL2 = np.zeros((nl,4)); objL2_folds = np.zeros((nl, est.p['kfold']))
    MVE = [None]*nl

    print(f"\nEstimating {nl} L2 models from κ={x[0]:.4f} to κ={x[-1]:.4f}")
    for i in range(nl):
        if i % 10 == 0: print(f"  Progress: {i+1}/{nl}")
        params = est.p.copy(); params['L2pen'] = l[i]
        phi[:,i], _, se[:,i] = l2est(X, y, params, True)

        params['L2pen'] = lCV[i]
        objL2[i,:], params, objL2_folds_ = cross_validate(l2est, _asnp(dd), r_train, params)
        objL2_folds[i,:] = objL2_folds_[:,1]
        MVE[i] = params.get('cv_MVE', {})

    # Optimal pick
    iopt = est.p['optfunc_np'](objL2[:,1])
    sr = _sharpe_from_cv(MVE[iopt], est.p['freq'])
    df = np.sum(D.reshape(-1,1)/(D.reshape(-1,1)+l.reshape(1,-1)), axis=0)

    print(f"\nOptimal model: κ={x[iopt]:.4f}, OOS {est.p['sObjective']}={objL2[iopt,1]:.4f}")

    out = est.p.copy()
    out.update({
        'coeffsPaths':phi, 'df':df, 'objL2_IS':objL2[:,0], 'objL2_OOS':objL2[:,1],
        'bL2':phi[:,iopt], 'R2oos':objL2[iopt,1],
        'optimal_model_L2': {'coefficients':phi[:,iopt], 'se':se[:,iopt], 'objective':objL2[iopt,1],
                             'dof':df[iopt], 'kappa':x[iopt], 'SR':sr},
        'x':x, 'l':l, 'lCV':lCV, 'r_train':r_train, 'r_test':r_test, 'dd':dd,
        'anomalies':est.anomalies, 'rotation_matrix':Q,
        'xlbl':'Root Expected SR$^2$ (prior), $\\kappa$'
    })

    if est.p['plot_dof']:        _plot_dof(df, x, out)
    if est.p['plot_coefpaths']:  _plot_coefpaths(x, phi, iopt, est.anomalies, 'SDF Coefficient, $b$', out); _plot_coefpaths(x, phi/se, iopt, est.anomalies, 'SDF Coefficient $t$-statistic', out)
    if est.p['plot_objective']:  _plot_cv(x, objL2, out)
    _table_coefs(phi[:,iopt], se[:,iopt], est.anomalies, out)
    return out

def _sharpe_from_cv(mve_dict, freq):
    if not mve_dict: return np.nan
    v=[]
    for r in mve_dict.values():
        if r is None: continue
        if isinstance(r,(np.ndarray,pd.Series)): v.extend(np.ravel(r))
        elif np.isscalar(r): v.append(float(r))
    if len(v)>1:
        v=np.array(v); s=np.std(v); 
        if s>0: return np.mean(v)/s*np.sqrt(freq)
    return np.nan

def format_func(value, tick_number):
    """Custom formatter: only show decimals when needed"""
    if value >= 1:
        return f'{value:.0f}'  # 1, 10, 100
    elif value >= 0.1:
        return f'{value:.1f}'  # 0.1, 0.2, 0.5
    else:
        return f'{value:.2f}'  # 0.01, 0.02

def _plot_dof(df, x, p):
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(x,df,linewidth=1.5)
    if p.get('L2_log_scale',True): 
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    ax.grid(True,alpha=.3)
    ax.set_xlabel(p.get('xlbl','Kappa'))
    ax.set_ylabel('Effective degrees of freedom')
    _save_show(fig,p,'degrees_of_freedom.png')

def _plot_coefpaths(x, phi, iopt, names, ylab, p):
    fig,ax=plt.subplots(figsize=(12,8))
    iSort = iopt if p.get('L2_sort_loc','opt')=='opt' else 0
    maxl  = min(p.get('L2_max_legends',20), phi.shape[0])
    I = np.argsort(-np.abs(phi[:,iSort]))[:maxl] if phi.shape[0]>maxl else np.argsort(-phi[:,iSort])
    for i in I: ax.plot(x,phi[i,:],linewidth=1.5)
    if p.get('L2_log_scale',True): 
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    ax.axvline(x[iopt],color='k',ls='--',alpha=.7)
    ax.grid(True,alpha=.3)
    ax.set_xlabel(p.get('xlbl','Kappa'))
    ax.set_ylabel(ylab)
    ax.legend([f'Asset_{i}' for i in I[:5]],fontsize=10)
    ax.set_xlim([x.min(),x.max()])
    _save_show(fig,p,'tstats_paths.png' if 'statistic' in ylab else 'coefficients_paths.png')

def _plot_cv(x, obj, p):
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(x,obj[:,0],'--',lw=1.5,label='In-sample')
    ax.plot(x,obj[:,1],'-', lw=1.5,label=f"OOS {p.get('method','CV')}")
    if obj.shape[1]>3:
        ax.plot(x,obj[:,1]+obj[:,3],':',lw=1.5,color='tab:orange',alpha=.7,label='±1 s.e.')
        ax.plot(x,obj[:,1]-obj[:,3],':',lw=1.5,color='tab:orange',alpha=.7)
    if p.get('L2_log_scale',True):
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    ax.grid(True,alpha=.3)
    ax.set_xlabel(p.get('xlbl','Kappa'))
    ax.set_ylabel(f"IS/OOS {p.get('sObjective','Objective')}")
    ax.legend(loc='upper left',fontsize=10)
    ax.set_ylim([0,0.9])
    ax.set_xlim([x.min(),1.0])
    _save_show(fig,p,'cross_validation.png')

def _table_coefs(phi, se, names, p):
    nrows = p.get('L2_table_rows',11); t=phi/se; idx=np.argsort(np.abs(t))[::-1][:nrows]
    df = pd.DataFrame({'Portfolio':[names[i] if i<len(names) else f"Portfolio_{i}" for i in idx], 'b':phi[idx], 't_stat':np.abs(t[idx])})
    print("\n"+"="*50+"\nTop portfolios by |t-statistic|:\n"+"="*50); print(df.to_string(index=False, float_format='%.4f')); print("="*50)
    if p.get('results_export',False):
        df.to_latex('results_export/coefficients_table.tex', index=False, float_format='%.4f'); print("Saved: results_export/coefficients_table.tex")

def _save_show(fig, p, fname):
    if p.get('results_export',False):
        fig.savefig(f'results_export/{fname}', dpi=300, bbox_inches='tight'); print(f"Saved: results_export/{fname}")
    plt.show() if p.get('show_plot',False) else plt.close(fig)
