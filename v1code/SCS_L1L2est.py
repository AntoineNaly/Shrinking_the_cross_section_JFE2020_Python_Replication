# -*- coding: utf-8 -*-

"""
SCS_L1L2est.py — L¹–L² (Elastic-Net like) estimator for Kozak, Nagel & Santosh (2020 JFE)
------------------------------------------------------------------------------------
Goal:
    Combine ridge (L²) and lasso (L¹) regularization to estimate sparse SDF models,
    performing full cross-validation across grids of L¹ and L² penalties.
    Produces Figures 1a–b and 3a–b (OOS R² contours).

Inputs:
    - dates, re, market, freq: Time series data
    - anomalies: list of portfolio returns or factors
    - parameters: configuration dictionary for L¹–L² estimation

Outputs:
    - Optimal Elastic-Net like model (coefficients, R², sparsity)
    - Contour plots in 'results_export/'
"""


import numpy as np, pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import os

from base_estimator import BaseEstimator
from SCS_L2est import L2est
from cross_validate import cross_validate
from elasticnet_sdf_HJdist import elasticnet_sdf_HJdist

_asnp = lambda x: x.values if hasattr(x,'values') else np.asarray(x)

def _proc_one_L2(l2pen, p_cache, L1range, dd, r_train):
    """Run CV across L1 grid for a single L2 penalty."""
    p = p_cache.copy(); p.update({'bpath':{}, 'storepath':True, 'L2pen':l2pen})
    kfold, L1rn = p_cache['kfold'], len(L1range)
    cv, phis, cv_folds_i, first = np.full((L1rn,4),np.nan), {}, np.full((L1rn,kfold),np.nan), True
    Xd = _asnp(dd); Rt = r_train
    for j in range(L1rn-1, -1, -1):
        p.update({'stop':-L1range[j], 'use_precomputed': not first}); first=False
        cv_res, p, cv_f = cross_validate(elasticnet_sdf_HJdist, Xd, Rt, p)
        cv[j,:], cv_folds_i[j,:] = cv_res, cv_f[:,1]
        if 'cv_phi' in p: phis[j] = p['cv_phi'].copy()
    return cv[:,1], cv[:,3], [phis.get(j) for j in range(L1rn)], cv_folds_i

def L1L2est(dates, re, market, freq, anomalies, parameters):
    print("Running L2 estimation first...")
    est = L2est(dates, re, market, freq, anomalies, parameters)
    est.update({'plot_L1L2map':parameters.get('plot_L1L2map',True),
                'n_jobs':parameters.get('n_jobs',-1),
                'L1_truncPath':parameters.get('L1_truncPath',True)})

    x, lCV = est['x'], est['lCV']; nl = len(x); n_total = len(anomalies)
    has_re_ew = isinstance(anomalies, list) and ('re_ew' in anomalies)

    # L1 grid
    l1min, l1max, l1n = parameters.get('l1_lambda_min'), parameters.get('l1_lambda_max'), parameters.get('l1_grid_size',50)
    if l1min is not None and l1max is not None:
        l1min, l1max = max(1,min(l1min,n_total)), max(l1min,min(l1max,n_total))
        if l1max - l1min < l1n: L1range = np.arange(l1min, l1max+1)
        else:                   L1range = np.unique(np.logspace(np.log10(l1min), np.log10(l1max), l1n).astype(int))
        if 'target_factors' in parameters:
            t = parameters['target_factors']
            if (t not in L1range) and (l1min<=t<=l1max): L1range = np.sort(np.append(L1range,t))
    else:
        L1range = np.unique(np.logspace(0, np.log10(n_total), 150).astype(int))
    L1rn = len(L1range)

    print(f"\n{'='*50}\nStarting L1-L2 (Elastic Net) estimation...\n  L2 grid: {nl} points\n  L1 grid: {L1rn} points ({L1range[0]} to {L1range[-1]} variables)")
    if has_re_ew:
        print(f"  Using {n_total} portfolios: {n_total-1} anomalies + re_ew (equal-weighted market return)\n  NOTE: re_ew included to match paper results")
    else:
        print(f"  Using {n_total} anomaly portfolios")

    r_train, r_test, dd, processed_anoms = [est.get(k) for k in ['r_train','r_test','dd','anomalies']]
    if (r_train is None) or (dd is None) or (isinstance(r_train,pd.DataFrame) and r_train.empty) or (isinstance(r_train,np.ndarray) and r_train.size==0):
        print("WARNING: No processed data from L2; rebuilding...")
        r_train, r_test, dd, processed_anoms = _recreate_fallback(dates, re, market, freq, anomalies, parameters)

    # cache
    p_cache_base = est.copy(); p_cache_base['cache_run']=True
    _, p_cache, _ = cross_validate(elasticnet_sdf_HJdist, _asnp(dd), r_train, p_cache_base)
    p_cache['cache_run']=False

    # run across L2 penalties
    shape = (nl, L1rn); cv_test = np.full(shape,np.nan); cv_se = np.full(shape,np.nan)
    cv_phi = [[None]*L1rn for _ in range(nl)]; cv_folds = np.full((*shape, est['kfold']), np.nan)

    if est.get('n_jobs',1)==1:
        print("\nRunning elastic-net like estimation (sequential)...")
        it = [ _proc_one_L2(lCV[i], p_cache, L1range, dd, r_train) for i in tqdm(range(nl),desc="L2 penalties") ]
    else:
        print(f"\nRunning elastic-net like estimation (parallel with {est['n_jobs']} jobs)...")
        it = Parallel(n_jobs=est['n_jobs'])( delayed(_proc_one_L2)(lCV[i], p_cache, L1range, dd, r_train)
                                            for i in tqdm(range(nl),desc="L2 penalties") )

    for i,(cvt,cse,phis,cf) in enumerate(it):
        cv_test[i,:], cv_se[i,:], cv_phi[i], cv_folds[i,:,:] = cvt, cse, phis, cf

    # choose optimal combo
    print("\nFinding optimal L1-L2 combination...")
    optfunc = est['optfunc']
    if 'target_factors' in parameters:
        t, w = parameters['target_factors'], parameters.get('target_window',2)
        tidx = np.argmin(np.abs(L1range - t)); s,e = max(0,tidx-w), min(L1rn, tidx+w+1)
        I = (np.nanargmax if optfunc==max else np.nanargmin)(cv_test[:,s:e], axis=0)
        M = np.array([cv_test[I[j-s], j] for j in range(s,e)])
        j_star = (np.nanargmax if optfunc==max else np.nanargmin)(M); cv_L1pen = s + j_star; cv_L2pen = I[j_star]; best = M[j_star]
    else:
        I = (np.nanargmax if optfunc==max else np.nanargmin)(cv_test, axis=0)
        M = np.array([cv_test[I[j], j] for j in range(L1rn)])
        cv_L1pen = (np.nanargmax if optfunc==max else np.nanargmin)(M); cv_L2pen = I[cv_L1pen]; best = M[cv_L1pen]

    Mk = (np.nanmax if optfunc==max else np.nanmin)(cv_folds, axis=(0,1))
    M_fold = cv_folds[cv_L2pen, cv_L1pen, :]
    bias_mean = np.mean(Mk - M_fold); bias_se = np.std(Mk - M_fold, ddof=1)/np.sqrt(est['kfold'])

    # nonzero count from stored phis
    opt_phi = cv_phi[cv_L2pen][cv_L1pen]
    n_nz = 0
    if opt_phi is not None:
        if isinstance(opt_phi, dict):
            vals = [v.flatten() for v in opt_phi.values() if v is not None]
            opt_phi_mean = np.mean(vals, axis=0) if vals else None
        else:
            opt_phi_mean = opt_phi
        if opt_phi_mean is not None: n_nz = int(np.sum(np.abs(opt_phi_mean) > 1e-10))

    _print_opt(L1range, cv_L1pen, x, cv_L2pen, best, est['sObjective'], bias_mean, bias_se, est, n_total, n_nz)

    est.update({
        'optimal_model_L1L2': _opt_dict(L1range[cv_L1pen], x[cv_L2pen], best, bias_mean, bias_se, cv_L2pen, cv_L1pen, est, n_total, n_nz),
        'cv_test_L1L2': cv_test, 'cv_test_se_L1L2': cv_se, 'L1range': L1range,
        'cv_phi_L1L2': cv_phi, 'anomalies': processed_anoms
    })
    if opt_phi is not None:
        if isinstance(opt_phi, dict):
            vals = [v.flatten() for v in opt_phi.values() if v is not None]
            opt_phi = np.mean(vals, axis=0) if vals else None
        if opt_phi is not None: est['optimal_model_L1L2']['coefficients']=opt_phi

    if est['plot_L1L2map']: _plot_map(x, L1range, cv_test, est, fname="elasticnet_contour.png")
    return est

def _recreate_fallback(dates, re, market, freq, anomalies, p):
    est = BaseEstimator(dates, re, market, freq, anomalies, p)
    r_train, r_test, idx_train = est._prepare_data()
    dd = dates[idx_train] if isinstance(dates,pd.Series) else dates[idx_train]
    if p.get('rotate_PC',False):
        print("Applying PC rotation (fallback)...")
        from regcov import regcov
        U,S,Vh = np.linalg.svd(regcov(r_train)); Q = U
        def _rot(d): 
            if len(d)==0: return d
            if isinstance(d,pd.DataFrame): return pd.DataFrame(d.values@Q, index=d.index, columns=[f'PC{i+1}' for i in range(Q.shape[1])])
            return d @ Q
        r_train, r_test = _rot(r_train), _rot(r_test)
        anoms = [f'PC{i+1}' for i in range(Q.shape[1])]
        print(f"Applied PC rotation to {len(anoms)} components")
    else:
        anoms = anomalies
    return r_train, r_test, dd, anoms

def _print_opt(L1range, j, x, i, obj, sObj, bmean, bse, est, n_tot, n_nz):
    l2obj = est['optimal_model_L2']['objective']
    imp = (obj - l2obj)/l2obj*100 if l2obj!=0 else 0
    print("\nOptimal L1-L2 model:")
    print(f"  Total variables available: {n_tot}\n  Target sparsity: {L1range[j]}\n  Actual non-zero coefficients: {n_nz}\n  Kappa (L2): {x[i]:.4f}\n  OOS {sObj}: {obj:.4f}\n  Bias correction: {bmean:.4f} ± {bse:.4f}\n  Bias-corrected OOS {sObj}: {obj+bmean:.4f}")
    print("\nComparison with L2-only:")
    print(f"  L2-only OOS {sObj}: {l2obj:.4f}\n  Improvement: {imp:+.1f}%")

def _opt_dict(tgt, kappa, obj, bmean, bse, iL2, iL1, est, n_tot, n_nz):
    l2obj = est['optimal_model_L2']['objective']
    imp = (obj - l2obj)/l2obj*100 if l2obj!=0 else 0
    return {'n_total_variables':n_tot,'target_sparsity':tgt,'n_nonzero':n_nz,'kappa':kappa,
            'objective':obj,'bias':bmean,'bias_se':bse,'objective_corrected':obj+bmean,
            'L2_idx':iL2,'L1_idx':iL1,'improvement_over_L2':imp}

def _plot_map(x, L1range, cv_test, p, fname="elasticnet_contour.png"):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(figsize=(10, 8))
    # Mesh + values (Z shape: len(L1range) × len(x))
    X, Y = np.meshgrid(x, L1range)
    Z = np.asarray(cv_test).T
    # Clip for nicer dynamic range
    if p.get('objective') in ['CSR2', 'GLSR2', 'SRexpl', 'MVU']:
        Z_plot = np.maximum(-0.1, Z)
    else:
        Z_plot = np.minimum(Z, np.nanmin(Z) + 3 * np.nanstd(Z))
    # Filled contours
    levels = np.linspace(np.nanmin(Z_plot), np.nanmax(Z_plot), 50)
    contf = ax.contourf(X, Y, Z_plot, levels=levels, cmap='viridis', antialiased=True)
    # Contour lines
    ax.contour(X, Y, Z_plot, levels=contf.levels, colors='k',
               linewidths=0.45, alpha=0.35, antialiased=True)
    # Mark optimal point
    iL2, iL1 = p['optimal_model_L1L2']['L2_idx'], p['optimal_model_L1L2']['L1_idx']
    ax.plot(x[iL2], L1range[iL1], marker='*', markersize=15,
            markeredgecolor='white', markeredgewidth=2, color='k')
    # Log axes
    ax.set_xscale('log'); ax.set_yscale('log')
    # X ticks
    x_min, x_max = float(np.min(x)), float(np.max(x))
    xticks = [t for t in [0.01, 0.1, 1, 10] if x_min <= t <= x_max] or [x_min, x_max]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:g}" for t in xticks])
    ax.set_xlim([x_min, max(xticks)])
    # Y ticks
    y_min, y_max = int(np.min(L1range)), int(np.max(L1range))
    ytgrid = [1, 2, 5, 10, 20, 50, 100, 200]
    yticks = [t for t in ytgrid if y_min <= t <= y_max] or [y_min, y_max]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(t)) for t in yticks])
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel(r'Root Expected $SR^2$ (prior), $\kappa$', fontsize=12)
    ax.set_ylabel('Number of nonzero coefficients', fontsize=12)
    ax.set_title('L1–L2 Regularization: Cross-sectional $R^2$', fontsize=14)
    # Info box
    opt = p['optimal_model_L1L2']
    box = (r'$\star$ Optimal:'      f"\n{opt['n_nonzero']} variables"
           f"\n$\kappa$ = {opt['kappa']:.3f}"
           f"\nCross-sectional $R^2$ = {opt['objective']:.3f}")
    props = dict(boxstyle='round', facecolor='white', alpha=0.92,
                 edgecolor='black', linewidth=0.6)
    ax.text(0.98, 0.02, box, transform=ax.transAxes, va='bottom',
            ha='right', fontsize=10, bbox=props)
    # Colorbar
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.1)
    cbar = plt.colorbar(contf, cax=cax)
    cbar.set_label('Cross-sectional $R^2$', fontsize=12, labelpad=12)
    cticks = np.linspace(np.nanmin(Z_plot), np.nanmax(Z_plot), 6)
    cbar.set_ticks(cticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in cticks])
    cbar.ax.tick_params(labelsize=10)
    ax.margins(0); plt.tight_layout()

    # --- Save logic (raw vs PC, dataset-aware) ---
    if p.get('results_export', False):
        os.makedirs('results_export', exist_ok=True)
        provider = p.get('dataprovider', '')
        rotate_pc = p.get('rotate_PC', False)
        if provider == 'ff25':
            fname_out = 'results_export/figure1b.png' if rotate_pc else 'results_export/figure1a.png'
        elif provider == 'anom':
            fname_out = 'results_export/figure3b.png' if rotate_pc else 'results_export/figure3a.png'
        else:
            suffix = '_pc' if rotate_pc else '_raw'
            fname_out = f'results_export/elasticnet_contour{suffix}.png'
        fig.savefig(fname_out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {fname_out}")

    if p.get('show_plot', False): 
        plt.show()
    else: 
        plt.close(fig)
    return fig

