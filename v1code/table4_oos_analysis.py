# -*- coding: utf-8 -*-

"""
table4_oos_analysis.py — Spec-accurate replication of KNS (2020) JFE Table 4 (2005–2017 OOS)

What this file does
-------------------
Implements the exact pre/OOS design used in Table 4:

  1) PRE/OOS split (strict):
     • PRE: dates < pure_oos_date
     • OOS: dates ≥ pure_oos_date
     Slicing is **positional** (no label-based indexing) to avoid alignment errors.

  2) De-market all candidate returns using PRE betas:
     • Estimate betas on PRE only, apply the same betas to OOS.
     • This is done both for (i) the 50 anomaly portfolios and (ii) the FF non-market factors.

  3) SDF (L2) on PRE → OOS MVE:
     • Run `L2est` on PRE demarketed anomalies (plot/export disabled).
     • Obtain b̂ and form OOS MVE_t = F_t' b̂ on the OOS demarketed anomalies.
     • Scale MVE to match the **OOS market volatility** (ddof=1) before regressions.

  4) Benchmarks (each estimated on PRE, applied to OOS):
     • CAPM            : zero benchmark (alpha = mean of scaled OOS MVE).
     • FF6 single MVE  : use the 5 non-market factors {SMB, HML, RMW, CMA, Mom}.
                         Demarket them with PRE betas; compute w = Σ^{-1} μ on PRE; apply to OOS.
     • Char-sparse (5) : run `L1L2est` in the original anomaly space on PRE with grid fixed to exactly 5
                         non-zeros; apply the 5-sparse b̂ to OOS demarketed anomalies.
     • PC-sparse (5)   : PCA on PRE only (mean-centered, covariance eigendecomposition).
                         Rotate both PRE/OOS by PRE eigenvectors; run `L1L2est` in PC space with exactly
                         5 non-zeros; apply to OOS PCs. Also report top-5 variance share.

  5) Regressions (OOS only):
     • For each benchmark B_t, run OLS:  MVE_t = α + β B_t + ε_t
     • Report **annualized** α and s.e. (homoskedastic OLS, ddof consistent with code), plus R².

Inputs / expectations
---------------------
Requires a params dict `p` produced by upstream estimation, containing:
  • p['dd']             : date vector aligned to p['r_train'] and `market`
  • p['r_train']        : T×N panel of managed portfolios (as used by L2/L1L2)
  • p['anomalies']      : list of N portfolio names
  • p['freq']           : 252 (daily) or 12 (monthly)
  • p['pure_oos_date']  : datetime split (OOS starts here)
  • p['daily']          : bool to choose daily vs monthly FF files in ./Data
Plus a `market` series aligned to p['r_train'].

Outputs
-------
`add_table4_to_main_results(p, dd, market)`:
  • Runs the full pipeline, attaches a dict under p['table4_results'] with:
      capm_alpha/se/tstat,
      ff6_alpha/se/tstat/r_squared,
      sparse5_alpha/se/tstat/r_squared,
      pc_sparse5_alpha/se/tstat/r_squared,
      pc_sparse5_var_explained,
      mve_mean_return, mve_volatility,
      n_pre_2005, n_post_2005, optimal_kappa
  • Prints a compact Table-4 summary to stdout.

Numerical guarantees
--------------------
  • PRE/OOS split, demarketing, and scaling match the specification.
  • Uses existing `L2est`/`L1L2est` for coefficients; plotting/export disabled in PRE fits.
  • Operation order and ddof settings are fixed to preserve numerical equivalence (e.g., CAPM column).
"""

import numpy as np
import pandas as pd
import glob, os

from SCS_L2est import L2est
from SCS_L1L2est import L1L2est
from demarket import demarket
from pathlib import Path


def add_table4_to_main_results(p, dd, market):
    print("\n" + "="*60)
    print("="*60)
    res = _run_table4(p, dd, market)
    p['table4_results'] = res
    _print_table(res)
    export_table4_to_latex(res)

# ---------------- core ----------------
def _run_table4(p, dd, market):
    """
    Strict pre-2005 fit / 2005–2017 apply for:
      - SDF (L2) MVE
      - CAPM (zero benchmark)
      - FF6 unregularized MVE (5 non-market)
      - Char-sparse(5) (dual-penalty in anomaly space)
      - PC-sparse(5)   (dual-penalty in PC space with PCA fit on pre only)
    Uses positional slicing to avoid index-alignment errors.
    """
    split = p['pure_oos_date']

    # Dates aligned BY POSITION to r_train & market
    dates = pd.to_datetime(p['dd'])
    r = p['r_train']                  # T×N (DataFrame or ndarray)
    names = p['anomalies']

    # --- positional masks (no label alignment) ---
    pre_mask_pos  = (dates.values < np.datetime64(split))
    post_mask_pos = ~pre_mask_pos

    def _pos_slice(obj, mask):
        if hasattr(obj, 'iloc'):     # pandas
            return obj.iloc[mask]
        else:                        # numpy
            arr = np.asarray(obj)
            return arr[mask]

    # Slices (positional)
    r_pre  = _pos_slice(r, pre_mask_pos).copy()
    r_post = _pos_slice(r, post_mask_pos).copy()
    m_pre  = _pos_slice(market, pre_mask_pos)
    m_post = _pos_slice(market, post_mask_pos)

    # Matching date slices (use these anywhere you need dates)
    dates_pre  = dates[pre_mask_pos]
    dates_post = dates[post_mask_pos]

    # Basic sanity checks
    T = len(dates)
    if hasattr(r, '__len__'):
        assert _as_np(r).shape[0] == T, "r_train rows must match length of dates"
    if hasattr(market, '__len__'):
        assert len(market) == T, "market length must match dates"
    assert pre_mask_pos.sum() + post_mask_pos.sum() == T
    
    # 1) Orthogonalize using betas fit on PRE only, then apply to OOS
    # Ensure correct shapes for demarket(): r = (T×N), mkt = (T,)
    r_pre_arr  = np.asarray(r_pre)
    r_post_arr = np.asarray(r_post)
    
    # Flatten market series but preserve time dimension alignment
    m_pre_arr  = np.asarray(m_pre).reshape(-1)
    m_post_arr = np.asarray(m_post).reshape(-1)
    
    # Sanity check: ensure matching time dimension
    if r_pre_arr.shape[0] != m_pre_arr.shape[0]:
        raise ValueError(f"PRE mismatch: r_pre has {r_pre_arr.shape[0]} rows, m_pre has {m_pre_arr.shape[0]}")
    if r_post_arr.shape[0] != m_post_arr.shape[0]:
        raise ValueError(f"OOS mismatch: r_post has {r_post_arr.shape[0]} rows, m_post has {m_post_arr.shape[0]}")
    
    # Run demarket consistently with your demarket.py function
    r_pre_t, betas = demarket(r_pre_arr, m_pre_arr)
    r_post_t, _    = demarket(r_post_arr, m_post_arr, betas)

    # 2) L2 on PRE only (pass the PRE dates), then build OOS MVE from SDF
    L2p = p.copy()
    for k in ['plot_dof','plot_coefpaths','plot_objective','plot_L1L2map','results_export']:
        L2p[k] = False

    est_pre = L2est(dates_pre, r_pre_t, m_pre, p['freq'], names, L2p)
    b_hat = _as_np(est_pre['optimal_model_L2']['coefficients']).reshape(-1)

    mve_sdf_oos = _as_np(r_post_t) @ b_hat

    # 3) Scale SDF MVE to OOS market σ
    sigma_M = _std_np(m_post)
    mve_sdf_scaled = _scale_to_sigma(mve_sdf_oos, sigma_M)

    # 4) Build benchmarks (all fitted on PRE only, then applied to OOS)

    # 4a) CAPM (benchmark = 0)
    capm = _alpha_const(mve_sdf_scaled, p['freq'])

    # 4b) FF6 single MVE: Σ^{-1}μ on PRE (demarketed 5 non-market factors), apply OOS
    ff6_panel = _ff6_single_mve(dates, split, p.get('daily', True))   # DF (SMB,HML,RMW,CMA,MOM) aligned to dates
    ff6_pre   = ff6_panel.loc[dates_pre]
    ff6_post  = ff6_panel.loc[dates_post]

    mkt_pre_ff  = _ff_mkt(dates, split, p.get('daily', True), pre=True).loc[dates_pre]
    mkt_post_ff = _ff_mkt(dates, split, p.get('daily', True), pre=False).loc[dates_post]

    ff6_pre_t, bet_ff = demarket(ff6_pre,  mkt_pre_ff)            # fit betas on PRE
    ff6_post_t, _     = demarket(ff6_post, mkt_post_ff, bet_ff)   # apply to OOS

    w_ff = _unreg_mve(ff6_pre_t)                                  # w = Σ^{-1} μ (PRE)
    ff6_oos = _as_np(ff6_post_t) @ w_ff
    ff6_scaled = _scale_to_sigma(ff6_oos, sigma_M)
    ff6 = _alpha_ols(mve_sdf_scaled, ff6_scaled, p['freq'])

    # 4c) Char-sparse(5): dual-penalty on PRE (anomaly basis), apply to OOS
    sparse5_oos = _sparse_k_bench(r_pre_t, r_post_t, dates_pre, m_pre, names, p, k=5)
    sparse5_scaled = _scale_to_sigma(sparse5_oos, sigma_M)
    sparse5 = _alpha_ols(mve_sdf_scaled, sparse5_scaled, p['freq'])

    # 4d) PC-sparse(5): PCA on PRE only, rotate, dual-penalty in PC space (k=5), apply to OOS
    pc5_oos, pc_var5 = _pc_sparse_k_bench(r_pre_t, r_post_t, dates_pre, p, k=5)
    pc5_scaled = _scale_to_sigma(pc5_oos, sigma_M)
    pc5 = _alpha_ols(mve_sdf_scaled, pc5_scaled, p['freq'])

    # 5) Return consolidated results
    return dict(
        capm_alpha=capm['alpha'], capm_se=capm['se'], capm_tstat=capm['t'],

        ff6_alpha=ff6['alpha'], ff6_se=ff6['se'], ff6_tstat=ff6['t'], ff6_r_squared=ff6['r2'],

        sparse5_alpha=sparse5['alpha'], sparse5_se=sparse5['se'],
        sparse5_tstat=sparse5['t'], sparse5_r_squared=sparse5['r2'],

        pc_sparse5_alpha=pc5['alpha'], pc_sparse5_se=pc5['se'],
        pc_sparse5_tstat=pc5['t'], pc_sparse5_r_squared=pc5['r2'],
        pc_sparse5_var_explained=pc_var5,

        mve_mean_return=float(np.mean(mve_sdf_scaled) * p['freq']),
        mve_volatility=float(np.std(mve_sdf_scaled, ddof=1) * np.sqrt(p['freq'])),

        n_pre_2005=int(pre_mask_pos.sum()),
        n_post_2005=int(post_mask_pos.sum()),
        optimal_kappa=float(est_pre['optimal_model_L2']['kappa']),
    )


# ---------------- benchmarks ----------------
def _sparse_k_bench(r_pre_t, r_post_t, dates_pre, m_pre, names, p, k=5):
    params = p.copy()
    for kk in ['plot_dof','plot_coefpaths','plot_objective','plot_L1L2map','results_export']:
        params[kk] = False
    params.update(dict(l1_lambda_min=k, l1_lambda_max=k, l1_grid_size=1,
                       target_factors=k, target_window=0, rotate_PC=False))
    est = L1L2est(dates_pre, r_pre_t, m_pre, p['freq'], names, params)
    b = _as_np(est['optimal_model_L1L2'].get('coefficients'))
    return _as_np(r_post_t) @ b

def _pc_sparse_k_bench(r_pre_t, r_post_t, dates_pre, p, k=5):
    # PCA on pre only, freeze Q, rotate both samples
    Rpre = _as_np(r_pre_t); Rpost = _as_np(r_post_t)
    mu = Rpre.mean(axis=0, keepdims=True)
    C = np.cov((Rpre - mu).T, ddof=1)
    eigval, eigvec = np.linalg.eigh(C)
    idx = eigval.argsort()[::-1]
    L, Q = eigval[idx], eigvec[:, idx]
    PCpre, PCpost = Rpre @ Q, Rpost @ Q
    pc_names = [f'PC{i+1}' for i in range(PCpre.shape[1])]

    # dual-penalty on PCs (pre only), exactly 5 nonzeros
    params = p.copy()
    for kk in ['plot_dof','plot_coefpaths','plot_objective','plot_L1L2map','results_export']:
        params[kk] = False
    params.update({
        'l1_lambda_min': k, 'l1_lambda_max': k, 'l1_grid_size': 1,
        'target_factors': k, 'target_window': 0,
        # CRITICAL for PCs with dummy market:
        'demarket_unconditionally': False,
        'devol_unconditionally':   False,
    })
    dummy_mkt = np.zeros(PCpre.shape[0])
    est = L1L2est(dates_pre, PCpre, dummy_mkt, p['freq'], pc_names, params)
    b_pc = _as_np(est['optimal_model_L1L2'].get('coefficients'))
    return PCpost @ b_pc, (L[:k].sum()/L.sum() if L.sum()>0 else np.nan)

def _ff6_single_mve(dates, split, daily=True):
    """Return DataFrame of 5 non-market FF factors aligned to your sample dates (decimals)."""
    REPO_ROOT = Path(__file__).resolve().parent.parent
    datapath = REPO_ROOT / "Data"
    if daily:
        ff5 = sorted(glob.glob(os.path.join(datapath, '*5_Factors*_daily*.csv')))[-1]
        mom = sorted(glob.glob(os.path.join(datapath, '*Momentum*_daily*.csv')))[-1]
        date_fmt = "%Y/%m/%d"
    else:
        ff5 = sorted(glob.glob(os.path.join(datapath, '*5_Factors*_monthly*.csv')))[-1]
        mom = sorted(glob.glob(os.path.join(datapath, '*Momentum*_monthly*.csv')))[-1]
        date_fmt = "%Y%m"

    ff5_df = pd.read_csv(ff5)
    ff5_df['Date'] = pd.to_datetime(ff5_df.iloc[:,0].astype(str), format=date_fmt)
    mom_df = pd.read_csv(mom)
    mom_df.columns = mom_df.columns.str.strip()
    mom_df['Date'] = pd.to_datetime(mom_df.iloc[:,0].astype(str), format=date_fmt)

    ff = (ff5_df.merge(mom_df[['Date','Mom']], on='Date', how='left')
                  .set_index('Date')
                  .rename(columns=str.strip))/100.0
    # keep only non-market five
    return ff[['SMB','HML','RMW','CMA','Mom']].loc[pd.to_datetime(dates)]

def _ff_mkt(dates, split, daily, pre=True):
    # helper to get aligned market (Mkt-RF) series for demarket of FF factors
    datapath = os.path.join(os.getcwd(), 'Data')
    patt = '*5_Factors*_daily*.csv' if daily else '*5_Factors*_monthly*.csv'
    ff5 = sorted(glob.glob(os.path.join(datapath, patt)))[-1]
    date_fmt = "%Y/%m/%d" if daily else "%Y%m"
    df = pd.read_csv(ff5)
    df['Date'] = pd.to_datetime(df.iloc[:,0].astype(str), format=date_fmt)
    s = (df.set_index('Date').rename(columns=str.strip)['Mkt-RF']/100.0).loc[pd.to_datetime(dates)]
    return s.loc[s.index < split] if pre else s.loc[s.index >= split]


# ---------------- small math utils ----------------
def _as_np(x):  return x.values if hasattr(x, "values") else np.asarray(x)
def _std(x):   return np.std(_as_np(x), ddof=1)
def _std_np(x): return np.std(_as_np(x), ddof=1)

def _scale_to_sigma(series, target_sigma):
    s = _as_np(series)
    cur = np.std(s, ddof=1)
    return s if cur == 0 else s * (target_sigma/cur)

def _unreg_mve(df_pre_demarketed):
    mu = _as_np(df_pre_demarketed.mean())
    Sigma = _as_np(df_pre_demarketed.cov())
    try:        return np.linalg.solve(Sigma, mu)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(Sigma) @ mu

def _alpha_const(y, freq):
    y = _as_np(y).ravel()
    n = len(y)
    a = y.mean()
    se = y.std(ddof=1)/np.sqrt(n)
    return dict(alpha=a*freq, se=se*freq, t=(a/se if se>0 else np.nan))

def _alpha_ols(y, x, freq):
    y = _as_np(y).ravel(); x = _as_np(x).ravel()
    n = min(len(y), len(x)); y, x = y[:n], x[:n]
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.solve(X.T@X, X.T@y)
    res = y - X@beta
    s2 = (res@res)/(n-2)
    varb = s2*np.linalg.inv(X.T@X)
    se_a = np.sqrt(varb[0,0])
    r2 = 1 - (res@res)/np.sum((y - y.mean())**2)
    return dict(alpha=beta[0]*freq, se=se_a*freq, t=(beta[0]/se_a if se_a>0 else np.nan), r2=r2)



# ---------------- console print ----------------
def _print_table(r):
    print("\n" + "-"*86)
    print("Table 4: OOS MVE α (2005–2017), %, annualised; s.e. in parentheses")
    f = lambda a,s: f"{a*100:6.2f}  ({s*100:5.2f})"
    print(f"{'SDF factors':22s} {'CAPM':>16s} {'FF 6-factor':>16s} {'Char.-sparse':>16s} {'PC-sparse':>16s}")
    print("-"*86)
    print(f"{'50 anomaly portfolios':22s} "
          f"{f(r['capm_alpha'],r['capm_se']):>16s} "
          f"{f(r['ff6_alpha'],r['ff6_se']):>16s} "
          f"{f(r['sparse5_alpha'],r['sparse5_se']):>16s} "
          f"{f(r['pc_sparse5_alpha'],r['pc_sparse5_se']):>16s}")
    print("-"*86)
    print(f"R² — FF6: {r['ff6_r_squared']:.3f}, Char5: {r['sparse5_r_squared']:.3f}, PC5: {r['pc_sparse5_r_squared']:.3f}")
    print(f"PC var explained (top 5): {r['pc_sparse5_var_explained']*100:.1f}%")
    print(f"Diagnostics: pre={r['n_pre_2005']}, oos={r['n_post_2005']}, "
          f"κ*={r['optimal_kappa']:.3f}, μ_MVE={r['mve_mean_return']*100:.2f}%, "
          f"σ_MVE={r['mve_volatility']*100:.2f}%")
    print("-"*86)



# ------------------ Export Table 4 results to LaTeX (results_export folder) ----------------------
def export_table4_to_latex(r, outdir="results_export", fname="table4_summary.tex"):
    """
    Export Table 4 summary to LaTeX in the same style as coefficients_table_pc.tex.
    """
    import os, pandas as pd
    os.makedirs(outdir, exist_ok=True)

    # Format results into a DataFrame
    df = pd.DataFrame({
        "Benchmark": ["CAPM", "FF6", "Char.-sparse(5)", "PC-sparse(5)"],
        "Alpha": [r["capm_alpha"], r["ff6_alpha"], r["sparse5_alpha"], r["pc_sparse5_alpha"]],
        "s.e.": [r["capm_se"], r["ff6_se"], r["sparse5_se"], r["pc_sparse5_se"]],
        "t-stat": [r["capm_tstat"], r["ff6_tstat"], r["sparse5_tstat"], r["pc_sparse5_tstat"]],
        "R²": [np.nan, r["ff6_r_squared"], r["sparse5_r_squared"], r["pc_sparse5_r_squared"]],
    })

    # Style and round similar to coefficients_table
    df_rounded = df.copy()
    for col in ["Alpha", "s.e.", "t-stat", "R²"]:
        df_rounded[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    tex_path = os.path.join(outdir, fname)
    df_rounded.to_latex(
        tex_path,
        index=False,
        float_format="%.4f",
        column_format="lcccc",
        caption="Out-of-sample MVE $\\alpha$ (2005–2017), annualised. Standard errors in parentheses.",
        label="tab:table4_oos",
    )

    print(f"Saved: {tex_path}")

