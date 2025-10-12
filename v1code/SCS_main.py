# -*- coding: utf-8 -*-

"""
SCS_main.py — Main orchestration script for Kozak, Nagel & Santosh (2020 JFE) replication
----------------------------------------------------------------------------------------
Goal:
    Coordinate the full replication pipeline for Fama–French (FF25) and 50 anomaly portfolios,
    including L² and L1–L² (Elastic Net) estimation, plotting, and optional Table 4 analysis.

Inputs:
    cfg (dict, optional): Configuration dictionary specifying data provider ('ff25' or 'anom'),
                          frequency (daily/monthly), plotting flags, and OOS test dates.

Outputs:
    - Figures 1–4 in 'results_export/'
    - Table 4 replication summary (if run_table4=True)
    - Console summaries of model fit and sparsity
"""


import os, glob, sys
import numpy as np
from datetime import datetime
from pathlib import Path

# Add local folder (works if run whole script)
os.chdir(os.path.dirname(__file__))
sys.path.insert(0, os.getcwd())

# import local functions
from load_ff25 import load_ff25
from load_managed_portfolios import load_managed_portfolios
from SCS_L1L2est import L1L2est
from SCS_L2est import plot_sparsity_ridge
from table4_oos_analysis import add_table4_to_main_results


def main(cfg=None):
    if cfg is None:
        cfg = {
            'dataprovider': 'anom',       # 'anom' (50 anomalies from Kozak et al.) or 'ff25' (Fama-French 25 portfolios)
            'daily': True,                # set to True in paper
            'interactions': False,        # can't run interactions, we would need fundamental data
            'withhold_test_sample': False,
            'oos_test_date': datetime.strptime('01-Jan-2005','%d-%b-%Y'),
            'pure_oos_date': datetime.strptime('01-Jan-2005','%d-%b-%Y'),
            'run_table4': True,  # run only for anomalies in paper
            'results_export': True,
            'show_plot': False
        }

    try:
        V1CODE_DIR = Path(__file__).resolve().parent
    except NameError:
        V1CODE_DIR = Path.cwd()

    REPO_ROOT = V1CODE_DIR.parent
    datapath  = REPO_ROOT / "Data"
    instrpath = datapath / "instruments"
    os.chdir(REPO_ROOT)

    freq, suffix = (252, '_d') if cfg['daily'] else (12, '')
    np.random.seed(0)
    p = _init_params(cfg, freq)

    print(f"\nStarting {cfg['dataprovider'].upper()} | Daily={cfg['daily']} | Interactions={cfg['interactions']}")
    
    if cfg['dataprovider'] == 'ff25':
        results = _run_ff25(datapath, freq, p, cfg)
    else:
        results = _run_anom(instrpath, suffix, freq, p, cfg)

    # === Optional Table 4 (2005–2017 OOS) ===
    if cfg.get('run_table4', False):
        market_series = results.get('market', None)
        if market_series is None:
            print("[WARN] No market series found — cannot run Table 4.")
        else:
            print("\n[INFO] Running Table 4 (2005–2017 OOS) replication...")
            try:
                add_table4_to_main_results(results, results['dd'], market_series)
            except Exception as e:
                print(f"[ERROR] Table 4 analysis failed: {e}")

    print("\nDone!")
    return cfg


def _init_params(cfg, freq):
    p = {
        'gridsize': 100, 'contour_levelstep': 0.01, 'objective': 'CSR2', 'rotate_PC': False,
        'devol_unconditionally': False, 'kfold': 3,
        'plot_dof': True, 'plot_coefpaths': True, 'plot_objective': True, 'plot_L1L2map': True,
        'results_export': cfg.get('results_export', True),
        'show_plot': cfg.get('show_plot', False),
        'fig_options': {'fig_sizes': ['width=half'], 'close_after_print': True},
        'pure_oos_date': cfg['pure_oos_date']
    }
    if cfg.get('interactions'): 
        p['kfold'] = 2
    if cfg.get('withhold_test_sample'): 
        p['oos_test_date'] = cfg['oos_test_date']
    p['freq'] = freq
    return p


def _run_ff25(datapath, freq, p, cfg):
    """FF25 pipeline (Figs. 1a–1b & 2b)"""
    print("\n=== Running FF25 portfolios ===")

    cfg["t0"], cfg["tN"] = datetime(1926,7,1), datetime(2017,12,31)
    dates, re, market, DATA, labels = load_ff25(
        datapath=datapath,
        daily=cfg["daily"],
        t0=cfg["t0"], tN=cfg["tN"],
    )
    print(f"[FF25] range: {dates.min()} → {dates.max()} | T={len(dates)}")

    p_ff = dict(p)
    p_ff["devol_unconditionally"] = True
    p_ff["demarket_unconditionally"] = True
    p_ff.setdefault("n_jobs", -1)
    p_ff["dataprovider"] = "ff25"

    # --- No rotation (Figure 1a)
    est_raw = L1L2est(dates, re, market, freq, labels, p_ff)
    _print_summary(est_raw)

    # --- With rotation (Figure 1b)
    p_pc = dict(p_ff)
    p_pc["rotate_PC"] = True
    est_pc = L1L2est(dates, re, market, freq, labels, p_pc)
    _print_summary(est_pc)

    # --- Plot both versions (Figure 2b)
    plot_sparsity_ridge(est_raw, est_pc, fname="figure2b.png", freq=freq)

    # Attach market for Table 4
    est_raw["market"] = market
    return est_raw


def _run_anom(instrpath, suffix, freq, p, cfg):
    """50 anomalies pipeline (Figs. 3a–3b & 4b)"""
    print("\n=== Running 50 anomalies portfolios ===")

    cfg["t0"], cfg["tN"] = datetime(1963,7,1), datetime(2017,12,31)
    pattern = os.path.join(instrpath, f"managed_portfolios_{cfg['dataprovider']}{suffix}_*.csv")
    flist = glob.glob(pattern)
    if not flist:
        raise FileNotFoundError(f"Missing managed_portfolios file matching: {pattern}")
    filename = sorted(flist)[0]
    print(f"[INFO] Using managed portfolio file: {filename}")

    dd, re, mkt, anoms = load_managed_portfolios(filename, cfg["daily"], 0.2)

    p_anom = dict(p)
    p_anom["dataprovider"] = "anom"

    # --- No rotation (Figure 3a)
    est_raw = L1L2est(dd, re, mkt, freq, anoms, p_anom)
    _print_summary(est_raw)

    # --- With rotation (Figure 3b)
    p_pc = dict(p_anom)
    p_pc["rotate_PC"] = True
    est_pc = L1L2est(dd, re, mkt, freq, anoms, p_pc)
    _print_summary(est_pc)

    # --- Plot both versions (Figure 4b)
    plot_sparsity_ridge(est_raw, est_pc, fname="figure4b.png", freq=freq)

    est_raw["market"] = mkt
    return est_raw



def _print_summary(p):
    print("\n" + "="*60)
    print("ESTIMATION RESULTS SUMMARY")
    print("="*60)
    if 'optimal_model_L2' in p:
        m = p['optimal_model_L2']
        print(f"\nOptimal L2 Model:\n  Objective: {m['objective']:.4f}\n  Kappa: {m['kappa']:.4f}\n  DOF: {m.get('dof','N/A'):.2f}\n  SR: {m.get('SR','N/A'):.4f}")
    if 'optimal_model_L1L2' in p:
        m = p['optimal_model_L1L2']
        print(f"\nOptimal L1-L2 Model:\n  Total vars: {m['n_total_variables']}\n  Non-zero: {m['n_nonzero']}\n  Sparsity: {m['target_sparsity']}\n  Objective: {m['objective']:.4f}\n  Kappa: {m['kappa']:.4f}\n  ΔL2: {m['improvement_over_L2']:+.1f}%\n  Corrected: {m.get('objective_corrected','N/A'):.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
