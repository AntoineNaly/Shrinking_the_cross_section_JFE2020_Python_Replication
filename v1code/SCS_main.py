# -*- coding: utf-8 -*-

"""
scs_main.py — Orchestrator for Kozak, Nagel and Santosh (2020, JFE) replication
===========================================================================

Based on Kozak, Nagel, and Santosh (2020) https://www.sciencedirect.com/science/article/abs/pii/S0304405X19301655
This is an adaptation of the authors' Matlab code https://github.com/serhiykozak/SCS

Purpose
-------
Coordinates data loading, estimation, and (optionally) Table 4 OOS evaluation for the
“Shrinking the Cross-Section” replication. This file is deliberately light on logic:
it configures a run, dispatches to `SCS_L2est`/`SCS_L1L2est`, and (for anomaly runs)
invokes the Table-4 pipeline. The design preserves **numerical equivalence** to the
authors’ Matlab implementation; all heavy lifting (estimation, CV, plotting) happens
downstream in the dedicated modules.

Pipeline overview
-----------------
1) Configuration
   • Set global run options in `config`:
       - frequency: `daily=True` (252) or monthly (12)
       - scope:     `dataprovider in {'anom','ff25'}`
       - timing:    `t0`, `tN`, `oos_test_date`, `pure_oos_date`
       - features:  `rotate_PC`, `interactions` (not used for FF25), `withhold_test_sample`
       - output:    `run_folder`, `run_table4`
   • RNG is seeded (`np.random.seed(0)`) for reproducible CV splits and paths.
   • `_initialize_default_parameters` merges sensible defaults (objective, grids, k-fold,
     plotting/export flags) with overrides from `config`.

2) Data & dispatch
   • FF25 portfolios path (`_run_ff25_analysis`):
       - Loads Fama–French 25 portfolios + factors via `load_ff25`.
       - Injects FF-specific params (e.g., SMB/HML series) and forces unconditional
         de-volatilization to match original figures.
       - Calls `L1L2est` (which internally runs `L2est` first) and prints a compact summary.

   • Anomaly portfolios path (`_run_managed_portfolios_analysis`):
       - Locates `Data/instruments/managed_portfolios_anom_{d|m}_*.csv`.
       - Loads managed portfolios via `load_managed_portfolios`.
       - Enables fast L1 truncation and sets `n_jobs=-1` for parallel L1×L2 evaluation.
       - Calls `L1L2est` and prints a compact summary.
       - If `run_table4=True` and not rotating PCs, imports (with reload) and executes
         `table4_oos_analysis.add_table4_to_main_results(p, dd, mkt)`.

3) Reporting
   • `_print_results_summary` emits the optimal L2 and L1–L2 outcomes (objective, κ, dof,
     improvement, bias-corrected objective when available).
   • When Table 4 is run, the Table-4 module prints its own formatted panel and attaches
     results under `p['table4_results']`.

Inputs & directory expectations
-------------------------------
• Working directory must contain:
   - `Data/` with Fama–French CSVs (daily or monthly) and (for anomalies) the input returns.
   - `Data/instruments/managed_portfolios_anom_{suffix}_*.csv` for anomaly runs.
• The script changes into your code folder (see `os.chdir(...)`) and uses the previous cwd
  (captured in `projpath`) to locate `./Data`. Adjust paths if your layout differs.

Key configuration flags (summary)
---------------------------------
• `daily`                  : True=252 freq with `_d` file suffix; False=12 freq (monthly).
• `dataprovider`           : 'anom' (50 anomaly portfolios) or 'ff25'.
• `rotate_PC`              : Rotate returns to PC space (exact Matlab logic in L2est); off for Table 4.
• `interactions`           : Placeholder; requires fundamental data, not used with FF25.
• `withhold_test_sample`   : If True, L2/L1L2 evaluation uses an explicit OOS block (`oos_test_date`).
• `pure_oos_date`          : Table-4 PRE/OOS split (strict PRE fit / OOS apply).
• `run_table4`             : Only applies for `dataprovider='anom'` and `rotate_PC=False`.

Numerical guarantees
--------------------
• This orchestrator does **not** alter estimation math; it only sets flags and calls
  the established routines. 
• Random seed and deterministic iteration order are fixed for replicability.

Outputs
-------
• Console logs:
   - Run configuration, data file used, and brief estimation summaries.
   - If Table 4 is enabled, a compact 4-column panel (CAPM, FF6, Char-sparse(5), PC-sparse(5))
     with annualized alphas, s.e., and diagnostics.
• On `results_export=True` (handled downstream), figures/tables are written to `./results_export/`.
• The `p` dict returned by `L1L2est` is augmented in-place; when Table 4 runs, its results
  are stored as `p['table4_results']`.

Error handling
--------------
• Missing data files raise a `FileNotFoundError` with the expected pattern.
• Table-4 import/execute is wrapped; any exception is printed with a traceback while the
  main estimation results remain available.

Usage
-----
Run as a script (`python scs_main.py`) or import and call `main()`. Adjust `config` (or pass
your own) to switch between FF25 vs anomaly runs, daily vs monthly, and to toggle Table 4.

Notes
-----
• Keep `rotate_PC=False` when running Table 4—KNS evaluate sparse **characteristics** and
  **PC-sparse** benchmarks separately in the OOS stage.
• If you reorganize folders, update the `os.chdir(...)` and the `Data/` paths accordingly.
"""

import os, glob, sys
import numpy as np
from datetime import datetime
from pathlib import Path

# Add local folder (works if run whole script)
os.chdir(os.path.dirname(__file__))
sys.path.insert(0, os.getcwd())
#import local functions
from load_ff25 import load_ff25
from load_managed_portfolios import load_managed_portfolios
from SCS_L1L2est import L1L2est



def main():
    cfg = {
        'dataprovider': 'anom', # 'anom' (50 anomalies) or 'ff25' (Fama French double sorted portfolios on market equity and book-to-market ratio)
        'daily': True, # set to True in original paper
        'interactions': False, # can't run interractions, we need underlying fundamental data
        'withhold_test_sample': False, # set to False in original paper
        'rotate_PC': True, # use True or False
        't0': datetime.strptime('01-Jul-1963','%d-%b-%Y'), # whole sample start date
        'tN': datetime.strptime('31-Dec-2017','%d-%b-%Y'), # whole sample end   date
        'oos_test_date': datetime.strptime('01-Jan-2005','%d-%b-%Y'), # only used if withhold_test_sample set to True
        'pure_oos_date': datetime.strptime('01-Jan-2005','%d-%b-%Y'), # only used in Table 4
        'run_folder': datetime.today().strftime('%d%b%Y').upper()+"/",
        'run_table4': False # use True or False, Table 4 is only run for anomalies since we don't have interactions and WRDS financial ratios
        }


    # Get path to current files
    try:
        V1CODE_DIR = Path(__file__).resolve().parent          # .../repo/v1code
    except NameError:  # e.g., interactive session (Spyder/Jupyter)
        V1CODE_DIR = Path.cwd()
    REPO_ROOT = V1CODE_DIR.parent
    datapath  = REPO_ROOT / "Data"
    instrpath = datapath / "instruments"
    os.chdir(REPO_ROOT)

    freq, suffix = (252,'_d') if cfg['daily'] else (12,'')
    np.random.seed(0)
    p = _init_params(cfg, freq)

    print(f"\nStarting {cfg['dataprovider'].upper()} | Daily={cfg['daily']} | Interactions={cfg['interactions']}")
    if cfg['dataprovider']=='ff25':
        _run_ff25(datapath, freq, p, cfg['interactions'])
    else:
        _run_anom(instrpath, suffix, freq, p, cfg)

    print("\nDone!")

def _init_params(cfg, freq):
    p = {
        'gridsize':100,'contour_levelstep':0.01,'objective':'CSR2','rotate_PC':False,
        'devol_unconditionally':False,'kfold':3,
        'plot_dof':True,'plot_coefpaths':True,'plot_objective':True,'plot_L1L2map':True,
        'results_export':True,'show_plot':False,
        'fig_options':{'fig_sizes':['width=half'],'close_after_print':True},
        'pure_oos_date':cfg['pure_oos_date']
    }
    if cfg.get('interactions'): p['kfold']=2
    if cfg.get('withhold_test_sample'): p['oos_test_date']=cfg['oos_test_date']
    for k in ['rotate_PC','objective','gridsize','kfold','devol_unconditionally','pure_oos_date']:
        if k in cfg: p[k]=cfg[k]
    p['freq']=freq
    return p

def _run_ff25(datapath, freq, p, interactions):
    if interactions:
        print("Skipping FF25 with interactions (not implemented)"); return
    print("\nRunning FF25 estimation...")
    dd,re,mkt,DATA,labels = load_ff25(datapath, p['freq']==252, 0, None)
    p.update({
        'smb':DATA.get('SMB',{}).get('values',None),
        'hml':DATA.get('HML',{}).get('values',None),
        'L2_table_rows':10,'table_L2coefs_posneg_sort': not p['rotate_PC'],
        'table_L2coefs_extra_space': p['rotate_PC'],'L2_sort_loc':'OLS',
        'devol_unconditionally':True,'n_jobs':-1
    })
    p = L1L2est(dd,re,mkt,freq,labels,p); _print_summary(p)

def _run_anom(instrpath, suffix, freq, p, cfg):
    flist = glob.glob(os.path.join(instrpath, f'managed_portfolios_{cfg["dataprovider"]}{suffix}_*.csv'))
    if not flist: raise FileNotFoundError(f"Missing managed_portfolios file in {instrpath}")
    filename = flist[0].strip()
    print(f"\nUsing managed portfolio file: {filename}")
    p.update({'L1_truncPath':True,'devol_unconditionally':False,'n_jobs':-1})
    dd,re,mkt,anoms = load_managed_portfolios(filename, cfg['daily'], 0.2)
    p = L1L2est(dd,re,mkt,freq,anoms,p); _print_summary(p)

    if cfg['dataprovider']=='anom' and not p.get('rotate_PC',False) and cfg.get('run_table4',True):
        try:
            import importlib, sys
            if 'table4_oos_analysis' in sys.modules:
                import table4_oos_analysis; importlib.reload(table4_oos_analysis)
            else:
                import table4_oos_analysis
            table4_oos_analysis.add_table4_to_main_results(p, dd, mkt)
        except ImportError as e:
            print(f"\nTable 4 module not found: {e}")
        except Exception as e:
            print(f"\nTable 4 analysis failed: {e}"); import traceback; traceback.print_exc()

def _print_summary(p):
    print("\n"+"="*60); print("ESTIMATION RESULTS SUMMARY"); print("="*60)
    if 'optimal_model_L2' in p:
        m=p['optimal_model_L2']
        print(f"\nOptimal L2 Model:\n  Objective: {m['objective']:.4f}\n  Kappa: {m['kappa']:.4f}\n  Degrees of Freedom: {m.get('dof','N/A'):.2f}\n  Sharpe Ratio: {m.get('SR','N/A'):.4f}")
    if 'optimal_model_L1L2' in p:
        m=p['optimal_model_L1L2']
        print(f"\nOptimal L1-L2 Model:\n  Total variables: {m['n_total_variables']}\n  Non-zero coefficients: {m['n_nonzero']}\n  Target sparsity: {m['target_sparsity']}\n  Objective: {m['objective']:.4f}\n  Kappa: {m['kappa']:.4f}\n  Improvement over L2: {m['improvement_over_L2']:+.1f}%\n  Bias-corrected objective: {m.get('objective_corrected','N/A'):.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
