# -*- coding: utf-8 -*-

"""
Make README tables and plots
-----------------------------------
This script builds the figures/tables used in the README using the exact settings:
    dataprovider = 'anom'
    daily        = True
    interactions = False
    withhold_test_sample = False
    rotate_PC    = False (for Table 4; contours also shown for rotate_PC=True)
    run_table4   = True

Artifacts produced:
  • Contours (FF25 raw & PC; Anom-50 raw & PC)
  • Table 1 (Anom-50): panel (a) raw, panel (b) PCs — Markdown
  • Table 4 (OOS alphas for 50 anomalies, daily) — Markdown

Numerics: This script *only* orchestrates loaders/estimators you already use.
It does not alter any estimator logic or penalty grids.
"""

import os, glob, sys
import numpy as np
import pandas as pd
from datetime import datetime

# Get current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # root folder
V1_DIR = os.path.join(BASE_DIR, "v1code")
if V1_DIR not in sys.path:
    sys.path.insert(0, V1_DIR)

from load_ff25 import load_ff25
from load_managed_portfolios import load_managed_portfolios
from SCS_L2est import L2est
from SCS_L1L2est import L1L2est, _plot_map
import table4_oos_analysis as t4

# Output folders
OUT_FIGS = "docs/figs"
OUT_DOCS = "docs"
os.makedirs(OUT_FIGS, exist_ok=True)
os.makedirs(OUT_DOCS, exist_ok=True)

# ---------- helpers ----------
def _save_contour_png(tag: str, est: dict):
    """Save L1–L2 contour as PNG using your plot_map."""
    fig = _plot_map(est["x"], est["L1range"], est["cv_test_L1L2"], est)
    fig.savefig(f"{OUT_FIGS}/{tag}.png", dpi=300, bbox_inches="tight")
    print(f"saved {OUT_FIGS}/{tag}.png")

def _table1_from_L2(est: dict, top: int = 10) -> pd.DataFrame:
    """Build Table 1 panel from an L2 result (top |t|-stats)."""
    phi = np.asarray(est["optimal_model_L2"]["coefficients"]).reshape(-1)
    se  = np.asarray(est["optimal_model_L2"]["se"]).reshape(-1)
    t   = np.abs(phi / se)
    idx = np.argsort(-t)[:top]
    return pd.DataFrame({
        "Portfolio": [est["anomalies"][i] for i in idx],
        "b": phi[idx],
        "t_stat": t[idx]
    })

def _write_md_table(path: str, df: pd.DataFrame, floatfmt: str = ".2f"):
    """Write a small Markdown table (no external deps)."""
    # headers
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    # rows
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:{floatfmt}}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"saved {path}")

# ---------- FF25 (contours only) ----------
def run_ff25(daily: bool = True):
    print("\n== FF25 (contours) ==")
    freq = 252 if daily else 12

    dd, re, mkt, DATA, labels = load_ff25("Data", daily, 0, pd.Timestamp("2017-12-31"))

    base = dict(
        gridsize=100, kfold=3, objective="CSR2",
        plot_dof=False, plot_coefpaths=False, plot_objective=False, plot_L1L2map=False,
        results_export=False, show_plot=False, n_jobs=-1
    )

    # raw
    est_raw = L1L2est(dd, re, mkt, freq, labels, {**base, "rotate_PC": False})
    _save_contour_png("ff25_L1L2_contour_raw", est_raw)

    # PCs
    est_pc = L1L2est(dd, re, mkt, freq, labels, {**base, "rotate_PC": True})
    _save_contour_png("ff25_L1L2_contour_pc", est_pc)

# ---------- Anom-50 (contours, Table 1, Table 4) ----------
def run_anom_50(daily: bool = True):
    print("\n== 50 anomalies (daily) ==")
    assert daily, "Table 4 in README uses daily anomalies only."
    freq   = 252
    suffix = "_d"
    instrpath = os.path.join(os.getcwd(), "Data", "instruments")
    fmask  = os.path.join(instrpath, f"managed_portfolios_anom{suffix}_*.csv")
    flist  = sorted(glob.glob(fmask))
    if not flist:
        raise FileNotFoundError(f"No anomaly file found matching: {fmask}")
    filename = flist[0]
    print(f"Using: {filename}")

    dd, re, mkt, labels = load_managed_portfolios(filename, daily, 0.2)

    base = dict(
        gridsize=100, kfold=3, objective="CSR2",
        plot_dof=False, plot_coefpaths=False, plot_objective=False, plot_L1L2map=False,
        results_export=False, show_plot=False, n_jobs=-1,
        L1_truncPath=True, devol_unconditionally=False
    )

    # ---- Table 1 (panel a) — raw anomalies via L2 ----
    est_L2_raw = L2est(dd, re, mkt, freq, labels, {**base, "rotate_PC": False})
    df_a = _table1_from_L2(est_L2_raw, top=11)
    _write_md_table(f"{OUT_DOCS}/Table1a_anom_raw.md", df_a, ".2f")

    # ---- Contour (raw anomalies) ----
    est_L1L2_raw = L1L2est(dd, re, mkt, freq, labels, {**base, "rotate_PC": False})
    _save_contour_png("anom50_L1L2_contour_raw", est_L1L2_raw)

    # ---- PCs: contour + Table 1 (panel b) ----
    est_L1L2_pc = L1L2est(dd, re, mkt, freq, labels, {**base, "rotate_PC": True})
    _save_contour_png("anom50_L1L2_contour_pc", est_L1L2_pc)

    est_L2_pc = L2est(dd, re, mkt, freq, labels, {**base, "rotate_PC": True})
    df_b = _table1_from_L2(est_L2_pc, top=11)
    _write_md_table(f"{OUT_DOCS}/Table1b_anom_pc.md", df_b, ".2f")

    # ---- Table 4 (strict settings) ----
    p_t4 = dict(
        freq=freq,
        pure_oos_date=datetime(2005, 1, 1),  # strict split
        daily=True,
        run_table4=True,
        rotate_PC=False,                     # Table 4 SDF estimated in anomaly space
        # keep everything else default; the helper uses p['r_train'], p['dd'], p['anomalies']
        r_train=re, dd=dd, anomalies=labels
    )

    # Run and capture results
    t4.add_table4_to_main_results(p_t4, dd, mkt)
    r = p_t4["table4_results"]

    def _fmt(v):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "—"
        return f"{100.0 * v:6.2f}"

    md = []
    md.append("| SDF factors              | CAPM (α, %) | FF6 (α, %) | Char.-sparse (α, %) | PC-sparse (α, %) |")
    md.append("|--------------------------|------------:|-----------:|--------------------:|-----------------:|")
    md.append(f"| 50 anomaly portfolios    | {_fmt(r['capm_alpha'])} | {_fmt(r['ff6_alpha'])} | {_fmt(r['sparse5_alpha'])} | {_fmt(r['pc_sparse5_alpha'])} |")
    md.append(f"| (s.e.)                   | ({_fmt(r['capm_se'])}) | ({_fmt(r['ff6_se'])}) | ({_fmt(r['sparse5_se'])}) | ({_fmt(r['pc_sparse5_se'])}) |")
    with open(f"{OUT_DOCS}/Table4_OOS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"saved {OUT_DOCS}/Table4_OOS.md")


if __name__ == "__main__":
    run_ff25(daily=True)
    run_anom_50(daily=True)
