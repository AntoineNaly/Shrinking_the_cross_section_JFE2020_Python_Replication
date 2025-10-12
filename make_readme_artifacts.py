# -*- coding: utf-8 -*-

"""
make_readme_artifacts.py — Generate paper-style figures & tables
for Kozak, Nagel & Santosh (2020 JFE) replication

Outputs (for README):
  /paper_figures/
      figure1_combined.png
      figure2_combined.png
      figure3_combined.png
      figure4_combined.png
      Table1_combined.md
      table4_summary.md
"""

import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent
V1_DIR      = REPO_ROOT / "v1code"
RESULTS_EXP = REPO_ROOT / "results_export"
PAPER_FIGS  = REPO_ROOT / "paper_figures"

os.makedirs(PAPER_FIGS, exist_ok=True)
if str(V1_DIR) not in sys.path:
    sys.path.insert(0, str(V1_DIR))

from SCS_main import main as run_main

# ---------------------------------------------------------------------------
# Markdown helper
# ---------------------------------------------------------------------------
def _write_md_table(path: Path, df: pd.DataFrame, floatfmt: str = ".2f"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        cells = [f"{v:{floatfmt}}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# Combine panels into paper-style figures
# ---------------------------------------------------------------------------
def rename_contours():
    mapping = {
        "ff25": {"elasticnet_contour_raw.png": "figure1a.png", "elasticnet_contour_pc.png": "figure1b.png"},
        "anom": {"elasticnet_contour_raw.png": "figure3a.png", "elasticnet_contour_pc.png": "figure3b.png"},
    }
    for maps in mapping.values():
        for src, dst in maps.items():
            p = RESULTS_EXP / src
            if p.exists():
                p.rename(RESULTS_EXP / dst)
                print(f"Renamed {src} → {dst}")

def combine_two_panels(left, right, title, subtitles, outfile):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, sub in zip(axes, [left, right], subtitles):
        ax.imshow(Image.open(img))
        ax.axis("off")
        ax.set_title(sub, fontsize=11, y=-0.15)
    fig.suptitle(title, fontsize=13, fontweight="normal", y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = PAPER_FIGS / outfile
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

def combine_all_figures():
    rename_contours()
    combine_two_panels(RESULTS_EXP/"figure1a.png", RESULTS_EXP/"figure1b.png",
        "Figure 1: OOS $R^2$ (Fama–French 25 portfolios)",
        ["(a) Raw portfolios", "(b) PCs of portfolios"], "figure1_combined.png")
    combine_two_panels(RESULTS_EXP/"figure2a.png", RESULTS_EXP/"figure2b.png",
        "Figure 2: L2 Model Selection and Sparsity (Fama–French 25 portfolios)",
        ["(a) L2 model selection", "(b) Sparsity"], "figure2_combined.png")
    combine_two_panels(RESULTS_EXP/"figure3a.png", RESULTS_EXP/"figure3b.png",
        "Figure 3: OOS $R^2$ (50 anomaly portfolios)",
        ["(a) Raw anomalies", "(b) PCs of anomalies"], "figure3_combined.png")
    combine_two_panels(RESULTS_EXP/"figure4a.png", RESULTS_EXP/"figure4b.png",
        "Figure 4: L2 Model Selection and Sparsity (50 anomaly portfolios)",
        ["(a) L2 model selection", "(b) Sparsity"], "figure4_combined.png")

# ---------------------------------------------------------------------------
# Table 1 (Markdown-only)
# ---------------------------------------------------------------------------
def combine_table1():
    raw, pc = RESULTS_EXP / "coefficients_table_raw.tex", RESULTS_EXP / "coefficients_table_pc.tex"
    if not (raw.exists() and pc.exists()):
        print("[WARN] Missing coefficient tables; skipping Table 1.")
        return

    def _read_tex(path):
        lines = [l.strip() for l in open(path, encoding="utf-8") if "&" in l]
        data = [[x.strip().replace("\\\\", "") for x in l.split("&")] for l in lines]
        df = pd.DataFrame(data[1:], columns=[c.strip() for c in data[0]])
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    df_raw, df_pc = _read_tex(raw), _read_tex(pc)
    df_raw.columns, df_pc.columns = ["Portfolio", "b", "t"], ["Portfolio", "b", "t"]

    md_path = PAPER_FIGS / "Table1_combined.md"
    md_lines = [
        "### Table 1 – Largest SDF factors (50 anomaly portfolios)\n",
        "_Coefficient estimates and absolute t-statistics at the optimal prior root expected SR² (cross-validated)._",
        "\n\n**(a) Raw 50 anomaly portfolios**\n",
        "| Portfolio | b | t |", "| --- | --- | --- |",
    ]
    for _, r in df_raw.iterrows():
        md_lines.append(f"| {r['Portfolio']} | {r['b']:.3f} | {r['t']:.3f} |")
    md_lines += [
        "\n**(b) PCs of 50 anomaly portfolios**\n",
        "| Portfolio | b | t |", "| --- | --- | --- |",
    ]
    for _, r in df_pc.iterrows():
        md_lines.append(f"| {r['Portfolio']} | {r['b']:.3f} | {r['t']:.3f} |")
    md_lines.append("\n_Coefficients are sorted by absolute t-statistic values._")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"✅ Saved GitHub-style Table 1 → {md_path}")

# ---------------------------------------------------------------------------
# Table 4 → Markdown only
# ---------------------------------------------------------------------------
def move_table4_to_paper_figures():
    """
    Convert table4_summary.tex → Markdown and save in paper_figures/.
    """
    tex_src = RESULTS_EXP / "table4_summary.tex"
    if not tex_src.exists():
        print("[WARN] table4_summary.tex not found in results_export/")
        return

    try:
        lines = [l.strip() for l in tex_src.read_text(encoding="utf-8").splitlines() if "&" in l]
        data = [[x.strip().replace('\\', '') for x in l.split('&')] for l in lines]
        df = pd.DataFrame(data[1:], columns=[c.strip() for c in data[0]])
        md_path = PAPER_FIGS / "table4_summary.md"
        _write_md_table(md_path, df, floatfmt=".3f")
        print(f"✅ Saved Table 4 Markdown → {md_path}")
    except Exception as e:
        print(f"[WARN] Could not create Markdown version of Table 4: {e}")

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_all_artifacts():
    print("\n=== Building paper-style figures and tables ===")
    base = dict(
        daily=True, interactions=False, withhold_test_sample=False,
        results_export=True, show_plot=False,
        oos_test_date=datetime(2005,1,1), pure_oos_date=datetime(2005,1,1)
    )

    print("\n[FF25 portfolios → Figures 1–2]")
    run_main(dict(base, dataprovider="ff25", run_table4=False))

    print("\n[50 anomalies → Figures 3–4 + Tables 1 and 4]")
    run_main(dict(base, dataprovider="anom", run_table4=True))

    combine_all_figures()
    combine_table1()
    move_table4_to_paper_figures()

    print(f"\n✅ All figures and Markdown tables saved in {PAPER_FIGS}/")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_all_artifacts()
