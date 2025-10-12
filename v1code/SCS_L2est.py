# -*- coding: utf-8 -*-

"""
SCS_L2est.py — L² (ridge) estimator for Kozak, Nagel & Santosh (2020 JFE)
--------------------------------------------------------------------------
Goal:
    Implement the ridge (L²) estimation of the SDF, perform cross-validation, and produce
    Figures 2a and 4a (model selection and sparsity ridge). Includes paper-style gridlines.

Inputs:
    - dates, re, market, freq: Time series data
    - anomalies: list of portfolio returns or factors
    - parameters: configuration dictionary

Outputs:
    - Optimal L² model coefficients and diagnostics
    - Plots: degrees of freedom, coefficient paths, CV curve
    - Table of top coefficients in 'results_export/'
"""


import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from base_estimator import BaseEstimator
from regcov import regcov
from cross_validate import cross_validate

_asnp = lambda x: x.values if hasattr(x, "values") else np.asarray(x)

# ============================================================================
# Main L2 estimation
# ============================================================================
def L2est(dates, re, market, freq, anomalies, parameters):
    if isinstance(re, pd.DataFrame):
        assert not re.isnull().any().any(), "Missing observations in returns"
    else:
        assert not np.isnan(re).any(), "Missing observations in returns"
    from l2est import l2est

    freq = 252 if parameters.get("daily", True) else 12
    est = BaseEstimator(dates, re, market, freq, anomalies, parameters)
    r_train, r_test, idx_train = est._prepare_data()
    dd = dates[idx_train] if isinstance(dates, pd.Series) else dates[idx_train]

    # === optional PC rotation ===
    Q = None
    if est.p.get("rotate_PC", False):
        print("Rotating returns into PC space...")
        U, S, Vh = np.linalg.svd(regcov(r_train))
        Q = U

        def _rot(d):
            if len(d) == 0:
                return d
            if isinstance(d, pd.DataFrame):
                return pd.DataFrame(
                    d.values @ Q,
                    index=d.index,
                    columns=[f"PC{i+1}" for i in range(Q.shape[1])],
                )
            return d @ Q

        r_train, r_test = _rot(r_train), _rot(r_test)
        est.anomalies = [f"PC{i+1}" for i in range(Q.shape[1])]
        print(f"Rotated to {len(est.anomalies)} principal components")

    # === setup grid ===
    X = regcov(r_train)
    y = _asnp(np.mean(r_train, axis=0)).reshape(-1, 1)
    x, l, lCV = est._setup_grid(X, y, est.p["T"])
    nl = len(l)

    Qsvd, D, _ = np.linalg.svd(X)
    est.p.update({"Q": Qsvd, "d": D, "Xinv": np.linalg.pinv(X)})

    n = est.p["n"]
    phi = np.zeros((n, nl))
    se = np.zeros((n, nl))
    objL2 = np.zeros((nl, 4))
    objL2_folds = np.zeros((nl, est.p["kfold"]))
    MVE = [None] * nl

    print(f"\nEstimating {nl} L2 models from κ={x[0]:.4f} to κ={x[-1]:.4f}")
    for i in range(nl):
        if i % 10 == 0:
            print(f"  Progress: {i+1}/{nl}")
        params = est.p.copy()
        params["L2pen"] = l[i]
        phi[:, i], _, se[:, i] = l2est(X, y, params, True)

        params["L2pen"] = lCV[i]
        objL2[i, :], params, objL2_folds_ = cross_validate(
            l2est, _asnp(dd), r_train, params
        )
        objL2_folds[i, :] = objL2_folds_[:, 1]
        MVE[i] = params.get("cv_MVE", {})

    iopt = est.p["optfunc_np"](objL2[:, 1])
    sr = _sharpe_from_cv(MVE[iopt], freq)
    df = np.sum(D.reshape(-1, 1) / (D.reshape(-1, 1) + l.reshape(1, -1)), axis=0)

    print(f"\nOptimal model: κ={x[iopt]:.4f}, OOS {est.p['sObjective']}={objL2[iopt,1]:.4f}")

    out = est.p.copy()
    out.update(
        {
            "coeffsPaths": phi,
            "df": df,
            "objL2_IS": objL2[:, 0],
            "objL2_OOS": objL2[:, 1],
            "bL2": phi[:, iopt],
            "R2oos": objL2[iopt, 1],
            "optimal_model_L2": {
                "coefficients": phi[:, iopt],
                "se": se[:, iopt],
                "objective": objL2[iopt, 1],
                "dof": df[iopt],
                "kappa": x[iopt],
                "SR": sr,
            },
            "x": x,
            "l": l,
            "lCV": lCV,
            "r_train": r_train,
            "r_test": r_test,
            "dd": dd,
            "anomalies": est.anomalies,
            "rotation_matrix": Q,
            "xlbl": "Root Expected SR$^2$ (prior), $\\kappa$",
            "freq": freq,
        }
    )

    if est.p["plot_dof"]:
        _plot_dof(df, x, out)
    if est.p["plot_coefpaths"]:
        _plot_coefpaths(x, phi, iopt, est.anomalies, "SDF Coefficient, $b$", out)
        _plot_coefpaths(
            x, phi / se, iopt, est.anomalies, "SDF Coefficient $t$-statistic", out
        )
    if est.p["plot_objective"]:
        _plot_cv(x, objL2, out)
    _table_coefs(phi[:, iopt], se[:, iopt], est.anomalies, out)

    return out


# ============================================================================
# Helpers
# ============================================================================
def _sharpe_from_cv(mve_dict, freq):
    if not mve_dict:
        return np.nan
    v = []
    for r in mve_dict.values():
        if r is None:
            continue
        if isinstance(r, (np.ndarray, pd.Series)):
            v.extend(np.ravel(r))
        elif np.isscalar(r):
            v.append(float(r))
    if len(v) > 1:
        v = np.array(v)
        s = np.std(v)
        if s > 0:
            return np.mean(v) / s * np.sqrt(freq)
    return np.nan


def format_func(value, tick_number):
    if value >= 1:
        return f"{value:.0f}"
    elif value >= 0.1:
        return f"{value:.1f}"
    else:
        return f"{value:.2f}"


# ============================================================================
# Improved paper-style gridlines (visible verticals for log κ)
# ============================================================================
def _add_paper_style_grid(ax):
    """
    Add visible vertical dotted gridlines for log-scale x-axis
    (used in Figures 2a and 4a, matching JFE paper style).
    """
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_func))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
    # vertical dotted lines
    ax.grid(True, which="major", axis="x", linestyle=":", color="k", linewidth=0.7, alpha=0.6)
    ax.grid(True, which="minor", axis="x", linestyle=":", color="k", linewidth=0.4, alpha=0.45)
    # light horizontal reference grid
    ax.grid(True, which="major", axis="y", linestyle="-", color="0.85", linewidth=0.5)


# ============================================================================
# Plots
# ============================================================================
def _plot_dof(df, x, p):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x, df, linewidth=1.5)
    if p.get("L2_log_scale", True):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    _add_paper_style_grid(ax)
    ax.set_xlabel(p.get("xlbl", "Kappa"))
    ax.set_ylabel("Effective degrees of freedom")
    _save_show(fig, p, "degrees_of_freedom.png")


def _plot_coefpaths(x, phi, iopt, names, ylab, p):
    fig, ax = plt.subplots(figsize=(5, 5))
    iSort = iopt if p.get("L2_sort_loc", "opt") == "opt" else 0
    maxl = min(p.get("L2_max_legends", 20), phi.shape[0])
    I = np.argsort(-np.abs(phi[:, iSort]))[:maxl]
    for i in I:
        ax.plot(x, phi[i, :], linewidth=1.5)
    if p.get("L2_log_scale", True):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    ax.axvline(x[iopt], color="k", ls="--", alpha=0.7)
    _add_paper_style_grid(ax)
    ax.set_xlabel(p.get("xlbl", "Kappa"))
    ax.set_ylabel(ylab)
    _save_show(fig, p, "tstats_paths.png" if "statistic" in ylab else "coefficients_paths.png")


def _plot_cv(x, obj, p):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x, obj[:, 0], "--", lw=1.5, label="In-sample")
    ax.plot(x, obj[:, 1], "-", lw=1.8, label=f"OOS {p.get('method','CV')}")
    if obj.shape[1] > 3:
        ax.plot(x, obj[:, 1] + obj[:, 3], ":", lw=1.2, color="tab:orange", alpha=0.7, label="±1 s.e.")
        ax.plot(x, obj[:, 1] - obj[:, 3], ":", lw=1.2, color="tab:orange", alpha=0.7)
    if p.get("L2_log_scale", True):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    _add_paper_style_grid(ax)
    ax.set_xlabel(p.get("xlbl", "Kappa"))
    ax.set_ylabel(f"IS/OOS {p.get('sObjective','Objective')}")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim([0, 0.9])
    ax.set_xlim([x.min(), 1.0])
    fname = "figure2a.png" if p.get("dataprovider") == "ff25" else "figure4a.png"
    _save_show(fig, p, fname)


def _table_coefs(phi, se, names, p):
    nrows = p.get("L2_table_rows", 11)
    t = phi / se
    idx = np.argsort(np.abs(t))[::-1][:nrows]
    df = pd.DataFrame(
        {"Portfolio": [names[i] for i in idx], "b": phi[idx], "t_stat": np.abs(t[idx])}
    )
    if p.get("results_export", False):
        os.makedirs("results_export", exist_ok=True)
        suffix = "_pc" if p.get("rotate_PC", False) else "_raw"
        tex_path = f"results_export/coefficients_table{suffix}.tex"
        df.to_latex(tex_path, index=False, float_format="%.4f")
        print(f"Saved: {tex_path}")


def _save_show(fig, p, fname):
    if p.get("results_export", False):
        os.makedirs("results_export", exist_ok=True)
        fig.savefig(f"results_export/{fname}", dpi=300, bbox_inches="tight")
        print(f"Saved: results_export/{fname}")
    plt.show() if p.get("show_plot", False) else plt.close(fig)


# ============================================================================
# Sparsity ridge (Figures 2b / 4b)
# ============================================================================
def plot_sparsity_ridge(est_raw, est_pc, fname="figure2b.png", **kwargs):
    def _as_np(x): return x.values if hasattr(x, "values") else np.asarray(x)
    cv_raw, se_raw, x_raw = map(_as_np, (est_raw["cv_test_L1L2"], est_raw["cv_test_se_L1L2"], est_raw["L1range"]))
    cv_pc, se_pc, x_pc = map(_as_np, (est_pc["cv_test_L1L2"], est_pc["cv_test_se_L1L2"], est_pc["L1range"]))
    idx_raw = np.nanargmax(cv_raw, axis=0)
    ridge_raw, ser_raw = cv_raw[idx_raw, np.arange(cv_raw.shape[1])], se_raw[idx_raw, np.arange(cv_raw.shape[1])]
    idx_pc = np.nanargmax(cv_pc, axis=0)
    ridge_pc, ser_pc = cv_pc[idx_pc, np.arange(cv_pc.shape[1])], se_pc[idx_pc, np.arange(cv_pc.shape[1])]
    m = min(len(x_raw), len(x_pc))
    x_raw, ridge_raw, ser_raw, x_pc, ridge_pc, ser_pc = x_raw[:m], ridge_raw[:m], ser_raw[:m], x_pc[:m], ridge_pc[:m], ser_pc[:m]
    fig, ax = plt.subplots(figsize=(5, 5))
    line_raw, = ax.plot(x_raw, ridge_raw, lw=2, label="Characteristics")
    line_pc, = ax.plot(x_pc, ridge_pc, "--", lw=2, label="PCs")
    ax.plot(x_raw, ridge_raw - ser_raw, ":", lw=1, color=line_raw.get_color())
    ax.plot(x_pc, ridge_pc - ser_pc, ":", lw=1, color=line_pc.get_color())
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    ax.set_xlabel("Number of variables in the SDF")
    ax.set_ylabel("OOS Cross-sectional $R^2$")
    ax.legend()
    _add_paper_style_grid(ax)
    os.makedirs("results_export", exist_ok=True)
    out_path = os.path.join("results_export", fname)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)
    return fig
