"""
build_section_a_pr_curve.py

Rebuilds Section A's precision-recall curve figure (generic scoring
function benchmarking on the MAO-B/DUD-E (1S3B) benchmark), using the
real underlying per-molecule score files and the manuscript-wide
shared plot_style module, for visual consistency with all later
figures.

Verified against Section_A_MAOB_Report.docx's Table 3 before use:
recomputing EF1%/NEF1% directly from these four raw score files
reproduces the reported values exactly (RF-Score-VS: 56/168 actives,
NEF1%=0.789; CNN-Score: 14/168, NEF1%=0.197; Smina: 11/168, NEF1%=0.155;
IFP: 2/168, NEF1%=0.028).

Usage:
    python build_section_a_pr_curve.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import plot_pr_curve, clean_axes, PLOT_SETTINGS

# (label, filename, ascending) -- ascending=True for Smina, since a more
# negative (lower) docking score indicates stronger predicted binding;
# the other three scoring functions rank higher score = better.
CONFIGS = [
    ("Smina", "Smina_results.csv", True),
    ("CNN-Score", "CNN_results.csv", False),
    ("RF-Score-VS", "RF_results.csv", False),
    ("IFP", "IFP_results.csv", False),
]


def compute_top1pct_point(df, ascending):
    """Returns (recall, precision) at the top-1% cutoff, matching the
    exact same rank-based logic used for EF1%/NEF1% throughout this
    project (not sklearn's threshold-based precision_recall_curve,
    which can behave differently under heavy score ties)."""
    df_sorted = df.sort_values("Score", ascending=ascending).reset_index(drop=True)
    n_total = len(df_sorted)
    n_top1pct = max(1, round(0.01 * n_total))
    a_total = (df_sorted["Real_Class"] == "Active").sum()
    a_top1pct = (df_sorted.iloc[:n_top1pct]["Real_Class"] == "Active").sum()
    recall = a_top1pct / a_total
    precision = a_top1pct / n_top1pct
    return recall, precision


def main():
    fig, ax = plt.subplots(figsize=PLOT_SETTINGS["figsize_combined"])

    for name, fname, ascending in CONFIGS:
        df = pd.read_csv(fname)
        y_true = (df["Real_Class"] == "Active").astype(int).values
        # sklearn's precision_recall_curve expects higher score = more
        # likely positive; for Smina (lower = better), the score is
        # negated so the curve is computed in the correct direction.
        y_score = -df["Score"].values if ascending else df["Score"].values

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        top1pct_point = compute_top1pct_point(df, ascending)

        plot_pr_curve(ax, recall, precision, name, top1pct_point=top1pct_point)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    #ax.set_title(f"PR curves — {PLOT_SETTINGS['title_suffix']}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    clean_axes(ax)
    ax.legend(frameon=False)

    plt.tight_layout()
    out_path = f"Section_A_PR_curve.{PLOT_SETTINGS['save_format']}"
    plt.savefig(out_path, dpi=PLOT_SETTINGS["dpi"])
    plt.savefig("Section_A_PR_curve.png", dpi=150)  # quick-preview raster copy
    print(f"Saved {out_path} and Section_A_PR_curve.png")


if __name__ == "__main__":
    main()
