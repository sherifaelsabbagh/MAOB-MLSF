"""
build_plec_vs_mfplec_slopechart.py

Slope chart comparing baseline (untuned) NEF1% between PLEC-only and
Morgan+PLEC (MF+PLEC) features, one line per algorithm, following the
visual convention of Caba et al. (2024), Fig. 5 (there used to compare
full vs. dissimilar test sets; here adapted to compare PLEC vs.
Morgan+PLEC feature representations). Two panels: classification and
regression. Uses the manuscript-wide shared plot_style module for
consistent algorithm colors.

Values are hard-coded directly from Table 3 (PLEC baseline) and
Table 4 (MF+PLEC baseline).

Usage:
    python build_plec_vs_mfplec_slopechart.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import clean_axes, PLOT_SETTINGS, COLORS

# (Algorithm, Type): (PLEC baseline, MF+PLEC baseline) -- from Table 3 / Table 4
DATA = {
    ("RF", "Classification"): (0.000, 0.000),
    ("RF", "Regression"):     (0.102, 0.102),
    ("XGB", "Classification"): (0.041, 0.102),
    ("XGB", "Regression"):     (0.061, 0.143),
    ("SVM", "Classification"): (0.122, 0.122),
    ("SVM", "Regression"):     (0.143, 0.184),
    ("ANN", "Classification"): (0.041, 0.020),
    ("ANN", "Regression"):     (0.082, 0.122),
    ("DNN", "Classification"): (0.000, 0.020),
    ("DNN", "Regression"):     (0.143, 0.122),
}

ALGOS = ["RF", "XGB", "SVM", "ANN", "DNN"]
X_POSITIONS = [0, 1]
X_LABELS = ["PLEC", "MF+PLEC"]


def main():
    fig, axes = plt.subplots(1, 2, figsize=PLOT_SETTINGS["figsize_2panel"])

    for ax, model_type in zip(axes, ["Classification", "Regression"]):
        for algo in ALGOS:
            plec_val, mfplec_val = DATA[(algo, model_type)]
            color = COLORS[algo]
            ax.plot(X_POSITIONS, [plec_val, mfplec_val], marker="o", markersize=5,
                    color=color, linewidth=1.8, label=algo)

        ax.set_xticks(X_POSITIONS)
        ax.set_xticklabels(X_LABELS)
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(model_type)
        ax.set_ylabel("NEF1% (baseline)")
        clean_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=5, frameon=False, fontsize=PLOT_SETTINGS["legend_font_size"])

    plt.tight_layout()
    out_path = f"PLEC_vs_MFPLEC_slopechart.{PLOT_SETTINGS['save_format']}"
    plt.savefig(out_path, dpi=PLOT_SETTINGS["dpi"], bbox_inches="tight")
    plt.savefig("PLEC_vs_MFPLEC_slopechart.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_path} and PLEC_vs_MFPLEC_slopechart.png")


if __name__ == "__main__":
    main()
