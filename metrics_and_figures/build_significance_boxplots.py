"""
build_significance_boxplots.py

Three-panel box plot figure showing the distribution of NEF1% values
for each of this study's three central comparisons (Regression vs.
Classification; PLEC vs. MF+PLEC; Baseline vs. Tuned), matching the
visual convention of Caba et al. (2024), Fig. 6. Each panel is
annotated with its Welch's t-test p-value (see
statistical_significance_tests.py for the full test output).

Values are hard-coded directly from Table 3 and Table 4 (the same 40
NEF1% values used for the significance tests).

Usage:
    python build_significance_boxplots.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import clean_axes, PLOT_SETTINGS

DATA = [
    ("RF", "PLEC", "Classification", "Baseline", 0.000),
    ("RF", "PLEC", "Classification", "Tuned", 0.041),
    ("RF", "PLEC", "Regression", "Baseline", 0.102),
    ("RF", "PLEC", "Regression", "Tuned", 0.102),
    ("XGB", "PLEC", "Classification", "Baseline", 0.041),
    ("XGB", "PLEC", "Classification", "Tuned", 0.041),
    ("XGB", "PLEC", "Regression", "Baseline", 0.061),
    ("XGB", "PLEC", "Regression", "Tuned", 0.122),
    ("SVM", "PLEC", "Classification", "Baseline", 0.122),
    ("SVM", "PLEC", "Classification", "Tuned", 0.041),
    ("SVM", "PLEC", "Regression", "Baseline", 0.143),
    ("SVM", "PLEC", "Regression", "Tuned", 0.143),
    ("ANN", "PLEC", "Classification", "Baseline", 0.041),
    ("ANN", "PLEC", "Classification", "Tuned", 0.041),
    ("ANN", "PLEC", "Regression", "Baseline", 0.082),
    ("ANN", "PLEC", "Regression", "Tuned", 0.122),
    ("DNN", "PLEC", "Classification", "Baseline", 0.000),
    ("DNN", "PLEC", "Classification", "Tuned", 0.020),
    ("DNN", "PLEC", "Regression", "Baseline", 0.143),
    ("DNN", "PLEC", "Regression", "Tuned", 0.061),
    ("RF", "MF+PLEC", "Classification", "Baseline", 0.000),
    ("RF", "MF+PLEC", "Classification", "Tuned", 0.020),
    ("RF", "MF+PLEC", "Regression", "Baseline", 0.102),
    ("RF", "MF+PLEC", "Regression", "Tuned", 0.082),
    ("XGB", "MF+PLEC", "Classification", "Baseline", 0.102),
    ("XGB", "MF+PLEC", "Classification", "Tuned", 0.082),
    ("XGB", "MF+PLEC", "Regression", "Baseline", 0.143),
    ("XGB", "MF+PLEC", "Regression", "Tuned", 0.245),
    ("SVM", "MF+PLEC", "Classification", "Baseline", 0.122),
    ("SVM", "MF+PLEC", "Classification", "Tuned", 0.061),
    ("SVM", "MF+PLEC", "Regression", "Baseline", 0.184),
    ("SVM", "MF+PLEC", "Regression", "Tuned", 0.082),
    ("ANN", "MF+PLEC", "Classification", "Baseline", 0.020),
    ("ANN", "MF+PLEC", "Classification", "Tuned", 0.041),
    ("ANN", "MF+PLEC", "Regression", "Baseline", 0.122),
    ("ANN", "MF+PLEC", "Regression", "Tuned", 0.000),
    ("DNN", "MF+PLEC", "Classification", "Baseline", 0.020),
    ("DNN", "MF+PLEC", "Classification", "Tuned", 0.020),
    ("DNN", "MF+PLEC", "Regression", "Baseline", 0.122),
    ("DNN", "MF+PLEC", "Regression", "Tuned", 0.041),
]

# Precomputed p-values from statistical_significance_tests.py
P_VALUES = {
    "Regression vs.\nClassification": 0.0001,
    "PLEC vs.\nMF+PLEC": 0.695,
    "Baseline vs.\nTuned": 0.464,
}


def p_label(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def main():
    regression_vals = [row[4] for row in DATA if row[2] == "Regression"]
    classification_vals = [row[4] for row in DATA if row[2] == "Classification"]
    plec_vals = [row[4] for row in DATA if row[1] == "PLEC"]
    mfplec_vals = [row[4] for row in DATA if row[1] == "MF+PLEC"]
    baseline_vals = [row[4] for row in DATA if row[3] == "Baseline"]
    tuned_vals = [row[4] for row in DATA if row[3] == "Tuned"]

    panels = [
        ("Regression vs.\nClassification", ["Regression", "Classification"], [regression_vals, classification_vals]),
        ("PLEC vs.\nMF+PLEC", ["PLEC", "MF+PLEC"], [plec_vals, mfplec_vals]),
        ("Baseline vs.\nTuned", ["Baseline", "Tuned"], [baseline_vals, tuned_vals]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(PLOT_SETTINGS["figsize_2panel"][0] * 1.4,
                                              PLOT_SETTINGS["figsize_2panel"][1]))

    for ax, (title, labels, values) in zip(axes, panels):
        bp = ax.boxplot(values, labels=labels, patch_artist=True, widths=0.5,
                         medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], ["#999999", "#0072B2"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for i, vals in enumerate(values, start=1):
            jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, color="black", s=12, alpha=0.5, zorder=3)

        p = P_VALUES[title]
        y_max = max(max(v) for v in values)
        ax.text(1.5, y_max * 1.08, p_label(p), ha="center", fontsize=PLOT_SETTINGS["label_font_size"])

        ax.set_title(title)
        ax.set_ylabel("NEF1%")
        ax.set_ylim(-0.02, y_max * 1.20)
        clean_axes(ax)

    plt.tight_layout()
    out_path = f"significance_boxplots.{PLOT_SETTINGS['save_format']}"
    plt.savefig(out_path, dpi=PLOT_SETTINGS["dpi"], bbox_inches="tight")
    plt.savefig("significance_boxplots.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_path} and significance_boxplots.png")


if __name__ == "__main__":
    main()
