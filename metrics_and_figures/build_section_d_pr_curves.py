"""
build_section_d_pr_curves.py

Builds the 2x2 panel PR curve figure for Section D: rows = Baseline /
Tuned, columns = Classification / Regression. Each panel shows all
five target-specific algorithms (using each algorithm's median-
representative run) plus the three generic scoring functions
(identical across all four panels, since generic SFs have no
"tuned" version), using the manuscript-wide shared plot_style module.

Requires plot_style.py in the same directory, plus:
    - 20 target-specific prediction .npy files (5 algorithms x
      {baseline, tuned} x {classification, regression})
    - y_test_active.npy (true labels, shared across all panels)
    - test_smina.csv, test_CNN.csv, test_RF.csv (generic SF scores on
      the true test set, ID/Score/Real_Class format)

Usage:
    python build_section_d_pr_curves.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import plot_pr_curve, clean_axes, PLOT_SETTINGS

# ----------------------------------------------------------------------
# File mapping -- exact filenames as they exist on disk (confirmed,
# not guessed; note the inconsistent capitalization between baseline
# and tuned scripts, carried over as-is rather than renamed).
# ----------------------------------------------------------------------
PREDICTION_FILES = {
    ("Baseline", "Regression"): {
        "RF": "RF_median_run_predictions.npy",
        "XGB": "XGB_median_run_predictions.npy",
        "SVM": "SVM_median_run_predictions.npy",
        "ANN": "ANN_median_run_predictions.npy",
        "DNN": "DNN_median_run_predictions.npy",
    },
    ("Baseline", "Classification"): {
        "RF": "RF_classification_median_predictions.npy",
        "XGB": "XGB_classification_median_predictions.npy",
        "SVM": "SVM_classification_median_predictions.npy",
        "ANN": "ANN_classification_median_predictions.npy",
        "DNN": "DNN_classification_median_predictions.npy",
    },
    ("Tuned", "Regression"): {
        "RF": "RF_tuned_median_predictions.npy",
        "XGB": "xgb_tuned_median_predictions.npy",
        "SVM": "svm_tuned_median_predictions.npy",
        "ANN": "ann_tuned_median_predictions.npy",
        "DNN": "dnn_tuned_median_predictions.npy",
    },
    ("Tuned", "Classification"): {
        "RF": "rf_classification_tuned_median_predictions.npy",
        "XGB": "xgb_classification_tuned_median_predictions.npy",
        "SVM": "svm_classification_tuned_median_predictions.npy",
        "ANN": "ann_classification_tuned_median_predictions.npy",
        "DNN": "dnn_classification_tuned_median_predictions.npy",
    },
}

# Generic SF score files on the true test set. Column names assumed to
# match the ID/Score/Real_Class format used throughout this project
# (Section A's Smina_results.csv/CNN_results.csv/RF_results.csv) --
# VERIFY these three files use the same column names before running;
# adjust the column names in load_generic_sf() below if they differ.
GENERIC_SF_FILES = {
    "Smina": ("test_smina.csv", True),       # ascending=True: lower Smina score = better
    "CNN-Score": ("test_CNN.csv", False),
    "RF-Score-VS": ("test_RF.csv", False),
}

ROWS = ["Baseline", "Tuned"]
COLS = ["Classification", "Regression"]  # left-to-right, per the agreed layout


def load_generic_sf(fname, ascending):
    df = pd.read_csv(fname)
    y_true = (df["Real_Class"] == "Active").astype(int).values
    y_score = -df["Score"].values if ascending else df["Score"].values
    return y_true, y_score


def main():
    y_test_active = np.load("y_test_active.npy")

    fig, axes = plt.subplots(2, 2, figsize=(PLOT_SETTINGS["figsize_2panel"][0],
                                              PLOT_SETTINGS["figsize_2panel"][0]))

    for row_idx, row_name in enumerate(ROWS):
        for col_idx, col_name in enumerate(COLS):
            ax = axes[row_idx, col_idx]

            # Target-specific algorithms for this panel
            for algo, fname in PREDICTION_FILES[(row_name, col_name)].items():
                y_score = np.load(fname)
                precision, recall, _ = precision_recall_curve(y_test_active, y_score)
                plot_pr_curve(ax, recall, precision, algo)

            # Generic SFs: identical in every panel
            for sf_name, (fname, ascending) in GENERIC_SF_FILES.items():
                y_true_sf, y_score_sf = load_generic_sf(fname, ascending)
                precision, recall, _ = precision_recall_curve(y_true_sf, y_score_sf)
                plot_pr_curve(ax, recall, precision, sf_name)

            clean_axes(ax)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)

            if row_idx == 0:
                ax.set_title(col_name)
            if col_idx == 0:
                ax.set_ylabel(f"{row_name}\n\nPrecision")
            else:
                ax.set_ylabel("Precision")
            if row_idx == 1:
                ax.set_xlabel("Recall")

    # Single shared legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=4, frameon=False, fontsize=PLOT_SETTINGS["legend_font_size"])

    plt.tight_layout()
    out_path = f"Section_D_PR_curves_2x2.{PLOT_SETTINGS['save_format']}"
    plt.savefig(out_path, dpi=PLOT_SETTINGS["dpi"], bbox_inches="tight")
    plt.savefig("Section_D_PR_curves_2x2.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_path} and Section_D_PR_curves_2x2.png")


if __name__ == "__main__":
    main()
