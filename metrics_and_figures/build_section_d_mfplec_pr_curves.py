"""
build_section_d_mfplec_pr_curves.py

Builds the 2x2 panel PR curve figure for Section 3.4 (Morgan+PLEC
features): rows = Baseline / Tuned, columns = Classification /
Regression. Each panel shows all five target-specific algorithms
(using each algorithm's median-representative run) plus the three
generic scoring functions (identical across all four panels, since
generic SFs have no "tuned" version and do not use Morgan+PLEC
features at all), using the manuscript-wide shared plot_style module.

This is the Morgan+PLEC counterpart to build_section_d_pr_curves.py
(the PLEC-only figure) -- same structure, same styling, different
underlying prediction files and (likely) a different y-axis range,
given this study's Morgan+PLEC results reach considerably higher
NEF1% values (up to 0.347) than the PLEC-only results (up to 0.143).

Suggested caption:
    Figure X. Precision-recall curves for target-specific MAO-B
    scoring functions (dashed lines: RF, XGB, SVM, ANN, DNN) using
    combined Morgan fingerprint and PLEC (MF+PLEC) features, and
    generic scoring functions (solid lines: Smina, CNN-Score,
    RF-Score-VS) on the true-inactive test set (117 actives, 4,828
    experimentally confirmed inactive compounds). Panels show baseline
    and hyperparameter-tuned configurations for classification (left)
    and regression (right) models. The random baseline (NEF1% = 0.024)
    is shown as a horizontal dashed grey line. Each curve corresponds
    to the run giving the median NEF1% across five repeats (see
    Section 3.4 for a note on the ANN regression, tuned configuration,
    where one of five initial repeats was a low-performing outlier).

Requires plot_style.py in the same directory, plus:
    - 20 target-specific prediction .npy files (5 algorithms x
      {baseline, tuned} x {classification, regression}), Morgan+PLEC
    - y_test_active.npy (true labels, shared across all panels --
      identical file to the one used for the PLEC-only figure, since
      the test set itself is unchanged)
    - test_smina.csv, test_CNN.csv, test_RF.csv (generic SF scores on
      the true test set, ID/Score/Real_Class format -- identical files
      to the PLEC-only figure, since generic SFs do not use Morgan+PLEC)

Usage:
    python build_section_d_mfplec_pr_curves.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import plot_pr_curve, clean_axes, PLOT_SETTINGS

RANDOM_BASELINE_NEF1 = 0.024  # 1 / maximal EF1% (41.88) -- see Table 3
Y_AXIS_MAX = 0.40  # wider than the PLEC-only figure's 0.25, given this
                    # study's Morgan+PLEC results reach up to NEF1% = 0.347

# ----------------------------------------------------------------------
# File mapping -- exact filenames as they exist on disk (confirmed,
# not guessed; note the inconsistent capitalization between baseline
# and tuned scripts, carried over as-is rather than renamed).
# ----------------------------------------------------------------------
PREDICTION_FILES = {
    ("Baseline", "Regression"): {
        "RF": "RF_mfplec_median_predictions.npy",
        "XGB": "XGB_mfplec_median_predictions.npy",
        "SVM": "SVM_mfplec_median_predictions.npy",
        "ANN": "ANN_mfplec_median_predictions.npy",
        "DNN": "DNN_mfplec_median_predictions.npy",
    },
    ("Baseline", "Classification"): {
        "RF": "RF_classification_mfplec_median_predictions.npy",
        "XGB": "XGB_classification_mfplec_median_predictions.npy",
        "SVM": "SVM_classification_mfplec_median_predictions.npy",
        "ANN": "ANN_classification_mfplec_median_predictions.npy",
        "DNN": "DNN_classification_mfplec_median_predictions.npy",
    },
    ("Tuned", "Regression"): {
        "RF": "RF_regression_mfplec_tuned_median_predictions.npy",
        "XGB": "xgb_regression_mfplec_tuned_median_predictions.npy",
        "SVM": "svm_mfplec_tuned_median_predictions.npy",
        "ANN": "ann_regression_mfplec_tuned_median_predictions.npy",
        "DNN": "dnn_regression_mfplec_tuned_median_predictions.npy",
    },
    ("Tuned", "Classification"): {
        "RF": "rf_classification_mfplec_tuned_median_predictions.npy",
        "XGB": "xgb_classification_mfplec_tuned_median_predictions.npy",
        "SVM": "svm_classification_mfplec_tuned_median_predictions.npy",
        "ANN": "ann_classification_mfplec_tuned_median_predictions.npy",
        "DNN": "dnn_classification_mfplec_tuned_median_predictions.npy",
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
            ax.set_ylim(-0.005, Y_AXIS_MAX)
            ax.axhline(RANDOM_BASELINE_NEF1, color="grey", linestyle="--", linewidth=1.0, zorder=1,
                       label="Random expectation" if (row_idx == 0 and col_idx == 0) else None)

            if row_idx == 0:
                ax.set_title(col_name)
            if col_idx == 0:
                ax.set_ylabel(f"{row_name}\n\nPrecision")
            else:
                ax.set_ylabel("Precision")
            if row_idx == 1:
                ax.set_xlabel("Recall")

    # Single shared legend, grouped: target-specific algorithms first,
    # then generic scoring functions, then the random-baseline line.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    target_specific_order = ["RF", "XGB", "SVM", "ANN", "DNN"]
    generic_order = ["Smina", "CNN-Score", "RF-Score-VS"]
    ordered_labels = target_specific_order + generic_order + ["Random expectation"]
    label_to_handle = dict(zip(labels, handles))
    ordered_handles = [label_to_handle[l] for l in ordered_labels if l in label_to_handle]
    ordered_labels_present = [l for l in ordered_labels if l in label_to_handle]

    fig.legend(ordered_handles, ordered_labels_present, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=5, frameon=False, fontsize=PLOT_SETTINGS["legend_font_size"])

    plt.tight_layout()
    out_path = f"Section_D_MFPLEC_PR_curves_2x2.{PLOT_SETTINGS['save_format']}"
    plt.savefig(out_path, dpi=PLOT_SETTINGS["dpi"], bbox_inches="tight")
    plt.savefig("Section_D_MFPLEC_PR_curves_2x2.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_path} and Section_D_MFPLEC_PR_curves_2x2.png")


if __name__ == "__main__":
    main()
