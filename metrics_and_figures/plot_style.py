"""
plot_style.py

Unified, manuscript-wide plotting style for all figures in this
project (Section A generic-SF benchmarking, Section D target-specific
model comparisons). Every figure-generating script should import from
this module rather than defining its own PLOT_SETTINGS, colors, or
line styles, to guarantee visual consistency across the whole
manuscript (a given scoring function or algorithm always has the same
color and line style in every figure it appears in).

Design principles:
    - Colorblind-safe Okabe-Ito palette throughout.
    - GENERIC scoring functions (Smina, CNN-Score, RF-Score-VS, IFP)
      are always drawn as SOLID lines.
    - TARGET-SPECIFIC ML algorithms (RF, XGB, SVM, ANN, DNN) are always
      drawn as DASHED lines.
    - This solid-vs-dashed convention matches the one used in Caba
      et al. 2024 (PARP1 ML-SF paper, Fig. 3/4), and provides a second,
      redundant visual encoding of the generic/target-specific
      distinction beyond color alone -- useful for grayscale printing
      and for readers with color vision deficiency who may still
      struggle with some palette entries despite Okabe-Ito's design.
"""

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Master color + line-style mapping -- every entity that could appear
# in ANY figure across the manuscript is defined here ONCE.
# ----------------------------------------------------------------------
COLORS = {
    # Generic scoring functions (Section A + Section D baseline comparisons)
    "Smina":        "#0072B2",  # blue
    "CNN-Score":    "#E69F00",  # orange
    "RF-Score-VS":  "#009E73",  # green
    "IFP":          "#D55E00",  # vermillion (Section A only)

    # Target-specific ML algorithms (Section D)
    "RF":           "#CC79A7",  # pink
    "XGB":          "#56B4E9",  # sky blue
    "SVM":          "#B8860B",  # dark goldenrod (darker than standard Okabe-Ito yellow for
                                 # sufficient contrast against white; pure yellow #F0E442
                                 # was tested and found too low-contrast, especially in
                                 # legends and at line intersections)
    "ANN":          "#8B4513",  # saddle brown (distinct from the black top-1% cutoff
                                 # markers -- pure black for both was tested and found to
                                 # cause visual confusion between ANN's line and marker dots)
    "DNN":          "#999999",  # grey
}

LINESTYLES = {
    # Generic SFs: always solid
    "Smina": "-", "CNN-Score": "-", "RF-Score-VS": "-", "IFP": "-",
    # Target-specific algorithms: always dashed
    "RF": "--", "XGB": "--", "SVM": "--", "ANN": "--", "DNN": "--",
}

GENERIC_SFS = {"Smina", "CNN-Score", "RF-Score-VS", "IFP"}
TARGET_SPECIFIC_ALGOS = {"RF", "XGB", "SVM", "ANN", "DNN"}


def get_style(name):
    """Returns (color, linestyle) for a given scoring function/algorithm name."""
    if name not in COLORS:
        raise KeyError(f"'{name}' has no defined style -- add it to COLORS/LINESTYLES "
                        f"in plot_style.py before using it in a figure.")
    return COLORS[name], LINESTYLES[name]


# ----------------------------------------------------------------------
# Figure-level settings
# ----------------------------------------------------------------------
PLOT_SETTINGS = {
    'figsize_single': (4.5, 4.5),     # one scoring function / one algorithm, standalone
    'figsize_combined': (5, 5),        # single-panel, multiple lines overlaid (e.g. Section A's 4-SF comparison)
    'figsize_2panel': (9, 4.5),        # two side-by-side panels (e.g. Section D: Classification | Regression)
    'figsize_bar': (4.5, 4),
    'dpi': 300,
    'save_format': 'pdf',              # vector format; use 'png' only for quick previews
    'linewidth': 1.8,
    'marker_size': 45,
    'marker_color': 'black',
    'marker_edgecolor': 'white',
    'marker_edgewidth': 0.8,
    'font_family': 'Arial',
    'font_size': 9,
    'title_font_size': 10,
    'label_font_size': 9,
    'tick_font_size': 8,
    'legend_font_size': 8,
    'title_suffix': 'MAO-B (1S3B)',
    'grid': True,
    'spine_top_right_off': False,  # False = full framed box (all 4 spines visible);
                                     # set True for the open top/right-removed style
}

plt.rcParams.update({
    'font.family': PLOT_SETTINGS['font_family'],
    'font.size': PLOT_SETTINGS['font_size'],
    'axes.titlesize': PLOT_SETTINGS['title_font_size'],
    'axes.labelsize': PLOT_SETTINGS['label_font_size'],
    'xtick.labelsize': PLOT_SETTINGS['tick_font_size'],
    'ytick.labelsize': PLOT_SETTINGS['tick_font_size'],
    'legend.fontsize': PLOT_SETTINGS['legend_font_size'],
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'savefig.bbox': 'tight',
    'svg.fonttype': 'none',
})


def clean_axes(ax):
    """Apply the publication spine/tick style consistently across all plots."""
    if PLOT_SETTINGS['spine_top_right_off']:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=3)
    if PLOT_SETTINGS['grid']:
        ax.grid(True, linewidth=0.5, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)  # ensures gridlines sit behind the data lines, not on top


def plot_pr_curve(ax, recall, precision, name, top1pct_point=None, label=None):
    """
    Draws one PR curve on the given axes, using this entity's defined
    color/linestyle. Optionally marks its top-1% cutoff point.
    top1pct_point: (recall, precision) tuple, or None to skip the marker.
    """
    color, linestyle = get_style(name)
    ax.plot(recall, precision, color=color, linestyle=linestyle,
             linewidth=PLOT_SETTINGS['linewidth'], label=label or name)
    if top1pct_point is not None:
        ax.scatter([top1pct_point[0]], [top1pct_point[1]],
                   s=PLOT_SETTINGS['marker_size'], color=PLOT_SETTINGS['marker_color'],
                   edgecolor=PLOT_SETTINGS['marker_edgecolor'],
                   linewidth=PLOT_SETTINGS['marker_edgewidth'], zorder=5)
