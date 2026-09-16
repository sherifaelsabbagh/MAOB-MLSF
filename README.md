# Development and Evaluation of Target-Specific Machine-Learning Scoring Functions for Monoamine Oxidase B

Code accompanying the manuscript "Development and Evaluation of Target-Specific Machine-Learning Scoring Functions for Monoamine Oxidase B" (Elsabbagh, [year]). This repository contains the scripts used to build, evaluate, and analyze target-specific machine-learning scoring functions (ML SFs) for human MAO-B, following the general protocol of Tran-Nguyen et al. (2023) and directly informed by Caba et al. (2024) and James & Ballester (2026).

## Overview

Three modeling questions are addressed:
1. Classification versus regression modeling
2. Structure-based (PLEC) versus combined structure- and ligand-based (PLEC + Morgan fingerprint) featurization
3. Untuned (baseline) versus hyperparameter-tuned models

All models are benchmarked against established generic scoring functions (Smina, CNN-Score, RF-Score-VS), evaluated on both a public DUD-E benchmark and a test set composed exclusively of experimentally confirmed inactive compounds.

## Repository structure

```
Scripts/
├── ligand_and_receptor_preparation/   Receptor preparation (PyMOL/PDBFixer)
├── data_curation/                     Bioactivity data curation (PubChem/ChEMBL, IC50-only)
├── decoy_generation_and_splitting/    DeepCoy decoy generation, Butina clustering, train/test assembly
├── featurization/                     PLEC and Morgan+PLEC feature extraction
├── hit_list_generation/               Generic scoring function rescoring (Smina, CNN-Score, RF-Score-VS)
├── model_training_testing/            Target-specific model training (one representative script per algorithm; all algorithms follow the same structure)
└── metrics_and_figures/               EF1%/NEF1% calculation, all manuscript figures, statistical significance testing

results/
├── Table3_PLEC_results.csv
├── Table4_MFPLEC_results.csv
└── best_model_predictions/            Saved predictions for the two best-performing models (XGB regression MF+PLEC tuned; SVM regression MF+PLEC baseline)
```

## Environment

See `environment.yml`. Key dependencies: RDKit, ODDT, scikit-learn, XGBoost, Optuna, scipy, matplotlib.

```bash
conda env create -f environment.yml
conda activate maob-mlsf
```

## Data availability

Raw and intermediate data files (curated bioactivity datasets, docked poses, full training/test feature sets) are not included in this repository due to size, but are available upon request. Model training/testing scripts expect `train_PLEC_features.csv`, `test_PLEC_features.csv`, `train_MFPLEC_features.csv`, and `test_MFPLEC_features.csv` in the working directory; column structure is documented in each script's docstring.

## Notes on reproducibility

- All random seeds and hyperparameter search spaces are specified in each script.
- `model_training_testing/` includes one representative baseline and tuning script per algorithm (RF, XGB, SVM, ANN, DNN); the remaining algorithm/task/feature combinations reported in the manuscript follow an identical structure, differing only in the model class and hyperparameter search space (see manuscript Table 2 for search spaces).
- `metrics_and_figures/plot_style.py` defines the shared visual style (colors, line styles, formatting) used consistently across every figure in the manuscript.

## Citation

If you use this code, please cite:

[Full citation, once published]

## License

See `LICENSE.txt`.
