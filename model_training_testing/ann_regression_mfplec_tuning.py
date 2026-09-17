"""
ann_tuning.py

Hyperparameter-tuned ANN (scikit-learn MLPRegressor, single hidden
layer) regression for MAO-B (Section D, v2 dataset, PLEC features).

Phase 1 -- Search: 10 Optuna trials, each evaluating one candidate
hyperparameter combination via proper 5-fold cross-validation on the
training set, scored by mean Spearman rank correlation.

Search space (bracketing the baseline's hidden_layer_size=100 in both
directions):
    hidden_layer_size:   {32, 64, 128, 256} (categorical)
    alpha (L2 reg):       1e-5 to 1e-1 (log scale)
    learning_rate_init:   1e-4 to 1e-1 (log scale)

Phase 2 -- Final evaluation: 5 repeats with the winning combination,
trained on the full training set, evaluated on the real test set.

Outputs:
    ann_regression_mfplec_tuning_history.csv
    ann_regression_mfplec_tuned_results.csv
    ann_regression_mfplec_tuned_median_predictions.npy
    ANN_regression_mfplec_tuned_PR_curve.png
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 10
N_CV_FOLDS = 5
N_FINAL_REPEATS = 5

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
train_df = pd.read_csv('train_MFPLEC_features.csv')
test_df = pd.read_csv('test_MFPLEC_features.csv')

feature_cols = [c for c in train_df.columns if (c.startswith('f') or c.startswith('mf')) and c.lstrip('mf').isdigit()]
X_train = train_df[feature_cols].values
y_train_pic50 = train_df['pIC50'].values

X_test = test_df[feature_cols].values
y_test_active = (test_df['activity'] == 'Active').astype(int).values

print(f"Train: {X_train.shape}, Test: {X_test.shape}", flush=True)


def compute_ef1_nef1(y_true_active, y_score):
    order = np.argsort(-y_score)
    y_true_sorted = y_true_active[order]
    n_total = len(y_true_sorted)
    n_top1pct = max(1, round(0.01 * n_total))
    a_total = y_true_sorted.sum()
    a_top1pct = y_true_sorted[:n_top1pct].sum()
    ef1 = (a_top1pct / a_total) * 100
    max_ef1 = (min(n_top1pct, a_total) / a_total) * 100
    nef1 = ef1 / max_ef1
    return ef1, max_ef1, nef1


def build_ann(params, seed):
    return MLPRegressor(
        hidden_layer_sizes=(params["hidden_layer_size"],),
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        max_iter=1000,
        early_stopping=True,
        random_state=seed,
    )


tuning_log = []


def objective(trial):
    params = {
        "hidden_layer_size": trial.suggest_categorical("hidden_layer_size", [32, 64, 128, 256]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
    }

    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=0)
    fold_scores = []
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        model = build_ann(params, seed=0)
        model.fit(X_train[tr_idx], y_train_pic50[tr_idx])
        preds = model.predict(X_train[val_idx])
        rho, _ = spearmanr(y_train_pic50[val_idx], preds)
        fold_scores.append(rho if not np.isnan(rho) else 0.0)

    mean_rho = float(np.mean(fold_scores))
    print(f"Trial {trial.number + 1}/{N_TRIALS}: CV Spearman rho = {mean_rho:.4f}, "
          f"params = {params}", flush=True)

    tuning_log.append({**params, "trial": trial.number, "cv_spearman_rho": mean_rho})
    return mean_rho


print(f"\n=== Phase 1: Optuna search ({N_TRIALS} trials, {N_CV_FOLDS}-fold CV) ===", flush=True)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params
print(f"\nBest trial: CV Spearman rho = {study.best_value:.4f}, params = {best_params}", flush=True)

pd.DataFrame(tuning_log).to_csv('ann_regression_mfplec_tuning_history.csv', index=False)
print("Saved ann_regression_mfplec_tuning_history.csv")


print(f"\n=== Phase 2: Final evaluation ({N_FINAL_REPEATS} repeats) ===", flush=True)

results = []
for run in range(N_FINAL_REPEATS):
    model = build_ann(best_params, seed=run)
    model.fit(X_train, y_train_pic50)
    y_pred = model.predict(X_test)
    ef1, max_ef1, nef1 = compute_ef1_nef1(y_test_active, y_pred)
    print(f"Run {run + 1}/{N_FINAL_REPEATS}: EF1% = {ef1:.3f}, NEF1% = {nef1:.3f}", flush=True)
    results.append({"run": run, "EF1%": ef1, "maximal_EF1%": max_ef1, "NEF1%": nef1})

results_df = pd.DataFrame(results)
results_df.to_csv('ann_regression_mfplec_tuned_results.csv', index=False)

median_nef1 = results_df['NEF1%'].median()
std_nef1 = results_df['NEF1%'].std()
print(f"\nMedian EF1% = {results_df['EF1%'].median():.3f}, "
      f"Median NEF1% = {median_nef1:.3f} (std = {std_nef1:.3f})", flush=True)

closest_idx = (results_df['NEF1%'] - median_nef1).abs().idxmin()
closest_seed = int(results_df.loc[closest_idx, 'run'])
model = build_ann(best_params, seed=closest_seed)
model.fit(X_train, y_train_pic50)
y_pred = model.predict(X_test)

np.save('ann_regression_mfplec_tuned_median_predictions.npy', y_pred)

precision, recall, _ = precision_recall_curve(y_test_active, y_pred)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(recall, precision, linewidth=1.8)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title(f'ANN (tuned) regression — median run (seed={closest_seed})')
plt.savefig('ANN_regression_mfplec_tuned_PR_curve.png', dpi=200)
print("Saved ANN_regression_mfplec_tuned_PR_curve.png and ann_regression_mfplec_tuned_median_predictions.npy")
