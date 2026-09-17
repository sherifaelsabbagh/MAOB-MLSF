"""
rf_classification_tuning.py

Hyperparameter-tuned RF CLASSIFICATION for MAO-B (Section D, v2
dataset, PLEC features) -- the classification counterpart to
rf_tuning.py (regression), on the IDENTICAL train/test data.

Phase 1 -- Search: 10 Optuna trials, each evaluating one candidate
hyperparameter combination via proper 5-fold cross-validation on the
training set, scored by mean AVERAGE PRECISION (area under the
precision-recall curve) across the 5 folds.

Average Precision (not MCC, the metric used in the reference PARP1/
TRPM8 tuning code) was chosen deliberately: this project's actual
reported metric, NEF1%, is entirely rank-based (it never uses a fixed
classification threshold) -- a model could have poor MCC at the
default 0.5 threshold while still ranking true actives above true
inactives correctly. Average Precision, like NEF1%, is threshold-free
and evaluates ranking quality across the entire precision-recall
curve, making it a better-aligned proxy for this project's specific
goal, at the cost of not matching the reference code's own choice of
metric.

Search space (same ranges as rf_tuning.py's regression search, since
the underlying RF algorithm's sensitivity to n_estimators/max_depth/
min_samples_leaf is not expected to differ fundamentally by target
type):
    n_estimators:      100-800
    max_depth:         5-30
    min_samples_leaf:  1-20
class_weight="balanced" is applied throughout (fixed, not tuned),
carrying over the fix already validated for RF classification earlier
in this project (recall RF collapsed to predicting the majority class
without it).

Phase 2 -- Final evaluation: 5 repeats with the winning combination,
trained on the full training set, evaluated on the real test set,
using predict_proba (not predict) for scoring.

Outputs:
    rf_classification_tuning_history.csv
    rf_classification_tuned_results.csv
    rf_classification_tuned_median_predictions.npy
    RF_classification_tuned_PR_curve.png
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 10
N_CV_FOLDS = 5
N_FINAL_REPEATS = 5

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
train_df = pd.read_csv('train_PLEC_features.csv')
test_df = pd.read_csv('test_PLEC_features.csv')

feature_cols = [c for c in train_df.columns if c.startswith('f') and c[1:].isdigit()]
X_train = train_df[feature_cols].values
y_train_active = (train_df['activity'] == 'Active').astype(int).values

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


def build_rf(params, seed):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


tuning_log = []


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }

    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=0)
    fold_scores = []
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        model = build_rf(params, seed=0)
        model.fit(X_train[tr_idx], y_train_active[tr_idx])
        preds = model.predict_proba(X_train[val_idx])[:, 1]
        ap = average_precision_score(y_train_active[val_idx], preds)
        fold_scores.append(ap)

    mean_ap = float(np.mean(fold_scores))
    print(f"Trial {trial.number + 1}/{N_TRIALS}: CV Average Precision = {mean_ap:.4f}, "
          f"params = {params}", flush=True)

    tuning_log.append({**params, "trial": trial.number, "cv_average_precision": mean_ap})
    return mean_ap


print(f"\n=== Phase 1: Optuna search ({N_TRIALS} trials, {N_CV_FOLDS}-fold CV) ===", flush=True)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params
print(f"\nBest trial: CV Average Precision = {study.best_value:.4f}, params = {best_params}", flush=True)

pd.DataFrame(tuning_log).to_csv('rf_classification_tuning_history.csv', index=False)
print("Saved rf_classification_tuning_history.csv")


print(f"\n=== Phase 2: Final evaluation ({N_FINAL_REPEATS} repeats) ===", flush=True)

results = []
for run in range(N_FINAL_REPEATS):
    model = build_rf(best_params, seed=run)
    model.fit(X_train, y_train_active)
    y_score = model.predict_proba(X_test)[:, 1]
    ef1, max_ef1, nef1 = compute_ef1_nef1(y_test_active, y_score)
    print(f"Run {run + 1}/{N_FINAL_REPEATS}: EF1% = {ef1:.3f}, NEF1% = {nef1:.3f}", flush=True)
    results.append({"run": run, "EF1%": ef1, "maximal_EF1%": max_ef1, "NEF1%": nef1})

results_df = pd.DataFrame(results)
results_df.to_csv('rf_classification_tuned_results.csv', index=False)

median_nef1 = results_df['NEF1%'].median()
std_nef1 = results_df['NEF1%'].std()
print(f"\nMedian EF1% = {results_df['EF1%'].median():.3f}, "
      f"Median NEF1% = {median_nef1:.3f} (std = {std_nef1:.3f})", flush=True)

closest_idx = (results_df['NEF1%'] - median_nef1).abs().idxmin()
closest_seed = int(results_df.loc[closest_idx, 'run'])
model = build_rf(best_params, seed=closest_seed)
model.fit(X_train, y_train_active)
y_score = model.predict_proba(X_test)[:, 1]

np.save('rf_classification_tuned_median_predictions.npy', y_score)

precision, recall, _ = precision_recall_curve(y_test_active, y_score)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(recall, precision, linewidth=1.8)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title(f'RF (tuned) classification — median run (seed={closest_seed})')
plt.savefig('RF_classification_tuned_PR_curve.png', dpi=200)
print("Saved RF_classification_tuned_PR_curve.png and rf_classification_tuned_median_predictions.npy")
