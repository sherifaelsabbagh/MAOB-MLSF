"""
xgb_classification_tuning.py

Hyperparameter-tuned XGB CLASSIFICATION for MAO-B (Section D, v2
dataset, PLEC features) -- classification counterpart to
xgb_tuning.py (regression), on the IDENTICAL train/test data.

Same rationale as rf_classification_tuning.py: Average Precision (not
MCC) as the CV objective, since NEF1% is threshold-free/rank-based.

Search space (same ranges as xgb_tuning.py's regression search):
    n_estimators:      100-800
    max_depth:         3-15
    learning_rate:      0.01-0.3 (log scale)
    subsample:          0.5-1.0
    colsample_bytree:   0.5-1.0
scale_pos_weight (n_negative/n_positive) applied throughout (fixed,
not tuned), carrying over XGB's classification-imbalance fix already
validated earlier in this project.

Outputs:
    xgb_classification_mfplec_tuning_history.csv
    xgb_classification_mfplec_tuned_results.csv
    xgb_classification_mfplec_tuned_median_predictions.npy
    XGB_classification_mfplec_tuned_PR_curve.png
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
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
train_df = pd.read_csv('train_MFPLEC_features.csv')
test_df = pd.read_csv('test_MFPLEC_features.csv')

feature_cols = [c for c in train_df.columns if (c.startswith('f') or c.startswith('mf')) and c.lstrip('mf').isdigit()]
X_train = train_df[feature_cols].values
y_train_active = (train_df['activity'] == 'Active').astype(int).values

X_test = test_df[feature_cols].values
y_test_active = (test_df['activity'] == 'Active').astype(int).values

print(f"Train: {X_train.shape}, Test: {X_test.shape}", flush=True)

n_pos = y_train_active.sum()
n_neg = len(y_train_active) - n_pos
scale_pos_weight = n_neg / n_pos
print(f"Class balance: {n_pos} active / {n_neg} inactive, scale_pos_weight = {scale_pos_weight:.3f}", flush=True)


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


def build_xgb(params, seed):
    return XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
    )


tuning_log = []


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=0)
    fold_scores = []
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        model = build_xgb(params, seed=0)
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

pd.DataFrame(tuning_log).to_csv('xgb_classification_mfplec_tuning_history.csv', index=False)
print("Saved xgb_classification_mfplec_tuning_history.csv")


print(f"\n=== Phase 2: Final evaluation ({N_FINAL_REPEATS} repeats) ===", flush=True)

results = []
for run in range(N_FINAL_REPEATS):
    model = build_xgb(best_params, seed=run)
    model.fit(X_train, y_train_active)
    y_score = model.predict_proba(X_test)[:, 1]
    ef1, max_ef1, nef1 = compute_ef1_nef1(y_test_active, y_score)
    print(f"Run {run + 1}/{N_FINAL_REPEATS}: EF1% = {ef1:.3f}, NEF1% = {nef1:.3f}", flush=True)
    results.append({"run": run, "EF1%": ef1, "maximal_EF1%": max_ef1, "NEF1%": nef1})

results_df = pd.DataFrame(results)
results_df.to_csv('xgb_classification_mfplec_tuned_results.csv', index=False)

median_nef1 = results_df['NEF1%'].median()
std_nef1 = results_df['NEF1%'].std()
print(f"\nMedian EF1% = {results_df['EF1%'].median():.3f}, "
      f"Median NEF1% = {median_nef1:.3f} (std = {std_nef1:.3f})", flush=True)

closest_idx = (results_df['NEF1%'] - median_nef1).abs().idxmin()
closest_seed = int(results_df.loc[closest_idx, 'run'])
model = build_xgb(best_params, seed=closest_seed)
model.fit(X_train, y_train_active)
y_score = model.predict_proba(X_test)[:, 1]

np.save('xgb_classification_mfplec_tuned_median_predictions.npy', y_score)

precision, recall, _ = precision_recall_curve(y_test_active, y_score)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(recall, precision, linewidth=1.8)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title(f'XGB (tuned) classification — median run (seed={closest_seed})')
plt.savefig('XGB_classification_mfplec_tuned_PR_curve.png', dpi=200)
print("Saved XGB_classification_mfplec_tuned_PR_curve.png and xgb_classification_mfplec_tuned_median_predictions.npy")
