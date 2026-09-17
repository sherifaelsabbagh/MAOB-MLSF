"""
xgb_tuning.py

Hyperparameter-tuned XGBoost regression for MAO-B (Section D, v2
dataset, PLEC features).

Phase 1 -- Search: 10 Optuna trials, each evaluating one candidate
hyperparameter combination via proper 5-fold cross-validation on the
training set (never touching the real test set), scored by mean
Spearman rank correlation between predicted and true pIC50 across the
5 folds. Same rationale as rf_tuning.py: Spearman (a ranking-quality
metric) is used instead of raw MSE or MCC.

Search space (defined from first principles for this project's scale,
not copied from the PARP1/TRPM8 reference notebook's own ranges,
calibrated for a ~79x smaller training set):
    n_estimators:      100-800
    max_depth:         3-15 (kept shallower than RF's 5-30, since
                        boosting corrects errors iteratively across
                        many trees rather than needing each tree to
                        capture everything alone)
    learning_rate:      0.01-0.3 (log scale)
    subsample:          0.5-1.0
    colsample_bytree:   0.5-1.0

NOTE: subsample and colsample_bytree are tuned (not left at their
default of 1.0) specifically because, without them, XGBoost training
is nearly deterministic given fixed hyperparameters -- a real bug
diagnosed earlier in this project (identical results across repeated
runs with different random_state, since no other randomness source
was present). Tuning these here both searches a genuinely useful
regularization axis AND ensures the required randomness is present.

Phase 2 -- Final evaluation: 5 repeats with the winning combination,
trained on the full training set, evaluated on the real test set,
matching this project's established convention.

Outputs:
    xgb_tuning_history.csv
    xgb_tuned_results.csv
    xgb_tuned_median_predictions.npy
    XGB_tuned_PR_curve.png
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
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
train_df = pd.read_csv('train_PLEC_features.csv')
test_df = pd.read_csv('test_PLEC_features.csv')

feature_cols = [c for c in train_df.columns if c.startswith('f') and c[1:].isdigit()]
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


def build_xgb(params, seed):
    return XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=seed,
        n_jobs=-1,
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

pd.DataFrame(tuning_log).to_csv('xgb_tuning_history.csv', index=False)
print("Saved xgb_tuning_history.csv")


print(f"\n=== Phase 2: Final evaluation ({N_FINAL_REPEATS} repeats) ===", flush=True)

results = []
for run in range(N_FINAL_REPEATS):
    model = build_xgb(best_params, seed=run)
    model.fit(X_train, y_train_pic50)
    y_pred = model.predict(X_test)
    ef1, max_ef1, nef1 = compute_ef1_nef1(y_test_active, y_pred)
    print(f"Run {run + 1}/{N_FINAL_REPEATS}: EF1% = {ef1:.3f}, NEF1% = {nef1:.3f}", flush=True)
    results.append({"run": run, "EF1%": ef1, "maximal_EF1%": max_ef1, "NEF1%": nef1})

results_df = pd.DataFrame(results)
results_df.to_csv('xgb_tuned_results.csv', index=False)

median_nef1 = results_df['NEF1%'].median()
std_nef1 = results_df['NEF1%'].std()
print(f"\nMedian EF1% = {results_df['EF1%'].median():.3f}, "
      f"Median NEF1% = {median_nef1:.3f} (std = {std_nef1:.3f})", flush=True)

closest_idx = (results_df['NEF1%'] - median_nef1).abs().idxmin()
closest_seed = int(results_df.loc[closest_idx, 'run'])
model = build_xgb(best_params, seed=closest_seed)
model.fit(X_train, y_train_pic50)
y_pred = model.predict(X_test)

np.save('xgb_tuned_median_predictions.npy', y_pred)

precision, recall, _ = precision_recall_curve(y_test_active, y_pred)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(recall, precision, linewidth=1.8)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title(f'XGB (tuned) regression — median run (seed={closest_seed})')
plt.savefig('XGB_tuned_PR_curve.png', dpi=200)
print("Saved XGB_tuned_PR_curve.png and xgb_tuned_median_predictions.npy")
