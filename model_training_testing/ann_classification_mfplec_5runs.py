import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

train_df = pd.read_csv('train_MFPLEC_features.csv')
test_df = pd.read_csv('test_MFPLEC_features.csv')

feature_cols = [c for c in train_df.columns if (c.startswith('f') or c.startswith('mf')) and c.lstrip('mf').isdigit()]
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

results = []
for run in range(5):
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000,
                           early_stopping=True, random_state=run)
    model.fit(X_train, y_train_active)
    y_score = model.predict_proba(X_test)[:, 1]
    ef1, max_ef1, nef1 = compute_ef1_nef1(y_test_active, y_score)
    print(f"Run {run+1}/5: EF1% = {ef1:.3f}, NEF1% = {nef1:.3f}", flush=True)
    results.append({"run": run, "EF1%": ef1, "maximal_EF1%": max_ef1, "NEF1%": nef1})

results_df = pd.DataFrame(results)
results_df.to_csv('ann_classification_mfplec_results.csv', index=False)

median_nef1 = results_df['NEF1%'].median()
std_nef1 = results_df['NEF1%'].std()
print(f"\nMedian EF1% = {results_df['EF1%'].median():.3f}, Median NEF1% = {median_nef1:.3f} (std = {std_nef1:.3f})")

closest_idx = (results_df['NEF1%'] - median_nef1).abs().idxmin()
closest_seed = int(results_df.loc[closest_idx, 'run'])
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000,
                       early_stopping=True, random_state=closest_seed)
model.fit(X_train, y_train_active)
y_score = model.predict_proba(X_test)[:, 1]

np.save('ANN_classification_mfplec_median_predictions.npy', y_score)

precision, recall, _ = precision_recall_curve(y_test_active, y_score)
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(recall, precision, linewidth=1.8)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title(f'ANN classification (MF+PLEC) — median run (seed={closest_seed})')
plt.savefig('ANN_classification_mfplec_PR_curve.png', dpi=200)
print("Saved individual PR curve and predictions.")
