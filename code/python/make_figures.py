import os, json, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = '/mnt/data/stat4052_project'
DATA = os.path.join(PROJECT, 'data', 'risk_factors_cervical_cancer.csv')
OUT = os.path.join(PROJECT, 'outputs')
FIG = os.path.join(PROJECT, 'figures')
os.makedirs(FIG, exist_ok=True)

df_raw = pd.read_csv(DATA)
df_num = df_raw.replace('?', np.nan).apply(pd.to_numeric, errors='coerce')
y = df_num['Biopsy'].astype(int)
metrics = pd.read_csv(os.path.join(OUT, 'test_metrics.csv'))
roc_df = pd.read_csv(os.path.join(OUT, 'roc_data.csv'))
coef_df = pd.read_csv(os.path.join(OUT, 'logistic_coefficients.csv'))
rf_imp = pd.read_csv(os.path.join(OUT, 'perm_importance_random_forest.csv'))
methods = ['Logistic regression', 'Radial SVM', 'Random forest']

def save(fig, name):
    # Avoid tight_layout hangs by using simple bbox_inches.
    fig.savefig(os.path.join(FIG, name), dpi=220, bbox_inches='tight')
    plt.close(fig)

# Class balance
counts = [int((y == 0).sum()), int((y == 1).sum())]
labels = ['Biopsy = 0\n(no positive finding)', 'Biopsy = 1\n(positive finding)']
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.bar(labels, counts)
ax.set_ylabel('Number of patients')
ax.set_title('Strong class imbalance in the biopsy endpoint')
for i, v in enumerate(counts):
    ax.text(i, v + 12, f'{v} ({v/len(y):.1%})', ha='center', fontsize=9)
ax.set_ylim(0, max(counts)*1.15)
save(fig, 'class_balance.png')
print('class_balance')

# Missingness top 10
miss = df_raw.replace('?', np.nan).isna().sum().sort_values(ascending=False).head(10)[::-1]
fig, ax = plt.subplots(figsize=(7.2, 4.5))
ypos = np.arange(len(miss))
labels = [textwrap.fill(x, width=28) for x in miss.index]
ax.barh(ypos, miss.values)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Missing values')
ax.set_title('Top 10 variables by missingness')
ax.set_xlim(0, max(miss.values)*1.12)
for y0, v in zip(ypos, miss.values):
    ax.text(v + max(miss.values)*0.01, y0, str(int(v)), va='center', fontsize=8)
fig.subplots_adjust(left=0.38, right=0.95, top=0.88, bottom=0.15)
save(fig, 'missingness_top10.png')
print('missingness')

# ROC curves
fig, ax = plt.subplots(figsize=(6.3, 5.0))
for name in methods:
    sub = roc_df[roc_df['method'] == name]
    auc = metrics[(metrics['method'] == name) & (metrics['threshold_type'] == 'CV Youden')]['auc'].iloc[0]
    ax.plot(sub['fpr'], sub['tpr'], label=f'{name} (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1, label='Chance')
ax.set_xlabel('False positive rate')
ax.set_ylabel('Sensitivity')
ax.set_title('Test-set ROC curves')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.25)
save(fig, 'roc_curves.png')
print('roc')

# Metrics bar
met = metrics[metrics['threshold_type'] == 'CV Youden'].set_index('method')
metrics_list = ['auc', 'sensitivity', 'specificity', 'balanced_accuracy']
pretty = ['AUC', 'Sensitivity', 'Specificity', 'Balanced acc.']
fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(len(methods))
width = 0.18
for j, col in enumerate(metrics_list):
    vals = [met.loc[m, col] for m in methods]
    ax.bar(x + (j - 1.5) * width, vals, width, label=pretty[j])
ax.set_xticks(x)
ax.set_xticklabels(['Logistic\nregression', 'Radial\nSVM', 'Random\nforest'])
ax.set_ylim(0, 1.05)
ax.set_ylabel('Metric value')
ax.set_title('Test performance at training-selected thresholds')
ax.legend(fontsize=8, ncol=2, loc='upper right')
ax.grid(axis='y', alpha=0.25)
fig.subplots_adjust(bottom=0.18, top=0.88, right=0.97)
save(fig, 'metrics_bar.png')
print('metrics')

# Logistic coefficients top 10
coef_top = coef_df.head(10)[::-1].copy()
coef_top['label'] = coef_top['feature'].str.replace('num__', '', regex=False).str.replace('missingindicator_', 'missing: ', regex=False)
coef_top['label'] = [textwrap.fill(x, width=30) for x in coef_top['label']]
fig, ax = plt.subplots(figsize=(7.0, 4.8))
ypos = np.arange(len(coef_top))
ax.barh(ypos, coef_top['coefficient'].values)
ax.set_yticks(ypos)
ax.set_yticklabels(coef_top['label'], fontsize=8)
ax.axvline(0, linewidth=0.8)
ax.set_xlabel('Standardized logistic coefficient')
ax.set_title('Largest logistic regression coefficients')
fig.subplots_adjust(left=0.42, right=0.95, top=0.88, bottom=0.15)
save(fig, 'logistic_coefficients.png')
print('coef')

# RF permutation importance top 10
rf_top = rf_imp.head(10)[::-1].copy()
rf_top['label'] = [textwrap.fill(x, width=32) for x in rf_top['feature']]
fig, ax = plt.subplots(figsize=(7.0, 4.8))
ypos = np.arange(len(rf_top))
ax.barh(ypos, rf_top['importance_mean_auc_drop'].values)
ax.set_yticks(ypos)
ax.set_yticklabels(rf_top['label'], fontsize=8)
ax.set_xlabel('Mean decrease in test AUC')
ax.set_title('Random forest permutation importance')
fig.subplots_adjust(left=0.38, right=0.95, top=0.88, bottom=0.15)
save(fig, 'rf_permutation_importance.png')
print('rf')
