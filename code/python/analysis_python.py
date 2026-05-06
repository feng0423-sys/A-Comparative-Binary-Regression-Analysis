import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.inspection import permutation_importance
from sklearn.base import clone
import joblib

PROJECT = '/mnt/data/stat4052_project'
DATA = os.path.join(PROJECT, 'data', 'risk_factors_cervical_cancer.csv')
OUT = os.path.join(PROJECT, 'outputs')
FIG = os.path.join(PROJECT, 'figures')
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SEED=4052

df_raw = pd.read_csv(DATA)
df = df_raw.replace('?', np.nan).copy()
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Exclude alternative diagnostic target/test outcomes to keep a risk-factor prediction question.
target = 'Biopsy'
exclude = ['Hinselmann', 'Schiller', 'Citology', target]
# Drop columns with >50% missingness; these are two STD time-since-diagnosis fields.
missing_rate = df.isna().mean()
drop_high_missing = missing_rate[missing_rate > 0.50].index.tolist()
features = [c for c in df.columns if c not in exclude + drop_high_missing]
X = df[features]
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)

# Preprocessors
num_cols = features
preprocess_scaled = ColumnTransformer([
    ('num', Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler())
    ]), num_cols)
], remainder='drop')
preprocess_rf = ColumnTransformer([
    ('num', SimpleImputer(strategy='median', add_indicator=True), num_cols)
], remainder='drop')

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

models = {
    'Logistic regression': {
        'pipe': Pipeline([
            ('prep', preprocess_scaled),
            ('model', LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=2000, random_state=SEED))
        ]),
        'param_grid': {'model__C': [0.05, 0.1, 0.5, 1, 5]}
    },
    'Radial SVM': {
        'pipe': Pipeline([
            ('prep', preprocess_scaled),
            ('model', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=SEED))
        ]),
        'param_grid': {'model__C': [0.5, 2, 10], 'model__gamma': ['scale', 0.01, 0.1]}
    },
    'Random forest': {
        'pipe': Pipeline([
            ('prep', preprocess_rf),
            ('model', RandomForestClassifier(n_estimators=75, class_weight='balanced_subsample', random_state=SEED, n_jobs=1))
        ]),
        'param_grid': {'model__max_features': ['sqrt'], 'model__min_samples_leaf': [1, 5], 'model__max_depth': [None, 4]}
    }
}

def youden_threshold(y_true, probs):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    ix = int(np.nanargmax(j))
    return float(thresholds[ix]), float(tpr[ix]), float(1 - fpr[ix]), float(j[ix])

def metric_row(name, y_true, probs, thr, extra=None):
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    row = {
        'method': name,
        'threshold': thr,
        'auc': roc_auc_score(y_true, probs),
        'brier': brier_score_loss(y_true, probs),
        'log_loss': log_loss(y_true, probs, labels=[0,1]),
        'accuracy': accuracy_score(y_true, pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, pred),
        'sensitivity': recall_score(y_true, pred, pos_label=1, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) else np.nan,
        'precision': precision_score(y_true, pred, pos_label=1, zero_division=0),
        'f1': f1_score(y_true, pred, pos_label=1, zero_division=0),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }
    if extra:
        row.update(extra)
    return row

results=[]
cv_results=[]
roc_data=[]
trained={}
thresholds=[]
for name, spec in models.items():
    print("START", name, flush=True)
    gs = GridSearchCV(spec['pipe'], spec['param_grid'], scoring='roc_auc', cv=cv, n_jobs=1, refit=True, return_train_score=True)
    gs.fit(X_train, y_train)
    print("GRID DONE", name, gs.best_params_, flush=True)
    best = gs.best_estimator_
    trained[name] = best
    # Cross-val predicted probabilities with tuned hyperparameters, for threshold choice without test leakage.
    print("CV PROBS", name, flush=True)
    cv_probs = cross_val_predict(best, X_train, y_train, cv=cv, method='predict_proba', n_jobs=1)[:,1]
    thr, cv_sens, cv_spec, cv_j = youden_threshold(y_train, cv_probs)
    thresholds.append({'method': name, 'cv_auc': roc_auc_score(y_train, cv_probs), 'chosen_threshold': thr, 'cv_sensitivity_at_threshold': cv_sens, 'cv_specificity_at_threshold': cv_spec, 'cv_youden_j': cv_j, 'best_params': str(gs.best_params_)})
    test_probs = best.predict_proba(X_test)[:,1]
    results.append(metric_row(name, y_test, test_probs, 0.5, {'threshold_type':'0.50'}))
    results.append(metric_row(name, y_test, test_probs, thr, {'threshold_type':'CV Youden'}))
    fpr, tpr, roc_thr = roc_curve(y_test, test_probs)
    for a,b,c in zip(fpr,tpr,roc_thr):
        roc_data.append({'method':name,'fpr':a,'tpr':b,'threshold':c})
    # Save grid summary
    cvres = pd.DataFrame(gs.cv_results_)
    cols = ['mean_test_score','std_test_score','rank_test_score'] + [c for c in cvres.columns if c.startswith('param_')]
    cvres[cols].sort_values('rank_test_score').to_csv(os.path.join(OUT, f'cv_{name.lower().replace(" ", "_")}.csv'), index=False)

metrics = pd.DataFrame(results)
metrics.to_csv(os.path.join(OUT, 'test_metrics.csv'), index=False)
pd.DataFrame(thresholds).to_csv(os.path.join(OUT, 'training_thresholds.csv'), index=False)
pd.DataFrame(roc_data).to_csv(os.path.join(OUT, 'roc_data.csv'), index=False)

# Dataset summary
summary = {
    'n_total': int(len(df)),
    'n_features_raw': int(df.shape[1]-1),
    'target_positive': int(y.sum()),
    'target_negative': int((1-y).sum()),
    'positive_rate': float(y.mean()),
    'n_train': int(len(y_train)),
    'n_test': int(len(y_test)),
    'train_positive': int(y_train.sum()),
    'test_positive': int(y_test.sum()),
    'excluded_diagnostic_columns': exclude[:-1],
    'dropped_high_missing_columns': drop_high_missing,
    'n_predictors_used': len(features),
    'predictors_used': features,
    'missing_counts_top10': df_raw.replace('?', np.nan).isna().sum().sort_values(ascending=False).head(10).astype(int).to_dict()
}
with open(os.path.join(OUT, 'data_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# Coefficients for logistic regression
logit = trained['Logistic regression']
# get feature names after imputer indicators maybe not easy; do coefficients for original variables only by fitting without add_indicator? Better extract names.
prep = logit.named_steps['prep']
model = logit.named_steps['model']
try:
    feature_names = prep.get_feature_names_out()
except Exception:
    feature_names = [f'x{i}' for i in range(len(model.coef_[0]))]
coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': model.coef_[0]})
coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
coef_df.to_csv(os.path.join(OUT, 'logistic_coefficients.csv'), index=False)

print("PLOTTING/IMPORTANCE", flush=True)
# Permutation importance using AUC on test set (descriptive; test-only, not used for selection).
for name, est in trained.items():
    try:
        print("PERM", name, flush=True)
        pi = permutation_importance(est, X_test, y_test, scoring='roc_auc', n_repeats=10, random_state=SEED, n_jobs=1)
        imp = pd.DataFrame({'feature': features, 'importance_mean_auc_drop': pi.importances_mean, 'importance_sd': pi.importances_std})
        imp.sort_values('importance_mean_auc_drop', ascending=False).to_csv(os.path.join(OUT, f'perm_importance_{name.lower().replace(" ", "_")}.csv'), index=False)
    except Exception as e:
        print('perm importance error', name, e)

print("FIGURES", flush=True)
# figures
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# class balance figure
counts = pd.Series({ 'No biopsy': int((y==0).sum()), 'Positive biopsy': int((y==1).sum()) })
fig, ax = plt.subplots(figsize=(6.4,4.1), dpi=220)
counts.plot(kind='bar', ax=ax)
ax.set_ylabel('Number of patients')
ax.set_title('Biopsy outcome distribution')
for i,v in enumerate(counts.values):
    ax.text(i, v + max(counts)*0.02, f'{v}\n({v/len(y):.1%})', ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=0)
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'class_balance.png'), bbox_inches='tight')
plt.close(fig)

# missingness figure top 10
miss = df_raw.replace('?', np.nan).isna().sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(7.0,4.5), dpi=220)
miss.iloc[::-1].plot(kind='barh', ax=ax)
ax.set_xlabel('Missing values')
ax.set_title('Top 10 predictors by missingness')
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'missingness_top10.png'), bbox_inches='tight')
plt.close(fig)

# ROC figure
roc_df = pd.DataFrame(roc_data)
fig, ax = plt.subplots(figsize=(6.4,5.0), dpi=220)
for name in models.keys():
    sub=roc_df[roc_df.method==name]
    auc=metrics[(metrics.method==name)&(metrics.threshold_type=='CV Youden')]['auc'].iloc[0]
    ax.plot(sub['fpr'], sub['tpr'], label=f'{name} (AUC={auc:.3f})')
ax.plot([0,1],[0,1], linestyle='--', linewidth=1, label='Chance')
ax.set_xlabel('False positive rate')
ax.set_ylabel('Sensitivity / true positive rate')
ax.set_title('Test-set ROC curves')
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'roc_curves.png'), bbox_inches='tight')
plt.close(fig)

# metrics bar (sensitivity/specificity/AUC at chosen threshold)
metric_plot = metrics[metrics.threshold_type=='CV Youden'][['method','auc','sensitivity','specificity','balanced_accuracy']].melt(id_vars='method', var_name='metric', value_name='value')
fig, ax = plt.subplots(figsize=(7.2,4.3), dpi=220)
# custom grouped bars no seaborn
methods_list = list(models.keys())
metrics_list = ['auc','sensitivity','specificity','balanced_accuracy']
x = np.arange(len(methods_list))
width=0.18
for j,m in enumerate(metrics_list):
    vals=[metric_plot[(metric_plot.method==meth)&(metric_plot.metric==m)].value.iloc[0] for meth in methods_list]
    ax.bar(x + (j-1.5)*width, vals, width, label=m.replace('_',' '))
ax.set_xticks(x)
ax.set_xticklabels(methods_list, rotation=15, ha='right')
ax.set_ylim(0,1.05)
ax.set_ylabel('Metric value')
ax.set_title('Test metrics at training-selected thresholds')
ax.legend(fontsize=8, ncol=2)
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'metrics_bar.png'), bbox_inches='tight')
plt.close(fig)

# logit coefficient figure top 10 original/non-missing indicators
coef_top = coef_df.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.0,5.0), dpi=220)
ax.barh(coef_top['feature'].str.replace('num__','', regex=False), coef_top['coefficient'])
ax.set_xlabel('Standardized logistic coefficient')
ax.set_title('Largest logistic regression coefficients')
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'logistic_coefficients.png'), bbox_inches='tight')
plt.close(fig)

# RF permutation importance top 10
rf_imp = pd.read_csv(os.path.join(OUT, 'perm_importance_random_forest.csv')).head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.0,4.8), dpi=220)
ax.barh(rf_imp['feature'], rf_imp['importance_mean_auc_drop'])
ax.set_xlabel('Mean decrease in test AUC when permuted')
ax.set_title('Random forest permutation importance')
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'rf_permutation_importance.png'), bbox_inches='tight')
plt.close(fig)

print("SAVE", flush=True)
# Save trained artifacts for reproducibility
joblib.dump(trained, os.path.join(OUT, 'trained_models.joblib'))
print('done')
print(metrics)
print(pd.DataFrame(thresholds))
print(summary)
