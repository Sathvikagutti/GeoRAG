"""One-off generator for notebooks/Member2_LULC_ML.ipynb (viva-ready mirror of src/run_baseline.py)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

nb.cells = [
    md("""# GeoRAG — Member 2: LULC Classification Baseline (RF + XGBoost)

**Goal:** classify Landsat pixels into 4 land-use classes for 2015, 2020, 2025 and evaluate honestly.

| Code | Class |
|---|---|
| 1 | Built-up |
| 2 | Vegetation |
| 3 | Water |
| 4 | Open Land |

**Workflow per year:** load → deduplicate → stratified 80/20 split → train RF & XGBoost → accuracy / Kappa / macro-F1 / confusion matrix / feature importance.

**Known limitations (documented, not hidden):**
1. Random pixel split + spatial autocorrelation → accuracy may be optimistic (no X/Y in CSVs, so spatial block validation is not possible yet).
2. Label provenance unknown → if labels were thresholded from the same bands, scores are partly circular. Ask Member 1.
3. Class imbalance (e.g., Open Land 2.8% in 2015) → we use class weights and per-class metrics, not accuracy alone."""),
    code("""import time
from pathlib import Path
import joblib, numpy as np, pandas as pd
import xgboost as xgb, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score)

def find_root(p=Path.cwd()):
    # Locate the project root no matter where the notebook is launched from.
    for cand in [p, *p.parents]:
        if (cand/'data'/'training_2015.csv').exists():
            return cand
    raise FileNotFoundError('Could not locate project root (data/training_2015.csv)')

ROOT = find_root()
print('project root:', ROOT)
DATA, MODELS, RESULTS = ROOT/'data', ROOT/'models', ROOT/'results'
FILES = {2015: 'training_2015.csv', 2020: 'training_2020.csv', 2025: 'training_2025.csv'}
FEATURES = ['Blue','Green','Red','NIR','SWIR1','SWIR2','NDVI','NDBI','NDWI']
CLASS_NAMES = {1:'Built-up', 2:'Vegetation', 3:'Water', 4:'Open Land'}
RANDOM_STATE = 42  # reproducibility"""),
    md("## 1. Load and inspect one year (2015)"),
    code("""df = pd.read_csv(DATA/'training_2015.csv')
print('shape:', df.shape)
df.head()"""),
    code("""df['Label'].value_counts().sort_index().rename(index=CLASS_NAMES).plot(
    kind='bar', title='2015 class distribution (raw)', ylabel='pixels');
plt.tight_layout(); plt.show()"""),
    md("""## 2. Deduplicate **before** splitting

Exact duplicate rows are removed first. If we deduplicated *after* splitting, copies of the
same pixel could land in both train and test — that is data leakage."""),
    code("""n_raw = len(df)
df = df.drop_duplicates(subset=FEATURES + ['Label']).reset_index(drop=True)
print(f'{n_raw:,} -> {len(df):,} rows ({n_raw-len(df):,} exact duplicates removed)')"""),
    md("""## 3. Stratified train/test split

`stratify=y` keeps class proportions equal in both sets. `Label` is remapped 1..4 → 0..3
only because XGBoost requires zero-based integer labels; all reports show original codes."""),
    code("""X = df[FEATURES].to_numpy(np.float32)
y = df['Label'].to_numpy(np.int64)
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.20,
                                  random_state=RANDOM_STATE, stratify=y)
X_tr, X_te, y_tr, y_te = X[idx_tr], X[idx_te], y[idx_tr]-1, y[idx_te]
print('train:', len(idx_tr), ' test:', len(idx_te))"""),
    md("""## 4. Train Random Forest and XGBoost (2015)

- RF: `class_weight='balanced'` compensates for minority classes.
- XGBoost: `hist` tree method (fast on 260k+ rows), `mlogloss` metric.
- `Year` is deliberately **not** a feature — it is metadata, not spectral information."""),
    code("""rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                            n_jobs=-1, class_weight='balanced')
xgbc = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                         n_jobs=-1, tree_method='hist', eval_metric='mlogloss')

t0=time.time(); rf.fit(X_tr, y_tr);   print(f'RF fit: {time.time()-t0:.1f}s')
t0=time.time(); xgbc.fit(X_tr, y_tr); print(f'XGB fit: {time.time()-t0:.1f}s')"""),
    md("## 5. Evaluate (2015) — accuracy is never enough"),
    code("""def evaluate(name, model, X_te, y_te):
    pred = model.predict(X_te) + 1          # back to 1..4
    print(f'--- {name}: accuracy={accuracy_score(y_te,pred):.4f}  '
          f'kappa={cohen_kappa_score(y_te,pred):.4f}  '
          f'macroF1={f1_score(y_te,pred,average="macro"):.4f}')
    print(classification_report(y_te, pred, labels=[1,2,3,4], zero_division=0,
          target_names=[CLASS_NAMES[i] for i in (1,2,3,4)]))
    return pred

pred_rf  = evaluate('RF',  rf,   X_te, y_te)
pred_xgb = evaluate('XGB', xgbc, X_te, y_te)"""),
    code("""cm = confusion_matrix(y_te, pred_xgb, labels=[1,2,3,4])
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(4), [CLASS_NAMES[i] for i in (1,2,3,4)], rotation=30, ha='right')
ax.set_yticks(range(4), [CLASS_NAMES[i] for i in (1,2,3,4)])
for r in range(4):
    for c in range(4):
        ax.text(c, r, cm[r,c], ha='center', va='center',
                color='white' if cm[r,c] > cm.max()/2 else 'black')
ax.set_title('XGBoost 2015 — confusion matrix'); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
fig.colorbar(im, fraction=0.046); plt.tight_layout(); plt.show()"""),
    md("""## 6. All six experiments (2015/2020/2025 × RF/XGB)

The same procedure is repeated per year; results and models are saved to `results/` and `models/`."""),
    code("""rows = []
for year, fname in FILES.items():
    d = pd.read_csv(DATA/fname).drop_duplicates(subset=FEATURES+['Label']).reset_index(drop=True)
    Xd, yd = d[FEATURES].to_numpy(np.float32), d['Label'].to_numpy(np.int64)
    i_tr, i_te = train_test_split(np.arange(len(yd)), test_size=0.20,
                                  random_state=RANDOM_STATE, stratify=yd)
    for name, mdl in [('RF', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                                    n_jobs=-1, class_weight='balanced')),
                      ('XGB', xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                                subsample=0.8, colsample_bytree=0.8,
                                                random_state=RANDOM_STATE, n_jobs=-1,
                                                tree_method='hist', eval_metric='mlogloss'))]:
        t0 = time.time(); mdl.fit(Xd[i_tr], yd[i_tr]-1); fit_s = time.time()-t0
        p = mdl.predict(Xd[i_te]) + 1
        rows.append(dict(year=year, model=name,
                         accuracy=round(accuracy_score(yd[i_te], p), 4),
                         kappa=round(cohen_kappa_score(yd[i_te], p), 4),
                         macro_f1=round(f1_score(yd[i_te], p, average='macro'), 4),
                         fit_seconds=round(fit_s, 1)))
        joblib.dump(mdl, MODELS/f'{name}_{year}.pkl')

summary = pd.DataFrame(rows)
summary.to_csv(RESULTS/'accuracy_results.csv', index=False)
summary"""),
    code("""ax = summary.pivot(index='year', columns='model', values='kappa').plot(
    kind='bar', title="Cohen's Kappa by year and model", ylabel='kappa')
ax.set_ylim(0.6, 0.9); plt.tight_layout(); plt.show()"""),
    md("## 7. Feature importance (RF 2015)"),
    code("""imp = pd.DataFrame({'feature': FEATURES,
                    'importance': joblib.load(MODELS/'RF_2015.pkl').feature_importances_}
                  ).sort_values('importance')
ax = imp.plot(kind='barh', x='feature', y='importance', legend=False,
              title='Random Forest 2015 — feature importance')
plt.tight_layout(); plt.show()
# Note: NDVI/NDBI/NDWI are derived from the bands, so importance is shared between
# correlated features. Do not read this as physical causality."""),
    md("""## 8. Honest conclusions & next steps

**Results (stratified random split):** XGBoost slightly beats RF everywhere; Kappa 0.68 (2015) → 0.80–0.81 (2020/2025). Weakest spot: Built-up recall in 2015 (~0.67) — vegetation dominance confuses built-up pixels.

**For the report/viva, state clearly:**
1. Scores are from random pixel splits; spatial autocorrelation likely inflates them (no coordinates available).
2. Label provenance must be confirmed with Member 1 (circularity risk).
3. Class imbalance handled via class weights; Open Land/Water remain the hardest classes.

**Next phases:** raster-wide prediction (needs the .tif files still downloading), area statistics, 2015→2020→2025 change matrices."""),
]

nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata.language_info = {"name": "python"}

out = Path(__file__).resolve().parents[1] / "notebooks" / "Member2_LULC_ML.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("notebook written:", out)
