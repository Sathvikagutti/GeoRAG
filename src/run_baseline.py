"""
GeoRAG - Member 2 - Phase B: Baseline ML (Random Forest + XGBoost)
==================================================================
Per-year LULC classification baseline.

Design decisions (documented for the project report):
- Deduplicate exact duplicate rows BEFORE splitting (prevents train/test leakage).
- Stratified 80/20 split (preserves class proportions).
- Labels remapped 1..4 -> 0..3 internally for XGBoost; reports use original codes.
  1 = Built-up, 2 = Vegetation, 3 = Water, 4 = Open Land
- 'Year' is NOT used as a feature (metadata, constant within each year file).
- No spatial coordinates exist in the CSVs -> limitation documented, not hidden.
- Known limitation: random pixel split may be optimistic (spatial autocorrelation).

Run from project root:  python src/run_baseline.py
"""

import time
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS, RESULTS = ROOT / "data", ROOT / "models", ROOT / "results"
RESULTS.mkdir(exist_ok=True)

FILES = {
    2015: DATA / "training_2015.csv",
    2020: DATA / "training_2020.csv",
    2025: DATA / "training_2025.csv",
}
FEATURES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDBI", "NDWI"]
TARGET = "Label"
CLASS_NAMES = {1: "Built-up", 2: "Vegetation", 3: "Water", 4: "Open Land"}
RANDOM_STATE = 42

summary_rows = []


def train_rf(X_tr, y_tr):
    m = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    t0 = time.time()
    m.fit(X_tr, y_tr)
    return m, time.time() - t0


def train_xgb(X_tr, y_tr):
    m = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss",
    )
    t0 = time.time()
    m.fit(X_tr, y_tr)
    return m, time.time() - t0


for year, path in FILES.items():
    print("=" * 70)
    print(f"YEAR {year} - {path.name}")
    df = pd.read_csv(path)
    n_raw = len(df)

    # 1) Deduplicate exact duplicates (feature values + label identical)
    df = df.drop_duplicates(subset=FEATURES + [TARGET]).reset_index(drop=True)
    print(f"rows: {n_raw} -> {len(df)} after dedupe ({n_raw - len(df)} removed)")

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.int64)          # 1..4
    y_enc = y - 1                                    # 0..3 for XGBoost

    # 2) Stratified split
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr_enc, y_te_enc = y_enc[idx_tr], y_enc[idx_te]
    y_te = y[idx_te]
    print(f"train={len(idx_tr)}  test={len(idx_te)}")

    for name, trainer in [("RF", train_rf), ("XGB", train_xgb)]:
        model, fit_s = trainer(X_tr, y_tr_enc)
        t1 = time.time()
        pred_enc = model.predict(X_te)
        pred_s = time.time() - t1
        pred = pred_enc + 1                                    # back to 1..4

        acc = accuracy_score(y_te, pred)
        kappa = cohen_kappa_score(y_te, pred)
        mf1 = f1_score(y_te, pred, average="macro")

        tag = f"{year}_{name}"
        print(f"\n--- {name}: accuracy={acc:.4f}  kappa={kappa:.4f}  macroF1={mf1:.4f}  "
              f"(fit {fit_s:.1f}s, predict {pred_s:.1f}s)")

        # classification report
        (RESULTS / f"classification_report_{tag}.txt").write_text(
            classification_report(
                y_te, pred,
                labels=[1, 2, 3, 4],
                target_names=[CLASS_NAMES[i] for i in (1, 2, 3, 4)],
                digits=4,
                zero_division=0,
            )
        )

        # confusion matrix CSV
        cm = confusion_matrix(y_te, pred, labels=[1, 2, 3, 4])
        pd.DataFrame(
            cm,
            index=[f"actual_{CLASS_NAMES[i]}" for i in (1, 2, 3, 4)],
            columns=[f"pred_{CLASS_NAMES[i]}" for i in (1, 2, 3, 4)],
        ).to_csv(RESULTS / f"confusion_matrix_{tag}.csv")

        # confusion matrix plot
        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(4), [CLASS_NAMES[i] for i in (1, 2, 3, 4)], rotation=30, ha="right")
        ax.set_yticks(range(4), [CLASS_NAMES[i] for i in (1, 2, 3, 4)])
        for r in range(4):
            for c in range(4):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=9)
        ax.set_title(f"{name} {year} - confusion matrix\nacc={acc:.3f} kappa={kappa:.3f}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fig.savefig(RESULTS / f"confusion_matrix_{tag}.png", dpi=150)
        plt.close(fig)

        # feature importance plot (RF: gini impurity; XGB: gain)
        if name == "RF":
            imp = model.feature_importances_
        else:
            scores = model.get_booster().get_score(importance_type="gain")
            imp = np.array([scores.get(f"f{i}", 0.0) for i in range(len(FEATURES))])
        fi_df = pd.DataFrame({"feature": FEATURES, "importance": imp}).sort_values("importance")
        ax = fi_df.plot(kind="barh", x="feature", y="importance", legend=False, figsize=(6, 4))
        ax.set_title(f"{name} {year} - feature importance")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(RESULTS / f"feature_importance_{tag}.png", dpi=150)
        plt.close(fig)
        fi_df.assign(year=year, model=name).to_csv(
            RESULTS / f"feature_importance_{tag}.csv", index=False
        )

        # model artifact
        joblib.dump(model, MODELS / f"{name}_{year}.pkl")

        summary_rows.append(
            dict(year=year, model=name, accuracy=round(acc, 4), kappa=round(kappa, 4),
                 macro_f1=round(mf1, 4), train_rows=len(idx_tr), test_rows=len(idx_te),
                 rows_raw=n_raw, rows_after_dedupe=len(df), fit_seconds=round(fit_s, 1))
        )

# ---------------------------------------------------------------- summary
summary = pd.DataFrame(summary_rows)
summary.to_csv(RESULTS / "accuracy_results.csv", index=False)
print("\n" + "=" * 70)
print("SUMMARY (stratified random split - see limitations in script header)")
print(summary.to_string(index=False))
print("\nArtifacts in results/, models/. Done.")
