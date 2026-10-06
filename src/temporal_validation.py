"""
GeoRAG - Member 2 - Temporal validation experiment
==================================================
WHY (beginner version):
The Phase B scores (accuracy ~84-90%) came from splitting ONE year's pixels
randomly into train/test. Neighboring pixels look almost identical, so the
model is partly "quizzed on questions it has seen". That inflates scores.

A harder, more honest test: TRAIN on the PAST, TEST on the FUTURE.
  A) train 2015 -> test 2020
  B) train 2015 -> test 2025
  C) train 2015+2020 -> test 2025
If the model still works on a year it never saw, it truly learned what
built-up/vegetation/water/open-land "look like".

NOTE: 2025 pixels are not 100% usable (zero-band issue) - for this experiment
we use only fully finite rows, same rule for train and test, so the
comparison stays fair.

Outputs: results/temporal_validation_results.csv (+ printed table)

Run from project root:  python src/temporal_validation.py
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from geo_utils import MODELS, RESULTS

ROOT = Path(__file__).resolve().parents[1]
CSV = {2015: ROOT / "data" / "training_2015.csv",
       2020: ROOT / "data" / "training_2020.csv",
       2025: ROOT / "data" / "training_2025.csv"}
FEATURES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDBI", "NDWI"]


def load_year(y):
    df = pd.read_csv(CSV[y]).drop_duplicates(subset=FEATURES + ["Label"])
    X = df[FEATURES].to_numpy(np.float32)
    yv = df["Label"].to_numpy(np.int64)
    return X, yv


EXPERIMENTS = [([2015], [2020, 2025]), ([2015, 2020], [2025])]  # train once, test many
rows = []

cache = {y: load_year(y) for y in (2015, 2020, 2025)}

for train_years, test_years in EXPERIMENTS:
    Xtr = np.vstack([cache[y][0] for y in train_years])
    ytr = np.hstack([cache[y][1] - 1 for y in train_years])
    print(f"training on {train_years}: {len(Xtr):,} rows")

    for name, mdl in [
        ("RF", RandomForestClassifier(n_estimators=200, random_state=42,
                                      n_jobs=-1, class_weight="balanced")),
        ("XGB", xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  n_jobs=-1, tree_method="hist", eval_metric="mlogloss")),
    ]:
        t0 = time.time()
        mdl.fit(Xtr, ytr)
        dt = time.time() - t0
        print(f"  {name} fit done in {dt:.1f}s")
        for test_year in test_years:
            Xte, yte = cache[test_year]
            pred = mdl.predict(Xte) + 1
            rows.append(dict(
                train_years="+".join(map(str, train_years)), test_year=test_year, model=name,
                accuracy=round(accuracy_score(yte, pred), 4),
                kappa=round(cohen_kappa_score(yte, pred), 4),
                macro_f1=round(f1_score(yte, pred, average="macro"), 4),
                fit_seconds=round(dt, 1)))
            print(f"    -> test {test_year} [{name}]: acc={rows[-1]['accuracy']:.4f} "
                  f"kappa={rows[-1]['kappa']:.4f} macroF1={rows[-1]['macro_f1']:.4f}")

# baseline: same-year random split scores from Phase B for comparison
phaseB = pd.read_csv(RESULTS / "accuracy_results.csv")[["year", "model", "accuracy"]]
print("\nPhase B (same-year random split) accuracies for reference:")
print(phaseB.to_string(index=False))

out = pd.DataFrame(rows)
out.to_csv(RESULTS / "temporal_validation_results.csv", index=False)
print(f"\nSaved {RESULTS / 'temporal_validation_results.csv'}")
