"""
GeoRAG - Member 2 - Final consolidated results (checkpointed)
=============================================================
Recomputes the six same-year experiments AND the temporal experiments from
scratch and writes one master table:

    results/final_model_results.csv
    (Training_Years, Test_Year, Model, Accuracy, Precision_Macro,
     Recall_Macro, F1_Macro, Kappa)

Usage (three passes, each well under the 10-minute command limit):

    python src/final_results.py sameyear    # ~3-4 min (RF on 2015 is the big one)
    python src/final_results.py temporal1   # train 2015 -> test 2020+2025 (~4 min)
    python src/final_results.py temporal2   # train 2015+2020 -> test 2025 (~4 min)
    python src/final_results.py merge       # combine + reproducibility check

Methodology (identical to earlier runs, fixed random_state=42):
- predictors: only the 9 spectral features (no Label, no Year)
- dedupe before splitting; stratified 80/20 for same-year runs
- temporal runs train on ALL rows of the training year(s) (that is the
  experiment's definition), test on whole unseen years
- no preprocessing, no tuning: test labels are never used for anything
  except final scoring
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CSV = {2015: ROOT / "data/training_2015.csv",
       2020: ROOT / "data/training_2020.csv",
       2025: ROOT / "data/training_2025.csv"}
FEATURES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDBI", "NDWI"]


def load(y):
    df = pd.read_csv(CSV[y]).drop_duplicates(subset=FEATURES + ["Label"])
    return df[FEATURES].to_numpy(np.float32), df["Label"].to_numpy(np.int64)


def models():
    return [
        ("RF", RandomForestClassifier(n_estimators=200, random_state=42,
                                      n_jobs=-1, class_weight="balanced")),
        ("XGB", xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  n_jobs=-1, tree_method="hist", eval_metric="mlogloss")),
    ]


def score(train_label, ty, name, mdl, Xte, yte, fit_s):
    pred = mdl.predict(Xte) + 1
    return dict(Training_Years=train_label, Test_Year=ty, Model=name,
                Accuracy=round(accuracy_score(yte, pred), 4),
                Precision_Macro=round(precision_score(yte, pred, average="macro"), 4),
                Recall_Macro=round(recall_score(yte, pred, average="macro"), 4),
                F1_Macro=round(f1_score(yte, pred, average="macro"), 4),
                Kappa=round(cohen_kappa_score(yte, pred), 4),
                fit_seconds=round(fit_s, 1))


rows = []
mode = sys.argv[1]

cache = {y: load(y) for y in (2015, 2020, 2025)}

if mode == "sameyear":
    for y in (2015, 2020, 2025):
        X, yv = cache[y]
        i_tr, i_te = train_test_split(np.arange(len(yv)), test_size=0.20,
                                      random_state=42, stratify=yv)
        for name, mdl in models():
            t0 = time.time()
            mdl.fit(X[i_tr], yv[i_tr] - 1)
            dt = time.time() - t0
            rows.append(score(str(y), y, name, mdl, X[i_te], yv[i_te], dt))
            print(f"{y} {name}: acc={rows[-1]['Accuracy']} F1m={rows[-1]['F1_Macro']} "
                  f"kappa={rows[-1]['Kappa']} ({dt:.0f}s)")

elif mode == "temporal1":
    Xtr = cache[2015][0]
    ytr = cache[2015][1] - 1
    print(f"train 2015: {len(Xtr):,} rows")
    for name, mdl in models():
        t0 = time.time()
        mdl.fit(Xtr, ytr)
        dt = time.time() - t0
        print(f"  {name} fit {dt:.0f}s")
        for ty in (2020, 2025):
            Xte, yte = cache[ty]
            rows.append(score("2015", ty, name, mdl, Xte, yte, dt))
            print(f"  -> {ty}: acc={rows[-1]['Accuracy']} F1m={rows[-1]['F1_Macro']}")

elif mode == "temporal2":
    Xtr = np.vstack([cache[2015][0], cache[2020][0]])
    ytr = np.hstack([cache[2015][1] - 1, cache[2020][1] - 1])
    print(f"train 2015+2020: {len(Xtr):,} rows")
    for name, mdl in models():
        t0 = time.time()
        mdl.fit(Xtr, ytr)
        dt = time.time() - t0
        print(f"  {name} fit {dt:.0f}s")
        Xte, yte = cache[2025]
        rows.append(score("2015+2020", 2025, name, mdl, Xte, yte, dt))
        print(f"  -> 2025: acc={rows[-1]['Accuracy']} F1m={rows[-1]['F1_Macro']}")

elif mode == "merge":
    parts = []
    for p in ["final_sameyear.csv", "final_temporal1.csv", "final_temporal2.csv"]:
        fp = RES / p
        if fp.exists():
            parts.append(pd.read_csv(fp))
    combined = pd.concat(parts, ignore_index=True)
    order = {"2015": 0, "2020": 1, "2025": 2, "2015+2020": 3}
    combined = combined.sort_values(
        by=["Training_Years", "Test_Year", "Model"],
        key=lambda s: s.map(order) if s.name == "Training_Years" else s).reset_index(drop=True)
    combined.to_csv(RES / "final_model_results.csv", index=False)
    print("Saved", RES / "final_model_results.csv")
    print(combined.drop(columns=["fit_seconds"]).to_string(index=False))

    # ---- reproducibility check vs the earlier temporal run
    print("\n== REPRODUCIBILITY: temporal metrics vs earlier run ==")
    prev = pd.read_csv(RES / "temporal_validation_results.csv")
    ok_all = True
    for _, r in prev.iterrows():
        m = combined[(combined.Training_Years == r.train_years)
                     & (combined.Test_Year == int(r.test_year))
                     & (combined.Model == r.model)]
        if m.empty:
            continue
        match = abs(float(r.accuracy) - float(m.Accuracy.iloc[0])) < 1e-6
        ok_all &= match
        print(f"  train {r.train_years} -> test {int(r.test_year)} [{r.model}]: "
              f"earlier={r.accuracy:.4f} recomputed={m.Accuracy.iloc[0]:.4f} "
              f"{'MATCH' if match else 'MISMATCH'}")
    print("RESULT:", "ALL TEMPORAL RESULTS REPRODUCIBLE" if ok_all
          else "REPRODUCIBILITY FAILURE")

else:
    sys.exit("usage: python src/final_results.py [sameyear|temporal1|temporal2|merge]")

if mode != "merge":
    out = RES / {"sameyear": "final_sameyear.csv",
                 "temporal1": "final_temporal1.csv",
                 "temporal2": "final_temporal2.csv"}[mode]
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved", out)
