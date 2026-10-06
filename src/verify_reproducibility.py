"""
GeoRAG - Member 2 - Standalone reproducibility check
====================================================
Reads results/final_model_results.csv and results/temporal_validation_results.csv
FROM DISK and verifies every temporal experiment's accuracy matches exactly.
No retraining: this compares persisted, computed results.
"""
import pandas as pd

RES = "results/"
comb = pd.read_csv(RES + "final_model_results.csv")
prev = pd.read_csv(RES + "temporal_validation_results.csv")

ok_all = True
n_checked = 0
for _, r in prev.iterrows():
    ty = int(r["test_year"])
    m = comb[(comb["Training_Years"].astype(str) == str(r["train_years"]))
             & (comb["Test_Year"].astype(int) == ty)
             & (comb["Model"].astype(str) == str(r["model"]))]
    if len(m) != 1:
        print(f"  train {r['train_years']} -> test {ty} [{r['model']}]: "
              f"FOUND {len(m)} rows in final_model_results.csv (expected 1)")
        ok_all = False
        continue
    a_old, a_new = float(r["accuracy"]), float(m["Accuracy"].iloc[0])
    k_old, k_new = float(r["kappa"]), float(m["Kappa"].iloc[0])
    f_old, f_new = float(r["macro_f1"]), float(m["F1_Macro"].iloc[0])
    match = (abs(a_old - a_new) < 1e-6 and abs(k_old - k_new) < 1e-6
             and abs(f_old - f_new) < 1e-6)
    ok_all &= match
    n_checked += 1
    print(f"  train {r['train_years']} -> test {ty} [{r['model']}]: "
          f"acc {a_old:.4f}={a_new:.4f}, kappa {k_old:.4f}={k_new:.4f}, "
          f"F1m {f_old:.4f}={f_new:.4f}  {'MATCH' if match else 'MISMATCH'}")

print(f"\n{n_checked}/6 temporal experiments checked")
print("VERDICT:", "ALL TEMPORAL RESULTS REPRODUCIBLE (exact match)"
      if ok_all and n_checked == 6 else "REPRODUCIBILITY FAILURE - INVESTIGATE")
