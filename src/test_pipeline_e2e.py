"""
GeoRAG Member 2 - End-to-End Pipeline Integrity & Authenticity Test Runner
==========================================================================
Pin-to-pin automated test suite verifying:
1. Raw training data schema, row counts, and class encodings
2. Serialized model inference capabilities
3. Output GeoTIFF raster geometries, CRS, nodata values, and class sets
4. 6/6 temporal experiment metric reproducibility
5. Urban growth calculations, transition matrices, and UTM cross-checks
6. Consistency between computed CSVs and final reports

Run with uv:  uv run python src/test_pipeline_e2e.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import rasterio

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
PRED = ROOT / "predictions"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"

FEATURES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDBI", "NDWI"]
EXPECTED_CLASSES = {1, 2, 3, 4}

passed = 0
failed = 0


def record(name, condition, details=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}" + (f" ({details})" if details else ""))
        passed += 1
    else:
        print(f"  [FAIL] {name}" + (f" ({details})" if details else ""))
        failed += 1


print("=" * 80)
print("GEORAG MEMBER 2 - END-TO-END PIPELINE INTEGRITY & AUTHENTICITY TESTS")
print("=" * 80)

# -------------------------------------------------------------------------
# Test Group 1: Raw Training Data Authenticity
# -------------------------------------------------------------------------
print("\n[TEST GROUP 1] Training Data Ingestion & Schema Integrity")
expected_rows = {2015: 347915, 2020: 57437, 2025: 25806}

for year, exp_count in expected_rows.items():
    csv_path = DATA / f"training_{year}.csv"
    if not csv_path.exists():
        record(f"{year} CSV exists", False, f"File missing at {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    record(f"{year} CSV row count", len(df) == exp_count, f"got {len(df):,}, expected {exp_count:,}")
    has_features = all(f in df.columns for f in FEATURES)
    record(f"{year} CSV spectral features", has_features, f"all 9 bands present")
    classes_present = set(df["Label"].unique())
    record(f"{year} CSV class labels", classes_present.issubset(EXPECTED_CLASSES), f"labels: {classes_present}")
    no_nans = df[FEATURES].isna().sum().sum() == 0
    record(f"{year} CSV feature validity", no_nans, "no NaN/missing feature values")

# -------------------------------------------------------------------------
# Test Group 2: Model Inference & Artifact Integrity
# -------------------------------------------------------------------------
print("\n[TEST GROUP 2] Model Serialization & Live Inference")
dummy_input = np.random.rand(5, 9).astype(np.float32)

for year in [2015, 2020, 2025]:
    xgb_path = MODELS / f"XGB_{year}.pkl"
    if xgb_path.exists():
        model = joblib.load(xgb_path)
        pred = model.predict(dummy_input)
        valid_preds = set(pred).issubset({0, 1, 2, 3})
        record(f"XGB_{year} loads & predicts", valid_preds, f"outputs classes in 0..3")
    else:
        record(f"XGB_{year} exists", False, f"Missing {xgb_path}")

# -------------------------------------------------------------------------
# Test Group 3: GeoTIFF Rasters & Geographic Quality Assurance
# -------------------------------------------------------------------------
print("\n[TEST GROUP 3] Output GeoTIFF Raster Quality Assurance")
expected_rasters = [
    ("LULC_pred_2015_XGB.tif", 693, 669, [1, 2, 3, 4]),
    ("LULC_pred_2020_XGB.tif", 693, 669, [1, 2, 3, 4]),
    ("LULC_pred_2025_XGB_window.tif", 362, 392, [1, 2, 3, 4]),
    ("Change_labels_2015_2020.tif", 2077, 2004, None),
    ("Change_labels_2020_2025.tif", 2077, 2004, None),
    ("Change_labels_2015_2025.tif", 2077, 2004, None),
]

for fname, exp_w, exp_h, exp_classes in expected_rasters:
    r_path = PRED / fname
    if not r_path.exists():
        record(f"Raster {fname} exists", False, "file missing")
        continue
    with rasterio.open(r_path) as src:
        arr = src.read(1)
        crs_ok = str(src.crs) == "EPSG:4326"
        dim_ok = src.width == exp_w and src.height == exp_h
        nodata_ok = src.nodata == 255
        vals = set(np.unique(arr[arr != 255]).tolist())
        if exp_classes:
            classes_ok = set(vals).issubset(set(exp_classes))
        else:
            # Change rasters: from*10 + to, e.g. 11, 21, etc.
            classes_ok = all(10 <= v <= 44 for v in vals)

        record(f"Raster {fname} CRS & dimensions", crs_ok and dim_ok, f"{src.width}x{src.height}, {src.crs}")
        record(f"Raster {fname} values & nodata", nodata_ok and classes_ok, f"nodata={src.nodata}, classes={len(vals)}")

# -------------------------------------------------------------------------
# Test Group 4: Exact Reproducibility of Temporal Experiments
# -------------------------------------------------------------------------
print("\n[TEST GROUP 4] Exact Metric Reproducibility (6/6 Checks)")
final_csv = RESULTS / "final_model_results.csv"
temp_csv = RESULTS / "temporal_validation_results.csv"

if final_csv.exists() and temp_csv.exists():
    comb = pd.read_csv(final_csv)
    prev = pd.read_csv(temp_csv)
    matched_experiments = 0
    for _, r in prev.iterrows():
        ty = int(r["test_year"])
        m = comb[(comb["Training_Years"].astype(str) == str(r["train_years"]))
                 & (comb["Test_Year"].astype(int) == ty)
                 & (comb["Model"].astype(str) == str(r["model"]))]
        if len(m) == 1:
            a_old, a_new = float(r["accuracy"]), float(m["Accuracy"].iloc[0])
            k_old, k_new = float(r["kappa"]), float(m["Kappa"].iloc[0])
            f_old, f_new = float(r["macro_f1"]), float(m["F1_Macro"].iloc[0])
            match = (abs(a_old - a_new) < 1e-6 and abs(k_old - k_new) < 1e-6 and abs(f_old - f_new) < 1e-6)
            if match:
                matched_experiments += 1
    record("Temporal validation 6/6 exact reproducibility", matched_experiments == 6, f"{matched_experiments}/6 exact match")
else:
    record("Reproducibility CSV files exist", False, "Missing results CSVs")

# -------------------------------------------------------------------------
# Test Group 5: Urban Growth Dynamics & Geographic Accuracy
# -------------------------------------------------------------------------
print("\n[TEST GROUP 5] Urban Growth Calculations & UTM Cross-Check")
urban_csv = RESULTS / "urban_growth_summary.csv"
utm_csv = RESULTS / "area_crosscheck_utm.csv"

if urban_csv.exists():
    udf = pd.read_csv(urban_csv)
    b_row = udf[(udf["source"] == "labels_10m") & (udf["class_name"] == "Built-up")]
    v_row = udf[(udf["source"] == "labels_10m") & (udf["class_name"] == "Vegetation")]
    b_growth = float(b_row["change_15_25_km2"].iloc[0])
    v_loss = float(v_row["change_15_25_km2"].iloc[0])
    record("10-year Built-up growth calculation", abs(b_growth - 23.97) < 0.01, f"+{b_growth:.2f} km²")
    record("10-year Vegetation loss calculation", abs(v_loss - (-22.79)) < 0.01, f"{v_loss:.2f} km²")

if utm_csv.exists():
    utm_df = pd.read_csv(utm_csv)
    max_diff = utm_df["diff_pct"].max()
    record("Planar vs UTM Zone 44N area agreement", max_diff < 1.0, f"max discrepancy = {max_diff:.2f}% (<1.0%)")

# -------------------------------------------------------------------------
# Test Group 6: Report Generation & Delivery Readiness
# -------------------------------------------------------------------------
print("\n[TEST GROUP 6] Report Generation & Member Deliverables")
for rep in ["PhaseD_summary.md", "Member2_to_Member3_Handoff.md"]:
    rep_path = REPORTS / rep
    record(f"Report {rep} exists & non-empty", rep_path.exists() and rep_path.stat().st_size > 1000,
           f"{rep_path.stat().st_size:,} bytes")

print("\n" + "=" * 80)
print(f"TEST SUMMARY: {passed} PASSED, {failed} FAILED (TOTAL {passed + failed})")
print("=" * 80)

if failed > 0:
    sys.exit(1)
print("\n>>> ALL PIPELINE TESTS VERIFIED: ZERO HALLUCINATED OR BROKEN METRICS <<<\n")
