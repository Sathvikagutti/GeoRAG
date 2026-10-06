# GeoRAG — Member 2: ML + Urban Growth Analysis

**GitHub Repository:** [https://github.com/Sathvikagutti/GeoRAG](https://github.com/Sathvikagutti/GeoRAG)

Member 2 module of **GeoRAG** (AI-powered geospatial decision support for
sustainable land-use planning, Vijayawada): **LULC classification with Random
Forest and XGBoost → accuracy assessment → LULC map prediction → temporal
validation → change detection → urban-growth analysis**.

---

## Quickstart & Environment Setup (using `uv`)

This project uses the modern Python package manager **`uv`**.

```bash
# 1. Clone the repository
git clone https://github.com/Sathvikagutti/GeoRAG.git
cd GeoRAG

# 2. Sync virtual environment and install exact dependencies
uv sync

# (Alternative) If using standard pip:
pip install -r requirements.txt

# 3. Run automated end-to-end pipeline integrity test suite
uv run python src/test_pipeline_e2e.py

# 4. Run reproducibility verification (6/6 exact match check)
uv run python src/verify_reproducibility.py
```

---

## 1. Project objective and Member 2 responsibilities

Build a **scientifically defensible** ML pipeline that (a) classifies Landsat
pixels into four land-use/land-cover classes, (b) reports honest accuracy with
per-class metrics, (c) produces georeferenced LULC maps for the GIS team, and
(d) quantifies urban growth 2015→2020→2025 from transitions into Built-up.

Responsibilities covered by this workspace:

1. Data audit of Member 1's training CSVs
2. Random Forest + XGBoost classification (3 years)
3. Same-year validation (stratified random pixel split)
4. Temporal validation (train past → test future)
5. Raster-wide LULC prediction + comparison against Member 1's label rasters
6. Area statistics, transition matrices, change maps, urban-growth analysis

## 2. Input datasets (Member 1 — never modified; MD5s recorded in the audit)

| File | Rows / size | Role |
|---|---|---|
| `data/training_2015.csv` | 347,915 | pixel training samples 2015 |
| `data/training_2020.csv` | 57,437 | pixel training samples 2020 |
| `data/training_2025.csv` | 25,806 | pixel training samples 2025 |
| `data/training_all_years.csv` | 431,158 | combined copy of the above |
| `data/Member1_Final_QC_Report.txt` | — | Member 1's QC (checks, not methods) |
| `GeoRag/GeoRAG_Satellite/*.tif` | 7-band Landsat, 30 m, EPSG:4326 | prediction input |
| `GeoRag/GeoRAG_LandUse_Labels/*.tif` | 10 m, values 0–4 | reference maps |

## 3. Features and classes

- **Predictors (exactly 9, nothing else):** Blue, Green, Red, NIR, SWIR1,
  SWIR2, NDVI, NDBI, NDWI. The three indices are deterministic functions of
  the bands (verified); they are inputs, not labels.
- **Excluded from predictors:** `Label` (the target) and `Year` (metadata;
  used only to define temporal experiments).
- **Classes:** 1 = Built-up, 2 = Vegetation, 3 = Water, 4 = Open Land.

## 4. Random Forest methodology

`sklearn.ensemble.RandomForestClassifier`, 200 trees, `class_weight="balanced"`
(justified: documented class imbalance, e.g. Open Land 2.8% in 2015),
`n_jobs=-1`, `random_state=42` (reproducibility). No preprocessing (trees use
raw features), no hyperparameter search (test labels never used for tuning).

## 5. XGBoost methodology

`xgboost.XGBClassifier`, 300 trees, `max_depth=6`, `learning_rate=0.05`,
`subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`, `eval_metric=
"mlogloss"`, `random_state=42`. Labels internally remapped 1..4 → 0..3
(library requirement); all reported metrics use original codes. No tuning.

## 6. Same-year validation — methodology statement

For each year: **deduplicate exact duplicate rows before splitting**, then a
**stratified random pixel split** (80% train / 20% test, class proportions
preserved, `random_state=42`).

> **This is NOT spatial validation.** Pixels are split randomly; neighboring
> pixels can appear in both sets (spatial autocorrelation), so these scores
> are likely optimistic. **Spatially independent (block) validation was NOT
> implemented** because the CSVs contain no coordinates — stated as a
> limitation, not hidden. Temporal validation is our partial mitigation.

Reported metrics: Accuracy, Precision (macro), Recall (macro), F1 (macro and
per-class), Cohen's Kappa, confusion matrices (CSV + PNG), feature importance.

## 7. Temporal validation — methodology statement

Models train on **all rows** of the training year(s) and are tested on whole
**unseen future years** (no test-year rows in training, no tuning on test):

| Experiment | Train | Test |
|---|---|---|
| T1 | 2015 | 2020 |
| T2 | 2015 | 2025 |
| T3 | 2015 + 2020 | 2025 |

Each model is trained **once per training set** and reused across test years.

## 8. Accuracy results (all measured; `results/final_model_results.csv`)

### Same-year (stratified random pixel split — optimistic, see §6)

| Train→Test | Model | Accuracy | Prec(m) | Rec(m) | F1(m) | Kappa |
|---|---|---|---|---|---|---|
| 2015→2015 | RF | 0.8402 | 0.8648 | 0.7442 | 0.7937 | 0.6823 |
| 2015→2015 | XGB | 0.8411 | 0.8652 | 0.7476 | 0.7957 | 0.6836 |
| 2020→2020 | RF | 0.8923 | 0.9027 | 0.8438 | 0.8708 | 0.7847 |
| 2020→2020 | XGB | 0.8993 | 0.9060 | 0.8592 | 0.8808 | 0.8006 |
| 2025→2025 | RF | 0.8678 | 0.8940 | 0.8745 | 0.8836 | 0.8046 |
| 2025→2025 | XGB | 0.8739 | 0.8963 | 0.8807 | 0.8877 | 0.8141 |

### Temporal (train past → test future; exact reproduction verified 6/6)

| Train→Test | Model | Accuracy | F1(m) | Kappa |
|---|---|---|---|---|
| 2015→2020 | RF | 0.7600 | 0.7734 | 0.5895 |
| 2015→2020 | XGB | 0.7616 | 0.7797 | 0.5943 |
| 2015→2025 | RF | 0.7766 | 0.8091 | 0.6756 |
| 2015→2025 | XGB | 0.7817 | 0.8156 | 0.6841 |
| 2015+2020→2025 | RF | 0.8210 | 0.8480 | 0.7387 |
| 2015+2020→2025 | XGB | 0.8145 | 0.8433 | 0.7305 |

Forward-prediction skill drops ~8–13 points vs same-year scores and improves
as training years are added — evidence of genuine temporal generalization.

## 9. LULC raster outputs (`predictions/`, all verified in Step 8)

| Raster | Grid | Notes |
|---|---|---|
| `LULC_pred_2015_XGB.tif` / `_RF.tif` | 693×669 | full city, aligned to `Vijayawada_Landsat_2015.tif` |
| `LULC_pred_2020_XGB.tif` / `_RF.tif` | 693×669 | full city, aligned to `Vijayawada_Landsat_2020.tif` |
| `LULC_pred_2025_XGB_window.tif` | 362×392 | **healthy window only** (2025.tif is ~68% empty) |
| `Change_labels_<a>_<b>.tif` | 2077×2004 | transition codes from*10+to (21 = Veg→Built-up), 255 = nodata |

All: EPSG:4326, classes ∈ {1,2,3,4}, nodata 255, pixel-perfect alignment with
their source rasters (`results/output_verification.csv`).

**Map agreement with Member 1's label rasters** (computed live in figures;
per-pixel, inside city, valid pixels only): 2015 XGB 84.8%, RF 96.7%;
2020 XGB 79.6%, RF 78.9%; 2025 window XGB 90.6%.
**This agreement is a consistency check, NOT independent accuracy** — RF's
96.7% in 2015 suggests labels are highly reproducible from the same imagery.

## 10. Change detection and urban growth (`results/change_matrix_*.csv`, `transition_areas.csv`)

Transition matrices and per-transition areas (10 m label rasters). Built-up
**expansion** (transitions INTO Built-up, not net change):

| Period | V→Built-up (px) | Total into Built-up (km²) |
|---|---|---|
| 2015→2020 | 166,944 | 16.50 |
| 2020→2025 | 144,302 | 14.55 |
| 2015→2025 | 264,075 | 26.09 |

Built-up→Vegetation is **not** counted as urban expansion anywhere.

## 11. Area statistics (verified two independent ways)

Areas from pixel counts × verified pixel area; the cos(latitude) degree-based
method agrees with on-the-fly **UTM 44N reprojection within 0.4–1.0%**
(`results/area_crosscheck_utm.csv`). Label-raster areas (10 m):

| Class | 2015 | 2020 | 2025 | Δ 15–25 |
|---|---|---|---|---|
| Built-up | 89.25 km² | 103.39 km² | 113.22 km² | **+23.97** |
| Vegetation | 183.36 km² | 170.26 km² | 160.57 km² | **−22.79** |
| Water | 19.85 km² | 26.16 km² | 24.95 km² | +5.10 |
| Open Land | 8.34 km² | 11.67 km² | 12.72 km² | +4.38 |

## 12. Feature importance

RF impurity-based importance per year in `results/feature_importance_<year>_RF.csv`
(+ XGBoost gain-based variant). **Importance ≠ causality**: indices derive from
the bands, so importance is shared among correlated features; this describes
usefulness under this model setup, not physical causation.

## 13. How to re-run everything

```bash
python src/run_baseline.py            # Phase B: dedupe -> split -> RF+XGB per year
python src/predict_lulc.py            # Phase D: pixel-wise prediction + validation
python src/change_analysis.py         # Phase E: matrices, areas, change maps
python src/temporal_validation.py     # temporal experiments (earlier run)
python src/final_results.py sameyear  # consolidated recomputation (3 passes)
python src/final_results.py temporal1
python src/final_results.py temporal2
python src/final_results.py merge     # -> results/final_model_results.csv
python src/verify_reproducibility.py  # 6/6 exact-match check
python src/transition_areas.py        # transition areas in km2
python src/verify_outputs.py          # raster QA + UTM area cross-check
python src/make_figures.py            # slide figures (agreements computed live)
python src/make_report.py             # rebuild reports/PhaseD_summary.md
```

Deterministic (`random_state=42`). `src/inspect_rasters*.py` are read-only
audits. All scripts are read-only w.r.t. `data/` and `GeoRag/`.

## 14. Limitations (state these explicitly in the report)

1. **Label provenance could not be independently verified** from available
   project materials; the 2015 RF agreement (96.7%) and exact CSV/map class
   match in 2015 are consistent with labels derived from the same imagery,
   but this remains unconfirmed.
2. **No spatially independent validation** — CSVs lack coordinates; random
   pixel splits are optimistic; temporal validation is the mitigation.
3. **2025 satellite image ~68% empty** → 2025 LULC map covers the healthy
   window only; 2025 areas come from the label rasters.
4. **2020 CSV imbalance** (66% Built-up vs 33% in the map) degrades 2020
   calibration and map agreement.
5. **Year-wise sample-count difference is not fully explainable** from
   available materials: 2015's CSV matches the label map (11.09% subsample,
   same class mix); 2020/2025 use class-stratified sampling (fractions
   1.77% / 0.79%) whose design is undocumented.
6. Area conversion uses a local planar approximation (verified against UTM
   within 1%).
7. XGBoost gain importance is reported alongside RF impurity importance;
   both are model-dependent, neither is causal.

## 15. Handoff to Member 4 (GIS)

- **Maps:** `predictions/LULC_pred_2015_XGB.tif`, `LULC_pred_2020_XGB.tif`,
  `LULC_pred_2025_XGB_window.tif` (uint8, 255=nodata, EPSG:4326)
- **Change rasters:** `predictions/Change_labels_<a>_<b>.tif` (code = from*10+to)
- **Tables:** `results/urban_growth_summary.csv`, `transition_areas.csv`,
  `change_matrix_labels_*.csv`, `area_statistics_labels_*.csv`,
  `final_model_results.csv`, `area_crosscheck_utm.csv`
- **Figures:** `maps/figures/fig1–fig4` (captions state validation type)
- **Summary:** `reports/PhaseD_summary.md`; **full handoff text:**
  `results/Member2_Final_Handoff.txt`

## 16. Open questions for Member 1

1. How exactly were the label rasters produced (visual interpretation?
   existing product? index thresholds? classifier)?
2. Can the full 2025 satellite image be provided?
3. What sampling design produced the 2020/2025 CSVs (why 66% Built-up)?
