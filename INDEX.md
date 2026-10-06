# GeoRAG — Member 2 Bundle (ML + Urban Growth)

Complete, verified Member 2 module: trained models, LULC maps, change analysis,
result tables, figures, full source code, and handoff documents.

**Classes:** 1 = Built-up · 2 = Vegetation · 3 = Water · 4 = Open Land
**Study area:** Vijayawada · **CRS:** EPSG:4326 · **Random seed:** 42 (all results reproducible)

---

## What's in this bundle

| Folder / file | Contents |
|---|---|
| `INDEX.md` | This file |
| `README.md` | Full methodology, results, limitations, how to re-run everything |
| `src/` | All 15 pipeline scripts (see execution order below) |
| `models/` | 3 XGBoost models (`.pkl`) that generated the delivered maps |
| `predictions/` | 9 final GeoTIFFs: LULC maps + change/transition rasters |
| `results/` | 57 result tables (accuracy, areas, transitions, QA) + Member 2→4 handoff text |
| `maps/` | 9 map previews + `figures/` with 4 slide-ready figures |
| `notebooks/` | Executed viva notebook (`Member2_LULC_ML.ipynb`) |
| `reports/` | `PhaseD_summary.md` (results) + `Member2_to_Member3_Handoff.md` (complete code + data handoff) |

**Not included (deliberately):**
- `data/` and `GeoRag/` — Member 1's inputs; they live in Member 1's workspace (never modified by Member 2).
- `models/RF_*.pkl` — the Random Forest binary for 2015 alone is 1.3 GB; the XGBoost models (12 MB total) produced the delivered maps, and the RF models retrain deterministically in ~3 minutes via `python src/run_baseline.py`.

---

## Start here, by role

- **Member 3 (RAG / decision support):** open `reports/Member2_to_Member3_Handoff.md` — it contains the complete source code of all 15 scripts (byte-verified), execution order, data contract, class/change encodings, usage steps, and limitations. You need nothing else.
- **Member 4 (GIS / suitability):** open `results/Member2_Final_Handoff.txt` (sections A–P) and load the GeoTIFFs from `predictions/` in QGIS. Class colors: red = Built-up, green = Vegetation, blue = Water, yellow = Open Land; 255 = NoData.
- **Supervisor / report writing:** `README.md` (methodology + limitations) and `reports/PhaseD_summary.md` (all result tables).
- **Viva prep:** `notebooks/Member2_LULC_ML.ipynb` (executed, outputs visible) + `maps/figures/fig1..fig4`.

---

## Execution order (reproduce everything from scratch)

```bash
python src/run_baseline.py            # 1. train RF+XGB per year + same-year validation
python src/temporal_validation.py     # 2. train past -> test future
python src/final_results.py sameyear  # 3. consolidated recomputation (3 passes)
python src/final_results.py temporal1
python src/final_results.py temporal2
python src/final_results.py merge     #    -> results/final_model_results.csv
python src/verify_reproducibility.py  # 4. 6/6 exact-match proof
python src/predict_lulc.py            # 5. raster-wide LULC prediction + validation
python src/change_analysis.py         # 6. transition matrices + change maps
python src/transition_areas.py        # 7. transitions in km2
python src/verify_outputs.py          # 8. raster QA + UTM area cross-check
python src/make_report.py             # 9. reports/PhaseD_summary.md
python src/make_handoff.py            #    results/Member2_Final_Handoff.txt
python src/make_figures.py            # 10. slide figures
```

Requires: Python 3.13, pandas, numpy, scikit-learn, xgboost, rasterio, matplotlib,
joblib, tabulate. Member 1's `data/` and `GeoRag/` folders must sit in the project
root (not part of this bundle).

---

## Headline results

| Result | Value |
|---|---|
| Same-year accuracy (XGB, random pixel split) | 0.841 / 0.899 / 0.874 (2015 / 2020 / 2025) |
| Temporal accuracy (2015+2020 → 2025) | RF 0.821 · XGB 0.815 |
| Map agreement with label rasters | 84.8% (2015) · 79.6% (2020) · 90.6% (2025 window) |
| Built-up growth 2015→2025 (label rasters) | **+23.97 km²** (89.25 → 113.22) |
| Vegetation loss 2015→2025 | −22.79 km² |
| Land that BECAME Built-up 2015→2025 | 26.09 km² (codes 21/31/41) |

**Honest limitations (also in README §14):** label provenance not independently
verified · no spatially independent validation (no coordinates in CSVs) · 2025 map
covers the healthy window only · 2020/2025 CSV sampling design undocumented ·
areas use a cos(lat) approximation verified against UTM within 1%.
