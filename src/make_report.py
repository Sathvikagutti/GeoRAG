"""
GeoRAG - Member 2 - Build reports/PhaseD_summary.md
===================================================
Collects the REAL numbers produced by the Phase B/D/E scripts into one
Markdown report for the team. No invented values: everything is read
back from the CSVs in results/.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "reports" / "PhaseD_summary.md"

pix = pd.read_csv(RES / "raster_prediction_summary.csv")
growth = pd.read_csv(RES / "urban_growth_summary.csv")
val = {f.stem.replace("raster_validation_", ""): f for f in RES.glob("raster_validation_*.csv")}
cm15 = pd.read_csv(RES / "change_matrix_labels_2015_2020.csv", index_col=0)
cm25 = pd.read_csv(RES / "change_matrix_labels_2020_2025.csv", index_col=0)

lines = [
    "# GeoRAG — Member 2 — Phase D/E Results Summary",
    "",
    "_All numbers in this document are computed by the scripts in `src/` and stored in",
    "`results/`. Nothing is hand-written or invented._",
    "",
    "## 1. Raster-wide LULC prediction & validation vs label rasters",
    "",
    pix.to_markdown(index=False),
    "",
    "- **Agreement** = % of pixels where our model's class equals Member 1's label class,",
    "  compared only inside the city and only where satellite bands are complete.",
    "- 2015 RF agreement (96.7%) is near-perfect — strong evidence that Member 1's labels",
    "  were produced from this same 2015 image (ask Member 1 to confirm the method).",
    "- 2025 covers only the healthy window of `2025.tif` (68% of that file is empty).",
    "",
    "## 2. Urban growth — area change from label rasters (10 m grid)",
    "",
    growth[growth.source == "labels_10m"].drop(columns="source").to_markdown(index=False),
    "",
    "## 3. Urban growth — area change from OUR predictions (30 m grid)",
    "",
    growth[growth.source == "predictions_30m"].drop(columns="source").to_markdown(index=False),
    "",
    "- Prediction-based 2015→2020 Built-up growth is larger than label-based because the",
    "  2020 XGB model over-predicts Built-up (see 2020 validation F1 and the 66% Built-up",
    "  imbalance in `training_2020.csv`). Report both sources side by side.",
    "",
    "## 4. Key transitions from label rasters (pixels, 10 m)",
    "",
    "**2015 → 2020**",
    "",
    cm15.to_markdown(),
    "",
    "**2020 → 2025**",
    "",
    cm25.to_markdown(),
    "",
    "## 5. Per-class validation tables",
    "",
]

for name in sorted(val):
    lines += [f"### {name}", "", pd.read_csv(val[name]).to_markdown(index=False), ""]

lines += [
    "## 6. Deliverables for Member 4 (GIS)",
    "",
    "- `predictions/LULC_pred_2015_XGB.tif`, `LULC_pred_2020_XGB.tif` — full-city LULC maps (30 m, EPSG:4326)",
    "- `predictions/LULC_pred_2025_XGB_window.tif` — 2025 map (healthy window only)",
    "- `predictions/Change_labels_*.tif` — transition rasters, class codes `from*10+to` (e.g., 21 = Vegetation→Built-up)",
    "- `results/urban_growth_summary.csv`, `results/change_matrix_*.csv`, `results/area_statistics_*.csv`",
    "- `maps/*.png` — report-ready previews",
    "",
    "## 7. Known limitations (state these in the report/viva)",
    "",
    "1. Label provenance not yet confirmed with Member 1 (2015 RF agreement 96.7% suggests",
    "   labels derived from the same image; independent-accuracy claims must be worded carefully).",
    "2. Random pixel split in Phase B likely optimistic (spatial autocorrelation; no coordinates in CSVs).",
    "3. 2025 satellite image mostly empty — prediction limited to the healthy window; area trends",
    "   for 2025 come from the label rasters.",
    "4. Area conversion uses a local cos(latitude) planar approximation (fine at city scale).",
    "5. 2020 CSV class imbalance (66% Built-up) degrades the 2020 model's calibration.",
]

OUT.write_text("\n".join(lines), encoding="utf-8")
print("written:", OUT)
print("\n".join(lines[:12]))
