"""
GeoRAG - Member 2 - Phase D: LULC prediction + validation
=========================================================
WHAT THIS DOES (beginner version):
1. Loads our trained XGBoost and Random Forest models (from Phase B).
2. Feeds them EVERY valid pixel of the big 2015 and 2020 satellite images
   -> produces full LULC maps (our own predictions, not Member 1's).
3. For 2025, predicts only the small healthy window of 2025.tif.
4. "Grades" every prediction against Member 1's label rasters
   (agreement % + confusion matrix) and saves everything:
     predictions/*.tif   full-geo maps for Member 4 (GIS)
     maps/*.png          colored previews for the report
     results/*.csv       accuracy + class-share tables
"""

import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from geo_utils import (BAND_ORDER, CLASS_NAMES, CLASSES, LABELS, MAPS, MODELS,
                       NODATA, PRED_DIR, RESULTS, SAT, PLAIN, area_table,
                       load_feature_stack, predict_raster, sample_labels_at,
                       save_map, save_png)

rows = []

for year in (2015, 2020):
    print("=" * 78)
    print(f"YEAR {year} - {SAT[year].name}")
    X, bands_ok, shape, tr, crs = load_feature_stack(SAT[year])
    h, w = shape
    print(f"  grid {w}x{h}, pixels with complete bands: {int(bands_ok.sum()):,}")

    lab = sample_labels_at(LABELS[year], tr, shape)
    inside = (lab > 0) & (lab <= 4) & bands_ok          # compare only there

    for mname in ("XGB", "RF"):
        model = joblib.load(MODELS / f"{mname}_{year}.pkl")
        t0 = time.time()
        pred = predict_raster(model, X, bands_ok, shape)
        dt = time.time() - t0
        print(f"  {mname}: predicted {int((pred != NODATA).sum()):,} px in {dt:.1f}s")

        # ---- validation against label raster
        p, l = pred.ravel()[inside], lab[inside]
        agree = float((p == l).mean())
        cm = np.zeros((4, 4), np.int64)
        for la, pb in zip(l, p):
            cm[la - 1, pb - 1] += 1
        per_class = {}
        for i, c in enumerate(CLASSES):
            tp = cm[i, i]
            prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
            rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[c] = dict(precision=round(prec, 4), recall=round(rec, 4),
                                f1=round(f1, 4), support=int(cm[i, :].sum()))
        print(f"    agreement vs labels: {agree * 100:.2f}%  | " +
              "  ".join(f"{CLASS_NAMES[c]} F1={per_class[c]['f1']:.3f}" for c in CLASSES))

        # ---- class shares (valid pixels only)
        tot = int((pred != NODATA).sum())
        shares = Counter(pred[pred != NODATA].tolist())
        share_txt = {CLASS_NAMES[c]: f"{shares.get(c, 0) / tot * 100:.1f}%" for c in CLASSES}

        rows.append(dict(year=year, model=mname,
                         agreement_pct=round(agree * 100, 2),
                         n_compared=int(inside.sum()), n_predicted=tot,
                         builtup_share=share_txt["Built-up"],
                         veget_share=share_txt["Vegetation"],
                         water_share=share_txt["Water"],
                         open_share=share_txt["Open Land"],
                         seconds=round(dt, 1)))
        pd.DataFrame(per_class).T.to_csv(
            RESULTS / f"raster_validation_{year}_{mname}.csv",
            index_label="class")

        # ---- artifacts: GeoTIFF + PNG + area table for BOTH models
        save_map(PRED_DIR / f"LULC_pred_{year}_{mname}.tif", pred, tr, crs)
        save_png(MAPS / f"LULC_pred_{year}_{mname}.png", pred,
                 f"GeoRAG LULC prediction {year} ({mname}, Member 2)")
        areas, total, px_km2 = area_table(pred, tr)
        pd.DataFrame(areas).T.to_csv(RESULTS / f"area_statistics_pred_{year}_{mname}.csv",
                                     index_label="class")
        print(f"    shares: {share_txt} | pixel = {px_km2 * 1e6:.0f} m^2 "
              f"({px_km2:.6f} km^2)")

# ---------------------------------------------------------------- 2025 window
print("=" * 78)
print("2025 - healthy window of 2025.tif (documented gap; ask Member 1 for full image)")
X, bands_ok, shape, tr, crs = load_feature_stack(PLAIN["2025"])
model = joblib.load(MODELS / "XGB_2025.pkl")
pred = predict_raster(model, X, bands_ok & np.isfinite(X).all(axis=1), shape)
lab = sample_labels_at(LABELS[2025], tr, shape)
inside = (lab > 0) & (lab <= 4) & (pred.ravel() != NODATA)
p, l = pred.ravel()[inside], lab[inside]
agree = float((p == l).mean())
tot = int((pred != NODATA).sum())
shares = Counter(pred[pred != NODATA].tolist())
share_txt = {CLASS_NAMES[c]: f"{shares.get(c, 0) / tot * 100:.1f}%" for c in CLASSES}
print(f"  valid px: {tot:,}  agreement vs labels: {agree * 100:.2f}%  shares: {share_txt}")
rows.append(dict(year=2025, model="XGB_window", agreement_pct=round(agree * 100, 2),
                 n_compared=int(inside.sum()), n_predicted=tot,
                 builtup_share=share_txt["Built-up"], veget_share=share_txt["Vegetation"],
                 water_share=share_txt["Water"], open_share=share_txt["Open Land"],
                 seconds=0.0))
save_map(PRED_DIR / "LULC_pred_2025_XGB_window.tif", pred, tr, crs)
save_png(MAPS / "LULC_pred_2025_XGB_window.png", pred,
         "GeoRAG LULC prediction 2025 - healthy window only (XGBoost)")

summary = pd.DataFrame(rows)
summary.to_csv(RESULTS / "raster_prediction_summary.csv", index=False)
print("=" * 78)
print("Saved:")
print("  predictions/LULC_pred_2015_XGB.tif, LULC_pred_2020_XGB.tif, LULC_pred_2025_XGB_window.tif")
print("  maps/LULC_pred_*.png, results/raster_prediction_summary.csv")
print(summary.to_string(index=False))
