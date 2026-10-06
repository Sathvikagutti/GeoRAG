"""
GeoRAG - Member 2 - Phase E: Urban growth / temporal change analysis
====================================================================
BEGINNER VERSION:
A "transition matrix" answers: between 2015 and 2020, how many pixels that
WERE vegetation became built-up? Each row = old class, each column = new
class. Big diagonal numbers = "stayed the same". Off-diagonal numbers tell
the change story (e.g., Vegetation -> Built-up = urban growth).

We run this TWICE, on two parallel universes:
  A) MEMBER 1's label rasters (10 m, full city)   = the reference truth
  B) OUR model predictions (30 m, XGBoost)        = our own ML output

Outputs:
  results/change_matrix_labels_<y1>_<y2>.csv
  results/change_matrix_predictions_2015_2020.csv
  results/area_statistics_labels_<year>.csv
  results/urban_growth_summary.csv        <- headline numbers
  predictions/Change_labels_<y1>_<y2>.tif <- transition codes for GIS
  maps/change_<y1>_<y2>.png               <- "what became built-up" maps
"""

import numpy as np
import pandas as pd
import rasterio
from matplotlib import patches as mpatches

from geo_utils import (CLASS_NAMES, CLASSES, LABELS, MAPS, NODATA, PRED_DIR,
                       RESULTS, area_table, save_map, transition_matrix)

PAIRS = [(2015, 2020), (2020, 2025), (2015, 2025)]
M_LAT = 111_320.0                      # meters per degree latitude
LON_FACTOR = float(np.cos(np.deg2rad(16.5)))   # shrink lon at Vijayawada's latitude


def matrix_to_df(T):
    return pd.DataFrame(
        T,
        index=[f"from_{CLASS_NAMES[c]}" for c in CLASSES],
        columns=[f"to_{CLASS_NAMES[c]}" for c in CLASSES],
    )


def transition_codes(a2d, b2d):
    """Encode transitions as from*10+to (21 = vegetation -> built-up)."""
    code = np.full(a2d.shape, NODATA, np.uint8)
    m = (a2d >= 1) & (a2d <= 4) & (b2d >= 1) & (b2d <= 4)
    code[m] = (a2d[m] * 10 + b2d[m]).astype(np.uint8)
    return code


def save_change_png(path, a2d, b2d, title):
    """Simple report map: what stayed, what became Built-up, other changes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = np.zeros(a2d.shape, np.uint8)           # 0 grey = no data
    both = (a2d >= 1) & (a2d <= 4) & (b2d >= 1) & (b2d <= 4)
    img[both & (a2d == b2d)] = 1                  # dark grey = unchanged
    img[both & (a2d != b2d) & (b2d != 1)] = 2     # orange = changed (not to built-up)
    img[both & (a2d != b2d) & (b2d == 1)] = 3     # red = NEW built-up
    colors = {0: "#eeeeee", 1: "#c8c8c8", 2: "#f2a33c", 3: "#c81e1e"}
    cmap = matplotlib.colors.ListedColormap([colors[i] for i in range(4)])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    handles = [mpatches.Patch(color=colors[1], label="unchanged"),
               mpatches.Patch(color=colors[2], label="changed (non-built)"),
               mpatches.Patch(color=colors[3], label="NEW built-up")]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


growth_rows = []

# ============================================================ A) LABELS (10 m)
print("#" * 78)
print("# A) LABEL RASTERS (10 m, full city) - area stats + transitions")
print("#" * 78)
label_rasters, ltr, lcrs = {}, None, None
for y in (2015, 2020, 2025):
    with rasterio.open(LABELS[y]) as src:
        label_rasters[y] = src.read(1)
        ltr, lcrs = src.transform, src.crs
    areas, total, px_km2 = area_table(label_rasters[y], ltr)
    pd.DataFrame(areas).T.to_csv(RESULTS / f"area_statistics_labels_{y}.csv",
                                 index_label="class")
px_km2_10m = abs(ltr.a * ltr.e) * M_LAT * LON_FACTOR * LON_FACTOR / 1e6
print(f"10 m-grid pixel area: {px_km2_10m * 1e6:.1f} m^2")

a_lab = {y: pd.read_csv(RESULTS / f"area_statistics_labels_{y}.csv", index_col=0)
         for y in (2015, 2020, 2025)}
for c in CLASSES:
    n = CLASS_NAMES[c]
    growth_rows.append(dict(
        source="labels_10m", class_name=n,
        area_2015_km2=a_lab[2015].loc[c, "area_km2"],
        area_2020_km2=a_lab[2020].loc[c, "area_km2"],
        area_2025_km2=a_lab[2025].loc[c, "area_km2"],
        change_15_20_km2=round(a_lab[2020].loc[c, "area_km2"] - a_lab[2015].loc[c, "area_km2"], 2),
        change_20_25_km2=round(a_lab[2025].loc[c, "area_km2"] - a_lab[2020].loc[c, "area_km2"], 2),
        change_15_25_km2=round(a_lab[2025].loc[c, "area_km2"] - a_lab[2015].loc[c, "area_km2"], 2)))

for y1, y2 in PAIRS:
    T = transition_matrix(label_rasters[y1], label_rasters[y2])
    matrix_to_df(T).to_csv(RESULTS / f"change_matrix_labels_{y1}_{y2}.csv")
    vb = int(((label_rasters[y1] == 2) & (label_rasters[y2] == 1)).sum())
    ob = int(((label_rasters[y1] == 4) & (label_rasters[y2] == 1)).sum())
    print(f"\n{y1} -> {y2}:  Vegetation->Built-up {vb:,} px | OpenLand->Built-up {ob:,} px")
    print(matrix_to_df(T).to_string())
    save_map(PRED_DIR / f"Change_labels_{y1}_{y2}.tif",
             transition_codes(label_rasters[y1], label_rasters[y2]), ltr, lcrs)
    save_change_png(MAPS / f"change_labels_{y1}_{y2}.png",
                    label_rasters[y1], label_rasters[y2],
                    f"Change {y1} -> {y2} (label rasters, 10 m)")

# ============================================================ B) PREDICTIONS (30 m)
print()
print("#" * 78)
print("# B) OUR XGB PREDICTIONS (30 m) - area stats + 2015->2020 transition")
print("#" * 78)
pred_rasters, ptr, pcrs = {}, None, None
for y in (2015, 2020):
    with rasterio.open(PRED_DIR / f"LULC_pred_{y}_XGB.tif") as src:
        pred_rasters[y] = src.read(1)
        ptr, pcrs = src.transform, src.crs
    areas, total, px_km2_30m = area_table(pred_rasters[y], ptr)
    pd.DataFrame(areas).T.to_csv(RESULTS / f"area_statistics_pred_{y}_XGB.csv",
                                 index_label="class")
px_km2_30m = abs(ptr.a * ptr.e) * M_LAT * LON_FACTOR * LON_FACTOR / 1e6
print(f"30 m-grid pixel area: {px_km2_30m * 1e6:.1f} m^2")

a_pred = {y: pd.read_csv(RESULTS / f"area_statistics_pred_{y}_XGB.csv", index_col=0)
          for y in (2015, 2020)}
for c in CLASSES:
    growth_rows.append(dict(
        source="predictions_30m", class_name=CLASS_NAMES[c],
        area_2015_km2=a_pred[2015].loc[c, "area_km2"],
        area_2020_km2=a_pred[2020].loc[c, "area_km2"],
        area_2025_km2=np.nan,      # 2025 prediction covers only the healthy window
        change_15_20_km2=round(a_pred[2020].loc[c, "area_km2"] - a_pred[2015].loc[c, "area_km2"], 2),
        change_20_25_km2=np.nan, change_15_25_km2=np.nan))

# 2015 -> 2020 transition on the common full grid
T = transition_matrix(pred_rasters[2015], pred_rasters[2020])
matrix_to_df(T).to_csv(RESULTS / "change_matrix_predictions_2015_2020.csv")
vb = int(((pred_rasters[2015] == 2) & (pred_rasters[2020] == 1)).sum())
print(f"\n2015 -> 2020 (predictions): Vegetation->Built-up {vb:,} px")
print(matrix_to_df(T).to_string())
save_map(PRED_DIR / "Change_predictions_2015_2020.tif",
         transition_codes(pred_rasters[2015], pred_rasters[2020]), ptr, pcrs)
save_change_png(MAPS / "change_predictions_2015_2020.png",
                pred_rasters[2015], pred_rasters[2020],
                "Change 2015 -> 2020 (XGBoost predictions, 30 m)")

# ============================================================ summary
summary = pd.DataFrame(growth_rows)
summary.to_csv(RESULTS / "urban_growth_summary.csv", index=False)
print()
print("=" * 78)
print("HEADLINE: urban_growth_summary.csv")
print(summary.to_string(index=False))
print("\nDone.")
