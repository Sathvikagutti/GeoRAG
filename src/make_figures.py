"""
GeoRAG - Member 2 - Presentation figures
========================================
Builds report/slide figures from REAL rasters and CSVs into maps/figures/:
  fig1_lulc_labels.png       3-panel label maps 2015/2020/2025
  fig2_labels_vs_preds.png   label vs our XGB map (2015, 2020) with agreements
  fig3_area_change.png       class areas over the years (from label rasters)
  fig4_temporal_validation.png  same-year vs cross-year accuracy
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib import colors as mcolors, patches as mpatches

from geo_utils import CLASS_NAMES, CLASSES, LABELS, MAPS, RESULTS

FIG = MAPS / "figures"
FIG.mkdir(exist_ok=True)

COLORS = {1: "#c81e1e", 2: "#28a03c", 3: "#2859c8", 4: "#e6c83c"}
CMAP = mcolors.ListedColormap([COLORS[c] for c in CLASSES])


def panel(ax, arr, title):
    img = np.zeros(arr.shape, np.uint8)
    for c in CLASSES:
        img[arr == c] = CLASSES.index(c) + 1
    masked = np.ma.masked_where(arr == 0, img)          # hide outside-city
    ax.imshow(masked, cmap=CMAP, vmin=1, vmax=4, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.axis("off")


legend_handles = [mpatches.Patch(color=COLORS[c], label=f"{c} {CLASS_NAMES[c]}")
                  for c in CLASSES]

# ---------------------------------------------------------------- Figure 1
label_arrs = {}
for y in (2015, 2020, 2025):
    with rasterio.open(LABELS[y]) as src:
        label_arrs[y] = src.read(1)
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, y in zip(axes, (2015, 2020, 2025)):
    panel(ax, label_arrs[y], f"LULC {y} (reference labels, 10 m)")
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Vijayawada land use / land cover 2015 - 2020 - 2025", fontsize=14)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(FIG / "fig1_lulc_labels.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig1 saved")

# ---------------------------------------------------------------- Figure 2
# Agreements are COMPUTED live from the rasters (never hardcoded), using the
# same georeferenced label sampling as src/predict_lulc.py.
from geo_utils import SAT, sample_labels_at

agreements = {}
preds = {}
for y in (2015, 2020):
    with rasterio.open(RESULTS.parent / "predictions" / f"LULC_pred_{y}_XGB.tif") as src:
        preds[y] = src.read(1)
        tr = src.transform
    lab = sample_labels_at(LABELS[y], tr, preds[y].shape)
    p = preds[y].ravel()
    inside = (lab > 0) & (lab <= 4) & (p >= 1) & (p <= 4)   # exclude nodata(255)
    agreements[str(y)] = round(float((p[inside] == lab[inside]).mean() * 100), 1)
    print(f"computed agreement {y}: {agreements[str(y)]}% over {int(inside.sum()):,} px")
fig, axes = plt.subplots(2, 2, figsize=(11, 11))
for r, y in enumerate((2015, 2020)):
    panel(axes[r, 0], label_arrs[y], f"Reference labels {y}")
    panel(axes[r, 1], preds[y],
          f"Our XGBoost prediction {y} (agreement {agreements[str(y)]}%)\n"
          "agreement vs label raster - NOT independent validation")
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=10,
           frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Member 1 labels vs Member 2 model predictions", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(FIG / "fig2_labels_vs_preds.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig2 saved")

# ---------------------------------------------------------------- Figure 3
g = pd.read_csv(RESULTS / "urban_growth_summary.csv")
gl = g[g.source == "labels_10m"].set_index("class_name")
years = [2015, 2020, 2025]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for c in CLASSES:
    n = CLASS_NAMES[c]
    axes[0].plot(years, [gl.loc[n, f"area_{y}_km2"] for y in years],
                 marker="o", color=COLORS[c], linewidth=2, label=n)
axes[0].set_title("Class area over time (label rasters)")
axes[0].set_ylabel("area (km²)")
axes[0].set_xticks(years)
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=9)

x = np.arange(3)
w = 0.2
for i, c in enumerate(CLASSES):
    n = CLASS_NAMES[c]
    axes[1].bar(x + (i - 1.5) * w, [gl.loc[n, f"area_{y}_km2"] for y in years],
                width=w, color=COLORS[c], label=n)
axes[1].set_title("Area by class and year")
axes[1].set_ylabel("area (km²)")
axes[1].set_xticks(x, [str(y) for y in years])
axes[1].grid(alpha=0.3, axis="y")
fig.suptitle("Urban growth: Built-up rises as Vegetation falls", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG / "fig3_area_change.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig3 saved")

# ---------------------------------------------------------------- Figure 4
tv = pd.read_csv(RESULTS / "temporal_validation_results.csv")
b = pd.read_csv(RESULTS / "accuracy_results.csv")
bars = [("2015\n(same-year split)", b.query("year == 2015 and model == 'XGB'").accuracy.iloc[0]),
        ("2020\n(same-year split)", b.query("year == 2020 and model == 'XGB'").accuracy.iloc[0]),
        ("2025\n(same-year split)", b.query("year == 2025 and model == 'XGB'").accuracy.iloc[0]),
        ("train 2015 ->\ntest 2020", tv.query("train_years == '2015' and test_year == 2020 and model == 'XGB'").accuracy.iloc[0]),
        ("train 2015 ->\ntest 2025", tv.query("train_years == '2015' and test_year == 2025 and model == 'XGB'").accuracy.iloc[0]),
        ("train 2015+2020 ->\ntest 2025", tv.query("train_years == '2015+2020' and test_year == 2025 and model == 'XGB'").accuracy.iloc[0])]
labels_txt = [t for t, _ in bars]
vals = [v * 100 for _, v in bars]
cols = ["#4477aa"] * 3 + ["#ee6677"] * 2 + ["#228833"]
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(labels_txt, vals, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_ylim(60, 95)
ax.set_ylabel("accuracy (%)")
ax.set_title("XGBoost accuracy: stratified random-pixel split vs unseen future years\n"
             "(all splits are random pixel-level; no spatially independent validation available)")
ax.grid(alpha=0.3, axis="y")
plt.xticks(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig4_temporal_validation.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig4 saved")
print("All figures in", FIG)
