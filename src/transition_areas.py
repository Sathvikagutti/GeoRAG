"""
GeoRAG - Member 2 - Transition AREAS (km2)
==========================================
The change matrices count pixels; this script converts every transition to
area using the verified pixel areas (Step 10 cross-checked both methods,
agreeing within 1%). Reads only existing matrices + rasters. READ-ONLY.
Output: results/transition_areas.csv
"""
import numpy as np
import pandas as pd
import rasterio

from geo_utils import CLASS_NAMES, CLASSES, LABELS, PRED_DIR, RESULTS

M_LAT = 111_320.0
LON_F = float(np.cos(np.deg2rad(16.5)))


def px_km2(transform):
    return abs(transform.a * transform.e) * M_LAT * (M_LAT * LON_F) / 1e6


with rasterio.open(LABELS[2015]) as src:
    a10 = px_km2(src.transform)          # ~0.0000285 km2 (10 m grid)

rows = []
for y1, y2 in [(2015, 2020), (2020, 2025), (2015, 2025)]:
    T = pd.read_csv(RESULTS / f"change_matrix_labels_{y1}_{y2}.csv", index_col=0)
    for i, fa in enumerate(CLASSES):
        for j, tb in enumerate(CLASSES):
            n = int(T.iloc[i, j])
            rows.append(dict(period=f"{y1}->{y2}", from_class=CLASS_NAMES[fa],
                             to_class=CLASS_NAMES[tb], pixels=n,
                             area_km2=round(n * a10, 2),
                             kind="unchanged" if fa == tb else
                                  ("to_Built-up" if tb == 1 else "other_change")))
ta = pd.DataFrame(rows)
ta.to_csv(RESULTS / "transition_areas.csv", index=False)

growth = ta[(ta["kind"] == "to_Built-up")].groupby("period")["area_km2"].sum()
print("Built-up expansion by source class (km2, label rasters 10 m):")
print(growth.round(2).to_string())
print(f"\nSaved {RESULTS / 'transition_areas.csv'} ({len(ta)} transitions)")
