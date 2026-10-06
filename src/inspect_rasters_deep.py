"""
GeoRAG - Member 2 - Step 2: Deep raster inspection
==================================================
QUESTIONS WE ANSWER HERE (beginner version):
1. How much of each satellite image is REAL data vs empty (NaN) pixels?
   (NaN pixels are usually "outside the city boundary" or cloudy areas.)
2. What are the bands called? (7 bands in the raster vs 9 features in our CSV -
   we need to know the mapping, e.g. band 1 = Blue? band 7 = ???)
3. Do the label maps agree with the training CSVs? (provenance check:
   if the label raster class shares match the CSV class shares, the CSVs
   were almost certainly sampled FROM these label rasters.)
4. Grid alignment: can label pixels be matched 1:1 to satellite pixels?

READ-ONLY: nothing is modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / "GeoRag"

SAT = {
    2015: GEO / "GeoRAG_Satellite" / "Vijayawada_Landsat_2015.tif",
    2020: GEO / "GeoRAG_Satellite" / "Vijayawada_Landsat_2020.tif",
}
PLAIN = {
    "2015": GEO / "GeoRAG_Satellite" / "2015.tif",
    "2025": GEO / "GeoRAG_Satellite" / "2025.tif",
}
LABELS = {
    y: GEO / "GeoRAG_LandUse_Labels" / f"Vijayawada_LandUse_Labels_{y}.tif"
    for y in (2015, 2020, 2025)
}
CSV = {y: ROOT / "data" / f"training_{y}.csv" for y in (2015, 2020, 2025)}
CLASS_NAMES = {1: "Built-up", 2: "Vegetation", 3: "Water", 4: "Open Land"}


def valid_stats(path):
    """Read every band; report % of finite pixels per band."""
    with rasterio.open(path) as src:
        desc = src.descriptions
        arr = src.read()
        finite = np.isfinite(arr).all(axis=0)          # pixel valid only if ALL bands finite
        n_all = arr.shape[1] * arr.shape[2]
        n_ok = int(finite.sum())
    return desc, n_all, n_ok, arr.shape[0]


print("#" * 78)
print("# 1) SATELLITE IMAGES - how many pixels are real?")
print("#" * 78)
for y, f in {**SAT, **PLAIN}.items():
    desc, n_all, n_ok, nb = valid_stats(f)
    print(f"{f.name:35s} {nb} bands | valid pixels: {n_ok:>7,} / {n_all:,} "
          f"({n_ok / n_all * 100:5.1f}%) | band names: {desc}")

print()
print("#" * 78)
print("# 2) LABEL MAPS - class shares (full image, no sampling)")
print("#" * 78)
label_counts = {}
for y, f in LABELS.items():
    with rasterio.open(f) as src:
        a = src.read(1)
    vals, cnts = np.unique(a, return_counts=True)
    label_counts[y] = dict(zip(vals.tolist(), cnts.tolist()))
    total = cnts.sum()
    inside = int(cnts[vals > 0].sum())
    print(f"{y}: inside-city pixels = {inside:,} | class shares:")
    for v, c in zip(vals, cnts):
        if v == 0:
            print(f"    0 (outside) : {c:>9,}  ({c / total * 100:5.2f}%)")
        else:
            print(f"    {v} {CLASS_NAMES[v]:<10s}: {c:>9,}  ({c / total * 100:5.2f}%)")

print()
print("#" * 78)
print("# 3) CSV CLASS SHARES - do they match the label rasters?")
print("#" * 78)
for y in (2015, 2020, 2025):
    df = pd.read_csv(CSV[y])
    shares = df["Label"].value_counts(normalize=True).sort_index() * 100
    print(f"{y} CSV :", "  ".join(f"{k}={v:.2f}%" for k, v in shares.items()))

print()
print("#" * 78)
print("# 4) GRID ALIGNMENT - label raster vs satellite raster")
print("#" * 78)
with rasterio.open(LABELS[2015]) as lab, rasterio.open(SAT[2015]) as sat:
    print("label  :", lab.width, "x", lab.height, "res", lab.res)
    print("sat    :", sat.width, "x", sat.height, "res", sat.res)
    print("same size?  ", (lab.width, lab.height) == (sat.width, sat.height))
    print("same bounds?", np.allclose(lab.bounds, sat.bounds))
    ratio = lab.width / sat.width
    print(f"size ratio label/sat = {ratio:.3f}  (3.0 means: 3x3 label px per sat px)")

print()
print("Done. READ-ONLY - nothing modified.")
