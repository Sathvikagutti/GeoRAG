"""
GeoRAG - Member 2 - Step 1: Inspect all raster files
====================================================
WHAT THIS DOES (beginner version):
A GeoTIFF is like a PNG photo, but every pixel is a number and the file
"knows" its real-world location (CRS = coordinate reference system) and
pixel size (resolution, in meters for projected maps).

We print, for every raster:
  - width x height  (number of pixels)
  - number of bands (a band = one "color layer", e.g. Red, NIR...)
  - CRS             (the "address system" of the map, e.g. EPSG:32644)
  - resolution      (real-world size of one pixel, here 30 m = Landsat)
  - nodata value    (the number that means "no information here")
  - min/max/mean of a few pixels (quick sanity check that data is real)

This is READ-ONLY: we open files, we never modify them.
"""

from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / "GeoRag"

FILES = [
    GEO / "GeoRAG_Satellite" / "Vijayawada_Landsat_2015.tif",
    GEO / "GeoRAG_Satellite" / "Vijayawada_Landsat_2020.tif",
    GEO / "GeoRAG_Satellite" / "2015.tif",
    GEO / "GeoRAG_Satellite" / "2020.tif",
    GEO / "GeoRAG_Satellite" / "2025.tif",
    GEO / "GeoRAG_Indices" / "Vijayawada_Indices_2015.tif",
    GEO / "GeoRAG_Indices" / "Vijayawada_Indices_2020.tif",
    GEO / "GeoRAG_LandUse_Labels" / "Vijayawada_LandUse_Labels_2015.tif",
    GEO / "GeoRAG_LandUse_Labels" / "Vijayawada_LandUse_Labels_2020.tif",
    GEO / "GeoRAG_LandUse_Labels" / "Vijayawada_LandUse_Labels_2025.tif",
]

for f in FILES:
    print("=" * 78)
    if not f.exists():
        print(f"MISSING: {f.name}")
        continue
    with rasterio.open(f) as src:
        print(f"FILE    : {f.relative_to(ROOT)}")
        print(f"  size  : {src.width} x {src.height} px   ({src.width * src.height:,} pixels)")
        print(f"  bands : {src.count}")
        print(f"  dtypes: {src.dtypes}")
        print(f"  CRS   : {src.crs}")
        print(f"  res   : {src.res[0]:.2f} x {src.res[1]:.2f} (map units per pixel)")
        print(f"  nodata: {src.nodata}")
        print(f"  bounds: {tuple(round(b, 1) for b in src.bounds)}")
        # read up to 3 bands, subsample every 7th pixel for speed
        n = min(3, src.count)
        for b in range(1, n + 1):
            arr = src.read(b)[::7, ::7]
            valid = arr[arr != (src.nodata if src.nodata is not None else -9999)]
            if valid.size:
                print(f"  band {b}: min={valid.min():.4f} max={valid.max():.4f} "
                      f"mean={valid.mean():.4f} ({valid.size} sampled px)")
print("=" * 78)
print("Done. READ-ONLY inspection - no files were modified.")
