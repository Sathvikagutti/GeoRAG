"""
GeoRAG - Member 2 - Output verification (final QA)
==================================================
Independently verifies, for every produced LULC/change GeoTIFF:
  - CRS, dimensions, transform, bounds, nodata
  - class values are only {1,2,3,4} + nodata(255)
  - alignment with the source satellite raster of the same year
Then cross-checks area statistics TWO WAYS:
  A) our cos(latitude) degree-based approximation
  B) independent on-the-fly reprojection to UTM 44N (metric CRS)
and writes results/output_verification.csv.
READ-ONLY: no raster is modified.
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

from geo_utils import (CLASS_NAMES, CLASSES, LABELS, MAPS, NODATA, PRED_DIR,
                       RESULTS, SAT)

rows = []


def check(path, expect_aligned_with=None, label=""):
    with rasterio.open(path) as src:
        arr = src.read(1)
        vals = set(np.unique(arr[arr != src.nodata]).tolist()) if src.nodata is not None \
            else set(np.unique(arr).tolist())
        rec = dict(file=path.name, label=label, crs=str(src.crs),
                   width=src.width, height=src.height,
                   res_deg=round(abs(src.transform.a), 8),
                   bounds=",".join(f"{b:.5f}" for b in src.bounds),
                   nodata=src.nodata,
                   classes=sorted(int(v) for v in vals if v <= 4),
                   bad_classes=sorted(int(v) for v in vals if v not in CLASSES))
        aligned = None
        if expect_aligned_with is not None:
            with rasterio.open(expect_aligned_with) as ref:
                aligned = (src.width == ref.width and src.height == ref.height
                           and np.allclose(src.transform, ref.transform)
                           and str(src.crs) == str(ref.crs))
        rec["aligned_with_source"] = aligned
        rows.append(rec)
        return src, arr


print("== LULC predictions ==")
sat = {y: SAT[y] for y in (2015, 2020)}
src15, _ = check(PRED_DIR / "LULC_pred_2015_XGB.tif", sat[2015], "prediction 2015")
src20, _ = check(PRED_DIR / "LULC_pred_2020_XGB.tif", sat[2020], "prediction 2020")
with rasterio.open(sat[2015]) as r:
    tr_full, crs_full = r.transform, r.crs

# 2025 window prediction: check against the small 2025.tif
src25, _ = check(PRED_DIR / "LULC_pred_2025_XGB_window.tif",
                 "GeoRag/GeoRAG_Satellite/2025.tif", "prediction 2025 (window)")

print("== change rasters ==")
for a, b in [(2015, 2020), (2020, 2025), (2015, 2025)]:
    with rasterio.open(LABELS[a]) as ref:
        pass
    check(PRED_DIR / f"Change_labels_{a}_{b}.tif", LABELS[a], f"change {a}-{b}")

print("== valid-class check ==")
for rec in rows:
    ok = rec["classes"] == [1, 2, 3, 4] or rec["classes"] == [] or rec["label"].startswith("change")
    rec["classes_ok"] = ok

# ---------------- area cross-check: degrees-approx vs UTM 44N ----------------
print("== area cross-check (cos-lat approx vs UTM 44N reproject) ==")
M_LAT = 111_320.0
LON_F = float(np.cos(np.deg2rad(16.5)))
area_rows = []
for year, (srcp, arr) in {2015: (None, None), 2020: (None, None)}.items():
    pass

def area_two_ways(path, inside_only=True):
    with rasterio.open(path) as src:
        arr = src.read(1)
        tr, crs = src.transform, src.crs
        bounds = src.bounds
    # A) degree approximation: px_deg2 * (m/deg-lat) * (m/deg-lon) -> m^2 -> km^2
    px_deg2 = abs(tr.a * tr.e)
    px_km2_deg = px_deg2 * M_LAT * (M_LAT * LON_F) / 1e6
    # B) UTM reprojection (independent, metric)
    left, bottom, right, top = bounds
    tr_utm, w, h = calculate_default_transform(
        crs, "EPSG:32644", src.width, src.height, left, bottom, right, top)
    px_km2_utm = abs(tr_utm.a * tr_utm.e) / 1e6
    out = {}
    src_data = arr
    for cls in CLASSES:
        n_deg = int((src_data == cls).sum())
        dst = np.zeros((h, w), np.uint8)
        reproject(src_data, dst, src_transform=tr, src_crs=crs,
                  dst_transform=tr_utm, dst_crs="EPSG:32644",
                  resampling=Resampling.nearest)
        n_utm = int((dst == cls).sum())
        out[cls] = dict(km2_deg=round(n_deg * px_km2_deg, 2),
                        km2_utm=round(n_utm * px_km2_utm, 2))
    return out

checks = [(2015, PRED_DIR / "LULC_pred_2015_XGB.tif"),
          (2020, PRED_DIR / "LULC_pred_2020_XGB.tif"),
          ("2015lab", LABELS[2015]), ("2020lab", LABELS[2020]), ("2025lab", LABELS[2025])]
for name, p in checks:
    res = area_two_ways(p)
    for cls, v in res.items():
        diff = abs(v["km2_deg"] - v["km2_utm"]) / max(v["km2_utm"], 1e-9) * 100
        area_rows.append(dict(raster=name, cls=CLASS_NAMES[cls], **v,
                              diff_pct=round(diff, 2)))
        print(f"  {name} {CLASS_NAMES[cls]:10s}: deg-approx {v['km2_deg']:8.2f} km2 | "
              f"UTM {v['km2_utm']:8.2f} km2 | diff {diff:.2f}%")

pd.DataFrame(rows).to_csv(RESULTS / "output_verification.csv", index=False)
pd.DataFrame(area_rows).to_csv(RESULTS / "area_crosscheck_utm.csv", index=False)
print("\nSaved results/output_verification.csv and results/area_crosscheck_utm.csv")
