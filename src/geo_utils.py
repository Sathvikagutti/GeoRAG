"""
GeoRAG - Member 2 - Shared geospatial helpers
=============================================
Small utility functions used by all Phase D scripts. Keeping them in one
place means every script treats the data exactly the same way.

Beginner notes:
- "transform"  : an affine object inside a GeoTIFF that converts
                 (row, col) pixel positions -> (longitude, latitude).
- "sampling"   : reading a value from the label raster at the same
                 geographic location as a satellite pixel. The label raster
                 is finer (10 m) than the satellite (30 m), so we look up
                 which label pixel sits under each satellite pixel center.
- "nodata"     : we encode "no prediction / outside image" as 255 (uint8).
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio import Affine

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
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MAPS = ROOT / "maps"
PRED_DIR = ROOT / "predictions"
for d in (RESULTS, MAPS, PRED_DIR):
    d.mkdir(exist_ok=True)

CLASS_NAMES = {1: "Built-up", 2: "Vegetation", 3: "Water", 4: "Open Land"}
CLASSES = [1, 2, 3, 4]
NODATA = 255

# Classic Landsat band names inside the rasters (verified in Step 1/2).
BAND_ORDER = ["blue", "green", "red", "nir", "swir1", "swir2", "thermal"]
FEATURES = BAND_ORDER[:6] + ["ndvi", "ndbi", "ndwi"]  # model input layout


def safe_div(a, b):
    """(a-b)/(a+b) that returns NaN instead of warning when a+b == 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def load_feature_stack(path):
    """
    Read a 7-band satellite raster and build the 9-feature stack
    (6 bands + NDVI + NDBI + NDWI) exactly as in the training CSVs.
    Returns (X [H*W, 9] float32, bands_finite [H*W] bool, shape (H, W)).
    """
    with rasterio.open(path) as src:
        arr = src.read()
        transform = src.transform
        crs = src.crs
    h, w = arr.shape[1], arr.shape[2]
    blue, green, red, nir, s1, s2 = (arr[i].astype(np.float32) for i in range(6))
    stack = np.stack([blue, green, red, nir, s1, s2,
                      safe_div(nir, red), safe_div(s1, nir), safe_div(green, nir)], -1)
    bands_finite = np.isfinite(stack[..., :6]).all(axis=-1)
    return stack.reshape(-1, 9), bands_finite.reshape(-1), (h, w), transform, crs


def sample_labels_at(lab_path, transform, shape):
    """
    For every satellite pixel center, fetch the label-raster value underneath.
    Returns flat uint8 array (255 = outside image / class 0).
    """
    h, w = shape
    with rasterio.open(lab_path) as lsrc:
        lab = lsrc.read(1)
        ltr = lsrc.transform
    rows, cols = np.mgrid[0:h, 0:w]
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    lcols, lrows = (~ltr) * (xs, ys)
    lcols = np.floor(lcols).astype(np.int64)
    lrows = np.floor(lrows).astype(np.int64)
    ok = (lrows >= 0) & (lrows < lab.shape[0]) & (lcols >= 0) & (lcols < lab.shape[1])
    out = np.full(h * w, NODATA, np.uint8)
    out[ok.reshape(-1)] = lab.reshape(-1)[(lrows * lab.shape[1] + lcols)[ok]]
    return out


def predict_raster(model, X, valid_mask, shape):
    """Run the model on all valid pixels; returns 2-D uint8 (255 = unclassified)."""
    pred = np.full(shape[0] * shape[1], NODATA, np.uint8)
    pred[valid_mask] = model.predict(X[valid_mask].astype(np.float32)).astype(np.uint8) + 1
    return pred.reshape(shape)


def save_map(path, arr2d, transform, crs):
    """Write a single-band uint8 GeoTIFF (GeoTIFF = a PNG that knows where it is on Earth)."""
    profile = dict(driver="GTiff", height=arr2d.shape[0], width=arr2d.shape[1],
                   count=1, dtype="uint8", crs=crs, transform=transform,
                   nodata=NODATA, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr2d, 1)


CLASS_COLORS = {1: (200, 30, 30), 2: (40, 160, 60), 3: (40, 90, 200), 4: (230, 200, 60)}


def save_png(path, arr2d, title=""):
    """Save a colorized PNG preview (red=Built-up, green=Vegetation, blue=Water, yellow=Open)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors, patches as mpatches
    cmap = mcolors.ListedColormap(
        [(1, 1, 1)] + [tuple(c / 255 for c in CLASS_COLORS[c]) for c in CLASSES]
    )
    img = np.full(arr2d.shape, 0, np.uint8)
    for c in CLASSES:
        img[arr2d == c] = c
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    handles = [mpatches.Patch(color=tuple(v / 255 for v in CLASS_COLORS[c]),
                              label=f"{c} {CLASS_NAMES[c]}") for c in CLASSES]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def area_table(pred2d, transform, inside_mask=None):
    """
    Pixel counts -> area per class.
    Deg^2 pixel area is projected to km^2 with a local cos(latitude) factor
    (accurate at city scale; documented in the report).
    """
    px_area_deg2 = abs(transform.a * transform.e)
    mid_lat = None  # filled by caller if needed; default 16.5 (Vijayawada)
    lat = 16.5 if mid_lat is None else mid_lat
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat))
    px_km2 = px_area_deg2 * m_per_deg_lon * m_per_deg_lat / 1e6
    counts = {}
    for c in CLASSES:
        m = (pred2d == c)
        if inside_mask is not None:
            m = m & inside_mask
        counts[c] = int(m.sum())
    total = sum(counts.values())
    return {c: dict(pixels=counts[c], area_km2=round(counts[c] * px_km2, 2))
            for c in CLASSES}, total, px_km2


def transition_matrix(a2d, b2d, inside_mask=None):
    """Count transitions a->b between two classified rasters (same grid)."""
    T = np.zeros((4, 4), np.int64)
    m = (a2d <= 4) & (b2d <= 4)
    if inside_mask is not None:
        m &= inside_mask
    for la in CLASSES:
        for lb in CLASSES:
            T[la - 1, lb - 1] = int(((a2d == la) & (b2d == lb) & m).sum())
    return T
