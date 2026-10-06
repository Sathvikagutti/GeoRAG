# GeoRAG — Member 2 — Phase D/E Results Summary

_All numbers in this document are computed by the scripts in `src/` and stored in
`results/`. Nothing is hand-written or invented._

## 1. Raster-wide LULC prediction & validation vs label rasters

|   year | model      |   agreement_pct |   n_compared |   n_predicted | builtup_share   | veget_share   | water_share   | open_share   |   seconds |
|-------:|:-----------|----------------:|-------------:|--------------:|:----------------|:--------------|:--------------|:-------------|----------:|
|   2015 | XGB        |           84.76 |       348549 |        361732 | 22.2%           | 70.7%         | 5.0%          | 2.2%         |       1   |
|   2015 | RF         |           96.66 |       348549 |        361732 | 27.7%           | 63.5%         | 6.2%          | 2.6%         |       5.5 |
|   2020 | XGB        |           79.56 |       344072 |        344888 | 41.0%           | 48.5%         | 7.0%          | 3.5%         |       1   |
|   2020 | RF         |           78.91 |       344072 |        344888 | 43.3%           | 46.5%         | 6.9%          | 3.4%         |       1.6 |
|   2025 | XGB_window |           90.62 |        32085 |         32085 | 51.5%           | 27.5%         | 8.8%          | 12.2%        |       0   |

- **Agreement** = % of pixels where our model's class equals Member 1's label class,
  compared only inside the city and only where satellite bands are complete.
- 2015 RF agreement (96.7%) is near-perfect — strong evidence that Member 1's labels
  were produced from this same 2015 image (ask Member 1 to confirm the method).
- 2025 covers only the healthy window of `2025.tif` (68% of that file is empty).

## 2. Urban growth — area change from label rasters (10 m grid)

| class_name   |   area_2015_km2 |   area_2020_km2 |   area_2025_km2 |   change_15_20_km2 |   change_20_25_km2 |   change_15_25_km2 |
|:-------------|----------------:|----------------:|----------------:|-------------------:|-------------------:|-------------------:|
| Built-up     |           89.25 |          103.39 |          113.22 |              14.14 |               9.83 |              23.97 |
| Vegetation   |          183.36 |          170.26 |          160.57 |             -13.1  |              -9.69 |             -22.79 |
| Water        |           19.85 |           26.16 |           24.95 |               6.31 |              -1.21 |               5.1  |
| Open Land    |            8.34 |           11.67 |           12.72 |               3.33 |               1.05 |               4.38 |

## 3. Urban growth — area change from OUR predictions (30 m grid)

| class_name   |   area_2015_km2 |   area_2020_km2 |   area_2025_km2 |   change_15_20_km2 |   change_20_25_km2 |   change_15_25_km2 |
|:-------------|----------------:|----------------:|----------------:|-------------------:|-------------------:|-------------------:|
| Built-up     |           69.14 |          122.16 |             nan |              53.02 |                nan |                nan |
| Vegetation   |          220.77 |          144.38 |             nan |             -76.39 |                nan |                nan |
| Water        |           15.5  |           20.77 |             nan |               5.27 |                nan |                nan |
| Open Land    |            6.74 |           10.31 |             nan |               3.57 |                nan |                nan |

- Prediction-based 2015→2020 Built-up growth is larger than label-based because the
  2020 XGB model over-predicts Built-up (see 2020 validation F1 and the 66% Built-up
  imbalance in `training_2020.csv`). Report both sources side by side.

## 4. Key transitions from label rasters (pixels, 10 m)

**2015 → 2020**

|                 |   to_Built-up |   to_Vegetation |   to_Water |   to_Open Land |
|:----------------|--------------:|----------------:|-----------:|---------------:|
| from_Built-up   |        886076 |           32233 |      10863 |           1673 |
| from_Vegetation |        166944 |         1613574 |      74480 |          57372 |
| from_Water      |          4266 |           30248 |     170429 |           2083 |
| from_Open Land  |           818 |           10545 |      14983 |          60585 |

**2020 → 2025**

|                 |   to_Built-up |   to_Vegetation |   to_Water |   to_Open Land |
|:----------------|--------------:|----------------:|-----------:|---------------:|
| from_Built-up   |       1029011 |           40686 |       3599 |           4973 |
| from_Vegetation |        144302 |         1604908 |      13573 |          12875 |
| from_Water      |          6496 |           24859 |     218351 |          23090 |
| from_Open Land  |           981 |            4245 |      24729 |          91758 |

## 5. Per-class validation tables

### 2015_RF

|   class |   precision |   recall |     f1 |   support |
|--------:|------------:|---------:|-------:|----------:|
|       1 |      0.9771 |   0.9329 | 0.9545 |    103421 |
|       2 |      0.9609 |   0.9878 | 0.9742 |    212502 |
|       3 |      0.9721 |   0.9341 | 0.9527 |     22960 |
|       4 |      0.975  |   0.9374 | 0.9559 |      9666 |

### 2015_XGB

|   class |   precision |   recall |     f1 |   support |
|--------:|------------:|---------:|-------:|----------:|
|       1 |      0.8897 |   0.6759 | 0.7682 |    103421 |
|       2 |      0.8303 |   0.955  | 0.8883 |    212502 |
|       3 |      0.8826 |   0.682  | 0.7694 |     22960 |
|       4 |      0.8877 |   0.7175 | 0.7936 |      9666 |

### 2020_RF

|   class |   precision |   recall |     f1 |   support |
|--------:|------------:|---------:|-------:|----------:|
|       1 |      0.6217 |   0.8851 | 0.7304 |    104615 |
|       2 |      0.9117 |   0.74   | 0.8169 |    196966 |
|       3 |      0.9406 |   0.7654 | 0.844  |     28964 |
|       4 |      0.939  |   0.8119 | 0.8709 |     13527 |

### 2020_XGB

|   class |   precision |   recall |     f1 |   support |
|--------:|------------:|---------:|-------:|----------:|
|       1 |      0.6383 |   0.862  | 0.7335 |    104615 |
|       2 |      0.9004 |   0.7627 | 0.8258 |    196966 |
|       3 |      0.9335 |   0.7739 | 0.8462 |     28964 |
|       4 |      0.9152 |   0.808  | 0.8583 |     13527 |

## 6. Deliverables for Member 4 (GIS)

- `predictions/LULC_pred_2015_XGB.tif`, `LULC_pred_2020_XGB.tif` — full-city LULC maps (30 m, EPSG:4326)
- `predictions/LULC_pred_2025_XGB_window.tif` — 2025 map (healthy window only)
- `predictions/Change_labels_*.tif` — transition rasters, class codes `from*10+to` (e.g., 21 = Vegetation→Built-up)
- `results/urban_growth_summary.csv`, `results/change_matrix_*.csv`, `results/area_statistics_*.csv`
- `maps/*.png` — report-ready previews

## 7. Known limitations (state these in the report/viva)

1. Label provenance not yet confirmed with Member 1 (2015 RF agreement 96.7% suggests
   labels derived from the same image; independent-accuracy claims must be worded carefully).
2. Random pixel split in Phase B likely optimistic (spatial autocorrelation; no coordinates in CSVs).
3. 2025 satellite image mostly empty — prediction limited to the healthy window; area trends
   for 2025 come from the label rasters.
4. Area conversion uses a local cos(latitude) planar approximation (fine at city scale).
5. 2020 CSV class imbalance (66% Built-up) degrades the 2020 model's calibration.