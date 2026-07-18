"""Aggregate pixel-level TWS trend (see `trend.py`) to basin/aquifer polygons.

Assigns each mascon-grid pixel to the polygon containing its centroid — a
standard simplification for coarse-to-polygon zonal stats, and consistent
with the pilot's "native mascon resolution, no downscaling" stance: we are
not claiming sub-pixel precision by area-weighting fractional overlaps.
"""

import geopandas as gpd
import xarray as xr


def _trend_to_points(trend_ds: xr.Dataset) -> gpd.GeoDataFrame:
    df = trend_ds.to_dataframe().reset_index().dropna(subset=["trend"])
    lon_180 = ((df["lon"] + 180) % 360) - 180
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(lon_180, df["lat"]),
        crs="EPSG:4326",
    )


def aggregate_trend_to_basins(
    trend_ds: xr.Dataset, basin_polygons: gpd.GeoDataFrame, id_column: str, majority_threshold: float = 0.5
) -> gpd.GeoDataFrame:
    """Aggregate a pixel-level trend Dataset (from `trend.fit_trend`) to basin polygons.

    Adds `mean_trend`, `n_pixels`, `frac_pixels_significant_decline`, and
    `basin_significant_decline` (mean trend negative and more than
    `majority_threshold` of contributing pixels individually significant)
    to a copy of `basin_polygons`. Basins with no contributing pixels (e.g.
    smaller than the mascon grid) get NaN/0, not dropped.
    """
    points = _trend_to_points(trend_ds)
    joined = gpd.sjoin(points, basin_polygons[[id_column, "geometry"]], predicate="within", how="inner")

    grouped = joined.groupby(id_column)
    stats = grouped.agg(
        mean_trend=("trend", "mean"),
        n_pixels=("trend", "size"),
        frac_pixels_significant_decline=("significant_decline", "mean"),
    )
    stats["basin_significant_decline"] = (stats["mean_trend"] < 0) & (
        stats["frac_pixels_significant_decline"] > majority_threshold
    )

    result = basin_polygons.merge(stats, on=id_column, how="left")
    result["n_pixels"] = result["n_pixels"].fillna(0).astype(int)
    result["basin_significant_decline"] = result["basin_significant_decline"].fillna(False)
    return result
