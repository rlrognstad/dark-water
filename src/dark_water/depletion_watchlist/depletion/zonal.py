"""Aggregate pixel-level TWS trend (see `trend.py`) to basin/aquifer polygons.

Assigns each mascon-grid pixel to the polygon containing its centroid — a
standard simplification for coarse-to-polygon zonal stats, and consistent
with the pilot's "native mascon resolution, no downscaling" stance: we are
not claiming sub-pixel precision by area-weighting fractional overlaps.
"""

import geopandas as gpd
import xarray as xr


def _summary_variables(trend_ds: xr.Dataset, dim: str = "time") -> list:
    # Select only the time-less summary variables. trend.fit_trend also
    # returns trend_line (dims time, lat, lon, for hydrograph overlays) --
    # to_dataframe() on the whole Dataset would broadcast trend/p_value
    # across trend_line's time dimension, multiplying every pixel's row by
    # the number of time steps.
    #
    # Detected rather than hardcoded so that the richer Datasets from
    # precipitation.adjusted_trend (total_trend, precip_explained_trend,
    # fraction_unexplained) reach the basin table without this module
    # needing to know their names.
    return [name for name, var in trend_ds.data_vars.items() if dim not in var.dims]


def _trend_to_points(trend_ds: xr.Dataset) -> gpd.GeoDataFrame:
    df = trend_ds[_summary_variables(trend_ds)].to_dataframe().reset_index().dropna(subset=["trend"])
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

    Any further time-less variable on `trend_ds` is carried through as
    `mean_<name>` -- so a Dataset from `precipitation.adjusted_trend`
    arrives with `mean_fraction_unexplained` alongside the rest.
    """
    points = _trend_to_points(trend_ds)
    joined = gpd.sjoin(points, basin_polygons[[id_column, "geometry"]], predicate="within", how="inner")

    extra = [c for c in _summary_variables(trend_ds) if c not in ("trend", "p_value", "significant_decline")]

    grouped = joined.groupby(id_column)
    stats = grouped.agg(
        mean_trend=("trend", "mean"),
        n_pixels=("trend", "size"),
        frac_pixels_significant_decline=("significant_decline", "mean"),
        **{f"mean_{name}": (name, "mean") for name in extra},
    )
    stats["basin_significant_decline"] = (stats["mean_trend"] < 0) & (
        stats["frac_pixels_significant_decline"] > majority_threshold
    )

    result = basin_polygons.merge(stats, on=id_column, how="left")
    result["n_pixels"] = result["n_pixels"].fillna(0).astype(int)
    result["basin_significant_decline"] = result["basin_significant_decline"].fillna(False)
    return result
