import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box

from dark_water.depletion_watchlist.depletion import zonal


def _trend_dataset(lats, lons, trend_grid, significant_grid):
    return xr.Dataset(
        {
            "trend": (("lat", "lon"), trend_grid),
            "p_value": (("lat", "lon"), np.full_like(trend_grid, 0.01)),
            "significant_decline": (("lat", "lon"), significant_grid),
        },
        coords={"lat": lats, "lon": lons},
    )


def test_aggregate_trend_to_basins_assigns_by_centroid_and_handles_lon_wraparound():
    # lon in 0-360 (GRACE convention); basin polygon spans -1..1 in -180..180
    # convention, i.e. lon 359-360 and 0-1 in 0-360 convention.
    lats = [0.5]
    lons = [359.5, 0.5]
    trend_grid = np.array([[-3.0, -1.0]])
    significant_grid = np.array([[True, False]])
    trend_ds = _trend_dataset(lats, lons, trend_grid, significant_grid)

    basin = gpd.GeoDataFrame(
        {"basin_id": [1]}, geometry=[box(-1, -1, 1, 1)], crs="EPSG:4326"
    )

    result = zonal.aggregate_trend_to_basins(trend_ds, basin, id_column="basin_id")

    row = result.loc[result["basin_id"] == 1].iloc[0]
    assert row["n_pixels"] == 2
    assert row["mean_trend"] == -2.0
    assert row["frac_pixels_significant_decline"] == 0.5
    assert bool(row["basin_significant_decline"]) is False  # exactly at threshold, not > 0.5


def test_aggregate_trend_to_basins_flags_majority_significant_decline():
    lats = [0.5]
    lons = [0.5, 1.5, 2.5]
    trend_grid = np.array([[-3.0, -2.0, -1.0]])
    significant_grid = np.array([[True, True, False]])
    trend_ds = _trend_dataset(lats, lons, trend_grid, significant_grid)

    basin = gpd.GeoDataFrame(
        {"basin_id": [1]}, geometry=[box(0, 0, 3, 1)], crs="EPSG:4326"
    )

    result = zonal.aggregate_trend_to_basins(trend_ds, basin, id_column="basin_id")

    row = result.loc[result["basin_id"] == 1].iloc[0]
    assert bool(row["basin_significant_decline"]) is True


def test_aggregate_trend_to_basins_leaves_empty_basin_unflagged():
    lats = [40.5]
    lons = [40.5]
    trend_grid = np.array([[-5.0]])
    significant_grid = np.array([[True]])
    trend_ds = _trend_dataset(lats, lons, trend_grid, significant_grid)

    basins = gpd.GeoDataFrame(
        {"basin_id": [1, 2]},
        geometry=[box(40, 40, 41, 41), box(-10, -10, -9, -9)],
        crs="EPSG:4326",
    )

    result = zonal.aggregate_trend_to_basins(trend_ds, basins, id_column="basin_id")

    empty_row = result.loc[result["basin_id"] == 2].iloc[0]
    assert empty_row["n_pixels"] == 0
    assert bool(empty_row["basin_significant_decline"]) is False
