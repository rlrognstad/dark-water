import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from dark_water.depletion_watchlist.darkness import density


def _basins():
    return gpd.GeoDataFrame(
        {"basin_id": [1, 2, 3], "area_km2_hydrobasins": [100.0, 100.0, 100.0]},
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1), box(4, 0, 5, 1)],
        crs="EPSG:4326",
    )


def _stations(records):
    # records: list of (lon, lat, end_date)
    return gpd.GeoDataFrame(
        {"end_date": [pd.Timestamp(r[2]) if r[2] else pd.NaT for r in records]},
        geometry=[Point(r[0], r[1]) for r in records],
        crs="EPSG:4326",
    )


def test_station_density_by_basin_counts_and_uses_provided_area_column():
    stations = _stations([(0.5, 0.5, None), (0.5, 0.6, None), (2.5, 0.5, None)])
    basins = _basins()

    result = density.station_density_by_basin(stations, basins, "basin_id", area_column="area_km2_hydrobasins")

    row1 = result.loc[result["basin_id"] == 1].iloc[0]
    row2 = result.loc[result["basin_id"] == 2].iloc[0]
    row3 = result.loc[result["basin_id"] == 3].iloc[0]
    assert row1["n_stations"] == 2
    assert row2["n_stations"] == 1
    assert row3["n_stations"] == 0
    assert row1["station_density"] == pytest.approx(2 / 100)


def test_station_density_by_basin_computes_area_when_no_area_column_given():
    stations = _stations([(0.5, 0.5, None)])
    basins = _basins()

    result = density.station_density_by_basin(stations, basins, "basin_id")

    row1 = result.loc[result["basin_id"] == 1].iloc[0]
    assert row1["area_km2"] > 0
    assert row1["station_density"] == row1["n_stations"] / row1["area_km2"]


def test_station_density_by_basin_excludes_inactive_stations_by_default():
    as_of = pd.Timestamp("2026-01-01")
    stations = _stations(
        [
            (0.5, 0.5, None),  # still reporting
            (0.5, 0.6, "2025-06-01"),  # recently inactive, within grace period
            (0.5, 0.7, "2010-01-01"),  # long inactive
        ]
    )
    basins = _basins()

    result = density.station_density_by_basin(
        stations, basins, "basin_id", area_column="area_km2_hydrobasins", as_of=as_of
    )

    row1 = result.loc[result["basin_id"] == 1].iloc[0]
    assert row1["n_stations"] == 2  # excludes the long-inactive station


def test_station_density_by_basin_active_only_false_counts_everything():
    as_of = pd.Timestamp("2026-01-01")
    stations = _stations([(0.5, 0.5, "2010-01-01")])
    basins = _basins()

    result = density.station_density_by_basin(
        stations, basins, "basin_id", area_column="area_km2_hydrobasins", active_only=False, as_of=as_of
    )

    row1 = result.loc[result["basin_id"] == 1].iloc[0]
    assert row1["n_stations"] == 1


def test_observability_score_ranks_zero_density_basins_at_the_bottom_not_the_middle():
    # Heavy zero-inflation: most basins have zero stations. A naive
    # average-tie percentile rank would pull the zero-density majority up
    # toward the middle of the distribution; method="min" keeps them low.
    density_values = pd.Series([0.0, 0.0, 0.0, 0.0, 5.0])

    scores = density.observability_score(density_values)

    assert scores.iloc[:4].max() < 0.3
    assert scores.iloc[4] == 1.0


def test_darkness_score_is_complement_of_observability():
    density_values = pd.Series([0.0, 1.0, 2.0])
    scores = density.observability_score(density_values)
    darkness = density.darkness_score(density_values)

    assert np.allclose(darkness.to_numpy(), 1 - scores.to_numpy())
