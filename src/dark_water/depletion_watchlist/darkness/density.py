"""Darkness axis (v1 substitute): a GGMN station-density proxy for O_GW.

Per the concept doc: "If the observability work hasn't yet produced O_GW
for the relevant basins, DDW v1 substitutes a lightweight standalone
GGMN-density proxy." This is that proxy -- a static *level* (how well a
basin can currently be seen), not the net-decay *velocity* that
`dark_basins` computes separately.

The observability score is a percentile rank of station density across
basins, not an absolute scale -- station density has no natural zero-to-one
range, and most basins globally have zero GGMN stations, so an absolute
threshold would be arbitrary. Ranking is scale-free but means the score is
*relative to this basin set*, not an absolute observability standard.
"""

import geopandas as gpd
import pandas as pd

_EQUAL_AREA_CRS = "EPSG:6933"  # World Cylindrical Equal Area
_M2_PER_KM2 = 1_000_000


def _active_stations(stations: gpd.GeoDataFrame, as_of: pd.Timestamp, max_inactive_years: float) -> gpd.GeoDataFrame:
    cutoff = as_of - pd.Timedelta(days=max_inactive_years * 365.25)
    return stations[stations["end_date"].isna() | (stations["end_date"] >= cutoff)]


def station_density_by_basin(
    stations: gpd.GeoDataFrame,
    basin_polygons: gpd.GeoDataFrame,
    id_column: str,
    area_column: str | None = None,
    active_only: bool = True,
    max_inactive_years: float = 5.0,
    as_of: pd.Timestamp | None = None,
) -> gpd.GeoDataFrame:
    """Count stations per basin and compute density (stations / km^2).

    `area_column`, if given, is used directly (e.g. HydroBASINS' own
    `SUB_AREA`, already in km^2) rather than recomputing it -- avoids a
    reprojection when the source already provides a real area. Otherwise
    area is computed by reprojecting to an equal-area CRS.

    `active_only` filters to stations with no `end_date` (still reporting)
    or one within `max_inactive_years` of `as_of` (default: now). This is a
    simple recency filter for the static darkness *level*, not the
    cessation/replacement analysis `dark_basins` does -- a station with a
    recent end_date still counts as observability that existed until
    recently, not a decay event.

    Adds `n_stations`, `area_km2`, and `station_density` (stations/km^2) to
    a copy of `basin_polygons`. Basins with no stations get 0, not NaN.
    """
    if active_only:
        stations = _active_stations(stations, as_of or pd.Timestamp.now(), max_inactive_years)

    joined = gpd.sjoin(stations[["geometry"]], basin_polygons[[id_column, "geometry"]], predicate="within", how="inner")
    counts = joined.groupby(id_column).size().rename("n_stations")

    result = basin_polygons.merge(counts, on=id_column, how="left")
    result["n_stations"] = result["n_stations"].fillna(0).astype(int)

    if area_column is not None:
        result["area_km2"] = result[area_column]
    else:
        result["area_km2"] = result.geometry.to_crs(_EQUAL_AREA_CRS).area / _M2_PER_KM2

    result["station_density"] = result["n_stations"] / result["area_km2"]
    return result


def observability_score(density: pd.Series) -> pd.Series:
    """Percentile rank of station density across basins, in [0, 1].

    `method="min"` so every basin tied at the minimum (typically zero
    stations, usually most basins globally) gets the same low percentile
    rather than pandas' default tie-averaging, which would otherwise pull
    a zero-inflated majority up toward the middle of the distribution.
    """
    return density.rank(pct=True, method="min")


def darkness_score(density: pd.Series) -> pd.Series:
    """1 - observability_score: high darkness = sparse/absent monitoring."""
    return 1 - observability_score(density)
