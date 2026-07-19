"""Assemble the watchlist: join depletion and darkness, assign tiers.

Per the concept doc, DDW is a labeled scatter, not a ranked index, and the
tiers are the rhetorical engine:

    Tier 1 -- Dark & Falling:    significant depletion, near-zero observability
    Tier 2 -- Dim & Falling:     significant depletion, sparse network
    Tier 3 -- Watched & Falling: significant depletion, adequate network

A basin without a statistically significant decline gets no tier -- the
tiers are explicitly "... & Falling", not a general observability ranking.
"""

import geopandas as gpd
import pandas as pd

TIER_1 = "1 -- Dark & Falling"
TIER_2 = "2 -- Dim & Falling"
TIER_3 = "3 -- Watched & Falling"
TIER_ORDER = [TIER_1, TIER_2, TIER_3]

# dataviz skill's status palette (reserved, never themed): tier color encodes
# monitoring state (dark/dim/watched), not water-supply health or a
# categorical series -- shared by every product view of the watchlist so a
# tier always reads the same color regardless of which chart it's in.
TIER_COLORS = {
    TIER_1: "#d03b3b",  # critical
    TIER_2: "#fab219",  # warning
    TIER_3: "#0ca30c",  # good (well-observed, not "healthy aquifer")
}


def join_depletion_and_darkness(
    depletion: gpd.GeoDataFrame, darkness: pd.DataFrame, id_column: str
) -> gpd.GeoDataFrame:
    """Join basin-level depletion stats (`zonal.aggregate_trend_to_basins`)
    with basin-level darkness stats (`density.station_density_by_basin`).

    Keeps `depletion`'s geometry and drops `darkness`'s to avoid a duplicate
    geometry column -- both are built from the same basin polygon set, so
    the geometries are identical and only need to be carried once.
    """
    darkness_cols = [c for c in darkness.columns if c != "geometry"]
    return depletion.merge(darkness[darkness_cols], on=id_column, how="left")


def assign_tiers(
    gdf: gpd.GeoDataFrame,
    depletion_column: str = "basin_significant_decline",
    darkness_column: str = "darkness_score",
    dark_threshold: float = 2 / 3,
    dim_threshold: float = 1 / 3,
) -> pd.Series:
    """Assign a tier label to each basin, or `None` if not significantly declining.

    Thresholds are applied directly to `darkness_column` (a global percentile
    rank across the *entire* basin set -- see `darkness/density.py`), not
    re-ranked within just the declining subset. Re-ranking would make a
    basin's tier depend on which other declining basins happened to be in
    the same run; a fixed cut on the global score keeps tier assignment
    stable as the declining population changes.
    """
    declining = gdf[depletion_column].astype(bool)
    darkness = gdf[darkness_column]

    tier = pd.Series(pd.NA, index=gdf.index, dtype="object")
    tier[declining & (darkness >= dark_threshold)] = TIER_1
    tier[declining & (darkness >= dim_threshold) & (darkness < dark_threshold)] = TIER_2
    tier[declining & (darkness < dim_threshold)] = TIER_3
    return tier
