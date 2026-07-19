import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from dark_water.depletion_watchlist.product import tiers


def _depletion():
    return gpd.GeoDataFrame(
        {
            "basin_id": [1, 2, 3, 4],
            "mean_trend": [-2.0, -1.0, -0.5, 0.3],
            "basin_significant_decline": [True, True, True, False],
        },
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1), box(4, 0, 5, 1), box(6, 0, 7, 1)],
        crs="EPSG:4326",
    )


def _darkness():
    return pd.DataFrame(
        {
            "basin_id": [1, 2, 3, 4],
            "darkness_score": [0.9, 0.5, 0.1, 0.9],
            "n_stations": [0, 3, 40, 0],
        }
    )


def test_join_depletion_and_darkness_merges_on_id_and_keeps_one_geometry():
    joined = tiers.join_depletion_and_darkness(_depletion(), _darkness(), "basin_id")

    assert "darkness_score" in joined.columns
    assert "n_stations" in joined.columns
    assert joined.crs == _depletion().crs
    assert list(joined["basin_id"]) == [1, 2, 3, 4]


def test_assign_tiers_buckets_by_darkness_and_requires_decline():
    joined = tiers.join_depletion_and_darkness(_depletion(), _darkness(), "basin_id")

    tier = tiers.assign_tiers(joined)

    assert tier.loc[joined["basin_id"] == 1].iloc[0] == tiers.TIER_1
    assert tier.loc[joined["basin_id"] == 2].iloc[0] == tiers.TIER_2
    assert tier.loc[joined["basin_id"] == 3].iloc[0] == tiers.TIER_3
    # basin 4 has high darkness but no significant decline -- no tier
    assert pd.isna(tier.loc[joined["basin_id"] == 4].iloc[0])


def test_assign_tiers_respects_custom_thresholds():
    joined = tiers.join_depletion_and_darkness(_depletion(), _darkness(), "basin_id")

    tier = tiers.assign_tiers(joined, dark_threshold=0.6, dim_threshold=0.4)

    # basin 2 (darkness 0.5) now falls in the dim band under looser thresholds
    assert tier.loc[joined["basin_id"] == 2].iloc[0] == tiers.TIER_2
