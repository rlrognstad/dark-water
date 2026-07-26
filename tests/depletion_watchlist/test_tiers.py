import geopandas as gpd
import pandas as pd
import pytest
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


def _with_unexplained(fractions):
    joined = tiers.join_depletion_and_darkness(_depletion(), _darkness(), "basin_id")
    joined["mean_fraction_unexplained"] = fractions
    return joined


def test_precipitation_gate_is_off_by_default():
    # Every basin's decline is fully explained by precipitation, but without
    # the gate the tiers must be unchanged from the pre-covariate behavior.
    joined = _with_unexplained([0.0, 0.0, 0.0, 0.0])

    tier = tiers.assign_tiers(joined)

    assert tier.loc[joined["basin_id"] == 1].iloc[0] == tiers.TIER_1


def test_precipitation_gate_untiers_a_drought_driven_decline():
    # Basin 1 is dark and falling, but the fall is a dry decade; basin 2's
    # decline survives the control.
    joined = _with_unexplained([0.1, 0.8, 0.8, 0.0])

    tier = tiers.assign_tiers(joined, min_unexplained_fraction=0.5)

    assert pd.isna(tier.loc[joined["basin_id"] == 1].iloc[0])
    assert tier.loc[joined["basin_id"] == 2].iloc[0] == tiers.TIER_2


def test_precipitation_gate_treats_undefined_fraction_as_not_passing():
    joined = _with_unexplained([float("nan"), 0.8, 0.8, 0.8])

    tier = tiers.assign_tiers(joined, min_unexplained_fraction=0.5)

    assert pd.isna(tier.loc[joined["basin_id"] == 1].iloc[0])


def test_precipitation_gate_raises_when_the_covariate_is_missing():
    joined = tiers.join_depletion_and_darkness(_depletion(), _darkness(), "basin_id")

    with pytest.raises(KeyError, match="mean_fraction_unexplained"):
        tiers.assign_tiers(joined, min_unexplained_fraction=0.5)
