import pandas as pd

from dark_water.depletion_watchlist.product import tables, tiers


def _watchlist():
    return pd.DataFrame(
        {
            "basin_id": [1, 2, 3, 4, 5],
            "tier": [tiers.TIER_1, tiers.TIER_1, tiers.TIER_3, None, tiers.TIER_1],
            "mean_trend": [-1.0, -5.0, -3.0, 0.2, -2.0],
            "darkness_score": [0.9, 0.95, 0.1, 0.9, 0.85],
            "n_stations": [0, 0, 40, 0, 1],
            "frac_pixels_significant_decline": [0.9, 1.0, 0.8, 0.0, 0.7],
        }
    )


def test_rank_top_basins_filters_to_tier_and_sorts_by_severity():
    table = tables.rank_top_basins(_watchlist(), id_column="basin_id", n=2)

    assert list(table["basin_id"]) == [2, 5]  # tier 1 only, most negative trend first
    assert list(table.columns) == ["basin_id", "tier", "mean_trend", "darkness_score", "n_stations", "frac_pixels_significant_decline"]


def test_rank_top_basins_across_all_tiers_when_tier_is_none():
    table = tables.rank_top_basins(_watchlist(), id_column="basin_id", n=10, tier=None)

    assert list(table["basin_id"]) == [2, 3, 5, 1]  # excludes basin 4 (no tier)


def test_save_table_writes_csv(tmp_path):
    table = tables.rank_top_basins(_watchlist(), id_column="basin_id", n=2)
    output_path = tmp_path / "top_basins.csv"

    result = tables.save_table(table, output_path)

    assert result == output_path
    written = pd.read_csv(output_path)
    assert list(written["basin_id"]) == [2, 5]
