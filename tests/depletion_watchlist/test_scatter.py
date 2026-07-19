import geopandas as gpd
from shapely.geometry import box

from dark_water.depletion_watchlist.product import scatter, tiers


def _watchlist():
    return gpd.GeoDataFrame(
        {
            "basin_id": [1, 2, 3, 4],
            "name": ["Alpha", "Beta", "Gamma", "Delta"],
            "mean_trend": [-2.0, -1.0, -0.5, 0.3],
            "darkness_score": [0.9, 0.5, 0.1, 0.9],
            "tier": [tiers.TIER_1, tiers.TIER_2, tiers.TIER_3, None],
        },
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1), box(4, 0, 5, 1), box(6, 0, 7, 1)],
        crs="EPSG:4326",
    )


def test_plot_watchlist_scatter_writes_a_file(tmp_path):
    output_path = tmp_path / "scatter.png"

    result = scatter.plot_watchlist_scatter(_watchlist(), output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_watchlist_scatter_with_labels_writes_a_file(tmp_path):
    output_path = tmp_path / "scatter_labeled.png"

    result = scatter.plot_watchlist_scatter(_watchlist(), output_path, label_column="name", label_top_n=2)

    assert result.exists()
