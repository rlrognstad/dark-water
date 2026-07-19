import geopandas as gpd
from shapely.geometry import box

from dark_water.depletion_watchlist.product import darkness_distribution


def _basins():
    return gpd.GeoDataFrame(
        {
            "basin_id": range(6),
            "darkness_score": [0.0, 0.1, 0.2, 0.9, 0.95, 0.99],
        },
        geometry=[box(i, 0, i + 1, 1) for i in range(6)],
        crs="EPSG:4326",
    )


def test_plot_darkness_distribution_writes_a_file(tmp_path):
    output_path = tmp_path / "distribution.png"

    result = darkness_distribution.plot_darkness_distribution(_basins(), output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_darkness_distribution_respects_custom_thresholds(tmp_path):
    output_path = tmp_path / "distribution_custom.png"

    result = darkness_distribution.plot_darkness_distribution(
        _basins(), output_path, dark_threshold=0.5, dim_threshold=0.15
    )

    assert result.exists()
