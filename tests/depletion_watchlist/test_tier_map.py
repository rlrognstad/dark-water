import geopandas as gpd
from PIL import Image
from shapely.geometry import box

from dark_water.depletion_watchlist.product import tier_map, tiers


def _watchlist():
    return gpd.GeoDataFrame(
        {
            "basin_id": [1, 2, 3, 4],
            "tier": [tiers.TIER_1, tiers.TIER_2, tiers.TIER_3, None],
        },
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1), box(4, 0, 5, 1), box(6, 0, 7, 1)],
        crs="EPSG:4326",
    )


def test_plot_tier_map_writes_a_file(tmp_path):
    output_path = tmp_path / "tier_map.png"

    result = tier_map.plot_tier_map(_watchlist(), output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_tier_map_dark_theme_uses_a_dark_surface(tmp_path):
    output_path = tmp_path / "tier_map_dark.png"

    result = tier_map.plot_tier_map(_watchlist(), output_path, theme="dark")

    assert result.exists()
    corner = Image.open(output_path).convert("RGB").getpixel((0, 0))
    assert sum(corner) < 200


def test_plot_tier_map_handles_no_untiered_basins(tmp_path):
    output_path = tmp_path / "tier_map_all_tiered.png"
    gdf = _watchlist()
    gdf = gdf[gdf["tier"].notna()]

    result = tier_map.plot_tier_map(gdf, output_path)

    assert result.exists()
