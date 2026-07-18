import zipfile

import geopandas as gpd
import pytest
from shapely.geometry import box

from dark_water.common import basins


def _write_shapefile_zip(dest_dir, stem):
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
        crs="EPSG:4326",
    )
    shp_dir = dest_dir / "shp_src"
    shp_dir.mkdir()
    shp_path = shp_dir / f"{stem}.shp"
    gdf.to_file(shp_path)

    zip_path = dest_dir / f"{stem}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in shp_dir.iterdir():
            zf.write(f, f.name)
    return zip_path


def test_hydrobasins_url_rejects_unknown_continent():
    with pytest.raises(ValueError):
        basins._hydrobasins_url("zz", 4)


def test_hydrobasins_url_pads_level():
    assert basins._hydrobasins_url("af", 4).endswith("hybas_af_lev04_v1c.zip")


def test_download_hydrobasins_extracts_and_loads(local_http_server, tmp_path, monkeypatch):
    served_dir, base_url = local_http_server
    zip_path = _write_shapefile_zip(served_dir, "hybas_af_lev04_v1c")
    assert zip_path.exists()

    monkeypatch.setattr(
        basins, "_hydrobasins_url", lambda continent, level: f"{base_url}/hybas_af_lev04_v1c.zip"
    )

    shp_path = basins.download_hydrobasins("af", 4, tmp_path / "dest")
    gdf = basins.load_basins(shp_path)

    assert len(gdf) == 2


def test_download_whymap_aquifers_picks_aquifer_layer(local_http_server, tmp_path, monkeypatch):
    served_dir, base_url = local_http_server

    shp_dir = served_dir / "shp_src"
    shp_dir.mkdir()
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    gdf.to_file(shp_dir / "whymap_GW_aquifers_v1_poly.shp")
    other_gdf = gpd.GeoDataFrame({"id": [1, 2, 3]}, geometry=[box(0, 0, 1, 1)] * 3, crs="EPSG:4326")
    other_gdf.to_file(shp_dir / "whymap_rivers__v1_line.shp")

    zip_path = served_dir / "WHYMAP_GWR_v1.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in shp_dir.iterdir():
            zf.write(f, f"WHYMAP_GWR/shp/{f.name}")

    monkeypatch.setattr(basins, "WHYMAP_GWR_URL", f"{base_url}/WHYMAP_GWR_v1.zip")

    shp_path = basins.download_whymap_aquifers(tmp_path / "dest")
    loaded = basins.load_basins(shp_path)

    assert shp_path.name == basins.WHYMAP_AQUIFERS_SHP_NAME
    assert len(loaded) == 1
