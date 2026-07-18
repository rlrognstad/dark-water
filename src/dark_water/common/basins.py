"""Basin unit ingestion: HydroBASINS (Pfafstetter levels) and WHYMAP aquifer polygons.

Shared between depletion_watchlist (zonal-aggregating the TWS trend) and
dark_basins (aggregating net station decay), per the pilot scope's basin-unit
inputs.
"""

import zipfile
from pathlib import Path

import geopandas as gpd

from dark_water.common.http import stream_download

HYDROBASINS_CONTINENTS = ("af", "ar", "as", "au", "eu", "gr", "na", "sa", "si")

WHYMAP_GWR_URL = "https://download.bgr.de/bgr/grundwasser/whymap/shp/WHYMAP_GWR_v1.zip"
WHYMAP_AQUIFERS_SHP_NAME = "whymap_GW_aquifers_v1_poly.shp"


def _hydrobasins_url(continent: str, level: int) -> str:
    if continent not in HYDROBASINS_CONTINENTS:
        raise ValueError(f"Unknown HydroBASINS continent {continent!r}, expected one of {HYDROBASINS_CONTINENTS}")
    return f"https://data.hydrosheds.org/file/hydrobasins/standard/hybas_{continent}_lev{level:02d}_v1c.zip"


def download_hydrobasins(continent: str, level: int, dest_dir: Path) -> Path:
    """Download and extract a HydroBASINS continent/level shapefile.

    `continent` is one of the standard HydroBASINS region codes
    (af, ar, as, au, eu, gr, na, sa, si); `level` is the Pfafstetter level
    (the pilot scope targets 4-5).
    """
    dest_dir = Path(dest_dir)
    zip_path = stream_download(_hydrobasins_url(continent, level), dest_dir)
    extract_dir = dest_dir / zip_path.stem
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    (shp_path,) = extract_dir.glob("*.shp")
    return shp_path


def download_whymap_aquifers(dest_dir: Path) -> Path:
    """Download and extract the WHYMAP Groundwater Resources of the World aquifer polygons.

    The WHYMAP GWR archive bundles many thematic layers (wetlands, rivers,
    permafrost, etc.) alongside the aquifer typology polygons — this picks
    out just the aquifer layer.
    """
    dest_dir = Path(dest_dir)
    zip_path = stream_download(WHYMAP_GWR_URL, dest_dir)
    extract_dir = dest_dir / zip_path.stem
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    (shp_path,) = extract_dir.glob(f"**/{WHYMAP_AQUIFERS_SHP_NAME}")
    return shp_path


def load_basins(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)
