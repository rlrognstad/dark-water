"""GRACE/GRACE-FO mascon ingestion: JPL and GSFC (auto), CSR (manual URL).

Each mascon center reports Total Water Storage anomaly independently. DDW
uses all three as an ensemble and reports sensitivity across them, so this
module treats them as separate sources rather than reconciling them here.
"""

from pathlib import Path

import earthaccess
import xarray as xr

from dark_water.common.http import stream_download

JPL_SHORT_NAME = "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4"

GSFC_URL = (
    "https://earth.gsfc.nasa.gov/sites/default/files/geo/"
    "gsfc.glb_.200204_202603_rl06v2.0_obp-ice6gd_halfdegree.nc"
)


def download_jpl_mascons(dest_dir: Path) -> Path:
    """Download the JPL GRACE/GRACE-FO mascon CRI grid via NASA Earthdata.

    Requires Earthdata credentials, discovered by earthaccess from
    ~/.netrc or the EARTHDATA_USERNAME/EARTHDATA_PASSWORD env vars.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    earthaccess.login()
    granules = earthaccess.search_data(short_name=JPL_SHORT_NAME)
    if not granules:
        raise RuntimeError(f"No granules found for {JPL_SHORT_NAME!r}")
    (path,) = earthaccess.download(granules[:1], str(dest_dir))
    return Path(path)


def download_gsfc_mascons(dest_dir: Path) -> Path:
    """Download the GSFC GRACE/GRACE-FO mascon half-degree grid."""
    return stream_download(GSFC_URL, dest_dir)


def download_csr_mascons(url: str, dest_dir: Path) -> Path:
    """Download the CSR GRACE/GRACE-FO mascon solution grid.

    Unlike JPL and GSFC, CSR does not serve the mascon solution grid from a
    stable public URL — obtain the current download link from
    https://www2.csr.utexas.edu/grace/RL0603_mascons.html and pass it here.
    """
    return stream_download(url, dest_dir)


def load_mascons(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)
