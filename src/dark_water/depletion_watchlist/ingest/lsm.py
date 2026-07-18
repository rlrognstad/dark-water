"""GLDAS-2.1 land surface model ingestion, for the groundwater-attribution layer.

GLDAS is distributed by NASA GES DISC, which requires accepting its EULA on
your Earthdata profile — a separate step from a plain Earthdata login/
password. Without it, downloads fail with `EulaNotAccepted`. Accept it at
https://disc.gsfc.nasa.gov/earthdata-login (log in, then look for the GES
DISC / GESDISC DATA ARCHIVE application under your profile's authorized
apps) before using this module.
"""

from pathlib import Path

import earthaccess
import xarray as xr

GLDAS_MODELS = {
    "noah": "GLDAS_NOAH10_M",
    "vic": "GLDAS_VIC10_M",
    "clsm": "GLDAS_CLSM10_M",
}
GLDAS_VERSION = "2.1"


def download_gldas(model: str, dest_dir: Path, temporal: tuple | None = None) -> list:
    """Download monthly GLDAS-2.1 granules for one land surface model.

    `model` is one of "noah", "vic", "clsm". `temporal` bounds the search
    as (start, end) ISO date strings; omit to fetch the full 2000-present
    record (hundreds of monthly granules, one per month).
    """
    if model not in GLDAS_MODELS:
        raise ValueError(f"Unknown GLDAS model {model!r}, expected one of {tuple(GLDAS_MODELS)}")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    earthaccess.login()
    granules = earthaccess.search_data(short_name=GLDAS_MODELS[model], version=GLDAS_VERSION, temporal=temporal)
    if not granules:
        raise RuntimeError(f"No granules found for GLDAS model {model!r}")
    return [Path(p) for p in earthaccess.download(granules, str(dest_dir))]


def load_gldas(paths) -> xr.Dataset:
    return xr.open_mfdataset(sorted(str(p) for p in paths), combine="by_coords")
