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


def _first_variable(ds: xr.Dataset, candidates: tuple[str, ...], path: Path) -> xr.DataArray:
    for name in candidates:
        if name in ds:
            return ds[name]
    raise KeyError(f"None of {candidates} found in {path}; got {tuple(ds.data_vars)}")


def apply_scale_factor(ds: xr.Dataset, scale_factor_path: Path, variable: str = "lwe_thickness") -> xr.Dataset:
    """Multiply mascon LWE thickness by JPL's CLM4-derived gain factor grid.

    Mascon solutions are spatially smoothed by the gravity inversion, which
    leaks signal out of high-amplitude basins into their surroundings and
    damps the basin's own amplitude. JPL ships a per-pixel gain factor
    (derived by passing CLM4 output through the same processing) precisely
    to undo that damping, and the dataset documentation treats applying it
    as mandatory, not optional, for basin-scale amplitudes. Skipping it
    biases depletion magnitudes low -- materially so in Central Valley,
    where the factor exceeds 1.

    The gain grid is a separate ancillary file in the same PO.DAAC
    collection as the mascon granule (for RL06 CRI, distributed as a
    `*SCALE_FACTOR*.nc`); download it alongside the granule and pass the
    path here. It is not applied automatically because the correct file is
    release-specific and pairing the wrong release's grid to a granule is a
    silent error, not a loud one.
    """
    with xr.open_dataset(scale_factor_path) as scale_ds:
        scale = _first_variable(scale_ds, ("scale_factor", "SCALE_FACTOR", "gain_factor"), scale_factor_path).load()

    scale = scale.assign_coords(lon=scale["lon"] % 360).sortby("lon")
    result = ds.copy()
    result[variable] = ds[variable] * scale.interp(lat=ds["lat"], lon=ds["lon"], method="nearest")
    result[variable].attrs = dict(ds[variable].attrs, scale_factor_applied=str(scale_factor_path))
    return result


def apply_land_mask(ds: xr.Dataset, land_mask_path: Path) -> xr.Dataset:
    """Mask ocean pixels using the mascon collection's land-mask ancillary file.

    The mascon granule itself carries no `land_mask` variable -- the mask is
    a separate file. Ocean mascons carry real ocean-mass signal that is not
    terrestrial water storage, so leaving them in pollutes any zonal or
    global statistic that touches a coastline.
    """
    with xr.open_dataset(land_mask_path) as mask_ds:
        mask = _first_variable(mask_ds, ("land_mask", "LO_val", "land"), land_mask_path).load()

    mask = mask.assign_coords(lon=mask["lon"] % 360).sortby("lon")
    return ds.where(mask.interp(lat=ds["lat"], lon=ds["lon"], method="nearest") > 0)


def load_mascons(
    path: Path, scale_factor_path: Path | None = None, land_mask_path: Path | None = None
) -> xr.Dataset:
    """Open a mascon granule, optionally applying the gain factor and land mask.

    Both ancillary steps are opt-in by path so that a caller working with
    GSFC or CSR solutions -- which ship their own conventions and are not
    interchangeable with JPL's ancillary grids -- is never silently handed a
    JPL correction.
    """
    ds = xr.open_dataset(path)
    if scale_factor_path is not None:
        ds = apply_scale_factor(ds, scale_factor_path)
    if land_mask_path is not None:
        ds = apply_land_mask(ds, land_mask_path)
    return ds
