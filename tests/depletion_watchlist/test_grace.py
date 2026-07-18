import numpy as np
import xarray as xr

from dark_water.depletion_watchlist.ingest import grace


def test_load_mascons(tmp_path):
    ds = xr.Dataset({"lwe_thickness": (("time", "lat", "lon"), np.zeros((2, 3, 4)))})
    path = tmp_path / "mascons.nc"
    ds.to_netcdf(path)

    loaded = grace.load_mascons(path)

    assert "lwe_thickness" in loaded.data_vars
    assert loaded["lwe_thickness"].shape == (2, 3, 4)
