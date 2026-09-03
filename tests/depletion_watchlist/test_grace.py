import numpy as np
import pytest
import xarray as xr

from dark_water.depletion_watchlist.ingest import grace


_LAT = [10.0, 20.0]
_LON = [100.0, 200.0]


def _mascon_ds(values):
    return xr.Dataset(
        {"lwe_thickness": (("time", "lat", "lon"), np.asarray(values, dtype=float))},
        coords={"time": [0, 1], "lat": _LAT, "lon": _LON},
    )


def test_load_mascons(tmp_path):
    ds = xr.Dataset({"lwe_thickness": (("time", "lat", "lon"), np.zeros((2, 3, 4)))})
    path = tmp_path / "mascons.nc"
    ds.to_netcdf(path)

    loaded = grace.load_mascons(path)

    assert "lwe_thickness" in loaded.data_vars
    assert loaded["lwe_thickness"].shape == (2, 3, 4)


def test_apply_scale_factor_amplifies_lwe_thickness(tmp_path):
    path = tmp_path / "mascons.nc"
    _mascon_ds(np.full((2, 2, 2), 10.0)).to_netcdf(path)

    scale_path = tmp_path / "scale.nc"
    xr.Dataset(
        {"scale_factor": (("lat", "lon"), np.full((2, 2), 1.3))},
        coords={"lat": _LAT, "lon": _LON},
    ).to_netcdf(scale_path)

    loaded = grace.load_mascons(path, scale_factor_path=scale_path)

    assert np.allclose(loaded["lwe_thickness"].values, 13.0)
    assert loaded["lwe_thickness"].attrs["scale_factor_applied"] == str(scale_path)


def test_load_mascons_without_scale_factor_leaves_values_untouched(tmp_path):
    path = tmp_path / "mascons.nc"
    _mascon_ds(np.full((2, 2, 2), 10.0)).to_netcdf(path)

    assert np.allclose(grace.load_mascons(path)["lwe_thickness"].values, 10.0)


def test_apply_land_mask_nans_ocean_pixels(tmp_path):
    path = tmp_path / "mascons.nc"
    _mascon_ds(np.full((2, 2, 2), 10.0)).to_netcdf(path)

    mask_path = tmp_path / "mask.nc"
    xr.Dataset(
        {"land_mask": (("lat", "lon"), np.array([[1.0, 0.0], [1.0, 0.0]]))},
        coords={"lat": _LAT, "lon": _LON},
    ).to_netcdf(mask_path)

    masked = grace.load_mascons(path, land_mask_path=mask_path)["lwe_thickness"]

    assert np.allclose(masked.sel(lon=100.0).values, 10.0)
    assert np.isnan(masked.sel(lon=200.0).values).all()


def test_scale_factor_file_missing_expected_variable_raises(tmp_path):
    path = tmp_path / "mascons.nc"
    _mascon_ds(np.full((2, 2, 2), 10.0)).to_netcdf(path)

    scale_path = tmp_path / "scale.nc"
    xr.Dataset({"wrong_name": (("lat", "lon"), np.ones((2, 2)))}, coords={"lat": _LAT, "lon": _LON}).to_netcdf(
        scale_path
    )

    with pytest.raises(KeyError, match="wrong_name"):
        grace.load_mascons(path, scale_factor_path=scale_path)
