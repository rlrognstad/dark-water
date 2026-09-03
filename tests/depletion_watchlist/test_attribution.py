import numpy as np
import pandas as pd
import xarray as xr

from dark_water.depletion_watchlist.depletion import attribution

# A GLDAS-like 2x2 grid (1.0 deg, -180/180 lon) whose bounding box contains
# the GRACE-like point used below, so bilinear interpolation never
# extrapolates.
_LSM_LAT = [10.0, 20.0]
_LSM_LON = [-100.0, -90.0]

# A GRACE-like single point (0.25 deg, 0/360 lon) inside the LSM grid's
# bounding box: lon -100..-90 == 260..270 in 0/360 convention.
_GRACE_LAT = [15.0]
_GRACE_LON = [265.0]

_GLDAS_MONTH_START = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"])
_GRACE_MID_MONTH = pd.to_datetime(["2020-01-17", "2020-02-15", "2020-03-17", "2020-04-16"])


def _grid_var(value_per_time):
    # Uniform across the 2x2 spatial grid, so any interpolation inside the
    # bounding box returns the same value regardless of exact weights.
    n = len(value_per_time)
    data = np.broadcast_to(np.asarray(value_per_time)[:, None, None], (n, 2, 2))
    return ("time", "lat", "lon"), data


def _noah_ds(soil_layers_kg_m2, swe_kg_m2, canopy_kg_m2, n=4):
    def const(v):
        return _grid_var(np.full(n, v))

    return xr.Dataset(
        {
            "SoilMoi0_10cm_inst": const(soil_layers_kg_m2[0]),
            "SoilMoi10_40cm_inst": const(soil_layers_kg_m2[1]),
            "SoilMoi40_100cm_inst": const(soil_layers_kg_m2[2]),
            "SoilMoi100_200cm_inst": const(soil_layers_kg_m2[3]),
            "SWE_inst": const(swe_kg_m2) if np.isscalar(swe_kg_m2) else _grid_var(swe_kg_m2),
            "CanopInt_inst": const(canopy_kg_m2),
        },
        coords={"time": _GLDAS_MONTH_START[:n], "lat": _LSM_LAT, "lon": _LSM_LON},
    )


def _grace_da(values):
    n = len(values)
    data = np.asarray(values).reshape(n, 1, 1)
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": _GRACE_MID_MONTH[:n], "lat": _GRACE_LAT, "lon": _GRACE_LON},
    )


def test_noah_non_gw_storage_sums_layers_and_converts_units():
    ds = _noah_ds(soil_layers_kg_m2=[100.0, 200.0, 300.0, 400.0], swe_kg_m2=50.0, canopy_kg_m2=10.0)
    result = attribution._noah_non_gw_storage(ds)
    # (100+200+300+400+50+10) kg/m^2 = 1060 mm = 106 cm
    assert np.allclose(result.values, 106.0)


def test_clsm_uses_root_zone_not_profile_soil_moisture():
    ds = xr.Dataset(
        {
            "SoilMoist_RZ_inst": _grid_var(np.full(4, 200.0)),
            "SoilMoist_P_inst": _grid_var(np.full(4, 999.0)),  # must be ignored
            "SWE_inst": _grid_var(np.full(4, 0.0)),
            "CanopInt_inst": _grid_var(np.full(4, 0.0)),
        },
        coords={"time": _GLDAS_MONTH_START, "lat": _LSM_LAT, "lon": _LSM_LON},
    )
    result = attribution._clsm_non_gw_storage(ds)
    assert np.allclose(result.values, 20.0)  # 200 kg/m^2 = 20 cm; ignores the 999 profile value


def test_to_grace_grid_handles_longitude_convention_and_resolution():
    ds = _noah_ds(soil_layers_kg_m2=[10.0, 0.0, 0.0, 0.0], swe_kg_m2=0.0, canopy_kg_m2=0.0)
    non_gw = attribution._noah_non_gw_storage(ds)
    grace_da = _grace_da([0.0, 0.0, 0.0, 0.0])

    regridded = attribution._to_grace_grid(non_gw, grace_da)

    assert regridded.sizes == {"time": 4, "lat": 1, "lon": 1}
    assert np.allclose(regridded.values, 1.0)  # 10 kg/m^2 = 1 cm, uniform across the grid


def test_monthly_aligns_mid_month_and_month_start_dates():
    grace_da = _grace_da([1.0, 2.0, 3.0, 4.0])
    gldas_da = xr.DataArray(
        np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1, 1),
        dims=("time", "lat", "lon"),
        coords={"time": _GLDAS_MONTH_START, "lat": _GRACE_LAT, "lon": _GRACE_LON},
    )
    assert (attribution._monthly(grace_da)["time"].values == attribution._monthly(gldas_da)["time"].values).all()


def test_monthly_averages_duplicate_sub_monthly_solutions():
    # GRACE occasionally has two mascon solutions within the same calendar
    # month (a known orbit/battery-driven quirk, not a data error).
    times = pd.to_datetime(["2012-01-06", "2012-01-24", "2012-02-15"])
    da = xr.DataArray(
        np.array([10.0, 20.0, 30.0]).reshape(3, 1, 1),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": _GRACE_LAT, "lon": _GRACE_LON},
    )

    result = attribution._monthly(da)

    assert result.sizes["time"] == 2
    assert np.allclose(sorted(result.values.ravel()), [15.0, 30.0])


def test_groundwater_storage_anomaly_removes_lsm_own_mean_not_raw_value():
    # A constant LSM offset (e.g. a bias in absolute soil moisture) must not
    # show up in the GWS anomaly -- only deviations from the LSM's own mean.
    ds = _noah_ds(soil_layers_kg_m2=[1000.0, 1000.0, 1000.0, 1000.0], swe_kg_m2=0.0, canopy_kg_m2=0.0)
    grace_tws_anomaly = _grace_da([0.0, 0.0, 0.0, 0.0])

    result = attribution.groundwater_storage_anomaly(grace_tws_anomaly, ds, "noah")

    assert np.allclose(result.values, 0.0)


def test_groundwater_storage_anomaly_reflects_lsm_time_variation():
    ds = _noah_ds(soil_layers_kg_m2=[0.0, 0.0, 0.0, 0.0], swe_kg_m2=[0.0, 100.0, 200.0, 300.0], canopy_kg_m2=0.0)
    grace_tws_anomaly = _grace_da([0.0, 0.0, 0.0, 0.0])

    result = attribution.groundwater_storage_anomaly(grace_tws_anomaly, ds, "noah")

    assert result.values[0, 0, 0] > result.values[-1, 0, 0]


def test_to_grace_grid_interpolates_across_the_prime_meridian():
    # GLDAS cell centers stop at 0.5 and 359.5 after the 0-360 conversion;
    # GRACE's grid runs past both. Without periodic padding these columns
    # fall outside the interpolation domain and come back NaN.
    lsm = xr.Dataset(
        {
            "SoilMoi0_10cm_inst": (("time", "lat", "lon"), np.full((1, 2, 4), 10.0)),
            "SoilMoi10_40cm_inst": (("time", "lat", "lon"), np.zeros((1, 2, 4))),
            "SoilMoi40_100cm_inst": (("time", "lat", "lon"), np.zeros((1, 2, 4))),
            "SoilMoi100_200cm_inst": (("time", "lat", "lon"), np.zeros((1, 2, 4))),
            "SWE_inst": (("time", "lat", "lon"), np.zeros((1, 2, 4))),
            "CanopInt_inst": (("time", "lat", "lon"), np.zeros((1, 2, 4))),
        },
        coords={
            "time": _GLDAS_MONTH_START[:1],
            "lat": [10.0, 20.0],
            "lon": [-179.5, -0.5, 0.5, 179.5],
        },
    )
    grace_da = xr.DataArray(
        np.zeros((1, 1, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": _GRACE_MID_MONTH[:1], "lat": [15.0], "lon": [0.125, 359.875]},
    )

    regridded = attribution._to_grace_grid(attribution._noah_non_gw_storage(lsm), grace_da)

    assert not np.isnan(regridded.values).any()
    assert np.allclose(regridded.values, 1.0)  # 10 kg/m^2 = 1 cm, uniform


def test_baselines_are_removed_over_common_months_only():
    # GRACE carries a 4th month that GLDAS does not -- stand-in for the
    # 2017-06..2018-05 GRACE/GRACE-FO gap and the scattered missing months,
    # which GLDAS does not share. With the LSM constant, the whole residual
    # is GRACE's own anomaly, so it must be centered on the 3 shared months.
    grace_tws = _grace_da([0.0, 0.0, 0.0, 100.0])
    lsm = _noah_ds(soil_layers_kg_m2=[500.0, 0.0, 0.0, 0.0], swe_kg_m2=0.0, canopy_kg_m2=0.0, n=3)

    result = attribution.groundwater_storage_anomaly(grace_tws, lsm, "noah")

    assert result.sizes["time"] == 3
    assert np.allclose(result.values, 0.0)


def test_result_does_not_depend_on_the_callers_baseline_choice():
    # The old dashboard de-meaned GRACE over its full record before calling
    # in, while attribution differenced only the shared months -- so the
    # difference between the two baselines survived as a constant offset in
    # the residual. Output must now be invariant to what the caller passes.
    lsm = _noah_ds(soil_layers_kg_m2=[500.0, 0.0, 0.0, 0.0], swe_kg_m2=0.0, canopy_kg_m2=0.0, n=3)
    raw = _grace_da([0.0, 0.0, 0.0, 100.0])
    pre_centered = raw - raw.mean(dim="time")  # what the caller used to do

    from_raw = attribution.groundwater_storage_anomaly(raw, lsm, "noah")
    from_pre_centered = attribution.groundwater_storage_anomaly(pre_centered, lsm, "noah")

    assert np.allclose(from_raw.values, from_pre_centered.values)
    assert np.allclose(from_raw.mean(dim="time").values, 0.0)


def test_ensemble_spread_is_not_inflated_by_uneven_model_coverage():
    # VIC stops a month early. A union-join concat would pad it with NaN and
    # poison ensemble_min/max on that month.
    grace_tws = _grace_da([0.0, 0.0, 0.0, 0.0])
    noah_ds = _noah_ds(soil_layers_kg_m2=[0.0, 0.0, 0.0, 0.0], swe_kg_m2=0.0, canopy_kg_m2=0.0, n=4)
    vic_ds = xr.Dataset(
        {
            "SoilMoi0_30cm_inst": _grid_var(np.full(3, 100.0)),
            "SoilMoi_depth2_inst": _grid_var(np.full(3, 0.0)),
            "SoilMoi_depth3_inst": _grid_var(np.full(3, 0.0)),
            "SWE_inst": _grid_var(np.full(3, 0.0)),
            "CanopInt_inst": _grid_var(np.full(3, 0.0)),
        },
        coords={"time": _GLDAS_MONTH_START[:3], "lat": _LSM_LAT, "lon": _LSM_LON},
    )

    result = attribution.ensemble_attribution(grace_tws, {"noah": noah_ds, "vic": vic_ds})

    assert result.sizes["time"] == 3
    assert not np.isnan(result["ensemble_spread"].values).any()
    assert np.allclose(result["ensemble_spread"].values, 0.0, atol=1e-10)


def test_ensemble_attribution_reports_mean_and_spread_across_models():
    grace_tws_anomaly = _grace_da([0.0, 0.0, 0.0, 0.0])
    noah_ds = _noah_ds(soil_layers_kg_m2=[0.0, 0.0, 0.0, 0.0], swe_kg_m2=0.0, canopy_kg_m2=0.0)
    vic_ds = xr.Dataset(
        {
            "SoilMoi0_30cm_inst": _grid_var(np.full(4, 100.0)),
            "SoilMoi_depth2_inst": _grid_var(np.full(4, 0.0)),
            "SoilMoi_depth3_inst": _grid_var(np.full(4, 0.0)),
            "SWE_inst": _grid_var(np.full(4, 0.0)),
            "CanopInt_inst": _grid_var(np.full(4, 0.0)),
        },
        coords={"time": _GLDAS_MONTH_START, "lat": _LSM_LAT, "lon": _LSM_LON},
    )

    result = attribution.ensemble_attribution(grace_tws_anomaly, {"noah": noah_ds, "vic": vic_ds})

    # Both models are constant over time (no anomaly relative to their own
    # mean), so GWS estimates should agree and the spread should be ~0.
    assert np.allclose(result["ensemble_spread"].values, 0.0, atol=1e-10)
    assert "noah" in result and "vic" in result
