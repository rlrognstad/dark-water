import numpy as np
import xarray as xr

from dark_water.depletion_watchlist.depletion import attribution


def _time_coord(n=4):
    return xr.DataArray(np.arange(n), dims="time")


def _noah_ds(soil_layers_kg_m2, swe_kg_m2, canopy_kg_m2, n=4):
    return xr.Dataset(
        {
            "SoilMoi0_10cm_inst": ("time", np.full(n, soil_layers_kg_m2[0])),
            "SoilMoi10_40cm_inst": ("time", np.full(n, soil_layers_kg_m2[1])),
            "SoilMoi40_100cm_inst": ("time", np.full(n, soil_layers_kg_m2[2])),
            "SoilMoi100_200cm_inst": ("time", np.full(n, soil_layers_kg_m2[3])),
            "SWE_inst": ("time", np.full(n, swe_kg_m2)),
            "CanopInt_inst": ("time", np.full(n, canopy_kg_m2)),
        },
        coords={"time": _time_coord(n)},
    )


def test_noah_non_gw_storage_sums_layers_and_converts_units():
    ds = _noah_ds(soil_layers_kg_m2=[100.0, 200.0, 300.0, 400.0], swe_kg_m2=50.0, canopy_kg_m2=10.0)
    result = attribution._noah_non_gw_storage(ds)
    # (100+200+300+400+50+10) kg/m^2 = 1060 mm = 106 cm
    assert np.allclose(result.values, 106.0)


def test_clsm_uses_root_zone_not_profile_soil_moisture():
    ds = xr.Dataset(
        {
            "SoilMoist_RZ_inst": ("time", np.full(4, 200.0)),
            "SoilMoist_P_inst": ("time", np.full(4, 999.0)),  # must be ignored
            "SWE_inst": ("time", np.full(4, 0.0)),
            "CanopInt_inst": ("time", np.full(4, 0.0)),
        },
        coords={"time": _time_coord(4)},
    )
    result = attribution._clsm_non_gw_storage(ds)
    assert np.allclose(result.values, 20.0)  # 200 kg/m^2 = 20 cm; ignores the 999 profile value


def test_groundwater_storage_anomaly_removes_lsm_own_mean_not_raw_value():
    # A constant LSM offset (e.g. a bias in absolute soil moisture) must not
    # show up in the GWS anomaly -- only deviations from the LSM's own mean.
    ds = _noah_ds(soil_layers_kg_m2=[1000.0, 1000.0, 1000.0, 1000.0], swe_kg_m2=0.0, canopy_kg_m2=0.0)
    grace_tws_anomaly = xr.DataArray(np.zeros(4), dims="time", coords={"time": _time_coord(4)})

    result = attribution.groundwater_storage_anomaly(grace_tws_anomaly, ds, "noah")

    assert np.allclose(result.values, 0.0)


def test_groundwater_storage_anomaly_reflects_lsm_time_variation():
    ds = _noah_ds(soil_layers_kg_m2=[0.0, 0.0, 0.0, 0.0], swe_kg_m2=0.0, canopy_kg_m2=0.0)
    # Vary SWE over time: LSM storage rises, so the GWS residual should fall.
    ds["SWE_inst"] = ("time", np.array([0.0, 100.0, 200.0, 300.0]))
    grace_tws_anomaly = xr.DataArray(np.zeros(4), dims="time", coords={"time": _time_coord(4)})

    result = attribution.groundwater_storage_anomaly(grace_tws_anomaly, ds, "noah")

    assert result.values[0] > result.values[-1]


def test_ensemble_attribution_reports_mean_and_spread_across_models():
    grace_tws_anomaly = xr.DataArray(np.zeros(4), dims="time", coords={"time": _time_coord(4)})
    noah_ds = _noah_ds(soil_layers_kg_m2=[0.0, 0.0, 0.0, 0.0], swe_kg_m2=0.0, canopy_kg_m2=0.0)
    vic_ds = xr.Dataset(
        {
            "SoilMoi0_30cm_inst": ("time", np.full(4, 100.0)),
            "SoilMoi_depth2_inst": ("time", np.full(4, 0.0)),
            "SoilMoi_depth3_inst": ("time", np.full(4, 0.0)),
            "SWE_inst": ("time", np.full(4, 0.0)),
            "CanopInt_inst": ("time", np.full(4, 0.0)),
        },
        coords={"time": _time_coord(4)},
    )

    result = attribution.ensemble_attribution(grace_tws_anomaly, {"noah": noah_ds, "vic": vic_ds})

    # Both models are constant over time (no anomaly relative to their own
    # mean), so GWS estimates should agree and the spread should be ~0.
    assert np.allclose(result["ensemble_spread"].values, 0.0, atol=1e-10)
    assert "noah" in result and "vic" in result
