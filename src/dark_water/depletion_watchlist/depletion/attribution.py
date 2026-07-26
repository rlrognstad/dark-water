"""Groundwater-attribution layer: GRACE-minus-LSM residual, per model.

TWS trend/significance (see `trend.py`) is the headline signal. Attribution
to groundwater specifically requires subtracting a land-surface model's
modeled soil moisture + snow + canopy storage from GRACE's TWS anomaly —
and the answer depends on which LSM. This reports a spread across models
(Noah, VIC, CLSM), not a single number: the spread is not an embarrassment,
it's part of the message about how blind these basins are. Do not let this
become the headline — it's the hypothesis, not the finding.

The formula for each model's non-groundwater ("everything GRACE sees that
isn't groundwater") storage follows NASA's own GLDAS-2 README (section 3.5,
"Data Interpretation"), not an independent derivation:

- Noah and VIC don't model groundwater as a separate reservoir, so their
  full soil-moisture-layers + snow + canopy sum stands in for "everything
  but groundwater".
- CLSM does model groundwater explicitly, and NASA's own formula for
  extracting it is GWS = TWS - RootZoneSoilMoisture - SWE -
  CanopyInterception. Note this uses root-zone soil moisture, not profile
  (total-column) soil moisture — using profile here would double-count.
"""

import xarray as xr

_KG_M2_TO_CM = 0.1  # GLDAS reports kg/m^2 (== mm); GRACE trend.py works in cm.


def _wrap_longitude(da: xr.DataArray, dim: str = "lon") -> xr.DataArray:
    """Pad a 0-360 longitude axis periodically, one column at each end.

    Longitude is circular but `interp` is not: GLDAS cell centers span
    0.5..359.5 after the 0-360 conversion, while GRACE's grid runs past
    both (0.25..359.75 at half-degree, 0.125..359.875 at quarter). Without
    padding, every GRACE column outside the GLDAS center range falls
    outside the interpolation domain and comes back NaN -- a dead stripe
    straddling the prime meridian, silently voiding basins in western
    Europe and Africa. Copying the last column to lon-360 and the first to
    lon+360 lets bilinear interpolation cross the seam the way the sphere
    actually does.
    """
    lons = da[dim].values
    left = da.isel({dim: [-1]}).assign_coords({dim: [lons[-1] - 360]})
    right = da.isel({dim: [0]}).assign_coords({dim: [lons[0] + 360]})
    return xr.concat([left, da, right], dim=dim)


def _to_grace_grid(da: xr.DataArray, grace_da: xr.DataArray) -> xr.DataArray:
    """Regrid a GLDAS DataArray (1.0 deg, -180/180 lon) onto GRACE's grid (0.25 deg, 0/360 lon).

    GLDAS and GRACE disagree on both spatial resolution and longitude
    convention -- a naive elementwise op between them would silently
    compare the wrong pixels (and, on lon, mostly nothing at all).
    """
    da = da.assign_coords(lon=da["lon"] % 360).sortby("lon")
    return _wrap_longitude(da).interp(lat=grace_da["lat"], lon=grace_da["lon"], method="linear")


def _monthly(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Collapse a DataArray onto one value per calendar month.

    GRACE timestamps land mid-month; GLDAS timestamps land on the 1st --
    direct alignment on raw `time` would intersect to nothing. GRACE also
    occasionally carries two sub-monthly mascon solutions within the same
    calendar month (an orbit/battery-driven split, not a data error), so
    truncating the coordinate alone can produce duplicate labels; group and
    average instead of just reassigning, so those collapse into one value.
    """
    months = da[dim].values.astype("datetime64[M]").astype(da[dim].dtype)
    return da.assign_coords({dim: months}).groupby(dim).mean(dim)


def monthly_mean(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Public form of the calendar-month collapse used internally.

    Exposed so callers plotting GRACE alongside this module's output can put
    both on the same monthly axis rather than reimplementing the
    duplicate-solution handling described in `_monthly`.
    """
    return _monthly(da, dim=dim)


def _noah_non_gw_storage(ds: xr.Dataset) -> xr.DataArray:
    soil = (
        ds["SoilMoi0_10cm_inst"]
        + ds["SoilMoi10_40cm_inst"]
        + ds["SoilMoi40_100cm_inst"]
        + ds["SoilMoi100_200cm_inst"]
    )
    return (soil + ds["SWE_inst"] + ds["CanopInt_inst"]) * _KG_M2_TO_CM


def _vic_non_gw_storage(ds: xr.Dataset) -> xr.DataArray:
    soil = ds["SoilMoi0_30cm_inst"] + ds["SoilMoi_depth2_inst"] + ds["SoilMoi_depth3_inst"]
    return (soil + ds["SWE_inst"] + ds["CanopInt_inst"]) * _KG_M2_TO_CM


def _clsm_non_gw_storage(ds: xr.Dataset) -> xr.DataArray:
    return (ds["SoilMoist_RZ_inst"] + ds["SWE_inst"] + ds["CanopInt_inst"]) * _KG_M2_TO_CM


_NON_GW_STORAGE = {
    "noah": _noah_non_gw_storage,
    "vic": _vic_non_gw_storage,
    "clsm": _clsm_non_gw_storage,
}


def groundwater_storage_anomaly(
    grace_tws: xr.DataArray, lsm_ds: xr.Dataset, model: str, dim: str = "time"
) -> xr.DataArray:
    """GRACE-minus-LSM groundwater storage anomaly estimate for one model.

    `grace_tws` (cm) is regridded/time-aligned against automatically;
    `lsm_ds` may be on GLDAS's native 1.0 deg / -180-180 lon grid and monthly
    dates. Returns an anomaly, in the same units as `grace_tws`, relative to
    the mean over the months the two records share.

    Both baselines are removed *after* restricting to the common months, not
    before. GRACE has the 2017-06..2018-05 GRACE/GRACE-FO gap plus scattered
    missing months; GLDAS has none. De-meaning each record over its own time
    axis therefore references two different sets of months, and the
    difference of those two baselines survives the subtraction as a constant
    offset in the residual -- an offset with no physical meaning that the
    inner-join alignment of the final `-` would otherwise hide. Passing an
    already-de-meaned series is harmless: re-centering on the common period
    is what the fix is.
    """
    if model not in _NON_GW_STORAGE:
        raise ValueError(f"Unknown GLDAS model {model!r}, expected one of {tuple(_NON_GW_STORAGE)}")
    non_gw_storage = _monthly(_to_grace_grid(_NON_GW_STORAGE[model](lsm_ds), grace_tws), dim=dim)
    grace_common, non_gw_common = xr.align(_monthly(grace_tws, dim=dim), non_gw_storage, join="inner")

    grace_anomaly = grace_common - grace_common.mean(dim=dim)
    non_gw_anomaly = non_gw_common - non_gw_common.mean(dim=dim)
    return grace_anomaly - non_gw_anomaly


def ensemble_attribution(grace_tws: xr.DataArray, lsm_datasets: dict, dim: str = "time") -> xr.Dataset:
    """Per-model groundwater storage anomaly ensemble and its spread.

    `lsm_datasets` maps model name ("noah", "vic", "clsm") to its loaded
    GLDAS Dataset. Returns a Dataset with one variable per model plus
    `ensemble_mean`, `ensemble_min`, `ensemble_max`, and `ensemble_spread`
    (max - min) — the spread is the flagged secondary layer, not a single
    attributed number.

    Models are aligned to their mutually common months before stacking. Each
    model's usable period is set by its own GLDAS coverage, so a bare
    `concat` would take the union and pad the short models with NaN, which
    then propagates into `ensemble_min`/`max` and inflates `ensemble_spread`
    on exactly the months where the ensemble is thinnest -- the spread is a
    headline uncertainty layer here, so it must not be an artifact of
    coverage.
    """
    per_model = dict(
        zip(
            lsm_datasets,
            xr.align(
                *(
                    groundwater_storage_anomaly(grace_tws, ds, model, dim=dim)
                    for model, ds in lsm_datasets.items()
                ),
                join="inner",
            ),
        )
    )
    stacked = xr.concat(list(per_model.values()), dim="model")

    result = xr.Dataset(per_model)
    result["ensemble_mean"] = stacked.mean(dim="model")
    result["ensemble_min"] = stacked.min(dim="model")
    result["ensemble_max"] = stacked.max(dim="model")
    result["ensemble_spread"] = result["ensemble_max"] - result["ensemble_min"]
    return result
