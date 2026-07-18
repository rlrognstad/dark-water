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
    grace_tws_anomaly: xr.DataArray, lsm_ds: xr.Dataset, model: str, dim: str = "time"
) -> xr.DataArray:
    """GRACE-minus-LSM groundwater storage anomaly estimate for one model.

    `grace_tws_anomaly` (cm) and `lsm_ds` must share a `dim` (time) and
    spatial grid. Returns an anomaly, in the same units as
    `grace_tws_anomaly`, relative to `lsm_ds`'s own mean over its time range.
    """
    if model not in _NON_GW_STORAGE:
        raise ValueError(f"Unknown GLDAS model {model!r}, expected one of {tuple(_NON_GW_STORAGE)}")
    non_gw_storage = _NON_GW_STORAGE[model](lsm_ds)
    non_gw_anomaly = non_gw_storage - non_gw_storage.mean(dim=dim)
    return grace_tws_anomaly - non_gw_anomaly


def ensemble_attribution(grace_tws_anomaly: xr.DataArray, lsm_datasets: dict, dim: str = "time") -> xr.Dataset:
    """Per-model groundwater storage anomaly ensemble and its spread.

    `lsm_datasets` maps model name ("noah", "vic", "clsm") to its loaded
    GLDAS Dataset. Returns a Dataset with one variable per model plus
    `ensemble_mean`, `ensemble_min`, `ensemble_max`, and `ensemble_spread`
    (max - min) — the spread is the flagged secondary layer, not a single
    attributed number.
    """
    per_model = {
        model: groundwater_storage_anomaly(grace_tws_anomaly, ds, model, dim=dim)
        for model, ds in lsm_datasets.items()
    }
    stacked = xr.concat(list(per_model.values()), dim="model")

    result = xr.Dataset(per_model)
    result["ensemble_mean"] = stacked.mean(dim="model")
    result["ensemble_min"] = stacked.min(dim="model")
    result["ensemble_max"] = stacked.max(dim="model")
    result["ensemble_spread"] = result["ensemble_max"] - result["ensemble_min"]
    return result
