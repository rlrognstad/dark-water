"""Precipitation covariate: how much of a storage decline the weather explains.

Precipitation is a flux, not a storage term, so it has no place in the
GRACE-minus-LSM mass balance in `attribution.py` -- and GLDAS is
precipitation-forced anyway, so it is already implicit in the soil moisture
subtracted there. Adding it to that sum would be a unit error and a
double-count.

It matters for the *claim*. DDW's tiers read "... & Falling" and the
product is described as surfacing aquifers "being drawn down beyond any
local capacity to verify it" -- language about abstraction. But a
significant negative trend is equally consistent with a dry decade: the
runbook's own Central Valley validation spans the 2012-2016 and 2020-2022
California droughts and cites them as corroboration. Without a
precipitation covariate, a drought-driven decline in a poorly monitored
basin earns a Tier 1 label on evidence that does not distinguish it from
over-pumping.

This module fits the trend a second time with accumulated precipitation
anomaly as an extra regressor. The trend that survives that control is the
part the weather does not account for -- the number the tier language
actually rests on.

The regressor is *cumulative* precipitation anomaly, not the monthly rate:
storage integrates flux, so a storage anomaly responds to accumulated
surplus or deficit, not to any single month's rainfall. Regressing a
storage series on an instantaneous rate would mostly measure noise.

No extra download -- `Rainf_f_tavg` already ships inside the GLDAS granules
`ingest/lsm.py` pulls for the attribution ensemble.
"""

import numpy as np
import pandas as pd
import xarray as xr

from dark_water.depletion_watchlist.depletion import trend as trend_module

_KG_M2_TO_CM = 0.1  # kg/m^2 == mm; this module reports cm, matching GRACE.
_SECONDS_PER_DAY = 86400

# GLDAS names the total-precipitation forcing field with an `_f_` tag. The
# rain/snow split (`Rainf_tavg` + `Snowf_tavg`) is the fallback: it sums to
# the same total, but only if both are present -- taking rain alone would
# silently drop snowfall, which is most of the cold-season input in exactly
# the snow-fed basins where the seasonal cycle is largest.
_TOTAL_PRECIP = "Rainf_f_tavg"
_RAIN_AND_SNOW = ("Rainf_tavg", "Snowf_tavg")


def precipitation_depth(lsm_ds: xr.Dataset, dim: str = "time") -> xr.DataArray:
    """Monthly precipitation depth in cm, from a GLDAS monthly Dataset.

    GLDAS reports precipitation as a time-averaged *rate* in kg/m^2/s, so
    each month's depth is that rate times the length of that specific month
    -- February and July do not carry the same multiplier, and using a
    nominal 30-day month would alias a spurious annual cycle straight into
    the regressor meant to control for the real one.
    """
    if _TOTAL_PRECIP in lsm_ds:
        rate = lsm_ds[_TOTAL_PRECIP]
    elif all(name in lsm_ds for name in _RAIN_AND_SNOW):
        rate = lsm_ds[_RAIN_AND_SNOW[0]] + lsm_ds[_RAIN_AND_SNOW[1]]
    else:
        raise KeyError(
            f"GLDAS Dataset has neither {_TOTAL_PRECIP!r} nor both of {_RAIN_AND_SNOW}; got {tuple(lsm_ds.data_vars)}"
        )

    days = xr.DataArray(
        pd.to_datetime(rate[dim].values).days_in_month.to_numpy().astype("float64"),
        dims=(dim,),
        coords={dim: rate[dim]},
    )
    return rate * days * _SECONDS_PER_DAY * _KG_M2_TO_CM


def cumulative_anomaly(depth: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Running sum of precipitation departures from the record mean, in cm.

    The cumulative series is re-centered on its own mean so it enters the
    regression without competing with the intercept.
    """
    anomaly = depth - depth.mean(dim=dim)
    cumulative = anomaly.cumsum(dim=dim).transpose(*depth.dims)
    return cumulative - cumulative.mean(dim=dim)


def _solve_with_covariate(y: np.ndarray, common: np.ndarray, covariate: np.ndarray):
    """Least squares of `y` on `[common | covariate]`, per pixel.

    `common` is (n_time, k) and shared by every pixel; `covariate` is
    (n_time, n_pixels) and differs per pixel, so there is no single design
    matrix to factor once. Materializing a full (n_pixels, n_time, k+1)
    stack would run to gigabytes on a global grid, so this assembles the
    (k+1, k+1) normal-equation blocks instead -- the only per-pixel terms
    are the covariate's inner products.

    Uses `pinv` rather than `solve` because a pixel whose covariate is
    constant (a desert basin with no precipitation variance) makes its
    normal matrix singular, and a batched `solve` raises on the whole array
    if any single pixel is degenerate.
    """
    n_time, n_pixels = covariate.shape
    k = common.shape[1]

    a = common.T @ common
    common_t_cov = common.T @ covariate
    cov_t_cov = (covariate * covariate).sum(axis=0)

    normal = np.empty((n_pixels, k + 1, k + 1))
    normal[:, :k, :k] = a
    normal[:, :k, k] = common_t_cov.T
    normal[:, k, :k] = common_t_cov.T
    normal[:, k, k] = cov_t_cov

    rhs = np.empty((n_pixels, k + 1))
    rhs[:, :k] = (common.T @ y).T
    rhs[:, k] = (covariate * y).sum(axis=0)

    inverse = np.linalg.pinv(normal)
    coeffs = np.einsum("pij,pj->pi", inverse, rhs)

    design_cov_term = covariate * coeffs[:, k][None, :]
    fitted = common @ coeffs[:, :k].T + design_cov_term
    return coeffs, y - fitted, inverse


def adjusted_trend(
    gws_anomaly: xr.DataArray,
    cumulative_precip_anomaly: xr.DataArray,
    dim: str = "time",
    alpha: float = 0.05,
) -> xr.Dataset:
    """Refit the storage trend controlling for accumulated precipitation.

    `gws_anomaly` is the groundwater storage anomaly from
    `attribution.groundwater_storage_anomaly` (or its ensemble mean);
    `cumulative_precip_anomaly` comes from `cumulative_anomaly`. Both are
    aligned to their shared months first.

    Returns a Dataset carrying, per pixel:

    - `trend` -- the precipitation-adjusted trend (cm/yr). Named `trend` so
      it drops straight into `zonal.aggregate_trend_to_basins`.
    - `p_value`, `significant_decline` -- as in `trend.fit_trend`, using the
      same lag-1 autocorrelation correction.
    - `total_trend` -- the unadjusted trend, for comparison.
    - `precip_explained_trend` -- total minus adjusted.
    - `fraction_unexplained` -- adjusted / total.

    `fraction_unexplained` is deliberately not clipped to [0, 1]. Values
    above 1 are real and informative: they mean the basin lost storage
    *through* a wet period, which is stronger evidence of abstraction than
    a decline during drought, not a computational artifact. It is NaN where
    the total trend is zero, since the ratio has no meaning there.
    """
    gws, precip = xr.align(gws_anomaly, cumulative_precip_anomaly, join="inner")
    gws = gws.transpose(dim, ...)
    precip = precip.transpose(dim, ...).broadcast_like(gws)

    other_dims = [d for d in gws.dims if d != dim]
    other_shape = [gws.sizes[d] for d in other_dims]
    n_time = gws.sizes[dim]

    y = gws.values.reshape(n_time, -1)
    covariate = precip.values.reshape(n_time, -1)
    years = trend_module._decimal_years(gws[dim])
    common = trend_module._design_matrix(years)
    n_params = common.shape[1] + 1

    # pinv raises on non-finite input, and ocean pixels are NaN by design
    # once the land mask is applied. Solve on the finite columns only and
    # write NaN back everywhere else.
    valid = np.isfinite(y).all(axis=0) & np.isfinite(covariate).all(axis=0)
    n_pixels = y.shape[1]
    adjusted_flat = np.full(n_pixels, np.nan)
    p_value_flat = np.full(n_pixels, np.nan)

    if valid.any():
        coeffs, residuals, inverse = _solve_with_covariate(y[:, valid], common, covariate[:, valid])
        trend_valid = coeffs[:, 1]

        dof = max(n_time - n_params, 1)
        sigma2 = (residuals**2).sum(axis=0) / dof
        trend_variance = sigma2 * inverse[:, 1, 1]

        adjusted_flat[valid] = trend_valid
        p_value_flat[valid] = trend_module.trend_p_value(
            residuals, trend_valid, trend_variance, n_time, n_params
        )

    total = trend_module.fit_trend(gws, dim=dim, alpha=alpha)["trend"]

    adjusted = adjusted_flat.reshape(other_shape)
    p_value = p_value_flat.reshape(other_shape)
    total_values = total.values
    fraction = np.divide(
        adjusted,
        total_values,
        out=np.full_like(adjusted, np.nan),
        where=total_values != 0,
    )

    coords = {d: gws.coords[d] for d in other_dims if d in gws.coords}
    return xr.Dataset(
        {
            "trend": (other_dims, adjusted),
            "p_value": (other_dims, p_value),
            "significant_decline": (other_dims, (adjusted < 0) & (p_value < alpha)),
            "total_trend": (other_dims, total_values),
            "precip_explained_trend": (other_dims, total_values - adjusted),
            "fraction_unexplained": (other_dims, fraction),
        },
        coords=coords,
    )
