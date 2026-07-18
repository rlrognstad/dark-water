"""TWS trend and significance testing at native mascon resolution.

Fits trend + annual/semi-annual harmonics jointly, rather than a raw linear
fit — GRACE Total Water Storage varies far more within a year (snowpack,
soil moisture, monsoon cycles) than the depletion trend does, so an
un-deseasonalized fit would let the seasonal cycle bias the trend estimate.
Significance uses an effective-sample-size correction for residual lag-1
autocorrelation (Dawdy & Matalas 1964): monthly TWS anomalies are strongly
autocorrelated, so a naive OLS p-value would overstate confidence.
"""

import numpy as np
import xarray as xr
from scipy import stats

_HARMONIC_PERIODS_YEARS = (1.0, 0.5)  # annual, semi-annual
_SECONDS_PER_YEAR = 365.25 * 24 * 3600


def _decimal_years(time: xr.DataArray) -> np.ndarray:
    seconds = time.values.astype("datetime64[s]").astype("float64")
    years = seconds / _SECONDS_PER_YEAR
    return years - years.mean()


def _design_matrix(years: np.ndarray) -> np.ndarray:
    columns = [np.ones_like(years), years]
    for period in _HARMONIC_PERIODS_YEARS:
        omega = 2 * np.pi / period
        columns += [np.cos(omega * years), np.sin(omega * years)]
    return np.stack(columns, axis=1)  # (n_time, n_params); trend is column 1


def fit_trend(da: xr.DataArray, dim: str = "time", alpha: float = 0.05) -> xr.Dataset:
    """Fit a trend + seasonal-harmonics model per pixel and test trend significance.

    `da` must have `dim` as a datetime64 coordinate. Returns a Dataset with
    `trend` (units of `da` per year), `p_value`, and `significant_decline`
    (trend < 0 and p_value < alpha), on every non-`dim` dimension of `da`.
    """
    da = da.transpose(dim, ...)
    other_dims = [d for d in da.dims if d != dim]
    other_shape = [da.sizes[d] for d in other_dims]
    n_time = da.sizes[dim]

    y = da.values.reshape(n_time, -1)
    x = _design_matrix(_decimal_years(da[dim]))
    n_params = x.shape[1]

    coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coeffs

    dof = n_time - n_params
    sigma2 = (residuals**2).sum(axis=0) / dof
    xtx_inv = np.linalg.inv(x.T @ x)
    trend_variance = sigma2 * xtx_inv[1, 1]

    r1_num = (residuals[1:] * residuals[:-1]).sum(axis=0)
    r1_den = (residuals[:-1] ** 2).sum(axis=0)
    r1 = np.divide(r1_num, r1_den, out=np.zeros_like(r1_num), where=r1_den != 0)
    r1 = np.clip(r1, -0.99, 0.99)
    n_eff = np.clip(n_time * (1 - r1) / (1 + r1), 3, n_time)

    trend = coeffs[1]
    se_adjusted = np.sqrt(trend_variance * (n_time / n_eff))
    t_stat = np.divide(trend, se_adjusted, out=np.zeros_like(trend), where=se_adjusted != 0)
    dof_eff = np.clip(n_eff - n_params, 1, None)
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=dof_eff)

    trend = trend.reshape(other_shape)
    p_value = p_value.reshape(other_shape)
    significant_decline = (trend < 0) & (p_value < alpha)

    coords = {d: da.coords[d] for d in other_dims if d in da.coords}
    return xr.Dataset(
        {
            "trend": (other_dims, trend),
            "p_value": (other_dims, p_value),
            "significant_decline": (other_dims, significant_decline),
        },
        coords=coords,
    )
