import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dark_water.depletion_watchlist.depletion import trend


def _ar1_noise(n, phi, sigma, rng):
    noise = np.empty(n)
    noise[0] = rng.normal(scale=sigma)
    innovation_scale = sigma * np.sqrt(1 - phi**2)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal(scale=innovation_scale)
    return noise


def _synthetic_series(rng, true_trend, n_years=20, seasonal_amplitude=4.0, noise_sigma=1.0, phi=0.7):
    time = pd.date_range("2002-04-01", periods=n_years * 12, freq="MS")
    years = np.arange(len(time)) / 12.0
    years = years - years.mean()
    seasonal = seasonal_amplitude * np.cos(2 * np.pi * years)
    noise = _ar1_noise(len(time), phi, noise_sigma, rng)
    y = true_trend * years + seasonal + noise
    return time, y


def test_fit_trend_recovers_known_slope_and_flags_significant_decline():
    rng = np.random.default_rng(0)
    time, y = _synthetic_series(rng, true_trend=-2.0)
    da = xr.DataArray(y.reshape(-1, 1, 1), dims=("time", "lat", "lon"), coords={"time": time})

    result = trend.fit_trend(da)

    assert result["trend"].values[0, 0] == pytest.approx(-2.0, abs=0.5)
    assert result["p_value"].values[0, 0] < 0.05
    assert bool(result["significant_decline"].values[0, 0]) is True


def test_fit_trend_does_not_flag_flat_series_as_declining():
    rng = np.random.default_rng(1)
    time, y = _synthetic_series(rng, true_trend=0.0)
    da = xr.DataArray(y.reshape(-1, 1, 1), dims=("time", "lat", "lon"), coords={"time": time})

    result = trend.fit_trend(da)

    assert bool(result["significant_decline"].values[0, 0]) is False


def test_fit_trend_does_not_flag_positive_trend_as_declining():
    rng = np.random.default_rng(2)
    time, y = _synthetic_series(rng, true_trend=3.0)
    da = xr.DataArray(y.reshape(-1, 1, 1), dims=("time", "lat", "lon"), coords={"time": time})

    result = trend.fit_trend(da)

    assert result["trend"].values[0, 0] > 0
    assert bool(result["significant_decline"].values[0, 0]) is False


def test_fit_trend_is_vectorized_across_pixels():
    rng = np.random.default_rng(3)
    time, y_decline = _synthetic_series(rng, true_trend=-2.0)
    _, y_flat = _synthetic_series(rng, true_trend=0.0)
    stacked = np.stack([y_decline, y_flat], axis=1).reshape(-1, 1, 2)
    da = xr.DataArray(stacked, dims=("time", "lat", "lon"), coords={"time": time})

    result = trend.fit_trend(da)

    assert result["trend"].shape == (1, 2)
    assert bool(result["significant_decline"].values[0, 0]) is True
    assert bool(result["significant_decline"].values[0, 1]) is False
