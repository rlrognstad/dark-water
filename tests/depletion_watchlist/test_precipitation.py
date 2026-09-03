import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dark_water.depletion_watchlist.depletion import precipitation

_LAT = [10.0]
_LON = [100.0]
_MONTHS = pd.date_range("2012-01-01", periods=120, freq="MS")


def _precip_ds(rate_kg_m2_s, times=_MONTHS, name="Rainf_f_tavg"):
    values = np.broadcast_to(np.asarray(rate_kg_m2_s, dtype=float)[:, None, None], (len(times), 1, 1))
    return xr.Dataset(
        {name: (("time", "lat", "lon"), values.copy())},
        coords={"time": times, "lat": _LAT, "lon": _LON},
    )


def _series(values, times=_MONTHS):
    return xr.DataArray(
        np.asarray(values, dtype=float).reshape(len(times), 1, 1),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": _LAT, "lon": _LON},
    )


def test_precipitation_depth_converts_rate_to_monthly_cm():
    # 1 kg/m^2/s for a 31-day January = 86400*31 mm = 267840 cm/10 -> 267840 mm = 26784 cm
    ds = _precip_ds([1.0], times=pd.to_datetime(["2012-01-01"]))

    depth = precipitation.precipitation_depth(ds)

    assert np.isclose(depth.values.item(), 86400 * 31 * 0.1)


def test_precipitation_depth_uses_each_months_actual_length():
    # 2012 is a leap year: January has 31 days, February 29.
    ds = _precip_ds([1.0, 1.0], times=pd.to_datetime(["2012-01-01", "2012-02-01"]))

    depth = precipitation.precipitation_depth(ds).values.ravel()

    assert np.isclose(depth[0] / depth[1], 31 / 29)


def test_precipitation_depth_falls_back_to_rain_plus_snow():
    times = pd.to_datetime(["2012-01-01"])
    ds = xr.Dataset(
        {
            "Rainf_tavg": (("time", "lat", "lon"), np.full((1, 1, 1), 1.0)),
            "Snowf_tavg": (("time", "lat", "lon"), np.full((1, 1, 1), 3.0)),
        },
        coords={"time": times, "lat": _LAT, "lon": _LON},
    )

    depth = precipitation.precipitation_depth(ds)

    assert np.isclose(depth.values.item(), 4 * 86400 * 31 * 0.1)


def test_precipitation_depth_raises_when_no_usable_variable():
    ds = _precip_ds([1.0], times=pd.to_datetime(["2012-01-01"]), name="Rainf_tavg")

    with pytest.raises(KeyError, match="Rainf_tavg"):
        precipitation.precipitation_depth(ds)


def test_cumulative_anomaly_is_centered_and_integrates_deficit():
    # A record that is dry for the first half and wet for the second: the
    # cumulative anomaly must fall then rise, not oscillate with the input.
    depth = _series(np.concatenate([np.full(60, 1.0), np.full(60, 3.0)]))

    cumulative = precipitation.cumulative_anomaly(depth).values.ravel()

    assert np.isclose(cumulative.mean(), 0.0)
    assert cumulative[59] == cumulative.min()
    assert cumulative[-1] == cumulative.max()


def test_drought_driven_decline_is_mostly_explained():
    # Storage tracks the accumulated precipitation deficit exactly, with no
    # independent downward trend of its own.
    rng = np.random.default_rng(0)
    monthly = 2.0 + rng.normal(0, 0.05, len(_MONTHS)) - np.linspace(0, 2.0, len(_MONTHS))
    depth = _series(monthly)
    cumulative = precipitation.cumulative_anomaly(depth)
    gws = cumulative.copy()

    result = precipitation.adjusted_trend(gws, cumulative)

    assert result["total_trend"].values.item() < 0  # it really is declining
    assert np.isclose(result["fraction_unexplained"].values.item(), 0.0, atol=1e-6)
    assert not bool(result["significant_decline"].values.item())


def test_abstraction_like_decline_survives_the_precipitation_control():
    # Precipitation is stationary; storage falls steadily anyway.
    rng = np.random.default_rng(1)
    depth = _series(2.0 + rng.normal(0, 0.05, len(_MONTHS)))
    cumulative = precipitation.cumulative_anomaly(depth)
    gws = _series(np.linspace(0, -20.0, len(_MONTHS)))

    result = precipitation.adjusted_trend(gws, cumulative)

    assert result["fraction_unexplained"].values.item() > 0.9
    assert bool(result["significant_decline"].values.item())
    assert np.isclose(
        result["total_trend"].values.item(),
        result["trend"].values.item() + result["precip_explained_trend"].values.item(),
    )


def test_decline_through_a_wet_period_reports_fraction_above_one():
    # Getting wetter while storage falls is stronger evidence of abstraction
    # than a decline during drought, so the fraction is not clipped at 1.
    wetting = _series(np.linspace(1.0, 3.0, len(_MONTHS)))
    cumulative = precipitation.cumulative_anomaly(wetting)
    gws = _series(np.linspace(0, -20.0, len(_MONTHS)))

    result = precipitation.adjusted_trend(gws, cumulative)

    assert result["fraction_unexplained"].values.item() > 1.0


def test_adjusted_trend_preserves_nan_pixels():
    # Ocean pixels are NaN once the land mask is applied; pinv raises on
    # non-finite input, so they must be routed around rather than solved.
    values = np.zeros((len(_MONTHS), 1, 2))
    values[:, 0, 0] = np.linspace(0, -20.0, len(_MONTHS))
    values[:, 0, 1] = np.nan
    gws = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": _MONTHS, "lat": _LAT, "lon": [100.0, 200.0]},
    )
    depth = xr.DataArray(
        np.full((len(_MONTHS), 1, 2), 2.0),
        dims=("time", "lat", "lon"),
        coords={"time": _MONTHS, "lat": _LAT, "lon": [100.0, 200.0]},
    )
    depth.values[:, 0, 0] += np.linspace(0, 1.0, len(_MONTHS))

    result = precipitation.adjusted_trend(gws, precipitation.cumulative_anomaly(depth))

    assert np.isfinite(result["trend"].values[0, 0])
    assert np.isnan(result["trend"].values[0, 1])
    assert not bool(result["significant_decline"].values[0, 1])


def test_adjusted_trend_aligns_to_shared_months():
    gws = _series(np.linspace(0, -20.0, len(_MONTHS)))
    short = _MONTHS[:100]
    depth = _series(np.full(100, 2.0), times=short)

    result = precipitation.adjusted_trend(gws, precipitation.cumulative_anomaly(depth))

    assert np.isfinite(result["trend"].values.item())
