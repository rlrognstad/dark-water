import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

matplotlib.use("Agg")

from dark_water.depletion_watchlist.product import dashboards


def _grace_tws_anomaly():
    time = pd.date_range("2010-01-01", periods=60, freq="MS")
    years = np.arange(60) / 12.0
    values = -1.5 * years + 3.0 * np.cos(2 * np.pi * years)
    return xr.DataArray(values, dims="time", coords={"time": time})


def _attribution_ensemble():
    time = pd.date_range("2010-01-01", periods=60, freq="MS")
    years = np.arange(60) / 12.0
    mean = -1.2 * years
    return xr.Dataset(
        {
            "ensemble_mean": ("time", mean),
            "ensemble_min": ("time", mean - 1.0),
            "ensemble_max": ("time", mean + 1.0),
        },
        coords={"time": time},
    )


def test_plot_basin_case_study_saves_a_real_two_panel_figure(tmp_path):
    output_path = tmp_path / "case_study.png"

    result = dashboards.plot_basin_case_study(_grace_tws_anomaly(), _attribution_ensemble(), output_path)

    assert result == output_path
    assert output_path.exists()
    width, height = Image.open(output_path).size
    assert width > 400
    assert height > 400
