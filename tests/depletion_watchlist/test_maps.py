import matplotlib
import numpy as np
import xarray as xr
from PIL import Image

matplotlib.use("Agg")

from dark_water.depletion_watchlist.product import maps


def _trend_ds():
    lat = [10.0, 20.0]
    lon = [100.0, 200.0]
    # (significant decline, significant increase, non-significant decline, non-significant increase)
    trend = np.array([[-2.0, 2.0], [-0.1, 0.1]])
    p_value = np.array([[0.01, 0.01], [0.5, 0.5]])
    return xr.Dataset(
        {
            "trend": (("lat", "lon"), trend),
            "p_value": (("lat", "lon"), p_value),
        },
        coords={"lat": lat, "lon": lon},
    )


def test_plot_global_trend_map_saves_a_real_figure_not_a_cropped_one(tmp_path):
    output_path = tmp_path / "trend.png"

    result = maps.plot_global_trend_map(_trend_ds(), output_path)

    assert result == output_path
    assert output_path.exists()
    # A regression guard for the bbox_inches="tight" + gridline-labels bug
    # that cropped the map down to just the colorbar strip.
    width, height = Image.open(output_path).size
    assert width > 400
    assert height > 400


def test_not_significant_mask_does_not_flag_significant_increases():
    # significant_decline (one-sided, from trend.py) would be False for the
    # significant-increase cell -- masking on it would incorrectly fade a
    # genuinely significant pixel and mislabel it as uncertain.
    trend_ds = _trend_ds()

    mask = maps._not_significant_mask(trend_ds, alpha=0.05)

    assert mask.values.tolist() == [[False, False], [True, True]]
