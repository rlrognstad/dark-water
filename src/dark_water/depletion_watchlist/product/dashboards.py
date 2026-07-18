"""Single-basin case-study dashboard: TWS hydrograph + attribution ensemble.

Two stacked panels for one basin/point:

1. The raw monthly TWS anomaly (GRACE) with the deseasonalized trend line
   overlaid, so the reader can see the trend against the seasonal noise it
   was fit through, not instead of it.
2. The groundwater-attribution ensemble (see `depletion/attribution.py`):
   the multi-model mean as a line, the Noah/VIC/CLSM range as a band around
   it -- the spread is shown deliberately, not collapsed to one number.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from dark_water.depletion_watchlist.depletion import trend

_TWS_COLOR = "#2a78d6"  # categorical slot 1 (blue)
_TREND_LINE_COLOR = "#52514e"  # secondary ink -- a fitted reference, not a data series
_ENSEMBLE_COLOR = "#4a3aa7"  # categorical slot 5 (violet) -- distinct from panel 1's blue


def plot_basin_case_study(
    grace_tws_anomaly: xr.DataArray,
    attribution_ensemble: xr.Dataset,
    output_path: Path,
    basin_name: str = "Case study basin",
) -> Path:
    """Render the two-panel case-study dashboard for one basin/point.

    `grace_tws_anomaly` is a 1-D (time,) series in cm. `attribution_ensemble`
    is the output of `attribution.ensemble_attribution`, already selected
    down to the same single point/basin (also 1-D over time).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trend_ds = trend.fit_trend(grace_tws_anomaly)
    trend_value = float(trend_ds["trend"])
    is_significant = bool(trend_ds["significant_decline"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, facecolor="#fcfcfb")

    ax1.plot(grace_tws_anomaly["time"], grace_tws_anomaly, color=_TWS_COLOR, linewidth=1.5, label="TWS anomaly")
    ax1.plot(
        trend_ds["time"],
        trend_ds["trend_line"],
        color=_TREND_LINE_COLOR,
        linewidth=1.5,
        linestyle="--",
        label="trend",
    )
    significance_note = "significant decline" if is_significant else "not a significant decline"
    ax1.set_ylabel("TWS anomaly (cm)", color="#0b0b0b")
    ax1.set_title(f"{basin_name} — trend: {trend_value:+.2f} cm/yr ({significance_note})", color="#0b0b0b")
    ax1.legend(loc="best", frameon=False, labelcolor="#0b0b0b")
    ax1.grid(color="#e1e0d9", linewidth=0.5)

    ax2.plot(
        attribution_ensemble["time"],
        attribution_ensemble["ensemble_mean"],
        color=_ENSEMBLE_COLOR,
        linewidth=1.5,
        label="ensemble mean",
    )
    ax2.fill_between(
        attribution_ensemble["time"],
        attribution_ensemble["ensemble_min"],
        attribution_ensemble["ensemble_max"],
        color=_ENSEMBLE_COLOR,
        alpha=0.2,
        label="Noah / VIC / CLSM range",
    )
    ax2.set_ylabel("GWS anomaly (cm)", color="#0b0b0b")
    ax2.set_title(f"{basin_name} — groundwater-attribution ensemble", color="#0b0b0b")
    ax2.legend(loc="best", frameon=False, labelcolor="#0b0b0b")
    ax2.grid(color="#e1e0d9", linewidth=0.5)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return output_path
