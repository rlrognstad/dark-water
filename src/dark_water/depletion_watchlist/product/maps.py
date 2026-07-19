"""Global TWS trend map: the classic GRACE visualization.

Diverging red/blue choropleth (mass loss vs. gain) with a neutral gray
midpoint, following the dataviz skill's diverging-pair formula rather than
a hand-picked colormap. Pixels that fail the significance test (see
`depletion/trend.py`) are shown at reduced opacity rather than full color,
so the map doesn't imply false precision everywhere.

Supports both themes (`theme="light"` or `"dark"`), each pulling its own
token set from `_THEMES` rather than inverting one palette -- the dark
diverging arms brighten toward their extremes (readable against a dark
chart surface) instead of darkening toward them the way the light-mode
arms do, per the dataviz skill's light/dark surface guidance.
"""

from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

_THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "coastline": "#52514e",
        "gridline": "#e1e0d9",
        # dark red -> red -> neutral gray -> blue -> dark blue: on a light
        # surface, the most extreme values are the darkest, most saturated
        # steps of each arm.
        "cmap_stops": ["#462124", "#e34948", "#f0efec", "#2a78d6", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "coastline": "#c3c2b7",
        "gridline": "#2c2c2a",
        # inverted logic for a dark surface: extremes brighten rather than
        # darken, or they'd disappear into the background. Neutral midpoint
        # is the dataviz skill's dark diverging-pair gray (#383835), not the
        # light midpoint re-used as-is.
        "cmap_stops": ["#ffb3b0", "#e66767", "#383835", "#3987e5", "#bcdcff"],
    },
}

_NOT_SIGNIFICANT_ALPHA = 0.35


def _not_significant_mask(trend_ds: xr.Dataset, alpha: float) -> xr.DataArray:
    """Two-sided significance mask: True where a pixel fails p < alpha.

    Deliberately built from `p_value`, not `trend_ds["significant_decline"]`
    -- that field is one-sided by design (see `trend.py`), so using it here
    would fade out significant *increases* along with genuinely
    insignificant pixels.
    """
    return (trend_ds["p_value"] >= alpha) & trend_ds["trend"].notnull()


def plot_global_trend_map(
    trend_ds: xr.Dataset,
    output_path: Path,
    vlim: float = 3.0,
    alpha: float = 0.05,
    title: str = "TWS trend (cm/yr)",
    theme: str = "light",
) -> Path:
    """Render a global TWS trend map with significance masking.

    `trend_ds` is the output of `depletion.trend.fit_trend` (must have
    `trend` and `p_value` on lat/lon). `vlim` sets the symmetric color scale
    bounds in cm/yr; pixels outside it are clipped, not stretched, so a few
    extreme cells don't wash out the rest of the map.

    Significance masking uses `p_value` directly (two-sided), not
    `significant_decline` -- that field is one-sided by design (see
    `trend.py`), so using it here would fade out significant *increases*
    and mislabel them as statistically uncertain.

    `theme` selects a token set from `_THEMES` ("light" or "dark") -- see
    the module docstring for why dark mode isn't just light mode inverted.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = _THEMES[theme]
    cmap = LinearSegmentedColormap.from_list(f"tws_trend_diverging_{theme}", colors["cmap_stops"])

    fig = plt.figure(figsize=(12, 6))
    # Equal Earth, not PlateCarree: an equirectangular display projection
    # visually inflates polar area, overweighting Greenland/Antarctic ice
    # loss relative to the mid-latitude aquifer depletion this map is
    # actually about. `transform=` below stays PlateCarree -- that's the
    # data's own lat/lon coordinate system, unrelated to the display
    # projection.
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())
    ax.set_global()
    # GeoAxes default to a white background patch regardless of figure
    # facecolor -- without this, ocean/NaN cells (masked out via land_mask
    # upstream) show through as white even in dark theme.
    ax.set_facecolor(colors["surface"])
    ax.coastlines(linewidth=0.5, color=colors["coastline"])

    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    mesh = ax.pcolormesh(
        trend_ds["lon"],
        trend_ds["lat"],
        trend_ds["trend"],
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    overlay = np.where(_not_significant_mask(trend_ds, alpha), 1.0, np.nan)
    ax.pcolormesh(
        trend_ds["lon"],
        trend_ds["lat"],
        overlay,
        transform=ccrs.PlateCarree(),
        cmap=LinearSegmentedColormap.from_list("relief", [colors["surface"], colors["surface"]]),
        alpha=1 - _NOT_SIGNIFICANT_ALPHA,
        shading="auto",
        vmin=0,
        vmax=1,
    )

    # draw_labels=True on cartopy gridlines corrupts matplotlib's tight-bbox
    # calculation on save (cropping the map to just the labels/colorbar) --
    # a known cartopy/matplotlib interaction, not a config choice. Plain
    # gridlines plus coastlines are enough to read the map.
    ax.gridlines(draw_labels=False, linewidth=0.5, color=colors["gridline"], alpha=0.8)

    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
    cbar.set_label(title, color=colors["ink"])
    cbar.ax.xaxis.set_tick_params(color=colors["ink"], labelcolor=colors["ink"])

    ax.set_title(
        f"{title} — faded pixels are not statistically significant (p ≥ {alpha:g})",
        color=colors["ink"],
        fontsize=11,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=colors["surface"])
    plt.close(fig)
    return output_path
