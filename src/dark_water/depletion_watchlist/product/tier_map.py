"""Geographic complement to the scatter: where the tiered basins actually are.

The scatter (`scatter.py`) shows the depletion x darkness relationship but
throws away location. This answers the question the scatter can't -- reuses
`maps.py`'s Equal Earth projection and gridline workaround, and the same
tier status colors as the scatter so a tier reads the same color everywhere.

Supports both themes (`theme="light"` or `"dark"`). Tier colors themselves
don't change -- the dataviz skill's status palette is fixed, verified
against both surfaces already -- only the map chrome (surface, ink,
coastline, gridline, untiered-basin fill) switches per theme.
"""

from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from dark_water.depletion_watchlist.product.tiers import TIER_COLORS, TIER_ORDER

_THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary_ink": "#52514e",
        "coastline": "#52514e",
        "gridline": "#c3c2b7",
        "context": "#e1e0d9",  # untiered (non-declining) basin fill
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary_ink": "#c3c2b7",
        "coastline": "#c3c2b7",
        "gridline": "#2c2c2a",
        "context": "#383835",  # dataviz skill's dark diverging-pair neutral
    },
}


def plot_tier_map(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    tier_column: str = "tier",
    title: str = "Dark Depletion Watchlist -- basins by tier",
    theme: str = "light",
) -> Path:
    """Render a global choropleth of basins colored by tier.

    Untiered basins (no significant decline) are filled with a neutral
    context color rather than left blank, so the map reads as "all basins
    we assessed" rather than implying missing data.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = _THEMES[theme]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())
    ax.set_global()
    # GeoAxes default to a white background patch regardless of figure
    # facecolor (see maps.py) -- without this, any gap in basin coverage
    # (ocean, unassessed land) shows through as white even in dark theme.
    ax.set_facecolor(colors["surface"])
    ax.coastlines(linewidth=0.5, color=colors["coastline"])

    untiered = gdf[gdf[tier_column].isna()]
    if not untiered.empty:
        untiered.plot(ax=ax, transform=ccrs.PlateCarree(), color=colors["context"], linewidth=0, zorder=1)

    handles = []
    for tier in TIER_ORDER:
        subset = gdf[gdf[tier_column] == tier]
        if subset.empty:
            continue
        subset.plot(ax=ax, transform=ccrs.PlateCarree(), color=TIER_COLORS[tier], linewidth=0, zorder=2)
        handles.append(Patch(color=TIER_COLORS[tier], label=f"{tier} (n={len(subset)})"))

    # draw_labels=True corrupts matplotlib's tight-bbox calculation on save
    # (see maps.py) -- plain gridlines plus coastlines are enough to read.
    ax.gridlines(draw_labels=False, linewidth=0.5, color=colors["gridline"], alpha=0.6)

    legend = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8, labelcolor=colors["secondary_ink"])
    legend.set_title("Tier (declining basins only)", prop={"size": 8})

    ax.set_title(title, color=colors["ink"], fontsize=12)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=colors["surface"])
    plt.close(fig)
    return output_path
