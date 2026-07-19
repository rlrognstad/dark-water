"""Geographic complement to the scatter: where the tiered basins actually are.

The scatter (`scatter.py`) shows the depletion x darkness relationship but
throws away location. This answers the question the scatter can't -- reuses
`maps.py`'s Equal Earth projection and gridline workaround, and the same
tier status colors as the scatter so a tier reads the same color everywhere.
"""

from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from dark_water.depletion_watchlist.product.tiers import TIER_COLORS, TIER_ORDER

_CONTEXT_COLOR = "#e1e0d9"
_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"


def plot_tier_map(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    tier_column: str = "tier",
    title: str = "Dark Depletion Watchlist -- basins by tier",
) -> Path:
    """Render a global choropleth of basins colored by tier.

    Untiered basins (no significant decline) are filled with a light
    neutral context color rather than left blank, so the map reads as "all
    basins we assessed" rather than implying missing data.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())
    ax.set_global()
    ax.coastlines(linewidth=0.5, color="#52514e")

    untiered = gdf[gdf[tier_column].isna()]
    if not untiered.empty:
        untiered.plot(ax=ax, transform=ccrs.PlateCarree(), color=_CONTEXT_COLOR, linewidth=0, zorder=1)

    handles = []
    for tier in TIER_ORDER:
        subset = gdf[gdf[tier_column] == tier]
        if subset.empty:
            continue
        subset.plot(ax=ax, transform=ccrs.PlateCarree(), color=TIER_COLORS[tier], linewidth=0, zorder=2)
        handles.append(Patch(color=TIER_COLORS[tier], label=f"{tier} (n={len(subset)})"))

    # draw_labels=True corrupts matplotlib's tight-bbox calculation on save
    # (see maps.py) -- plain gridlines plus coastlines are enough to read.
    ax.gridlines(draw_labels=False, linewidth=0.5, color="#c3c2b7", alpha=0.6)

    legend = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8, labelcolor=_SECONDARY_INK)
    legend.set_title("Tier (declining basins only)", prop={"size": 8})

    ax.set_title(title, color=_INK, fontsize=12)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return output_path
