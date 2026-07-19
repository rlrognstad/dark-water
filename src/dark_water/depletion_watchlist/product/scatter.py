"""The DDW watchlist scatter: depletion (y) x darkness (x), tiers as color.

Per the concept doc, DDW is presented as a labeled scatter, not a ranked
index -- a scatter invites the reader to see the cluster, a ranked index
invites an unproductive fight about ordinal position. Tier color encodes
*monitoring status* (dark/dim/watched), not water-supply health, so this
uses the dataviz skill's status palette (reserved, never themed) rather
than a categorical series palette or a diverging one -- there's one axis
of severity here (how blind we are to a basin that's falling), not
independent categories or a positive/negative split.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

from dark_water.depletion_watchlist.product.tiers import TIER_1, TIER_2, TIER_3, TIER_ORDER

_TIER_COLORS = {
    TIER_1: "#d03b3b",  # critical
    TIER_2: "#fab219",  # warning
    TIER_3: "#0ca30c",  # good (well-observed, not "healthy aquifer")
}
_CONTEXT_COLOR = "#c3c2b7"
_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"


def plot_watchlist_scatter(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    tier_column: str = "tier",
    x_column: str = "darkness_score",
    y_column: str = "mean_trend",
    label_column: str | None = None,
    label_top_n: int = 10,
    title: str = "Dark Depletion Watchlist",
) -> Path:
    """Render the two-axis watchlist scatter.

    Basins with no tier (not significantly declining) are drawn as small
    muted context points, so the reader can see where the watchlist sits
    relative to the full basin population. Tiered basins are drawn on top,
    colored by status (see `_TIER_COLORS`), with a legend -- tier color is
    never the only channel carrying tier identity, since the legend and
    (for the top `label_top_n` Tier-1 basins, if `label_column` is given)
    direct labels both name it too.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    untiered = gdf[gdf[tier_column].isna()]
    ax.scatter(
        untiered[x_column], untiered[y_column], s=10, color=_CONTEXT_COLOR, alpha=0.5, linewidths=0, zorder=1
    )

    for tier in TIER_ORDER:
        subset = gdf[gdf[tier_column] == tier]
        if subset.empty:
            continue
        ax.scatter(
            subset[x_column],
            subset[y_column],
            s=28,
            color=_TIER_COLORS[tier],
            edgecolors="#fcfcfb",
            linewidths=0.5,
            label=f"{tier} (n={len(subset)})",
            zorder=2,
        )

    if label_column is not None:
        tier_1 = gdf[gdf[tier_column] == TIER_1].nsmallest(label_top_n, y_column)
        for _, row in tier_1.iterrows():
            ax.annotate(
                str(row[label_column]),
                (row[x_column], row[y_column]),
                fontsize=7,
                color=_SECONDARY_INK,
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.axhline(0, color="#e1e0d9", linewidth=0.5, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Darkness (1 - observability, percentile rank)", color=_INK)
    ax.set_ylabel("Mean TWS trend (cm/yr)", color=_INK)
    ax.set_title(title, color=_INK, fontsize=12)
    ax.tick_params(colors=_SECONDARY_INK)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")

    legend = ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=_SECONDARY_INK)
    legend.set_title("Tier (declining basins only)", prop={"size": 8})

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return output_path
