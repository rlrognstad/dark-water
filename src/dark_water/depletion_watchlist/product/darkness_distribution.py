"""Show why Tier 2 ("Dim & Falling") can come up empty under the v1 darkness proxy.

The GGMN station-density darkness score (see `darkness/density.py`) is a
percentile rank, so on its own it says nothing about the *shape* of the
underlying distribution. In practice it is sharply bimodal: basins with any
registered GGMN stations top out well below the tier thresholds, while the
majority with zero stations are tied at the top of the rank and cluster
near 1.0. A histogram makes that gap visible in a way the scatter (which
only shows tiered, i.e. declining, basins) does not.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

_BAR_COLOR = "#2a78d6"  # categorical slot 1
_BAND_COLOR = "#fab219"  # status: warning, matches Tier 2's scatter color
_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
_MUTED_INK = "#898781"


def plot_darkness_distribution(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    darkness_column: str = "darkness_score",
    dark_threshold: float = 2 / 3,
    dim_threshold: float = 1 / 3,
    bins: int = 50,
    title: str = "Darkness score is bimodal, not continuous",
) -> Path:
    """Histogram of `darkness_column` with the Tier 1/2/3 cut points overlaid.

    Counts use a log y-axis: the zero-station tied majority (darkness near
    1.0) vastly outnumbers the non-zero-station basins, and a linear axis
    would flatten the latter to invisibility. This is a log transform of one
    measure (basin count), not a second axis on a different measure -- not
    the dual-axis pattern the dataviz skill warns against.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = gdf[darkness_column].dropna()
    n_in_band = ((values >= dim_threshold) & (values < dark_threshold)).sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    counts, edges = np.histogram(values, bins=bins, range=(0, 1))
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=_BAR_COLOR, edgecolor="#fcfcfb", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8)

    ax.axvspan(dim_threshold, dark_threshold, color=_BAND_COLOR, alpha=0.15, zorder=0)
    ax.axvline(dim_threshold, color=_MUTED_INK, linestyle="--", linewidth=1)
    ax.axvline(dark_threshold, color=_MUTED_INK, linestyle="--", linewidth=1)
    ax.text(
        (dim_threshold + dark_threshold) / 2,
        ax.get_ylim()[1] * 0.6,
        f"Tier 2 band\n(n={n_in_band})",
        ha="center",
        va="top",
        fontsize=9,
        color=_SECONDARY_INK,
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Darkness score (1 - observability, percentile rank)", color=_INK)
    ax.set_ylabel("Basins (log scale)", color=_INK)
    ax.set_title(title, color=_INK, fontsize=12)
    ax.tick_params(colors=_SECONDARY_INK)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return output_path
