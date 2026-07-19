"""Ranked top-N table: the dossier-writer's starting point.

HydroBASINS level-4 units have no human-readable name, so this reports
`id_column` (e.g. `HYBAS_ID`) rather than inventing one -- pairing the ID
with a place name is exactly the kind of judgment call the dossier work
(concept doc, "What ships") does by hand, not something to fake here.
"""

from pathlib import Path

import pandas as pd

from dark_water.depletion_watchlist.product.tiers import TIER_1

_DEFAULT_COLUMNS = ["tier", "mean_trend", "darkness_score", "n_stations", "frac_pixels_significant_decline"]


def rank_top_basins(
    watchlist: pd.DataFrame,
    id_column: str,
    n: int = 10,
    tier: str | None = TIER_1,
    sort_column: str = "mean_trend",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return the `n` most severe basins, sorted by `sort_column` ascending.

    Ascending because `mean_trend` (cm/yr) is negative for depleting basins
    -- the most negative values are the most severe. Filters to `tier` by
    default (Tier 1, the headline); pass `tier=None` to rank across every
    tiered (i.e. significantly declining) basin regardless of tier.
    """
    tiered = watchlist[watchlist["tier"].notna()]
    subset = tiered[tiered["tier"] == tier] if tier is not None else tiered
    columns = columns or _DEFAULT_COLUMNS
    return subset.nsmallest(n, sort_column)[[id_column, *columns]].reset_index(drop=True)


def save_table(table: pd.DataFrame, output_path: Path) -> Path:
    """Write a ranked table to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return output_path
