#!/usr/bin/env python3
"""Build the DDW watchlist: join depletion trend and darkness by basin, tier, plot.

Usage:
    uv run scripts/build_watchlist_scatter.py
    uv run scripts/build_watchlist_scatter.py --output data/processed/figures/watchlist_scatter.png

Reuses whatever's already been fetched by build_global_trend_map.py and
build_darkness_axis.py (JPL mascons, HydroBASINS level-4 polygons, and the
processed darkness-by-basin table) rather than re-downloading -- run those
scripts first if their outputs aren't present yet.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from dark_water.common.basins import HYDROBASINS_CONTINENTS, download_hydrobasins, load_basins
from dark_water.depletion_watchlist.depletion import trend, zonal
from dark_water.depletion_watchlist.ingest import grace
from dark_water.depletion_watchlist.product import scatter, tiers


def _load_all_hydrobasins(basins_dir: Path, level: int) -> gpd.GeoDataFrame:
    frames = []
    for continent in HYDROBASINS_CONTINENTS:
        existing = list(basins_dir.glob(f"hybas_{continent}_lev{level:02d}_v1c/*.shp"))
        shp_path = existing[0] if existing else download_hydrobasins(continent, level, basins_dir)
        frames.append(load_basins(shp_path))
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jpl-dir", type=Path, default=Path("data/raw/grace/jpl"))
    parser.add_argument("--basins-dir", type=Path, default=Path("data/raw/basins/hydrobasins"))
    parser.add_argument("--level", type=int, default=4, help="HydroBASINS Pfafstetter level")
    parser.add_argument("--darkness-input", type=Path, default=Path("data/processed/darkness_by_basin.gpkg"))
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    parser.add_argument("--dark-threshold", type=float, default=2 / 3)
    parser.add_argument("--dim-threshold", type=float, default=1 / 3)
    parser.add_argument("--output", type=Path, default=Path("data/processed/figures/watchlist_scatter.png"))
    parser.add_argument("--table-output", type=Path, default=Path("data/processed/watchlist.gpkg"))
    args = parser.parse_args()

    existing = sorted(args.jpl_dir.glob("*.nc"))
    jpl_path = existing[0] if existing else grace.download_jpl_mascons(args.jpl_dir)
    ds = grace.load_mascons(jpl_path)
    land = ds.where(ds["land_mask"] == 1)
    trend_ds = trend.fit_trend(land["lwe_thickness"], alpha=args.alpha)

    print(f"Loading HydroBASINS level {args.level} for all continents...")
    basins = _load_all_hydrobasins(args.basins_dir, args.level)

    print("Aggregating trend to basins...")
    depletion = zonal.aggregate_trend_to_basins(trend_ds, basins, id_column="HYBAS_ID")

    print(f"Loading darkness scores from {args.darkness_input}...")
    darkness = gpd.read_file(args.darkness_input)

    watchlist = tiers.join_depletion_and_darkness(depletion, darkness, id_column="HYBAS_ID")
    watchlist["tier"] = tiers.assign_tiers(
        watchlist, dark_threshold=args.dark_threshold, dim_threshold=args.dim_threshold
    )

    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    watchlist.to_file(args.table_output, driver="GPKG")
    print(f"Saved {args.table_output}")

    for tier in tiers.TIER_ORDER:
        print(f"  {tier}: {(watchlist['tier'] == tier).sum()} basins")

    output_path = scatter.plot_watchlist_scatter(watchlist, args.output)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
