#!/usr/bin/env python3
"""Build the DDW watchlist: join depletion trend and darkness by basin, tier, plot.

Usage:
    uv run scripts/build_watchlist_scatter.py
    uv run scripts/build_watchlist_scatter.py --output data/processed/figures/watchlist_scatter.png

Reuses whatever's already been fetched by build_global_trend_map.py and
build_darkness_axis.py (JPL mascons, HydroBASINS level-4 polygons, and the
processed darkness-by-basin table) rather than re-downloading -- run those
scripts first if their outputs aren't present yet.

Produces, per run: the watchlist table (GeoParquet, every assessed basin),
the depletion x darkness scatter, the darkness-score distribution, a
geographic tier choropleth, and a ranked Tier-1 top-N CSV for dossier work.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from dark_water.common.basins import HYDROBASINS_CONTINENTS, download_hydrobasins, load_basins
from dark_water.depletion_watchlist.depletion import attribution, precipitation, trend, zonal
from dark_water.depletion_watchlist.ingest import grace, lsm
from dark_water.depletion_watchlist.product import darkness_distribution, scatter, tables, tier_map, tiers


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
    parser.add_argument("--darkness-input", type=Path, default=Path("data/processed/darkness_by_basin.parquet"))
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    parser.add_argument("--dark-threshold", type=float, default=2 / 3)
    parser.add_argument("--dim-threshold", type=float, default=1 / 3)
    parser.add_argument("--output", type=Path, default=Path("data/processed/figures/watchlist_scatter.png"))
    parser.add_argument(
        "--distribution-output", type=Path, default=Path("data/processed/figures/darkness_distribution.png")
    )
    parser.add_argument("--map-output", type=Path, default=Path("data/processed/figures/watchlist_tier_map.png"))
    parser.add_argument("--map-theme", choices=["light", "dark"], default="light")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-basins-output", type=Path, default=Path("data/processed/top_tier1_basins.csv"))
    parser.add_argument("--table-output", type=Path, default=Path("data/processed/watchlist.parquet"))
    parser.add_argument(
        "--scale-factor",
        type=Path,
        default=None,
        help="JPL gain-factor ancillary .nc. Omitting it leaves leakage uncorrected and biases trends low.",
    )
    parser.add_argument("--land-mask", type=Path, default=None, help="JPL land-mask ancillary .nc")
    parser.add_argument(
        "--gldas-dir",
        type=Path,
        default=None,
        help="GLDAS dir (any single model) to control the trend for accumulated precipitation.",
    )
    parser.add_argument(
        "--min-unexplained-fraction",
        type=float,
        default=None,
        help="Require this share of a basin's decline to survive the precipitation control before tiering it.",
    )
    args = parser.parse_args()

    if args.min_unexplained_fraction is not None and args.gldas_dir is None:
        parser.error("--min-unexplained-fraction needs --gldas-dir to compute the covariate")

    existing = sorted(args.jpl_dir.glob("*.nc"))
    jpl_path = existing[0] if existing else grace.download_jpl_mascons(args.jpl_dir)
    ds = grace.load_mascons(jpl_path, scale_factor_path=args.scale_factor, land_mask_path=args.land_mask)
    if args.scale_factor is None:
        print("WARNING: no --scale-factor given; mascon leakage uncorrected, trends biased low.")
    land = ds if args.land_mask is not None else ds.where(ds["land_mask"] == 1)
    trend_ds = trend.fit_trend(land["lwe_thickness"], alpha=args.alpha)

    if args.gldas_dir is not None:
        gldas_paths = sorted(args.gldas_dir.glob("**/*.nc4"))
        if not gldas_paths:
            parser.error(f"No GLDAS granules under {args.gldas_dir}")
        print(f"Controlling trend for accumulated precipitation ({len(gldas_paths)} GLDAS granules)...")
        precip = precipitation.precipitation_depth(lsm.load_gldas(gldas_paths))
        cumulative = precipitation.cumulative_anomaly(attribution._to_grace_grid(precip, land["lwe_thickness"]))
        trend_ds = precipitation.adjusted_trend(
            attribution.monthly_mean(land["lwe_thickness"]), cumulative, alpha=args.alpha
        )

    print(f"Loading HydroBASINS level {args.level} for all continents...")
    basins = _load_all_hydrobasins(args.basins_dir, args.level)

    print("Aggregating trend to basins...")
    depletion = zonal.aggregate_trend_to_basins(trend_ds, basins, id_column="HYBAS_ID")

    print(f"Loading darkness scores from {args.darkness_input}...")
    darkness = gpd.read_parquet(args.darkness_input)

    watchlist = tiers.join_depletion_and_darkness(depletion, darkness, id_column="HYBAS_ID")
    watchlist["tier"] = tiers.assign_tiers(
        watchlist,
        dark_threshold=args.dark_threshold,
        dim_threshold=args.dim_threshold,
        min_unexplained_fraction=args.min_unexplained_fraction,
    )

    if "mean_fraction_unexplained" in watchlist:
        declining = watchlist["basin_significant_decline"].astype(bool)
        share = watchlist.loc[declining, "mean_fraction_unexplained"]
        print(
            f"Precipitation control: median {share.median():.2f} of the decline unexplained "
            f"across {int(declining.sum())} declining basins; "
            f"{int((share < 0.5).sum())} are majority precipitation-driven."
        )

    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    watchlist.to_parquet(args.table_output)
    print(f"Saved {args.table_output}")

    for tier in tiers.TIER_ORDER:
        print(f"  {tier}: {(watchlist['tier'] == tier).sum()} basins")

    output_path = scatter.plot_watchlist_scatter(watchlist, args.output)
    print(f"Saved {output_path}")

    distribution_path = darkness_distribution.plot_darkness_distribution(
        watchlist, args.distribution_output, dark_threshold=args.dark_threshold, dim_threshold=args.dim_threshold
    )
    print(f"Saved {distribution_path}")

    map_path = tier_map.plot_tier_map(watchlist, args.map_output, theme=args.map_theme)
    print(f"Saved {map_path}")

    top_basins = tables.rank_top_basins(watchlist, id_column="HYBAS_ID", n=args.top_n)
    top_basins_path = tables.save_table(top_basins, args.top_basins_output)
    print(f"Saved {top_basins_path}")


if __name__ == "__main__":
    main()
