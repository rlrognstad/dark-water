#!/usr/bin/env python3
"""Build the darkness axis (v1 GGMN-density proxy) for a basin set.

Usage:
    uv run scripts/build_darkness_axis.py
    uv run scripts/build_darkness_axis.py --output data/processed/darkness_by_basin.parquet

Fetches the global GGMN groundwater station registry (via hydrostations)
and HydroBASINS level-4 polygons (downloading whatever continents aren't
already in --basins-dir), then computes station density, an observability
score, and darkness (1 - observability) per basin.
"""

import argparse
from pathlib import Path

from dark_water.common.basins import HYDROBASINS_CONTINENTS, download_hydrobasins, load_basins
from dark_water.depletion_watchlist.darkness import density
from dark_water.depletion_watchlist.ingest.stations import download_ggmn_stations

import geopandas as gpd
import pandas as pd


def _load_all_hydrobasins(basins_dir: Path, level: int) -> gpd.GeoDataFrame:
    frames = []
    for continent in HYDROBASINS_CONTINENTS:
        existing = list(basins_dir.glob(f"hybas_{continent}_lev{level:02d}_v1c/*.shp"))
        shp_path = existing[0] if existing else download_hydrobasins(continent, level, basins_dir)
        frames.append(load_basins(shp_path))
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basins-dir", type=Path, default=Path("data/raw/basins/hydrobasins"))
    parser.add_argument("--level", type=int, default=4, help="HydroBASINS Pfafstetter level")
    parser.add_argument("--output", type=Path, default=Path("data/processed/darkness_by_basin.parquet"))
    parser.add_argument("--max-inactive-years", type=float, default=5.0)
    args = parser.parse_args()

    print("Fetching GGMN stations (global, paginated -- can take several minutes)...")
    stations = download_ggmn_stations()
    print(f"  {len(stations)} GGMN stations")

    print(f"Loading HydroBASINS level {args.level} for all continents...")
    basins = _load_all_hydrobasins(args.basins_dir, args.level)
    print(f"  {len(basins)} basins")

    result = density.station_density_by_basin(
        stations, basins, id_column="HYBAS_ID", area_column="SUB_AREA", max_inactive_years=args.max_inactive_years
    )
    result["observability_score"] = density.observability_score(result["station_density"])
    result["darkness_score"] = density.darkness_score(result["station_density"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output)
    print(f"Saved {args.output}")
    print(f"Basins with zero GGMN stations: {(result['n_stations'] == 0).sum()} / {len(result)}")


if __name__ == "__main__":
    main()
