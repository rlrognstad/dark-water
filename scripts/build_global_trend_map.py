#!/usr/bin/env python3
"""Build the global TWS trend map from JPL GRACE/GRACE-FO mascons.

Usage:
    uv run scripts/build_global_trend_map.py
    uv run scripts/build_global_trend_map.py --output data/processed/figures/global_tws_trend.png --vlim 3.0

Downloads the JPL mascon file via NASA Earthdata if it isn't already in
--jpl-dir (see docs/runbook.md for credential setup), fits the trend, and
renders the map.
"""

import argparse
from pathlib import Path

from dark_water.depletion_watchlist.depletion import trend
from dark_water.depletion_watchlist.ingest import grace
from dark_water.depletion_watchlist.product import maps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jpl-dir", type=Path, default=Path("data/raw/grace/jpl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/figures/global_tws_trend.png"))
    parser.add_argument("--vlim", type=float, default=3.0, help="Symmetric color scale bound, cm/yr")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    args = parser.parse_args()

    existing = sorted(args.jpl_dir.glob("*.nc"))
    jpl_path = existing[0] if existing else grace.download_jpl_mascons(args.jpl_dir)

    ds = grace.load_mascons(jpl_path)
    land = ds.where(ds["land_mask"] == 1)
    trend_ds = trend.fit_trend(land["lwe_thickness"])

    output_path = maps.plot_global_trend_map(trend_ds, args.output, vlim=args.vlim, alpha=args.alpha, theme=args.theme)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
