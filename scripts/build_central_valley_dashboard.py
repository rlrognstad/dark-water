#!/usr/bin/env python3
"""Build the Central Valley case-study dashboard: TWS hydrograph + GWS attribution ensemble.

Usage:
    uv run scripts/build_central_valley_dashboard.py
    uv run scripts/build_central_valley_dashboard.py --start 2012-01-01 --end 2022-12-31

Downloads JPL GRACE mascons and GLDAS-2.1 Noah/VIC/CLSM (via NASA
Earthdata; GLDAS additionally needs the GES DISC EULA -- see
docs/runbook.md) if not already present, then renders the dashboard for
the single grid cell nearest the given lat/lon.
"""

import argparse
from pathlib import Path

from dark_water.depletion_watchlist.depletion import attribution
from dark_water.depletion_watchlist.ingest import grace, lsm
from dark_water.depletion_watchlist.product import dashboards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jpl-dir", type=Path, default=Path("data/raw/grace/jpl"))
    parser.add_argument("--gldas-dir", type=Path, default=Path("data/raw/gldas"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/figures/central_valley_case_study.png"))
    parser.add_argument("--lat", type=float, default=37.0)
    parser.add_argument("--lon", type=float, default=239.0, help="0-360 convention, matching GRACE's grid")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--basin-name", default="Central Valley, California")
    args = parser.parse_args()

    existing_jpl = sorted(args.jpl_dir.glob("*.nc"))
    jpl_path = existing_jpl[0] if existing_jpl else grace.download_jpl_mascons(args.jpl_dir)
    ds = grace.load_mascons(jpl_path)
    grace_tws = ds["lwe_thickness"].sel(time=slice(args.start, args.end))
    grace_tws_anomaly = grace_tws - grace_tws.mean(dim="time")

    models = {}
    for name in ["noah", "vic", "clsm"]:
        model_dir = args.gldas_dir / name
        existing = sorted(model_dir.glob("*.nc4")) if model_dir.exists() else []
        paths = existing if existing else lsm.download_gldas(name, model_dir, temporal=(args.start, args.end))
        models[name] = lsm.load_gldas(paths)

    ensemble = attribution.ensemble_attribution(grace_tws_anomaly, models)

    point_tws = grace_tws_anomaly.sel(lat=args.lat, lon=args.lon, method="nearest")
    point_ensemble = ensemble.sel(lat=args.lat, lon=args.lon, method="nearest")

    output_path = dashboards.plot_basin_case_study(point_tws, point_ensemble, args.output, basin_name=args.basin_name)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
