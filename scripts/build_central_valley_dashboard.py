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
    parser.add_argument(
        "--scale-factor",
        type=Path,
        default=None,
        help="JPL gain-factor ancillary .nc. Omitting it leaves leakage uncorrected and biases amplitudes low.",
    )
    parser.add_argument("--land-mask", type=Path, default=None, help="JPL land-mask ancillary .nc")
    parser.add_argument("--output", type=Path, default=Path("data/processed/figures/central_valley_case_study.png"))
    parser.add_argument("--lat", type=float, default=37.0)
    parser.add_argument("--lon", type=float, default=239.0, help="0-360 convention, matching GRACE's grid")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--basin-name", default="Central Valley, California")
    args = parser.parse_args()

    existing_jpl = sorted(args.jpl_dir.glob("*.nc"))
    jpl_path = existing_jpl[0] if existing_jpl else grace.download_jpl_mascons(args.jpl_dir)
    ds = grace.load_mascons(jpl_path, scale_factor_path=args.scale_factor, land_mask_path=args.land_mask)
    if args.scale_factor is None:
        print("WARNING: no --scale-factor given; mascon leakage uncorrected, amplitudes biased low.")
    grace_tws = ds["lwe_thickness"].sel(time=slice(args.start, args.end))

    models = {}
    for name in ["noah", "vic", "clsm"]:
        model_dir = args.gldas_dir / name
        existing = sorted(model_dir.glob("*.nc4")) if model_dir.exists() else []
        paths = existing if existing else lsm.download_gldas(name, model_dir, temporal=(args.start, args.end))
        models[name] = lsm.load_gldas(paths)

    ensemble = attribution.ensemble_attribution(grace_tws, models)

    # Center the plotted TWS hydrograph on the same months the ensemble was
    # centered on. The two curves share an axis, so a TWS baseline taken over
    # the full GRACE record and a GWS baseline taken over the GRACE-GLDAS
    # overlap would offset one against the other purely by construction.
    grace_monthly = attribution.monthly_mean(grace_tws).sel(time=ensemble["time"])
    point_tws = (grace_monthly - grace_monthly.mean(dim="time")).sel(lat=args.lat, lon=args.lon, method="nearest")
    point_ensemble = ensemble.sel(lat=args.lat, lon=args.lon, method="nearest")

    output_path = dashboards.plot_basin_case_study(point_tws, point_ensemble, args.output, basin_name=args.basin_name)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
