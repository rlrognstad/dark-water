# Runbook

How to set up credentials and run the ingestion/computation pipeline that exists so far.

## Setup

```bash
uv sync
uv run pytest
```

## Credentials

### NASA Earthdata (JPL/GSFC GRACE mascons, GLDAS)

Create `~/.netrc`:

```
machine urs.earthdata.nasa.gov
    login <your-earthdata-username>
    password <your-earthdata-password>
```

```bash
chmod 600 ~/.netrc
```

`earthaccess.login()` (called internally by `ingest/grace.py` and
`ingest/lsm.py`) reads this file automatically. If MFA is enabled on the
account, plain username/password `.netrc` auth will fail with
`invalid_credentials` regardless of formatting — use a non-MFA Earthdata
account, or an app-specific credential if Earthdata supports one.

### GES DISC EULA (required for GLDAS only)

GLDAS (`ingest/lsm.py`) is hosted by GES DISC, which gates its data behind
its own EULA — separate from the base Earthdata login above. Without
accepting it, downloads fail with `EulaNotAccepted`, not an auth error.

1. Log into https://urs.earthdata.nasa.gov
2. Profile → **Applications** → **Authorized Apps**
3. Approve **"NASA GESDISC DATA ARCHIVE"**

JPL and GSFC GRACE mascons and HydroBASINS/WHYMAP basin polygons need no
EULA beyond the base Earthdata login (GSFC's mascon grid isn't Earthdata-
gated at all — see below).

## Data sources implemented

| Source | Module | Auth | Notes |
|---|---|---|---|
| JPL GRACE/GRACE-FO mascons | `ingest/grace.py` | Earthdata | Single granule, whole record in one file |
| GSFC GRACE/GRACE-FO mascons | `ingest/grace.py` | None | Direct HTTPS, no login needed |
| CSR GRACE/GRACE-FO mascons | `ingest/grace.py` | None (manual URL) | CSR doesn't serve a stable download URL; get the current link from their request form and pass it to `download_csr_mascons` |
| HydroBASINS (Pfafstetter levels) | `common/basins.py` | None | Per continent + level, e.g. `download_hydrobasins("af", 4, dest)` |
| WHYMAP aquifer polygons | `common/basins.py` | None | One global zip; the function picks out the aquifer-polygon shapefile from ~10 bundled thematic layers |
| GLDAS-2.1 Noah/VIC/CLSM | `ingest/lsm.py` | Earthdata + GES DISC EULA | One granule per model per month — hundreds of files for a multi-year pull |

## Example: pull JPL GRACE + compute basin-level trend

```python
from pathlib import Path
from dark_water.depletion_watchlist.ingest import grace
from dark_water.depletion_watchlist.depletion import trend, zonal
from dark_water.common import basins

path = grace.download_jpl_mascons(Path("data/raw/grace/jpl"))
ds = grace.load_mascons(path)
land = ds.where(ds["land_mask"] == 1)

trend_ds = trend.fit_trend(land["lwe_thickness"])  # per-pixel trend + significance

aquifer_shp = basins.download_whymap_aquifers(Path("data/raw/basins/whymap"))
aquifers = basins.load_basins(aquifer_shp).reset_index().rename(columns={"index": "aquifer_id"})

result = zonal.aggregate_trend_to_basins(trend_ds, aquifers, id_column="aquifer_id")
```

`result` is `aquifers` with `mean_trend`, `n_pixels`,
`frac_pixels_significant_decline`, and `basin_significant_decline` columns
added.

## Example: groundwater-attribution ensemble

```python
from pathlib import Path
from dark_water.depletion_watchlist.ingest import grace, lsm
from dark_water.depletion_watchlist.depletion import attribution

models = {
    name: lsm.load_gldas(lsm.download_gldas(name, Path(f"data/raw/gldas/{name}"), temporal=("2012-01-01", "2022-12-31")))
    for name in ["noah", "vic", "clsm"]
}

path = grace.download_jpl_mascons(Path("data/raw/grace/jpl"))
ds = grace.load_mascons(path)
grace_tws = ds["lwe_thickness"].sel(time=slice("2012-01-01", "2022-12-31"))
grace_tws_anomaly = grace_tws - grace_tws.mean(dim="time")

ensemble = attribution.ensemble_attribution(grace_tws_anomaly, models)
```

`ensemble` has one variable per model plus `ensemble_mean`,
`ensemble_min`, `ensemble_max`, and `ensemble_spread` — the spread is the
flagged secondary layer itself, not a single attributed number (see the
module docstring for why).

## Gotchas already hit and fixed

- **Longitude conventions differ across sources.** GRACE mascons use
  0–360°; HydroBASINS/WHYMAP and GLDAS use -180/180°. `zonal.py` and
  `attribution.py` both convert before comparing — if you write new code
  that mixes these sources, check this first.
- **Grid resolution differs.** GLDAS is 1.0°; GRACE is 0.25°.
  `attribution._to_grace_grid` bilinearly interpolates GLDAS onto GRACE's
  grid.
- **GRACE occasionally has two sub-monthly solutions in the same calendar
  month** (an orbit/battery-driven quirk, confirmed at 2012-01 and
  2015-04 in the real JPL series — not a data error). Naively truncating
  timestamps to month precision produces duplicate time labels and breaks
  alignment; `attribution._monthly` groups and averages instead.
- **`open_mfdataset` requires `dask`** to combine GLDAS's one-file-per-
  month granules; it's in `pyproject.toml` for this reason.
- **Trend must be fit jointly with annual/semi-annual harmonics**, not as
  a raw linear regression — TWS varies far more seasonally than the
  depletion trend does. `trend.py`'s significance test also corrects for
  lag-1 residual autocorrelation, since monthly TWS series are strongly
  autocorrelated and a naive OLS p-value overstates confidence.

## Example: how much of a decline is just weather?

The tiers say "... & Falling" and the product describes basins "drawn down
beyond any local capacity to verify it" — abstraction language. A
significant negative trend alone does not support that: the Central Valley
validation below spans the 2012–2016 and 2020–2022 droughts. Refit the
trend with accumulated precipitation as a covariate and report what
survives.

```python
from dark_water.depletion_watchlist.depletion import attribution, precipitation

precip = precipitation.precipitation_depth(models["noah"])          # cm/month
on_grace = attribution._to_grace_grid(precip, land["lwe_thickness"])
cumulative = precipitation.cumulative_anomaly(on_grace)

result = precipitation.adjusted_trend(attribution.monthly_mean(land["lwe_thickness"]), cumulative)
```

`result["fraction_unexplained"]` is the share of the trend that the weather
does not account for. It is not clipped at 1 — above 1 means the basin lost
storage *through* a wet period, which is stronger evidence of abstraction
than a decline during drought.

`Rainf_f_tavg` already ships inside the GLDAS granules the attribution
ensemble downloads, so this costs no extra data. Any one model's forcing
will do; GLDAS-2.1 shares its precipitation forcing across Noah/VIC/CLSM.

In the pipeline:

```bash
uv run scripts/build_watchlist_scatter.py --gldas-dir data/raw/gldas/noah
uv run scripts/build_watchlist_scatter.py --gldas-dir data/raw/gldas/noah --min-unexplained-fraction 0.5
```

The first reports the covariate and leaves tiers alone. The second also
refuses to tier a basin whose decline precipitation mostly explains — a
deliberate editorial choice, off by default so adding the covariate never
silently reassigns tiers.

## Mascon gain factor (do not skip)

`grace.load_mascons` takes optional `scale_factor_path` and
`land_mask_path`. The gain grid is a separate ancillary file in the mascon
collection; without it, leakage from the gravity inversion damps
basin-scale amplitudes and every trend comes out biased low. It is not
fetched automatically because the correct file is release-specific and
pairing the wrong release's grid to a granule fails silently. Both scripts
warn when it is omitted.

## Verification

Central Valley, California (a known groundwater crisis basin) is the
running sanity check — the pilot's own validation gate is that known
crises should light up. As of this writing:

- `trend.fit_trend` on JPL data: significant decline, mean trend ≈ -1 to
  -1.3 cm/yr across overlapping WHYMAP aquifer polygons.
- `attribution.ensemble_attribution` (2012–2022): ensemble-mean
  groundwater storage anomaly falls from about +14 cm (early 2012) to
  about -11 cm (late 2022), with Noah/VIC/CLSM agreeing on direction and
  showing a multi-cm spread — consistent with the 2012-2016 and 2020-2022
  California droughts.
