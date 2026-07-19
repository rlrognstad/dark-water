# Dark Water

A product suite for finding and tracking water observability gaps.

## Modules

- **`depletion_watchlist`** — the Dark Depletion Watchlist (DDW): crosses satellite-observed water
  storage decline (GRACE/GRACE-FO) against groundwater monitoring darkness (`1 − O_GW`) to surface
  aquifers being drawn down beyond any local capacity to verify it.
  Concept doc: [scratch/dark-depletion-watchlist.md](scratch/dark-depletion-watchlist.md) (not tracked in git).
  - `ingest/` — data acquisition (GRACE mascons, basin units, groundwater station registries, LSM output, InSAR)
  - `depletion/` — TWS trend computation, significance testing, and groundwater attribution
  - `darkness/` — the darkness axis: v1 is a GGMN station-density proxy for `O_GW`, pending WOI
  - `product/` — tiering, maps, dashboards, and dossier generation
- **`dark_basins`** — net observability decay engine: detects station cessation, checks for
  replacement, and aggregates to a basin-level net decay rate. Feeds `depletion_watchlist`'s movement
  alerts and de-/re-darkening tracking.
- **`common`** — shared infrastructure (basin units, etc.) used across modules.
- more modules TBD.

## Layout

- `src/dark_water/` — the package, organized as above
- `data/{raw,interim,processed}/` — gitignored except for `.gitkeep`

## Setup

```bash
uv sync
uv run pytest
```
