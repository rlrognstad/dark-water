"""Groundwater station registry ingestion, via the `hydrostations` package.

`hydrostations` (a sibling project, depended on as a local path dependency
-- see pyproject.toml) already implements real network adapters (GGMN,
NWIS, WISE, BoM, ...); this module doesn't re-implement station fetching,
just selects the GGMN/groundwater slice DDW's darkness axis needs.

GGMN license note: CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike),
per the adapter's own citation of the dataset's own metadata -- not the
plainer "CC BY" sometimes quoted informally. Relevant if DDW output has any
commercial dimension.
"""

import geopandas as gpd
from hydrostations import get_stations


def download_ggmn_stations() -> gpd.GeoDataFrame:
    """Fetch the full global GGMN groundwater station registry.

    No bounding-box restriction -- GGMN is a genuinely global network, and
    the darkness axis needs global basin coverage, not the handful of named
    basins `hydrostations.basins` pre-declares bboxes for.
    """
    return get_stations(network="ggmn", compartment="GW")
