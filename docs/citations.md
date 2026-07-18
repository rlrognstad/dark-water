# Methodology & Dataset Citations

Every citation below was verified against a primary source before being
listed here — either the actual metadata embedded in a downloaded data
file, a bundled technical-documentation PDF, or the publisher/agency's own
citation page. None are reconstructed from memory alone. Where a source
couldn't be verified this way, that's noted explicitly rather than guessed.

---

## Datasets

### GRACE/GRACE-FO mascons — JPL (`ingest/grace.py`)

Verified directly from the `journal_reference` / `CRI_filter_journal_reference`
/ `C_20_substitution` / `C_30_substitution` / `GIA_removed` attributes of the
downloaded file `GRCTellus.JPL.200204_202605.GLO.RL06.3M.MSCNv04CRI.nc`.

- Watkins, M. M., Wiese, D. N., Yuan, D.-N., Boening, C., & Landerer, F. W. (2015). Improved methods for observing Earth's time variable mass distribution with GRACE using spherical cap mascons. *Journal of Geophysical Research: Solid Earth*, 120. https://doi.org/10.1002/2014JB011547
- Wiese, D. N., Landerer, F. W., & Watkins, M. M. (2016). Quantifying and reducing leakage errors in the JPL RL05M GRACE mascon solution. *Water Resources Research*, 52. https://doi.org/10.1002/2016WR019344 (the Coastline Resolution Improvement / CRI filter)
- Loomis, B. D., Rachlin, K. E., & Luthcke, S. B. (2019). Improved Earth oblateness rate reveals increased ice sheet losses and mass-driven sea level rise. *Geophysical Research Letters*, 46, 6910–6917. https://doi.org/10.1029/2019GL082929 (C20/C30 substitution, TN-14)
- Peltier, W. R., Argus, D. F., & Drummond, R. (2018). Comment on the paper by Purcell et al. 2016 entitled "An assessment of ICE-6G_C (VM5a) glacial isostatic adjustment model." *Journal of Geophysical Research: Solid Earth*, 122. (ICE6G-D GIA correction)

### GRACE/GRACE-FO mascons — GSFC (`ingest/grace.py`)

Verified from GSFC's own recommended-citations statement at
https://earth.gsfc.nasa.gov/geo/data/grace-mascons.

- Loomis, B. D., Luthcke, S. B., & Sabaka, T. J. (2019). Regularization and error characterization of GRACE mascons. *Journal of Geodesy*, 93, 1381–1398. https://doi.org/10.1007/s00190-019-01252-y

### GRACE/GRACE-FO mascons — CSR (`ingest/grace.py`)

Verified from CSR's dataset citation on the Texas Data Repository
(https://doi.org/10.18738/T8/UN91VR).

- Save, H. (2019). *CSR GRACE RL06 Mascon Solutions* [Data set]. Texas Data Repository. https://doi.org/10.18738/T8/UN91VR
- Save, H., Bettadpur, S., & Tapley, B. D. (2016). High resolution CSR GRACE RL05 mascons. *Journal of Geophysical Research: Solid Earth*, 121, 7547–7569. https://doi.org/10.1002/2016JB013007 (underlying method paper, cited alongside the RL06 dataset itself)

### Basin units — HydroBASINS (`common/basins.py`)

Verified from `HydroBASINS_TechDoc_v1c.pdf`, bundled inside every
HydroBASINS download zip.

- Lehner, B., & Grill, G. (2013). Global river hydrography and network routing: baseline data and new approaches to study the world's large river systems. *Hydrological Processes*, 27(15), 2171–2186. https://doi.org/10.1002/hyp.9740

### Basin units — WHYMAP aquifer polygons (`common/basins.py`)

Verified from `metadata_WHYMAP_GWR_v1.0.pdf`, bundled inside the WHYMAP GWR
download zip. Required attribution line, verbatim: "WHYMAP GWR © BGR &
UNESCO 2015."

- Bundesanstalt für Geowissenschaften und Rohstoffe (BGR) & UNESCO (2008). *Groundwater Resources of the World 1:25,000,000*. Hannover, Paris.

### Land surface models — GLDAS-2.1 Noah/VIC/CLSM (`ingest/lsm.py`)

Verified from the GES DISC-hosted `README_GLDAS2.pdf` (section 7.0
Acknowledgements / References), the same document that provided the
"Data Interpretation" formulas `depletion/attribution.py` implements.

- Rodell, M., Houser, P. R., Jambor, U., Gottschalck, J., Mitchell, K., Meng, C.-J., Arsenault, K., Cosgrove, A., Radakovich, J., Bosilovich, M., Entin, J. K., Walker, J. P., Lohmann, D., & Toll, D. (2004). The Global Land Data Assimilation System. *Bulletin of the American Meteorological Society*, 85(3), 381–394.

---

## Methodology

### TWS trend + significance testing (`depletion/trend.py`)

**Trend + annual/semi-annual harmonic regression, jointly fit.** This is
standard practice in the GRACE literature for separating a storage trend
from the (much larger) seasonal cycle before testing significance — see
it applied the same way in, e.g., the Rodell papers cited above. Not
attributed to one single originating paper here; presented as standard
practice rather than pinned to an unverified "first use" citation.

**Effective-sample-size correction for lag-1 residual autocorrelation.**
Verified: the formula implemented (`n_eff = n·(1−r₁)/(1+r₁)`) matches the
one attributed to Dawdy & Matalas across multiple independent secondary
sources describing it.

- Dawdy, D. R., & Matalas, N. C. (1964). Statistical and probability analysis of hydrologic data, Part III: Analysis of variance, covariance and time series. In V. T. Chow (Ed.), *Handbook of Applied Hydrology*. McGraw-Hill.

### Groundwater-attribution layer, GRACE-minus-LSM (`depletion/attribution.py`)

**General method** — subtracting independently-estimated non-groundwater
storage components (soil moisture, snow, canopy) from GRACE's TWS anomaly
to isolate a groundwater signal:

- Rodell, M., & Famiglietti, J. S. (2002). The potential for satellite-based monitoring of groundwater storage changes using GRACE: The High Plains aquifer, Central US. *Journal of Hydrology*, 263(1–4), 245–256. https://doi.org/10.1016/S0022-1694(02)00060-4
- Rodell, M., Velicogna, I., & Famiglietti, J. S. (2009). Satellite-based estimates of groundwater depletion in India. *Nature*, 460, 999–1002. https://doi.org/10.1038/nature08238
- Rodell, M., Famiglietti, J. S., Wiese, D. N., Reager, J. T., Beaudoing, H. K., Landerer, F. W., & Lo, M.-H. (2018). Emerging trends in global freshwater availability. *Nature*, 557(7707), 651–659. https://doi.org/10.1038/s41586-018-0123-1 (the closest existing analogue to DDW's own thesis: a systematic, global GRACE-based search for depleting/recharging basins)

**CLSM-specific formula** (`GWS = TWS − RootZoneSoilMoisture − SWE − CanopyInterception`,
using root-zone rather than profile soil moisture) — verified directly
from the GLDAS-2 README's own "Data Interpretation" section, itself
citing:

- Li, B., Rodell, M., Kumar, S., Beaudoing, H. K., Getirana, A., Zaitchik, B. F., et al. (2019). Global GRACE data assimilation for groundwater and drought monitoring: Advances and challenges. *Water Resources Research*, 55, 7564–7586. https://doi.org/10.1029/2018WR024618

---

## Not yet verified / open

- **WHYMAP's precise field semantics.** `common/basins.py` and
  `depletion/zonal.py` don't currently key off any WHYMAP attribute field
  (they use a synthetic id from `.reset_index()`), so this isn't a live
  bug — but for whoever builds product/tiering later: `HYGEO2` is
  officially "classification of groundwater recharge" (11–15 major
  groundwater basins, 22–25 complex hydrogeological structures, 33–34
  local/shallow aquifers), per the bundled metadata PDF, not a simple
  aquifer-type label. Worth knowing before using it to distinguish basin
  scales for tiering.
- **GGMN citation** for the eventual darkness-axis / station-density work —
  not yet pulled, since that module hasn't started.
- **InSAR subsidence citations** (Iran, Mexico, Tulare Basin studies
  mentioned in the concept doc) — not yet verified, since InSAR is
  explicitly deferred past v1 in the pilot scope.
