# Breathing Earth System Simulator (`BESS-JPL`) Python Implementation

[![CI](https://github.com/JPL-Evapotranspiration-Algorithms/BESS-JPL/actions/workflows/ci.yml/badge.svg)](https://github.com/JPL-Evapotranspiration-Algorithms/BESS-JPL/actions/workflows/ci.yml)

`BESS-JPL` is a NASA Jet Propulsion Laboratory Python implementation of the Breathing Earth System Simulator (BESS) model for estimating coupled Gross Primary Productivity ($\text{GPP}$) and Evapotranspiration ($\text{ET}$) from remote sensing observations. Developed for the ECOSTRESS mission and as a precursor for Surface Biology and Geology (SBG), it integrates atmospheric radiative transfer (`FLiESANN`), two-stream canopy shortwave and longwave radiation, $C_3$/$C_4$ photosynthesis, stomatal conductance, and two-source surface energy balance.

## Contributors

Gregory H. Halverson (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
Lead developer<br>
NASA Jet Propulsion Laboratory 321H

Youngryel Ryu (he/him)<br>
[yryu@snu.ac.kr](mailto:yryu@snu.ac.kr)<br>
BESS algorithm inventor<br>
Seoul National University

Hideki Kobayashi (he/him)<br>
[hkoba@jamstec.go.jp](mailto:hkoba@jamstec.go.jp)<br>
FLiES algorithm inventor<br>
Japan Agency for Marine-Earth Science and Technology

Robert Freepartner (he/him)<br>
[robert.freepartner@jpl.nasa.gov](mailto:robert.freepartner@jpl.nasa.gov)<br>
MATLAB to Python translation<br>
Raytheon

Joshua Fisher (he/him)<br>
[jbfisher@chapman.edu](mailto:jbfisher@chapman.edu)<br>
Concept development and project management<br>
Chapman University

Kerry Cawse-Nicholson (she/her)<br>
[kerry-anne.cawse-nicholson@jpl.nasa.gov](mailto:kerry-anne.cawse-nicholson@jpl.nasa.gov)<br>
Project management<br>
NASA Jet Propulsion Laboratory 329G

Zoe Pierrat (she/her)<br>
[zoe.a.pierrat@jpl.nasa.gov](mailto:zoe.a.pierrat@jpl.nasa.gov)<br>
Algorithm maintenance<br>
NASA Jet Propulsion Laboratory 329G

Claire Villanueva-Weeks (she/her)<br>
[claire.s.villanueva-weeks@jpl.nasa.gov](mailto:claire.s.villanueva-weeks@jpl.nasa.gov)<br>
Code maintenance<br>
NASA Jet Propulsion Laboratory 329G

## Citing This Package

If you use `BESS-JPL` in research, operational pipelines, or publications, please cite both this software implementation and the foundational BESS algorithm paper (Ryu et al., 2011):

```bibtex
@software{halverson_bess_jpl,
  author       = {Halverson, Gregory H. and Ryu, Youngryel and Kobayashi, Hideki and Freepartner, Robert and Fisher, Joshua B. and Cawse-Nicholson, Kerry and Pierrat, Zoe and Villanueva-Weeks, Claire},
  title        = {{BESS-JPL}: Breathing Earth System Simulator Python Implementation},
  publisher    = {NASA Jet Propulsion Laboratory},
  url          = {https://github.com/JPL-Evapotranspiration-Algorithms/BESS-JPL},
  version      = {1.29.0},
  year         = {2026}
}

@article{ryu2011integration,
  author       = {Ryu, Youngryel and Baldocchi, Dennis D. and Kobayashi, Hideki and van Ingen, Catharine and Li, Jie and Black, T. Andrew and Beringer, Jason and van Gorsel, Eva and Knohl, Alexander and Law, Beverly E. and Roupsard, Olivier},
  title        = {Integration of MODIS land and atmosphere products with a coupled-process model to estimate gross primary productivity and evapotranspiration from 1 km to global scales},
  journal      = {Remote Sensing of Environment},
  volume       = {115},
  number       = {8},
  pages        = {1865--1874},
  year         = {2011},
  doi          = {10.1016/j.rse.2011.03.009}
}
```

## Overview & Background

This software package is a Python re-implementation of the Breathing Earth System Simulator (BESS) model developed at NASA's Jet Propulsion Laboratory based on [MATLAB code](https://www.environment.snu.ac.kr/bess-flux) produced by Youngryel Ryu at Seoul National University. Originally formulated to quantify global gross primary productivity ($\text{GPP}$) and evapotranspiration ($\text{ET}$) from 1–5 km, 8-day MODIS products, `BESS-JPL` adapts the coupled-process framework to fine-resolution thermal and optical remote sensing observations.

Specifically, `BESS-JPL` is engineered to process $\text{GPP}$ and $\text{ET}$ at 70 m spatial resolution using land surface temperature ($\text{ST}$) from the ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) mission together with Normalized Difference Vegetation Index ($\text{NDVI}$) and surface albedo from the [Spatial Timeseries for Automated high-Resolution multi-Sensor (STARS)](https://github.com/STARS-Data-Fusion) data fusion system. Within the ECOSTRESS Collection 3 `JET3` ensemble (`ECOv003-L3T-L4T-JET`), `BESS-JPL` serves both as one of four constituent evapotranspiration models (`L3T_JET`) and as the sole provider of $\text{GPP}$ for Water Use Efficiency (`L4T_WUE`).

Developed under a NASA Research Opportunities in Space and Earth Sciences (ROSES) grant as an algorithmic precursor for the Surface Biology and Geology (SBG) mission, the package supports interactive research in Jupyter notebooks as well as automated, air-gapped production pipelines, and is released in accordance with NASA SPD-41 open-science requirements.

## Installation

Install the latest release from PyPI using `pip`:

```bash
pip install BESS-JPL
```

For local development and running the test and verification suites:

```bash
git clone https://github.com/JPL-Evapotranspiration-Algorithms/BESS-JPL.git
cd BESS-JPL
pip install -e ".[dev]"
pytest
```

## Usage Guide

### 1. Gridded Raster Scenes (`BESS_JPL`)

The primary entry point `BESS_JPL()` accepts `rasters.Raster` scenes (or NumPy arrays paired with a `RasterGeometry` and `time_UTC`) and automatically retrieves any omitted meteorological (`GEOS5FP`), topographic (`NASADEM`), clumping index (`MODISCI`), canopy height (`gedi-canopy-height`), and climate (`koppengeiger`) inputs:

```python
from datetime import datetime
import rasters as rt
from BESS_JPL import BESS_JPL

# Load 70 m ECOSTRESS / STARS input rasters
ST_C = rt.Raster.open("ECOv002_L2T_LSTE_LST.tif") - 273.15
NDVI = rt.Raster.open("ECOv002_L2T_STARS_NDVI.tif")
albedo = rt.Raster.open("ECOv002_L2T_STARS_albedo.tif")
time_UTC = datetime(2024, 7, 28, 20, 40, 25)

# Run coupled BESS-JPL model
results = BESS_JPL(
    ST_C=ST_C,
    NDVI=NDVI,
    albedo=albedo,
    time_UTC=time_UTC,
    upscale_to_daylight=True
)

# Inspect key gridded outputs
gpp_inst = results["GPP"]            # Instantaneous GPP [μmol CO2 m^-2 s^-1]
gpp_daily = results["GPP_daily"]     # Daily GPP [g C m^-2 day^-1]
le_inst = results["LE_Wm2"]          # Instantaneous latent heat flux [W m^-2]
rn_inst = results["Rn_Wm2"]          # Instantaneous net radiation [W m^-2]
g_inst = results["G_Wm2"]            # Soil heat flux [W m^-2]
et_daylight = results["ET_daylight_kg"]  # Daylight upscaled ET [kg m^-2 / mm]
```

### 2. Tabular Point & Cal-Val Processing (`process_BESS_table`)

Use `process_BESS_table()` to evaluate tower observations or sensitivity tables stored in a `pandas.DataFrame` or `geopandas.GeoDataFrame` containing `time_UTC`, `ST_C`, `NDVI`, `albedo`, and either a `geometry` column or `lat`/`lon` columns:

```python
from BESS_JPL import load_ECOv002_calval_BESS_inputs, process_BESS_table

# Load the ECOSTRESS Collection 2 Cal-Val table (1,065 tower records)
input_df = load_ECOv002_calval_BESS_inputs()

# Run in offline mode using pre-populated table inputs (preserves Cal-Val baseline)
output_df = process_BESS_table(input_df, offline_mode=True)

# Optional: enable directional PAR/NIR spectral albedos in tabular runs
spectral_df = process_BESS_table(
    input_df,
    offline_mode=True,
    use_spectral_albedo=True
)
```

### 3. Air-Gapped / Offline Execution & Verification

In operational Product Generating Executables (PGEs) where network access is disabled, pass `offline_mode=True` to `BESS_JPL()` or `process_BESS_table()`. If any required static or meteorological parameter is missing, `BESS_JPL.MissingOfflineParameter` is raised immediately rather than attempting a remote download.

To verify numerical parity against the frozen ECOSTRESS Cal-Val reference dataset (`rtol=1e-4, atol=1e-7` for `G_Wm2`, `Rn_Wm2`, and `LE_Wm2`):

```python
from BESS_JPL import verify

assert verify(), "BESS-JPL numerical verification failed"
```

## Methodology

`BESS-JPL` couples five physical sub-models across sunlit canopy, shaded canopy, and soil compartments:

1. **Atmospheric Radiative Transfer (`FLiESANN` & Spectral Albedo Scaling):**
   When incoming shortwave components are not supplied, `BESS-JPL` invokes the Forest Light Environmental Simulator artificial neural network emulator (`FLiESANN`) using cloud optical thickness ($\text{COT}$), aerosol optical thickness ($\text{AOT}$), precipitable water vapor, column ozone, surface elevation, solar zenith angle ($\text{SZA}$), and Köppen-Geiger climate class to partition total incoming shortwave radiation ($S_{\text{Win}}$) into direct and diffuse photosynthetically active radiation ($\text{PAR}$, $400\text{--}700\,\text{nm}$), near-infrared radiation ($\text{NIR}$), and ultraviolet radiation ($\text{UV}$). Broadband surface albedo ($\alpha$) is scaled to visible ($\alpha_{\text{PAR}}$) and near-infrared ($\alpha_{\text{NIR}}$) albedos using GEOS-5 FP directional albedo ratios ($\text{ALBVISDR}/\text{ALBEDO}$ and $\text{ALBNIRDR}/\text{ALBEDO}$).

2. **Canopy Structure & Two-Stream Shortwave/Longwave Transfer (`canopy_shortwave_radiation`, `canopy_longwave_radiation`):**
   Leaf Area Index ($\text{LAI}$) is derived from $\text{NDVI}$ via Beer-Lambert inversion ($\text{LAI} = -\ln(1 - \text{fIPAR}) / k_{\text{PAR}}$) and adjusted by MODIS clumping index ($\text{CI}$) to partition the canopy into sunlit and shaded leaf fractions ($f_{\text{sun}}$). Two-stream radiative transfer computes absorbed $\text{PAR}$ ($\text{APAR}_{\text{sunlit}}$, $\text{APAR}_{\text{shade}}$) and absorbed shortwave radiation ($\text{ASW}_{\text{sunlit}}$, $\text{ASW}_{\text{shade}}$, $\text{ASW}_{\text{soil}}$), while ground heat storage is parameterized from soil-absorbed $\text{PAR}$ ($G = 0.28\,\text{APAR}_{\text{soil}}$). Absorbed longwave radiation ($\text{ALW}$) is evaluated for sunlit leaves, shaded leaves, and soil following Wang et al. (2006) using Stefan-Boltzmann emission and clear/all-sky atmospheric emissivity ($\varepsilon_a$).

3. **Canopy Nitrogen Scaling & $\mathrm{C}_3$/$\mathrm{C}_4$ Biochemical Photosynthesis (`calculate_VCmax`, `C3_photosynthesis`, `C4_photosynthesis`):** Leaf-level peak carboxylation capacity at $25\,^\circ\mathrm{C}$ ($V_{\mathrm{cmax},25}$) is scaled across seasonal phenology ($\mathrm{LAI}_{\min}$ to $\mathrm{LAI}_{\max}$) and integrated vertically over sunlit and shaded canopy fractions using beam ($k_b$) and nitrogen ($k_n$) extinction coefficients. Net carbon assimilation ($A_n$) is computed separately for $\mathrm{C}_3$ pathways (Collatz et al., 1991, co-limited by Rubisco $J_C$, electron transport $J_E$, and export capacity $J_S$) and $\mathrm{C}_4$ pathways (Collatz et al., 1992).

4. **Coupled Stomatal Conductance & Two-Source Surface Energy Balance (`canopy_energy_balance`, `soil_energy_balance`, `process_paw_and_gao_LE`):**
   Stomatal conductance ($g_s$) is coupled to net assimilation via the Ball-Berry formulation ($g_s = m\,\text{RH}\,A_n / C_a + b_0\,f_{\text{stress}}$) and solved iteratively for intercellular $\text{CO}_2$ ($C_i$), leaf temperature ($T_f$), and soil temperature ($T_s$). Canopy latent heat flux ($\text{LE}_{\text{canopy}}$) is solved via the quadratic Penman-Monteith formulation of Paw U & Gao (1988), and soil evaporation ($\text{LE}_{\text{soil}}$) is computed via equilibrium evaporation moderated by relative humidity and vapor pressure deficit ($\text{RH}^{\text{VPD}/1000}$).

5. **$C_3$/$C_4$ Blending & Daylight Upscaling (`interpolate_C3_C4`, `daylight-evapotranspiration`):**
   Total ecosystem fluxes ($\text{GPP}$, $\text{LE}$, $\text{LE}_{\text{canopy}}$, $\text{LE}_{\text{soil}}$, $R_n$, $R_{n,\text{canopy}}$, $R_{n,\text{soil}}$) are linearly interpolated between $C_3$ and $C_4$ solutions using the local $C_4$ vegetation fraction. Instantaneous $\text{GPP}$ is upscaled to daily $\text{GPP}_{\text{daily}}$ ($\text{g C m}^{-2}\,\text{day}^{-1}$) via the solar radiation ratio $\text{SF}_d$ (Ryu et al., 2008), and instantaneous $\text{LE}$ is upscaled to daylight evapotranspiration ($\text{ET}_{\text{daylight}}$) using `daylight-evapotranspiration`.

## Operational Notes & Known Behaviors

* **Spectral vs. broadband albedo in `process_BESS_table()`:** The gridded `BESS_JPL()` entry point (and `JET3.JET()`) preserves separate `PAR_albedo` and `NIR_albedo` inputs or scales broadband `albedo` using GEOS-5 FP `ALBVISDR` and `ALBNIRDR`. In `process_BESS_table()`, `use_spectral_albedo` defaults to `False` to preserve exact parity with the frozen `ECOv002-cal-val-BESS-JPL-outputs.csv` verification baseline (`rtol=1e-4, atol=1e-7`); pass `use_spectral_albedo=True` to enable directional PAR/NIR albedos in tabular workflows.
* **Default atmospheric $\text{CO}_2$ (`Ca`) in `process_BESS_table()`:** When `Ca` is omitted from `input_df`, `process_BESS_table()` defaults `Ca` to `400.0` ppm so offline verification tables without `Ca` run deterministically without network access.
* **Kelvin-to-Celsius conversion in `meteorology.py`:** `meteorology()` retains `Ta_C = Ta_K - 273.16` from Youngryel Ryu's original MATLAB formulation to preserve `rtol=1e-4` numerical parity with the ECOSTRESS Collection 2/3 Cal-Val and `JET3` baseline.

## Release History & Technical Evolution

### Architectural Eras

1. **Initial Prototype & PyPI Packaging (`v1.0.5` – `v1.1.3`, Jul 2024 – Mar 2025):**
   Translated Youngryel Ryu's MATLAB BESS implementation into Python (`BESS` / `breathing_earth_system_simulator`), decomposed photosynthesis (`C3_photosynthesis`, `C4_photosynthesis`), radiative transfer (`canopy_shortwave_radiation`, `canopy_longwave_radiation`), energy balance (`canopy_energy_balance`, `soil_energy_balance`), and meteorology into standalone modules, externalized `FLiESANN` and `sun-angles`, and added GitHub Actions CI.
2. **Unified `BESS_JPL` Entry Point & Ecosystem Integration (`v1.2.0` – `v1.8.0`, Mar 2025 – Apr 2025):**
   Introduced the high-level `BESS` / `BESS_JPL` raster entry point with automatic retrieval of auxiliary inputs from `GEOS5FP`, `MODISCI`, `koppengeiger`, `gedi-canopy-height`, and `NASADEM`; rebranded the package to `BESS-JPL` (`v1.6.1`); standardized dictionary outputs (`v1.5.0`); added intermediate distribution checks (`check_distribution`); and implemented directional visible/NIR albedo scaling (`v1.7.0`).
3. **Tabular Cal-Val Pipelines, Upscaling & SI Standardization (`v1.9.0` – `v1.22.0`, May 2025 – Dec 2025):**
   Extracted `BESS_JPL/model.py`, added tabular batch processing (`process_BESS_table.py`, `v1.10.0`), exposed soil heat flux `G_Wm2` (`v1.9.0` / `v1.12.0`), added sensitivity analysis (`v1.18.0`) and daylight evapotranspiration upscaling (`v1.19.0` – `v1.21.0`), and standardized variable naming with explicit SI unit suffixes (`v1.20.0`).
4. **Offline Mode, Verification Suite & ECOSTRESS Alignment (`v1.23.0` – `v1.29.0`, Jan – Nov 2026):**
   Added `offline_mode` and `MissingOfflineParameter` validation (`v1.25.0` – `v1.26.0`), fixed the `BALL_BERRY_INTERCEPT_C4` default (`v1.27.0`), vectorized tabular solar time calculations (`v1.28.0`), and aligned `BESS-JPL` with `GEOS5FP v2.26.0` and `JET3 v3.2.0` using per-variable `GEOS5FPConnection` retrievals and opt-in spectral albedo table processing (`v1.29.0`).

### Complete Release Log

| Version | Date (UTC) | Summary |
| :--- | :--- | :--- |
| **`v1.29.0`** | 2026-11-23 | **ECOSTRESS consistency alignment:** replaced batched `GEOS5FP.query()` with per-variable `GEOS5FPConnection` methods (`#99`); added opt-in `use_spectral_albedo: bool = False` in `process_BESS_table()` (`#100`); fixed optional `Ta_C`/`RH`, per-row `list[Point]` geometry, and mixed timestamp parsing in `process_BESS_table()` (`#101`, `#102`, `#104`); consolidated `MissingOfflineParameter` (`#103`); pinned `GEOS5FP>=2.26.0` and direct dependencies (`#105`); replaced stray `print()` calls with `logger.debug()` (`#106`); and relocated ad-hoc root scripts out of `pytest` discovery (`#107`). |
| **`v1.28.0`** | 2026-01-29 | Optimized `process_BESS_table` and `retrieve_BESS_JPL_GEOS5FP_inputs` for vectorized `GEOS-5 FP` point queries and updated `verify()` baselines. |
| **`v1.27.0`** | 2026-01-22 | Fixed default `ball_berry_intercept_C4` handling (`BALL_BERRY_INTERCEPT_C4 = 0.04`). |
| **`v1.26.1`** | 2026-01-22 | Added progress logging for static vegetation and auxiliary parameter retrievals. |
| **`v1.26.0`** | 2026-01-22 | Required explicit clumping index (`CI`) and elevation (`elevation_m`) in `offline_mode=True`. |
| **`v1.25.0`** | 2026-01-21 | Added `offline_mode` and `MissingOfflineParameter` checks to prevent unintended network calls during offline runs. |
| **`v1.24.0`** | 2026-01-20 | Updated and verified ECOSTRESS Collection 2 Cal-Val input and output reference datasets. |
| **`v1.23.0`** | 2026-01-14 | Fixed input dataset generation and scalar extraction for Cal-Val tables. |
| **`v1.22.0`** | 2025-12-23 | Improved Cal-Val input dataset generation and sensitivity analysis workflows. |
| **`v1.21.0`** | 2025-12-11 | Included daylight-upscaled in-situ values in the validation dataset. |
| **`v1.20.1`** | 2025-11-18 | Cleaned up logger output across model and input retrieval routines. |
| **`v1.20.0`** | 2025-11-10 | Standardized variable names across model inputs, intermediates, and outputs with explicit SI unit suffixes (`_Wm2`, `_μmolm2s1`, `_deg`). |
| **`v1.19.1`** | 2025-11-10 | Reduced Python wheel package size by excluding non-package assets via `MANIFEST.in` and `pyproject.toml`. |
| **`v1.19.0`** | 2025-11-10 | Integrated `daylight-evapotranspiration` upscaling from instantaneous latent heat flux (`LE_Wm2`). |
| **`v1.18.0`** | 2025-09-10 | Added sensitivity analysis notebooks and multi-panel diagnostic figures. |
| **`v1.17.0`** | 2025-08-15 | Updated model verification checks and input handling. |
| **`v1.16.0`** | 2025-08-13 | Refined input retrieval and model execution workflows. |
| **`v1.15.1`** | 2025-07-30 | Maintenance and dependency compatibility updates. |
| **`v1.15.0`** | 2025-07-08 | Fixed digital elevation model (`NASADEM`) retrieval and added Python 3.12 compatibility. |
| **`v1.14.0`** | 2025-07-08 | Fixed MODIS clumping index (`MODISCI`) download directory configuration. |
| **`v1.13.0`** | 2025-07-03 | Added configurable `GEDI_download_directory` for canopy height retrieval. |
| **`v1.12.0`** | 2025-07-01 | Included soil heat flux (`G_Wm2`) in model output dictionaries. |
| **`v1.11.0`** | 2025-06-30 | Updated CI test matrix and dependencies for Python 3.11+. |
| **`v1.10.0`** | 2025-06-03 | Extracted `BESS_JPL/model.py` and added `process_BESS_table.py` for Cal-Val and sensitivity table processing. |
| **`v1.9.2`** | 2025-05-14 | Added missing-variable and array-shape validation checks for `FLiESANN` radiative transfer inputs. |
| **`v1.9.1`** | 2025-05-12 | Updated unit tests (`tests/test_import_BESS_JPL.py`) to match the `BESS-JPL` package namespace. |
| **`v1.9.0`** | 2025-05-12 | Included soil heat flux `G` in `BESS_JPL` model outputs. |
| **`v1.8.0`** | 2025-04-02 | Separated required and optional parameters in the `BESS_JPL` function signature. |
| **`v1.7.0`** | 2025-04-02 | Introduced `BESS_JPL` as the primary entry point, added `NASADEM` default elevation, and implemented GEOS-5 FP visible/NIR albedo scaling. |
| **`v1.6.1`** | 2025-04-01 | Rebranded package from `breathing-earth-system-simulator` to `BESS-JPL`. |
| **`v1.5.4`** | 2025-03-28 | Added `check_distribution` assertions for inputs and intermediates in `canopy_shortwave_radiation`. |
| **`v1.5.3`** | 2025-03-28 | Added intermediate radiative transfer checks and zero-LAI handling in `canopy_shortwave_radiation`. |
| **`v1.5.2`** | 2025-03-28 | Made `check_distribution` compatible with NumPy arrays in addition to `Raster` objects. |
| **`v1.5.1`** | 2025-03-28 | Added intermediate distribution diagnostics across meteorology, $V_{\mathrm{cmax}}$, radiation, and flux modules. |
| **`v1.5.0`** | 2025-03-28 | Updated `BESS()` to return a named dictionary of flux and radiation outputs instead of a positional tuple. |
| **`v1.4.8`** | 2025-03-28 | Updated clumping index integration to use the uppercase `MODISCI` package (`>=1.3.0`). |
| **`v1.4.7`** | 2025-03-28 | Added automatic clumping index (`CI`) retrieval via `MODISCI`. |
| **`v1.4.6`** | 2025-03-28 | Added `None` parameter validation in `canopy_shortwave_radiation`. |
| **`v1.4.5`** | 2025-03-28 | Fixed `geometry` handling and removed legacy `CoordinateArray` overrides. |
| **`v1.4.4`** | 2025-03-28 | Fixed atmospheric $\text{CO}_2$ retrieval using `GEOS5FP_connection.CO2SC()`. |
| **`v1.4.3`** | 2025-03-28 | Migrated meteorological data retrieval to the uppercase `GEOS5FP` package (`>=1.1.1`). |
| **`v1.4.1`** | 2025-03-27 | Updated `FLiESANN` dependency pin (`>=1.4.0` / `>=1.4.1`). |
| **`v1.4.0`** | 2025-03-27 | Switched radiative transfer calls to the unified `FLiESANN()` entry point. |
| **`v1.3.0`** | 2025-03-25 | Refined `BESS()` signature (`ST_C`, `NDVI`, `albedo` first) and integrated `solar-apparent-time`. |
| **`v1.2.3`** | 2025-03-24 | Updated `FLiESANN` interface arguments (`day_of_year`, `hour_of_day`, `geometry`, `GEOS5FP_connection`). |
| **`v1.2.2`** | 2025-03-19 | Standardized `datetime_UTC` parameter usage across remaining `GEOS5FP` calls. |
| **`v1.2.1`** | 2025-03-19 | Fixed `datetime_UTC` parameter passing in `Ta_C` and `RH` lookups. |
| **`v1.2.0`** | 2025-03-13 | Introduced `BESS()` as the main entry point with optional default parameters (`koppengeiger`, `gedi-canopy-height`, `geos5fp`). |
| **`v1.1.3`** | 2025-03-11 | Reduced wheel size by excluding example imagery and adding `dev` optional dependencies. |
| **`v1.1.2`** | 2025-03-11 | Added GitHub Actions CI workflow and cleaned up `README.md`. |
| **`v1.1.1`** | 2025-03-11 | Restored `vegetation_conversion` sub-module (`FVC_from_NDVI`, `LAI_from_NDVI`, `SAVI_from_NDVI`). |
| **`v1.0.5`** | 2024-11-02 | Initial PyPI release (`breathing_earth_system_simulator`) translating Ryu's MATLAB BESS model into modular Python. |

## References

The following scientific references provide detailed information about the BESS model and its underlying algorithms:

1. Ryu, Y., Baldocchi, D. D., Kobayashi, H., van Ingen, C., Li, J., Black, T. A., ... & Ueyama, M. (2011). Integration of MODIS land and atmosphere products with a coupled-process model to estimate gross primary productivity and evapotranspiration from 1 km to global scales. *Remote Sensing of Environment, 115*(8), 1865-1874. https://doi.org/10.1016/j.rse.2011.03.009

2. Kobayashi, H., Ryu, Y., Baldocchi, D. D., Welles, J. M., & Norman, J. M. (2012). On the correct estimation of gap fraction: How to remove scattered radiation in gap fraction measurements? *Agricultural and Forest Meteorology, 160*, 14-25. https://doi.org/10.1016/j.agrformet.2012.03.001

3. Fisher, J. B., Lee, B., Purdy, A. J., Halverson, G. H., Dohlen, M. B., & Tu, K. P. (2020). ECOSTRESS: NASA's next generation mission to measure evapotranspiration from the International Space Station. *Water Resources Research, 56*(4), e2019WR026058. https://doi.org/10.1029/2019WR026058

4. Ryu, Y., Jiang, C., Kobayashi, H., & Detto, M. (2018). Modis-derived global land products of shortwave radiation and diffuse and total photosynthetically active radiation at 5 km resolution from 2000. *Remote Sensing of Environment, 204*, 812-825. https://doi.org/10.1016/j.rse.2017.09.021

5. Kobayashi, H., & Iwabuchi, H. (2008). A coupled 1-D atmosphere and canopy radiative transfer model for an atmosphere with a nonlambertian surface. *Journal of Quantitative Spectroscopy and Radiative Transfer, 109*(17-18), 2955-2970. https://doi.org/10.1016/j.jqsrt.2008.07.008