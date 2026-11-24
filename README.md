# Breathing Earth System Simulator (`BESS-JPL`) Python Implementation

[![CI](https://github.com/JPL-Evapotranspiration-Algorithms/BESS-JPL/actions/workflows/ci.yml/badge.svg)](https://github.com/JPL-Evapotranspiration-Algorithms/BESS-JPL/actions/workflows/ci.yml)

This software package is a Python implementation of the Breathing Earth System Simulator (BESS) model. It was re-implemented in Python by Gregory Halverson at Jet Propulsion Laboratory based on [MATLAB code](https://www.environment.snu.ac.kr/bess-flux) produced by Youngryel Ryu at Seoul University. The BESS model was designed to quantify global gross primary productivity (GPP) and evapotranspiration (ET) using MODIS with a spatial resolution of 1–5 km and a temporal resolution of 8 days. It couples atmospheric and canopy radiative transfer processes with photosynthesis, stomatal conductance, and transpiration models on sunlit and shaded portions of vegetation and soil. An artificial neural network emulator of Hideki Kobayashi's Forest Light Environmental Simulator (FLiES) radiative transfer model is used to estimate incoming solar radiation. This implementation of BESS was designed to process GPP at fine spatial resolution using surface temperature from the ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) mission and normalized difference vegetation index (NDVI) and albedo from the [Spatial Timeseries for Automated high-Resolution multi-Sensor (STARS) data fusion system](https://github.com/STARS-Data-Fusion). The software was developed as part of a research grant by the NASA Research Opportunities in Space and Earth Sciences (ROSES) program. It was designed for use by the ECOSTRESS mission as a precursor for the Surface Biology and Geology (SBG) mission. However, it may also be useful for general remote sensing and GIS projects in Python. This package can be utilized for remote sensing research in Jupyter notebooks and deployed for operations in data processing pipelines. This software is being released according to the SPD-41 open-science requirements of NASA-funded ROSES projects.

Gregory H. Halverson (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
Lead developer<br>
NASA Jet Propulsion Laboratory 329G

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
MATLAB to python translation<br>
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

## Installation

Use the pip package manager to install this package:

```bash
pip install BESS-JPL
```

For local development and running the test suite:

```bash
git clone https://github.com/JPL-Evapotranspiration-Algorithms/BESS-JPL.git
cd BESS-JPL
pip install -e ".[dev]"
pytest
```

## Known Issues & Operational Notes

* **Single-variable `GEOS5FP` queries (`CO2SC` only) under `GEOS5FP >=2.24.0`:** In `v1.28.0`, `retrieve_BESS_JPL_GEOS5FP_inputs()` batches missing variables into `GEOS5FP_connection.query(...)` and expects a `dict`. When only a single variable is missing (such as `Ca` / `CO2SC` when invoked from `JET3` / `ECOv003-L3T-L4T-JET`), `GEOS5FP.query()` returns a bare `Raster` or `np.ndarray`, raising `TypeError: 'NoneType' object is not callable` on `retrieved.keys()`. **Workaround:** Pass `Ca` explicitly (or pre-populate all meteorological inputs) until `v1.29.0` switches to direct per-variable `GEOS5FPConnection` methods.
* **Spectral vs. broadband albedo in `process_BESS_table()`:** The gridded `BESS_JPL()` entry point (and `JET3.JET()`) preserves separate `PAR_albedo` and `NIR_albedo` inputs or scales broadband `albedo` using GEOS-5 FP `ALBVISDR` and `ALBNIRDR`. By contrast, `process_BESS_table()` in `v1.28.0` overrides both channels with broadband `albedo` to match the frozen `ECOv002-cal-val-BESS-JPL-outputs.csv` verification baseline (`rtol=1e-4, atol=1e-7`). An opt-in `use_spectral_albedo: bool = False` flag is planned for `v1.29.0` to enable spectral PAR/NIR albedos in tabular workflows without breaking `BESS_JPL.verify()`.
* **Required columns in `process_BESS_table()`:** In `v1.28.0`, `process_BESS_table()` expects `Ta_C` (or `Ta`), `RH`, and ISO-8601-formatted `time_UTC` strings even when `offline_mode=False`, and passes a single `MultiPoint` rather than per-row `list[Point]` geometries into `GEOS-5 FP` retrieval. When `Ca` is omitted from `input_df`, `process_BESS_table()` defaults `Ca` to `400.0` ppm rather than querying `CO2SC`.
* **Duplicate `MissingOfflineParameter` exception class:** `BESS_JPL/retrieve_BESS_JPL_GEOS5FP_inputs.py` defines a local `MissingOfflineParameter` exception class separate from `BESS_JPL.exceptions.MissingOfflineParameter`. Catching `BESS_JPL.MissingOfflineParameter` will not intercept offline errors raised during `GEOS-5 FP` input checks until the classes are consolidated in `v1.29.0`.
* **Root-level diagnostic `test_*.py` scripts:** Running bare `pytest` from the repository root in `v1.28.0` also collects ad-hoc live-network scripts (`test_co2sc.py`, `test_geos5fp_query.py`, `test_process_BESS_table_performance.py`). Run `pytest tests/` in offline environments.

## Release History & Technical Evolution

### Architectural Eras

1. **Initial Prototype & PyPI Packaging (`v1.0.5` – `v1.1.3`, Jul 2024 – Mar 2025):**
   Translated Youngryel Ryu's MATLAB BESS implementation into Python (`BESS` / `breathing_earth_system_simulator`), decomposed photosynthesis (`C3_photosynthesis`, `C4_photosynthesis`), radiative transfer (`canopy_shortwave_radiation`, `canopy_longwave_radiation`), energy balance (`canopy_energy_balance`, `soil_energy_balance`), and meteorology into standalone modules, externalized `FLiESANN` and `sun-angles`, and added GitHub Actions CI.
2. **Unified `BESS_JPL` Entry Point & Ecosystem Integration (`v1.2.0` – `v1.8.0`, Mar 2025 – Apr 2025):**
   Introduced the high-level `BESS` / `BESS_JPL` raster entry point with automatic retrieval of auxiliary inputs from `GEOS5FP`, `MODISCI`, `koppengeiger`, `gedi-canopy-height`, and `NASADEM`; rebranded the package to `BESS-JPL` (`v1.6.1`); standardized dictionary outputs (`v1.5.0`); added intermediate distribution checks (`check_distribution`); and implemented directional visible/NIR albedo scaling (`v1.7.0`).
3. **Tabular Cal-Val Pipelines, Upscaling & SI Standardization (`v1.9.0` – `v1.22.0`, May 2025 – Dec 2025):**
   Extracted `BESS_JPL/model.py`, added tabular batch processing (`process_BESS_table.py`, `v1.10.0`), exposed soil heat flux `G_Wm2` (`v1.9.0` / `v1.12.0`), added sensitivity analysis (`v1.18.0`) and daylight evapotranspiration upscaling (`v1.19.0` – `v1.21.0`), and standardized variable naming with explicit SI unit suffixes (`v1.20.0`).
4. **Offline Mode, Verification Suite & Table Optimization (`v1.23.0` – `v1.28.0`, Jan 2026):**
   Added `offline_mode` and `MissingOfflineParameter` validation (`v1.25.0` – `v1.26.0`), fixed the `BALL_BERRY_INTERCEPT_C4` default (`v1.27.0`), and refactored `process_BESS_table` and `retrieve_BESS_JPL_GEOS5FP_inputs` for vectorized `GEOS-5 FP` batch retrieval and `verify()` regression testing (`v1.28.0`).

### Complete Release Log

| Version | Date (UTC) | Summary |
| :--- | :--- | :--- |
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
| **`v1.5.1`** | 2025-03-28 | Added intermediate distribution diagnostics across meteorology, $V_{c,\max}$, radiation, and flux modules. |
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