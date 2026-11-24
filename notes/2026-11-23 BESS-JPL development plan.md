# `BESS-JPL` Comprehensive Development History & `v1.29.0` Engineering Plan

* **Repository:** [`JPL-Evapotranspiration-Algorithms/BESS-JPL`](https://github.com/JPL-Evapotranspiration-Algorithms/BESS-JPL)
* **Current Release:** `v1.28.0` (January 29, 2026)
* **Target Release:** `v1.29.0` (November 23, 2026)
* **Lead Developer:** Gregory H. Halverson (NASA Jet Propulsion Laboratory, 329G)
* **Upstream Dependencies:** `GEOS5FP` (`v2.26.0`), `FLiESANN` (`v2.13.1`), `MODISCI` (`>=1.4.0`), `NASADEM` (`>=1.3.0`), `gedi-canopy-height` (`>=1.1.0`), `koppengeiger` (`>=1.1.0`), `daylight-evapotranspiration`, `rasters` (`>=1.9.0`)
* **Downstream Ecosystem Consumers:** `JET3` (`v3.2.0`), `ECOv003-L3T-L4T-JET` (`v1.23.0`), `PTJPL` (`v1.9.0`), `PTJPLSM` (`v1.10.0`), `STIC-JPL` (`v1.6.0`), `PM-JPL` (`v1.11.0`), `AquaSEBS` (`v1.3.0`)

---

## 1. Executive Summary & ECOSTRESS Operational Mandate

`BESS-JPL` implements the Breathing Earth System Simulator (Ryu et al., 2011) in Python as a coupled atmospheric-canopy radiative transfer, $C_3$/$C_4$ biochemical photosynthesis, stomatal conductance, and two-source energy balance model. Within NASA's **ECOSTRESS Collection 3** and **Surface Biology and Geology (SBG)** processing systems, `BESS-JPL` serves two irreplaceable functions inside the **`JET3`** ensemble engine and the operational **`ECOv003-L3T-L4T-JET`** Product Generating Executable (PGE):

1. **Ensemble Evapotranspiration ($LE$ / $ET_{\text{daily}}$) Constituent:** `BESS-JPL` provides one of four independent physical evapotranspiration estimates (`LE_Wm2` and daylight-upscaled `ET_daylight_kg`) combined via ensemble median in `L3T_JET` alongside `PT-JPL-SM`, `STIC-JPL`, and `PM-JPL`.
2. **Sole Provider of Gross Primary Production ($GPP$) and Water Use Efficiency ($WUE$):** `BESS-JPL` is the **only** model in the `JET3` ensemble that computes terrestrial carbon assimilation (`GPP`), directly driving the `L4T_WUE` operational data product ($\text{WUE} = \text{GPP}_{\text{inst}} / \text{ET}_{\text{canopy,inst}}$).

### System Architecture & Operational Data Flow

```text
+-----------------------------------------------------------------------------+
|                    ECOv003-L3T-L4T-JET (v1.23.0 Operational PGE)            |
|  Parses XML RunConfig, reads L2T_LSTE + L2T_STARS, writes COG/JSON products |
+--------------------------------------+--------------------------------------+
                                       | calls JET(...) on 70 m RasterGrids
                                       v
+-----------------------------------------------------------------------------+
|                         JET3 (v3.2.0 Science Ensemble)                      |
|  Downscales Ta_C/RH/SM, runs FLiES-ANN & Verma Rn, aggregates 4 ET models   |
+------+--------------------+--------------------+--------------------+-------+
       |                    |                    |                    |
       v                    v                    v                    v
+-------------+      +-------------+      +-------------+      +-------------+
|  PT-JPL-SM  |      |  STIC-JPL   |      |   PM-JPL    |      |  BESS-JPL   |
|  (v1.10.0)  |      |  (v1.6.0)   |      |  (v1.11.0)  |      |  (v1.29.0)  |
+------+------+      +------+------+      +------+------+      +------+------+
       |                    |                    |                    |
       +--------------------+--------------------+--------------------+
                                       | per-variable GEOS5FPConnection calls
                                       v
+-----------------------------------------------------------------------------+
|                         GEOS5FP (v2.26.0 Meteorology)                       |
|  Gridded .nc4 granule interpolation (Raster) & OPeNDAP point queries        |
+-----------------------------------------------------------------------------+
```

Following the November 2026 ecosystem upgrade to Python 3.12+, `GEOS5FP v2.26.0`, and `JET3 v3.2.0`, end-to-end integration testing of `ECOv003-L3T-L4T-JET` exposed a `TypeError` in `BESS-JPL v1.28.0`: because `BESS-JPL` is the sole constituent package calling `GEOS5FP.query(target_variables=list)` instead of dedicated per-variable `GEOS5FPConnection` methods, `GEOS5FP >=2.24.0` returns a bare `Raster` rather than a `dict` whenever `JET3` leaves only a single missing variable (`Ca` / `CO2SC`).

This development plan documents the full evolution of `BESS-JPL` from `v1.0.5` through `v1.28.0` and defines the exact engineering tasks, mathematical safeguards, and verification gates required for **`v1.29.0`**.

---

## 2. Governing Consistency Invariants

Every change in `BESS-JPL v1.29.0` is governed by two strict, non-negotiable invariants across the ECOSTRESS Collection 3 processing chain:

### Invariant 1: Operational Gridded `ECOv003-L3T-L4T-JET` & `JET3` Parity
* ECOSTRESS Collection 3 forward processing is anchored to the frozen scientific baseline established in late 2025 (`eco-level-2-3-4-application` `v1.7.0` / `v1.7.1`) and validated by `JET3.verify()` at `rtol=1e-4, atol=1e-7`.
* **Requirement:** `BESS-JPL v1.29.0` must execute cleanly under `GEOS5FP >=2.26.0` inside `JET3` and `ECOv003-L3T-L4T-JET` while producing numerically identical gridded outputs across all four operational products:
  * `ECOv003_L3T_JET` (`ETdaily`, `ETinst`, `ETuncertainty`)
  * `ECOv003_L3T_ETAUX` (`Ta`, `RH`, `SM`, `Rg`, `Rn`)
  * `ECOv003_L4T_ESI` (`ESI`)
  * `ECOv003_L4T_WUE` (`WUE` derived from `BESS-JPL` `GPP` and `LE_canopy_Wm2`)

### Invariant 2: Tabular `process_BESS_table()` & `BESS_JPL.verify()` Baseline Parity
* `BESS_JPL/verify.py` (invoked via `make verify` and `pytest tests/test_verify.py`) executes `process_BESS_table(input_df, offline_mode=True)` against the 1,065-record ECOSTRESS Collection 2 Cal-Val reference dataset (`ECOv002-cal-val-BESS-JPL-inputs.csv`) and asserts that `G_Wm2`, `Rn_Wm2`, and `LE_Wm2` match `ECOv002-cal-val-BESS-JPL-outputs.csv` within `rtol=1e-4, atol=1e-7`.
* **Requirement:** Default execution of `process_BESS_table()` must preserve exact numerical parity with the frozen reference outputs (`rtol=1e-4, atol=1e-7`). Any behavioral enhancement that alters tabular radiative transfer physics—specifically switching from broadband `albedo` to directional `PAR_albedo` / `NIR_albedo`—must be gated behind an opt-in parameter (`use_spectral_albedo: bool = False`).

---

## 3. Complete Historical Development (`v1.0.5` $\rightarrow$ `v1.28.0`)

### 3.1. Architectural Eras

1. **Era 1 — Prototype Translation, Modularization & Initial PyPI Packaging (`v1.0.5` – `v1.1.3`, Jul 2024 – Mar 2025):**
   Ported Youngryel Ryu's MATLAB BESS formulation into Python (`BESS` / `breathing_earth_system_simulator`), decomposed photosynthesis (`C3_photosynthesis.py`, `C4_photosynthesis.py`), radiative transfer (`canopy_shortwave_radiation.py`, `canopy_longwave_radiation.py`), surface energy balance (`canopy_energy_balance.py`, `soil_energy_balance.py`), and meteorology (`meteorology.py`) into standalone modules, externalized `FLiESANN` and `sun-angles`, and established GitHub Actions CI.
2. **Era 2 — High-Level `BESS_JPL` Entry Point, Rebranding & Spectral Albedo Scaling (`v1.2.0` – `v1.8.0`, Mar – Apr 2025):**
   Introduced a unified raster entry point (`BESS` $\rightarrow$ `BESS_JPL`) with automated retrieval of auxiliary inputs from `GEOS5FP`, `MODISCI`, `koppengeiger`, `gedi-canopy-height`, and `NASADEM`; rebranded the package to `BESS-JPL` (`v1.6.1`); standardized dictionary outputs (`v1.5.0`); added intermediate distribution checks (`check_distribution`); and implemented directional visible/NIR albedo scaling (`v1.7.0`).
3. **Era 3 — Tabular Cal-Val Pipelines, SI Unit Standardization & Daylight Upscaling (`v1.9.0` – `v1.22.0`, May – Dec 2025):**
   Extracted core orchestration into `BESS_JPL/model.py`, introduced `process_BESS_table.py` (`v1.10.0`) for ECOSTRESS Cal-Val and Monte Carlo sensitivity analysis, added soil heat flux (`G_Wm2`), standardized variable naming with explicit SI unit suffixes (`v1.20.0`), integrated daylight evapotranspiration upscaling (`daylight-evapotranspiration`), and decomposed aerodynamic resistance and Paw & Gao latent heat calculations.
4. **Era 4 — Input Retrieval Separation, Offline PGE Mode & Table Optimization (`v1.23.0` – `v1.28.0`, Jan 2026):**
   Separated input acquisition into `retrieve_BESS_inputs.py` and `retrieve_BESS_JPL_GEOS5FP_inputs.py`, added `GeoDataFrame` input loaders and `verify.py` regression testing, implemented `offline_mode` and `MissingOfflineParameter` for air-gapped PGE containers (`v1.25.0`–`v1.26.0`), fixed C4 Ball-Berry intercept defaults (`v1.27.0`), and vectorized tabular solar time calculations (`v1.28.0`).

### 3.2. Version-by-Version Release Changelog

| Version | Date (UTC) | Key Changes & Architectural Significance |
| :--- | :--- | :--- |
| **`v1.0.5`** | 2024-11-02 | Initial PyPI release (`breathing-earth-system-simulator`) translating Youngryel Ryu's MATLAB BESS code into modular Python (`C3_photosynthesis`, `C4_photosynthesis`, `VCmax`, `canopy_shortwave_radiation`, `canopy_longwave_radiation`, `canopy_energy_balance`, `soil_energy_balance`, `meteorology`). |
| **`v1.1.1`** | 2025-03-11 | Externalized `FLiESANN` and `sun-angles`; decomposed `FVC_from_NDVI`, `LAI_from_NDVI`, `interpolate_C3_C4`, and static GeoTIFF `load_*.py` functions; restored `vegetation_conversion` helper functions. |
| **`v1.1.2`** | 2025-03-11 | Configured GitHub Actions CI workflow and cleaned up `README.md`. |
| **`v1.1.3`** | 2025-03-11 | Added `[project.optional-dependencies] dev` (`pytest`, `build`, `twine`) and reduced wheel package size. |
| **`v1.2.0`** | 2025-03-13 | Introduced `BESS()` as the main high-level entry point with optional static, meteorological (`geos5fp`), climate (`koppengeiger`), and canopy height (`gedi-canopy-height`) parameters. |
| **`v1.2.1`** | 2025-03-19 | Fixed `datetime_UTC` variable references in `Ta_C` and `RH` retrievals. |
| **`v1.2.2`** | 2025-03-19 | Fixed `datetime_UTC` variable references in `COT`, `AOT`, `Ca`, and `wind_speed` retrievals. |
| **`v1.2.3`** | 2025-03-24 | Updated `FLiESANN` call signature (`day_of_year`, `hour_of_day`, `geometry`, `GEOS5FP_connection`) and pinned `FLiESANN>=1.2.1`. |
| **`v1.3.0`** | 2025-03-25 | Refined `BESS()` signature (`ST_C`, `NDVI`, `albedo` required first), standardized on `time_UTC`, and integrated `solar-apparent-time` (`solar_day_of_year_for_area`, `solar_hour_of_day_for_area`). |
| **`v1.4.0`** | 2025-03-27 | Upgraded radiative transfer calls to `FLiESANN` (`>=1.4.0`) unified entry point. |
| **`v1.4.1`** | 2025-03-27 | Pinned `FLiESANN>=1.4.1` dependency requirement. |
| **`v1.4.3`** | 2025-03-28 | Migrated meteorological data retrieval from `geos5fp` to uppercase `GEOS5FP>=1.1.1`. |
| **`v1.4.4`** | 2025-03-28 | Fixed atmospheric $\text{CO}_2$ concentration retrieval (`GEOS5FP_connection.CO2SC`). |
| **`v1.4.5`** | 2025-03-28 | Removed redundant `rt.CoordinateArray` geometry override in `BESS`. |
| **`v1.4.6`** | 2025-03-28 | Added explicit `None` parameter validation checks in `canopy_shortwave_radiation.py`. |
| **`v1.4.7`** | 2025-03-28 | Added automatic clumping index (`CI`) retrieval via `MODISCI`. |
| **`v1.4.8`** | 2025-03-28 | Migrated clumping index dependency from `modisci` to uppercase `MODISCI>=1.3.0`. |
| **`v1.5.0`** | 2025-03-28 | Changed `BESS()` return value from an 8-element positional tuple to a named `dict` (`GPP`, `GPP_daily`, `Rn`, `Rn_soil`, `Rn_canopy`, `LE`, `LE_soil`, `LE_canopy`). |
| **`v1.5.1`** | 2025-03-28 | Instrumented meteorology, $V_{c,\max}$, canopy radiation, and carbon/water flux outputs with `check_distribution` diagnostics. |
| **`v1.5.2`** | 2025-03-28 | Extended `check_distribution` to support `numpy.ndarray` inputs alongside `Raster` objects. |
| **`v1.5.3`** | 2025-03-28 | Added `check_distribution` assertions across all intermediate radiative transfer variables and guarded `fSun` against `LAI == 0`. |
| **`v1.5.4`** | 2025-03-28 | Added `check_distribution` validation on input parameters in `canopy_shortwave_radiation.py`. |
| **`v1.6.1`** | 2025-04-01 | Rebranded package and module namespace to `BESS-JPL` / `BESS_JPL` and adopted external `check-distribution`. |
| **`v1.7.0`** | 2025-04-02 | Renamed entry point to `BESS_JPL()`, added `NASADEM` default elevation lookup, implemented GEOS-5 FP directional visible/NIR albedo scaling (`ALBVISDR / ALBEDO` and `ALBNIRDR / ALBEDO`), and added raster processing example. |
| **`v1.8.0`** | 2025-04-02 | Separated required positional parameters (`ST_C`, `NDVI`, `albedo`) from optional keyword parameters in `BESS_JPL()`. |
| **`v1.9.0`** | 2025-05-12 | Removed legacy `SZA` and `vegetation_conversion` sub-modules and included soil heat flux `G` in `BESS_JPL` outputs. |
| **`v1.9.1`** | 2025-05-12 | Updated unit tests (`tests/test_import_BESS_JPL.py`) for the `BESS_JPL` entry point. |
| **`v1.9.2`** | 2025-05-14 | Added missing-variable and array-shape validation checks prior to invoking `FLiESANN`. |
| **`v1.10.0`** | 2025-06-03 | Extracted `BESS_JPL/model.py` and introduced `process_BESS_table.py` for ECOSTRESS Cal-Val and sensitivity analysis tables. |
| **`v1.11.0`** | 2025-06-30 | Updated `FLiESANN>=1.5.0` and Python 3.11+ compatibility. |
| **`v1.12.0`** | 2025-07-01 | Included soil heat flux `G` in `BESS_JPL/model.py` return dictionary. |
| **`v1.13.0`** | 2025-07-03 | Added configurable `GEDI_download_directory` parameter (`GEDI_DOWNLOAD_DIRECTORY`) for canopy height loading. |
| **`v1.14.0`** | 2025-07-08 | Added configurable `MODISCI_connection` parameter for clumping index caching. |
| **`v1.15.0`** | 2025-07-08 | Fixed digital elevation model integration using `NASADEMConnection` (`NASADEM>=1.3.0`) and expanded CI matrix to Python 3.10–3.12. |
| **`v1.15.1`** | 2025-07-30 | Reverted an erroneous `check-distribution` replacement commit and restored external `check-distribution` dependency. |
| **`v1.16.0`** | 2025-08-13 | Fixed `Raster.open` calls across static loaders (`load_C4_fraction`, `load_NDVI_minimum/maximum`, `load_peakVCmax_C3/C4`, `load_ball_berry_*`, `load_kn`) to pass `nodata=np.nan` and honor `resampling`. |
| **`v1.17.0`** | 2025-08-15 | Scaled `C4_fraction` from $[0, 100]$ percentages to $[0, 1]$ fractions, removed erroneous `fIPAR == 0` NaN mask in `LAI_from_NDVI.py`, and replaced `None in (...)` array checks with explicit `is None` loops. |
| **`v1.18.0`** | 2025-09-10 | Parameterized `C4_FRACTION_SCALE_FACTOR = 0.01`, extracted `process_paw_and_gao_LE.py`, introduced `importlib.metadata` versioning (`version.py`), and expanded `process_BESS_table.py` for Monte Carlo sensitivity analysis. |
| **`v1.19.0`** | 2025-11-10 | Integrated `daylight_ET_from_instantaneous_LE` (`daylight-evapotranspiration`) and updated `FLiESANN>=2.1.0` interface (`elevation_m`, `SZA_deg`, `SWin_Wm2`, `PAR_*`, `NIR_*`). |
| **`v1.19.1`** | 2025-11-10 | Fixed Python wheel size via explicit `[tool.setuptools.package-data]` and `[tool.setuptools.packages.find]` exclusions. |
| **`v1.20.0`** | 2025-11-10 | Standardized parameter and output variable names across all modules with explicit SI unit suffixes (`elevation_m`, `SWin_Wm2`, `PAR_diffuse_Wm2`, `NIR_diffuse_Wm2`, `SZA_deg`, `peakVCmax_C3_μmolm2s1`, `Rn_Wm2`, `LE_Wm2`, `G_Wm2`). |
| **`v1.20.1`** | 2025-11-18 | Cleaned up logger output and updated `FLiESANN>=2.2.1`. |
| **`v1.21.0`** | 2025-12-11 | Included daylight-upscaled in-situ values in validation dataset (`ECOv002-calval-tables>=1.11.0`). |
| **`v1.22.0`** | 2025-12-23 | Consolidated `GEOS-5 FP` retrievals into `retrieve_BESS_JPL_GEOS5FP_inputs.py`, extracted `calculate_bulk_aerodynamic_resistance.py` and `calculate_friction_velocity.py`, enabled `UPSCALE_TO_DAYLIGHT = True`, and added `generate_input_dataset.py`. |
| **`v1.23.0`** | 2026-01-14 | Separated `retrieve_BESS_inputs.py`, added `generate_BESS_inputs_table.py` and `generate_output_dataset.py`, and fixed scalar extraction in Cal-Val table generation. |
| **`v1.24.0`** | 2026-01-20 | Converted `load_ECOv002_calval_BESS_inputs()` to return a `geopandas.GeoDataFrame`, added defaults for `Ca` ($400\,\text{ppm}$) and canopy/soil temperatures (`ST_C`), and calibrated `verify()` tolerances to `rtol=1e-4, atol=1e-7`. |
| **`v1.25.0`** | 2026-01-21 | Added `offline_mode: bool = False` and `MissingOfflineParameter` validation across `BESS_JPL()`, `process_BESS_table()`, `retrieve_BESS_inputs()`, and `retrieve_BESS_JPL_GEOS5FP_inputs()`, and enforced input completeness checks in `verify.py`. |
| **`v1.26.0`** | 2026-01-22 | Required explicit clumping index (`CI`) and elevation (`elevation_m`) when `offline_mode=True`. |
| **`v1.26.1`** | 2026-01-22 | Added `INFO`-level logging prior to retrieving each static vegetation and auxiliary parameter. |
| **`v1.27.0`** | 2026-01-22 | Fixed `ball_berry_intercept_C4` defaulting to `BALL_BERRY_INTERCEPT_C4` (`0.04`) when `None` is passed to `BESS_JPL()`. |
| **`v1.28.0`** | 2026-01-29 | Vectorized `calculate_solar_day_of_year` and `calculate_solar_hour_of_day` across full `GeoSeries` arrays and added `TicToc` stage timing in `process_BESS_table.py`. |

---

## 4. `v1.29.0` Engineering & ECOSTRESS Alignment Plan

### 4.1. Overview of Planned Changes

| Task ID | Priority | Target File(s) | Summary & Ecosystem Rationale | Invariant Protected |
| :--- | :--- | :--- | :--- | :--- |
| **BESS-1** | **Critical** | `BESS_JPL/retrieve_BESS_JPL_GEOS5FP_inputs.py` | Replace batched `GEOS5FP.query()` with direct per-variable `GEOS5FPConnection` methods (`COT()`, `AOT()`, `vapor_gccm()`, `ozone_cm()`, `Ta_C()`, `RH()`, `CO2SC()`, `wind_speed_mps()`, `ALBEDO()`, `ALBVISDR()`, `ALBNIRDR()`). Fixes `TypeError` when `JET3` invokes `BESS_JPL` with only `Ca=None` (`CO2SC`). | **Invariant 1** (`ECOv003-L3T-L4T-JET` & `JET3` execution) |
| **BESS-2** | **High** | `BESS_JPL/process_BESS_table.py` | Add opt-in `use_spectral_albedo: bool = False` parameter. Pass `PAR_albedo if use_spectral_albedo else albedo` and `NIR_albedo if use_spectral_albedo else albedo` to both `retrieve_BESS_JPL_GEOS5FP_inputs()` and `BESS_JPL()`. | **Invariant 1 & Invariant 2** (`BESS_JPL.verify()` parity + opt-in spectral fidelity) |
| **BESS-3** | **High** | `BESS_JPL/process_BESS_table.py` | Extract `Ta_C` (with `"Ta"` alias fallback) and `RH` conditionally, defaulting to `None` when absent so `process_BESS_table(..., offline_mode=False)` can retrieve missing meteorology from `GEOS-5 FP`. | **Invariant 2** |
| **BESS-4** | **High** | `BESS_JPL/process_BESS_table.py` | Pass per-row `geometry` (`list[Point]`) to `retrieve_BESS_JPL_GEOS5FP_inputs()` while retaining `geometry_multipoint` for `BESS_JPL()`. Ensures 1-to-1 coordinate-timestamp pairing in online batch table runs. | **Invariant 2** |
| **BESS-5** | **Medium** | `BESS_JPL/retrieve_BESS_JPL_GEOS5FP_inputs.py`, `BESS_JPL/exceptions.py` | Remove duplicate local `MissingOfflineParameter` class in `retrieve_BESS_JPL_GEOS5FP_inputs.py` and import `MissingOfflineParameter` from `.exceptions`. | **Invariant 1 & Invariant 2** |
| **BESS-6** | **Medium** | `BESS_JPL/process_BESS_table.py` | Update `pd.to_datetime(input_df.time_UTC, format='ISO8601')` to `format='mixed'` to support space-separated and mixed timestamp formats. | **Invariant 2** |
| **BESS-7** | **Medium** | `pyproject.toml`, `README.md` | Bump `version = "1.29.0"`, update `"GEOS5FP>=2.26.0"`, declare direct imports (`"pandas"`, `"geopandas"`, `"shapely"`, `"matplotlib"`, `"python-dateutil"`), and update `README.md` install command to `pip install BESS-JPL`. | **Invariant 1 & Invariant 2** |
| **BESS-8** | **Low** | `BESS_JPL/retrieve_BESS_inputs.py` | Replace stray `print(type(SZA_deg))` and `print(SZA_deg.dtype ...)` calls with `logger.debug(...)` to keep operational PGE stdout logs clean. | **Invariant 1** |
| **BESS-9** | **Low** | Repo root $\rightarrow$ `examples/` or `debugging/` | Relocate root ad-hoc scripts (`test_co2sc.py`, `test_geos5fp_query.py`, `test_process_BESS_table_performance.py`, `debug_GEOS5FP_timeseries.py`, `check_input_distribution.py`, `fix_input_dataset.py`, `generate_*.py`) so `pytest` only runs offline tests in `tests/`. | **Invariant 1 & Invariant 2** |
| **BESS-10** | **Low** | `BESS_JPL/meteorology.py` | Standardize Kelvin-to-Celsius conversion from `Ta_K - 273.16` to `Ta_K - 273.15` and verify both `BESS_JPL.verify()` and `JET3.verify()` remain within `rtol=1e-4, atol=1e-7`. | **Invariant 1 & Invariant 2** |

---

### 4.2. Detailed Implementation Specifications

#### 4.2.1. Per-Variable `GEOS5FPConnection` Retrieval (`BESS-1` & `BESS-5`)
In `BESS_JPL/retrieve_BESS_JPL_GEOS5FP_inputs.py`:
1. Remove the local `class MissingOfflineParameter(Exception):` and import `from .exceptions import MissingOfflineParameter`.
2. Check missing variables up front in `offline_mode=True` and raise `MissingOfflineParameter` listing all missing inputs (`COT`, `AOT`, `vapor_gccm`, `ozone_cm`, `Ta_C`, `RH`, `CO2SC`, `wind_speed_mps`, `PAR_albedo`, `NIR_albedo`).
3. Replace `GEOS5FP_connection.query(target_variables=variables_to_retrieve, ...)` with direct method calls on `GEOS5FPConnection` (`COT()`, `AOT()`, `vapor_gccm()`, `ozone_cm()`, `Ta_C()`, `RH()`, `CO2SC()`, `wind_speed_mps()`, `ALBEDO()`, `ALBVISDR()`, `ALBNIRDR()`), clipping `wind_speed_mps` to `[0.1, None]` and directional albedos to `[0, 1]`. Because `GEOS5FPConnection` caches local NetCDF granules and HTTP directory listings, individual method calls perform zero redundant downloads while returning consistent `Raster` (gridded) or `np.ndarray` (point) types regardless of how many variables are requested.

#### 4.2.2. Opt-In Spectral Albedo & Tabular Processing Fixes (`BESS-2`, `BESS-3`, `BESS-4`, `BESS-6`)
In `BESS_JPL/process_BESS_table.py`:
1. Add `use_spectral_albedo: bool = False` to `process_BESS_table(...)`.
2. Resolve `PAR_albedo_input = PAR_albedo if use_spectral_albedo else albedo` and `NIR_albedo_input = NIR_albedo if use_spectral_albedo else albedo`, passing them into `retrieve_BESS_JPL_GEOS5FP_inputs(...)` and forwarding the resulting `PAR_albedo` and `NIR_albedo` into `BESS_JPL(...)`.
3. Extract `Ta_C` (falling back to `"Ta"`, or `None` if neither column exists) and `RH` (`None` if absent). Keep `Ca = np.full_like(ST_C, 400.0, dtype=np.float64)` when `"Ca"` is absent so `offline_mode=True` verification tables without `Ca` continue to pass.
4. Parse timestamps with `pd.to_datetime(input_df.time_UTC, format='mixed').tolist()`.
5. Pass `geometry=geometry` (`list[Point]`) to `retrieve_BESS_JPL_GEOS5FP_inputs(...)` and `geometry=geometry_multipoint` to `BESS_JPL(...)`.

---

## 5. Verification, Testing & Release Rollout Plan

### 5.1. Automated Regression Test Suite (`tests/test_regressions.py`)
All new unit tests must run completely offline using mocked `GEOS5FP` connections:

1. **`test_retrieve_bess_geos5fp_inputs_single_missing_co2sc` (`BESS-1`):**
   Supplies all parameters except `Ca=None` (mirroring `JET3.JET()`), verifies `GEOS5FP.CO2SC(...)` is called and `.query()` is never invoked, and confirms `Ca` is returned in the output dictionary.
2. **`test_retrieve_bess_geos5fp_inputs_all_missing` (`BESS-1`):**
   Leaves all optional `GEOS-5 FP` inputs as `None`, verifies all 11 `GEOS5FPConnection` methods are invoked, and asserts `PAR_albedo` and `NIR_albedo` are properly scaled from `ALBEDO`, `ALBVISDR`, and `ALBNIRDR` and clipped to `[0, 1]`.
3. **`test_missing_offline_parameter_unified_exception` (`BESS-5`):**
   Asserts `pytest.raises(BESS_JPL.MissingOfflineParameter)` catches missing `GEOS-5 FP` inputs when `retrieve_BESS_JPL_GEOS5FP_inputs(..., offline_mode=True)` is called.
4. **`test_process_bess_table_spectral_albedo_opt_in` (`BESS-2`):**
   Verifies that `use_spectral_albedo=False` (default) passes broadband `albedo` to `PAR_albedo`/`NIR_albedo` without querying `ALBEDO`/`ALBVISDR`/`ALBNIRDR`, while `use_spectral_albedo=True` preserves table `PAR_albedo`/`NIR_albedo` or scales them from `GEOS-5 FP` when missing.
5. **`test_process_bess_table_optional_met_and_mixed_timestamps` (`BESS-3`, `BESS-4`, `BESS-6`):**
   Verifies `process_BESS_table(..., offline_mode=False)` succeeds when `Ta_C` (or `Ta`) and `RH` are omitted, parses mixed space-separated/ISO timestamps, and passes `list[Point]` to `retrieve_BESS_JPL_GEOS5FP_inputs`.
6. **`test_verify_calval_baseline` (Invariant 2):**
   Runs `tests/test_verify.py` (`BESS_JPL.verify()`) to confirm `G_Wm2`, `Rn_Wm2`, and `LE_Wm2` match `ECOv002-cal-val-BESS-JPL-outputs.csv` within `rtol=1e-4, atol=1e-7`.

### 5.2. Cross-Package Release & PGE Deployment Sequence

1. **Implement & Verify `BESS-JPL v1.29.0`:**
   ```bash
   cd ~/Projects/BESS-JPL
   make test
   make verify
   make dist
   ```
2. **Verify `JET3` Ensemble Consistency:**
   ```bash
   cd ~/Projects/JET3
   pip install --upgrade "BESS-JPL>=1.29.0" "GEOS5FP>=2.26.0"
   make test
   make verify
   ```
3. **Update & Smoke-Test `ECOv003-L3T-L4T-JET` Operational PGE:**
   * Pin `"BESS-JPL~=1.29.0"`, `"GEOS5FP~=2.26.0"`, and `"JET3~=3.2.0"` in `ECOv003-L3T-L4T-JET/pyproject.toml`.
   * Run `make test`, `make smoke-test`, `python examples/example_L3T_L4T_JET_run_with_download.py`, and `make docker-smoke-test`.
   * Confirm all four operational granules (`ECOv003_L3T_JET`, `ECOv003_L3T_ETAUX`, `ECOv003_L4T_ESI`, and `ECOv003_L4T_WUE`) are generated without errors and match the Collection 3 baseline.