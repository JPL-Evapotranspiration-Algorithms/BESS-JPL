#!/usr/bin/env fish
# Publish v1.29.0 ecosystem alignment and bug-fix issues to JPL-Evapotranspiration-Algorithms/BESS-JPL

set REPO "JPL-Evapotranspiration-Algorithms/BESS-JPL"

# Ensure required labels exist (silently succeeds if already present)
set -l labels \
    "bug:d73a4a:Something isn't working" \
    "enhancement:a2eeef:New feature or request" \
    "critical:b60205:Critical blocking issue" \
    "high:d93f0b:High priority" \
    "medium:fbca04:Medium priority" \
    "low:0e8a16:Low priority" \
    "ecosystem:1d76db:Cross-package ECOSTRESS / JET3 alignment" \
    "maintenance:5319e7:Routine maintenance and dependency updates" \
    "packaging:0075ca:Build, wheel, and pyproject.toml configuration" \
    "cleanup:cfd3d7:Code and repository organization" \
    "tests:bfd4f2:Unit tests and verification suite" \
    "science:006b75:Scientific formulation and physical constants"

for entry in $labels
    set -l parts (string split ":" $entry)
    gh label create $parts[1] --color $parts[2] --description $parts[3] --repo $REPO --force >/dev/null 2>&1
end

function create_issue --argument-names title labels body
    echo "Creating issue: $title"
    gh issue create \
        --repo $REPO \
        --title "$title" \
        --label "$labels" \
        --body "$body"
end

# Issue 1
create_issue \
    "Fix TypeError when GEOS5FP.query() returns a single Raster (CO2SC only)" \
    "bug,critical,ecosystem" \
'## Summary
When `JET3` invokes `BESS_JPL` with all meteorological inputs pre-populated except `Ca`, `variables_to_retrieve` evaluates to `["CO2SC"]` (length 1). Under `GEOS5FP >=2.24.0`, `.query()` returns a bare `rasters.Raster` instead of a `dict`, causing `list(retrieved.keys())` in `BESS_JPL/retrieve_BESS_JPL_GEOS5FP_inputs.py` to raise `TypeError: '\''NoneType'\'' object is not callable`.

## Consistency Context
- Every other package in the `JET3` ecosystem (`JET3`, `PT-JPL`, `PT-JPL-SM`, `STIC-JPL`, `PM-JPL`, `FLiESANN`, `AquaSEBS`) retrieves `GEOS-5 FP` inputs via dedicated per-variable methods on `GEOS5FPConnection` (`Ta_C()`, `RH()`, `wind_speed_mps()`, etc.).
- Direct per-variable calls consistently return a `Raster` on gridded geometries and an `np.ndarray` on points, restoring gridded `ECOv003-L3T-L4T-JET` execution with identical retrieved values while leaving `offline_mode=True` (`BESS_JPL.verify()`) untouched.

## Action Items
- [ ] Replace the batched `GEOS5FP_connection.query(target_variables=variables_to_retrieve, ...)` call in `BESS_JPL/retrieve_BESS_JPL_GEOS5FP_inputs.py` with direct per-variable method calls (`COT()`, `AOT()`, `vapor_gccm()`, `ozone_cm()`, `Ta_C()`, `RH()`, `CO2SC()`, `wind_speed_mps()`, `ALBEDO()`, `ALBVISDR()`, `ALBNIRDR()`).
- [ ] Add unit tests covering single-missing-variable (`Ca=None`, mirroring `JET3`) and all-missing-variable execution paths.'

# Issue 2
create_issue \
    "Add opt-in use_spectral_albedo parameter to process_BESS_table()" \
    "enhancement,bug,high" \
'## Summary
`process_BESS_table()` in `BESS_JPL/process_BESS_table.py` extracts `PAR_albedo` and `NIR_albedo` from `input_df`, then hardcodes `NIR_albedo=albedo` when calling `retrieve_BESS_JPL_GEOS5FP_inputs()` and `PAR_albedo=albedo, NIR_albedo=albedo` when calling `BESS_JPL()`.

## Consistency Context
- **Gridded `ECOv003-L3T-L4T-JET` path:** `JET3.JET()` computes spectral `PAR_albedo` and `NIR_albedo` from broadband `albedo` and GEOS-5 FP directional albedos (`ALBVISDR`, `ALBNIRDR`) and passes them directly to `BESS_JPL(...)`, which preserves them intact.
- **Tabular `process_BESS_table()` path:** Because `ECOv002-cal-val-BESS-JPL-outputs.csv` was generated under broadband substitution, enabling spectral albedos unconditionally shifts `G_Wm2`, `Rn_Wm2`, and `LE_Wm2` beyond the `rtol=1e-4, atol=1e-7` verification threshold in `BESS_JPL.verify()`.
- Gating spectral albedo forwarding behind `use_spectral_albedo: bool = False` preserves exact `BESS_JPL.verify()` and `ECOv003-L3T-L4T-JET` parity while allowing tabular workflows to opt into spectral albedo physics.

## Action Items
- [ ] Add `use_spectral_albedo: bool = False` to `process_BESS_table()`.
- [ ] Pass `PAR_albedo=PAR_albedo if use_spectral_albedo else albedo` and `NIR_albedo=NIR_albedo if use_spectral_albedo else albedo` to both `retrieve_BESS_JPL_GEOS5FP_inputs()` and `BESS_JPL()`.
- [ ] Add regression tests verifying `use_spectral_albedo=False` (preserves `verify()` baseline and avoids redundant GEOS-5 FP albedo fetches) and `use_spectral_albedo=True` (forwards table or GEOS-5 FP-scaled `PAR_albedo` and `NIR_albedo`).'

# Issue 3
create_issue \
    "Fix UnboundLocalError / AttributeError when Ta_C or RH is omitted in process_BESS_table()" \
    "bug,high" \
'## Summary
`process_BESS_table()` in `BESS_JPL/process_BESS_table.py` unconditionally reads `input_df.RH` and leaves `Ta_C` unbound if neither `"Ta_C"` nor `"Ta"` is present in `input_df`, breaking online mode (`offline_mode=False`) when meteorological inputs are expected to be retrieved from `GEOS-5 FP`.

## Consistency Context
- Preserves existing behavior when `Ta_C` and `RH` columns are present (as in `ECOv002-cal-val-BESS-JPL-inputs.csv`), while aligning online table processing with `JET3/process_JET_table.py`.
- Retain the `Ca = np.full_like(ST_C, 400.0, dtype=np.float64)` default when `"Ca"` is omitted from `input_df` so `BESS_JPL.verify(..., offline_mode=True)` continues to pass without requiring a `Ca` column.

## Action Items
- [ ] Extract `Ta_C` (with `"Ta"` column alias fallback) and `RH` conditionally, defaulting to `None` when absent.
- [ ] Add a regression test for `process_BESS_table(..., offline_mode=False)` when `Ta_C`/`RH` are omitted.'

# Issue 4
create_issue \
    "Pass per-row list[Point] geometry to retrieve_BESS_JPL_GEOS5FP_inputs() in process_BESS_table()" \
    "bug,high" \
'## Summary
`process_BESS_table()` in `BESS_JPL/process_BESS_table.py` passes a single `MultiPoint` of length $N$ alongside `time_UTC` (`list` of length $N$) into `retrieve_BESS_JPL_GEOS5FP_inputs()`, risking mismatched spatial pairing compared to `list[Point]` (as documented in `BESS_JPL/generate_input_dataset.py`).

## Action Items
- [ ] Pass `geometry=geometry` (`list[Point]`) to `retrieve_BESS_JPL_GEOS5FP_inputs()` while retaining `geometry=geometry_multipoint` for `BESS_JPL()`.'

# Issue 5
create_issue \
    "Consolidate duplicate MissingOfflineParameter exception class" \
    "bug,medium" \
'## Summary
`BESS_JPL/retrieve_BESS_JPL_GEOS5FP_inputs.py` defines a local `MissingOfflineParameter` class that shadows `BESS_JPL.exceptions.MissingOfflineParameter` (exported via `from BESS_JPL import MissingOfflineParameter`), preventing callers from catching offline validation errors raised during `GEOS-5 FP` input checks.

## Action Items
- [ ] Remove the duplicate `MissingOfflineParameter` class definition in `BESS_JPL/retrieve_BESS_JPL_GEOS5FP_inputs.py` and import `MissingOfflineParameter` from `.exceptions`.
- [ ] Add a unit test asserting `retrieve_BESS_JPL_GEOS5FP_inputs(..., offline_mode=True)` raises `BESS_JPL.MissingOfflineParameter`.'

# Issue 6
create_issue \
    "Support mixed timestamp formats (format='\''mixed'\'') in process_BESS_table()" \
    "bug,medium" \
'## Summary
`pd.to_datetime(input_df.time_UTC, format='\''ISO8601'\'')` in `BESS_JPL/process_BESS_table.py` raises a `ValueError` on non-ISO space-separated or mixed timestamp strings in calibration/validation tables.

## Action Items
- [ ] Update `pd.to_datetime(input_df.time_UTC, format='\''ISO8601'\'')` to `format='\''mixed'\''` (matching `JET3/process_JET_table.py`).'

# Issue 7
create_issue \
    "Update pyproject.toml dependencies (GEOS5FP>=2.26.0, pandas, etc.) and README.md install command" \
    "maintenance,packaging,medium" \
'## Summary
`pyproject.toml` pins an outdated `"GEOS5FP>=2.19.0"` floor and omits direct imports (`"pandas"`, `"geopandas"`, `"shapely"`, `"matplotlib"`, `"python-dateutil"`) from `dependencies`, while `README.md` references the obsolete `pip install breathing-earth-system-simulator` package name.

## Action Items
- [ ] Update `"GEOS5FP>=2.26.0"` and declare missing direct imports in `[project] dependencies` in `pyproject.toml`.
- [ ] Update the `README.md` installation command to `pip install BESS-JPL`.'

# Issue 8
create_issue \
    "Replace stray print() statements in retrieve_BESS_inputs.py with logger.debug()" \
    "cleanup,low" \
'## Summary
Unconditional `print(type(SZA_deg))` and `print(SZA_deg.dtype ...)` statements in `BESS_JPL/retrieve_BESS_inputs.py` write directly to `stdout` during operational `ECOv003-L3T-L4T-JET` PGE runs.

## Action Items
- [ ] Replace both `print(...)` calls in `BESS_JPL/retrieve_BESS_inputs.py` with `logger.debug(...)`.'

# Issue 9
create_issue \
    "Relocate root ad-hoc scripts (test_*.py, debug_*.py, generate_*.py) out of repository root" \
    "cleanup,tests,low" \
'## Summary
Top-level `test_co2sc.py` and `test_geos5fp_query.py` execute live NASA NCCS OPeNDAP queries when collected by `pytest` during CI and offline test runs, causing hangs or failures.

## Action Items
- [ ] Move root-level ad-hoc scripts (`test_co2sc.py`, `test_geos5fp_query.py`, `test_process_BESS_table_performance.py`, `debug_GEOS5FP_timeseries.py`, `check_input_distribution.py`, `fix_input_dataset.py`, `generate_*.py`) into `examples/` or `debugging/` so `pytest` runs only unit tests in `tests/`.'

# Issue 10
create_issue \
    "Standardize Kelvin-to-Celsius offset (273.15) in meteorology.py" \
    "science,cleanup,low" \
'## Summary
`BESS_JPL/meteorology.py` converts Kelvin to Celsius using `Ta_K - 273.16` instead of the standard `273.15` offset used elsewhere in `BESS-JPL` and `GEOS5FP`, introducing a `0.01 K` discrepancy.

## Consistency Context
Because a `0.01 K` offset could affect pixels near the `0 °C` latent-heat freeze threshold (`process_paw_and_gao_LE.py`), this change must be validated against both `BESS_JPL.verify()` and `JET3.verify()` (`rtol=1e-4, atol=1e-7`) before merging.

## Action Items
- [ ] Update `Ta_C = Ta_K - 273.16` to `Ta_C = Ta_K - 273.15` in `BESS_JPL/meteorology.py` and confirm both `BESS_JPL.verify()` and `JET3.verify()` pass within `rtol=1e-4, atol=1e-7`.'

echo "Done! All 10 issues published to $REPO."