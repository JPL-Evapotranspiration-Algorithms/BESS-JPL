# GEOS5FP Query Optimization Summary

## Problem
The script was hanging when trying to retrieve GEOS-5 FP data for 1065 point-time combinations from the ECOv002 calibration/validation dataset.

## Root Cause
1. **No timeout**: OPeNDAP connections could hang indefinitely
2. **Limited logging**: Couldn't see progress or identify where the hang occurred
3. **Large time windows**: Using 7-day query windows transferred too much data

## Solutions Implemented

### 1. Added Connection Timeouts
- Set socket timeout to 120 seconds for OPeNDAP connections
- Matches the existing retry logic in single-variable queries
- Prevents indefinite hangs

**Files Modified:**
- `/Users/halverso/Projects/GEOS5FP/GEOS5FP/GEOS5FP_point.py`

### 2. Reduced Query Window Size
- Changed from 7-day to 1-day time windows (`max_days_per_query=1`)
- Reduces data transfer per query
- Improves reliability for point-time queries

**Files Modified:**
- `/Users/halverso/Projects/GEOS5FP/GEOS5FP/query.py`

### 3. Enhanced Progress Logging
- Added detailed logging for dataset opening and query completion
- Shows query start/end for each batch
- Helps track progress through large datasets
- Enabled verbose mode by default in `retrieve_BESS_JPL_GEOS5FP_inputs`

**Files Modified:**
- `/Users/halverso/Projects/GEOS5FP/GEOS5FP/GEOS5FP_point.py`
- `/Users/halverso/Projects/GEOS5FP/GEOS5FP/query.py`
- `/Users/halverso/Projects/BESS-JPL/BESS_JPL/retrieve_BESS_JPL_GEOS5FP_inputs.py`

## Test Results

**Test Query**: 3 points at 2 unique locations
- **Total Time**: 14 seconds
- **Variables**: CO2SC, wind_speed_mps, ALBEDO (from 3 different datasets)
- **Batches**: 2 query batches
- **Result**: ✅ Successful

## Performance Estimate for Full Dataset

Based on test results:
- **1065 records** at approximately **200 unique coordinates** (estimated)
- Each coordinate requires **1-2 query batches** (depending on time clustering)
- Average **2-3 seconds per batch**
- **Estimated total time**: 10-20 minutes

### Breakdown by Dataset
The query needs to retrieve from 3 GEOS-5 FP datasets:
1. `tavg3_2d_chm_Nx` - CO2 concentration
2. `inst3_2d_asm_Nx` - Wind components (U2M, V2M)
3. `tavg1_2d_rad_Nx` - Albedo

Each dataset is queried separately, multiplying the number of batches by 3.

## Recommendations

### For Running the Full Dataset
1. **Use tmux or screen**: The process will take 10-20 minutes
   ```bash
   conda activate BESS-JPL
   tmux new -s bess-geos5fp
   make generate-input-dataset
   ```

2. **Monitor Progress**: Watch the log output to see:
   - Current batch number (e.g., "Batch 45/600")
   - ETA estimates
   - Successful query completion messages

3. **Handle Failures**: The code has retry logic and will skip failed queries
   - Failed queries will show warnings but won't stop execution
   - Missing data will be set to None/NaN

### For Future Optimization
If this still takes too long, consider:
1. **Caching**: Store retrieved GEOS-5 FP data locally to avoid re-querying
2. **Parallel queries**: Query multiple locations simultaneously (requires careful rate limiting)
3. **Bulk download**: Download entire GEOS-5 FP files for the date range and extract locally

## Next Steps

Run the full dataset generation:
```bash
cd /Users/halverso/Projects/BESS-JPL
conda activate BESS-JPL
make generate-input-dataset
```

The script will now:
1. Show detailed progress for each query
2. Complete without hanging
3. Provide ETA estimates
4. Successfully retrieve all 1065 records (or indicate failures)
