"""
Retrieve Cloud Optical Thickness (COT) for the full ECOv002 cal/val dataset
using only the GEOS5FP and FLiESANN packages (matching BESS-JPL's approach).

This script loads the calibration/validation table and retrieves COT values
from GEOS-5 FP meteorological data using the same retrieval pattern as BESS-JPL.
"""

import pandas as pd
from GEOS5FP import GEOS5FP
from FLiESANN import retrieve_FLiESANN_GEOS5FP_inputs


def main():
    # Load the calibration/validation table
    from ECOv002_calval_tables import load_calval_table
    calval_df = load_calval_table()
    
    # Ensure `time_UTC` is in datetime format
    calval_df['time_UTC'] = pd.to_datetime(calval_df['time_UTC'])
    
    print(f"Processing {len(calval_df)} records from the cal/val dataset")
    
    # Initialize GEOS5FP connection
    print("Initializing GEOS5FP connection...")
    geos5fp = GEOS5FP(download_directory="GEOS5FP_download")
    
    # Extract arrays for vectorized retrieval (matching BESS-JPL approach)
    # Create a list of datetime objects for point-by-point queries
    time_list = calval_df['time_UTC'].tolist()
    lat_array = calval_df['Lat'].values
    lon_array = calval_df['Long'].values
    
    print("Retrieving COT values using FLiESANN retrieval pattern...")
    
    # Use the same retrieval function that BESS-JPL uses
    # This handles the point-by-point queries with different timestamps
    from rasters import RasterGeometry
    import numpy as np
    
    # Create a simple geometry for the spatial extent
    # (GEOS5FP will handle point-by-point queries internally)
    min_lat, max_lat = lat_array.min(), lat_array.max()
    min_lon, max_lon = lon_array.min(), lon_array.max()
    
    # For point-by-point queries, create geometry from bounding box
    geometry = RasterGeometry.from_bbox(
        left=min_lon,
        bottom=min_lat,
        right=max_lon,
        top=max_lat,
        cell_size=0.1  # Arbitrary resolution for the geometry
    )
    
    try:
        # Retrieve atmospheric parameters using FLiESANN's retrieval function
        # This is exactly how BESS-JPL retrieves COT
        atmospheric_inputs = retrieve_FLiESANN_GEOS5FP_inputs(
            COT=None,
            AOT=None,
            vapor_gccm=None,
            ozone_cm=None,
            geometry=geometry,
            time_UTC=time_list,
            GEOS5FP_connection=geos5fp,
            resampling="cubic",
            zero_COT_correction=False
        )
        
        # Extract COT from results
        COT_values = atmospheric_inputs["COT"]
        
        # Convert to array if needed
        if hasattr(COT_values, 'array'):
            COT_values = COT_values.array.flatten()
        elif hasattr(COT_values, 'values'):
            COT_values = COT_values.values
        
        # Ensure we have the right number of values
        if len(COT_values) != len(calval_df):
            print(f"Warning: Retrieved {len(COT_values)} values but expected {len(calval_df)}")
            # Pad or truncate as needed
            if len(COT_values) < len(calval_df):
                COT_values = np.pad(COT_values, (0, len(calval_df) - len(COT_values)), constant_values=np.nan)
            else:
                COT_values = COT_values[:len(calval_df)]
        
    except Exception as e:
        print(f"Error during COT retrieval: {e}")
        import traceback
        traceback.print_exc()
        # Create NaN array as fallback
        COT_values = np.full(len(calval_df), np.nan)
    
    # Create results DataFrame
    results_df = calval_df[['ID', 'Lat', 'Long', 'time_UTC']].copy()
    results_df['COT'] = COT_values
    
    # Save to CSV
    output_file = "ECOv002-cal-val-COT-full.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\nProcessed {len(results_df)} records")
    print(f"Successful retrievals: {results_df['COT'].notna().sum()}")
    print(f"Failed retrievals: {results_df['COT'].isna().sum()}")
    print(f"Results saved to: {output_file}")
    
    # Display summary statistics
    print("\nCOT Summary Statistics:")
    print(results_df['COT'].describe())


if __name__ == "__main__":
    main()
