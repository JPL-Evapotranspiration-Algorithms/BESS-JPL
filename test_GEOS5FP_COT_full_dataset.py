"""
Test script to reproduce COT retrieval for the full ECOv002 cal-val dataset
using only the GEOS5FP package (without BESS_JPL dependency).

This script demonstrates how to:
1. Load the cal-val table
2. Query GEOS5FP for COT values for all records
3. Handle the time series and coordinate data properly
"""

import pandas as pd
import numpy as np
from datetime import datetime
from ECOv002_calval_tables import load_calval_table
from GEOS5FP import GEOS5FP_connection


def main():
    """Query COT for the full ECOv002 cal-val dataset"""
    
    print("=" * 80)
    print("GEOS5FP COT Retrieval Test - Full Dataset")
    print("=" * 80)
    print()
    
    # Load the calibration/validation table
    print("Loading cal-val table...")
    calval_df = load_calval_table()
    
    # Ensure time_UTC is in datetime format
    calval_df['time_UTC'] = pd.to_datetime(calval_df['time_UTC'])
    
    print(f"Loaded {len(calval_df)} records")
    print(f"Date range: {calval_df['time_UTC'].min()} to {calval_df['time_UTC'].max()}")
    print(f"Sites: {calval_df['ID'].nunique()} unique sites")
    print()
    
    # Display sample of the data
    print("Sample of input data:")
    print(calval_df[['ID', 'lat', 'lon', 'time_UTC']].head(10))
    print()
    
    # Initialize GEOS5FP connection
    print("Initializing GEOS5FP connection...")
    connection = GEOS5FP_connection()
    print()
    
    # Method 1: Query COT for all records at once (vectorized)
    print("=" * 80)
    print("METHOD 1: Vectorized query (all records at once)")
    print("=" * 80)
    
    try:
        print(f"Querying COT for {len(calval_df)} records...")
        print(f"  Time UTC type: {type(calval_df['time_UTC'])}")
        print(f"  Latitude type: {type(calval_df['lat'])}")
        print(f"  Longitude type: {type(calval_df['lon'])}")
        
        COT_values = connection.COT(
            time_UTC=calval_df['time_UTC'],
            lat=calval_df['lat'],
            lon=calval_df['lon']
        )
        
        print(f"✓ SUCCESS: Retrieved {len(COT_values)} COT values")
        print(f"COT statistics:")
        print(f"  Type: {type(COT_values)}")
        if isinstance(COT_values, (pd.Series, np.ndarray)):
            print(f"  Mean: {np.nanmean(COT_values):.4f}")
            print(f"  Std: {np.nanstd(COT_values):.4f}")
            print(f"  Min: {np.nanmin(COT_values):.4f}")
            print(f"  Max: {np.nanmax(COT_values):.4f}")
            print(f"  NaN count: {np.sum(np.isnan(COT_values))}")
        
        # Add COT to the dataframe
        calval_df['COT'] = COT_values
        
        print("\nSample of results:")
        print(calval_df[['ID', 'time_UTC', 'lat', 'lon', 'COT']].head(10))
        print()
        
        # Save results to CSV
        output_file = "COT_full_dataset_results.csv"
        calval_df[['ID', 'time_UTC', 'lat', 'lon', 'COT']].to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        print("\nTrying alternative method...")
        print()
        
        # Method 2: Query COT row by row (fallback)
        print("=" * 80)
        print("METHOD 2: Row-by-row query (fallback)")
        print("=" * 80)
        
        COT_list = []
        
        for idx, row in calval_df.iterrows():
            try:
                COT = connection.COT(
                    time_UTC=row['time_UTC'],
                    lat=row['lat'],
                    lon=row['lon']
                )
                COT_list.append(COT)
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(calval_df)} records...")
                    
            except Exception as e:
                print(f"  ✗ Error at row {idx}: {e}")
                COT_list.append(np.nan)
        
        print(f"\n✓ Completed: Retrieved {len(COT_list)} COT values")
        
        # Add COT to the dataframe
        calval_df['COT'] = COT_list
        
        print("\nCOT statistics:")
        print(f"  Mean: {np.nanmean(COT_list):.4f}")
        print(f"  Std: {np.nanstd(COT_list):.4f}")
        print(f"  Min: {np.nanmin(COT_list):.4f}")
        print(f"  Max: {np.nanmax(COT_list):.4f}")
        print(f"  NaN count: {np.sum(np.isnan(COT_list))}")
        
        print("\nSample of results:")
        print(calval_df[['ID', 'time_UTC', 'lat', 'lon', 'COT']].head(10))
        print()
        
        # Save results to CSV
        output_file = "COT_full_dataset_results.csv"
        calval_df[['ID', 'time_UTC', 'lat', 'lon', 'COT']].to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
    
    print()
    print("=" * 80)
    print("Test completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
