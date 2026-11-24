#!/usr/bin/env python3
"""
Performance test script for process_BESS_table function.

This script loads a sample of data from the ECOv002 calibration/validation
inputs CSV and runs it through process_BESS_table to analyze performance.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from pytictoc import TicToc

# Configure logging to show timing information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import BESS-JPL package
from BESS_JPL.process_BESS_table import process_BESS_table

def main():
    """Run performance test on process_BESS_table with sample data."""
    
    timer = TicToc()
    
    # Path to input CSV file
    csv_path = Path(__file__).parent / "BESS_JPL" / "ECOv002-cal-val-BESS-JPL-inputs.csv"
    
    if not csv_path.exists():
        print(f"Error: Input CSV file not found at {csv_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("BESS-JPL process_BESS_table Performance Test")
    print("=" * 80)
    
    # Load the full dataset
    print(f"\nLoading data from: {csv_path}")
    timer.tic()
    base_df = pd.read_csv(csv_path)
    elapsed = timer.tocvalue()
    print(f"Loaded {len(base_df)} rows in {elapsed:.2f} seconds")
    
    # Duplicate the data 10 times for larger volume testing
    print(f"\nDuplicating data 10 times for larger volume testing...")
    timer.tic()
    input_df = pd.concat([base_df] * 10, ignore_index=True)
    elapsed = timer.tocvalue()
    print(f"Created duplicated dataset with {len(input_df)} rows in {elapsed:.2f} seconds")
    
    print(f"\nInput data shape: {input_df.shape}")
    print(f"Columns: {len(input_df.columns)}")
    
    # Display sample info
    if 'geometry' in input_df.columns:
        print(f"Geometry column present: Yes")
    if 'time_UTC' in input_df.columns:
        print(f"Time range: {input_df['time_UTC'].min()} to {input_df['time_UTC'].max()}")
        print(f"Unique times: {input_df['time_UTC'].nunique()}")
    
    # Run process_BESS_table with timing
    print(f"\nRunning process_BESS_table on full dataset (offline mode)...")
    timer.tic()
    
    try:
        output_df = process_BESS_table(
            input_df,
            GEOS5FP_connection=None,
            verbose=True,
            offline_mode=True  # Use offline mode to avoid external data fetches
        )
        
        total_elapsed = timer.tocvalue()
        
        print(f"\n{'=' * 80}")
        print(f"COMPLETED in {total_elapsed:.2f} seconds")
        print(f"{'=' * 80}")
        print(f"Output shape: {output_df.shape}")
        print(f"Processing rate: {len(input_df) / total_elapsed:.2f} rows/second")
        print(f"Time per row: {total_elapsed / len(input_df) * 1000:.2f} milliseconds")
        
        # Show some output columns
        new_columns = set(output_df.columns) - set(input_df.columns)
        if new_columns:
            print(f"\nNew columns added ({len(new_columns)}): {sorted(list(new_columns))[:10]}...")
        
        # Check for any issues
        null_counts = output_df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"\nWarning: Found null values in output")
            print(null_counts[null_counts > 0].head())
            
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
            
    print(f"\n{'=' * 80}")
    print("Performance test complete")
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    main()
