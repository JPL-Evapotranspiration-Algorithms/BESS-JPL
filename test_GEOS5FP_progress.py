"""
Test script to verify GEOS5FP progress logging improvements.
Tests with a small subset of the full dataset.
"""

import pandas as pd
import logging
from ECOv002_calval_tables import load_calval_table
from GEOS5FP import GEOS5FP
from shapely.geometry import Point

# Configure logging to see all INFO messages
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    print("=" * 80)
    print("GEOS5FP Progress Logging Test")
    print("=" * 80)
    print()
    
    # Load the calibration/validation table
    print("Loading cal-val table...")
    calval_df = load_calval_table()
    
    # Take only first 10 records for quick test
    test_df = calval_df.head(10).copy()
    test_df['time_UTC'] = pd.to_datetime(test_df['time_UTC'])
    
    print(f"Testing with {len(test_df)} records")
    print(f"Date range: {test_df['time_UTC'].min()} to {test_df['time_UTC'].max()}")
    print()
    
    # Display sample of the data
    print("Sample of test data:")
    print(test_df[['ID', 'Lat', 'Long', 'time_UTC']].head())
    print()
    
    # Initialize GEOS5FP connection
    print("Initializing GEOS5FP connection...")
    connection = GEOS5FP()
    print()
    
    # Create Point geometries
    print("Creating point geometries...")
    geometries = [Point(row['Long'], row['Lat']) for _, row in test_df.iterrows()]
    print()
    
    # Test with time series and geometries
    print("=" * 80)
    print("Testing COT retrieval with time series...")
    print("=" * 80)
    print()
    
    try:
        # This should trigger the new progress logging
        COT_values = connection.COT(
            time_UTC=test_df['time_UTC'].values,
            geometry=geometries
        )
        
        print()
        print("=" * 80)
        print(f"SUCCESS: Retrieved {len(COT_values)} COT values")
        print("=" * 80)
        print()
        
        if isinstance(COT_values, pd.DataFrame):
            print("Result type: DataFrame")
            print(f"Columns: {COT_values.columns.tolist()}")
            print(f"Shape: {COT_values.shape}")
            print("\nFirst few results:")
            print(COT_values.head())
        else:
            print(f"Result type: {type(COT_values)}")
            print(f"Length: {len(COT_values)}")
            
    except Exception as e:
        print()
        print("=" * 80)
        print(f"ERROR: {type(e).__name__}: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
