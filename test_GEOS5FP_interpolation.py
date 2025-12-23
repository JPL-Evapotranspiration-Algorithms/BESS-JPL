"""
Test GEOS5FP with OPeNDAP disabled (using interpolation method instead).
This is more reliable when OPeNDAP server is having issues.
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
    print("GEOS5FP Test with OPeNDAP Disabled (Using Interpolation)")
    print("=" * 80)
    print()
    
    # Load the calibration/validation table
    print("Loading cal-val table...")
    calval_df = load_calval_table()
    
    # Take only first 5 records for quick test
    test_df = calval_df.head(5).copy()
    test_df['time_UTC'] = pd.to_datetime(test_df['time_UTC'])
    
    print(f"Testing with {len(test_df)} records")
    print(f"Date range: {test_df['time_UTC'].min()} to {test_df['time_UTC'].max()}")
    print()
    
    # Display sample of the data
    print("Sample of test data:")
    print(test_df[['ID', 'Lat', 'Long', 'time_UTC']].head())
    print()
    
    # Initialize GEOS5FP connection with OPeNDAP disabled
    print("Initializing GEOS5FP connection (OPeNDAP disabled)...")
    connection = GEOS5FP(use_opendap=False)
    print()
    
    # Create Point geometries
    print("Creating point geometries...")
    geometries = [Point(row['Long'], row['Lat']) for _, row in test_df.iterrows()]
    print()
    
    # Test with time series and geometries
    print("=" * 80)
    print("Testing COT retrieval with interpolation method...")
    print("=" * 80)
    print()
    
    try:
        # This should use interpolation instead of OPeNDAP
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
        elif isinstance(COT_values, list):
            print(f"Result type: list of {type(COT_values[0])}")
            print(f"Length: {len(COT_values)}")
            print(f"\nSample values: {COT_values[:3]}")
        else:
            print(f"Result type: {type(COT_values)}")
            print(f"Length: {len(COT_values)}")
            print(f"\nSample values: {COT_values[:3] if hasattr(COT_values, '__getitem__') else 'N/A'}")
            
    except Exception as e:
        print()
        print("=" * 80)
        print(f"ERROR: {type(e).__name__}: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
