#!/usr/bin/env python3
"""
Quick test script for generate_BESS_inputs_table function
"""

import pandas as pd
from datetime import datetime
from BESS_JPL import generate_BESS_inputs_table

def test_generate_BESS_inputs_table():
    """Test the generate_BESS_inputs_table function with sample data."""
    
    # Create a small test DataFrame with required columns
    test_data = {
        'time_UTC': [datetime(2020, 6, 15, 12, 0, 0), datetime(2020, 6, 15, 13, 0, 0)],
        'geometry': ['POINT (-120.5 38.5)', 'POINT (-121.0 39.0)'],
        'ST_C': [25.0, 26.5],
        'NDVI': [0.7, 0.65],
        'albedo': [0.15, 0.18],
        'Ta_C': [24.0, 25.0],
        'RH': [0.5, 0.55],
        'elevation_m': [100.0, 150.0],
        'COT': [0.1, 0.2],
        'AOT': [0.05, 0.08],
        'vapor_gccm': [1.5, 1.8],
        'ozone_cm': [0.3, 0.32]
    }
    
    input_df = pd.DataFrame(test_data)
    
    print("Input DataFrame:")
    print(input_df)
    print("\nGenerating BESS inputs...")
    
    # Call the function
    output_df = generate_BESS_inputs_table(input_df)
    
    print("\nOutput DataFrame columns:")
    print(output_df.columns.tolist())
    print(f"\nOutput shape: {output_df.shape}")
    print(f"Input shape: {input_df.shape}")
    
    # Check that output has more columns than input (retrieved/calculated parameters)
    assert len(output_df.columns) >= len(input_df.columns), \
        "Output should have at least as many columns as input"
    
    # Check that all input rows are preserved
    assert len(output_df) == len(input_df), \
        "Output should have same number of rows as input"
    
    # Check for some expected output columns
    expected_columns = ['ST_C', 'NDVI', 'albedo', 'Ta_C', 'RH', 'elevation_m']
    for col in expected_columns:
        assert col in output_df.columns, f"Expected column '{col}' not found in output"
    
    print("\n✓ Test passed successfully!")
    print("\nSample output row:")
    print(output_df.iloc[0])
    
    return output_df

if __name__ == "__main__":
    result = test_generate_BESS_inputs_table()
