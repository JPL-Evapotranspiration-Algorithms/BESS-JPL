"""
Debug script to reproduce GEOS5FP time series issue

This script demonstrates the problem where GEOS5FP_connection receives
an entire pandas Series of timestamps instead of individual timestamp values
when processing data with row_wise=False.

Error: Cannot convert input [pandas Series of timestamps] to Timestamp
"""

import pandas as pd
import numpy as np
from datetime import datetime
from GEOS5FP import GEOS5FP_connection

def test_single_timestamp():
    """Test GEOS5FP with a single timestamp - this should work"""
    print("=" * 80)
    print("TEST 1: Single timestamp (should work)")
    print("=" * 80)
    
    connection = GEOS5FP_connection()
    
    # Single point and time
    lat = 31.8214
    lon = -110.8661
    time_UTC = pd.Timestamp('2019-10-02 17:09:40')
    
    print(f"Latitude: {lat}")
    print(f"Longitude: {lon}")
    print(f"Time UTC: {time_UTC}")
    print(f"Time type: {type(time_UTC)}")
    
    try:
        COT = connection.COT(
            time_UTC=time_UTC,
            lat=lat,
            lon=lon
        )
        print(f"✓ SUCCESS: COT = {COT}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
    
    print()


def test_series_timestamp():
    """Test GEOS5FP with a Series of timestamps - this reproduces the error"""
    print("=" * 80)
    print("TEST 2: Series of timestamps (reproduces error)")
    print("=" * 80)
    
    connection = GEOS5FP_connection()
    
    # Create a Series of timestamps (mimics what happens with row_wise=False)
    times = pd.Series([
        '2019-10-02 17:09:40',
        '2019-06-23 16:17:17',
        '2019-06-27 14:35:42',
        '2019-06-30 13:44:10',
        '2019-07-01 12:53:48'
    ])
    time_UTC = pd.to_datetime(times)
    
    # Single point but series of times
    lat = 31.8214
    lon = -110.8661
    
    print(f"Latitude: {lat}")
    print(f"Longitude: {lon}")
    print(f"Time UTC type: {type(time_UTC)}")
    print(f"Time UTC:\n{time_UTC}")
    
    try:
        COT = connection.COT(
            time_UTC=time_UTC,
            lat=lat,
            lon=lon
        )
        print(f"✓ SUCCESS: COT = {COT}")
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
    
    print()


def test_arrays_of_data():
    """Test GEOS5FP with arrays of lat/lon/time - expected vectorized behavior"""
    print("=" * 80)
    print("TEST 3: Arrays of lat/lon/time (expected vectorized use case)")
    print("=" * 80)
    
    connection = GEOS5FP_connection()
    
    # Multiple points and times (what row_wise=False should produce)
    lats = np.array([31.8214, 45.7624, 43.9397, 44.3233, 35.803])
    lons = np.array([-110.8661, -122.3303, -71.7181, -121.6078, -76.6685])
    times = pd.to_datetime([
        '2019-10-02 17:09:40',
        '2019-06-23 16:17:17',
        '2019-06-27 14:35:42',
        '2019-06-30 13:44:10',
        '2019-07-01 12:53:48'
    ])
    
    print(f"Latitudes: {lats}")
    print(f"Longitudes: {lons}")
    print(f"Times type: {type(times)}")
    print(f"Times:\n{times}")
    
    try:
        COT = connection.COT(
            time_UTC=times,
            lat=lats,
            lon=lons
        )
        print(f"✓ SUCCESS: COT = {COT}")
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
    
    print()


def test_dataframe_scenario():
    """Test scenario that mimics the actual cal/val table processing"""
    print("=" * 80)
    print("TEST 4: DataFrame scenario (mimics cal/val table)")
    print("=" * 80)
    
    connection = GEOS5FP_connection()
    
    # Create a mini cal/val table
    df = pd.DataFrame({
        'ID': ['US-SRM', 'US-Wrc', 'US-Oho', 'US-Me6', 'US-NC4'],
        'lat': [31.8214, 45.7624, 43.9397, 44.3233, 35.803],
        'lon': [-110.8661, -122.3303, -71.7181, -121.6078, -76.6685],
        'time_UTC': pd.to_datetime([
            '2019-10-02 17:09:40',
            '2019-06-23 16:17:17',
            '2019-06-27 14:35:42',
            '2019-06-30 13:44:10',
            '2019-07-01 12:53:48'
        ])
    })
    
    print("DataFrame:")
    print(df)
    print()
    
    # This is what happens when row_wise=False and entire columns are passed
    print("Attempting to query with entire DataFrame columns:")
    print(f"  time_UTC type: {type(df['time_UTC'])}")
    print(f"  lat type: {type(df['lat'])}")
    print(f"  lon type: {type(df['lon'])}")
    
    try:
        COT = connection.COT(
            time_UTC=df['time_UTC'],
            lat=df['lat'],
            lon=df['lon']
        )
        print(f"✓ SUCCESS: COT =\n{COT}")
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        print("\nThis is the error we're seeing in the actual code!")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GEOS5FP Time Series Debug Script")
    print("=" * 80 + "\n")
    
    # Run all tests
    test_single_timestamp()
    test_series_timestamp()
    test_arrays_of_data()
    test_dataframe_scenario()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The issue occurs when GEOS5FP_connection receives pandas Series objects
instead of individual values or numpy arrays. This happens in the FLiESANN
processing pipeline when row_wise=False.

Expected behavior:
- Single values should work (Test 1)
- Arrays/Series for vectorized operations should work (Test 3, 4)

Actual behavior:
- Single values work (Test 1)
- Series/DataFrame columns fail with timestamp conversion error (Test 2, 4)

The GEOS5FP package needs to handle pandas Series inputs properly when
used in vectorized mode (row_wise=False).
""")
