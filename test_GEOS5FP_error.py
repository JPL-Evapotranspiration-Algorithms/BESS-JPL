"""
Minimal script to reproduce GEOS5FP error:
"unsupported operand type(s) for -: 'list' and 'datetime.timedelta'"
"""

import pandas as pd
from datetime import datetime
from shapely.geometry import Point
from GEOS5FP import GEOS5FP

def main():
    # Create GEOS5FP connection
    print("Creating GEOS5FP connection...")
    geos5fp = GEOS5FP()
    
    # Test points from the error log
    test_points = [
        {"lat": 29.7381, "lon": -82.2188, "time": "2019-01-28 21:08:10"},
        {"lat": 32.5907, "lon": -106.8425, "time": "2019-01-28 22:40:54"}
    ]
    
    for point_data in test_points:
        print(f"\nTesting point: ({point_data['lat']}, {point_data['lon']}) at {point_data['time']}")
        
        # Create geometry
        geometry = Point(point_data['lon'], point_data['lat'])
        
        # Parse time
        time_utc = pd.to_datetime(point_data['time'])
        
        print(f"  Geometry: {geometry}")
        print(f"  Time (type): {type(time_utc)}")
        print(f"  Time (value): {time_utc}")
        
        try:
            # This is where the error occurs according to the traceback
            print("  Attempting to retrieve vapor_gccm...")
            vapor = geos5fp.vapor_gccm(
                time_UTC=time_utc,
                geometry=geometry
            )
            print(f"  Success! vapor_gccm = {vapor}")
            
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
