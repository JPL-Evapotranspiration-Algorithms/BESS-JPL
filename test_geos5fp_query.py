"""
Quick test of GEOS5FP query with a small subset of data.
"""
import pandas as pd
from datetime import datetime
from rasters import Point
from GEOS5FP import GEOS5FP
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

# Test with just 3 points
test_times = [
    datetime(2024, 1, 15, 12, 0, 0),
    datetime(2024, 1, 15, 13, 0, 0),
    datetime(2024, 1, 15, 14, 0, 0),
]

test_geometry = [
    Point(-118.25, 34.05),  # Los Angeles
    Point(-118.25, 34.05),  # Same location, different time
    Point(-118.26, 34.06),  # Nearby location
]

logger.info(f"Testing with {len(test_times)} points")

# Create GEOS5FP connection
geos5fp = GEOS5FP()

# Query some variables
variables = ["CO2SC", "wind_speed_mps", "ALBEDO"]

logger.info(f"Querying variables: {variables}")

try:
    results = geos5fp.query(
        target_variables=variables,
        time_UTC=test_times,
        geometry=test_geometry,
        verbose=True
    )
    
    logger.info(f"Query succeeded!")
    logger.info(f"Result keys: {list(results.keys())}")
    for key, value in results.items():
        if hasattr(value, 'shape'):
            logger.info(f"  {key}: shape={value.shape}")
        else:
            logger.info(f"  {key}: {type(value)}")
    
except Exception as e:
    logger.error(f"Query failed: {e}", exc_info=True)
