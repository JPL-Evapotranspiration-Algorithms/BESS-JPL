#!/usr/bin/env python
"""Test CO2SC retrieval from GEOS-5 FP"""

from datetime import datetime
from GEOS5FP import GEOS5FP
from rasters import Point, MultiPoint, WGS84
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Create GEOS5FP connection
geos = GEOS5FP()

# Test data - use coordinates and time from a typical BESS-JPL input
test_time = pd.Timestamp("2019-10-02 19:09:40")
test_lat = 35.799
test_lon = -76.656

logger.info(f"Testing CO2SC retrieval")
logger.info(f"Time: {test_time}")
logger.info(f"Location: ({test_lat}, {test_lon})")

# Create geometry
geometry = Point(test_lon, test_lat, crs=WGS84)

# Test 1: Use the CO2SC method directly
logger.info("\n" + "="*70)
logger.info("Test 1: Using .CO2SC() method directly")
logger.info("="*70)
try:
    result1 = geos.CO2SC(time_UTC=test_time, geometry=geometry)
    logger.info(f"✓ CO2SC method succeeded!")
    logger.info(f"Result type: {type(result1)}")
    logger.info(f"Result value: {result1}")
except Exception as e:
    logger.error(f"✗ CO2SC method failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Use the query method with "CO2SC"
logger.info("\n" + "="*70)
logger.info("Test 2: Using .query() method with 'CO2SC' variable")
logger.info("="*70)
try:
    result2 = geos.query(
        target_variables="CO2SC",
        time_UTC=test_time,
        geometry=geometry,
        verbose=True
    )
    logger.info(f"✓ query() method succeeded!")
    logger.info(f"Result type: {type(result2)}")
    logger.info(f"Result value: {result2}")
except Exception as e:
    logger.error(f"✗ query() method failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Use the Ca alias method
logger.info("\n" + "="*70)
logger.info("Test 3: Using .Ca() alias method")
logger.info("="*70)
try:
    result3 = geos.Ca(time_UTC=test_time, geometry=geometry)
    logger.info(f"✓ Ca method succeeded!")
    logger.info(f"Result type: {type(result3)}")
    logger.info(f"Result value: {result3}")
except Exception as e:
    logger.error(f"✗ Ca method failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Try with MultiPoint (like the actual BESS-JPL code uses)
logger.info("\n" + "="*70)
logger.info("Test 4: Using .query() with MultiPoint geometry")
logger.info("="*70)
geometry_multi = MultiPoint([(test_lon, test_lat)], crs=WGS84)
try:
    result4 = geos.query(
        target_variables=["CO2SC"],
        time_UTC=[test_time],
        geometry=geometry_multi,
        verbose=True
    )
    logger.info(f"✓ query() with MultiPoint succeeded!")
    logger.info(f"Result type: {type(result4)}")
    logger.info(f"Result:\n{result4}")
    if hasattr(result4, 'columns'):
        logger.info(f"Columns: {list(result4.columns)}")
        if 'CO2SC' in result4.columns:
            logger.info(f"CO2SC values: {result4['CO2SC'].values}")
except Exception as e:
    logger.error(f"✗ query() with MultiPoint failed: {e}")
    import traceback
    traceback.print_exc()

logger.info("\n" + "="*70)
logger.info("Testing complete!")
logger.info("="*70)
