import logging

import numpy as np
import rasters as rt
from dateutil import parser
from pandas import DataFrame

from .constants import *
from .model import BESS_JPL

logger = logging.getLogger(__name__)

def process_BESS_table(
        input_df: DataFrame,
        C4_fraction_scale_factor: float = C4_FRACTION_SCALE_FACTOR) -> DataFrame:
    ST_C = np.array(input_df.ST_C).astype(np.float64)
    NDVI = np.array(input_df.NDVI).astype(np.float64)

    NDVI = np.where(NDVI > 0.06, NDVI, np.nan).astype(np.float64)

    albedo = np.array(input_df.albedo).astype(np.float64)
    
    if "Ta_C" in input_df:
        Ta_C = np.array(input_df.Ta_C).astype(np.float64)
    elif "Ta" in input_df:
        Ta_C = np.array(input_df.Ta).astype(np.float64)

    RH = np.array(input_df.RH).astype(np.float64)

    if "elevation_km" in input_df:
        elevation_km = np.array(input_df.elevation_km).astype(np.float64)
    else:
        elevation_km = None

    # --- Handle geometry and time columns ---
    import pandas as pd
    from rasters import MultiPoint, WGS84
    from shapely.geometry import Point

    def ensure_geometry(df):
        if "geometry" in df:
            if isinstance(df.geometry.iloc[0], str):
                def parse_geom(s):
                    s = s.strip()
                    if s.startswith("POINT"):
                        coords = s.replace("POINT", "").replace("(", "").replace(")", "").strip().split()
                        return Point(float(coords[0]), float(coords[1]))
                    elif "," in s:
                        coords = [float(c) for c in s.split(",")]
                        return Point(coords[0], coords[1])
                    else:
                        coords = [float(c) for c in s.split()]
                        return Point(coords[0], coords[1])
                df = df.copy()
                df['geometry'] = df['geometry'].apply(parse_geom)
        return df

    input_df = ensure_geometry(input_df)

    logger.info("started extracting geometry from PT-JPL-SM input table")

    if "geometry" in input_df:
        # Convert Point objects to coordinate tuples for MultiPoint
        if hasattr(input_df.geometry.iloc[0], "x") and hasattr(input_df.geometry.iloc[0], "y"):
            coords = [(pt.x, pt.y) for pt in input_df.geometry]
            geometry = MultiPoint(coords, crs=WGS84)
        else:
            geometry = MultiPoint(input_df.geometry, crs=WGS84)
    elif "lat" in input_df and "lon" in input_df:
        lat = np.array(input_df.lat).astype(np.float64)
        lon = np.array(input_df.lon).astype(np.float64)
        geometry = MultiPoint(x=lon, y=lat, crs=WGS84)
    else:
        raise KeyError("Input DataFrame must contain either 'geometry' or both 'lat' and 'lon' columns.")

    logger.info("completed extracting geometry from PT-JPL-SM input table")

    logger.info("started extracting time from PT-JPL-SM input table")
    time_UTC = pd.to_datetime(input_df.time_UTC).tolist()
    logger.info("completed extracting time from PT-JPL-SM input table")
    
    results = BESS_JPL(
        geometry=geometry,
        time_UTC=time_UTC,
        ST_C=ST_C,
        albedo=albedo,
        NDVI=NDVI,
        Ta_C=Ta_C,
        RH=RH,
        elevation_km=elevation_km,
        C4_fraction_scale_factor=C4_fraction_scale_factor
    )

    output_df = input_df.copy()

    for key, value in results.items():
        output_df[key] = value

    return output_df
