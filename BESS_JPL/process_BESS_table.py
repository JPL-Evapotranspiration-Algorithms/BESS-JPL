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
        elevation_m = elevation_km * 1000
    else:
        elevation_km = None
        elevation_m = None

    if "NDVI_minimum" in input_df:
        NDVI_minimum = np.array(input_df.NDVI_minimum).astype(np.float64)
    else:
        NDVI_minimum = None

    if "NDVI_maximum" in input_df:
        NDVI_maximum = np.array(input_df.NDVI_maximum).astype(np.float64).astype(np.float64)
    else:
        NDVI_maximum = None
    
    if "C4_fraction" in input_df:
        C4_fraction = np.array(input_df.C4_fraction).astype(np.float64)
    else:
        C4_fraction = None

    if "carbon_uptake_efficiency" in input_df:
        carbon_uptake_efficiency = np.array(input_df.carbon_uptake_efficiency).astype(np.float64)
    else:
        carbon_uptake_efficiency = None

    if "kn" in input_df:
        kn = np.array(input_df.kn).astype(np.float64)
    else:
        kn = None
    
    if "peakVCmax_C3" in input_df:
        peakVCmax_C3 = np.array(input_df.peakVCmax_C3).astype(np.float64)
    else:
        peakVCmax_C3 = None

    if "peakVCmax_C4" in input_df:
        peakVCmax_C4 = np.array(input_df.peakVCmax_C4).astype(np.float64)
    else:
        peakVCmax_C4 = None
    
    if "ball_berry_slope_C3" in input_df:
        ball_berry_slope_C3 = np.array(input_df.ball_berry_slope_C3).astype(np.float64)
    else:
        ball_berry_slope_C3 = None
    
    if "ball_berry_slope_C4" in input_df:
        ball_berry_slope_C4 = np.array(input_df.ball_berry_slope_C4).astype(np.float64)
    else:
        ball_berry_slope_C4 = None

    if "ball_berry_intercept_C3" in input_df:
        ball_berry_intercept_C3 = np.array(input_df.ball_berry_intercept_C3).astype(np.float64)
    else:
        ball_berry_intercept_C3 = None

    if "KG_climate" in input_df:
        KG_climate = np.array(input_df.KG_climate)
    else:
        KG_climate = None

    if "CI" in input_df:
        CI = np.array(input_df.CI).astype(np.float64)
    else:
        CI = None

    if "canopy_height_meters" in input_df:
        canopy_height_meters = np.array(input_df.canopy_height_meters).astype(np.float64)
    else:
        canopy_height_meters = None

    if "COT" in input_df:
        COT = np.array(input_df.COT).astype(np.float64)
    else:
        COT = None

    if "AOT" in input_df:
        AOT = np.array(input_df.AOT).astype(np.float64)
    else:
        AOT = None

    if "Ca" in input_df:
        Ca = np.array(input_df.Ca).astype(np.float64)
    else:
        Ca = None

    if "wind_speed_mps" in input_df:
        wind_speed_mps = np.array(input_df.wind_speed_mps).astype(np.float64)
    else:
        wind_speed_mps = None

    if "vapor_gccm" in input_df:
        vapor_gccm = np.array(input_df.vapor_gccm).astype(np.float64)
    else:
        vapor_gccm = None
    
    if "ozone_cm" in input_df:
        ozone_cm = np.array(input_df.ozone_cm).astype(np.float64)
    else:
        ozone_cm = None

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
        elevation_m=elevation_m,
        NDVI_minimum=NDVI_minimum,
        NDVI_maximum=NDVI_maximum,
        C4_fraction=C4_fraction,
        carbon_uptake_efficiency=carbon_uptake_efficiency,
        kn=kn,
        peakVCmax_C3_μmolm2s1=peakVCmax_C3,
        peakVCmax_C4_μmolm2s1=peakVCmax_C4,
        ball_berry_slope_C3=ball_berry_slope_C3,
        ball_berry_slope_C4=ball_berry_slope_C4,
        ball_berry_intercept_C3=ball_berry_intercept_C3,
        KG_climate=KG_climate,
        CI=CI,
        canopy_height_meters=canopy_height_meters,
        COT=COT,
        AOT=AOT,
        Ca=Ca,
        wind_speed_mps=COT * 0 + 7.4,
        vapor_gccm=vapor_gccm,
        ozone_cm=ozone_cm,
        albedo_visible=albedo,
        albedo_NIR=albedo,
        C4_fraction_scale_factor=C4_fraction_scale_factor
    )

    output_df = input_df.copy()

    for key, value in results.items():
        output_df[key] = value

    return output_df
