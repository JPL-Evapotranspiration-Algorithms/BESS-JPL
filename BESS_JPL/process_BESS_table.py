import logging

import numpy as np
import pandas as pd
import rasters as rt
from dateutil import parser
from pandas import DataFrame
from pytictoc import TicToc

# Import functions for calculating solar time
from solar_apparent_time import calculate_solar_day_of_year, calculate_solar_hour_of_day
from geopandas import GeoSeries
from shapely.geometry import Point as ShapelyPoint

from rasters import MultiPoint

from GEOS5FP import GEOS5FP

from .constants import *
from .model import BESS_JPL
from .retrieve_BESS_JPL_GEOS5FP_inputs import retrieve_BESS_JPL_GEOS5FP_inputs

logger = logging.getLogger(__name__)

def _is_notebook() -> bool:
    """Check if code is running in a Jupyter notebook environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except (ImportError, NameError):
        return False      # Probably standard Python interpreter

def process_BESS_table(
        input_df: DataFrame,
        GEOS5FP_connection: GEOS5FP = None,
        C4_fraction_scale_factor: float = C4_FRACTION_SCALE_FACTOR,
        verbose: bool = None,
        offline_mode: bool = False) -> DataFrame:
    # Set verbose default based on environment if not explicitly provided
    if verbose is None:
        verbose = not _is_notebook()
    
    timer = TicToc()
    
    ST_C = np.array(input_df.ST_C).astype(np.float64)
    NDVI = np.array(input_df.NDVI).astype(np.float64)

    NDVI = np.where(NDVI > 0.06, NDVI, np.nan).astype(np.float64)

    albedo = np.array(input_df.albedo).astype(np.float64)
    
    if "PAR_albedo" in input_df:
        PAR_albedo = np.array(input_df.PAR_albedo).astype(np.float64)
    else:
        PAR_albedo = None
        
    if "NIR_albedo" in input_df:
        NIR_albedo = np.array(input_df.NIR_albedo).astype(np.float64)
    else:
        NIR_albedo = None
    
    if "Ta_C" in input_df:
        Ta_C = np.array(input_df.Ta_C).astype(np.float64)
    elif "Ta" in input_df:
        Ta_C = np.array(input_df.Ta).astype(np.float64)

    RH = np.array(input_df.RH).astype(np.float64)

    if "elevation_m" in input_df:
        elevation_m = np.array(input_df.elevation_m).astype(np.float64)
        elevation_km = elevation_m / 1000
    elif "elevation_km" in input_df:
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
        # Default to 400 ppm when Ca is not provided
        Ca = np.full_like(ST_C, 400.0, dtype=np.float64)

    if "wind_speed_mps" in input_df:
        wind_speed_mps = np.array(input_df.wind_speed_mps).astype(np.float64)
        # Apply default wind speed of 7.4 m/s when wind speed is 0 or very low
        # to avoid numerical instability in aerodynamic resistance calculations
        # wind_speed_mps = np.where(wind_speed_mps < 0.1, 7.4, wind_speed_mps)
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

    # Handle temperature defaults
    if "canopy_temperature_C" in input_df:
        canopy_temperature_C = np.array(input_df.canopy_temperature_C).astype(np.float64)
    else:
        # Default to surface temperature when canopy temperature is not provided
        canopy_temperature_C = ST_C.copy()

    if "soil_temperature_C" in input_df:
        soil_temperature_C = np.array(input_df.soil_temperature_C).astype(np.float64)
    else:
        # Default to surface temperature when soil temperature is not provided
        soil_temperature_C = ST_C.copy()

    # --- Handle geometry and time columns ---
    import pandas as pd
    from rasters import MultiPoint, WGS84
    # from shapely.geometry import Point
    from rasters import Point

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

    logger.info("started extracting geometry from BESS input table")
    timer.tic()

    if "geometry" in input_df:
        # Convert Point objects to a list of Points
        if hasattr(input_df.geometry.iloc[0], "x") and hasattr(input_df.geometry.iloc[0], "y"):
            geometry = [Point(pt.x, pt.y) for pt in input_df.geometry]
        else:
            geometry = [Point(pt) for pt in input_df.geometry]
    elif "lat" in input_df and "lon" in input_df:
        lat = np.array(input_df.lat).astype(np.float64)
        lon = np.array(input_df.lon).astype(np.float64)
        geometry = [Point(lon[i], lat[i]) for i in range(len(lat))]
    else:
        raise KeyError("Input DataFrame must contain either 'geometry' or both 'lat' and 'lon' columns.")

    elapsed = timer.tocvalue()
    logger.info(f"completed extracting geometry from BESS input table ({elapsed:.2f} seconds)")

    logger.info("started extracting time from BESS input table")
    timer.tic()
    time_UTC_list = pd.to_datetime(input_df.time_UTC, format='ISO8601').tolist()
    elapsed = timer.tocvalue()
    logger.info(f"completed extracting time from BESS input table ({elapsed:.2f} seconds)")
    
    logger.info("started calculating day of year and hour of day")
    timer.tic()
    
    # Create GeoSeries once for all geometry
    geoseries_all = GeoSeries([ShapelyPoint(geom.x, geom.y) for geom in geometry])
    
    # Call functions once with full arrays - they should handle broadcasting
    day_of_year = np.asarray(calculate_solar_day_of_year(time_UTC=time_UTC_list, geometry=geoseries_all))
    hour_of_day = np.asarray(calculate_solar_hour_of_day(time_UTC=time_UTC_list, geometry=geoseries_all))
    
    elapsed = timer.tocvalue()
    logger.info(f"completed calculating day of year and hour of day ({elapsed:.2f} seconds)")
    
    # Convert list of rasters.Point to MultiPoint for compatibility with FLiESANN and other functions
    
    logger.info("started extracting geometry")
    timer.tic()
    
    # Extract (x, y) tuples from rasters.Point objects
    point_tuples = [(pt.x, pt.y) for pt in geometry]
    geometry_multipoint = MultiPoint(point_tuples)
    time_UTC = time_UTC_list
    
    elapsed = timer.tocvalue()
    logger.info(f"completed extracting geometry ({elapsed:.2f} seconds)")
        
    logger.info("started retrieving BESS inputs")
    timer.tic()

    BESS_GEOS5FP_inputs = retrieve_BESS_JPL_GEOS5FP_inputs(
        time_UTC=time_UTC,
        geometry=geometry_multipoint,
        albedo=albedo,
        GEOS5FP_connection=GEOS5FP_connection,
        Ta_C=Ta_C,
        RH=RH,
        COT=COT,
        AOT=AOT,
        vapor_gccm=vapor_gccm,
        ozone_cm=ozone_cm,
        PAR_albedo=PAR_albedo,
        NIR_albedo=albedo,
        Ca=Ca,
        wind_speed_mps=wind_speed_mps,
        verbose=verbose,
        offline_mode=offline_mode
    )
    
    elapsed = timer.tocvalue()
    logger.info(f"finished retrieving BESS inputs ({elapsed:.2f} seconds)")
    
    albedo = BESS_GEOS5FP_inputs['albedo']
    Ta_C = BESS_GEOS5FP_inputs['Ta_C']
    RH = BESS_GEOS5FP_inputs['RH']
    COT = BESS_GEOS5FP_inputs['COT']
    AOT = BESS_GEOS5FP_inputs['AOT']
    vapor_gccm = BESS_GEOS5FP_inputs['vapor_gccm']
    ozone_cm = BESS_GEOS5FP_inputs['ozone_cm']
    PAR_albedo = BESS_GEOS5FP_inputs['PAR_albedo']
    NIR_albedo = BESS_GEOS5FP_inputs['NIR_albedo']
    Ca = BESS_GEOS5FP_inputs['Ca']
    wind_speed_mps = BESS_GEOS5FP_inputs['wind_speed_mps']

    results = BESS_JPL(
        geometry=geometry_multipoint,
        time_UTC=time_UTC,
        day_of_year=day_of_year,
        hour_of_day=hour_of_day,
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
        wind_speed_mps=wind_speed_mps,
        vapor_gccm=vapor_gccm,
        ozone_cm=ozone_cm,
        PAR_albedo=albedo,
        NIR_albedo=albedo,
        canopy_temperature_C=canopy_temperature_C,
        soil_temperature_C=soil_temperature_C,
        C4_fraction_scale_factor=C4_fraction_scale_factor,
        GEOS5FP_connection=GEOS5FP_connection,
        offline_mode=offline_mode
    )

    output_df = input_df.copy()

    # Update or add columns from results, overwriting existing columns to avoid duplicates
    for key, value in results.items():
        # Skip non-array-like objects (e.g., MultiPoint geometry)
        if hasattr(value, '__len__') and not isinstance(value, (str, MultiPoint)):
            try:
                output_df[key] = value  # Direct assignment overwrites existing columns
            except (ValueError, TypeError):
                # Skip values that can't be assigned to DataFrame
                logger.warning(f"Skipping assignment of key '{key}' to output DataFrame")
                continue
        elif isinstance(value, (int, float, np.number)):
            # Handle scalar values
            output_df[key] = value

    return output_df
