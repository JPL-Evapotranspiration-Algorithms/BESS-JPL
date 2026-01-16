from os.path import join, abspath, dirname
import pandas as pd
import numpy as np
from ECOv002_calval_tables import load_calval_table
from FLiESANN import process_FLiESANN_table, load_ECOv002_calval_FLiESANN_inputs
from .ECOv002_static_tower_BESS_inputs import load_ECOv002_static_tower_BESS_inputs
from .process_BESS_table import process_BESS_table
from .retrieve_BESS_JPL_GEOS5FP_inputs import retrieve_BESS_JPL_GEOS5FP_inputs

import logging
import warnings
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

# Configure GEOS5FP logging to be visible
geos5fp_logger = logging.getLogger('GEOS5FP')
geos5fp_logger.setLevel(logging.INFO)
if not geos5fp_logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    geos5fp_logger.addHandler(handler)

def generate_input_dataset():
    logger.info("Generating BESS-JPL input dataset from ECOv002 cal/val FLiESANN inputs")
    # calval_df = load_calval_table()
    inputs_df = load_ECOv002_calval_FLiESANN_inputs()

    # Ensure `time_UTC` is in datetime format
    inputs_df['time_UTC'] = pd.to_datetime(inputs_df['time_UTC'], errors='coerce')

    # Create a `date_UTC` column by extracting the date from `time_UTC`
    inputs_df['date_UTC'] = inputs_df['time_UTC'].dt.date

    # Convert any array-like values to scalars by extracting first element if needed
    def extract_scalar(x):
        if isinstance(x, pd.DataFrame):
            # Handle DataFrame - extract first value
            return x.iloc[0, 0] if not x.empty else x
        elif isinstance(x, pd.Series):
            # Handle Series - extract first value
            return x.iloc[0] if len(x) > 0 else x
        elif isinstance(x, np.ndarray):
            # Handle numpy arrays
            return x.item() if x.size == 1 else x.flat[0] if x.size > 0 else x
        elif isinstance(x, list):
            # Handle lists
            return x[0] if len(x) > 0 else x
        else:
            # Return as-is for scalars
            return x

    # Apply extraction to all columns
    for col in inputs_df.columns:
        inputs_df[col] = inputs_df[col].apply(extract_scalar)

    # Load static tower BESS inputs
    static_inputs_df = load_ECOv002_static_tower_BESS_inputs()

    # Merge FLiESANN outputs with static BESS inputs on Site ID
    # FLiESANN outputs contain time-varying atmospheric and radiation inputs
    # Static inputs contain vegetation parameters
    inputs_df = inputs_df.merge(
        static_inputs_df,
        left_on='ID',
        right_on='ID',
        how='left',
        suffixes=('', '_static')
    )
    
    

    # Remove duplicate columns from the merge (keep non-static versions)
    duplicate_cols = [col for col in inputs_df.columns if col.endswith('_static')]
    inputs_df = inputs_df.drop(columns=duplicate_cols)

    # Extract required parameters from inputs_df for retrieve_BESS_inputs
    ST_C = np.array(inputs_df.ST_C).astype(np.float64)
    NDVI = np.array(inputs_df.NDVI).astype(np.float64)
    NDVI = np.where(NDVI > 0.06, NDVI, np.nan).astype(np.float64)
    albedo = np.array(inputs_df.albedo).astype(np.float64)
    
    # Extract time and geometry
    from rasters import Point
    from solar_apparent_time import calculate_solar_day_of_year, calculate_solar_hour_of_day
    from geopandas import GeoSeries
    from shapely.geometry import Point as ShapelyPoint
    
    # Handle geometry construction
    if "geometry" in inputs_df:
        if isinstance(inputs_df.geometry.iloc[0], str):
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
            inputs_df = inputs_df.copy()
            inputs_df['geometry'] = inputs_df['geometry'].apply(parse_geom)
        geometry = [Point(pt.x, pt.y) for pt in inputs_df.geometry]
    elif "lat" in inputs_df and "lon" in inputs_df:
        lat = np.array(inputs_df.lat).astype(np.float64)
        lon = np.array(inputs_df.lon).astype(np.float64)
        geometry = [Point(lon[i], lat[i]) for i in range(len(lat))]
    else:
        raise KeyError("Input DataFrame must contain either 'geometry' or both 'lat' and 'lon' columns.")
    
    # Extract time
    time_UTC_list = pd.to_datetime(inputs_df.time_UTC).tolist()
    
    # Calculate solar time
    day_of_year_list = []
    hour_of_day_list = []
    
    for i, (time_utc, geom) in enumerate(zip(time_UTC_list, geometry)):
        shapely_point = ShapelyPoint(geom.x, geom.y)
        geoseries = GeoSeries([shapely_point])
        doy = calculate_solar_day_of_year(time_UTC=time_utc, geometry=geoseries)
        hod = calculate_solar_hour_of_day(time_UTC=time_utc, geometry=geoseries)
        doy_scalar = doy[0] if hasattr(doy, '__getitem__') else doy
        hod_scalar = hod[0] if hasattr(hod, '__getitem__') else hod
        day_of_year_list.append(doy_scalar)
        hour_of_day_list.append(hod_scalar)
    
    day_of_year = np.array(day_of_year_list)
    hour_of_day = np.array(hour_of_day_list)
    
    # Keep geometry as list of Points - do NOT convert to MultiPoint
    # This allows proper matching of each point with its corresponding time
    time_UTC = time_UTC_list
    
    # Extract optional inputs if present
    Ta_C = np.array(inputs_df.Ta_C).astype(np.float64) if "Ta_C" in inputs_df else (np.array(inputs_df.Ta).astype(np.float64) if "Ta" in inputs_df else None)
    RH = np.array(inputs_df.RH).astype(np.float64) if "RH" in inputs_df else None
    elevation_m = np.array(inputs_df.elevation_m).astype(np.float64) if "elevation_m" in inputs_df else (np.array(inputs_df.elevation_km).astype(np.float64) * 1000 if "elevation_km" in inputs_df else None)
    COT = np.array(inputs_df.COT).astype(np.float64) if "COT" in inputs_df else None
    AOT = np.array(inputs_df.AOT).astype(np.float64) if "AOT" in inputs_df else None
    vapor_gccm = np.array(inputs_df.vapor_gccm).astype(np.float64) if "vapor_gccm" in inputs_df else None
    ozone_cm = np.array(inputs_df.ozone_cm).astype(np.float64) if "ozone_cm" in inputs_df else None
    PAR_albedo = np.array(inputs_df.PAR_albedo).astype(np.float64) if "PAR_albedo" in inputs_df else None
    NIR_albedo = np.array(inputs_df.NIR_albedo).astype(np.float64) if "NIR_albedo" in inputs_df else None
    Ca = np.array(inputs_df.Ca).astype(np.float64) if "Ca" in inputs_df else None
    wind_speed_mps = np.array(inputs_df.wind_speed_mps).astype(np.float64) if "wind_speed_mps" in inputs_df else None
    NDVI_minimum = np.array(inputs_df.NDVI_minimum).astype(np.float64) if "NDVI_minimum" in inputs_df else None
    NDVI_maximum = np.array(inputs_df.NDVI_maximum).astype(np.float64) if "NDVI_maximum" in inputs_df else None
    C4_fraction = np.array(inputs_df.C4_fraction).astype(np.float64) if "C4_fraction" in inputs_df else None
    carbon_uptake_efficiency = np.array(inputs_df.carbon_uptake_efficiency).astype(np.float64) if "carbon_uptake_efficiency" in inputs_df else None
    kn = np.array(inputs_df.kn).astype(np.float64) if "kn" in inputs_df else None
    peakVCmax_C3 = np.array(inputs_df.peakVCmax_C3).astype(np.float64) if "peakVCmax_C3" in inputs_df else None
    peakVCmax_C4 = np.array(inputs_df.peakVCmax_C4).astype(np.float64) if "peakVCmax_C4" in inputs_df else None
    ball_berry_slope_C3 = np.array(inputs_df.ball_berry_slope_C3).astype(np.float64) if "ball_berry_slope_C3" in inputs_df else None
    ball_berry_slope_C4 = np.array(inputs_df.ball_berry_slope_C4).astype(np.float64) if "ball_berry_slope_C4" in inputs_df else None
    ball_berry_intercept_C3 = np.array(inputs_df.ball_berry_intercept_C3).astype(np.float64) if "ball_berry_intercept_C3" in inputs_df else None
    KG_climate = np.array(inputs_df.KG_climate) if "KG_climate" in inputs_df else None
    CI = np.array(inputs_df.CI).astype(np.float64) if "CI" in inputs_df else None
    canopy_height_meters = np.array(inputs_df.canopy_height_meters).astype(np.float64) if "canopy_height_meters" in inputs_df else None

    logger.info("Retrieving GEOS-5 FP meteorological inputs")
    logger.info(f"Calling retrieve_BESS_JPL_GEOS5FP_inputs with {len(time_UTC)} time points and {len(geometry)} geometry points")
    
    # Retrieve only GEOS-5 FP meteorological inputs (vegetation params already in inputs_df)
    # Pass geometry as list of Points to match each time with its corresponding location
    GEOS5FP_inputs_dict = retrieve_BESS_JPL_GEOS5FP_inputs(
        time_UTC=time_UTC,
        geometry=geometry,
        albedo=albedo,
        Ta_C=Ta_C,
        RH=RH,
        COT=COT,
        AOT=AOT,
        vapor_gccm=vapor_gccm,
        ozone_cm=ozone_cm,
        PAR_albedo=PAR_albedo,
        NIR_albedo=NIR_albedo,
        Ca=Ca,
        wind_speed_mps=wind_speed_mps
    )
    
    logger.info("Completed retrieving GEOS-5 FP meteorological inputs")
    
    # Create complete inputs dataframe by starting with original inputs_df and updating with retrieved values
    complete_inputs_df = inputs_df.copy()
    
    # Add primary inputs
    complete_inputs_df['ST_C'] = ST_C
    complete_inputs_df['NDVI'] = NDVI
    complete_inputs_df['albedo'] = albedo
    complete_inputs_df['time_UTC'] = time_UTC_list
    complete_inputs_df['day_of_year'] = day_of_year
    complete_inputs_df['hour_of_day'] = hour_of_day
    
    # Add geometry as lat/lon if not already present
    if 'lat' not in complete_inputs_df:
        complete_inputs_df['lat'] = [pt.y for pt in geometry]
    if 'lon' not in complete_inputs_df:
        complete_inputs_df['lon'] = [pt.x for pt in geometry]
    
    # Add all retrieved GEOS5FP inputs to complete_inputs_df
    for key, value in GEOS5FP_inputs_dict.items():
        if hasattr(value, '__len__') and not isinstance(value, str):
            try:
                complete_inputs_df[key] = value
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping assignment of key '{key}' to inputs DataFrame: {e}")
                continue
        elif isinstance(value, (int, float, np.number)):
            complete_inputs_df[key] = value

    logger.info("Processing BESS model to generate outputs")
    
    # Process with BESS-JPL model to get outputs
    outputs_df = process_BESS_table(inputs_df)

    inputs_filename = join(abspath(dirname(__file__)), "ECOv002-cal-val-BESS-JPL-inputs.csv")
    outputs_filename = join(abspath(dirname(__file__)), "ECOv002-cal-val-BESS-JPL-outputs.csv")

    # Save the complete input dataset to a CSV file
    complete_inputs_df.to_csv(inputs_filename, index=False)

    # Save the processed results to a CSV file
    outputs_df.to_csv(outputs_filename, index=False)

    logger.info(f"Processed {len(outputs_df)} records from the full cal/val dataset")
    logger.info(f"Complete input dataset saved to: {inputs_filename}")
    logger.info(f"  - Contains {len(complete_inputs_df.columns)} input columns")
    logger.info(f"Output dataset saved to: {outputs_filename}")
    logger.info(f"  - Contains {len(outputs_df.columns)} total columns")
    
    return outputs_df
