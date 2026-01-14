import logging

import numpy as np
import pandas as pd
import rasters as rt
from dateutil import parser
from pandas import DataFrame
from rasters import MultiPoint, WGS84
from shapely.geometry import Point
from GEOS5FP import GEOS5FP
from MODISCI import MODISCI
from NASADEM import NASADEMConnection
from gedi_canopy_height import GEDI_DOWNLOAD_DIRECTORY
from .retrieve_BESS_inputs import retrieve_BESS_inputs
from .constants import C4_FRACTION_SCALE_FACTOR, RESAMPLING

logger = logging.getLogger(__name__)

def generate_BESS_inputs_table(
        input_df: DataFrame,
        GEOS5FP_connection: GEOS5FP = None,
        MODISCI_connection: MODISCI = None,
        NASADEM_connection: NASADEMConnection = None,
        C4_fraction_scale_factor: float = C4_FRACTION_SCALE_FACTOR,
        resampling: str = RESAMPLING,
        GEDI_download_directory: str = GEDI_DOWNLOAD_DIRECTORY) -> DataFrame:
    """
    Generates a DataFrame of BESS-JPL inputs by retrieving atmospheric, vegetation, and static data.
    
    This is a simple wrapper around retrieve_BESS_inputs that handles DataFrame
    input/output and geometry parsing.

    Parameters:
    input_df (pd.DataFrame): A DataFrame containing the following columns:
        - time_UTC (str or datetime): Time in UTC.
        - geometry (str or shapely.geometry.Point) or (lat, lon): Spatial coordinates. 
          If "geometry" is a string, it should be in WKT format (e.g., "POINT (lon lat)").
        - ST_C (float): Surface temperature in Celsius.
        - NDVI (float): Normalized Difference Vegetation Index.
        - albedo (float): Surface albedo.
        - Ta_C (float, optional): Air temperature in Celsius.
        - RH (float, optional): Relative humidity as a proportion.
        - elevation_m (float, optional): Elevation in meters.
        - NDVI_minimum (float, optional): Minimum NDVI.
        - NDVI_maximum (float, optional): Maximum NDVI.
        - COT (float, optional): Cloud optical thickness.
        - AOT (float, optional): Aerosol optical thickness.
        - vapor_gccm (float, optional): Water vapor in grams per cubic centimeter.
        - ozone_cm (float, optional): Ozone concentration in centimeters.
        - KG or KG_climate (str, optional): Köppen-Geiger climate classification.
        - canopy_height_meters (float, optional): Canopy height in meters.
        - Ca (float, optional): Atmospheric CO2 concentration in ppm.
        - wind_speed_mps (float, optional): Wind speed in meters per second.
        - SZA_deg (float, optional): Solar zenith angle in degrees.
        - canopy_temperature_C (float, optional): Canopy temperature in Celsius.
        - soil_temperature_C (float, optional): Soil temperature in Celsius.
        - C4_fraction (float, optional): Fraction of C4 plants.
        - carbon_uptake_efficiency (float, optional): Intrinsic quantum efficiency for carbon uptake.
        - kn (float, optional): Nitrogen decay coefficient.
        - ball_berry_intercept_C3 (float, optional): Ball-Berry intercept for C3 plants.
        - ball_berry_slope_C3 (float, optional): Ball-Berry slope for C3 plants.
        - ball_berry_slope_C4 (float, optional): Ball-Berry slope for C4 plants.
        - peakVCmax_C3_μmolm2s1 (float, optional): Peak maximum carboxylation rate for C3 plants.
        - peakVCmax_C4_μmolm2s1 (float, optional): Peak maximum carboxylation rate for C4 plants.
        - CI (float, optional): Clumping index.
        - PAR_albedo (float, optional): Surface albedo in visible wavelengths.
        - NIR_albedo (float, optional): Surface albedo in near-infrared wavelengths.
        - day_of_year (float, optional): Day of year.
        - hour_of_day (float, optional): Hour of day.
    GEOS5FP_connection (GEOS5FP, optional): Connection object for GEOS-5 FP data.
    MODISCI_connection (MODISCI, optional): Connection object for MODIS clumping index data.
    NASADEM_connection (NASADEMConnection, optional): Connection object for NASADEM data.
    C4_fraction_scale_factor (float, optional): Scale factor for C4 fraction adjustment.
    resampling (str, optional): Resampling method for data processing.
    GEDI_download_directory (str, optional): Directory for GEDI canopy height data downloads.

    Returns:
    pd.DataFrame: A DataFrame with the same structure as the input, but with additional columns
        containing all BESS-JPL input variables that were retrieved or calculated.

    Raises:
    KeyError: If required columns ("geometry" or "lat" and "lon", "time_UTC", "ST_C", "NDVI", "albedo") are missing.
    """
    def ensure_geometry(row):
        if "geometry" in row:
            if isinstance(row.geometry, str):
                s = row.geometry.strip()
                if s.startswith("POINT"):
                    coords = s.replace("POINT", "").replace("(", "").replace(")", "").strip().split()
                    return Point(float(coords[0]), float(coords[1]))
                elif "," in s:
                    coords = [float(c) for c in s.split(",")]
                    return Point(coords[0], coords[1])
                else:
                    coords = [float(c) for c in s.split()]
                    return Point(coords[0], coords[1])
        return row.geometry

    logger.info("started generating BESS inputs table")

    # Ensure geometry column is properly formatted
    input_df = input_df.copy()
    input_df["geometry"] = input_df.apply(ensure_geometry, axis=1)

    # Prepare output DataFrame
    output_df = input_df.copy()
    
    # Prepare geometries
    if "geometry" in input_df.columns:
        geometries = MultiPoint([(geom.x, geom.y) for geom in input_df.geometry], crs=WGS84)
    elif "lat" in input_df.columns and "lon" in input_df.columns:
        geometries = MultiPoint([(lon, lat) for lon, lat in zip(input_df.lon, input_df.lat)], crs=WGS84)
    else:
        raise KeyError("Input DataFrame must contain either 'geometry' or both 'lat' and 'lon' columns.")
    
    # Convert time column to datetime
    times_UTC = pd.to_datetime(input_df.time_UTC)
    
    logger.info(f"generating inputs for {len(input_df)} rows")

    # Helper function to get column values or None if column doesn't exist
    def get_column_or_none(df, col_name, default_col_name=None):
        if col_name in df.columns:
            return df[col_name].values
        elif default_col_name and default_col_name in df.columns:
            return df[default_col_name].values
        else:
            return None

    # Retrieve all inputs at once using vectorized retrieve_BESS_inputs call
    BESS_inputs = retrieve_BESS_inputs(
        ST_C=get_column_or_none(input_df, "ST_C"),
        NDVI=get_column_or_none(input_df, "NDVI"),
        albedo=get_column_or_none(input_df, "albedo"),
        geometry=geometries,
        time_UTC=times_UTC,
        hour_of_day=get_column_or_none(input_df, "hour_of_day"),
        day_of_year=get_column_or_none(input_df, "day_of_year"),
        GEOS5FP_connection=GEOS5FP_connection,
        elevation_m=get_column_or_none(input_df, "elevation_m"),
        Ta_C=get_column_or_none(input_df, "Ta_C"),
        RH=get_column_or_none(input_df, "RH"),
        NDVI_minimum=get_column_or_none(input_df, "NDVI_minimum"),
        NDVI_maximum=get_column_or_none(input_df, "NDVI_maximum"),
        PAR_albedo=get_column_or_none(input_df, "PAR_albedo"),
        NIR_albedo=get_column_or_none(input_df, "NIR_albedo"),
        COT=get_column_or_none(input_df, "COT"),
        AOT=get_column_or_none(input_df, "AOT"),
        vapor_gccm=get_column_or_none(input_df, "vapor_gccm"),
        ozone_cm=get_column_or_none(input_df, "ozone_cm"),
        KG_climate=get_column_or_none(input_df, "KG_climate", "KG"),
        canopy_height_meters=get_column_or_none(input_df, "canopy_height_meters"),
        Ca=get_column_or_none(input_df, "Ca"),
        wind_speed_mps=get_column_or_none(input_df, "wind_speed_mps"),
        SZA_deg=get_column_or_none(input_df, "SZA_deg", "SZA"),
        canopy_temperature_C=get_column_or_none(input_df, "canopy_temperature_C"),
        soil_temperature_C=get_column_or_none(input_df, "soil_temperature_C"),
        C4_fraction=get_column_or_none(input_df, "C4_fraction"),
        carbon_uptake_efficiency=get_column_or_none(input_df, "carbon_uptake_efficiency"),
        kn=get_column_or_none(input_df, "kn"),
        ball_berry_intercept_C3=get_column_or_none(input_df, "ball_berry_intercept_C3"),
        ball_berry_slope_C3=get_column_or_none(input_df, "ball_berry_slope_C3"),
        ball_berry_slope_C4=get_column_or_none(input_df, "ball_berry_slope_C4"),
        peakVCmax_C3_μmolm2s1=get_column_or_none(input_df, "peakVCmax_C3_μmolm2s1", "peakVCmax_C3"),
        peakVCmax_C4_μmolm2s1=get_column_or_none(input_df, "peakVCmax_C4_μmolm2s1", "peakVCmax_C4"),
        CI=get_column_or_none(input_df, "CI"),
        C4_fraction_scale_factor=C4_fraction_scale_factor,
        MODISCI_connection=MODISCI_connection,
        NASADEM_connection=NASADEM_connection,
        resampling=resampling,
        GEDI_download_directory=GEDI_download_directory
    )

    # Add retrieved inputs to the output DataFrame
    for key, values in BESS_inputs.items():
        # Skip values with mismatched lengths
        if hasattr(values, '__len__') and not isinstance(values, str):
            if len(values) != len(output_df):
                logger.warning(f"Skipping {key}: length mismatch ({len(values)} != {len(output_df)})")
                continue
        logger.info(f"Adding {key} to output DataFrame (type: {type(values)}, length: {len(values) if hasattr(values, '__len__') and not isinstance(values, str) else 'N/A'})")
        output_df[key] = values

    logger.info("completed generating BESS inputs table")

    return output_df
