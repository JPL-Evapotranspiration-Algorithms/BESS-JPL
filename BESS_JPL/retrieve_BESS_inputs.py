from typing import Union
from datetime import datetime
import logging
import numpy as np

import rasters as rt
from rasters import Raster, RasterGeometry

from check_distribution import check_distribution

from sun_angles import calculate_SZA_from_DOY_and_hour
from solar_apparent_time import calculate_solar_day_of_year, calculate_solar_hour_of_day
from koppengeiger import load_koppen_geiger
from gedi_canopy_height import load_canopy_height, GEDI_DOWNLOAD_DIRECTORY
from FLiESANN import FLiESANN
from GEOS5FP import GEOS5FP
from MODISCI import MODISCI
from NASADEM import NASADEMConnection

from .constants import *
from .exceptions import *
from .colors import *
from .C3_photosynthesis import *
from .C4_photosynthesis import *
from .canopy_energy_balance import *
from .canopy_longwave_radiation import *
from .canopy_shortwave_radiation import *
from .carbon_water_fluxes import *
from .FVC_from_NDVI import *
from .interpolate_C3_C4 import *
from .LAI_from_NDVI import *
from .load_C4_fraction import *
from .load_carbon_uptake_efficiency import *
from .load_kn import *
from .load_NDVI_minimum import *
from .load_NDVI_maximum import *
from .load_peakVCmax_C3 import *
from .load_peakVCmax_C4 import *
from .load_ball_berry_intercept_C3 import *
from .load_ball_berry_slope_C3 import *
from .load_ball_berry_slope_C4 import *
from .calculate_VCmax import *
from .meteorology import *
from .soil_energy_balance import *
from .retrieve_BESS_JPL_GEOS5FP_inputs import retrieve_BESS_JPL_GEOS5FP_inputs

logger = logging.getLogger(__name__)

def retrieve_BESS_inputs(ST_C: Union[Raster, np.ndarray],  # surface temperature in Celsius
        NDVI: Union[Raster, np.ndarray],  # NDVI
        albedo: Union[Raster, np.ndarray],  # surface albedo
        geometry: RasterGeometry = None,
        time_UTC: datetime = None,
        hour_of_day: np.ndarray = None,
        day_of_year: np.ndarray = None,
        GEOS5FP_connection: GEOS5FP = None,
        elevation_m: Union[Raster, np.ndarray] = None,  # elevation in meters
        Ta_C: Union[Raster, np.ndarray] = None,  # air temperature in Celsius
        RH: Union[Raster, np.ndarray] = None,  # relative humidity as a proportion
        NDVI_minimum: Union[Raster, np.ndarray] = None,  # minimum NDVI
        NDVI_maximum: Union[Raster, np.ndarray] = None,  # maximum NDVI
        PAR_albedo: Union[Raster, np.ndarray] = None, # surface albedo in visible wavelengths (initialized to surface albedo if left as None)
        NIR_albedo: Union[Raster, np.ndarray] = None, # surface albedo in near-infrared wavelengths (initialized to surface albedo if left as None)
        COT: Union[Raster, np.ndarray] = None,  # cloud optical thickness
        AOT: Union[Raster, np.ndarray] = None,  # aerosol optical thickness
        vapor_gccm: Union[Raster, np.ndarray] = None,  # water vapor in g/ccm
        ozone_cm: Union[Raster, np.ndarray] = None,  # ozone in cm
        KG_climate: Union[Raster, np.ndarray] = None,  # KG climate
        canopy_height_meters: Union[Raster, np.ndarray] = None,  # canopy height in meters
        Ca: Union[Raster, np.ndarray] = None,  # atmospheric CO2 concentration in ppm
        wind_speed_mps: Union[Raster, np.ndarray] = None,  # wind speed in meters per second
        SZA_deg: Union[Raster, np.ndarray] = None,  # solar zenith angle in degrees
        canopy_temperature_C: Union[Raster, np.ndarray] = None, # canopy temperature in Celsius (initialized to surface temperature if left as None)
        soil_temperature_C: Union[Raster, np.ndarray] = None, # soil temperature in Celsius (initialized to surface temperature if left as None)
        C4_fraction: Union[Raster, np.ndarray] = None,  # fraction of C4 plants
        carbon_uptake_efficiency: Union[Raster, np.ndarray] = None,  # intrinsic quantum efficiency for carbon uptake
        kn: np.ndarray = None,
        ball_berry_intercept_C3: np.ndarray = None,  # Ball-Berry intercept for C3 plants
        ball_berry_intercept_C4: Union[np.ndarray, float] = BALL_BERRY_INTERCEPT_C4, # Ball-Berry intercept for C4 plants
        ball_berry_slope_C3: np.ndarray = None,  # Ball-Berry slope for C3 plants
        ball_berry_slope_C4: np.ndarray = None,  # Ball-Berry slope for C4 plants
        peakVCmax_C3_μmolm2s1: np.ndarray = None,  # peak maximum carboxylation rate for C3 plants
        peakVCmax_C4_μmolm2s1: np.ndarray = None,  # peak maximum carboxylation rate for C4 plants
        CI: Union[Raster, np.ndarray] = None,
        C4_fraction_scale_factor: float = C4_FRACTION_SCALE_FACTOR,
        MODISCI_connection: MODISCI = None,
        NASADEM_connection: NASADEMConnection = None,
        resampling: str = RESAMPLING,
        GEDI_download_directory: str = GEDI_DOWNLOAD_DIRECTORY,
        offline_mode: bool = False) -> dict:
    results = {}

    if (day_of_year is None or hour_of_day is None) and time_UTC is not None and geometry is not None:
        day_of_year = calculate_solar_day_of_year(time_UTC=time_UTC, geometry=geometry)
        hour_of_day = calculate_solar_hour_of_day(time_UTC=time_UTC, geometry=geometry)

    if time_UTC is None and day_of_year is None and hour_of_day is None:
        raise ValueError("no time given between time_UTC, day_of_year, and hour_of_day")

    # calculate solar zenith angle if not provided
    if SZA_deg is None:
        SZA_deg = calculate_SZA_from_DOY_and_hour(geometry.lat, geometry.lon, day_of_year, hour_of_day)

    if isinstance(SZA_deg, np.ndarray):
        # If array contains string representations, convert them first
        if SZA_deg.dtype == object or SZA_deg.dtype.kind in ['U', 'S']:
            # Handle string arrays by converting each element
            # This handles cases like '[71.46303285]' or '71.46303285'
            SZA_deg = np.array([float(str(x).strip('[]')) for x in SZA_deg], dtype=np.float32)
        else:
            # cast SZA_deg numpy array to float32
            SZA_deg = SZA_deg.astype(np.float32)

    print(type(SZA_deg))
    print(SZA_deg.dtype if isinstance(SZA_deg, np.ndarray) else type(SZA_deg))

    check_distribution(SZA_deg, "SZA_deg")
    results["SZA_deg"] = SZA_deg


    if CI is None and geometry is not None:
        if offline_mode:
            raise MissingOfflineParameter("CI not provided in offline mode")
        
        if MODISCI_connection is None:
            MODISCI_connection = MODISCI()

        logger.info("loading clumping index")
        CI = MODISCI_connection.CI(geometry=geometry, resampling=resampling)

    check_distribution(CI, "CI")
    results["CI"] = CI

    if elevation_m is None and geometry is not None:
        if offline_mode:
            raise MissingOfflineParameter("elevation_m not provided in offline mode")
        
        if NASADEM_connection is None:
            NASADEM_connection = NASADEMConnection()

        logger.info("loading elevation")  
        elevation_m = NASADEM_connection.elevation_m(geometry=geometry)

    check_distribution(elevation_m, "elevation_m")
    results["elevation_m"] = elevation_m


    # load minimum NDVI if not provided
    if NDVI_minimum is None and geometry is not None:
        logger.info("loading minimum NDVI")
        NDVI_minimum = load_NDVI_minimum(geometry=geometry, resampling=resampling)

    check_distribution(NDVI_minimum, "NDVI_minimum")
    results["NDVI_minimum"] = NDVI_minimum

    # load maximum NDVI if not provided
    if NDVI_maximum is None and geometry is not None:
        logger.info("loading maximum NDVI")
        NDVI_maximum = load_NDVI_maximum(geometry=geometry, resampling=resampling)

    check_distribution(NDVI_maximum, "NDVI_maximum")
    results["NDVI_maximum"] = NDVI_maximum

    # load C4 fraction if not provided
    if C4_fraction is None:
        logger.info("loading C4 fraction")
        C4_fraction = load_C4_fraction(
            geometry=geometry, 
            resampling=resampling,
            scale_factor=C4_fraction_scale_factor
        )

    check_distribution(C4_fraction, "C4_fraction")
    results["C4_fraction"] = C4_fraction

    # load carbon uptake efficiency if not provided
    if carbon_uptake_efficiency is None:
        logger.info("loading carbon uptake efficiency")
        carbon_uptake_efficiency = load_carbon_uptake_efficiency(geometry=geometry, resampling=resampling)
    
    check_distribution(carbon_uptake_efficiency, "carbon_uptake_efficiency")
    results["carbon_uptake_efficiency"] = carbon_uptake_efficiency

    # load kn if not provided
    if kn is None:
        logger.info("loading kn")
        kn = load_kn(geometry=geometry, resampling=resampling)

    check_distribution(kn, "kn")
    results["kn"] = kn

    # load peak VC max for C3 plants if not provided
    if peakVCmax_C3_μmolm2s1 is None:
        logger.info("loading peak VCmax for C3 plants")
        peakVCmax_C3_μmolm2s1 = load_peakVCmax_C3(geometry=geometry, resampling=resampling)

    check_distribution(peakVCmax_C3_μmolm2s1, "peakVCmax_C3_μmolm2s1")
    results["peakVCmax_C3_μmolm2s1"] = peakVCmax_C3_μmolm2s1

    # load peak VC max for C4 plants if not provided
    if peakVCmax_C4_μmolm2s1 is None:
        logger.info("loading peak VCmax for C4 plants")
        peakVCmax_C4_μmolm2s1 = load_peakVCmax_C4(geometry=geometry, resampling=resampling)

    check_distribution(peakVCmax_C4_μmolm2s1, "peakVCmax_C4_μmolm2s1")
    results["peakVCmax_C4_μmolm2s1"] = peakVCmax_C4_μmolm2s1

    # load Ball-Berry slope for C3 plants if not provided
    if ball_berry_slope_C3 is None:
        logger.info("loading Ball-Berry slope for C3 plants")
        ball_berry_slope_C3 = load_ball_berry_slope_C3(geometry=geometry, resampling=resampling)
    
    check_distribution(ball_berry_slope_C3, "ball_berry_slope_C3")
    results["ball_berry_slope_C3"] = ball_berry_slope_C3

    # load Ball-Berry slope for C4 plants if not provided
    if ball_berry_slope_C4 is None:
        logger.info("loading Ball-Berry slope for C4 plants")
        ball_berry_slope_C4 = load_ball_berry_slope_C4(geometry=geometry, resampling=resampling)

    check_distribution(ball_berry_slope_C4, "ball_berry_slope_C4")
    results["ball_berry_slope_C4"] = ball_berry_slope_C4

    # load Ball-Berry intercept for C3 plants if not provided
    if ball_berry_intercept_C3 is None:
        logger.info("loading Ball-Berry intercept for C3 plants")
        ball_berry_intercept_C3 = load_ball_berry_intercept_C3(geometry=geometry, resampling=resampling)

    check_distribution(ball_berry_intercept_C3, "ball_berry_intercept_C3")
    results["ball_berry_intercept_C3"] = ball_berry_intercept_C3

    # load koppen geiger climate classification if not provided
    if KG_climate is None:
        logger.info("loading Koppen-Geiger climate classification")
        KG_climate = load_koppen_geiger(geometry=geometry)

    check_distribution(np.float32(KG_climate), "KG_climate")
    results["KG_climate"] = KG_climate

    # load canopy height in meters if not provided
    if canopy_height_meters is None:
        logger.info("loading canopy height")
        canopy_height_meters = load_canopy_height(
            geometry=geometry, 
            resampling=resampling,
            source_directory=GEDI_download_directory
        )

    # canopy height defaults to zero
    canopy_height_meters = np.where(np.isnan(canopy_height_meters), 0, canopy_height_meters)

    check_distribution(canopy_height_meters, "canopy_height_meters")
    results["canopy_height_meters"] = canopy_height_meters


    # Retrieve GEOS-5 FP inputs if not provided
    GEOS5FP_inputs = retrieve_BESS_JPL_GEOS5FP_inputs(
        time_UTC=time_UTC,
        geometry=geometry,
        albedo=albedo,
        GEOS5FP_connection=GEOS5FP_connection,
        Ta_C=Ta_C,
        RH=RH,
        COT=COT,
        AOT=AOT,
        vapor_gccm=vapor_gccm,
        ozone_cm=ozone_cm,
        PAR_albedo=PAR_albedo,
        NIR_albedo=NIR_albedo,
        Ca=Ca,
        wind_speed_mps=wind_speed_mps,
        resampling=resampling,
        offline_mode=offline_mode
    )
    
    # Merge GEOS-5 FP inputs into results dictionary
    results.update(GEOS5FP_inputs)

    # canopy temperature defaults to surface temperature
    if canopy_temperature_C is None:
        canopy_temperature_C = ST_C

    check_distribution(canopy_temperature_C, "canopy_temperature_C")
    results["canopy_temperature_C"] = canopy_temperature_C

    # soil temperature defaults to surface temperature
    if soil_temperature_C is None:
        soil_temperature_C = ST_C

    check_distribution(soil_temperature_C, "soil_temperature_C")
    results["soil_temperature_C"] = soil_temperature_C

    return results
