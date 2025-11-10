from typing import Union
from datetime import datetime
import logging
import numpy as np
from pytictoc import TicToc

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

from daylight_evapotranspiration import daylight_ET_from_instantaneous_LE, daylight_ET_from_daylight_LE

from .constants import *
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

logger = logging.getLogger(__name__)

def BESS_JPL(
        ST_C: Union[Raster, np.ndarray],  # surface temperature in Celsius
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
        SWin_Wm2: Union[Raster, np.ndarray] = None,  # incoming shortwave radiation in W/m^2
        PAR_diffuse_Wm2: Union[Raster, np.ndarray] = None,  # diffuse visible radiation in W/m^2
        PAR_direct_Wm2: Union[Raster, np.ndarray] = None,  # direct visible radiation in W/m^2
        NIR_diffuse_Wm2: Union[Raster, np.ndarray] = None,  # diffuse near-infrared radiation in W/m^2
        NIR_direct_Wm2: Union[Raster, np.ndarray] = None,  # direct near-infrared radiation in W/m^2
        UV_Wm2: Union[Raster, np.ndarray] = None,  # incoming ultraviolet radiation in W/m^2
        albedo_visible: Union[Raster, np.ndarray] = None, # surface albedo in visible wavelengths (initialized to surface albedo if left as None)
        albedo_NIR: Union[Raster, np.ndarray] = None, # surface albedo in near-infrared wavelengths (initialized to surface albedo if left as None)
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
        upscale_to_daylight: bool = UPSCALE_TO_DAYLIGHT,
        resampling: str = RESAMPLING,
        GEDI_download_directory: str = GEDI_DOWNLOAD_DIRECTORY):  # clumping index
    """
    Breathing Earth System Simulator (BESS) model for estimating gross primary productivity (GPP)
    and evapotranspiration (ET) using coupled atmospheric and canopy radiative transfer processes.
    
    The BESS model couples atmospheric and canopy radiative transfer processes with photosynthesis,
    stomatal conductance, and transpiration models on sunlit and shaded portions of vegetation and soil.
    
    References:
        Ryu, Y., et al. (2011). Integration of MODIS land and atmosphere products with a coupled-process 
        model to estimate gross primary productivity and evapotranspiration from 1 km to global scales. 
        Remote Sensing of Environment, 115(8), 1865-1874.
    
    Parameters
    ----------
    ST_C : Union[Raster, np.ndarray]
        Surface temperature [°C]
    NDVI : Union[Raster, np.ndarray]  
        Normalized Difference Vegetation Index [-]
    albedo : Union[Raster, np.ndarray]
        Surface albedo [-]
    geometry : RasterGeometry, optional
        Raster geometry for spatial operations
    time_UTC : datetime, optional
        UTC time for solar calculations
    hour_of_day : np.ndarray, optional
        Solar hour of day [hours, 0-24]
    day_of_year : np.ndarray, optional
        Day of year [days, 1-365/366]
    GEOS5FP_connection : GEOS5FP, optional
        Connection to GEOS-5 FP meteorological data
    elevation_m : Union[Raster, np.ndarray], optional
        Elevation above sea level [m]
    Ta_C : Union[Raster, np.ndarray], optional
        Air temperature [°C]
    RH : Union[Raster, np.ndarray], optional
        Relative humidity [fraction, 0-1]
    NDVI_minimum : Union[Raster, np.ndarray], optional
        Minimum NDVI for phenology scaling [-]
    NDVI_maximum : Union[Raster, np.ndarray], optional
        Maximum NDVI for phenology scaling [-]
    SWin_Wm2 : Union[Raster, np.ndarray], optional
        Incoming shortwave radiation [W m⁻²]
    PAR_diffuse_Wm2 : Union[Raster, np.ndarray], optional
        Diffuse photosynthetically active radiation (400-700 nm) [W m⁻²]
    PAR_direct_Wm2 : Union[Raster, np.ndarray], optional
        Direct photosynthetically active radiation (400-700 nm) [W m⁻²]
    NIR_diffuse_Wm2 : Union[Raster, np.ndarray], optional
        Diffuse near-infrared radiation [W m⁻²]
    NIR_direct_Wm2 : Union[Raster, np.ndarray], optional
        Direct near-infrared radiation [W m⁻²]
    UV_Wm2 : Union[Raster, np.ndarray], optional
        Incoming ultraviolet radiation [W m⁻²]
    albedo_visible : Union[Raster, np.ndarray], optional
        Surface albedo in visible wavelengths (400-700 nm) [-]
    albedo_NIR : Union[Raster, np.ndarray], optional
        Surface albedo in near-infrared wavelengths [-]
    COT : Union[Raster, np.ndarray], optional
        Cloud optical thickness [-]
    AOT : Union[Raster, np.ndarray], optional
        Aerosol optical thickness [-]
    vapor_gccm : Union[Raster, np.ndarray], optional
        Water vapor column [g cm⁻²]
    ozone_cm : Union[Raster, np.ndarray], optional
        Ozone column [cm]
    KG_climate : Union[Raster, np.ndarray], optional
        Köppen-Geiger climate classification [-]
    canopy_height_meters : Union[Raster, np.ndarray], optional
        Canopy height [m]
    Ca : Union[Raster, np.ndarray], optional
        Atmospheric CO₂ concentration [ppm]
    wind_speed_mps : Union[Raster, np.ndarray], optional
        Wind speed [m s⁻¹]
    SZA_deg : Union[Raster, np.ndarray], optional
        Solar zenith angle [degrees]
    canopy_temperature_C : Union[Raster, np.ndarray], optional
        Canopy temperature [°C]. Defaults to surface temperature if None
    soil_temperature_C : Union[Raster, np.ndarray], optional
        Soil temperature [°C]. Defaults to surface temperature if None
    C4_fraction : Union[Raster, np.ndarray], optional
        Fraction of C4 plants [fraction, 0-1]
    carbon_uptake_efficiency : Union[Raster, np.ndarray], optional
        Intrinsic quantum efficiency for carbon uptake [mol CO₂ mol⁻¹ photons]
    kn : np.ndarray, optional
        Nitrogen decay coefficient [-]
    ball_berry_intercept_C3 : np.ndarray, optional
        Ball-Berry stomatal conductance intercept for C3 plants [mol m⁻² s⁻¹]
    ball_berry_intercept_C4 : Union[np.ndarray, float], optional
        Ball-Berry stomatal conductance intercept for C4 plants [mol m⁻² s⁻¹]
    ball_berry_slope_C3 : np.ndarray, optional
        Ball-Berry stomatal conductance slope for C3 plants [-]
    ball_berry_slope_C4 : np.ndarray, optional
        Ball-Berry stomatal conductance slope for C4 plants [-]
    peakVCmax_C3_μmolm2s1 : np.ndarray, optional
        Peak maximum carboxylation rate for C3 plants [μmol m⁻² s⁻¹]
    peakVCmax_C4_μmolm2s1 : np.ndarray, optional
        Peak maximum carboxylation rate for C4 plants [μmol m⁻² s⁻¹]
    CI : Union[Raster, np.ndarray], optional
        Clumping index [-]
    C4_fraction_scale_factor : float, optional
        Scale factor for C4 fraction adjustment [-]
    MODISCI_connection : MODISCI, optional
        Connection to MODIS clumping index data
    NASADEM_connection : NASADEMConnection, optional
        Connection to NASADEM elevation data
    resampling : str, optional
        Resampling method for data processing
    GEDI_download_directory : str, optional
        Directory for GEDI canopy height data downloads
    
    Returns
    -------
    dict
        Dictionary containing model outputs:
        
        GPP : Union[Raster, np.ndarray]
            Instantaneous gross primary productivity [μmol CO₂ m⁻² s⁻¹]
        GPP_daily : Union[Raster, np.ndarray]
            Daily gross primary productivity [g C m⁻² day⁻¹]
        Rn_Wm2 : Union[Raster, np.ndarray]
            Net radiation [W m⁻²]
        Rn_soil_Wm2 : Union[Raster, np.ndarray]
            Soil net radiation [W m⁻²]
        Rn_canopy_Wm2 : Union[Raster, np.ndarray]
            Canopy net radiation [W m⁻²]
        LE_Wm2 : Union[Raster, np.ndarray]
            Latent heat flux (evapotranspiration) [W m⁻²]
        LE_soil_Wm2 : Union[Raster, np.ndarray]
            Soil latent heat flux [W m⁻²]
        LE_canopy_Wm2 : Union[Raster, np.ndarray]
            Canopy latent heat flux (transpiration) [W m⁻²]
        G_Wm2 : Union[Raster, np.ndarray]
            Soil heat flux [W m⁻²]
    
    Notes
    -----
    The model uses the FLiES radiative transfer model to estimate incoming solar radiation
    components when they are not provided. The model calculates photosynthesis and energy
    balance separately for C3 and C4 vegetation types, then interpolates results based on
    the C4 fraction.
    
    For sunlit and shaded leaf fractions, the model applies different light extinction
    coefficients and photosynthetic parameters to account for canopy structure effects
    on radiation distribution and photosynthetic capacity.
    """
    if geometry is None and isinstance(ST_C, Raster):
        geometry = ST_C.geometry

    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP()

    if (day_of_year is None or hour_of_day is None) and time_UTC is not None and geometry is not None:
        day_of_year = calculate_solar_day_of_year(time_UTC=time_UTC, geometry=geometry)
        hour_of_day = calculate_solar_hour_of_day(time_UTC=time_UTC, geometry=geometry)

    if time_UTC is None and day_of_year is None and hour_of_day is None:
        raise ValueError("no time given between time_UTC, day_of_year, and hour_of_day")

    if elevation_m is None and geometry is not None:
        if NASADEM_connection is None:
            NASADEM_connection = NASADEMConnection()

        elevation_m = NASADEM_connection.elevation_m(geometry=geometry)

    check_distribution(elevation_m, "elevation_m")

    # load air temperature in Celsius if not provided
    if Ta_C is None:
        Ta_C = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    check_distribution(Ta_C, "Ta_C")

    # load relative humidity if not provided
    if RH is None:
        RH = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    check_distribution(RH, "RH")

    # load minimum NDVI if not provided
    if NDVI_minimum is None and geometry is not None:
        NDVI_minimum = load_NDVI_minimum(geometry=geometry, resampling=resampling)

    check_distribution(NDVI_minimum, "NDVI_minimum")

    # load maximum NDVI if not provided
    if NDVI_maximum is None and geometry is not None:
        NDVI_maximum = load_NDVI_maximum(geometry=geometry, resampling=resampling)

    check_distribution(NDVI_maximum, "NDVI_maximum")

    # load C4 fraction if not provided
    if C4_fraction is None:
        C4_fraction = load_C4_fraction(
            geometry=geometry, 
            resampling=resampling,
            scale_factor=C4_fraction_scale_factor
        )

    check_distribution(C4_fraction, "C4_fraction")

    # load carbon uptake efficiency if not provided
    if carbon_uptake_efficiency is None:
        carbon_uptake_efficiency = load_carbon_uptake_efficiency(geometry=geometry, resampling=resampling)
    
    check_distribution(carbon_uptake_efficiency, "carbon_uptake_efficiency")

    # load kn if not provided
    if kn is None:
        kn = load_kn(geometry=geometry, resampling=resampling)

    check_distribution(kn, "kn")

    # load peak VC max for C3 plants if not provided
    if peakVCmax_C3_μmolm2s1 is None:
        peakVCmax_C3_μmolm2s1 = load_peakVCmax_C3(geometry=geometry, resampling=resampling)

    check_distribution(peakVCmax_C3_μmolm2s1, "peakVCmax_C3")

    # load peak VC max for C4 plants if not provided
    if peakVCmax_C4_μmolm2s1 is None:
        peakVCmax_C4_μmolm2s1 = load_peakVCmax_C4(geometry=geometry, resampling=resampling)

    check_distribution(peakVCmax_C4_μmolm2s1, "peakVCmax_C4")

    # load Ball-Berry slope for C3 plants if not provided
    if ball_berry_slope_C3 is None:
        ball_berry_slope_C3 = load_ball_berry_slope_C3(geometry=geometry, resampling=resampling)
    
    check_distribution(ball_berry_slope_C3, "ball_berry_slope_C3")

    # load Ball-Berry slope for C4 plants if not provided
    if ball_berry_slope_C4 is None:
        ball_berry_slope_C4 = load_ball_berry_slope_C4(geometry=geometry, resampling=resampling)

    check_distribution(ball_berry_slope_C4, "ball_berry_slope_C4")

    # load Ball-Berry intercept for C3 plants if not provided
    if ball_berry_intercept_C3 is None:
        ball_berry_intercept_C3 = load_ball_berry_intercept_C3(geometry=geometry, resampling=resampling)

    check_distribution(ball_berry_intercept_C3, "ball_berry_intercept_C3")

    # Create a dictionary of variables to check
    variables_to_check = {
        "SWin_Wm2": SWin_Wm2,
        "PAR_diffuse_Wm2": PAR_diffuse_Wm2,
        "PAR_direct_Wm2": PAR_direct_Wm2,
        "NI_diffuse_Wm2": NIR_diffuse_Wm2,
        "NI_direct_Wm2": NIR_direct_Wm2,
        "UV_Wm2": UV_Wm2,
        "albedo_visible": albedo_visible,
        "albedo_NIR": albedo_NIR
    }

    # Check for None values and size mismatches
    reference_size = None
    for name, var in variables_to_check.items():
        if var is None:
            logger.warning(f"Variable '{name}' is None.")
        else:
            # Get the size of the variable if it's a numpy array
            size = var.shape if isinstance(var, np.ndarray) else None
            if reference_size is None:
                reference_size = size  # Set the first non-None size as the reference
            elif size != reference_size:
                logger.warning(f"Variable '{name}' has a different size: {size} (expected: {reference_size}).")

    # check if any of the FLiES outputs are not given
    flies_variables = [SWin_Wm2, PAR_diffuse_Wm2, PAR_direct_Wm2, NIR_diffuse_Wm2, NIR_direct_Wm2, UV_Wm2, albedo_visible, albedo_NIR]
    flies_variables_missing = False
    for variable in flies_variables:
        if variable is None:
            flies_variables_missing = True
    if flies_variables_missing:
        # load cloud optical thickness if not provided
        if COT is None:
            COT = GEOS5FP_connection.COT(time_UTC=time_UTC, geometry=geometry, resampling=resampling)

        # load aerosol optical thickness if not provided
        if AOT is None:
            AOT = GEOS5FP_connection.AOT(time_UTC=time_UTC, geometry=geometry, resampling=resampling)

        # run FLiES radiative transfer model
        FLiES_results = FLiESANN(
            time_UTC=time_UTC,
            day_of_year=day_of_year,
            hour_of_day=hour_of_day,
            geometry=geometry,
            albedo=albedo,
            COT=COT,
            AOT=AOT,
            vapor_gccm=vapor_gccm,
            ozone_cm=ozone_cm,
            elevation_m=elevation_m,
            SZA_deg=SZA_deg,
            KG_climate=KG_climate,
            GEOS5FP_connection=GEOS5FP_connection
        )

        # extract FLiES outputs
        SWin_Wm2 = FLiES_results["SWin_Wm2"]
        PAR_diffuse_Wm2 = FLiES_results["PAR_diffuse_Wm2"]
        PAR_direct_Wm2 = FLiES_results["PAR_direct_Wm2"]
        NIR_diffuse_Wm2 = FLiES_results["NIR_diffuse_Wm2"]
        NIR_direct_Wm2 = FLiES_results["NIR_direct_Wm2"]
        UV_Wm2 = FLiES_results["UV_Wm2"]
        # albedo_visible = FLiES_results["VIS"]
        # albedo_NIR = FLiES_results["NIR"]

        if albedo_visible is None:
            albedo_NWP = GEOS5FP_connection.ALBEDO(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
            RVIS_NWP = GEOS5FP_connection.ALBVISDR(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
            albedo_visible = rt.clip(albedo * (RVIS_NWP / albedo_NWP), 0, 1)

        # check_distribution(albedo_visible, "RVIS")
        
        if albedo_NIR is None:
            albedo_NWP = GEOS5FP_connection.ALBEDO(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
            RNIR_NWP = GEOS5FP_connection.ALBNIRDR(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
            albedo_NIR = rt.clip(albedo * (RNIR_NWP / albedo_NWP), 0, 1)
        
        # check_distribution(albedo_NIR, "RNIR")
        
        check_distribution(PAR_direct_Wm2, "PAR_direct_Wm2")
    else:
        logger.info("using given FLiES output as BESS parameters")

    # load koppen geiger climate classification if not provided
    if KG_climate is None:
        KG_climate = load_koppen_geiger(geometry=geometry)

    check_distribution(np.float32(KG_climate), "KG_climate")

    # load canopy height in meters if not provided
    if canopy_height_meters is None:
        canopy_height_meters = load_canopy_height(
            geometry=geometry, 
            resampling=resampling,
            source_directory=GEDI_download_directory
        )

    check_distribution(canopy_height_meters, "canopy_height_meters")

    # load CO2 concentration in ppm if not provided
    if Ca is None:
        Ca = GEOS5FP_connection.CO2SC(time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    check_distribution(Ca, "Ca")

    # load wind speed in meters per second if not provided
    if wind_speed_mps is None:
        wind_speed_mps = GEOS5FP_connection.wind_speed(time_UTC=time_UTC, geometry=geometry, resampling=resampling)    

    check_distribution(wind_speed_mps, "wind_speed_mps")

    # canopy temperature defaults to surface temperature
    if canopy_temperature_C is None:
        canopy_temperature_C = ST_C

    # soil temperature defaults to surface temperature
    if soil_temperature_C is None:
        soil_temperature_C = ST_C

    # visible albedo defaults to surface albedo
    if albedo_visible is None:
        albedo_visible = albedo

    # near-infrared albedo defaults to surface albedo
    if albedo_NIR is None:
        albedo_NIR = albedo

    # calculate solar zenith angle if not provided
    if SZA_deg is None:
        SZA_deg = calculate_SZA_from_DOY_and_hour(geometry.lat, geometry.lon, day_of_year, hour_of_day)

    if MODISCI_connection is None:
        MODISCI_connection = MODISCI()

    if CI is None and geometry is not None:
        CI = MODISCI_connection.CI(geometry=geometry, resampling=resampling)

    # canopy height defaults to zero
    canopy_height_meters = np.where(np.isnan(canopy_height_meters), 0, canopy_height_meters)

    # calculate saturation vapor pressure in Pascal from air temperature in Kelvin
    Ta_K = Ta_C + 273.15
    SVP_Pa = SVP_Pa_from_Ta_K(Ta_K)

    # calculate actual vapor pressure in Pascal from relative humidity and saturation vapor pressure
    Ea_Pa = RH * SVP_Pa

    latitude = geometry.lat

    meteorology_results = meteorology(
        day_of_year=day_of_year,
        hour_of_day=hour_of_day,
        latitude=latitude,
        elevation_m=elevation_m,
        SZA=SZA_deg,
        Ta_K=Ta_K,
        Ea_Pa=Ea_Pa,
        Rg_Wm2=SWin_Wm2,
        wind_speed_mps=wind_speed_mps,
        canopy_height_meters=canopy_height_meters
    )

    # Extract all variables from the dictionary returned by meteorology
    Ps_Pa = meteorology_results["Ps_Pa"]
    VPD_Pa = meteorology_results["VPD_Pa"]
    RH = meteorology_results["RH"]
    desTa = meteorology_results["desTa"]
    ddesTa = meteorology_results["ddesTa"]
    gamma = meteorology_results["gamma"]
    Cp = meteorology_results["Cp"]
    rhoa = meteorology_results["rhoa"]
    epsa = meteorology_results["epsa"]
    R = meteorology_results["R"]
    Rc = meteorology_results["Rc"]
    Rs = meteorology_results["Rs"]
    SFd = meteorology_results["SFd"]
    SFd2 = meteorology_results["SFd2"]
    DL = meteorology_results["DL"]
    Ra = meteorology_results["Ra"]
    fStress = meteorology_results["fStress"]

    # Check the distribution for each variable
    for var_name, var_value in meteorology_results.items():
        check_distribution(var_value, var_name)

    # convert NDVI to LAI
    LAI = LAI_from_NDVI(NDVI)
    LAI_minimum = LAI_from_NDVI(NDVI_minimum)
    LAI_maximum = LAI_from_NDVI(NDVI_maximum)

    VCmax_results = calculate_VCmax(
        LAI=LAI,
        LAI_minimum=LAI_minimum,
        LAI_maximum=LAI_maximum,
        peakVCmax_C3_μmolm2s1=peakVCmax_C3_μmolm2s1,
        peakVCmax_C4_μmolm2s1=peakVCmax_C4_μmolm2s1,
        SZA_deg=SZA_deg,
        kn=kn
    )

    VCmax_C3_sunlit_μmolm2s1 = VCmax_results["VCmax_C3_sunlit_μmolm2s1"]
    VCmax_C4_sunlit_μmolm2s1 = VCmax_results["VCmax_C4_sunlit_μmolm2s1"]
    VCmax_C3_shaded_μmolm2s1 = VCmax_results["VCmax_C3_shaded_μmolm2s1"]
    VCmax_C4_shaded_μmolm2s1 = VCmax_results["VCmax_C4_shaded_μmolm2s1"]

    # Check the distribution for each variable
    for var_name, var_value in VCmax_results.items():
        check_distribution(var_value, var_name)

    canopy_shortwave_radiation_results = canopy_shortwave_radiation(
        PAR_diffuse_Wm2=PAR_diffuse_Wm2,  # diffuse photosynthetically active radiation in W/m^2
        PAR_direct_Wm2=PAR_direct_Wm2,  # direct photosynthetically active radiation in W/m^2
        NIR_diffuse_Wm2=NIR_diffuse_Wm2,  # diffuse near-infrared radiation in W/m^2
        NIR_direct_Wm2=NIR_direct_Wm2,  # direct near-infrared radiation in W/m^2
        UV_Wm2=UV_Wm2,  # incoming ultraviolet radiation in W/m^2
        SZA_deg=SZA_deg,  # solar zenith angle in degrees
        LAI=LAI,  # leaf area index
        CI=CI,  # clumping index
        albedo_visible=albedo_visible,  # surface albedo in visible wavelengths
        albedo_NIR=albedo_NIR  # surface albedo in near-infrared wavelengths
    )

    # Check the distribution for each variable
    for var_name, var_value in canopy_shortwave_radiation_results.items():
        check_distribution(var_value, var_name)

    # Extract values from the dictionary
    sunlit_fraction = canopy_shortwave_radiation_results["fSun"]
    APAR_sunlit_μmolm2s1 = canopy_shortwave_radiation_results["APAR_sunlit_μmolm2s1"]
    APAR_shade_μmolm2s1 = canopy_shortwave_radiation_results["APAR_shade_μmolm2s1"]
    ASW_sunlit_Wm2 = canopy_shortwave_radiation_results["ASW_sunlit_Wm2"]
    ASW_shade_Wm2 = canopy_shortwave_radiation_results["ASW_shade_Wm2"]
    ASW_soil_Wm2 = canopy_shortwave_radiation_results["ASW_soil_Wm2"]
    G_Wm2 = canopy_shortwave_radiation_results["G_Wm2"]

    canopy_temperature_K = canopy_temperature_C + 273.15
    soil_temperature_K = soil_temperature_C + 273.15

    GPP_C3, LE_C3, LE_soil_C3, LE_canopy_C3, Rn_C3, Rn_soil_C3, Rn_canopy_C3 = carbon_water_fluxes(
        canopy_temperature_K=canopy_temperature_K,  # canopy temperature in Kelvin
        soil_temperature_K=soil_temperature_K,  # soil temperature in Kelvin
        LAI=LAI,  # leaf area index
        Ta_K=Ta_K,  # air temperature in Kelvin
        APAR_sunlit_μmolm2s1=APAR_sunlit_μmolm2s1,  # sunlit leaf absorptance of photosynthetically active radiation
        APAR_shaded_μmolm2s1=APAR_shade_μmolm2s1,  # shaded leaf absorptance of photosynthetically active radiation
        ASW_sunlit_Wm2=ASW_sunlit_Wm2,  # sunlit absorbed shortwave radiation
        ASW_shaded_Wm2=ASW_shade_Wm2,  # shaded absorbed shortwave radiation
        ASW_soil_Wm2=ASW_soil_Wm2,  # absorbed shortwave radiation of soil
        Vcmax25_sunlit=VCmax_C3_sunlit_μmolm2s1,  # sunlit maximum carboxylation rate at 25 degrees C
        Vcmax25_shaded=VCmax_C3_shaded_μmolm2s1,  # shaded maximum carboxylation rate at 25 degrees C
        ball_berry_slope=ball_berry_slope_C3,  # Ball-Berry slope for C3 photosynthesis
        ball_berry_intercept=ball_berry_intercept_C3,  # Ball-Berry intercept for C3 photosynthesis
        sunlit_fraction=sunlit_fraction,  # fraction of sunlit leaves
        G_Wm2=G_Wm2,  # soil heat flux
        SZA_deg=SZA_deg,  # solar zenith angle
        Ca=Ca,  # atmospheric CO2 concentration
        Ps_Pa=Ps_Pa,  # surface pressure in Pascal
        gamma=gamma,  # psychrometric constant
        Cp=Cp,  # specific heat of air in J/kg/K
        rhoa=rhoa,  # density of air in kg/m3
        VPD_Pa=VPD_Pa,  # vapor pressure deficit in Pascal
        RH=RH,  # relative humidity as a fraction
        desTa=desTa,
        ddesTa=ddesTa,
        epsa=epsa,
        Rc=Rc,
        Rs=Rs,
        carbon_uptake_efficiency=carbon_uptake_efficiency,  # intrinsic quantum efficiency for carbon uptake
        fStress=fStress,
        C4_photosynthesis=False  # C3 or C4 photosynthesis
    )

    # List of variable names and their corresponding values
    carbon_water_fluxes_outputs = {
        "GPP_C3": GPP_C3,
        "LE_C3": LE_C3,
        "LE_soil_C3": LE_soil_C3,
        "LE_canopy_C3": LE_canopy_C3,
        "Rn_C3": Rn_C3,
        "Rn_soil_C3": Rn_soil_C3,
        "Rn_canopy_C3": Rn_canopy_C3
    }

    # Check the distribution for each variable
    for var_name, var_value in carbon_water_fluxes_outputs.items():
        check_distribution(var_value, var_name)

    GPP_C4, LE_C4, LE_soil_C4, LE_canopy_C4, Rn_C4, Rn_soil_C4, Rn_canopy_C4 = carbon_water_fluxes(
        canopy_temperature_K=canopy_temperature_K,  # canopy temperature in Kelvin
        soil_temperature_K=soil_temperature_K,  # soil temperature in Kelvin
        LAI=LAI,  # leaf area index
        Ta_K=Ta_K,  # air temperature in Kelvin
        APAR_sunlit_μmolm2s1=APAR_sunlit_μmolm2s1,  # sunlit leaf absorptance of photosynthetically active radiation
        APAR_shaded_μmolm2s1=APAR_shade_μmolm2s1,  # shaded leaf absorptance of photosynthetically active radiation
        ASW_sunlit_Wm2=ASW_sunlit_Wm2,  # sunlit absorbed shortwave radiation
        ASW_shaded_Wm2=ASW_shade_Wm2,  # shaded absorbed shortwave radiation
        ASW_soil_Wm2=ASW_soil_Wm2,  # absorbed shortwave radiation of soil
        Vcmax25_sunlit=VCmax_C4_sunlit_μmolm2s1,  # sunlit maximum carboxylation rate at 25 degrees C
        Vcmax25_shaded=VCmax_C4_shaded_μmolm2s1,  # shaded maximum carboxylation rate at 25 degrees C
        ball_berry_slope=ball_berry_slope_C4,  # Ball-Berry slope for C4 photosynthesis
        ball_berry_intercept=ball_berry_intercept_C4,  # Ball-Berry intercept for C4 photosynthesis
        sunlit_fraction=sunlit_fraction,  # fraction of sunlit leaves
        G_Wm2=G_Wm2,  # soil heat flux
        SZA_deg=SZA_deg,  # solar zenith angle
        Ca=Ca,  # atmospheric CO2 concentration
        Ps_Pa=Ps_Pa,  # surface pressure in Pascal
        gamma=gamma,  # psychrometric constant
        Cp=Cp,  # specific heat of air in J/kg/K
        rhoa=rhoa,  # density of air in kg/m3
        VPD_Pa=VPD_Pa,  # vapor pressure deficit in Pascal
        RH=RH,  # relative humidity as a fraction
        desTa=desTa,
        ddesTa=ddesTa,
        epsa=epsa,
        Rc=Rc,
        Rs=Rs,
        carbon_uptake_efficiency=carbon_uptake_efficiency,  # intrinsic quantum efficiency for carbon uptake
        fStress=fStress,
        C4_photosynthesis=True  # C3 or C4 photosynthesis
    )

    # List of variable names and their corresponding values
    carbon_water_fluxes_C4_outputs = {
        "GPP_C4": GPP_C4,
        "LE_C4": LE_C4,
        "LE_soil_C4": LE_soil_C4,
        "LE_canopy_C4": LE_canopy_C4,
        "Rn_C4": Rn_C4,
        "Rn_soil_C4": Rn_soil_C4,
        "Rn_canopy_C4": Rn_canopy_C4
    }

    # Check the distribution for each variable
    for var_name, var_value in carbon_water_fluxes_C4_outputs.items():
        check_distribution(var_value, var_name)

    # interpolate C3 and C4 GPP
    ST_K = ST_C + 273.15
    GPP = np.clip(interpolate_C3_C4(GPP_C3, GPP_C4, C4_fraction), 0, 50)
    GPP = np.where(np.isnan(ST_K), np.nan, GPP)

    if isinstance(geometry, RasterGeometry):
        GPP = Raster(GPP, geometry=geometry)
        GPP.cmap = GPP_COLORMAP

    # upscale from instantaneous to daily

    # upscale GPP to daily
    GPP_daily = 1800 * GPP / SFd * 1e-6 * 12  # Eq. (3) in Ryu et al 2008
    GPP_daily = np.where(SFd < 0.01, 0, GPP_daily)
    GPP_daily = np.where(SZA_deg >= 90, 0, GPP_daily)

    if isinstance(geometry, RasterGeometry):
        GPP_daily = Raster(GPP_daily, geometry=geometry)
        GPP_daily.cmap = GPP_COLORMAP

    # interpolate C3 and C4 net radiation
    Rn_Wm2 = np.clip(interpolate_C3_C4(Rn_C3, Rn_C4, C4_fraction), 0, None)

    if isinstance(geometry, RasterGeometry):
        Rn_Wm2 = Raster(Rn_Wm2, geometry=geometry)

    # interpolate C3 and C4 soil net radiation
    Rn_soil_Wm2 = np.clip(interpolate_C3_C4(Rn_soil_C3, Rn_soil_C4, C4_fraction), 0, Rn_Wm2)

    if isinstance(geometry, RasterGeometry):
        Rn_soil_Wm2 = Raster(Rn_soil_Wm2, geometry=geometry)

    # interpolate C3 and C4 canopy net radiation
    Rn_canopy_Wm2 = np.clip(interpolate_C3_C4(Rn_canopy_C3, Rn_canopy_C4, C4_fraction), 0, Rn_Wm2)

    if isinstance(geometry, RasterGeometry):
        Rn_canopy_Wm2 = Raster(Rn_canopy_Wm2, geometry=geometry)

    # interpolate C3 and C4 latent heat flux
    LE_Wm2 = np.clip(interpolate_C3_C4(LE_C3, LE_C4, C4_fraction), 0, Rn_Wm2)

    if isinstance(geometry, RasterGeometry):
        LE_Wm2 = Raster(LE_Wm2, geometry=geometry)
        LE_Wm2.cmap = ET_COLORMAP

    # interpolate C3 and C4 soil latent heat flux
    LE_soil_Wm2 = np.clip(interpolate_C3_C4(LE_soil_C3, LE_soil_C4, C4_fraction), 0, LE_Wm2)

    if isinstance(geometry, RasterGeometry):
        LE_soil_Wm2 = Raster(LE_soil_Wm2, geometry=geometry)
        LE_soil_Wm2.cmap = ET_COLORMAP

    # interpolate C3 and C4 canopy latent heat flux
    LE_canopy_Wm2 = np.clip(interpolate_C3_C4(LE_canopy_C3, LE_canopy_C4, C4_fraction), 0, LE_Wm2)

    if isinstance(geometry, RasterGeometry):
        LE_canopy_Wm2 = Raster(LE_canopy_Wm2, geometry=geometry)
        LE_canopy_Wm2.cmap = ET_COLORMAP

    results = {
        "GPP": GPP,
        "GPP_daily": GPP_daily,
        "Rn_Wm2": Rn_Wm2,
        "Rn_soil_Wm2": Rn_soil_Wm2,
        "Rn_canopy_Wm2": Rn_canopy_Wm2,
        "LE_Wm2": LE_Wm2,
        "LE_soil_Wm2": LE_soil_Wm2,
        "LE_canopy_Wm2": LE_canopy_Wm2,
        "G_Wm2": G_Wm2
    }

    if upscale_to_daylight and time_UTC is not None:
        logger.info("started daylight ET upscaling")
        t_et = TicToc()
        t_et.tic()

        # Use new upscaling function from daylight_evapotranspiration
        daylight_results = daylight_ET_from_instantaneous_LE(
            LE_instantaneous_Wm2=LE_Wm2,
            Rn_instantaneous_Wm2=Rn_Wm2,
            G_instantaneous_Wm2=G_Wm2,
            day_of_year=day_of_year,
            time_UTC=time_UTC,
            geometry=geometry
        )
        # Add all returned daylight results to output
        results.update(daylight_results)

        elapsed_et = t_et.tocvalue()
        logger.info(f"completed daylight ET upscaling (elapsed: {elapsed_et:.2f} seconds)")

    return results
