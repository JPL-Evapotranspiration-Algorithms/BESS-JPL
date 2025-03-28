from typing import Union
from datetime import datetime
import numpy as np

import rasters as rt
from rasters import Raster, RasterGeometry

from sun_angles import calculate_SZA_from_DOY_and_hour

from koppengeiger import load_koppen_geiger
from gedi_canopy_height import load_canopy_height
from FLiESANN import FLiESANN
from GEOS5FP import GEOS5FP

from .constants import *
from .vegetation_conversion import LAI_from_NDVI
from .canopy_shortwave_radiation import canopy_shortwave_radiation
from .carbon_water_fluxes import carbon_water_fluxes
from .meteorology import meteorology, SVP_Pa_from_Ta_K
from .interpolate_C3_C4 import interpolate_C3_C4
from .calculate_VCmax import calculate_VCmax
from .load_NDVI_minimum import load_NDVI_minimum
from .load_NDVI_maximum import load_NDVI_maximum
from .load_C4_fraction import load_C4_fraction
from .load_carbon_uptake_efficiency import load_carbon_uptake_efficiency
from .load_kn import load_kn
from .load_peakVCmax_C3 import load_peakVCmax_C3
from .load_peakVCmax_C4 import load_peakVCmax_C4
from .load_ball_berry_intercept_C3 import load_ball_berry_intercept_C3
from .load_ball_berry_slope_C3 import load_ball_berry_slope_C3
from .load_ball_berry_slope_C4 import load_ball_berry_slope_C4

GEOS5FP_DOWNLOAD_DIRECTORY = "~/data/GEOS5FP_download"

## TODO automatic acquisition of SRTM for default elevation

def BESS(
        ST_C: Union[Raster, np.ndarray],  # surface temperature in Celsius
        NDVI: Union[Raster, np.ndarray],  # NDVI
        albedo: Union[Raster, np.ndarray],  # surface albedo
        elevation_km: Union[Raster, np.ndarray],  # elevation in kilometers
        geometry: RasterGeometry = None,
        time_UTC: datetime = None,
        hour_of_day: np.ndarray = None,
        day_of_year: np.ndarray = None,
        GEOS5FP_connection: GEOS5FP = None,
        Ta_C: Union[Raster, np.ndarray] = None,  # air temperature in Celsius
        RH: Union[Raster, np.ndarray] = None,  # relative humidity as a proportion
        NDVI_minimum: Union[Raster, np.ndarray] = None,  # minimum NDVI
        NDVI_maximum: Union[Raster, np.ndarray] = None,  # maximum NDVI
        Rg: Union[Raster, np.ndarray] = None,  # incoming shortwave radiation in W/m^2
        VISdiff: Union[Raster, np.ndarray] = None,  # diffuse visible radiation in W/m^2
        VISdir: Union[Raster, np.ndarray] = None,  # direct visible radiation in W/m^2
        NIRdiff: Union[Raster, np.ndarray] = None,  # diffuse near-infrared radiation in W/m^2
        NIRdir: Union[Raster, np.ndarray] = None,  # direct near-infrared radiation in W/m^2
        UV: Union[Raster, np.ndarray] = None,  # incoming ultraviolet radiation in W/m^2
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
        SZA: Union[Raster, np.ndarray] = None,  # solar zenith angle in degrees
        canopy_temperature_C: Union[Raster, np.ndarray] = None, # canopy temperature in Celsius (initialized to surface temperature if left as None)
        soil_temperature_C: Union[Raster, np.ndarray] = None, # soil temperature in Celsius (initialized to surface temperature if left as None)
        C4_fraction: Union[Raster, np.ndarray] = None,  # fraction of C4 plants
        carbon_uptake_efficiency: Union[Raster, np.ndarray] = None,  # intrinsic quantum efficiency for carbon uptake
        kn: np.ndarray = None,
        ball_berry_intercept_C3: np.ndarray = None,  # Ball-Berry intercept for C3 plants
        ball_berry_intercept_C4: Union[np.ndarray, float] = BALL_BERRY_INTERCEPT_C4, # Ball-Berry intercept for C4 plants
        ball_berry_slope_C3: np.ndarray = None,  # Ball-Berry slope for C3 plants
        ball_berry_slope_C4: np.ndarray = None,  # Ball-Berry slope for C4 plants
        peakVCmax_C3: np.ndarray = None,  # peak maximum carboxylation rate for C3 plants
        peakVCmax_C4: np.ndarray = None,  # peak maximum carboxylation rate for C4 plants
        CI: Union[Raster, np.ndarray] = None,
        GEOS5FP_download_directory: str = GEOS5FP_DOWNLOAD_DIRECTORY,
        resampling: str = RESAMPLING):  # clumping index
    if geometry is None and isinstance(ST_C, Raster):
        geometry = ST_C.geometry

    if (day_of_year is None or hour_of_day is None) and time_UTC is not None and geometry is not None:
        day_of_year = solar_day_of_year_for_area(time_UTC=time_UTC, geometry=geometry)
        hour_of_day = solar_hour_of_day_for_area(time_UTC=time_UTC, geometry=geometry)

    if time_UTC is None and day_of_year is None and hour_of_day is None:
        raise ValueError("no time given between time_UTC, day_of_year, and hour_of_day")

    # load air temperature in Celsius if not provided
    if Ta_C is None:
        Ta_C = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    # load relative humidity if not provided
    if RH is None:
        RH = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    # load minimum NDVI if not provided
    if NDVI_minimum is None and geometry is not None:
        NDVI_minimum = load_NDVI_minimum(geometry=geometry, resampling=resampling)

    # load maximum NDVI if not provided
    if NDVI_maximum is None and geometry is not None:
        NDVI_maximum = load_NDVI_maximum(geometry=geometry, resampling=resampling)

    # load C4 fraction if not provided
    if C4_fraction is None:
        C4_fraction = load_C4_fraction(geometry=geometry, resampling=resampling)

    # load carbon uptake efficiency if not provided
    if carbon_uptake_efficiency is None:
        carbon_uptake_efficiency = load_carbon_uptake_efficiency(geometry=geometry, resampling=resampling)
    
    # load kn if not provided
    if kn is None:
        kn = load_kn(geometry=geometry, resampling=resampling)

    # load peak VC max for C3 plants if not provided
    if peakVCmax_C3 is None:
        peakVCmax_C3 = load_peakVCmax_C3(geometry=geometry, resampling=resampling)

    # load peak VC max for C4 plants if not provided
    if peakVCmax_C4 is None:
        peakVCmax_C4 = load_peakVCmax_C4(geometry=geometry, resampling=resampling)

    # load Ball-Berry slope for C3 plants if not provided
    if ball_berry_slope_C3 is None:
        ball_berry_slope_C3 = load_ball_berry_slope_C3(geometry=geometry, resampling=resampling)
    
    # load Ball-Berry slope for C4 plants if not provided
    if ball_berry_slope_C4 is None:
        ball_berry_slope_C4 = load_ball_berry_slope_C4(geometry=geometry, resampling=resampling)

    # load Ball-Berry intercept for C3 plants if not provided
    if ball_berry_intercept_C3 is None:
        ball_berry_intercept_C3 = load_ball_berry_intercept_C3(geometry=geometry, resampling=resampling)

    # check if any of the FLiES outputs are not given
    if None in (Rg, VISdiff, VISdir, NIRdiff, NIRdir, UV, albedo_visible, albedo_NIR):
        # load cloud optical thickness if not provided
        if COT is None:
            COT = GEOS5FP_connection.COT(time_UTC=time_UTC, geometry=geometry, resampling=resampling)

        # load aerosol optical thickness if not provided
        if AOT is None:
            AOT = GEOS5FP_connection.AOT(time_UTC=time_UTC, geometry=geometry, resampling=resampling)

        ## FIXME fix FLiES interface

        # run FLiES radiative transfer model
        FLiES_results = FLiESANN(
            day_of_year=day_of_year,
            hour_of_day=hour_of_day,
            albedo=albedo,
            COT=COT,
            AOT=AOT,
            vapor_gccm=vapor_gccm,
            ozone_cm=ozone_cm,
            elevation_km=elevation_km,
            SZA=SZA,
            KG_climate=KG_climate,
            geometry=geometry,
            GEOS5FP_connection=GEOS5FP_connection
        )

        # extract FLiES outputs
        Rg = FLiES_results["Rg"]
        VISdiff = FLiES_results["VISdiff"]
        VISdir = FLiES_results["VISdir"]
        NIRdiff = FLiES_results["NIRdiff"]
        NIRdir = FLiES_results["NIRdir"]
        UV = FLiES_results["UV"]
        albedo_visible = FLiES_results["VIS"]
        albedo_NIR = FLiES_results["NIR"]

    # load koppen geiger climate classification if not provided
    if KG_climate is None:
        KG_climate = load_koppen_geiger(geometry=geometry)

    # load canopy height in meters if not provided
    if canopy_height_meters is None:
        canopy_height_meters = load_canopy_height(geometry=geometry, resampling=resampling)

    # load CO2 concentration in ppm if not provided
    if Ca is None:
        Ca = GEOS5FP_connection.CO2SC(time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    # load wind speed in meters per second if not provided
    if wind_speed_mps is None:
        wind_speed_mps = GEOS5FP_connection.wind_speed(time_UTC=time_UTC, geometry=geometry, resampling=resampling)    

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
    if SZA is None:
        SZA = calculate_SZA_from_DOY_and_hour(lat, lon, day_of_year, hour_of_day)

    # canopy height defaults to zero
    canopy_height_meters = np.where(np.isnan(canopy_height_meters), 0, canopy_height_meters)

    # calculate saturation vapor pressure in Pascal from air temperature in Kelvin
    Ta_K = Ta_C + 273.15
    SVP_Pa = SVP_Pa_from_Ta_K(Ta_K)

    # calculate actual vapor pressure in Pascal from relative humidity and saturation vapor pressure
    Ea_Pa = RH * SVP_Pa

    # convert elevation to meters
    elevation_m = elevation_km * 1000

    latitude = geometry.lat

    Ps_Pa, VPD_Pa, RH, desTa, ddesTa, gamma, Cp, rhoa, epsa, R, Rc, Rs, SFd, SFd2, DL, Ra, fStress = meteorology(
        day_of_year=day_of_year,
        hour_of_day=hour_of_day,
        latitude=latitude,
        elevation_m=elevation_m,
        SZA=SZA,
        Ta_K=Ta_K,
        Ea_Pa=Ea_Pa,
        Rg=Rg,
        wind_speed_mps=wind_speed_mps,
        canopy_height_meters=canopy_height_meters
    )

    # convert NDVI to LAI
    LAI = LAI_from_NDVI(NDVI)
    LAI_minimum = LAI_from_NDVI(NDVI_minimum)
    LAI_maximum = LAI_from_NDVI(NDVI_maximum)

    VCmax_C3_sunlit, VCmax_C4_sunlit, VCmax_C3_shaded, VCmax_C4_shaded = calculate_VCmax(
        LAI=LAI,
        LAI_minimum=LAI_minimum,
        LAI_maximum=LAI_maximum,
        peakVCmax_C3=peakVCmax_C3,
        peakVCmax_C4=peakVCmax_C4,
        SZA=SZA,
        kn=kn
    )

    sunlit_fraction, APAR_sunlit, APAR_shaded, ASW_sunlit, ASW_shaded, ASW_soil, G = canopy_shortwave_radiation(
        PARDiff=VISdiff,  # diffuse photosynthetically active radiation in W/m^2
        PARDir=VISdir,  # direct photosynthetically active radiation in W/m^2
        NIRDiff=NIRdiff,  # diffuse near-infrared radiation in W/m^2
        NIRDir=NIRdir,  # direct near-infrared radiation in W/m^2
        UV=UV,  # incoming ultraviolet radiation in W/m^2
        SZA=SZA,  # solar zenith angle in degrees
        LAI=LAI,  # leaf area index
        CI=CI,  # clumping index
        albedo_visible=albedo_visible,  # surface albedo in visible wavelengths
        albedo_NIR=albedo_NIR  # surface albedo in near-infrared wavelengths
    )

    canopy_temperature_K = canopy_temperature_C + 273.15
    soil_temperature_K = soil_temperature_C + 273.15

    GPP_C3, LE_C3, LE_soil_C3, LE_canopy_C3, Rn_C3, Rn_soil_C3, Rn_canopy_C3 = carbon_water_fluxes(
        canopy_temperature_K=canopy_temperature_K,  # canopy temperature in Kelvin
        soil_temperature_K=soil_temperature_K,  # soil temperature in Kelvin
        LAI=LAI,  # leaf area index
        Ta_K=Ta_K,  # air temperature in Kelvin
        APAR_sunlit=APAR_sunlit,  # sunlit leaf absorptance of photosynthetically active radiation
        APAR_shaded=APAR_shaded,  # shaded leaf absorptance of photosynthetically active radiation
        ASW_sunlit=ASW_sunlit,  # sunlit absorbed shortwave radiation
        ASW_shaded=ASW_shaded,  # shaded absorbed shortwave radiation
        ASW_soil=ASW_soil,  # absorbed shortwave radiation of soil
        Vcmax25_sunlit=VCmax_C3_sunlit,  # sunlit maximum carboxylation rate at 25 degrees C
        Vcmax25_shaded=VCmax_C3_shaded,  # shaded maximum carboxylation rate at 25 degrees C
        ball_berry_slope=ball_berry_slope_C3,  # Ball-Berry slope for C3 photosynthesis
        ball_berry_intercept=ball_berry_intercept_C3,  # Ball-Berry intercept for C3 photosynthesis
        sunlit_fraction=sunlit_fraction,  # fraction of sunlit leaves
        G=G,  # soil heat flux
        SZA=SZA,  # solar zenith angle
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

    GPP_C4, LE_C4, LE_soil_C4, LE_canopy_C4, Rn_C4, Rn_soil_C4, Rn_canopy_C4 = carbon_water_fluxes(
        canopy_temperature_K=canopy_temperature_K,  # canopy temperature in Kelvin
        soil_temperature_K=soil_temperature_K,  # soil temperature in Kelvin
        LAI=LAI,  # leaf area index
        Ta_K=Ta_K,  # air temperature in Kelvin
        APAR_sunlit=APAR_sunlit,  # sunlit leaf absorptance of photosynthetically active radiation
        APAR_shaded=APAR_shaded,  # shaded leaf absorptance of photosynthetically active radiation
        ASW_sunlit=ASW_sunlit,  # sunlit absorbed shortwave radiation
        ASW_shaded=ASW_shaded,  # shaded absorbed shortwave radiation
        ASW_soil=ASW_soil,  # absorbed shortwave radiation of soil
        Vcmax25_sunlit=VCmax_C4_sunlit,  # sunlit maximum carboxylation rate at 25 degrees C
        Vcmax25_shaded=VCmax_C4_shaded,  # shaded maximum carboxylation rate at 25 degrees C
        ball_berry_slope=ball_berry_slope_C4,  # Ball-Berry slope for C4 photosynthesis
        ball_berry_intercept=ball_berry_intercept_C4,  # Ball-Berry intercept for C4 photosynthesis
        sunlit_fraction=sunlit_fraction,  # fraction of sunlit leaves
        G=G,  # soil heat flux
        SZA=SZA,  # solar zenith angle
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

    # interpolate C3 and C4 GPP
    ST_K = ST_C + 273.15
    GPP = np.clip(interpolate_C3_C4(GPP_C3, GPP_C4, C4_fraction), 0, 50)
    GPP = np.where(np.isnan(ST_K), np.nan, GPP)

    # upscale from instantaneous to daily

    # upscale GPP to daily
    GPP_daily = 1800 * GPP / SFd * 1e-6 * 12  # Eq. (3) in Ryu et al 2008
    GPP_daily = np.where(SFd < 0.01, 0, GPP_daily)
    GPP_daily = np.where(SZA >= 90, 0, GPP_daily)

    # interpolate C3 and C4 net radiation
    Rn = np.clip(interpolate_C3_C4(Rn_C3, Rn_C4, C4_fraction), 0, 1000)

    # interpolate C3 and C4 soil net radiation
    Rn_soil = np.clip(interpolate_C3_C4(Rn_soil_C3, Rn_soil_C4, C4_fraction), 0, 1000)

    # interpolate C3 and C4 canopy net radiation
    Rn_canopy = np.clip(interpolate_C3_C4(Rn_canopy_C3, Rn_canopy_C4, C4_fraction), 0, 1000)

    # interpolate C3 and C4 latent heat flux
    LE = np.clip(interpolate_C3_C4(LE_C3, LE_C4, C4_fraction), 0, 1000)

    # interpolate C3 and C4 soil latent heat flux
    LE_soil = np.clip(interpolate_C3_C4(LE_soil_C3, LE_soil_C4, C4_fraction), 0, 1000)

    # interpolate C3 and C4 canopy latent heat flux
    LE_canopy = np.clip(interpolate_C3_C4(LE_canopy_C3, LE_canopy_C4, C4_fraction), 0, 1000)

    return GPP, GPP_daily, Rn, Rn_soil, Rn_canopy, LE, LE_soil, LE_canopy
