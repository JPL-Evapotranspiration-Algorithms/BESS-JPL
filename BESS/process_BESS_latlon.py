from typing import Union

import numpy as np
import rasters as rt
from sun_angles import calculate_SZA_from_DOY_and_hour

from .constants import *
from .meteorology import SVP_Pa_from_Ta_K
from .meteorology import meteorology
from .LAI_from_NDVI import LAI_from_NDVI
from .calculate_VCmax import calculate_VCmax
from .canopy_shortwave_radiation import canopy_shortwave_radiation
from .carbon_water_fluxes import carbon_water_fluxes
from .interpolate_C3_C4 import interpolate_C3_C4

def process_BESS_latlon(
        hour_of_day: Union[rt.Raster, np.ndarray],  # hour of day
        day_of_year: Union[rt.Raster, np.ndarray],  # day of year
        latitude: Union[rt.Raster, np.ndarray],  # latitude
        longitude: Union[rt.Raster, np.ndarray],  # longitude
        elevation_km: Union[rt.Raster, np.ndarray],  # elevation in kilometers
        ST_C: Union[rt.Raster, np.ndarray],  # surface temperature in Celsius
        NDVI: Union[rt.Raster, np.ndarray],  # NDVI
        NDVI_minimum: Union[rt.Raster, np.ndarray],  # minimum NDVI
        NDVI_maximum: Union[rt.Raster, np.ndarray],  # maximum NDVI
        albedo: Union[rt.Raster, np.ndarray],  # surface albedo
        Ta_C: Union[rt.Raster, np.ndarray],  # air temperature in Kelvin
        RH: Union[rt.Raster, np.ndarray],  # relative humidity as a proportion
        Rg: Union[rt.Raster, np.ndarray],  # incoming shortwave radiation in W/m^2
        VISdiff: Union[rt.Raster, np.ndarray],  # diffuse visible radiation in W/m^2
        VISdir: Union[rt.Raster, np.ndarray],  # direct visible radiation in W/m^2
        NIRdiff: Union[rt.Raster, np.ndarray],  # diffuse near-infrared radiation in W/m^2
        NIRdir: Union[rt.Raster, np.ndarray],  # direct near-infrared radiation in W/m^2
        UV: Union[rt.Raster, np.ndarray],  # incoming ultraviolet radiation in W/m^2
        canopy_height_meters: Union[rt.Raster, np.ndarray],  # canopy height in meters
        Ca: Union[rt.Raster, np.ndarray],  # atmospheric CO2 concentration in ppm
        wind_speed_mps: Union[rt.Raster, np.ndarray],  # wind speed in meters per second
        SZA: Union[rt.Raster, np.ndarray] = None,  # solar zenith angle in degrees
        canopy_temperature_C: Union[rt.Raster, np.ndarray] = None, # canopy temperature in Celsius (initialized to surface temperature if left as None)
        soil_temperature_C: Union[rt.Raster, np.ndarray] = None, # soil temperature in Celsius (initialized to surface temperature if left as None)
        albedo_visible: Union[rt.Raster, np.ndarray] = None, # surface albedo in visible wavelengths (initialized to surface albedo if left as None)
        albedo_NIR: Union[rt.Raster, np.ndarray] = None, # surface albedo in near-infrared wavelengths (initialized to surface albedo if left as None)
        C4_fraction: Union[rt.Raster, np.ndarray] = None,  # fraction of C4 plants
        carbon_uptake_efficiency: Union[rt.Raster, np.ndarray] = None,  # intrinsic quantum efficiency for carbon uptake
        kn: Union[rt.Raster, np.ndarray] = None,
        ball_berry_intercept_C3: Union[rt.Raster, np.ndarray] = None,  # Ball-Berry intercept for C3 plants
        ball_berry_intercept_C4: Union[rt.Raster, np.ndarray, float] = BALL_BERRY_INTERCEPT_C4, # Ball-Berry intercept for C4 plants
        ball_berry_slope_C3: Union[rt.Raster, np.ndarray] = None,  # Ball-Berry slope for C3 plants
        ball_berry_slope_C4: Union[rt.Raster, np.ndarray] = None,  # Ball-Berry slope for C4 plants
        peakVCmax_C3: Union[rt.Raster, np.ndarray] = None,  # peak maximum carboxylation rate for C3 plants
        peakVCmax_C4: Union[rt.Raster, np.ndarray] = None,  # peak maximum carboxylation rate for C4 plants
        CI: Union[rt.Raster, np.ndarray] = None):  # clumping index
    ST_K = ST_C + 273.15
    Ta_K = Ta_C + 273.15
    
    # canopy temperature defaults to surface temperature
    if canopy_temperature_C is None:
        canopy_temperature_C = ST_C

    canopy_temperature_K = canopy_temperature_C + 273.15

    # soil temperature defaults to surface temperature
    if soil_temperature_C is None:
        soil_temperature_C = ST_C

    soil_temperature_K = soil_temperature_C + 273.15

    # visible albedo defaults to surface albedo
    if albedo_visible is None:
        albedo_visible = albedo

    # near-infrared albedo defaults to surface albedo
    if albedo_NIR is None:
        albedo_NIR = albedo

    if SZA is None:
        SZA = calculate_SZA_from_DOY_and_hour(lat, lon, day_of_year, hour_of_day)

    geometry = rt.CoordinateArray(longitude, latitude)

    # canopy height defaults to zero
    canopy_height_meters = np.where(np.isnan(canopy_height_meters), 0, canopy_height_meters)

    # calculate saturation vapor pressure in Pascal from air temperature in Kelvin
    SVP_Pa = SVP_Pa_from_Ta_K(Ta_K)

    # calculate actual vapor pressure in Pascal from relative humidity and saturation vapor pressure
    Ea_Pa = RH * SVP_Pa

    # convert elevation to meters
    elevation_m = elevation_km * 1000

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

    results = {
        'GPP': GPP,
        'GPP_daily': GPP_daily,
        'Rn': Rn,
        'Rn_soil': Rn_soil,
        'Rn_canopy': Rn_canopy,
        'LE': LE,
        'LE_soil': LE_soil,
        'LE_canopy': LE_canopy
    }

    if isinstance(ST_C, rt.Raster):
        for key in results.keys():
            if not isinstance(results[key], rt.Raster):
                results[key] = rt.Raster(results[key], geometry=ST_C.geometry)

    return results
