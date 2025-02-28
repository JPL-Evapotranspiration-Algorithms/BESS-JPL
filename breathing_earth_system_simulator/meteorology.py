from typing import Tuple
import numpy as np


def SVP_kPa_from_Ta_C(Ta_C: np.ndarray) -> np.ndarray:
    """
    saturation vapor pressure in kPa from air temperature in celsius
    :param Ta_C: air temperature in celsius
    :return: saturation vapor pressure in kiloPascal
    """
    return 0.611 * np.exp((Ta_C * 17.27) / (Ta_C + 237.7))


def SVP_Pa_from_Ta_K(Ta_K: np.ndarray) -> np.ndarray:
    """
    saturation vapor pressure in kPa from air temperature in celsius
    :param Ta_K: air temperature in Kelvin
    :return: saturation vapor pressure in kiloPascal
    """
    Ta_C = Ta_K - 273.15
    SVP_kPa = SVP_kPa_from_Ta_C(Ta_C)
    SVP_Pa = SVP_kPa * 1000

    return SVP_Pa


def meteorology(
        day_of_year: np.ndarray,  # day of year
        hour_of_day: np.ndarray,  # hour of day
        latitude: np.ndarray,  # latitude
        elevation_m: np.ndarray,  # elevation in meters
        SZA: np.ndarray,  # solar zenith angle in degrees
        Ta_K: np.ndarray,  # air temperature in Kelvin
        Ea_Pa: np.ndarray,  # vapor pressure in Pascal
        Rg: np.ndarray,  # shortwave radiation in W/m2
        wind_speed_mps: np.ndarray,  # wind speed in meters per second
        canopy_height_meters: np.ndarray):  # canopy height in meters
    """
    Meteorological calculations for Breathing Earth System Simulator
    Adapted from Youngryel Ryu's MATLAB code by Gregory Halverson and Robert Freepartner
    :param day_of_year: day of year
    :param hour_of_day: hour of day
    :param latitude: latitude
    :param elevation_m: elevation in meters
    :param SZA: solar zenith angle in degrees
    :param Ta_K: air temperature in Kelvin
    :param Ea_Pa: vapor pressure in Pascal
    :param Rg: shortwave radiation in W/m2
    :param wind_speed_mps: wind speed in meters per second
    :param canopy_height_meters: canopy height in meters
    :return: 
    Ps_Pa surface pressure in Pascal
    VPD_Pa water vapor deficit in Pascal
    RH relative humidity as a fraction
    desTa 1st derivative of saturated vapor pressure
    ddesTa 2nd derivative of saturated vapor pressure
    gamma psychrometric constant in Pa K-1
    Cp specific heat of air in J kg-1 K-1
    rhoa air density in kg m-3
    epsa all-sky emissivity
    R
    Rc
    Rs
    SFd
    SFd2
    DL
    Ra
    fStress
    """""
    # Allen et al., 1998 (FAO)

    # surface pressure
    Ps_Pa = 101325.0 * (1.0 - 0.0065 * elevation_m / Ta_K) ** (9.807 / (0.0065 * 287.0))  # [Pa]

    # air temperature in Celsius
    Ta_C = Ta_K - 273.16  # [Celsius]

    # dewpoint temperature in Celsius
    # Td_C = Td_K - 273.16  # [Celsius]

    # ambient vapour pressure
    # Ea_Pa = 0.6108 * np.exp((17.27 * Td_C) / (Td_C + 237.3)) * 1000  # [Pa]

    # saturated vapour pressure
    SVP_Pa = 0.6108 * np.exp((17.27 * Ta_C) / (Ta_C + 237.3)) * 1000  # [Pa]

    # water vapour deficit
    VPD_Pa = np.clip(SVP_Pa - Ea_Pa, 0, None)  # [Pa]

    # relative humidity
    RH = np.clip(Ea_Pa / SVP_Pa, 0, 1)  # [-]

    # 1st derivative of saturated vapour pressure
    desTa = SVP_Pa * 4098.0 * pow((Ta_C + 237.3), (-2))  # [Pa K-1]

    # 2nd derivative of saturated vapour pressure
    ddesTa = 4098.0 * (desTa * pow((Ta_C + 237.3), (-2)) + (-2) * SVP_Pa * pow((Ta_C + 237.3), (-3)))  # [Pa K-2]

    # latent Heat of Vaporization
    latent_heat = 2.501 - (2.361e-3 * Ta_C)  # [J kg-1]

    # psychrometric constant
    gamma = 0.00163 * Ps_Pa / latent_heat  # [Pa K-1]

    # specific heat
    # this formula for specific heat was generating extreme negative values that threw off the energy balance calculation
    # Cp = 0.24 * 4185.5 * (1.0 + 0.8 * (0.622 * Ea_Pa / (Ps_Pa - Ea_Pa)))  # [J kg-1 K-1]

    # ratio molecular weight of water vapour dry air
    mv_ma = 0.622  # [-] (Wiki)

    # specific humidity
    q = (mv_ma * Ea_Pa) / (Ps_Pa - 0.378 * Ea_Pa)  # 3 [-] (Garratt, 1994)

    # specific heat of dry air
    Cpd = 1005 + (Ta_K - 250) ** 2 / 3364  # [J kg-1 K-1] (Garratt, 1994)

    # specific heat of air
    Cp = Cpd * (1 + 0.84 * q)  # [J kg-1 K-1] (Garratt, 1994)

    # virtual temperature
    # Tv_K = Ta_K * 1.0 / (1 - 0.378 * Ea_Pa / Ps_Pa)  # [K]

    # air density
    # rhoa = Ps_Pa / (287.0 * Tv_K)  # [kg m-3]

    # air density
    rhoa = Ps_Pa / (287.05 * Ta_K)  # [kg m-3] (Garratt, 1994)

    # inverse relative distance Earth-Sun
    dr = 1.0 + 0.033 * np.cos(2 * np.pi / 365.0 * day_of_year)  # [-]

    # solar declination
    delta = 0.409 * np.sin(2 * np.pi / 365.0 * day_of_year - 1.39)  # [rad]

    # sunset hour angle
    # Note: the value for arccos may be invalid (< -1.0 or > 1.0).
    # This will result in NaN values in omegaS.
    omegaS = np.arccos(-np.tan(latitude * np.pi / 180.0) * np.tan(delta))  # [rad]

    # omegaS[np.logical_or(np.isnan(omegaS), np.isinf(omegaS))] = 0.0
    omegaS = np.where(np.isnan(omegaS) | np.isinf(omegaS), 0, omegaS)
    omegaS = np.real(omegaS)

    # Day length
    DL = 24.0 / np.pi * omegaS

    # snapshot radiation
    Ra = 1333.6 * dr * np.cos(SZA * np.pi / 180.0)

    # Daily mean radiation
    RaDaily = 1333.6 / np.pi * dr * (omegaS * np.sin(latitude * np.pi / 180.0) * np.sin(delta)
                                     + np.cos(latitude * np.pi / 180.0) * np.cos(delta) * np.sin(omegaS))
    # clear-sky solar radiation
    Rgo = (0.75 + 2e-5 * elevation_m) * Ra  # [W m-2]

    # Choi et al., 2008: The Crawford and Duchon’s cloudiness factor with Brunt equation is recommended.

    # cloudy index
    cloudy = 1.0 - Rg / Rgo  # [-]
    # cloudy[cloudy < 0] = 0
    # cloudy[cloudy > 1] = 1
    cloudy = np.clip(cloudy, 0, 1)

    # clear-sky emissivity
    epsa0 = 0.605 + 0.048 * (Ea_Pa / 100) ** 0.5  # [-]

    # all-sky emissivity
    epsa = epsa0 * (1 - cloudy) + cloudy  # [-]

    # Ryu et al. 2008 2012

    # Upscaling factor
    # non0msk = RaDaily != 0
    # SFd = np.empty(np.shape(RaDaily))
    # SFd[non0msk] = 1800.0 * Ra[non0msk] / (RaDaily[non0msk] * 3600 * 24)
    # SFd[np.logical_not(non0msk)] = 1.0
    SFd = np.where(RaDaily != 0, 1800.0 * Ra / (RaDaily * 3600 * 24), 1)
    # SFd[SZA > 89.0] = 1.0
    SFd = np.where(SZA > 89.0, 1, SFd)
    # SFd[SFd > 1.0] = 1.0
    SFd = np.clip(SFd, None, 1)

    # bulk aerodynamic resistance
    k = 0.4  # von Karman constant
    z0 = np.clip(canopy_height_meters * 0.05, 0.05, None)
    ustar = wind_speed_mps * k / (np.log(10.0 / z0))  # Stability item ignored
    R = wind_speed_mps / (ustar * ustar) + 2.0 / (k * ustar)  # Eq. (2-4) in Ryu et al 2008
    R = np.clip(R, None, 1000)
    Rs = 0.5 * R
    Rc = R  # was: Rc = 0.5 * R * 2

    # Bisht et al., 2005
    DL = DL - 1.5

    # Time difference between overpass and midday
    dT = np.abs(hour_of_day - 12.0)

    # Upscaling factor for net radiation
    SFd2 = 1.5 / (np.pi * np.sin((DL - 2.0 * dT) / (2.0 * DL) * np.pi)) * DL / 24.0

    fStress = RH ** (VPD_Pa / 1000.0)

    return Ps_Pa, VPD_Pa, RH, desTa, ddesTa, gamma, Cp, rhoa, epsa, R, Rc, Rs, SFd, SFd2, DL, Ra, fStress
