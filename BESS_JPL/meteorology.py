from typing import Tuple, Union
import numpy as np
from .calculate_bulk_aerodynamic_resistance import calculate_bulk_aerodynamic_resistance


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


# Prefix all function names with `calculate_`
def calculate_surface_pressure(elevation_m: Union[np.ndarray, float], Ta_K: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Calculate surface pressure in Pascal (Pa) based on elevation and air temperature.

    Scientific basis:
        This formula is derived from the barometric formula, which relates atmospheric pressure to altitude and temperature.

    Args:
        elevation_m (Union[np.ndarray, float]): Elevation in meters.
        Ta_K (Union[np.ndarray, float]): Air temperature in Kelvin.

    Returns:
        Union[np.ndarray, float]: Surface pressure in Pascal (Pa).

    References:
        Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop evapotranspiration - Guidelines for computing crop water requirements. FAO Irrigation and drainage paper 56.
    """
    return 101325.0 * (1.0 - 0.0065 * elevation_m / Ta_K) ** (9.807 / (0.0065 * 287.0))


def calculate_saturation_vapor_pressure(Ta_C: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Calculate the saturation vapor pressure in Pascal (Pa) from air temperature in Celsius.

    Scientific basis:
        This function uses the Magnus-Tetens approximation to calculate saturation vapor pressure in kPa, then converts it to Pa.

    Args:
        Ta_C (Union[np.ndarray, float]): Air temperature in Celsius.

    Returns:
        Union[np.ndarray, float]: Saturation vapor pressure in Pascal (Pa).

    References:
        Alduchov, O. A., & Eskridge, R. E. (1996). Improved Magnus Form Approximation of Saturation Vapor Pressure. Journal of Applied Meteorology, 35(4), 601–609.
    """
    return 0.6108 * np.exp((17.27 * Ta_C) / (Ta_C + 237.3)) * 1000


# Define additional conversion functions with detailed docstrings and citations
def calculate_water_vapor_deficit(SVP_Pa: np.ndarray, Ea_Pa: np.ndarray) -> np.ndarray:
    """
    Calculate the water vapor deficit (VPD) in Pascal (Pa).

    Args:
        SVP_Pa (np.ndarray): Saturation vapor pressure in Pascal (Pa).
        Ea_Pa (np.ndarray): Actual vapor pressure in Pascal (Pa).

    Returns:
        np.ndarray: Water vapor deficit in Pascal (Pa).
    """
    return np.clip(SVP_Pa - Ea_Pa, 0, None)

def calculate_relative_humidity(SVP_Pa: np.ndarray, Ea_Pa: np.ndarray) -> np.ndarray:
    """
    Calculate the relative humidity (RH).

    Args:
        SVP_Pa (np.ndarray): Saturation vapor pressure in Pascal (Pa).
        Ea_Pa (np.ndarray): Actual vapor pressure in Pascal (Pa).

    Returns:
        np.ndarray: Relative humidity as a fraction (0 to 1).
    """
    return np.clip(Ea_Pa / SVP_Pa, 0, 1)

def calculate_latent_heat_of_vaporization(Ta_C: np.ndarray) -> np.ndarray:
    """
    Calculate the latent heat of vaporization in Joules per kilogram (J/kg).

    Args:
        Ta_C (np.ndarray): Air temperature in Celsius.

    Returns:
        np.ndarray: Latent heat of vaporization in J/kg.
    """
    return 2.501 - (2.361e-3 * Ta_C)

def calculate_psychrometric_constant(Ps_Pa: np.ndarray, latent_heat: np.ndarray) -> np.ndarray:
    """
    Calculate the psychrometric constant in Pascal per Kelvin (Pa/K).

    Args:
        Ps_Pa (np.ndarray): Surface pressure in Pascal (Pa).
        latent_heat (np.ndarray): Latent heat of vaporization in J/kg.

    Returns:
        np.ndarray: Psychrometric constant in Pa/K.
    """
    return 0.00163 * Ps_Pa / latent_heat

def calculate_specific_humidity(Ea_Pa: np.ndarray, Ps_Pa: np.ndarray) -> np.ndarray:
    """
    Calculate the specific humidity.

    Args:
        Ea_Pa (np.ndarray): Actual vapor pressure in Pascal (Pa).
        Ps_Pa (np.ndarray): Surface pressure in Pascal (Pa).

    Returns:
        np.ndarray: Specific humidity as a fraction.
    """
    mv_ma = 0.622  # Ratio of molecular weight of water vapor to dry air
    return (mv_ma * Ea_Pa) / (Ps_Pa - 0.378 * Ea_Pa)

def calculate_air_density(Ps_Pa: np.ndarray, Ta_K: np.ndarray) -> np.ndarray:
    """
    Calculate the air density in kilograms per cubic meter (kg/m³).

    Args:
        Ps_Pa (np.ndarray): Surface pressure in Pascal (Pa).
        Ta_K (np.ndarray): Air temperature in Kelvin.

    Returns:
        np.ndarray: Air density in kg/m³.
    """
    return Ps_Pa / (287.05 * Ta_K)

# Define additional functions for remaining formulas
def calculate_inverse_relative_distance_earth_sun(day_of_year: np.ndarray) -> np.ndarray:
    """
    Calculate the inverse relative distance between Earth and Sun.

    Args:
        day_of_year (np.ndarray): Day of the year.

    Returns:
        np.ndarray: Inverse relative distance Earth-Sun.
    """
    return 1.0 + 0.033 * np.cos(2 * np.pi / 365.0 * day_of_year)

def calculate_solar_declination(day_of_year: np.ndarray) -> np.ndarray:
    """
    Calculate the solar declination angle in radians.

    Args:
        day_of_year (np.ndarray): Day of the year.

    Returns:
        np.ndarray: Solar declination in radians.
    """
    return 0.409 * np.sin(2 * np.pi / 365.0 * day_of_year - 1.39)

def calculate_sunset_hour_angle(latitude: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """
    Calculate the sunset hour angle in radians.

    Args:
        latitude (np.ndarray): Latitude in degrees.
        delta (np.ndarray): Solar declination in radians.

    Returns:
        np.ndarray: Sunset hour angle in radians.
    """
    omegaS = np.arccos(-np.tan(latitude * np.pi / 180.0) * np.tan(delta))
    omegaS = np.where(np.isnan(omegaS) | np.isinf(omegaS), 0, omegaS)
    return np.real(omegaS)

def calculate_day_length(omegaS: np.ndarray) -> np.ndarray:
    """
    Calculate the day length in hours.

    Args:
        omegaS (np.ndarray): Sunset hour angle in radians.

    Returns:
        np.ndarray: Day length in hours.
    """
    return 24.0 / np.pi * omegaS

def calculate_snapshot_radiation(dr: np.ndarray, SZA: np.ndarray) -> np.ndarray:
    """
    Calculate the snapshot radiation in W/m².

    Args:
        dr (np.ndarray): Inverse relative distance Earth-Sun.
        SZA (np.ndarray): Solar zenith angle in degrees.

    Returns:
        np.ndarray: Snapshot radiation in W/m².
    """
    return 1333.6 * dr * np.cos(SZA * np.pi / 180.0)

def calculate_daily_mean_radiation(dr: np.ndarray, omegaS: np.ndarray, latitude: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """
    Calculate the daily mean radiation in W/m².

    Args:
        dr (np.ndarray): Inverse relative distance Earth-Sun.
        omegaS (np.ndarray): Sunset hour angle in radians.
        latitude (np.ndarray): Latitude in degrees.
        delta (np.ndarray): Solar declination in radians.

    Returns:
        np.ndarray: Daily mean radiation in W/m².
    """
    return 1333.6 / np.pi * dr * (
        omegaS * np.sin(latitude * np.pi / 180.0) * np.sin(delta)
        + np.cos(latitude * np.pi / 180.0) * np.cos(delta) * np.sin(omegaS)
    )

def calculate_clear_sky_radiation(elevation_m: np.ndarray, Ra: np.ndarray) -> np.ndarray:
    """
    Calculate the clear-sky solar radiation in W/m².

    Args:
        elevation_m (np.ndarray): Elevation in meters.
        Ra (np.ndarray): Snapshot radiation in W/m².

    Returns:
        np.ndarray: Clear-sky solar radiation in W/m².
    """
    return (0.75 + 2e-5 * elevation_m) * Ra

def calculate_cloudiness_index(Rg_Wm2: np.ndarray, Rgo: np.ndarray) -> np.ndarray:
    """
    Calculate the cloudiness index.

    Args:
        Rg_Wm2 (np.ndarray): Shortwave radiation in W/m².
        Rgo (np.ndarray): Clear-sky solar radiation in W/m².

    Returns:
        np.ndarray: Cloudiness index (0 to 1).
    """
    return np.clip(1.0 - Rg_Wm2 / Rgo, 0, 1)

def calculate_all_sky_emissivity(epsa0: np.ndarray, cloudy: np.ndarray) -> np.ndarray:
    """
    Calculate the all-sky emissivity.

    Args:
        epsa0 (np.ndarray): Clear-sky emissivity.
        cloudy (np.ndarray): Cloudiness index.

    Returns:
        np.ndarray: All-sky emissivity.
    """
    return epsa0 * (1 - cloudy) + cloudy

# Define additional functions for remaining formulas

def calculate_upscaling_factor(RaDaily: np.ndarray, Ra: np.ndarray, SZA: np.ndarray) -> np.ndarray:
    """
    Calculate the upscaling factor.

    Args:
        RaDaily (np.ndarray): Daily mean radiation in W/m².
        Ra (np.ndarray): Snapshot radiation in W/m².
        SZA (np.ndarray): Solar zenith angle in degrees.

    Returns:
        np.ndarray: Upscaling factor.
    """
    SFd = np.where(RaDaily != 0, 1800.0 * Ra / (RaDaily * 3600 * 24), 1)
    SFd = np.where(SZA > 89.0, 1, SFd)
    return np.clip(SFd, None, 1)

def calculate_upscaling_factor_net_radiation(hour_of_day: np.ndarray, DL: np.ndarray) -> np.ndarray:
    """
    Calculate the upscaling factor for net radiation.

    Args:
        hour_of_day (np.ndarray): Hour of the day.
        DL (np.ndarray): Day length in hours.

    Returns:
        np.ndarray: Upscaling factor for net radiation.
    """
    dT = np.abs(hour_of_day - 12.0)
    return 1.5 / (np.pi * np.sin((DL - 2.0 * dT) / (2.0 * DL) * np.pi)) * DL / 24.0

def calculate_stress_factor(RH: np.ndarray, VPD_Pa: np.ndarray) -> np.ndarray:
    """
    Calculate the stress factor.

    Args:
        RH (np.ndarray): Relative humidity as a fraction (0 to 1).
        VPD_Pa (np.ndarray): Vapor pressure deficit in Pascal (Pa).

    Returns:
        np.ndarray: Stress factor.
    """
    return RH ** (VPD_Pa / 1000.0)

# Update calculate_desTa to match the original formula
def calculate_desTa(SVP_Pa: np.ndarray, Ta_C: np.ndarray) -> np.ndarray:
    """
    Calculate the first derivative of saturated vapor pressure with respect to temperature.

    Args:
        SVP_Pa (np.ndarray): Saturation vapor pressure in Pascal (Pa).
        Ta_C (np.ndarray): Air temperature in Celsius.

    Returns:
        np.ndarray: First derivative of saturated vapor pressure.
    """
    return SVP_Pa * 4098.0 * pow((Ta_C + 237.3), -2)

# Update calculate_ddesTa to match the original formula
def calculate_ddesTa(desTa: np.ndarray, SVP_Pa: np.ndarray, Ta_C: np.ndarray) -> np.ndarray:
    """
    Calculate the second derivative of saturated vapor pressure with respect to temperature.

    Args:
        desTa (np.ndarray): First derivative of saturated vapor pressure.
        SVP_Pa (np.ndarray): Saturation vapor pressure in Pascal (Pa).
        Ta_C (np.ndarray): Air temperature in Celsius.

    Returns:
        np.ndarray: Second derivative of saturated vapor pressure.
    """
    return 4098.0 * (desTa * pow((Ta_C + 237.3), -2) + (-2) * SVP_Pa * pow((Ta_C + 237.3), -3))

# Update calculate_specific_heat_of_air to include q and Cpd
def calculate_specific_heat_of_air(Ta_C: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Calculate the specific heat of air in J/kg/K.

    Args:
        Ta_C (np.ndarray): Air temperature in Celsius.
        q (np.ndarray): Specific humidity as a fraction.

    Returns:
        np.ndarray: Specific heat of air in J/kg/K.
    """
    Cpd = 1005 + (Ta_C + 273.15 - 250) ** 2 / 3364  # Specific heat of dry air
    return Cpd * (1 + 0.84 * q)

# Update the meteorology function to use the corrected helper functions
def meteorology(
        day_of_year: np.ndarray,  # day of year
        hour_of_day: np.ndarray,  # hour of day
        latitude: np.ndarray,  # latitude
        elevation_m: np.ndarray,  # elevation in meters
        SZA: np.ndarray,  # solar zenith angle in degrees
        Ta_K: np.ndarray,  # air temperature in Kelvin
        Ea_Pa: np.ndarray,  # vapor pressure in Pascal
        Rg_Wm2: np.ndarray,  # shortwave radiation in W/m2
        wind_speed_mps: np.ndarray,  # wind speed in meters per second
        canopy_height_meters: np.ndarray):  # canopy height in meters
    """
    Meteorological calculations for Breathing Earth System Simulator
    Adapted from Youngryel Ryu's MATLAB code by Gregory Halverson and Robert Freepartner
    :return: Dictionary of meteorological outputs with keys identical to variable names
    """
    # Use the refactored helper functions
    Ps_Pa = calculate_surface_pressure(elevation_m, Ta_K)
    Ta_C = Ta_K - 273.16  # Convert Kelvin to Celsius
    SVP_Pa = calculate_saturation_vapor_pressure(Ta_C)
    VPD_Pa = calculate_water_vapor_deficit(SVP_Pa, Ea_Pa)
    RH = calculate_relative_humidity(SVP_Pa, Ea_Pa)
    latent_heat = calculate_latent_heat_of_vaporization(Ta_C)
    gamma = calculate_psychrometric_constant(Ps_Pa, latent_heat)
    q = calculate_specific_humidity(Ea_Pa, Ps_Pa)
    rhoa = calculate_air_density(Ps_Pa, Ta_K)

    dr = calculate_inverse_relative_distance_earth_sun(day_of_year)
    delta = calculate_solar_declination(day_of_year)
    omegaS = calculate_sunset_hour_angle(latitude, delta)
    DL = calculate_day_length(omegaS)
    Ra = calculate_snapshot_radiation(dr, SZA)
    RaDaily = calculate_daily_mean_radiation(dr, omegaS, latitude, delta)
    Rgo = calculate_clear_sky_radiation(elevation_m, Ra)
    cloudy = calculate_cloudiness_index(Rg_Wm2, Rgo)
    epsa0 = 0.605 + 0.048 * (Ea_Pa / 100) ** 0.5  # Clear-sky emissivity
    epsa = calculate_all_sky_emissivity(epsa0, cloudy)

    SFd = calculate_upscaling_factor(RaDaily, Ra, SZA)
    R, Rs, Rc = calculate_bulk_aerodynamic_resistance(wind_speed_mps, canopy_height_meters)

    # Bisht et al., 2005
    DL = DL - 1.5

    SFd2 = calculate_upscaling_factor_net_radiation(hour_of_day, DL)
    fStress = calculate_stress_factor(RH, VPD_Pa)

    desTa = calculate_desTa(SVP_Pa, Ta_C)
    ddesTa = calculate_ddesTa(desTa, SVP_Pa, Ta_C)
    Cp = calculate_specific_heat_of_air(Ta_C, q)

    # Include R, Rc, Rs, and SFd in the return dictionary
    return {
        "Ps_Pa": Ps_Pa,
        "VPD_Pa": VPD_Pa,
        "RH": RH,
        "desTa": desTa,
        "ddesTa": ddesTa,
        "gamma": gamma,
        "Cp": Cp,
        "rhoa": rhoa,
        "epsa": epsa,
        "R": R,
        "Rc": Rc,
        "Rs": Rs,
        "SFd": SFd,
        "SFd2": SFd2,
        "DL": DL,
        "Ra": Ra,
        "fStress": fStress
    }
