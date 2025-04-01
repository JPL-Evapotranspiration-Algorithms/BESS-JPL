"""
This module calculates sunrise hour and daylight hours.

Developed by Gregory Halverson in the Jet Propulsion Laboratory Year-Round Internship Program (Columbus Technologies and Services, ANRE Tech.), in coordination with the ECOSTRESS mission and master's thesis studies at California State University, Northridge.
"""
import warnings
from numpy import tan, cos, sin, pi, arccos, where, radians, degrees

__author__ = 'Gregory Halverson'


def day_angle_rad_from_doy(doy):
    """
    This function calculates day angle in radians from day of year between 1 and 365.
    """
    return (2 * pi * (doy - 1)) / 365


def solar_dec_deg_from_day_angle_rad(day_angle_rad):
    """
    This function calculates solar declination in degrees from day angle in radians. 
    """
    return (0.006918 - 0.399912 * cos(day_angle_rad) + 0.070257 * sin(day_angle_rad) - 0.006758 * cos(
        2 * day_angle_rad) + 0.000907 * sin(2 * day_angle_rad) - 0.002697 * cos(3 * day_angle_rad) + 0.00148 * sin(
        3 * day_angle_rad)) * (180 / pi)


def sha_deg_from_doy_lat(doy, latitude):
    """
    This function calculates sunrise hour angle in degrees from latitude in degrees and day of year between 1 and 365. 
    """
    # calculate day angle in radians
    day_angle_rad = day_angle_rad_from_doy(doy)

    # calculate solar declination in degrees
    solar_dec_deg = solar_dec_deg_from_day_angle_rad(day_angle_rad)

    # convert latitude to radians
    latitude_rad = radians(latitude)

    # convert solar declination to radians
    solar_dec_rad = radians(solar_dec_deg)

    # calculate cosine of sunrise angle at latitude and solar declination
    # need to keep the cosine for polar correction
    sunrise_cos = -tan(latitude_rad) * tan(solar_dec_rad)

    # calculate sunrise angle in radians from cosine
    warnings.filterwarnings('ignore')
    sunrise_rad = arccos(sunrise_cos)
    warnings.resetwarnings()

    # convert to degrees
    sunrise_deg = degrees(sunrise_rad)

    # apply polar correction
    sunrise_deg = where(sunrise_cos >= 1, 0, sunrise_deg)
    sunrise_deg = where(sunrise_cos <= -1, 180, sunrise_deg)

    return sunrise_deg


def sunrise_from_sha(sha_deg):
    """
    This function calculates sunrise hour from sunrise hour angle in degrees. 
    """
    return 12.0 - (sha_deg / 15.0)


def daylight_from_sha(sha_deg):
    """
    This function calculates daylight hours from sunrise hour angle in degrees.
    """
    return (2.0 / 15.0) * sha_deg
