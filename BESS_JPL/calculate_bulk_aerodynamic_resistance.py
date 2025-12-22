"""
Bulk Aerodynamic Resistance Calculation Module

This module calculates bulk aerodynamic resistance and related parameters for
energy balance and turbulent flux models.

References:
    Ryu, Y., Baldocchi, D. D., Ma, S., & Hehn, T. (2008). Interannual variability of 
    evapotranspiration and energy exchange over an annual grassland in California. 
    Journal of Geophysical Research, 113, D09104. doi:10.1029/2007JD009263
    
    Thom, A. S. (1975). Momentum, mass and heat exchange of plant communities. 
    In Vegetation and the Atmosphere (pp. 57-109). Academic Press.
    
    Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). 
    Crop evapotranspiration - Guidelines for computing crop water requirements. 
    FAO Irrigation and drainage paper 56.
"""

from typing import Tuple
import numpy as np
from .calculate_friction_velocity import calculate_friction_velocity


# Global constants for aerodynamic resistance calculations
# Minimum wind speed to prevent division by zero and represent natural convection
# Even in "still" conditions, buoyancy-driven convection typically maintains ~0.1-0.5 m/s
# Reference: Allen et al. (1998) FAO-56, Thom (1975)
MIN_WIND_SPEED_MPS = 0.1

# Maximum aerodynamic resistance representing the upper limit when forced convection
# becomes negligible and free (buoyancy-driven) convection dominates
# Typical values in literature range from 500-1000 s/m
# Reference: Ryu et al. (2008), SEBS model (Su, 2002)
MAX_AERODYNAMIC_RESISTANCE_SM = 1000.0

# Standard meteorological reference height for wind speed measurements
# WMO convention: 10 m above ground level
# Reference: WMO (2018) Guide to Instruments and Methods of Observation
REFERENCE_HEIGHT_M = 10.0

# von Karman constant - universal constant for turbulent flow
VON_KARMAN_CONSTANT = 0.4


def calculate_bulk_aerodynamic_resistance(
        wind_speed_mps: np.ndarray, 
        canopy_height_meters: np.ndarray,
        min_wind_speed: float = MIN_WIND_SPEED_MPS,
        max_resistance: float = MAX_AERODYNAMIC_RESISTANCE_SM,
        reference_height_m: float = REFERENCE_HEIGHT_M) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate bulk aerodynamic resistance and related parameters for turbulent transfer.
    
    Aerodynamic resistance represents the difficulty of transferring heat, water vapor, 
    and momentum through the air boundary layer above a surface. It depends on wind speed
    and surface roughness (characterized by canopy height).
    
    Physical basis:
        - When wind speed → 0, aerodynamic resistance → ∞ (no turbulent mixing)
        - In practice, buoyancy-driven convection sets an upper limit (~1000 s/m)
        - Roughness length (z0) is approximated as 5% of canopy height
        - Friction velocity (u*) characterizes turbulent momentum transfer

    Args:
        wind_speed_mps (np.ndarray): Wind speed in meters per second [m/s] at reference height.
        canopy_height_meters (np.ndarray): Canopy height in meters [m].
        min_wind_speed (float, optional): Minimum wind speed to prevent division by zero
            and represent natural convection. Defaults to 0.1 m/s.
        max_resistance (float, optional): Maximum aerodynamic resistance representing
            the upper limit when free convection dominates. Defaults to 1000 s/m.
        reference_height_m (float, optional): Height at which wind speed is measured.
            Defaults to 10 m (WMO standard for meteorological observations).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - R: Bulk aerodynamic resistance [s/m]
            - Rs: Surface aerodynamic resistance [s/m] (= 0.5 * R)
            - Rc: Canopy aerodynamic resistance [s/m] (= R)

    References:
        Ryu, Y., Baldocchi, D. D., Ma, S., & Hehn, T. (2008). Equation (2-4).
        Thom, A. S. (1975). Momentum, mass and heat exchange of plant communities.
        Allen, R. G., et al. (1998). FAO Irrigation and drainage paper 56.
    
    Notes:
        - Units: Aerodynamic resistance is in s/m (seconds per meter)
        - Typical range: 10-50 s/m (windy), 50-200 s/m (moderate), 200-500 s/m (calm)
        - Values > 1000 s/m indicate negligible turbulent transport
        - For flux tower data (often measured at 2-3 m), adjust reference_height_m accordingly
    """
    k = VON_KARMAN_CONSTANT
    
    # Apply minimum wind speed to avoid division by zero and represent natural convection
    # Even in "calm" conditions, buoyancy-driven mixing maintains some air movement
    wind_speed_mps = np.clip(wind_speed_mps, min_wind_speed, None)
    # Roughness length (z0) approximated as 5% of canopy height, with minimum of 0.05 m
    # This is a common approximation for vegetation surfaces
    z0 = np.clip(canopy_height_meters * 0.05, 0.05, None)
    
    # Calculate friction velocity using logarithmic wind profile
    # Characterizes turbulent momentum transfer intensity
    ustar = calculate_friction_velocity(wind_speed_mps, reference_height_m, z0, k)
    
    # Bulk aerodynamic resistance - Eq. (2-4) from Ryu et al. (2008)
    # First term: resistance in the inertial sublayer
    # Second term: resistance in the roughness sublayer
    R = wind_speed_mps / (ustar * ustar) + 2.0 / (k * ustar)
    
    # Cap resistance at maximum value representing transition to free convection
    # When forced convection (wind) is weak, buoyancy-driven convection takes over
    R = np.clip(R, None, max_resistance)
    
    # Surface and canopy resistances derived from bulk resistance
    # These are used in separate energy balance calculations
    Rs = 0.5 * R
    Rc = R
    
    return R, Rs, Rc
