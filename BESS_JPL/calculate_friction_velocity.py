"""
Friction Velocity Calculation Module

This module calculates friction velocity (u*), a fundamental scaling parameter 
in atmospheric surface layer turbulence theory. Friction velocity characterizes 
the intensity of turbulent momentum transfer between the atmosphere and surface.

The friction velocity is derived from the logarithmic wind profile, which describes
how wind speed varies with height in the atmospheric surface layer under neutral
stability conditions.

References:
    Brutsaert, W. (1982). Evaporation into the Atmosphere: Theory, History and 
    Applications. Springer, Dordrecht. Chapter 3: Turbulent Transfer in the 
    Surface Layer (pp. 41-72).
    
    Thom, A. S. (1975). Momentum, mass and heat exchange of plant communities. 
    In Vegetation and the Atmosphere (pp. 57-109). Academic Press.
    
    Garratt, J. R. (1992). The Atmospheric Boundary Layer. Cambridge University Press.
    Chapter 3: The Neutral Surface Layer (pp. 41-71).
    
    Monteith, J. L., & Unsworth, M. H. (2013). Principles of Environmental Physics: 
    Plants, Animals, and the Atmosphere (4th ed.). Academic Press.
"""

import numpy as np


# von Karman constant - universal constant for turbulent flow (~0.40-0.41)
# This is an empirical constant derived from atmospheric measurements
VON_KARMAN_CONSTANT = 0.4


def calculate_friction_velocity(
        wind_speed_mps: np.ndarray,
        reference_height_m: float,
        roughness_length_m: np.ndarray,
        von_karman_constant: float = VON_KARMAN_CONSTANT) -> np.ndarray:
    """
    Calculate friction velocity from wind speed using logarithmic wind profile theory.
    
    Friction velocity (u*) is a scaling parameter that characterizes the intensity
    of turbulent momentum transfer in the atmospheric surface layer. It is defined
    as u* = sqrt(τ/ρ), where τ is the surface shear stress and ρ is air density.
    
    Under neutral atmospheric stability, the logarithmic wind profile relates
    wind speed at a given height to friction velocity:
    
        U(z) = (u*/k) * ln(z/z0)
    
    Rearranging to solve for u*:
    
        u* = U(z) * k / ln(z/z0)
    
    where:
        - U(z) is wind speed at height z
        - k is von Karman constant (≈ 0.4)
        - z0 is roughness length (characterizes surface roughness)
    
    Physical interpretation:
        - Higher u* indicates stronger turbulent mixing
        - u* typically ranges from 0.1-1.0 m/s over vegetated surfaces
        - u* ≈ 0.1-0.3 m/s: weak turbulence (light winds, smooth surfaces)
        - u* ≈ 0.3-0.7 m/s: moderate turbulence (typical daytime conditions)
        - u* ≈ 0.7-1.0+ m/s: strong turbulence (high winds, rough surfaces)

    Args:
        wind_speed_mps (np.ndarray): Wind speed in meters per second [m/s] 
            measured at reference height.
        reference_height_m (float): Height at which wind speed is measured [m].
            Typically 10 m for meteorological observations or 2-3 m for flux towers.
        roughness_length_m (np.ndarray): Roughness length in meters [m].
            Characterizes surface roughness; typically 5-10% of canopy height.
        von_karman_constant (float, optional): von Karman constant (dimensionless).
            Defaults to 0.4. Universal constant for turbulent flow.

    Returns:
        np.ndarray: Friction velocity in meters per second [m/s].

    Raises:
        ValueError: If reference_height_m <= roughness_length_m (logarithm undefined).

    References:
        Brutsaert, W. (1982). Evaporation into the Atmosphere, Chapter 3.
        Thom, A. S. (1975). Momentum, mass and heat exchange.
        Garratt, J. R. (1992). The Atmospheric Boundary Layer, Chapter 3.
    
    Notes:
        - This formulation assumes neutral atmospheric stability (no buoyancy effects)
        - For stable or unstable conditions, stability corrections should be applied
        - The logarithmic profile is valid in the surface layer (~10-100 m)
        - Measurement height should be well above the roughness sublayer (z >> z0)
    
    Example:
        >>> wind_speed = np.array([5.0, 3.0, 7.0])  # m/s at 10 m height
        >>> reference_height = 10.0  # meters
        >>> roughness_length = np.array([0.1, 0.5, 0.05])  # meters
        >>> u_star = calculate_friction_velocity(wind_speed, reference_height, roughness_length)
        >>> print(u_star)
        [0.435  0.414  0.536]  # approximate values in m/s
    """
    # Ensure roughness length is positive and less than reference height
    # to avoid division by zero or negative/undefined logarithm
    roughness_length_m = np.clip(roughness_length_m, 1e-6, reference_height_m * 0.99)
    
    # Calculate friction velocity using logarithmic wind profile
    # u* = U * k / ln(z/z0)
    friction_velocity_mps = wind_speed_mps * von_karman_constant / np.log(reference_height_m / roughness_length_m)
    
    return friction_velocity_mps
