import numpy as np

def canopy_longwave_radiation(
        LAI: np.ndarray,
        SZA: np.ndarray,
        Ts_K: np.ndarray,
        Tf_K: np.ndarray,
        Ta_K: np.ndarray,
        epsa: np.ndarray,
        epsf: float,
        epss: float,
        ALW_min: float = None,
        intermediate_min: float = None,
        intermediate_max: float = None):
    """
    =============================================================================

    Module     : Canopy longwave radiation transfer
    Description: This function calculates the absorbed longwave radiation by sunlit leaves, shaded leaves, and soil within a vegetative canopy. It also computes the longwave radiation flux densities from air, soil, and foliage. The calculations are based on radiative transfer principles and empirical coefficients derived from literature.

    Inputs:
        LAI (np.ndarray): Leaf Area Index [-], a dimensionless measure of leaf area per unit ground area.
        SZA (np.ndarray): Solar Zenith Angle [degrees], the angle between the sun's rays and the vertical.
        Ts_K (np.ndarray): Soil temperature [K].
        Tf_K (np.ndarray): Foliage temperature [K].
        Ta_K (np.ndarray): Air temperature [K].
        epsa (np.ndarray): Clear-sky emissivity [-].
        epsf (float): Foliage emissivity [-].
        epss (float): Soil emissivity [-].
        ALW_min (float, optional): Minimum threshold for absorbed longwave radiation.
        intermediate_min (float, optional): Minimum threshold for intermediate calculations.
        intermediate_max (float, optional): Maximum threshold for intermediate calculations.

    Outputs:
        ALW_sunlit (np.ndarray): Absorbed longwave radiation by sunlit leaves [W/m^2].
        ALW_shaded (np.ndarray): Absorbed longwave radiation by shaded leaves [W/m^2].
        ALW_soil (np.ndarray): Absorbed longwave radiation by soil [W/m^2].
        Ls (np.ndarray): Longwave radiation flux density from soil [W/m^2].
        La (np.ndarray): Longwave radiation flux density from air [W/m^2].
        Lf (np.ndarray): Longwave radiation flux density from foliage [W/m^2].

    References:
        - Ryu, Y., Baldocchi, D. D., Kobayashi, H., et al. (2011). Integration of MODIS land and atmosphere products with a coupled-process model to estimate gross primary productivity and evapotranspiration. Remote Sensing of Environment, 115(8), 1866-1875.
        - Wang, Y., Law, R. M., Davies, H. L., McGregor, J. L., & Abramowitz, G. (2006). The CSIRO Atmosphere Biosphere Land Exchange (CABLE) model for use in climate models and as an offline model. Geoscientific Model Development.

    Conversion from MatLab by Robert Freepartner, JPL/Raytheon/JaDa Systems
    March 2020

    =============================================================================
    """
    # Clip SZA to a maximum of 89 degrees to avoid extreme values
    SZA = np.clip(SZA, None, 89)

    # Extinction coefficient for beam radiation (kb) based on solar zenith angle
    # Derived from Table A1 in Ryu et al. (2011)
    kb = 0.5 / np.cos(SZA * np.pi / 180.0)

    # Extinction coefficient for longwave radiation (kd)
    # Empirical value from Table A1 in Ryu et al. (2011)
    kd = 0.78

    # Stefan-Boltzmann constant [W m^-2 K^-4]
    sigma = 5.670373e-8

    # Longwave radiation flux densities from air, soil, and foliage
    # Calculated using the Stefan-Boltzmann law: L = eps * sigma * T^4
    La = np.clip(epsa * sigma * Ta_K ** 4, 0, None)  # Air
    Ls = np.clip(epss * sigma * Ts_K ** 4, 0, None)  # Soil
    Lf = np.clip(epsf * sigma * Tf_K ** 4, 0, None)  # Foliage

    # Precompute kd * LAI for efficiency
    kd_LAI = kd * LAI

    # Difference in longwave radiation flux densities between soil and foliage
    soil_leaf_difference = Ls - Lf
    
    if intermediate_min is not None:
        soil_leaf_difference = np.clip(soil_leaf_difference, intermediate_min, None)

    # Difference in longwave radiation flux densities between air and foliage
    air_leaf_difference = La - Lf

    if intermediate_min is not None:
        air_leaf_difference = np.clip(air_leaf_difference, intermediate_min, intermediate_max)

    # Absorbed longwave radiation by sunlit leaves
    # Numerator and denominator derived from Eq. (44) in Wang et al. (2006)
    numerator = soil_leaf_difference * kd * (np.exp(-kd_LAI) - np.exp(-kb * LAI)) / (kd - kb)
    numerator += kd * air_leaf_difference * (1.0 - np.exp(-(kb + kd) * LAI))
    denominator = kd + kb

    if ALW_min is not None:
        numerator = np.clip(numerator, ALW_min, None)

    ALW_sunlit = numerator / denominator

    # Combined longwave radiation flux densities from soil, air, and foliage
    soil_air_leaf = Ls + La - 2 * Lf

    if intermediate_min is not None or intermediate_max is not None:
        soil_air_leaf = np.clip(soil_air_leaf, intermediate_min, intermediate_max)

    # Absorbed longwave radiation by shaded leaves
    # Derived from Eq. (45) in Wang et al. (2006)
    ALW_shaded = (1.0 - np.exp(-kd_LAI)) * soil_air_leaf - ALW_sunlit

    if ALW_min is not None:
        ALW_shaded = np.clip(ALW_shaded, ALW_min, None)

    # Absorbed longwave radiation by soil
    # Derived from Eq. (41) in Wang et al. (2006)
    ALW_soil = (1.0 - np.exp(-kd_LAI)) * Lf + np.exp(-kd_LAI) * La

    if ALW_min is not None:
        ALW_soil = np.clip(ALW_soil, ALW_min, None)

    return ALW_sunlit, ALW_shaded, ALW_soil, Ls, La, Lf
