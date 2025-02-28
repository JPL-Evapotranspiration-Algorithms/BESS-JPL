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
    Input      : leaf area index (LAI) [-],
               : extinction coefficient for longwave radiation (kd) [m-1],
               : extinction coefficient for beam radiation (kb) [m-1],
               : air temperature (Ta) [K],
               : soil temperature (Ts) [K],
               : foliage temperature (Tf) [K],
               : clear-sky emissivity (epsa) [-],
               : soil emissivity (epss) [-],
               : foliage emissivity (epsf) [-].
    Output     : total absorbed LW by sunlit leaves (Q_LSun),
               : total absorbed LW by shade leaves (Q_LSh).
    References : Wang, Y., Law, R. M., Davies, H. L., McGregor, J. L., & Abramowitz, G. (2006).
                 The CSIRO Atmosphere Biosphere Land Exchange (CABLE) model for use in climate models and as an offline model.


    Conversion from MatLab by Robert Freepartner, JPL/Raytheon/JaDa Systems
    March 2020

    =============================================================================
    """
    # SZA[SZA > 89.0] = 89.0
    SZA = np.clip(SZA, None, 89)
    kb = 0.5 / np.cos(SZA * np.pi / 180.0)  # Table A1 in Ryu et al 2011
    kd = 0.78  # Table A1 in Ryu et al 2011

    # Stefan_Boltzmann_constant
    sigma = 5.670373e-8  # [W m-2 K-4] (Wiki)

    # Long wave radiation flux densities from air, soil and leaf
    La = np.clip(epsa * sigma * Ta_K ** 4, 0, None)
    Ls = np.clip(epss * sigma * Ts_K ** 4, 0, None)
    Lf = np.clip(epsf * sigma * Tf_K ** 4, 0, None)

    # For simplicity
    kd_LAI = kd * LAI

    soil_leaf_difference = Ls - Lf

    if intermediate_min is not None:
        soil_leaf_difference = np.clip(soil_leaf_difference, intermediate_min, None)

    air_leaf_difference = La - Lf

    if intermediate_min is not None:
        air_leaf_difference = np.clip(air_leaf_difference, intermediate_min, intermediate_max)
    
    # Absorbed longwave radiation by sunlit leaves
    numerator = soil_leaf_difference * kd * (np.exp(-kd_LAI) - np.exp(-kb * LAI)) / (kd - kb) + kd * air_leaf_difference * (1.0 - np.exp(-(kb + kd) * LAI))
    denominator = kd + kb
    
    if ALW_min is not None:
        numerator = np.clip(numerator, ALW_min, None)

    ALW_sunlit = numerator / denominator  # Eq. (44)

    soil_air_leaf = Ls + La - 2 * Lf

    if intermediate_min is not None or intermediate_max is not None:
        soil_air_leaf = np.clip(soil_air_leaf, intermediate_min, intermediate_max)
    
    # Absorbed longwave radiation by shaded leaves
    ALW_shaded = (1.0 - np.exp(-kd_LAI)) * soil_air_leaf - ALW_sunlit  # Eq. (45)
    
    if ALW_min is not None:
        ALW_shaded = np.clip(ALW_shaded, ALW_min, None)

    # Absorbed longwave radiation by soil
    ALW_soil = (1.0 - np.exp(-kd_LAI)) * Lf + np.exp(-kd_LAI) * La  # Eq. (41)

    if ALW_min is not None:
        ALW_soil = np.clip(ALW_soil, ALW_min, None)

    return ALW_sunlit, ALW_shaded, ALW_soil, Ls, La, Lf
