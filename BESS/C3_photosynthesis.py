import numpy as np


def calculate_C3_photosynthesis(
        Tf_K: np.ndarray,  # leaf temperature in Kelvin
        Ci: np.ndarray,  # intercellular CO2 concentration [umol mol-1]
        APAR: np.ndarray,  # leaf absorptance to photosynthetically active radiation [umol m-2 s-1]
        Vcmax25: np.ndarray,  # maximum carboxylation rate at 25C [umol m-2 s-1]
        Ps_Pa: np.ndarray,  # surface pressure in Pascal
        carbon_uptake_efficiency: np.ndarray) -> np.ndarray:  # intrinsic quantum efficiency for carbon uptake
    """
    photosynthesis for C3 plants
    Collatz et al., 1991
    https://www.sciencedirect.com/science/article/abs/pii/0168192391900028
    Adapted from Youngryel Ryu's code by Gregory Halverson and Robert Freepartner
    :param Tf_K: leaf temperature in Kelvin
    :param Ci: intercellular CO2 concentration [umol mol-1]
    :param APAR: leaf absorptance to photosynthetically active radiation [umol m-2 s-1]
    :param Vcmax25: maximum carboxylation rate at 25C [umol m-2 s-1]
    :param Ps_Pa: surface pressure in Pascal
    :param carbon_uptake_efficiency: intrinsic quantum efficiency for carbon uptake
    :return: net assimilation [umol m-2 s-1]
    """
    # gas constant
    R = 8.314e-3  # [kJ K-1 mol-1]

    # calculate oxygen concentration
    O2 = Ps_Pa * 0.21  # [Pa]

    # convert intercellular CO2 concentration to Pascal
    Pi = Ci * 1e-6 * Ps_Pa  # [umol mol-1] -> [Pa]

    # Temperature correction
    item = (Tf_K - 298.15) / 10
    KC25 = 30  # [Pa]
    KCQ10 = 2.1  # [-]
    KO25 = 30000  # [Pa]
    KOQ10 = 1.2  # [-]
    tao25 = 2600  # [Pa]
    taoQ10 = 0.57  # [-]
    KC = KC25 * KCQ10 ** item  # [Pa]
    KO = KO25 * KOQ10 ** item  # [Pa]
    K = KC * (1.0 + O2 / KO)  # [Pa]
    tao = tao25 * taoQ10 ** item  # [Pa]
    GammaS = O2 / (2.0 * tao)  # [Pa]
    VcmaxQ10 = 2.4  # [-]
    Vcmax_o = Vcmax25 * VcmaxQ10 ** item  # [umol m-2 s-1]
    Vcmax = Vcmax_o / (1.0 + np.exp((-220.0 + 0.703 * Tf_K) / (R * Tf_K)))  # [umol m-2 s-1]
    Rd_o = 0.015 * Vcmax  # [umol m-2 s-1]
    Rd = Rd_o * 1.0 / (1.0 + np.exp(1.3 * (Tf_K - 273.15 - 55.0)))  # [umol m-2 s-1]

    # Three limiting states
    JC = Vcmax * (Pi - GammaS) / (Pi + K)
    JE = carbon_uptake_efficiency * APAR * (Pi - GammaS) / (Pi + 2.0 * GammaS)
    JS = Vcmax / 2.0

    # Colimitation (not the case at canopy level according to DePury and Farquhar)
    a = 0.98
    b = -(JC + JE)
    c = JC * JE
    JCE = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    JCE = np.real(JCE)
    a = 0.95
    b = -(JCE + JS)
    c = JCE * JS
    JCES = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    JCES = np.real(JCES)

    # calculate net assimilation
    An = np.clip(JCES - Rd, 0, None)

    return An
