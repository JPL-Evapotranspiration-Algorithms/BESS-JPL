import numpy as np


def calculate_C3_photosynthesis(
        Tf_K: np.ndarray,  # leaf temperature in Kelvin (K)
        Ci: np.ndarray,  # intercellular CO2 concentration in micromoles per mole (μmol mol⁻¹)
        APAR_μmolm2s1: np.ndarray,  # absorbed photosynthetically active radiation in micromoles per square meter per second (μmol m⁻² s⁻¹)
        Vcmax25: np.ndarray,  # maximum carboxylation rate at 25°C in micromoles per square meter per second (μmol m⁻² s⁻¹)
        Ps_Pa: np.ndarray,  # surface pressure in Pascals (Pa)
        carbon_uptake_efficiency: np.ndarray) -> dict:  # intrinsic quantum efficiency for carbon uptake (unitless)
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
    :return: dictionary containing photosynthesis, respiration, and net assimilation rates
    """

    # Universal gas constant used in temperature corrections
    # Units: kilojoules per Kelvin per mole (kJ K⁻¹ mol⁻¹)
    R = 8.314e-3

    # Calculate oxygen concentration, assuming oxygen is 21% of the atmospheric pressure
    # Units: Pascals (Pa)
    O2 = Ps_Pa * 0.21

    # Convert intercellular CO2 concentration to partial pressure in Pascal
    # Units: Pascals (Pa)
    Pi = Ci * 1e-6 * Ps_Pa

    # Calculate temperature difference from 25°C (298.15 K), scaled by 10 for Q10 calculations
    # Unitless
    item = (Tf_K - 298.15) / 10

    # Michaelis-Menten constant for CO2 at 25°C
    # Units: Pascals (Pa)
    KC25 = 30

    # Michaelis-Menten constant for O2 at 25°C
    # Units: Pascals (Pa)
    KO25 = 30000

    # Specificity factor of RuBisCO at 25°C
    # Units: Pascals (Pa)
    tao25 = 2600

    # Q10 value for temperature sensitivity of KC
    # Unitless
    KCQ10 = 2.1

    # Q10 value for temperature sensitivity of KO
    # Unitless
    KOQ10 = 1.2

    # Q10 value for temperature sensitivity of tao
    # Unitless
    taoQ10 = 0.57

    # Adjusted Michaelis-Menten constant for CO2 at the given temperature
    # Units: Pascals (Pa)
    KC = KC25 * KCQ10 ** item

    # Adjusted Michaelis-Menten constant for O2 at the given temperature
    # Units: Pascals (Pa)
    KO = KO25 * KOQ10 ** item

    # Effective Michaelis-Menten constant for CO2 considering O2 competition
    # Units: Pascals (Pa)
    K = KC * (1.0 + O2 / KO)

    # Adjusted specificity factor of RuBisCO at the given temperature
    # Units: Pascals (Pa)
    tao = tao25 * taoQ10 ** item

    # CO2 compensation point in the absence of dark respiration
    # Units: Pascals (Pa)
    GammaS = O2 / (2.0 * tao)

    # Q10 value for temperature sensitivity of Vcmax
    # Unitless
    VcmaxQ10 = 2.4

    # Adjusted Vcmax at the given temperature without activation energy correction
    # Units: micromoles per square meter per second (μmol m⁻² s⁻¹)
    Vcmax_o = Vcmax25 * VcmaxQ10 ** item

    # Adjusted Vcmax at the given temperature with activation energy correction
    # Units: micromoles per square meter per second (μmol m⁻² s⁻¹)
    Vcmax = Vcmax_o / (1.0 + np.exp((-220.0 + 0.703 * Tf_K) / (R * Tf_K)))

    # Base dark respiration rate as a fraction of Vcmax
    # Units: micromoles per square meter per second (μmol m⁻² s⁻¹)
    Rd_o = 0.015 * Vcmax

    # Adjusted dark respiration rate at the given temperature
    # Units: micromoles per square meter per second (μmol m⁻² s⁻¹)
    respiration_C3_μmolm2s1 = Rd_o * 1.0 / (1.0 + np.exp(1.3 * (Tf_K - 273.15 - 55.0)))

    # Rubisco-limited rate, dependent on CO2 availability and RuBisCO activity
    # Units: micromoles per square meter per second (μmol m⁻² s⁻¹)
    JC = Vcmax * (Pi - GammaS) / (Pi + K)

    # Light-limited rate, dependent on light energy and CO2 availability
    # Units: micromoles per square meter per second (μmol m⁻² s⁻¹)
    JE = carbon_uptake_efficiency * APAR_μmolm2s1 * (Pi - GammaS) / (Pi + 2.0 * GammaS)

    # Export-limited rate, dependent on the capacity to export photosynthetic products
    # Units: micromoles per square meter per second (μmol m⁻² s⁻¹)
    JS = Vcmax / 2.0

    # Empirical colimitation factor for Rubisco-light limitation
    # Unitless
    a = 0.98

    # Quadratic coefficient for colimitation (Rubisco-light limitation)
    # Unitless
    b = -(JC + JE)

    # Quadratic coefficient for colimitation (Rubisco-light limitation)
    # Unitless
    c = JC * JE

    # Combined Rubisco-light limitation
    # Units: micromoles per square meter per second (μmol m⁻² s⁻¹)
    JCE = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)

    # Ensure the result is real-valued for Rubisco-light limitation
    # Units: micromoles per square meter per second (μmol m⁻² s⁻¹)
    JCE = np.real(JCE)

    # Empirical colimitation factor for Rubisco-light-export limitation
    # Unitless
    a = 0.95

    # Quadratic coefficient for colimitation (Rubisco-light-export limitation)
    # Unitless
    b = -(JCE + JS)

    # Quadratic coefficient for colimitation (Rubisco-light-export limitation)
    # Unitless
    c = JCE * JS

    # Combined Rubisco-light-export limitation
    # Units: micromoles per square meter per second (μmol m⁻² s⁻¹)
    photosynthesis_C3_μmolm2s1 = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)

    # Ensure the result is real-valued for Rubisco-light-export limitation
    # Units: micromoles per square meter per second (μmol m⁻² s⁻¹)
    photosynthesis_C3_μmolm2s1 = np.real(photosynthesis_C3_μmolm2s1)

    # Net assimilation rate, ensuring non-negative values
    # Units: micromoles per square meter per second (μmol m⁻² s⁻¹)
    net_assimilation_C3_μmolm2s1 = np.clip(photosynthesis_C3_μmolm2s1 - respiration_C3_μmolm2s1, 0, None)

    return {
        "photosynthesis_C3_μmolm2s1": photosynthesis_C3_μmolm2s1,
        "respiration_C3_μmolm2s1": respiration_C3_μmolm2s1,
        "net_assimilation_C3_μmolm2s1": net_assimilation_C3_μmolm2s1
    }
