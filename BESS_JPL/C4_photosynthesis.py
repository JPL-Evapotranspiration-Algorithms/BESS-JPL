import numpy as np


def calculate_C4_photosynthesis(
        Tf_K: np.ndarray, 
        Ci_μmol_per_mol: np.ndarray, 
        APAR_μmolm2s1: np.ndarray, 
        Vcmax25_μmolm2s1: np.ndarray) -> dict:
    """
    =============================================================================
    Collatz et al., 1992

    Module     : Photosynthesis for C4 plant
    Description: This function calculates the net assimilation rate (An) for C4 plants
                 based on the biochemical model described by Collatz et al. (1992).
                 The model accounts for temperature corrections, light absorption,
                 and CO2 availability to simulate photosynthetic rates under varying
                 environmental conditions.

    Parameters:
        Tf_K (np.ndarray): Leaf temperature (Tf) in Kelvin. Temperature influences 
                           enzyme kinetics and the rates of photosynthetic reactions.
        Ci_μmol_per_mol (np.ndarray): Intercellular CO2 concentration (Ci) in 
                                      micromoles per mole. Represents the CO2 available for fixation.
        APAR_μmolm2s1 (np.ndarray): Absorbed photosynthetically active radiation (APAR) 
                                     in micromoles per square meter per second. This is the light energy available for photosynthesis.
        Vcmax25_μmolm2s1 (np.ndarray): Maximum carboxylation rate at 25°C (Vcmax25) in 
                                       micromoles per square meter per second. Reflects the activity of the enzyme Rubisco.

    Returns:
        dict: A dictionary containing the following keys:
            - 'net_assimilation_C4_μmolm2s1' (np.ndarray): Net assimilation rate (An) in micromoles per square meter per second.
            - 'photosynthesis_C4_μmolm2s1' (np.ndarray): Photosynthesis rate in micromoles per square meter per second.
            - 'respiration_C4_μmolm2s1' (np.ndarray): Respiration rate in micromoles per square meter per second.

    Explanation:
        The net assimilation rate (An) is the balance between the carbon dioxide fixed 
        during photosynthesis and the carbon dioxide released during respiration. It 
        quantifies the net carbon gain by the plant under given environmental conditions. 
        This function models the biochemical processes that limit photosynthesis, 
        including light availability, CO2 concentration, and enzyme activity, while 
        accounting for temperature-dependent effects on these processes.

    References:
        Collatz, G. J., Ball, J. T., Grivet, C., & Berry, J. A. (1992). Physiological
        and environmental regulation of stomatal conductance, photosynthesis, and
        transpiration: A model that includes a laminar boundary layer. Agricultural
        and Forest Meteorology, 54(2-4), 107-136.

        DePury, D. G. G., & Farquhar, G. D. (1997). Simple scaling of photosynthesis
        from leaves to canopies without the errors of big-leaf models. Plant, Cell &
        Environment, 20(5), 537-557.
    =============================================================================
    """
    # Calculate the temperature deviation from 25°C (298.15 K)
    # `temperature_deviation` represents the temperature difference normalized to 10°C intervals
    temperature_deviation = (Tf_K - 298.15) / 10.0

    # Define the Q10 coefficient, which describes the rate increase for every 10°C rise
    # A Q10 of 2.0 means the reaction rate doubles for every 10°C increase in temperature
    Q10 = 2.0

    # Calculate the temperature-dependent rate constant for CO2 fixation
    # `k` is the rate constant for CO2 fixation, adjusted for temperature
    k = 0.7 * pow(Q10, temperature_deviation)  # [mol m-2 s-1]

    # Calculate the temperature-corrected maximum carboxylation rate
    # `Vcmax_o` is the base maximum carboxylation rate adjusted for temperature
    Vcmax_o = Vcmax25_μmolm2s1 * pow(Q10, temperature_deviation)  # [umol m-2 s-1]

    # Further adjust `Vcmax_o` for enzyme deactivation at extreme temperatures
    # `Vcmax` is the effective maximum carboxylation rate after accounting for temperature sensitivity
    Vcmax = Vcmax_o / (
        (1.0 + np.exp(0.3 * (286.15 - Tf_K))) * (1.0 + np.exp(0.3 * (Tf_K - 309.15)))
    )  # [umol m-2 s-1]

    # Calculate the temperature-corrected dark respiration rate
    # `Rd_o` is the base dark respiration rate adjusted for temperature
    Rd_o_μmolm2s1 = 0.8 * pow(Q10, temperature_deviation)  # [umol m-2 s-1]

    # `Rd` is the effective dark respiration rate after accounting for temperature sensitivity
    respiration_C4_μmolm2s1 = Rd_o_μmolm2s1 / (1.0 + np.exp(1.3 * (Tf_K - 328.15)))  # [umol m-2 s-1]

    # Define the three limiting states of photosynthesis
    # `Je` is the electron transport-limited rate, assumed to equal `Vcmax`
    Je = Vcmax  # [umol m-2 s-1]

    # Calculate the light-limited rate
    # `quantum_yield` is the quantum yield (mol CO2 fixed per mol photons absorbed)
    quantum_yield = 0.067  # Quantum yield

    # `Ji` is the light-limited rate, determined by absorbed light energy
    Ji_μmolm2s1 = quantum_yield * APAR_μmolm2s1  # [umol m-2 s-1]

    # Calculate the CO2-limited rate
    # `ci` is the intercellular CO2 concentration converted to mol/mol
    ci = Ci_μmol_per_mol * 1e-6  # Convert [umol mol-1] to [mol mol-1]

    # `Jc` is the CO2-limited rate, based on `ci` and the rate constant `k`
    Jc = ci * k * 1e6  # [umol m-2 s-1]

    # Colimitation between the three limiting states of photosynthesis
    # Step 1: Colimitation between `Je` and `Ji`
    # `a`, `b`, and `c` are coefficients for the quadratic equation
    a = 0.83  # Empirical coefficient for colimitation
    b = -(Je + Ji_μmolm2s1)
    c = Je * Ji_μmolm2s1

    # `Jei` is the intermediate colimited rate between `Je` and `Ji`
    Jei = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    Jei = np.real(Jei)  # Ensure real values

    # Step 2: Colimitation between `Jei` and `Jc`
    # Update coefficients for the quadratic equation
    a = 0.93  # Empirical coefficient for colimitation
    b = -(Jei + Jc)
    c = Jei * Jc

    # `Jeic` is the final colimited rate between `Jei` and `Jc`
    photosynthesis_C4_μmolm2s1 = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    photosynthesis_C4_μmolm2s1 = np.real(photosynthesis_C4_μmolm2s1)  # Ensure real values

    # Calculate the net assimilation rate
    # `An` is the net assimilation rate, clipped to ensure non-negative values
    # It represents the net carbon gain by the plant, accounting for photosynthesis and respiration
    net_assimilation_C4_μmolm2s1 = np.clip(photosynthesis_C4_μmolm2s1 - respiration_C4_μmolm2s1, 0, None)  # [umol m-2 s-1]

    # Return results as a dictionary
    return {
        'net_assimilation_C4_μmolm2s1': net_assimilation_C4_μmolm2s1,
        'photosynthesis_C4_μmolm2s1': photosynthesis_C4_μmolm2s1,
        'respiration_C4_μmolm2s1': respiration_C4_μmolm2s1
    }
