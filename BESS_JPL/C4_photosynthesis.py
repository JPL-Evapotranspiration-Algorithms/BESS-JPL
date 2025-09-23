import numpy as np


def calculate_C4_photosynthesis(Tf_K: np.ndarray, Ci: np.ndarray, APAR: np.ndarray, Vcmax25: np.ndarray) -> np.ndarray:
    """
    =============================================================================
    Collatz et al., 1992

    Module     : Photosynthesis for C4 plant
    Description: This function calculates the net assimilation rate (An) for C4 plants
                 based on the biochemical model described by Collatz et al. (1992).
                 The model accounts for temperature corrections, light absorption,
                 and CO2 availability to simulate photosynthetic rates under varying
                 environmental conditions.

    Inputs:
        Tf_K    : Leaf temperature (Tf) [K]. Temperature influences enzyme kinetics
                  and the rates of photosynthetic reactions.
        Ci      : Intercellular CO2 concentration (Ci) [umol mol-1]. Represents the
                  CO2 available for fixation.
        APAR    : Absorbed photosynthetically active radiation (APAR) [umol m-2 s-1].
                  This is the light energy available for photosynthesis.
        Vcmax25 : Maximum carboxylation rate at 25°C (Vcmax25) [umol m-2 s-1].
                  Reflects the activity of the enzyme Rubisco.

    Outputs:
        An      : Net assimilation rate (An) [umol m-2 s-1]. Represents the rate of
                  CO2 uptake by the plant.

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
    # `item` represents the temperature difference normalized to 10°C intervals
    item = (Tf_K - 298.15) / 10.0

    # Define the Q10 coefficient, which describes the rate increase for every 10°C rise
    Q10 = 2.0  # Reaction rate doubles for every 10°C increase

    # Calculate the temperature-dependent rate constant for CO2 fixation
    # `k` is the rate constant for CO2 fixation, adjusted for temperature
    k = 0.7 * pow(Q10, item)  # [mol m-2 s-1]

    # Calculate the temperature-corrected maximum carboxylation rate
    # `Vcmax_o` is the base maximum carboxylation rate adjusted for temperature
    Vcmax_o = Vcmax25 * pow(Q10, item)  # [umol m-2 s-1]

    # Further adjust `Vcmax_o` for enzyme deactivation at extreme temperatures
    # `Vcmax` is the effective maximum carboxylation rate after accounting for temperature sensitivity
    Vcmax = Vcmax_o / (
        (1.0 + np.exp(0.3 * (286.15 - Tf_K))) * (1.0 + np.exp(0.3 * (Tf_K - 309.15)))
    )  # [umol m-2 s-1]

    # Calculate the temperature-corrected dark respiration rate
    # `Rd_o` is the base dark respiration rate adjusted for temperature
    Rd_o = 0.8 * pow(Q10, item)  # [umol m-2 s-1]

    # `Rd` is the effective dark respiration rate after accounting for temperature sensitivity
    Rd = Rd_o / (1.0 + np.exp(1.3 * (Tf_K - 328.15)))  # [umol m-2 s-1]

    # Define the three limiting states of photosynthesis
    # `Je` is the electron transport-limited rate, assumed to equal `Vcmax`
    Je = Vcmax  # [umol m-2 s-1]

    # Calculate the light-limited rate
    # `alf` is the quantum yield (mol CO2 fixed per mol photons absorbed)
    alf = 0.067  # Quantum yield

    # `Ji` is the light-limited rate, determined by absorbed light energy
    Ji = alf * APAR  # [umol m-2 s-1]

    # Calculate the CO2-limited rate
    # `ci` is the intercellular CO2 concentration converted to mol/mol
    ci = Ci * 1e-6  # Convert [umol mol-1] to [mol mol-1]

    # `Jc` is the CO2-limited rate, based on `ci` and the rate constant `k`
    Jc = ci * k * 1e6  # [umol m-2 s-1]

    # Colimitation between the three limiting states
    # Step 1: Colimitation between `Je` and `Ji`
    # `a`, `b`, and `c` are coefficients for the quadratic equation
    a = 0.83  # Empirical coefficient for colimitation
    b = -(Je + Ji)
    c = Je * Ji

    # `Jei` is the intermediate colimited rate between `Je` and `Ji`
    Jei = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    Jei = np.real(Jei)  # Ensure real values

    # Step 2: Colimitation between `Jei` and `Jc`
    # Update coefficients for the quadratic equation
    a = 0.93  # Empirical coefficient for colimitation
    b = -(Jei + Jc)
    c = Jei * Jc

    # `Jeic` is the final colimited rate between `Jei` and `Jc`
    Jeic = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    Jeic = np.real(Jeic)  # Ensure real values

    # Calculate the net assimilation rate
    # `An` is the net assimilation rate, clipped to ensure non-negative values
    An = np.clip(Jeic - Rd, 0, None)  # [umol m-2 s-1]

    return An
