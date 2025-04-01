import numpy as np


def calculate_C4_photosynthesis(Tf_K: np.ndarray, Ci: np.ndarray, APAR: np.ndarray, Vcmax25: np.ndarray) -> np.ndarray:
    """
    =============================================================================
    Collatz et al., 1992

    Module     : Photosynthesis for C4 plant
    Input      : leaf temperature (Tf) [K],
               : intercellular CO2 concentration (Ci) [umol mol-1],
               : absorbed photosynthetically active radiation (APAR) [umol m-2 s-1],
               : maximum carboxylation rate at 25C (Vcmax25) [umol m-2 s-1].
    Output     : net assimilation (An) [umol m-2 s-1].


    Conversion from MatLab by Robert Freepartner, JPL/Raytheon/JaDa Systems
    March 2020

    =============================================================================
    """
    # Temperature correction
    item = (Tf_K - 298.15) / 10.0
    Q10 = 2.0
    k = 0.7 * pow(Q10, item)  # [mol m-2 s-1]
    Vcmax_o = Vcmax25 * pow(Q10, item)  # [umol m-2 s-1]
    Vcmax = Vcmax_o / (
            (1.0 + np.exp(0.3 * (286.15 - Tf_K))) * (1.0 + np.exp(0.3 * (Tf_K - 309.15))))  # [umol m-2 s-1]
    Rd_o = 0.8 * pow(Q10, item)  # [umol m-2 s-1]
    Rd = Rd_o / (1.0 + np.exp(1.3 * (Tf_K - 328.15)))  # [umol m-2 s-1]

    # Three limiting states
    Je = Vcmax  # [umol m-2 s-1]
    alf = 0.067  # [mol CO2 mol photons-1]
    Ji = alf * APAR  # [umol m-2 s-1]
    ci = Ci * 1e-6  # [umol mol-1] -> [mol CO2 mol CO2-1]
    Jc = ci * k * 1e6  # [umol m-2 s-1]

    # Colimitation (not the case at canopy level according to DePury and Farquhar)
    a = 0.83
    b = -(Je + Ji)
    c = Je * Ji
    Jei = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    Jei = np.real(Jei)
    a = 0.93
    b = -(Jei + Jc)
    c = Jei * Jc
    Jeic = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    Jeic = np.real(Jeic)

    # Net assimilation
    # An = nanmin(cat(3,Je,Ji,Jc),[],3) - Rd;    % [umol m-2 s-1]
    An = np.clip(Jeic - Rd, 0, None)
    # An[An<0.0] = 0.0

    return An
