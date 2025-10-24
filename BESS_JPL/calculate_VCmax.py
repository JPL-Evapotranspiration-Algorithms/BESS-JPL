from os.path import join, abspath, dirname

import rasters as rt
import numpy as np

from .constants import *

def calculate_VCmax(
        LAI: np.ndarray,
        LAI_minimum: np.ndarray,
        LAI_maximum: np.ndarray,
        peakVCmax_C3_μmolm2s1: np.ndarray,
        peakVCmax_C4_μmolm2s1: np.ndarray,
        SZA_deg: np.ndarray,
        kn: np.ndarray,
        A: np.ndarray = A):
    """
    Calculate the maximum carboxylation rate (VCmax) at 25°C for C3 and C4 plants.

    This function computes the sunlit and shaded maximum carboxylation rates for C3 and C4 plants
    based on the Leaf Area Index (LAI), solar zenith angle (SZA), and other physiological parameters.

    Parameters:
        LAI (np.ndarray): Leaf Area Index, the total one-sided green leaf area per unit ground area (dimensionless).
        LAI_minimum (np.ndarray): Minimum LAI for the vegetation type (dimensionless).
        LAI_maximum (np.ndarray): Maximum LAI for the vegetation type (dimensionless).
        peakVCmax_C3 (np.ndarray): Peak maximum carboxylation rate at 25°C for C3 plants (μmol CO₂ m⁻² s⁻¹).
        peakVCmax_C4 (np.ndarray): Peak maximum carboxylation rate at 25°C for C4 plants (μmol CO₂ m⁻² s⁻¹).
        SZA (np.ndarray): Solar Zenith Angle, the angle between the sun's rays and the vertical direction (degrees).
        kn (np.ndarray): Extinction coefficient for nitrogen, describing attenuation within the canopy (dimensionless).
        A (np.ndarray): Scaling factor, determines the proportion of peakVCmax used in the calculation (dimensionless).

    Returns:
        dict: A dictionary containing the following keys and their corresponding values:
            - VCmax_C3_sunlit_μmolm2s1
            - VCmax_C4_sunlit_μmolm2s1
            - VCmax_C3_shaded_μmolm2s1
            - VCmax_C4_shaded_μmolm2s1
    """
    # Normalize LAI to calculate the scaling factor (sf)
    sf = np.clip(np.clip(LAI - LAI_minimum, 0, None) / np.clip(LAI_maximum - LAI_minimum, 1, None), 0, 1)
    sf = np.where(np.isreal(sf), sf, 0)
    sf = np.where(np.isnan(sf), 0, sf)

    # Calculate maximum carboxylation rate at 25°C for C3 plants
    VCmax_C3_μmolm2s1 = A * peakVCmax_C3_μmolm2s1 + (1 - A) * peakVCmax_C3_μmolm2s1 * sf

    # Calculate maximum carboxylation rate at 25°C for C4 plants
    VCmax_C4_μmolm2s1 = A * peakVCmax_C4_μmolm2s1 + (1 - A) * peakVCmax_C4_μmolm2s1 * sf

    # Calculate the beam extinction coefficient (kb)
    kb = np.where(SZA_deg > 89, 50.0, 0.5 / np.cos(np.radians(SZA_deg)))

    # Combine extinction coefficients for nitrogen and beam attenuation
    kn_kb_Lc = kn + kb * LAI

    # Precompute exponential terms for light attenuation
    exp_neg_kn_kb_Lc = np.exp(-kn_kb_Lc)
    exp_neg_kn = np.exp(-kn)

    # Scale VCmax by LAI for C3 plants
    LAI_VCmax_C3 = LAI * VCmax_C3_μmolm2s1

    # Calculate total maximum carboxylation rate at 25°C for C3 plants
    VCmax_C3_total_μmolm2s1 = LAI_VCmax_C3 * (1 - exp_neg_kn) / kn

    # Calculate sunlit maximum carboxylation rate at 25°C for C3 plants
    VCmax_C3_sunlit_μmolm2s1 = LAI_VCmax_C3 * (1 - exp_neg_kn_kb_Lc) / kn_kb_Lc

    # Calculate shaded maximum carboxylation rate at 25°C for C3 plants
    VCmax_C3_shaded_μmolm2s1 = VCmax_C3_total_μmolm2s1 - VCmax_C3_sunlit_μmolm2s1

    # Scale VCmax by LAI for C4 plants
    LAI_VCmax_C4 = LAI * VCmax_C4_μmolm2s1

    # Calculate total maximum carboxylation rate at 25°C for C4 plants
    VCmax_C4_total_μmolm2s1 = LAI_VCmax_C4 * (1 - exp_neg_kn) / kn

    # Calculate sunlit maximum carboxylation rate at 25°C for C4 plants
    VCmax_C4_sunlit_μmolm2s1 = LAI_VCmax_C4 * (1 - exp_neg_kn_kb_Lc) / kn_kb_Lc

    # Calculate shaded maximum carboxylation rate at 25°C for C4 plants
    VCmax_C4_shaded_μmolm2s1 = VCmax_C4_total_μmolm2s1 - VCmax_C4_sunlit_μmolm2s1

    return {
        "VCmax_C3_sunlit_μmolm2s1": VCmax_C3_sunlit_μmolm2s1,
        "VCmax_C4_sunlit_μmolm2s1": VCmax_C4_sunlit_μmolm2s1,
        "VCmax_C3_shaded_μmolm2s1": VCmax_C3_shaded_μmolm2s1,
        "VCmax_C4_shaded_μmolm2s1": VCmax_C4_shaded_μmolm2s1
    }
