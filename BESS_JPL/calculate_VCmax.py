from os.path import join, abspath, dirname

import rasters as rt
import numpy as np

from .constants import *

def calculate_VCmax(
        LAI: np.ndarray,
        LAI_minimum: np.ndarray,
        LAI_maximum: np.ndarray,
        peakVCmax_C3: np.ndarray,
        peakVCmax_C4: np.ndarray,
        SZA: np.ndarray,
        kn: np.ndarray,
        A: np.ndarray = A):
    sf = np.clip(np.clip(LAI - LAI_minimum, 0, None) / np.clip(LAI_maximum - LAI_minimum, 1, None), 0, 1)
    sf = np.where(np.isreal(sf), sf, 0)
    sf = np.where(np.isnan(sf), 0, sf)

    # calculate maximum carboxylation rate at 25C for C3 plants
    VCmax_C3 = A * peakVCmax_C3 + (1 - A) * peakVCmax_C3 * sf

    # calculate maximum carboxylation rate at 25C for C4 plants
    VCmax_C4 = A * peakVCmax_C4 + (1 - A) * peakVCmax_C4 * sf

    # kb = 0.5 / np.cos(np.radians(SZA))
    kb = np.where(SZA > 89, 50.0, 0.5 / np.cos(np.radians(SZA)))
    kn_kb_Lc = kn + kb * LAI
    exp_neg_kn_kb_Lc = np.exp(-kn_kb_Lc)
    LAI_VCmax_C3 = LAI * VCmax_C3
    exp_neg_kn = np.exp(-kn)

    # calculate total maximum carboxylation rate at 25C for C3 plants
    VCmax_C3_total = LAI_VCmax_C3 * (1 - exp_neg_kn) / kn

    # calculate sunlit maximum carboxylation rate at 25C for C3 plants
    VCmax_C3_sunlit = LAI_VCmax_C3 * (1 - exp_neg_kn_kb_Lc) / kn_kb_Lc

    # calculate shaded maximum carboxylation rate at 25C for C3 plants
    VCmax_C3_shaded = VCmax_C3_total - VCmax_C3_sunlit

    LAI_VCmax_C4 = LAI * VCmax_C4

    # calculate total maximum carboxylation rate at 25C for C4 plants
    VCmax_C4_total = LAI_VCmax_C4 * (1 - exp_neg_kn) / kn

    # calculate sunlit maximum carboxylation rate at 25C for C4 plants
    VCmax_C4_sunlit = LAI_VCmax_C4 * (1 - exp_neg_kn_kb_Lc) / kn_kb_Lc

    # calculate shaded maximum carboxylation rate at 25C for C4 plants
    VCmax_C4_shaded = VCmax_C4_total - VCmax_C4_sunlit

    return VCmax_C3_sunlit, VCmax_C4_sunlit, VCmax_C3_shaded, VCmax_C4_shaded
