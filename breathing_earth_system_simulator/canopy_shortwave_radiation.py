import numpy as np


def canopy_shortwave_radiation(
        PARDiff: np.ndarray,
        PARDir: np.ndarray,
        NIRDiff: np.ndarray,
        NIRDir: np.ndarray,
        UV: np.ndarray,
        SZA: np.ndarray,
        LAI: np.ndarray,
        CI: np.ndarray,
        albedo_visible: np.ndarray,
        albedo_NIR: np.ndarray):
    """
    =============================================================================

    Module     : Canopy radiative transfer
    Input      : diffuse PAR radiation (PARDiff) [W m-2],
               : direct PAR radiation (PARDir) [W m-2],
               : diffuse NIR radiation (NIRDiff) [W m-2],
               : direct NIR radiation (NIRDir) [W m-2],
               : ultroviolet radiation (UV) [W m-2],
               : solar zenith angle (SZA) [degree],
               : leaf area index (LAI) [-],
               : clumping index (CI) [-],
               : VIS albedo (ALB_VIS) [-],
               : NIR albedo (ALB_NIR) [-],
               : leaf maximum carboxylation rate at 25C for C3 plant (Vcmax25_C3Leaf) [umol m-2 s-1],
               : leaf maximum carboxylation rate at 25C for C4 plant (Vcmax25_C4Leaf) [umol m-2 s-1].
    Output     : total absorbed PAR by sunlit leaves (APAR_Sun) [umol m-2 s-1],
               : total absorbed PAR by shade leaves (APAR_Sh) [umol m-2 s-1],
               : total absorbed SW by sunlit leaves (ASW_Sun) [W m-2],
               : total absorbed SW by shade leaves (ASW_Sh) [W m-2],
               : sunlit canopy maximum carboxylation rate at 25C for C3 plant (Vcmax25_C3Sun) [umol m-2 s-1],
               : shade canopy maximum carboxylation rate at 25C for C3 plant (Vcmax25_C3Sh) [umol m-2 s-1],
               : sunlit canopy maximum carboxylation rate at 25C for C4 plant (Vcmax25_C4Sun) [umol m-2 s-1],
               : shade canopy maximum carboxylation rate at 25C for C4 plant (Vcmax25_C4Sh) [umol m-2 s-1],
               : fraction of sunlit canopy (fSun) [-],
               : ground heat storage (G) [W m-2],
               : total absorbed SW by soil (ASW_Soil) [W m-2].
    References : Ryu, Y., Baldocchi, D. D., Kobayashi, H., Van Ingen, C., Li, J., Black, T. A., Beringer, J.,
                 Van Gorsel, E., Knohl, A., Law, B. E., & Roupsard, O. (2011).

                 Integration of MODIS land and atmosphere products with a coupled-process model i
                 to estimate gross primary productivity and evapotranspiration from 1 km to global scales.
                 Global Biogeochemical Cycles, 25(GB4017), 1-24. doi:10.1029/2011GB004053.1.


    Conversion from MatLab by Robert Freepartner, JPL/Raytheon/JaDa Systems
    March 2020

    =============================================================================
    """
    # Leaf scattering coefficients and soil reflectance (Sellers 1985)
    SIGMA_P = 0.175
    RHO_PSOIL = 0.15
    SIGMA_N = 0.825
    RHO_NSOIL = 0.30

    # Extinction coefficient for diffuse and scattered diffuse PAR
    kk_Pd = 0.72  # Table A1

    # self.diagnostic(PARDiff, "PARDiff", date_UTC, target)
    # self.diagnostic(PARDir, "PARDir", date_UTC, target)
    # self.diagnostic(NIRDiff, "NIRDiff", date_UTC, target)
    # self.diagnostic(NIRDir, "NIRDir", date_UTC, target)
    # self.diagnostic(UV, "UV", date_UTC, target)
    # self.diagnostic(SZA, "SZA", date_UTC, target)
    # self.diagnostic(LAI, "LAI", date_UTC, target)
    # self.diagnostic(CI, "CI", date_UTC, target)
    # self.diagnostic(RVIS, "RVIS", date_UTC, target)
    # self.diagnostic(RNIR, "RNIR", date_UTC, target)

    # Beam radiation extinction coefficient of canopy
    kb = np.where(SZA > 89, 50.0, 0.5 / np.cos(np.radians(SZA)))  # Table A1

    # Extinction coefficient for beam and scattered beam PAR
    kk_Pb = np.where(SZA > 89, 50.0, 0.46 / np.cos(np.radians(SZA)))  # Table A1

    # Extinction coefficient for beam and scattered beam NIR
    kk_Nb = kb * np.sqrt(1.0 - SIGMA_N)  # Table A1

    # Extinction coefficient for diffuse and scattered diffuse NIR
    kk_Nd = 0.35 * np.sqrt(1.0 - SIGMA_N)  # Table A1

    # Sunlit fraction
    fSun = np.clip(1.0 / kb * (1.0 - np.exp(-kb * LAI * CI)) / LAI, 0, 1)  # Integration of Eq. (1)

    # For simplicity
    L_CI = LAI * CI
    exp_kk_Pd_L_CI = np.exp(-kk_Pd * L_CI)
    exp_kk_Nd_L_CI = np.exp(-kk_Nd * L_CI)

    # Total absorbed incoming PAR
    Q_PDn = (1.0 - albedo_visible) * PARDir * (1.0 - np.exp(-kk_Pb * L_CI)) + (1.0 - albedo_visible) * PARDiff * (
            1.0 - exp_kk_Pd_L_CI)  # Eq. (2)

    # Absorbed incoming beam PAR by sunlit leaves
    Q_PbSunDn = PARDir * (1.0 - SIGMA_P) * (1.0 - np.exp(-kb * L_CI))  # Eq. (3)

    # Absorbed incoming diffuse PAR by sunlit leaves
    Q_PdSunDn = PARDiff * (1.0 - albedo_visible) * (1.0 - np.exp(-(kk_Pd + kb) * L_CI)) * kk_Pd / (kk_Pd + kb)  # Eq. (4)

    # Absorbed incoming scattered PAR by sunlit leaves
    Q_PsSunDn = PARDir * (
            (1.0 - albedo_visible) * (1.0 - np.exp(-(kk_Pb + kb) * L_CI)) * kk_Pb / (kk_Pb + kb) - (1.0 - SIGMA_P) * (
            1.0 - np.exp(-2.0 * kb * L_CI)) / 2.0)  # Eq. (5)
    Q_PsSunDn = np.clip(Q_PsSunDn, 0, None)

    # Absorbed incoming PAR by sunlit leaves
    Q_PSunDn = Q_PbSunDn + Q_PdSunDn + Q_PsSunDn  # Eq. (6)

    # Absorbed incoming PAR by shade leaves
    Q_PShDn = np.clip(Q_PDn - Q_PSunDn, 0, None)  # Eq. (7)

    # Incoming PAR at soil surface
    I_PSoil = np.clip((1.0 - albedo_visible) * PARDir + (1 - albedo_visible) * PARDiff - (Q_PSunDn + Q_PShDn), 0, None)

    # Absorbed PAR by soil
    APAR_Soil = np.clip((1.0 - RHO_PSOIL) * I_PSoil, 0, None)

    # Absorbed outgoing PAR by sunlit leaves
    Q_PSunUp = np.clip(I_PSoil * RHO_PSOIL * exp_kk_Pd_L_CI, 0, None)  # Eq. (8)

    # Absorbed outgoing PAR by shade leaves
    Q_PShUp = np.clip(I_PSoil * RHO_PSOIL * (1 - exp_kk_Pd_L_CI), 0, None)  # Eq. (9)

    # Total absorbed PAR by sunlit leaves
    APAR_Sun = Q_PSunDn + Q_PSunUp  # Eq. (10)

    # Total absorbed PAR by shade leaves
    APAR_Sh = Q_PShDn + Q_PShUp  # Eq. (11)

    # Absorbed incoming NIR by sunlit leaves
    Q_NSunDn = NIRDir * (1.0 - SIGMA_N) * (1.0 - np.exp(-kb * L_CI)) + NIRDiff * (1 - albedo_NIR) * (
            1.0 - np.exp(-(kk_Nd + kb) * L_CI)) * kk_Nd / (kk_Nd + kb) + NIRDir * (
                       (1.0 - albedo_NIR) * (1.0 - np.exp(-(kk_Nb + kb) * L_CI)) * kk_Nb / (kk_Nb + kb) - (
                       1.0 - SIGMA_N) * (1.0 - np.exp(-2.0 * kb * L_CI)) / 2.0)  # Eq. (14)
    Q_NSunDn = np.clip(Q_NSunDn, 0, None)

    # Absorbed incoming NIR by shade leaves
    Q_NShDn = (1.0 - albedo_NIR) * NIRDir * (1.0 - np.exp(-kk_Nb * L_CI)) + (1.0 - albedo_NIR) * NIRDiff * (
            1.0 - exp_kk_Nd_L_CI) - Q_NSunDn  # Eq. (15)
    Q_NShDn = np.clip(Q_NShDn, 0, None)

    # Incoming NIR at soil surface
    I_NSoil = (1.0 - albedo_NIR) * NIRDir + (1.0 - albedo_NIR) * NIRDiff - (Q_NSunDn + Q_NShDn)
    I_NSoil = np.clip(I_NSoil, 0, None)

    # Absorbed NIR by soil
    ANIR_Soil = (1.0 - RHO_NSOIL) * I_NSoil
    ANIR_Soil = np.clip(ANIR_Soil, 0, None)

    # Absorbed outgoing NIR by sunlit leaves
    Q_NSunUp = I_NSoil * RHO_NSOIL * exp_kk_Nd_L_CI  # Eq. (16)
    Q_NSunUp = np.clip(Q_NSunUp, 0, None)

    # Absorbed outgoing NIR by shade leaves
    Q_NShUp = I_NSoil * RHO_NSOIL * (1.0 - exp_kk_Nd_L_CI)  # Eq. (17)
    Q_NShUp = np.clip(Q_NShUp, 0, None)

    # Total absorbed NIR by sunlit leaves
    ANIR_Sun = Q_NSunDn + Q_NSunUp  # Eq. (18)

    # Total absorbed NIR by shade leaves
    ANIR_Sh = Q_NShDn + Q_NShUp  # Eq. (19)

    # UV
    UVDir = UV * PARDir / (PARDir + PARDiff + 1e-5)
    UVDiff = UV - UVDir
    Q_U = (1.0 - 0.05) * UVDiff * (1.0 - np.exp(-kk_Pb * L_CI)) + (1.0 - 0.05) * UVDiff * (1.0 - exp_kk_Pd_L_CI)
    AUV_Sun = Q_U * fSun
    AUV_Sh = Q_U * (1 - fSun)
    AUV_Soil = (1.0 - 0.05) * UV - Q_U

    # Ground heat storage
    G = APAR_Soil * 0.28

    # Summary
    ASW_Sun = APAR_Sun + ANIR_Sun + AUV_Sun
    ASW_Sun = np.where(LAI == 0, 0, ASW_Sun)
    ASW_Sh = APAR_Sh + ANIR_Sh + AUV_Sh
    ASW_Sh = np.where(LAI == 0, 0, ASW_Sh)
    ASW_Soil = APAR_Soil + ANIR_Soil + AUV_Soil
    APAR_Sun = np.where(LAI == 0, 0, APAR_Sun)
    APAR_Sun = APAR_Sun * 4.56
    APAR_Sh = np.where(LAI == 0, 0, APAR_Sh)
    APAR_Sh = APAR_Sh * 4.56

    #TODO not sure about these variables: Vcmax25_C3Sun, Vcmax25_C3Sh, Vcmax25_C4Sun, Vcmax25_C4Sh

    return fSun, APAR_Sun, APAR_Sh, ASW_Sun, ASW_Sh, ASW_Soil, G
