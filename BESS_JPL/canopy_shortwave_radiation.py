import numpy as np

from check_distribution import check_distribution


def canopy_shortwave_radiation(
        PAR_diffuse_Wm2: np.ndarray,
        PAR_direct_Wm2: np.ndarray,
        NIR_diffuse_Wm2: np.ndarray,
        NIR_direct_Wm2: np.ndarray,
        UV_Wm2: np.ndarray,
        SZA_deg: np.ndarray,
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
               : ultraviolet radiation (UV) [W m-2],
               : solar zenith angle (SZA) [degree],
               : leaf area index (LAI) [-],
               : clumping index (CI) [-],
               : visible albedo (albedo_visible) [-],
               : NIR albedo (albedo_NIR) [-].
    Output     : fraction of sunlit canopy (fSun) [-],
               : total absorbed PAR by sunlit leaves (APAR_Sun) [μmol m-2 s-1],
               : total absorbed PAR by shade leaves (APAR_Sh) [μmol m-2 s-1],
               : total absorbed shortwave radiation by sunlit leaves (ASW_Sun) [W m-2],
               : total absorbed shortwave radiation by shade leaves (ASW_Sh) [W m-2],
               : total absorbed shortwave radiation by soil (ASW_Soil) [W m-2],
               : ground heat storage (G) [W m-2].
    References : Ryu, Y., Baldocchi, D. D., Kobayashi, H., Van Ingen, C., Li, J., Black, T. A., Beringer, J.,
                 Van Gorsel, E., Knohl, A., Law, B. E., & Roupsard, O. (2011).

                 Integration of MODIS land and atmosphere products with a coupled-process model
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

    # Check for None parameters
    parameters = {
        "PARDiff": PAR_diffuse_Wm2,
        "PARDir": PAR_direct_Wm2,
        "NIRDiff": NIR_diffuse_Wm2,
        "NIRDir": NIR_direct_Wm2,
        "UV": UV_Wm2,
        "SZA": SZA_deg,
        "LAI": LAI,
        "CI": CI,
        "albedo_visible": albedo_visible,
        "albedo_NIR": albedo_NIR
    }

    for param_name, param_value in parameters.items():
        check_distribution(param_value, param_name)
        if param_value is None:
            raise ValueError(f"The parameter '{param_name}' cannot be None.")

    # Beam radiation extinction coefficient of canopy
    kb = np.where(SZA_deg > 89, 50.0, 0.5 / np.cos(np.radians(SZA_deg)))  # Table A1
    check_distribution(kb, "kb")

    # Extinction coefficient for beam and scattered beam PAR
    kk_Pb = np.where(SZA_deg > 89, 50.0, 0.46 / np.cos(np.radians(SZA_deg)))  # Table A1
    check_distribution(kk_Pb, "kk_Pb")

    # Extinction coefficient for beam and scattered beam NIR
    kk_Nb = kb * np.sqrt(1.0 - SIGMA_N)  # Table A1
    check_distribution(kk_Nb, "kk_Nb")

    # Extinction coefficient for diffuse and scattered diffuse NIR
    kk_Nd = 0.35 * np.sqrt(1.0 - SIGMA_N)  # Table A1
    check_distribution(kk_Nd, "kk_Nd")

    # Sunlit fraction
    fSun = np.clip(1.0 / kb * (1.0 - np.exp(-kb * LAI * CI)) / LAI, 0, 1)  # Integration of Eq. (1)
    fSun = np.where(LAI == 0, 0, fSun)  # Eq. (1)
    check_distribution(fSun, "fSun")

    # For simplicity
    L_CI = LAI * CI
    check_distribution(L_CI, "L_CI")
    exp_kk_Pd_L_CI = np.exp(-kk_Pd * L_CI)
    check_distribution(exp_kk_Pd_L_CI, "exp_kk_Pd_L_CI")
    exp_kk_Nd_L_CI = np.exp(-kk_Nd * L_CI)
    check_distribution(exp_kk_Nd_L_CI, "exp_kk_Nd_L_CI")

    # Total absorbed incoming PAR
    APARin_Wm2 = (1.0 - albedo_visible) * PAR_direct_Wm2 * (1.0 - np.exp(-kk_Pb * L_CI)) + (1.0 - albedo_visible) * PAR_diffuse_Wm2 * (
            1.0 - exp_kk_Pd_L_CI)  # Eq. (2)
    check_distribution(APARin_Wm2, "APARin_Wm2")

    # Absorbed incoming beam PAR by sunlit leaves
    APARin_sunlit_Wm2 = PAR_direct_Wm2 * (1.0 - SIGMA_P) * (1.0 - np.exp(-kb * L_CI))  # Eq. (3)
    check_distribution(APARin_sunlit_Wm2, "APARin_sunlit_Wm2")

    # Absorbed incoming diffuse PAR by sunlit leaves
    APARin_diffuse_sunlit_Wm2 = PAR_diffuse_Wm2 * (1.0 - albedo_visible) * (1.0 - np.exp(-(kk_Pd + kb) * L_CI)) * kk_Pd / (
            kk_Pd + kb)  # Eq. (4)
    check_distribution(APARin_diffuse_sunlit_Wm2, "APARin_diffuse_Wm2")

    # Absorbed incoming scattered PAR by sunlit leaves
    APARin_scattered_sunlit_Wm2 = PAR_direct_Wm2 * (
            (1.0 - albedo_visible) * (1.0 - np.exp(-(kk_Pb + kb) * L_CI)) * kk_Pb / (kk_Pb + kb) - (1.0 - SIGMA_P) * (
            1.0 - np.exp(-2.0 * kb * L_CI)) / 2.0)  # Eq. (5)
    APARin_scattered_sunlit_Wm2 = np.clip(APARin_scattered_sunlit_Wm2, 0, None)
    check_distribution(APARin_scattered_sunlit_Wm2, "APARin_scattered_sunlit_Wm2")

    # Absorbed incoming PAR by sunlit leaves
    APARin_sunlit_Wm2 = APARin_sunlit_Wm2 + APARin_diffuse_sunlit_Wm2 + APARin_scattered_sunlit_Wm2  # Eq. (6)
    check_distribution(APARin_sunlit_Wm2, "APARin_sunlit_Wm2")

    # Absorbed incoming PAR by shade leaves
    APARin_shade_Wm2 = np.clip(APARin_Wm2 - APARin_sunlit_Wm2, 0, None)  # Eq. (7)
    check_distribution(APARin_shade_Wm2, "APARin_shade_Wm2")

    # Incoming PAR at soil surface
    PARin_soil_Wm2 = np.clip((1.0 - albedo_visible) * PAR_direct_Wm2 + (1 - albedo_visible) * PAR_diffuse_Wm2 - (APARin_sunlit_Wm2 + APARin_shade_Wm2), 0, None)
    check_distribution(PARin_soil_Wm2, "PARin_soil_Wm2")

    # Absorbed PAR by soil
    APAR_soil_Wm2 = np.clip((1.0 - RHO_PSOIL) * PARin_soil_Wm2, 0, None)
    check_distribution(APAR_soil_Wm2, "APAR_Soil")

    # Absorbed outgoing PAR by sunlit leaves
    APARout_sunlit_Wm2 = np.clip(PARin_soil_Wm2 * RHO_PSOIL * exp_kk_Pd_L_CI, 0, None)  # Eq. (8)
    check_distribution(APARout_sunlit_Wm2, "APARout_sunlit_Wm2")

    # Absorbed outgoing PAR by shade leaves
    APARout_shade_Wm2 = np.clip(PARin_soil_Wm2 * RHO_PSOIL * (1 - exp_kk_Pd_L_CI), 0, None)  # Eq. (9)
    check_distribution(APARout_shade_Wm2, "APARout_shade_Wm2")

    # Total absorbed PAR by sunlit leaves
    APAR_sunlit_Wm2 = APARin_sunlit_Wm2 + APARout_sunlit_Wm2  # Eq. (10)
    check_distribution(APAR_sunlit_Wm2, "APAR_sunlit_Wm2")

    # Total absorbed PAR by shade leaves
    APAR_shade_Wm2 = APARin_shade_Wm2 + APARout_shade_Wm2  # Eq. (11)
    check_distribution(APAR_shade_Wm2, "APAR_shade_Wm2")

    # Absorbed incoming NIR by sunlit leaves
    ANIRin_sunlit_Wm2 = NIR_direct_Wm2 * (1.0 - SIGMA_N) * (1.0 - np.exp(-kb * L_CI)) + NIR_diffuse_Wm2 * (1 - albedo_NIR) * (
            1.0 - np.exp(-(kk_Nd + kb) * L_CI)) * kk_Nd / (kk_Nd + kb) + NIR_direct_Wm2 * (
                       (1.0 - albedo_NIR) * (1.0 - np.exp(-(kk_Nb + kb) * L_CI)) * kk_Nb / (kk_Nb + kb) - (
                       1.0 - SIGMA_N) * (1.0 - np.exp(-2.0 * kb * L_CI)) / 2.0)  # Eq. (14)
    ANIRin_sunlit_Wm2 = np.clip(ANIRin_sunlit_Wm2, 0, None)
    check_distribution(ANIRin_sunlit_Wm2, "ANIRin_sunlit_Wm2")

    # Absorbed incoming NIR by shade leaves
    ANIRin_shade_Wm2 = (1.0 - albedo_NIR) * NIR_direct_Wm2 * (1.0 - np.exp(-kk_Nb * L_CI)) + (1.0 - albedo_NIR) * NIR_diffuse_Wm2 * (
            1.0 - exp_kk_Nd_L_CI) - ANIRin_sunlit_Wm2  # Eq. (15)
    ANIRin_shade_Wm2 = np.clip(ANIRin_shade_Wm2, 0, None)
    check_distribution(ANIRin_shade_Wm2, "ANIRin_shade_Wm2")

    # Incoming NIR at soil surface
    NIRin_soil_Wm2 = (1.0 - albedo_NIR) * NIR_direct_Wm2 + (1.0 - albedo_NIR) * NIR_diffuse_Wm2 - (ANIRin_sunlit_Wm2 + ANIRin_shade_Wm2)
    NIRin_soil_Wm2 = np.clip(NIRin_soil_Wm2, 0, None)
    check_distribution(NIRin_soil_Wm2, "NIRin_soil_Wm2")

    # Absorbed NIR by soil
    ANIR_soil_Wm2 = (1.0 - RHO_NSOIL) * NIRin_soil_Wm2
    ANIR_soil_Wm2 = np.clip(ANIR_soil_Wm2, 0, None)
    check_distribution(ANIR_soil_Wm2, "ANIR_soil_Wm2")

    # Absorbed outgoing NIR by sunlit leaves
    ANIRout_sunlit_Wm2 = NIRin_soil_Wm2 * RHO_NSOIL * exp_kk_Nd_L_CI  # Eq. (16)
    ANIRout_sunlit_Wm2 = np.clip(ANIRout_sunlit_Wm2, 0, None)
    check_distribution(ANIRout_sunlit_Wm2, "ANIRout_sunlit_Wm2")

    # Absorbed outgoing NIR by shade leaves
    ANIRout_Wm2 = NIRin_soil_Wm2 * RHO_NSOIL * (1.0 - exp_kk_Nd_L_CI)  # Eq. (17)
    ANIRout_Wm2 = np.clip(ANIRout_Wm2, 0, None)
    check_distribution(ANIRout_Wm2, "ANIRout_Wm2")

    # Total absorbed NIR by sunlit leaves
    ANIR_sunlit_Wm2 = ANIRin_sunlit_Wm2 + ANIRout_sunlit_Wm2  # Eq. (18)
    check_distribution(ANIR_sunlit_Wm2, "ANIR_sunlit_Wm2")

    # Total absorbed NIR by shade leaves
    ANIR_shade_Wm2 = ANIRin_shade_Wm2 + ANIRout_Wm2  # Eq. (19)
    check_distribution(ANIR_shade_Wm2, "ANIR_shade_Wm2")

    # Direct UV radiation (W/m²), calculated as the proportion of total UV radiation that corresponds to direct PAR radiation.
    # This represents the UV radiation coming directly from the sun without scattering.
    UV_direct_Wm2 = UV_Wm2 * PAR_direct_Wm2 / (PAR_direct_Wm2 + PAR_diffuse_Wm2 + 1e-5)

    # Diffuse UV radiation (W/m²), calculated as the remaining portion of total UV radiation after subtracting the direct component.
    # This represents the UV radiation scattered in the atmosphere before reaching the canopy.
    UV_diffuse_Wm2 = UV_Wm2 - UV_direct_Wm2

    # Total absorbed UV radiation by the canopy (W/m²), combining contributions from diffuse UV radiation.
    AUV_Wm2 = (1.0 - 0.05) * UV_diffuse_Wm2 * (1.0 - np.exp(-kk_Pb * L_CI)) + (1.0 - 0.05) * UV_diffuse_Wm2 * (1.0 - exp_kk_Pd_L_CI)

    # Absorbed UV radiation by sunlit leaves (W/m²), calculated as the product of total absorbed UV radiation and the sunlit fraction.
    # This represents the portion of UV radiation absorbed by the sunlit canopy.
    AUV_sunlit_Wm2 = AUV_Wm2 * fSun

    # Absorbed UV radiation by shaded leaves (W/m²), calculated as the product of total absorbed UV radiation and the shaded fraction.
    # This represents the portion of UV radiation absorbed by the shaded canopy.
    AUV_shade_Wm2 = AUV_Wm2 * (1 - fSun)

    # Absorbed UV radiation by the soil (W/m²), calculated as the remaining UV radiation after accounting for absorption by the canopy.
    # This represents the portion of UV radiation absorbed by the ground surface.
    AUV_soil_Wm2 = (1.0 - 0.05) * UV_Wm2 - AUV_Wm2

    # Ground heat storage
    G_Wm2 = APAR_soil_Wm2 * 0.28
    check_distribution(G_Wm2, "G_Wm2")

    # Summary
    # Total absorbed shortwave radiation by sunlit leaves (ASW_Sun_Wm2) [W/m^2].
    # Calculated as the sum of absorbed PAR, NIR, and UV radiation by sunlit leaves.
    ASW_sunlit_Wm2 = APAR_sunlit_Wm2 + ANIR_sunlit_Wm2 + AUV_sunlit_Wm2
    ASW_sunlit_Wm2 = np.where(LAI == 0, 0, ASW_sunlit_Wm2)
    check_distribution(ASW_sunlit_Wm2, "ASW_sunlit_Wm2")

    # Total absorbed shortwave radiation by shaded leaves (ASW_Sh_Wm2) [W/m^2].
    # Calculated as the sum of absorbed PAR, NIR, and UV radiation by shaded leaves.
    ASW_shade_Wm2 = APAR_shade_Wm2 + ANIR_shade_Wm2 + AUV_shade_Wm2
    ASW_shade_Wm2 = np.where(LAI == 0, 0, ASW_shade_Wm2)
    check_distribution(ASW_shade_Wm2, "ASW_shade_Wm2")

    # Total absorbed shortwave radiation by soil (ASW_Soil_Wm2) [W/m^2].
    # Calculated as the sum of absorbed PAR, NIR, and UV radiation by the soil.
    ASW_soil_Wm2 = APAR_soil_Wm2 + ANIR_soil_Wm2 + AUV_soil_Wm2
    check_distribution(ASW_soil_Wm2, "ASW_soil_Wm2")

    # Total absorbed PAR by sunlit leaves (APAR_Sun_Wm2) [W/m^2].
    # Adjusted to zero where LAI is zero.
    APAR_sunlit_Wm2 = np.where(LAI == 0, 0, APAR_sunlit_Wm2)
    check_distribution(APAR_sunlit_Wm2, "APAR_sunlit_Wm2")

    # Total absorbed PAR by sunlit leaves (APAR_Sun_μmolm2s1) [μmol/m^2/s].
    # Converted from APAR_Sun_Wm2 using a factor of 4.56.
    APAR_sunlit_μmolm2s1 = APAR_sunlit_Wm2 * 4.56
    check_distribution(APAR_sunlit_μmolm2s1, "APAR_sunlit_μmolm2s1")

    # Total absorbed PAR by shaded leaves (APAR_Sh_Wm2) [W/m^2].
    # Adjusted to zero where LAI is zero.
    APAR_shade_Wm2 = np.where(LAI == 0, 0, APAR_shade_Wm2)
    check_distribution(APAR_shade_Wm2, "APAR_shade_Wm2")

    # Total absorbed PAR by shaded leaves (APAR_Sh_μmolm2s1) [μmol/m^2/s].
    # Converted from APAR_Sh_Wm2 using a factor of 4.56.
    APAR_shade_μmolm2s1 = APAR_shade_Wm2 * 4.56
    check_distribution(APAR_shade_μmolm2s1, "APAR_shade_μmolm2s1")

    # Return values as a dictionary
    return {
        "fSun": fSun,
        "APAR_sunlit_μmolm2s1": APAR_sunlit_μmolm2s1,
        "APAR_shade_μmolm2s1": APAR_shade_μmolm2s1,
        "ASW_sunlit_Wm2": ASW_sunlit_Wm2,
        "ASW_shade_Wm2": ASW_shade_Wm2,
        "ASW_soil_Wm2": ASW_soil_Wm2,
        "G_Wm2": G_Wm2
    }
