import numpy as np

def process_paw_and_gao_LE(
        Rn: np.ndarray,  # net radiation (W m-2)
        Ta_K: np.ndarray,  # air temperature (K)
        VPD_Pa: np.ndarray,  # vapor pressure (Pa)
        Cp: np.ndarray,  # specific heat of air (J kg-1 K-1)
        rhoa: np.ndarray,  # air density (kg m-3)
        gamma: np.ndarray,  # psychrometric constant (Pa K-1)
        Rc: np.ndarray,
        rs: np.ndarray,
        desTa: np.ndarray,
        ddesTa: np.ndarray) -> np.ndarray:
    """
    :param Rn:  net radiation (W m-2)
    :param Ta_K:  air temperature (K)
    :param VPD_Pa:  vapor pressure (Pa)
    :param Cp:  specific heat of air (J kg-1 K-1)
    :param rhoa:  air density (kg m-3)
    :param gamma:  psychrometric constant (Pa K-1)
    :param Rc:
    :param rs:
    :param desTa:
    :param ddesTa:
    :return:  latent heat flux (W m-2)
    """
    # To reduce redundant computation
    rc = rs
    ddesTa_Rc2 = ddesTa * Rc * Rc
    gamma_Rc_rc = gamma * (Rc + rc)
    rhoa_Cp_gamma_Rc_rc = rhoa * Cp * gamma_Rc_rc

    # Solution (Paw and Gao 1988)
    a = 1.0 / 2.0 * ddesTa_Rc2 / rhoa_Cp_gamma_Rc_rc  # Eq. (10b)
    b = -1.0 - Rc * desTa / gamma_Rc_rc - ddesTa_Rc2 * Rn / rhoa_Cp_gamma_Rc_rc  # Eq. (10c)
    c = rhoa * Cp / gamma_Rc_rc * VPD_Pa + desTa * Rc / gamma_Rc_rc * Rn + 1.0 / 2.0 * ddesTa_Rc2 / rhoa_Cp_gamma_Rc_rc * Rn * Rn  # Eq. (10d) in Paw and Gao (1988)

    # calculate latent heat flux
    LE = (-b + np.sign(b) * np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)  # Eq. (10a)
    LE = np.real(LE)

    # Constraints
    # LE[LE > Rn] = Rn[LE > Rn]
    LE = np.clip(LE, 0, Rn)
    # LE[Rn < 0.0] = 0.0
    # LE[LE < 0.0] = 0.0
    # LE[Ta < 273.15] = 0.0
    LE = np.where(Ta_K < 273.15, 0, LE)

    return LE

def canopy_energy_balance(
        An: np.ndarray,  # net assimulation (An) [umol m-2 s-1],
        ASW: np.ndarray,  # total absorbed shortwave radiation by sunlit/shade canopy (ASW) [umol m-2 s-1],
        ALW: np.ndarray,  # total absorbed longwave radiation by sunlit/shade canopy (ALW) [umol m-2 s-1],
        Tf_K: np.ndarray,  # leaf temperature in Kelvin
        Ps_Pa: np.ndarray,  # surface pressure in Pascal
        Ca: np.ndarray,  # ambient CO2 concentration [umol mol-1],
        Ta_K: np.ndarray,  # air temperature in Kelvin
        RH: np.ndarray,  # relative humidity as a fraction
        VPD_Pa: np.ndarray,  # vapor pressure deficit in Pascal
        desTa: np.ndarray,  # 1st derivative of saturated vapour pressure
        ddesTa: np.ndarray,  # 2nd derivative of saturated vapour pressure
        gamma: np.ndarray,  # psychrometric constant [pa K-1],
        Cp: np.ndarray,  # specific heat of air [J kg-1 K-1],
        rhoa: np.ndarray,  # air density [kg m-3],
        Rc: np.ndarray,  # aerodynamic resistance [s m-1],
        ball_berry_slope: np.ndarray,  # Ball-Berry slope [-],
        ball_berry_intercept: np.ndarray,  # Ball-Berry intercept [-].
        C4_photosynthesis: bool):  # flag for C4 photosynthesis
    """
    =============================================================================

    Module     : Canopy energy balance
    Input      : net assimulation (An) [umol m-2 s-1],
               : total absorbed shortwave radiation by sunlit/shade canopy (ASW) [umol m-2 s-1],
               : total absorbed longwave radiation by sunlit/shade canopy (ALW) [umol m-2 s-1],
               : sunlit/shade leaf temperature (Tf) [K],
               : surface pressure (Ps) [Pa],
               : ambient CO2 concentration (Ca) [umol mol-1],
               : air temperature (Ta) [K],
               : relative humidity (RH) [-],
               : water vapour deficit (VPD) [Pa],
               : 1st derivative of saturated vapour pressure (desTa),
               : 2nd derivative of saturated vapour pressure (ddesTa),
               : psychrometric constant (gamma) [pa K-1],
               : air density (rhoa) [kg m-3],
               : aerodynamic resistance (ra) [s m-1],
               : Ball-Berry slope (m) [-],
               : Ball-Berry intercept (b0) [-].
    Output     : sunlit/shade canopy net radiation (Rn) [W m-2],
               : sunlit/shade canopy latent heat (LE) [W m-2],
               : sunlit/shade canopy sensible heat (H) [W m-2],
               : sunlit/shade leaf temperature (Tl) [K],
               : stomatal resistance to vapour transfer from cell to leaf surface (rs) [s m-1],
               : inter-cellular CO2 concentration (Ci) [umol mol-1].
    References : Paw U, K. T., & Gao, W. (1988). Applications of solutions to non-linear energy budget equations.
                 Agricultural and Forest Meteorology, 43(2), 121�145. doi:10.1016/0168-1923(88)90087-1


    Conversion from MatLab by Robert Freepartner, JPL/Raytheon/JaDa Systems
    March 2020

    =============================================================================
    """
    # Convert factor
    cf = 0.446 * (273.15 / Tf_K) * (Ps_Pa / 101325.0)

    # Stefan_Boltzmann_constant
    sigma = 5.670373e-8  # [W m-2 K-4] (Wiki)

    # Ball-Berry stomatal conductance
    # https://onlinelibrary.wiley.com/doi/full/10.1111/pce.12990
    gs1 = ball_berry_slope * RH * An / Ca + ball_berry_intercept  # [mol m-2 s-1]

    # intercellular CO2 concentration
    Ci = Ca - 1.6 * An / gs1  # [umol./mol]

    # constrain intercellular CO2 concentration based on ambient CO2 concentration differently depending on C3 or C4 photosynthesis
    Ci = np.clip(Ci, 0.2 * Ca, 0.6 * Ca) if C4_photosynthesis else np.clip(Ci, 0.5 * Ca, 0.9 * Ca)

    # Stomatal resistance to vapour transfer from cell to leaf surface
    rs = 1.0 / (gs1 / cf * 1e-2)  # [s m-1]

    # Stomatal H2O conductance
    gs2 = 1.0 / rs  # [m s-1]

    # Canopy net radiation
    Rn = np.clip(ASW + ALW - 4.0 * 0.98 * sigma * (Ta_K ** 3) * (Tf_K - Ta_K), 0, None)

    # TODO explore options for alternate latent heat flux models in the BESS canopy energy balance calculation

    # caluclate latent heat flux using Paw and Gao (1988)
    # https://www.sciencedirect.com/science/article/abs/pii/0168192388900871
    LE = process_paw_and_gao_LE(Rn, Ta_K, VPD_Pa, Cp, rhoa, gamma, Rc, rs, desTa, ddesTa)

    # update sensible heat flux
    H = np.clip(Rn - LE, 0, Rn)

    # update difference between air and canopy temperature
    dT = np.clip(Rc / (rhoa * Cp) * H, -20, 20)  # Eq. (6)

    # update canopy temperature in Kelvin
    Tf_K = Ta_K + dT

    return Rn, LE, H, Tf_K, gs2, Ci
