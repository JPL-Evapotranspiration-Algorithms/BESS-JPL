import numpy as np

def process_paw_and_gao_LE(
        Rn: np.ndarray,  # net radiation (W m-2)
        Ta_C: np.ndarray,  # air temperature (C)
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
    :param Ta_C:  air temperature (C)
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
    # LE[Ta < 0.0] = 0.0  # Now using Celsius
    LE = np.where(Ta_C < 0.0, 0, LE)

    return LE
