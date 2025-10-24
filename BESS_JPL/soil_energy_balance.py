import numpy as np


def soil_energy_balance(
        Ts_K: np.ndarray,
        Ta_K: np.ndarray,
        G_Wm2: np.ndarray,
        VPD_Pa: np.ndarray,
        RH: np.ndarray,
        gamma: np.ndarray,
        Cp: np.ndarray,
        rhoa: np.ndarray,
        desTa: np.ndarray,
        Rs: np.ndarray,
        ASW_soil_Wm2: np.ndarray,
        ALW_soil_Wm2: np.ndarray,
        Ls: np.ndarray,
        epsa: np.ndarray):
    # Net radiation
    # Rn = Rnet - Rn_Sun - Rn_Sh
    sigma = 5.670373e-8  # [W m-2 K-4] (Wiki)
    Rn_soil_Wm2 = np.clip(ASW_soil_Wm2 + ALW_soil_Wm2 - Ls - 4.0 * epsa * sigma * (Ta_K ** 3) * (Ts_K - Ta_K), 0, None)
    # G = Rn * 0.35

    # Latent heat
    LE_soil_Wm2 = desTa / (desTa + gamma) * (Rn_soil_Wm2 - G_Wm2) * (RH ** (VPD_Pa / 1000.0))  # (Ryu et al., 2011)
    LE_soil_Wm2 = np.clip(LE_soil_Wm2, 0, Rn_soil_Wm2)
    # Sensible heat
    H_soil_Wm2 = np.clip(Rn_soil_Wm2 - G_Wm2 - LE_soil_Wm2, 0, Rn_soil_Wm2)

    # Update temperature
    dT = np.clip(Rs / (rhoa * Cp) * H_soil_Wm2, -20, 20)
    Ts_K = Ta_K + dT

    return Rn_soil_Wm2, LE_soil_Wm2, Ts_K