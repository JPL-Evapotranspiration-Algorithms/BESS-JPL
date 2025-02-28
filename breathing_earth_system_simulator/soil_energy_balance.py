import numpy as np


def soil_energy_balance(
        Ts_K: np.ndarray,
        Ta_K: np.ndarray,
        G: np.ndarray,
        VPD: np.ndarray,
        RH: np.ndarray,
        gamma: np.ndarray,
        Cp: np.ndarray,
        rhoa: np.ndarray,
        desTa: np.ndarray,
        Rs: np.ndarray,
        ASW_soil: np.ndarray,
        ALW_soil: np.ndarray,
        Ls: np.ndarray,
        epsa: np.ndarray):
    # Net radiation
    # Rn = Rnet - Rn_Sun - Rn_Sh
    sigma = 5.670373e-8  # [W m-2 K-4] (Wiki)
    Rn = np.clip(ASW_soil + ALW_soil - Ls - 4.0 * epsa * sigma * (Ta_K ** 3) * (Ts_K - Ta_K), 0, None)
    # G = Rn * 0.35

    # Latent heat
    LE = desTa / (desTa + gamma) * (Rn - G) * (RH ** (VPD / 1000.0))  # (Ryu et al., 2011)
    LE = np.clip(LE, 0, Rn)
    # Sensible heat
    H = np.clip(Rn - G - LE, 0, Rn)

    # Update temperature
    dT = np.clip(Rs / (rhoa * Cp) * H, -20, 20)
    Ts_K = Ta_K + dT

    return Rn, LE, Ts_K