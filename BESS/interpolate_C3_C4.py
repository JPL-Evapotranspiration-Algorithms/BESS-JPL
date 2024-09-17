import numpy as np


def interpolate_C3_C4(C3: np.ndarray, C4: np.ndarray, C4_fraction: np.ndarray) -> np.ndarray:
    """
    Interpolate between C3 and C4 plants based on C4 fraction
    :param C3: value for C3 plants
    :param C4: value for C4 plants
    :param C4_fraction: fraction of C4 plants
    :return: interpolated value
    """
    return C3 * (1 - C4_fraction) + C4 * C4_fraction
