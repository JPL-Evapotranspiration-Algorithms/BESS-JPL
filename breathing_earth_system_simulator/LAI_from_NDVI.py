from typing import Union

import numpy as np
import rasters as rt
from rasters import Raster

from .constants import KPAR, MIN_FIPAR, MAX_FIPAR, MIN_LAI, MAX_LAI


def LAI_from_NDVI(
        NDVI: Union[Raster, np.ndarray],
        min_fIPAR: float = MIN_FIPAR,
        max_fIPAR: float = MAX_FIPAR,
        min_LAI: float = MIN_LAI,
        max_LAI: float = MAX_LAI) -> Union[Raster, np.ndarray]:
    """
    Convert Normalized Difference Vegetation Index (NDVI) to Leaf Area Index (LAI).

    Parameters:
        NDVI (Union[Raster, np.ndarray]): Input NDVI data.

    Returns:
        Union[Raster, np.ndarray]: Converted LAI data.
    """
    fIPAR = rt.clip(NDVI - 0.05, min_fIPAR, max_fIPAR)
    fIPAR = np.where(fIPAR == 0, np.nan, fIPAR)
    LAI = rt.clip(-np.log(1 - fIPAR) * (1 / KPAR), min_LAI, max_LAI)

    return LAI
