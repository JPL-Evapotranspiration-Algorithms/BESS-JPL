from os.path import join, abspath, dirname

import rasters as rt
from rasters import Raster, RasterGeometry

import numpy as np

from .constants import *

def load_C4_fraction(
    geometry: RasterGeometry = None, 
    resampling: str = "nearest",
    scale_factor: float = C4_FRACTION_SCALE_FACTOR) -> Raster:
    filename = join(abspath(dirname(__file__)), "C4_fraction.tif")
    image = Raster.open(filename, geometry=geometry, resampling=resampling, nodata=np.nan)
    image = rt.clip(image, 0, 100)
    # Scale image to be between 0 and 1 using a multiplicative scale factor
    image *= scale_factor

    return image
