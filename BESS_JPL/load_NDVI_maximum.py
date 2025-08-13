from os.path import join, abspath, dirname

import rasters as rt
from rasters import Raster, RasterGeometry

import numpy as np

def load_NDVI_maximum(geometry: RasterGeometry = None, resampling: str = "nearest") -> Raster:
    filename = join(abspath(dirname(__file__)), "NDVI_maximum.tif")
    image = Raster.open(filename, geometry=geometry, resampling=resampling, nodata=np.nan)

    return image
