from os.path import join, abspath, dirname

import rasters as rt
from rasters import Raster, RasterGeometry

def load_NDVI_maximum(geometry: RasterGeometry = None) -> Raster:
    filename = join(abspath(dirname(__file__)), "NDVI_maximum.tif")
    image = Raster.open(filename, geometry=geometry, resampling="nearest")

    return image
