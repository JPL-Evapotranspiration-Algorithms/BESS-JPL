from os.path import join, abspath, dirname

import rasters as rt
from rasters import Raster, RasterGeometry

def load_C4_fraction(geometry: RasterGeometry = None) -> Raster:
    filename = join(abspath(dirname(__file__)), "C4_fraction.tif")
    image = Raster.open(filename, geometry=geometry, resampling="nearest")

    return image
