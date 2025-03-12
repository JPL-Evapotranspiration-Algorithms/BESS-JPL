from os.path import join, abspath, dirname

import rasters as rt
from rasters import Raster, RasterGeometry

def load_peakVCmax_C4(geometry: RasterGeometry = None, resampling: str = "nearest") -> Raster:
    filename = join(abspath(dirname(__file__)), "peakVCmax_C4.tif")
    image = Raster.open(filename, geometry=geometry, resampling="nearest")

    return image
