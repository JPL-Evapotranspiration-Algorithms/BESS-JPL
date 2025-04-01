from os.path import join, abspath, dirname

import rasters as rt
from rasters import Raster, RasterGeometry

def load_ball_berry_intercept_C3(geometry: RasterGeometry = None, resampling: str = "nearest") -> Raster:
    filename = join(abspath(dirname(__file__)), "ball_berry_intercept_C3.tif")
    image = Raster.open(filename, geometry=geometry, resampling=resampling)

    return image
