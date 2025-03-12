from os.path import join, abspath, dirname

import rasters as rt
from rasters import Raster, RasterGeometry

def load_carbon_uptake_efficiency(geometry: RasterGeometry = None, resampling: str = "nearest") -> Raster:
    filename = join(abspath(dirname(__file__)), "carbon_uptake_efficiency.tif")
    image = Raster.open(filename, geometry=geometry, resampling=resampling)

    return image
