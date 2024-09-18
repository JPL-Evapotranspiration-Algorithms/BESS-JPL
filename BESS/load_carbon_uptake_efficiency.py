from os.path import join, abspath, dirname

import rasters as rt
from rasters import RasterGeometry, Raster

def load_carbon_uptake_efficiency(geometry: RasterGeometry, resampling: str = "cubic") -> Raster:
    filename = join(abspath(dirname(__file__)), "carbon_uptake_efficiency.tif")
    image = rt.Raster.open(filename, geometry=geometry, resampling=resampling)

    return image
