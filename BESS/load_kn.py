from os.path import join, abspath, dirname

import rasters as rt
import numpy as np

def load_kn(geometry: rt.RasterGeometry, resampling: str = "cubic") -> rt.Raster:
    filename = join(abspath(dirname(__file__)), "kn.tif")
    image = rt.Raster.open(filename, geometry=geometry, resampling=resampling, nodata=np.nan)

    return image
