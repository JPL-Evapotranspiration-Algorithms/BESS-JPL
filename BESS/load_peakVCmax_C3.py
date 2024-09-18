from os.path import join, abspath, dirname

import numpy as np
import rasters as rt

def load_peakVCmax_C3(geometry: rt.RasterGeometry, resampling: str = "cubic") -> rt.Raster:
    filename = join(abspath(dirname(__file__)), "peakVCmax_C3.tif")
    image = rt.Raster.open(filename, geometry=geometry, resampling=resampling, nodata=np.nan)

    return image
