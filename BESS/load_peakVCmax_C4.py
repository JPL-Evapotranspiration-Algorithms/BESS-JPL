from os.path import join, abspath, dirname

import rasters as rt
import numpy as np

def load_peakVCmax_C4(geometry: rt.RasterGeometry, resampling: str = "cubic") -> rt.Raster:
    filename = join(abspath(dirname(__file__)), "peakVCmax_C4.tif")
    image = rt.Raster.open(filename, geometry=geometry, resampling=resampling, nodata=np.nan)

    return image
