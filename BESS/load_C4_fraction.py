from os.path import join, abspath, dirname

import rasters as rt


def load_C4_fraction(geometry: rt.RasterGeometry, resampling: str = "cubic") -> rt.Raster:
    filename = join(abspath(dirname(__file__)), "C4_fraction.tif")
    image = rt.Raster.open(filename, geometry=geometry, resampling=resampling)

    return image
