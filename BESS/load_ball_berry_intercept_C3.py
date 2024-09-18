from os.path import join, abspath, dirname

import rasters as rt


def load_ball_berry_intercept_C3(geometry: rt.RasterGeometry, resampling: str = "cubic") -> rt.Raster:
    filename = join(abspath(dirname(__file__)), "ball_berry_intercept_C3.tif")
    image = rt.Raster.open(filename, geometry=geometry, resampling=resampling)

    return image
