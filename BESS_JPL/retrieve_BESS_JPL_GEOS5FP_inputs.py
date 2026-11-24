from typing import Union, List
from datetime import datetime
import logging
import numpy as np

from rasters import Raster, RasterGeometry
import rasters as rt

from check_distribution import check_distribution
from GEOS5FP import GEOS5FP

from .exceptions import MissingOfflineParameter

logger = logging.getLogger(__name__)


def retrieve_BESS_JPL_GEOS5FP_inputs(
        time_UTC: Union[datetime, List[datetime]],
        geometry: RasterGeometry,
        albedo: Union[Raster, np.ndarray],
        GEOS5FP_connection: GEOS5FP = None,
        Ta_C: Union[Raster, np.ndarray] = None,
        RH: Union[Raster, np.ndarray] = None,
        COT: Union[Raster, np.ndarray] = None,
        AOT: Union[Raster, np.ndarray] = None,
        vapor_gccm: Union[Raster, np.ndarray] = None,
        ozone_cm: Union[Raster, np.ndarray] = None,
        PAR_albedo: Union[Raster, np.ndarray] = None,
        NIR_albedo: Union[Raster, np.ndarray] = None,
        Ca: Union[Raster, np.ndarray] = None,
        wind_speed_mps: Union[Raster, np.ndarray] = None,
        resampling: str = "cubic",
        verbose: bool = False,
        offline_mode: bool = False) -> dict:
    """
    Retrieve GEOS-5 FP meteorological inputs for BESS-JPL model.

    Retrieves each missing variable using dedicated per-variable methods on
    ``GEOS5FPConnection`` (``COT()``, ``AOT()``, ``vapor_gccm()``, ``ozone_cm()``,
    ``Ta_C()``, ``RH()``, ``CO2SC()``, ``wind_speed_mps()``, ``ALBEDO()``,
    ``ALBVISDR()``, ``ALBNIRDR()``) so gridded raster scenes and point/batch
    geometries return consistent array/raster types regardless of whether one
    or multiple variables are missing.
    """
    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP()

    missing_vars = [
        name for name, val in [
            ("COT", COT),
            ("AOT", AOT),
            ("vapor_gccm", vapor_gccm),
            ("ozone_cm", ozone_cm),
            ("Ta_C", Ta_C),
            ("RH", RH),
            ("CO2SC", Ca),
            ("wind_speed_mps", wind_speed_mps),
            ("PAR_albedo", PAR_albedo),
            ("NIR_albedo", NIR_albedo),
        ]
        if val is None
    ]

    if offline_mode and missing_vars:
        raise MissingOfflineParameter(
            f"missing offline parameters for BESS: {', '.join(missing_vars)}"
        )

    if verbose:
        if missing_vars:
            logger.info(f"Retrieving GEOS-5 FP variables: {', '.join(missing_vars)}")
        else:
            logger.info("All GEOS-5 FP inputs provided, no retrieval needed.")

    results = {"albedo": albedo}

    if COT is None:
        COT = GEOS5FP_connection.COT(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    results["COT"] = COT

    if AOT is None:
        AOT = GEOS5FP_connection.AOT(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    results["AOT"] = AOT

    if vapor_gccm is None:
        vapor_gccm = GEOS5FP_connection.vapor_gccm(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    results["vapor_gccm"] = vapor_gccm

    if ozone_cm is None:
        ozone_cm = GEOS5FP_connection.ozone_cm(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    results["ozone_cm"] = ozone_cm

    if Ta_C is None:
        Ta_C = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        check_distribution(Ta_C, "Ta_C")
    results["Ta_C"] = Ta_C

    if RH is None:
        RH = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        check_distribution(RH, "RH")
    results["RH"] = RH

    if Ca is None:
        Ca = GEOS5FP_connection.CO2SC(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        check_distribution(Ca, "Ca")
    results["Ca"] = Ca

    if wind_speed_mps is None:
        wind_speed_mps = GEOS5FP_connection.wind_speed_mps(
            time_UTC=time_UTC, geometry=geometry, resampling=resampling
        )
        wind_speed_mps = rt.clip(wind_speed_mps, 0.1, None)
        check_distribution(wind_speed_mps, "wind_speed_mps")
    results["wind_speed_mps"] = wind_speed_mps

    if PAR_albedo is None or NIR_albedo is None:
        albedo_NWP = GEOS5FP_connection.ALBEDO(time_UTC=time_UTC, geometry=geometry, resampling=resampling)

    if PAR_albedo is None:
        RVIS_NWP = GEOS5FP_connection.ALBVISDR(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        PAR_albedo = rt.clip(albedo * (RVIS_NWP / albedo_NWP), 0, 1)
    results["PAR_albedo"] = PAR_albedo

    if NIR_albedo is None:
        RNIR_NWP = GEOS5FP_connection.ALBNIRDR(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        NIR_albedo = rt.clip(albedo * (RNIR_NWP / albedo_NWP), 0, 1)
    results["NIR_albedo"] = NIR_albedo

    required_keys = [
        "albedo", "Ta_C", "RH", "COT", "AOT", "vapor_gccm", "ozone_cm",
        "PAR_albedo", "NIR_albedo", "Ca", "wind_speed_mps"
    ]
    missing_keys = [key for key in required_keys if key not in results]
    if missing_keys:
        raise ValueError(f"Missing required keys in results: {missing_keys}")

    return results
