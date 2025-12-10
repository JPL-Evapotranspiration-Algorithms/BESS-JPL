from typing import Union
from datetime import datetime
import numpy as np

from rasters import Raster, RasterGeometry
import rasters as rt

from GEOS5FP import GEOS5FP


def retrieve_BESS_JPL_GEOS5FP_inputs(
        time_UTC: datetime,
        geometry: RasterGeometry,
        albedo: Union[Raster, np.ndarray],
        GEOS5FP_connection: GEOS5FP = None,
        Ta_C: Union[Raster, np.ndarray] = None,
        RH: Union[Raster, np.ndarray] = None,
        COT: Union[Raster, np.ndarray] = None,
        AOT: Union[Raster, np.ndarray] = None,
        albedo_visible: Union[Raster, np.ndarray] = None,
        albedo_NIR: Union[Raster, np.ndarray] = None,
        Ca: Union[Raster, np.ndarray] = None,
        wind_speed_mps: Union[Raster, np.ndarray] = None,
        resampling: str = "cubic") -> dict:
    """
    Retrieve GEOS-5 FP meteorological inputs for BESS-JPL model.
    
    This function retrieves meteorological variables from GEOS-5 FP data products
    when they are not provided as inputs. It handles all the GEOS-5 FP data
    retrievals needed by the BESS-JPL model.
    
    Parameters
    ----------
    time_UTC : datetime
        UTC time for data retrieval
    geometry : RasterGeometry
        Raster geometry for spatial operations
    albedo : Union[Raster, np.ndarray]
        Surface albedo [-], used for albedo calculations
    GEOS5FP_connection : GEOS5FP, optional
        Connection to GEOS-5 FP meteorological data. If None, creates new connection.
    Ta_C : Union[Raster, np.ndarray], optional
        Air temperature [°C]. Retrieved from GEOS-5 FP if None.
    RH : Union[Raster, np.ndarray], optional
        Relative humidity [fraction, 0-1]. Retrieved from GEOS-5 FP if None.
    COT : Union[Raster, np.ndarray], optional
        Cloud optical thickness [-]. Retrieved from GEOS-5 FP if None.
    AOT : Union[Raster, np.ndarray], optional
        Aerosol optical thickness [-]. Retrieved from GEOS-5 FP if None.
    albedo_visible : Union[Raster, np.ndarray], optional
        Surface albedo in visible wavelengths (400-700 nm) [-]. 
        Calculated from GEOS-5 FP albedo products if None.
    albedo_NIR : Union[Raster, np.ndarray], optional
        Surface albedo in near-infrared wavelengths [-].
        Calculated from GEOS-5 FP albedo products if None.
    Ca : Union[Raster, np.ndarray], optional
        Atmospheric CO₂ concentration [ppm]. Retrieved from GEOS-5 FP if None.
    wind_speed_mps : Union[Raster, np.ndarray], optional
        Wind speed [m s⁻¹]. Retrieved from GEOS-5 FP if None.
    resampling : str, optional
        Resampling method for data processing. Default is "cubic".
    
    Returns
    -------
    dict
        Dictionary containing all meteorological inputs:
        - Ta_C : Air temperature [°C]
        - RH : Relative humidity [fraction, 0-1]
        - COT : Cloud optical thickness [-]
        - AOT : Aerosol optical thickness [-]
        - albedo_visible : Surface albedo in visible wavelengths [-]
        - albedo_NIR : Surface albedo in near-infrared wavelengths [-]
        - Ca : Atmospheric CO₂ concentration [ppm]
        - wind_speed_mps : Wind speed [m s⁻¹]
    
    Notes
    -----
    The visible and NIR albedo are calculated by scaling the input albedo with
    the ratio of GEOS-5 FP directional albedo products to total albedo.
    """
    # Create GEOS-5 FP connection if not provided
    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP()
    
    # Retrieve air temperature if not provided
    if Ta_C is None:
        Ta_C = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    
    # Retrieve relative humidity if not provided
    if RH is None:
        RH = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    
    # Retrieve cloud optical thickness if not provided
    if COT is None:
        COT = GEOS5FP_connection.COT(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    
    # Retrieve aerosol optical thickness if not provided
    if AOT is None:
        AOT = GEOS5FP_connection.AOT(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    
    # Calculate visible albedo from GEOS-5 FP products if not provided
    if albedo_visible is None:
        albedo_NWP = GEOS5FP_connection.ALBEDO(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        RVIS_NWP = GEOS5FP_connection.ALBVISDR(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        albedo_visible = rt.clip(albedo * (RVIS_NWP / albedo_NWP), 0, 1)
    
    # Calculate NIR albedo from GEOS-5 FP products if not provided
    if albedo_NIR is None:
        albedo_NWP = GEOS5FP_connection.ALBEDO(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        RNIR_NWP = GEOS5FP_connection.ALBNIRDR(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
        albedo_NIR = rt.clip(albedo * (RNIR_NWP / albedo_NWP), 0, 1)
    
    # Retrieve CO2 concentration if not provided
    if Ca is None:
        Ca = GEOS5FP_connection.CO2SC(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    
    # Retrieve wind speed if not provided
    if wind_speed_mps is None:
        wind_speed_mps = GEOS5FP_connection.wind_speed(time_UTC=time_UTC, geometry=geometry, resampling=resampling)
    
    return {
        "Ta_C": Ta_C,
        "RH": RH,
        "COT": COT,
        "AOT": AOT,
        "albedo_visible": albedo_visible,
        "albedo_NIR": albedo_NIR,
        "Ca": Ca,
        "wind_speed_mps": wind_speed_mps
    }
