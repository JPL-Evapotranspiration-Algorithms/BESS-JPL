from typing import Union, List
from datetime import datetime
import numpy as np

from rasters import Raster, RasterGeometry
import rasters as rt

from check_distribution import check_distribution

from GEOS5FP import GEOS5FP

import logging
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
        verbose: bool = False) -> dict:
    """
    Retrieve GEOS-5 FP meteorological inputs for BESS-JPL model.
    
    This function retrieves meteorological variables from GEOS-5 FP data products
    when they are not provided as inputs. All missing variables are retrieved in
    a single efficient `.query()` call to minimize network requests and improve
    performance.
    
    Parameters
    ----------
    time_UTC : Union[datetime, List[datetime]]
        UTC time for data retrieval. Can be a single datetime or list of datetimes
        for point-by-point queries.
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
    vapor_gccm : Union[Raster, np.ndarray], optional
        Water vapor [g cm⁻²]. Retrieved from GEOS-5 FP if None.
    ozone_cm : Union[Raster, np.ndarray], optional
        Ozone column [cm]. Retrieved from GEOS-5 FP if None.
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
        - vapor_gccm : Water vapor [g cm⁻²]
        - ozone_cm : Ozone column [cm]
        - albedo_visible : Surface albedo in visible wavelengths [-]
        - albedo_NIR : Surface albedo in near-infrared wavelengths [-]
        - Ca : Atmospheric CO₂ concentration [ppm]
        - wind_speed_mps : Wind speed [m s⁻¹]
    
    Notes
    -----
    The visible and NIR albedo are calculated by scaling the input albedo with
    the ratio of GEOS-5 FP directional albedo products to total albedo.
    
    All missing GEOS-5 FP variables are retrieved in a single `.query()` call
    for optimal performance, reducing network overhead and improving efficiency.
    
    When time_UTC is a list, it handles point-by-point queries where each point
    may have a different datetime.
    """
    # Create GEOS-5 FP connection if not provided
    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP()
    
    # Determine which variables need to be retrieved from GEOS-5 FP
    variables_to_retrieve = []
    
    # Atmospheric parameters (from FLiESANN)
    if COT is None:
        variables_to_retrieve.append("COT")
    if AOT is None:
        variables_to_retrieve.append("AOT")
    if vapor_gccm is None:
        variables_to_retrieve.append("vapor_gccm")
    if ozone_cm is None:
        variables_to_retrieve.append("ozone_cm")
    
    # Meteorological parameters
    if Ta_C is None:
        variables_to_retrieve.append("Ta_C")
    if RH is None:
        variables_to_retrieve.append("RH")
    if Ca is None:
        variables_to_retrieve.append("CO2SC")
    if wind_speed_mps is None:
        variables_to_retrieve.append("wind_speed_mps")
    
    # Albedo products needed for visible/NIR calculations
    if PAR_albedo is None or NIR_albedo is None:
        variables_to_retrieve.append("ALBEDO")
    if PAR_albedo is None:
        variables_to_retrieve.append("ALBVISDR")
    if NIR_albedo is None:
        variables_to_retrieve.append("ALBNIRDR")
    
    if len(variables_to_retrieve) == 0:
        logger.info("All GEOS-5 FP inputs provided, no retrieval needed.")
    else:
        logger.info(f"Retrieving GEOS-5 FP variables: {', '.join(variables_to_retrieve)}")

    # Retrieve all missing variables in a single query
    if variables_to_retrieve:
        retrieved = GEOS5FP_connection.query(
            target_variables=variables_to_retrieve,
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling,
            verbose=verbose
        )
        
        # Extract retrieved values
        if COT is None:
            COT = retrieved["COT"]
        if AOT is None:
            AOT = retrieved["AOT"]
        if vapor_gccm is None:
            vapor_gccm = retrieved["vapor_gccm"]
        if ozone_cm is None:
            ozone_cm = retrieved["ozone_cm"]
        if Ta_C is None:
            Ta_C = retrieved["Ta_C"]
            check_distribution(Ta_C, "Ta_C")
        if RH is None:
            RH = retrieved["RH"]
            check_distribution(RH, "RH")
        if Ca is None:
            Ca = retrieved["CO2SC"]
            check_distribution(Ca, "Ca")
        if wind_speed_mps is None:
            wind_speed_mps = rt.clip(retrieved["wind_speed_mps"], 0.1, None)
            check_distribution(wind_speed_mps, "wind_speed_mps")
        
        # Calculate visible and NIR albedo from retrieved products
        if PAR_albedo is None:
            albedo_NWP = retrieved["ALBEDO"]
            RVIS_NWP = retrieved["ALBVISDR"]
            PAR_albedo = rt.clip(albedo * (RVIS_NWP / albedo_NWP), 0, 1)
        
        if NIR_albedo is None:
            albedo_NWP = retrieved["ALBEDO"]
            RNIR_NWP = retrieved["ALBNIRDR"]
            NIR_albedo = rt.clip(albedo * (RNIR_NWP / albedo_NWP), 0, 1)

    return {
        "Ta_C": Ta_C,
        "RH": RH,
        "COT": COT,
        "AOT": AOT,
        "vapor_gccm": vapor_gccm,
        "ozone_cm": ozone_cm,
        "PAR_albedo": PAR_albedo,
        "NIR_albedo": NIR_albedo,
        "Ca": Ca,
        "wind_speed_mps": wind_speed_mps
    }
