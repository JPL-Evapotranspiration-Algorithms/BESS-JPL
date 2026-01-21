from typing import Union, List
from datetime import datetime
import numpy as np

from rasters import Raster, RasterGeometry
import rasters as rt

from check_distribution import check_distribution

from GEOS5FP import GEOS5FP

import logging
logger = logging.getLogger(__name__)

class MissingOfflineParameter(Exception):
    """Custom exception for missing offline parameters."""
    pass

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
        - albedo : Surface albedo [-]
        - Ta_C : Air temperature [°C]
        - RH : Relative humidity [fraction, 0-1]
        - COT : Cloud optical thickness [-]
        - AOT : Aerosol optical thickness [-]
        - vapor_gccm : Water vapor [g cm⁻²]
        - ozone_cm : Ozone column [cm]
        - PAR_albedo : Surface albedo in PAR wavelengths [-]
        - NIR_albedo : Surface albedo in near-infrared wavelengths [-]
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
    
    # Initialize results dictionary
    results = {}
    
    # Add albedo (always required)
    results["albedo"] = albedo
    
    # Add provided inputs to results
    if Ta_C is not None:
        results["Ta_C"] = Ta_C
    if RH is not None:
        results["RH"] = RH
    if COT is not None:
        results["COT"] = COT
    if AOT is not None:
        results["AOT"] = AOT
    if vapor_gccm is not None:
        results["vapor_gccm"] = vapor_gccm
    if ozone_cm is not None:
        results["ozone_cm"] = ozone_cm
    if PAR_albedo is not None:
        results["PAR_albedo"] = PAR_albedo
    if NIR_albedo is not None:
        results["NIR_albedo"] = NIR_albedo
    if Ca is not None:
        results["Ca"] = Ca
    if wind_speed_mps is not None:
        results["wind_speed_mps"] = wind_speed_mps
    
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

    if offline_mode and variables_to_retrieve:
        raise MissingOfflineParameter(f"missing offline parameters for BESS: {', '.join(variables_to_retrieve)}")

    # Retrieve all missing variables in a single query
    if variables_to_retrieve:
        logger.info(f"Retrieving GEOS-5 FP variables: {', '.join(variables_to_retrieve)}")
        logger.info(f"Time UTC type: {type(time_UTC)}")
        logger.info(f"Geometry type: {type(geometry)}")
        if hasattr(time_UTC, '__len__'):
            logger.info(f"Time UTC length: {len(time_UTC)}")
        if hasattr(geometry, '__len__'):
            logger.info(f"Geometry length: {len(geometry)}")
        
        retrieved = GEOS5FP_connection.query(
            target_variables=variables_to_retrieve,
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling,
            verbose=True  # Enable verbose logging to see progress
        )
        
        logger.info(f"Retrieved keys: {list(retrieved.keys())}")
        if "CO2SC" in retrieved:
            logger.info(f"CO2SC in retrieved: {retrieved['CO2SC']}")
        else:
            logger.warning("CO2SC not in retrieved dictionary!")
        
        # Extract retrieved values and add to results
        if "COT" in retrieved:
            results["COT"] = retrieved["COT"]
        if "AOT" in retrieved:
            results["AOT"] = retrieved["AOT"]
        if "vapor_gccm" in retrieved:
            results["vapor_gccm"] = retrieved["vapor_gccm"]
        if "ozone_cm" in retrieved:
            results["ozone_cm"] = retrieved["ozone_cm"]
        if "Ta_C" in retrieved:
            results["Ta_C"] = retrieved["Ta_C"]
            check_distribution(results["Ta_C"], "Ta_C")
        if "RH" in retrieved:
            results["RH"] = retrieved["RH"]
            check_distribution(results["RH"], "RH")
        if "CO2SC" in retrieved:
            results["Ca"] = retrieved["CO2SC"]
            logger.info(f"Retrieved Ca from GEOS-5 FP: {results['Ca']}")
            logger.info(f"Ca type: {type(results['Ca'])}")
            if isinstance(results["Ca"], np.ndarray):
                logger.info(f"Ca shape: {results['Ca'].shape}, dtype: {results['Ca'].dtype}")
                logger.info(f"Ca has NaN: {np.any(np.isnan(results['Ca']))}")
            check_distribution(results["Ca"], "Ca")
        if "wind_speed_mps" in retrieved:
            results["wind_speed_mps"] = rt.clip(retrieved["wind_speed_mps"], 0.1, None)
            check_distribution(results["wind_speed_mps"], "wind_speed_mps")
        
        # Calculate visible and NIR albedo from retrieved products and add to results
        if "PAR_albedo" not in results:
            albedo_NWP = retrieved["ALBEDO"]
            RVIS_NWP = retrieved["ALBVISDR"]
            results["PAR_albedo"] = rt.clip(albedo * (RVIS_NWP / albedo_NWP), 0, 1)
        
        if "NIR_albedo" not in results:
            albedo_NWP = retrieved["ALBEDO"]
            RNIR_NWP = retrieved["ALBNIRDR"]
            results["NIR_albedo"] = rt.clip(albedo * (RNIR_NWP / albedo_NWP), 0, 1)
    
    # Apply default for Ca if not provided and not retrieved
    if 'Ca' not in results:
        logger.info("Ca not provided, using default value of 400 ppm")
        # Create an array of 400.0 with the same shape as albedo
        if isinstance(albedo, np.ndarray):
            results['Ca'] = np.full_like(albedo, 400.0, dtype=np.float64)
        else:
            results['Ca'] = 400.0
    
    # Verify all required keys are present
    required_keys = ['albedo', 'Ta_C', 'RH', 'COT', 'AOT', 'vapor_gccm', 'ozone_cm', 
                     'PAR_albedo', 'NIR_albedo', 'Ca', 'wind_speed_mps']
    missing_keys = [key for key in required_keys if key not in results]
    if missing_keys:
        raise ValueError(f"Missing required keys in results: {missing_keys}")

    return results
