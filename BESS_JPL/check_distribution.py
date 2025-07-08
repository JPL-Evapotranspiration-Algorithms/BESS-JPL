"""
Replacement for check_distribution package to avoid pycksum dependency.

This module provides data validation functionality that was previously
provided by the check-distribution package, which depends on pycksum
that doesn't compile on Python 3.12.
"""

import logging
from typing import Union, Optional
from datetime import datetime

# Try to import numpy, but make it optional
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


logger = logging.getLogger(__name__)


def check_distribution(
    data: Union["np.ndarray", float, int, None], 
    name: str, 
    time_UTC: Optional[datetime] = None
) -> None:
    """
    Check and validate data distribution, logging warnings for invalid values.
    
    This function validates input data and logs information about the distribution,
    including warnings for NaN values, infinite values, and basic statistics.
    
    Args:
        data: The data to validate (numpy array, scalar, or None)
        name: Name of the variable for logging purposes
        time_UTC: Optional timestamp for logging context
        
    Returns:
        None
        
    Raises:
        Nothing - this function only logs warnings and information
    """
    
    # Create log prefix with time if provided
    time_prefix = f"[{time_UTC}] " if time_UTC else ""
    
    if data is None:
        logger.warning(f"{time_prefix}{name}: data is None")
        return
    
    # Handle numpy arrays or array-like objects
    if hasattr(data, 'shape') and hasattr(data, 'dtype'):
        # This is likely a numpy array or array-like object
        
        if hasattr(data, 'size') and data.size == 0:
            logger.warning(f"{time_prefix}{name}: array is empty")
            return
        
        if HAS_NUMPY:
            # Use numpy functions if available
            # Check for NaN values
            if np.any(np.isnan(data)):
                nan_count = np.sum(np.isnan(data))
                nan_percentage = (nan_count / data.size) * 100
                logger.warning(f"{time_prefix}{name}: contains {nan_count} NaN values ({nan_percentage:.1f}%)")
            
            # Check for infinite values  
            if np.any(np.isinf(data)):
                inf_count = np.sum(np.isinf(data))
                inf_percentage = (inf_count / data.size) * 100
                logger.warning(f"{time_prefix}{name}: contains {inf_count} infinite values ({inf_percentage:.1f}%)")
                
            # Calculate statistics for finite values only
            finite_data = data[np.isfinite(data)]
            if len(finite_data) > 0:
                min_val = np.min(finite_data)
                max_val = np.max(finite_data)
                mean_val = np.mean(finite_data)
                std_val = np.std(finite_data)
                
                logger.debug(f"{time_prefix}{name}: shape={data.shape}, "
                            f"min={min_val:.6f}, max={max_val:.6f}, "
                            f"mean={mean_val:.6f}, std={std_val:.6f}")
            else:
                logger.warning(f"{time_prefix}{name}: no finite values found")
        else:
            # Fallback when numpy is not available
            logger.debug(f"{time_prefix}{name}: array-like data validated (numpy not available for detailed analysis)")
            
    else:
        # Handle scalar values
        try:
            # Convert to float to check for NaN/inf
            float_val = float(data)
            
            if HAS_NUMPY:
                if np.isnan(float_val):
                    logger.warning(f"{time_prefix}{name}: scalar value is NaN")
                elif np.isinf(float_val):
                    logger.warning(f"{time_prefix}{name}: scalar value is infinite")
                else:
                    logger.debug(f"{time_prefix}{name}: scalar value = {float_val:.6f}")
            else:
                # Basic validation without numpy
                import math
                if math.isnan(float_val):
                    logger.warning(f"{time_prefix}{name}: scalar value is NaN")
                elif math.isinf(float_val):
                    logger.warning(f"{time_prefix}{name}: scalar value is infinite")
                else:
                    logger.debug(f"{time_prefix}{name}: scalar value = {float_val:.6f}")
                
        except (ValueError, TypeError) as e:
            logger.warning(f"{time_prefix}{name}: cannot convert to float for validation: {e}")