from os.path import dirname, join
import logging
import sys

# Allow running as script while preserving package imports
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, dirname(dirname(__file__)))
    __package__ = "BESS_JPL"

from ECOv002_calval_tables import load_times_locations, load_calval_table
from GEOS5FP import GEOS5FP
from BESS_JPL import GEOS5FP_INPUTS

logger = logging.getLogger(__name__)

def generate_BESS_GEOS5FP_inputs(
        filename: str = None,
        update_package_data: bool = True,
        sample_size: int = None) -> None:
    logger.info("Generating BESS-JPL GEOS-5 FP input table:")

    for item in GEOS5FP_INPUTS:
        logger.info(f"  - {item}")

    # Load sample times and locations
    targets_df = load_times_locations()
    calval_table_df = load_calval_table()
    
    if sample_size is not None:
        targets_df = targets_df.sample(n=sample_size).reset_index(drop=True)

    # Create GEOS5FP connection
    GEOS5FP_connection = GEOS5FP()
    
    target_variables = [
        variable
        for variable
        in GEOS5FP_INPUTS
        if variable not in calval_table_df.columns
    ]

    # Query for FLiESANN GEOS5FP input variables
    results_df = GEOS5FP_connection.query(
        target_variables=target_variables,
        targets_df=targets_df
    )

    if update_package_data and not sample_size:
        if filename is None:
            filename = join(dirname(__file__), "ECOv002-cal-val-BESS-JPL-GEOS5FP-inputs.csv")

        results_df.to_csv(filename, index=False)

    return results_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_BESS_GEOS5FP_inputs()
