# Import necessary libraries
import pandas as pd
import numpy as np
from GEOS5FP import GEOS5FP
from NASADEM import NASADEMConnection
from ECOv002_calval_tables import load_calval_table
from FLiESANN import process_FLiESANN_table
from BESS_JPL import load_ECOv002_static_tower_BESS_inputs, process_BESS_table

# Load the calibration/validation table
def main():
    calval_df = load_calval_table()

    # Ensure `time_UTC` is in datetime format
    calval_df['time_UTC'] = pd.to_datetime(calval_df['time_UTC'])

    # Create a `date_UTC` column by extracting the date from `time_UTC`
    calval_df['date_UTC'] = calval_df['time_UTC'].dt.date

    # Filter the dataset to only include the first date
    first_date = calval_df['date_UTC'].min()
    calval_df = calval_df[calval_df['date_UTC'] == first_date]

    # Initialize connections for GEOS5FP and NASADEM data
    GEOS5FP_connection = GEOS5FP(download_directory="GEOS5FP_download")
    NASADEM_connection = NASADEMConnection(download_directory="NASADEM_download")

    # Process the filtered dataset with FLiESANN to get atmospheric inputs
    # Defaults: COT=0, AOT=0, vapor_gccm=0, ozone_cm=0.3
    FLiES_results_df = process_FLiESANN_table(
        calval_df,  # Use dataset with atmospheric defaults
        GEOS5FP_connection=GEOS5FP_connection,
        NASADEM_connection=NASADEM_connection,
        row_wise=True
    )

    # Convert any array-like values to scalars by extracting first element if needed
    def extract_scalar(x):
        if isinstance(x, pd.DataFrame):
            # Handle DataFrame - extract first value
            return x.iloc[0, 0] if not x.empty else x
        elif isinstance(x, pd.Series):
            # Handle Series - extract first value
            return x.iloc[0] if len(x) > 0 else x
        elif isinstance(x, np.ndarray):
            # Handle numpy arrays
            return x.item() if x.size == 1 else x.flat[0] if x.size > 0 else x
        elif isinstance(x, list):
            # Handle lists
            return x[0] if len(x) > 0 else x
        else:
            # Return as-is for scalars
            return x
    
    # Apply extraction to all columns
    for col in FLiES_results_df.columns:
        FLiES_results_df[col] = FLiES_results_df[col].apply(extract_scalar)

    # Load static tower BESS inputs
    static_inputs_df = load_ECOv002_static_tower_BESS_inputs()

    # Merge FLiESANN outputs with static BESS inputs on Site ID
    # FLiESANN outputs contain time-varying atmospheric and radiation inputs
    # Static inputs contain vegetation parameters
    model_inputs_df = FLiES_results_df.merge(
        static_inputs_df,
        left_on='ID',
        right_on='ID',
        how='left',
        suffixes=('', '_static')
    )

    # Remove duplicate columns from the merge (keep non-static versions)
    duplicate_cols = [col for col in model_inputs_df.columns if col.endswith('_static')]
    model_inputs_df = model_inputs_df.drop(columns=duplicate_cols)

    # Process with BESS-JPL model
    BESS_results_df = process_BESS_table(model_inputs_df)

    # Save the processed results to a CSV file
    BESS_results_df.to_csv("ECOv002-cal-val-BESS-JPL-inputs-single-day.csv", index=False)

if __name__ == "__main__":
    main()
