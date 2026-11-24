# Import necessary libraries
import pandas as pd
import numpy as np
from ECOv002_calval_tables import load_calval_table
from FLiESANN import process_FLiESANN_table, load_ECOv002_calval_FLiESANN_inputs
from BESS_JPL import load_ECOv002_static_tower_BESS_inputs, process_BESS_table

# Load the calibration/validation table
def main():
    # calval_df = load_calval_table()
    model_inputs_df = load_ECOv002_calval_FLiESANN_inputs()

    # Ensure `time_UTC` is in datetime format
    model_inputs_df['time_UTC'] = pd.to_datetime(model_inputs_df['time_UTC'])

    # Create a `date_UTC` column by extracting the date from `time_UTC`
    model_inputs_df['date_UTC'] = model_inputs_df['time_UTC'].dt.date

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
    for col in model_inputs_df.columns:
        model_inputs_df[col] = model_inputs_df[col].apply(extract_scalar)

    # Load static tower BESS inputs
    static_inputs_df = load_ECOv002_static_tower_BESS_inputs()

    # Merge FLiESANN outputs with static BESS inputs on Site ID
    # FLiESANN outputs contain time-varying atmospheric and radiation inputs
    # Static inputs contain vegetation parameters
    model_inputs_df = model_inputs_df.merge(
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
    BESS_results_df.to_csv("ECOv002-cal-val-BESS-JPL-inputs-full.csv", index=False)
    
    print(f"Processed {len(BESS_results_df)} records from the full cal/val dataset")
    print(f"Results saved to: ECOv002-cal-val-BESS-JPL-inputs-full.csv")

if __name__ == "__main__":
    main()
