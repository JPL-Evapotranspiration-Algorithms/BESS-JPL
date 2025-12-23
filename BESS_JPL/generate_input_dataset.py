from os.path import join, abspath, dirname
import pandas as pd
import numpy as np
from ECOv002_calval_tables import load_calval_table
from FLiESANN import process_FLiESANN_table, load_ECOv002_calval_FLiESANN_inputs
from .ECOv002_static_tower_BESS_inputs import load_ECOv002_static_tower_BESS_inputs
from .process_BESS_table import process_BESS_table

import logging

logger = logging.getLogger(__name__)

def generate_input_dataset():
    logger.info("Generating BESS-JPL input dataset from ECOv002 cal/val FLiESANN inputs")
    # calval_df = load_calval_table()
    inputs_df = load_ECOv002_calval_FLiESANN_inputs()

    # Ensure `time_UTC` is in datetime format
    inputs_df['time_UTC'] = pd.to_datetime(inputs_df['time_UTC'])

    # Create a `date_UTC` column by extracting the date from `time_UTC`
    inputs_df['date_UTC'] = inputs_df['time_UTC'].dt.date

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
    for col in inputs_df.columns:
        inputs_df[col] = inputs_df[col].apply(extract_scalar)

    # Load static tower BESS inputs
    static_inputs_df = load_ECOv002_static_tower_BESS_inputs()

    # Merge FLiESANN outputs with static BESS inputs on Site ID
    # FLiESANN outputs contain time-varying atmospheric and radiation inputs
    # Static inputs contain vegetation parameters
    inputs_df = inputs_df.merge(
        static_inputs_df,
        left_on='ID',
        right_on='ID',
        how='left',
        suffixes=('', '_static')
    )

    # Remove duplicate columns from the merge (keep non-static versions)
    duplicate_cols = [col for col in inputs_df.columns if col.endswith('_static')]
    inputs_df = inputs_df.drop(columns=duplicate_cols)

    # Process with BESS-JPL model
    outputs_df = process_BESS_table(inputs_df)

    inputs_filename = join(abspath(dirname(__file__)), "ECOv002-cal-val-BESS-JPL-inputs.csv")
    outputs_filename = join(abspath(dirname(__file__)), "ECOv002-cal-val-BESS-JPL-outputs.csv")

    # Save the input dataset to a CSV file
    inputs_df.to_csv(inputs_filename, index=False)

    # Save the processed results to a CSV file
    outputs_df.to_csv(outputs_filename, index=False)

    logger.info(f"Processed {len(outputs_df)} records from the full cal/val dataset")
    logger.info(f"input dataset: {inputs_filename}")
    logger.info(f"output dataset: {outputs_filename}")
    
    return outputs_df
