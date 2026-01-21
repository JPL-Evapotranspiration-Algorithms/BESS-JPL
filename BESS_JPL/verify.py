def verify() -> bool:
    """
    Verifies the correctness of the BESS model implementation by comparing
    its outputs to a reference dataset.

    This function loads a known input table and the corresponding expected output table.
    It runs the model on the input data, then compares the resulting outputs to the
    reference outputs for key variables using strict numerical tolerances. If all
    outputs match within tolerance, the function returns True. Otherwise, it prints
    which column failed and returns False.

    Returns:
        bool: True if all model outputs match the reference outputs within tolerance, False otherwise.
    """
    import pandas as pd
    import numpy as np
    from .ECOv002_calval_BESS_inputs import load_ECOv002_calval_BESS_inputs
    from .process_BESS_table import process_BESS_table
    import os

    # Load input and output tables
    input_df = load_ECOv002_calval_BESS_inputs()
    module_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(module_dir, "ECOv002-cal-val-BESS-JPL-outputs.csv")
    output_df = pd.read_csv(output_file_path)

    # Check that input dataset contains all required inputs for BESS model
    # This ensures no on-the-fly data retrieval is needed
    required_inputs = [
        # Core inputs
        'ST_C', 'NDVI', 'albedo', 'time_UTC',
        # Geometry (either geometry or lat/lon)
        # 'geometry' or both 'lat' and 'lon' required - checked separately
        'day_of_year', 'hour_of_day',
        # Meteorological inputs
        'Ta_C',  # or 'Ta'
        'RH', 'elevation_m',  # or 'elevation_km'
        'wind_speed_mps',
        # Atmospheric inputs from GEOS5FP
        'COT', 'AOT', 'vapor_gccm', 'ozone_cm',
        # Note: Ca defaults to 400 ppm if not provided
        # Radiation inputs
        'PAR_albedo', 'NIR_albedo',
        # Vegetation parameters
        'NDVI_minimum', 'NDVI_maximum', 'C4_fraction',
        'carbon_uptake_efficiency', 'kn',
        # Photosynthesis parameters
        'peakVCmax_C3', 'peakVCmax_C4',
        'ball_berry_slope_C3', 'ball_berry_slope_C4', 'ball_berry_intercept_C3',
        # Other required inputs
        'KG_climate', 'CI', 'canopy_height_meters', 'SZA_deg',
        # Note: canopy_temperature_C and soil_temperature_C default to ST_C if not provided
    ]
    
    # Check for alternative column names
    missing_inputs = []
    for col in required_inputs:
        # Handle alternative names
        if col == 'Ta_C' and 'Ta_C' not in input_df and 'Ta' not in input_df:
            missing_inputs.append('Ta_C (or Ta)')
        elif col == 'elevation_m' and 'elevation_m' not in input_df and 'elevation_km' not in input_df:
            missing_inputs.append('elevation_m (or elevation_km)')
        elif col not in ['Ta_C', 'elevation_m'] and col not in input_df:
            missing_inputs.append(col)
    
    # Check geometry requirement
    has_geometry = 'geometry' in input_df or ('lat' in input_df and 'lon' in input_df)
    if not has_geometry:
        missing_inputs.append('geometry (or lat and lon)')
    
    if missing_inputs:
        print("Input verification failed: Missing required inputs for BESS model.")
        print("The following inputs are missing from the input dataset:")
        for inp in missing_inputs:
            print(f"  - {inp}")
        print("\nThese inputs must be present to run the model without on-the-fly data retrieval.")
        return False
    
    print("Input verification passed: All required BESS inputs are present.")
    
    # Run the model on the input table
    model_df = process_BESS_table(input_df, offline_mode=True)

    # Columns to compare (model outputs)
    output_columns = [
        "G_Wm2",
        "Rn_Wm2",
        "LE_Wm2"
    ]

    # Compare each output column and collect mismatches
    mismatches = []
    for col in output_columns:
        if col not in model_df or col not in output_df:
            mismatches.append((col, 'missing_column', None))
            continue
        model_vals = model_df[col].values
        ref_vals = output_df[col].values
        # Use numpy allclose for floating point comparison
        # Tolerances account for minor platform/version differences (macOS vs Ubuntu, different numpy versions)
        if not np.allclose(model_vals, ref_vals, rtol=1e-4, atol=1e-7, equal_nan=True):
            # Find indices where values differ
            diffs = np.abs(model_vals - ref_vals)
            max_diff = np.nanmax(diffs)
            idxs = np.where(~np.isclose(model_vals, ref_vals, rtol=1e-4, atol=1e-7, equal_nan=True))[0]
            mismatch_info = {
                'indices': idxs.tolist(),
                'model_values': model_vals[idxs].tolist(),
                'ref_values': ref_vals[idxs].tolist(),
                'diffs': diffs[idxs].tolist(),
                'max_diff': float(max_diff)
            }
            mismatches.append((col, 'value_mismatch', mismatch_info))
    if mismatches:
        print("Verification failed. Details:")
        for col, reason, info in mismatches:
            if reason == 'missing_column':
                print(f"  Missing column: {col}")
            elif reason == 'value_mismatch':
                print(f"  Mismatch in column: {col}")
                print(f"    Max difference: {info['max_diff']}")
                print(f"    Indices off: {info['indices']}")
                print(f"    Model values: {info['model_values']}")
                print(f"    Reference values: {info['ref_values']}")
                print(f"    Differences: {info['diffs']}")
        return False
    return True
