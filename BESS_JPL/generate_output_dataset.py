import os
from .ECOv002_calval_BESS_inputs import load_ECOv002_calval_BESS_inputs
from .process_BESS_table import process_BESS_table

def generate_output_dataset():
    """
    Generate the output dataset for the BESS-JPL model.
    """
    # Load the input data
    inputs_df = load_ECOv002_calval_BESS_inputs()
    
    # Perform any necessary processing to generate the outputs
    outputs_df = process_BESS_table(inputs_df)

    # Determine the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Save the outputs to a CSV file in the same directory as this script
    output_file_path = os.path.join(script_dir, "ECOv002-cal-val-BESS-JPL-outputs.csv")
    outputs_df.to_csv(output_file_path, index=False)

def main():
    generate_output_dataset()

if __name__ == "__main__":
    main()
