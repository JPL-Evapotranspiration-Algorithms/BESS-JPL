import pandas as pd

def fix_input_dataset(bess_file, flies_file):
    # Read the BESS and FLiES datasets
    bess_df = pd.read_csv(bess_file)
    flies_df = pd.read_csv(flies_file)

    # Overwrite columns in BESS dataset if they exist in FLiES dataset
    for column in flies_df.columns:
        if column in bess_df.columns:
            bess_df[column] = flies_df[column]

    # Write the updated BESS dataset back to the file
    bess_df.to_csv(bess_file, index=False)

if __name__ == "__main__":
    bess_file = "BESS_JPL/ECOv002-cal-val-BESS-JPL-inputs.csv"
    flies_file = "BESS_JPL/ECOv002-cal-val-FLiESANN-inputs.csv"

    fix_input_dataset(bess_file, flies_file)
    print(f"Updated {bess_file} with columns from {flies_file}.")