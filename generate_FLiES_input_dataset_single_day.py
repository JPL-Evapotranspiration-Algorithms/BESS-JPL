# Import necessary libraries
import pandas as pd
from GEOS5FP import GEOS5FP
from NASADEM import NASADEMConnection
from ECOv002_calval_tables import load_calval_table
from FLiESANN import process_FLiESANN_table

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

    # Process the filtered dataset with atmospheric parameter defaults
    # Defaults: COT=0, AOT=0, vapor_gccm=0, ozone_cm=0.3
    results_df = process_FLiESANN_table(
        calval_df,  # Use dataset with atmospheric defaults
        GEOS5FP_connection=GEOS5FP_connection,
        NASADEM_connection=NASADEM_connection
    )

    # Save the processed results to a CSV file
    results_df.to_csv("ECOv002-cal-val-FLiESANN-inputs-single-day.csv", index=False)

if __name__ == "__main__":
    main()