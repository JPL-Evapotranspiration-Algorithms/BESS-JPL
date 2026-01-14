import pandas as pd
from check_distribution import check_distribution

input_filename = "BESS_JPL/ECOv002-cal-val-BESS-JPL-outputs.csv"

input_df = pd.read_csv(input_filename)

for column in input_df.columns:
    try:
        values = input_df[column].dropna().values
        check_distribution(values, column)
    except Exception as e:
        continue
