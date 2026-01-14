import os
import pandas as pd
import geopandas as gpd

def load_ECOv002_calval_BESS_inputs() -> gpd.GeoDataFrame:
    """
    Load the input data for the BESS model from the ECOSTRESS Collection 2 Cal-Val dataset as a GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the input data with geometry.
    """

    # Define the path to the input CSV file relative to this module's directory
    module_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(module_dir, "ECOv002-cal-val-BESS-JPL-inputs.csv")

    # Load the input data into a DataFrame
    inputs_df = pd.read_csv(input_file_path)

    # Convert the DataFrame to a GeoDataFrame using the geometry column
    if 'geometry' not in inputs_df.columns:
        raise ValueError("The input CSV file must contain a 'geometry' column with WKT or similar geometry data.")

    inputs_gdf = gpd.GeoDataFrame(
        inputs_df,
        geometry=gpd.GeoSeries.from_wkt(inputs_df['geometry']),
        crs="EPSG:4326"  # Assuming WGS84 as the coordinate reference system
    )

    return inputs_gdf
