# %% [markdown]
# # Running BESS for an ECOSTRESS Scene
# 
# This is an example of running the artificial neural network emulator of the Breathing Earth Systems Simulator (BESS) corresponding to an ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) scene.

# %%
from dateutil import parser
from matplotlib.colors import LinearSegmentedColormap
from solar_apparent_time import UTC_to_solar
import rasters as rt
from BESS_JPL import BESS_JPL
import logging


# %% [markdown]
# Here's an example ECOSTRESS surface temperature scene.

# %%
ST_filename = "ECOv002_L2T_LSTE_34366_004_11SPS_20240728T204025_0712_01_LST.tif"
ST_cmap = "bwr"
ST_C = rt.Raster.open(ST_filename, cmap=ST_cmap) - 273.15
ST_C

# %% [markdown]
# Let's get the acquisition time of the scene.

# %%
time_UTC = parser.parse(ST_filename.split("_")[6])
geometry = ST_C.geometry
longitude = geometry.centroid_latlon.x
latitude = geometry.centroid_latlon.y
time_solar = UTC_to_solar(time_UTC, longitude)
doy_solar = time_solar.timetuple().tm_yday
hour_of_day_solar = time_solar.hour + time_solar.minute / 60 + time_solar.second / 3600
print(f"{time_UTC:%Y-%m-%d %H:%M:%S} UTC")
print(f"{time_solar:%Y-%m-%d %H:%M:%S} solar apparent time at longitude {longitude}")
print(f"day of year {doy_solar} at longitude {longitude}")
print(f"hour of day {hour_of_day_solar} at longitude {longitude}")

# %%
albedo_filename = "ECOv002_L2T_STARS_11SPS_20240728_0712_01_albedo.tif"
albedo_cmap = LinearSegmentedColormap.from_list(name="albedo", colors=["black", "white"])
albedo = rt.Raster.open(albedo_filename, cmap=albedo_cmap)
albedo

# %%
NDVI_filename = "ECOv002_L2T_STARS_11SPS_20240728_0712_01_NDVI.tif"
NDVI = rt. Raster.open(NDVI_filename)

NDVI_COLORMAP_ABSOLUTE = LinearSegmentedColormap.from_list(
    name="NDVI",
    colors=[
        (0, "#0000ff"),
        (0.4, "#000000"),
        (0.5, "#745d1a"),
        (0.6, "#e1dea2"),
        (0.8, "#45ff01"),
        (1, "#325e32")
    ]
)

NDVI.cmap = NDVI_COLORMAP_ABSOLUTE
NDVI

# %%
Ta_filename = "ECOv002_L3T_MET_34366_004_11SPS_20240728T204025_0712_01_Ta.tif"
Ta_C = rt.Raster.open(Ta_filename)
Ta_C.cmap = "bwr"
Ta_C

# %%
RH_filename = "ECOv002_L3T_MET_34366_004_11SPS_20240728T204025_0712_01_RH.tif"
RH = rt.Raster.open(RH_filename)
RH.cmap = "bwr_r"
RH

# %%
BESS_results = BESS_JPL(
    hour_of_day=hour_of_day_solar,
    day_of_year=doy_solar,
    geometry=geometry,
    time_UTC=time_UTC,
    ST_C=ST_C,
    NDVI=NDVI,
    albedo=albedo,
    Ta_C=Ta_C,
    RH=RH
)

# %%
BESS_results["GPP"]

# %%
BESS_results["Rn"]

# %%
BESS_results["LE"]

# %%



