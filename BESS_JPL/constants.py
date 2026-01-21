A = 0.3
KPAR = 0.5
MIN_FIPAR = 0.0
MAX_FIPAR = 1.0
MIN_LAI = 0.0
MAX_LAI = 10.0
BALL_BERRY_INTERCEPT_C4 = 0.04
RESAMPLING = "cubic"

# Default scale factor for C4 fraction
C4_FRACTION_SCALE_FACTOR = 0.01

UPSCALE_TO_DAYLIGHT = True

# GEOS-5 FP variables retrieved by the model
GEOS5FP_INPUTS = [
    "Ta_C",        # Air temperature (°C)
    "RH",          # Relative humidity (fraction)
    "COT",         # Cloud optical thickness
    "AOT",         # Aerosol optical thickness
    "PAR_albedo",    # Visible direct beam albedo
    "NIR_albedo",    # Near-infrared direct beam albedo
    "Ca",       # Atmospheric CO₂ concentration (ppm)
    "wind_speed_mps"   # Wind speed (m/s)
]
