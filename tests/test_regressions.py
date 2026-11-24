from datetime import datetime
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from rasters import Point, Raster, RasterGrid

import BESS_JPL
from BESS_JPL.exceptions import MissingOfflineParameter
from BESS_JPL.process_BESS_table import process_BESS_table
from BESS_JPL.retrieve_BESS_JPL_GEOS5FP_inputs import retrieve_BESS_JPL_GEOS5FP_inputs


def _make_mock_geos5fp():
    conn = MagicMock()
    conn.query.side_effect = AssertionError("GEOS5FP.query() should not be called directly")
    conn.COT.return_value = np.array([1.2])
    conn.AOT.return_value = np.array([0.15])
    conn.vapor_gccm.return_value = np.array([1.8])
    conn.ozone_cm.return_value = np.array([0.30])
    conn.Ta_C.return_value = np.array([25.0])
    conn.RH.return_value = np.array([0.55])
    conn.CO2SC.return_value = np.array([418.0])
    conn.wind_speed_mps.return_value = np.array([0.02])  # test clipping to 0.1
    conn.ALBEDO.return_value = np.array([0.20])
    conn.ALBVISDR.return_value = np.array([0.05])
    conn.ALBNIRDR.return_value = np.array([0.35])
    return conn


def test_retrieve_bess_geos5fp_inputs_single_missing_co2sc():
    """Issue #99: Single missing variable (Ca=None, as passed from JET3) calls CO2SC() and returns dict."""
    grid = RasterGrid.from_bbox(xmin=-118.5, ymin=34.0, xmax=-118.0, ymax=34.5, cell_size=0.25)
    ones = Raster(np.ones(grid.shape, dtype=np.float64), geometry=grid)
    co2_raster = Raster(np.full(grid.shape, 421.5, dtype=np.float64), geometry=grid)

    conn = MagicMock()
    conn.query.side_effect = AssertionError("GEOS5FP.query() must not be called")
    conn.CO2SC.return_value = co2_raster

    results = retrieve_BESS_JPL_GEOS5FP_inputs(
        time_UTC=datetime(2024, 7, 28, 20, 40),
        geometry=grid,
        albedo=ones * 0.18,
        GEOS5FP_connection=conn,
        Ta_C=ones * 28.0,
        RH=ones * 0.45,
        COT=ones * 0.5,
        AOT=ones * 0.1,
        vapor_gccm=ones * 2.1,
        ozone_cm=ones * 0.29,
        PAR_albedo=ones * 0.06,
        NIR_albedo=ones * 0.30,
        Ca=None,
        wind_speed_mps=ones * 3.2,
        offline_mode=False,
    )

    assert isinstance(results, dict)
    conn.CO2SC.assert_called_once()
    np.testing.assert_allclose(np.asarray(results["Ca"]), 421.5)


def test_retrieve_bess_geos5fp_inputs_all_missing():
    """Issue #99: All missing GEOS-5 FP variables retrieved via per-variable methods and properly scaled."""
    conn = _make_mock_geos5fp()
    pt = Point(-118.25, 34.05)
    albedo = np.array([0.20])

    results = retrieve_BESS_JPL_GEOS5FP_inputs(
        time_UTC=[datetime(2024, 7, 28, 20, 40)],
        geometry=[pt],
        albedo=albedo,
        GEOS5FP_connection=conn,
        offline_mode=False,
    )

    for method_name in [
        "COT", "AOT", "vapor_gccm", "ozone_cm", "Ta_C", "RH",
        "CO2SC", "wind_speed_mps", "ALBEDO", "ALBVISDR", "ALBNIRDR"
    ]:
        getattr(conn, method_name).assert_called_once()

    np.testing.assert_allclose(results["Ca"], [418.0])
    np.testing.assert_allclose(results["wind_speed_mps"], [0.1])  # clipped from 0.02
    np.testing.assert_allclose(results["PAR_albedo"], [0.05])      # 0.20 * (0.05 / 0.20)
    np.testing.assert_allclose(results["NIR_albedo"], [0.35])      # 0.20 * (0.35 / 0.20)


def test_missing_offline_parameter_unified_exception():
    """Issue #103: BESS_JPL.MissingOfflineParameter catches errors from retrieve_BESS_JPL_GEOS5FP_inputs."""
    assert BESS_JPL.MissingOfflineParameter is MissingOfflineParameter

    with pytest.raises(BESS_JPL.MissingOfflineParameter):
        retrieve_BESS_JPL_GEOS5FP_inputs(
            time_UTC=datetime(2024, 7, 28, 20, 40),
            geometry=Point(-118.25, 34.05),
            albedo=np.array([0.20]),
            Ca=None,
            offline_mode=True,
        )


def _sample_table(include_met: bool = True, include_spectral: bool = True) -> pd.DataFrame:
    data = {
        "time_UTC": ["2019-10-02 19:09:40"],  # space-separated mixed timestamp format (#104)
        "lat": [35.799],
        "lon": [-76.656],
        "ST_C": [26.0],
        "NDVI": [0.65],
        "albedo": [0.20],
        "elevation_m": [10.0],
        "wind_speed_mps": [2.5],
        "COT": [0.5],
        "AOT": [0.1],
        "vapor_gccm": [2.0],
        "ozone_cm": [0.3],
        "NDVI_minimum": [0.2],
        "NDVI_maximum": [0.8],
        "C4_fraction": [0.1],
        "carbon_uptake_efficiency": [0.06],
        "kn": [0.5],
        "peakVCmax_C3": [60.0],
        "peakVCmax_C4": [40.0],
        "ball_berry_slope_C3": [9.0],
        "ball_berry_slope_C4": [4.0],
        "ball_berry_intercept_C3": [0.01],
        "KG_climate": [3],
        "CI": [0.75],
        "canopy_height_meters": [15.0],
        "SZA_deg": [40.0],
    }
    if include_met:
        data["Ta"] = [25.0]  # 'Ta' alias (#101)
        data["RH"] = [0.55]
    if include_spectral:
        data["PAR_albedo"] = [0.05]
        data["NIR_albedo"] = [0.35]
    return pd.DataFrame(data)


def test_process_bess_table_spectral_albedo_opt_in():
    """Issue #100: use_spectral_albedo=False preserves broadband baseline; True forwards spectral albedos."""
    df = _sample_table(include_met=True, include_spectral=True)
    captured_calls = []

    def fake_bess_jpl(**kwargs):
        captured_calls.append((kwargs["PAR_albedo"].copy(), kwargs["NIR_albedo"].copy()))
        n = len(kwargs["ST_C"])
        return {
            "GPP": np.zeros(n),
            "GPP_daily": np.zeros(n),
            "Rn_Wm2": np.zeros(n),
            "Rn_soil_Wm2": np.zeros(n),
            "Rn_canopy_Wm2": np.zeros(n),
            "LE_Wm2": np.zeros(n),
            "LE_soil_Wm2": np.zeros(n),
            "LE_canopy_Wm2": np.zeros(n),
            "G_Wm2": np.zeros(n),
        }

    with patch("BESS_JPL.process_BESS_table.BESS_JPL", side_effect=fake_bess_jpl):
        process_BESS_table(df, offline_mode=True, use_spectral_albedo=False)
        process_BESS_table(df, offline_mode=True, use_spectral_albedo=True)

    # Default False -> broadband albedo (0.20, 0.20)
    np.testing.assert_allclose(captured_calls[0][0], [0.20])
    np.testing.assert_allclose(captured_calls[0][1], [0.20])

    # Opt-in True -> table spectral albedos (0.05, 0.35)
    np.testing.assert_allclose(captured_calls[1][0], [0.05])
    np.testing.assert_allclose(captured_calls[1][1], [0.35])


def test_process_bess_table_optional_met_and_point_list_geometry():
    """Issues #101, #102, #104: Omitted Ta_C/RH, mixed timestamp, and list[Point] passed to GEOS5FP."""
    df = _sample_table(include_met=False, include_spectral=False)
    conn = _make_mock_geos5fp()
    captured_geom = []

    def fake_bess_jpl(**kwargs):
        n = len(kwargs["ST_C"])
        return {
            "GPP": np.zeros(n),
            "GPP_daily": np.zeros(n),
            "Rn_Wm2": np.zeros(n),
            "Rn_soil_Wm2": np.zeros(n),
            "Rn_canopy_Wm2": np.zeros(n),
            "LE_Wm2": np.zeros(n),
            "LE_soil_Wm2": np.zeros(n),
            "LE_canopy_Wm2": np.zeros(n),
            "G_Wm2": np.zeros(n),
        }

    orig_retrieve = retrieve_BESS_JPL_GEOS5FP_inputs

    def spy_retrieve(*args, **kwargs):
        captured_geom.append(kwargs.get("geometry"))
        return orig_retrieve(*args, **kwargs)

    with patch("BESS_JPL.process_BESS_table.retrieve_BESS_JPL_GEOS5FP_inputs", side_effect=spy_retrieve),          patch("BESS_JPL.process_BESS_table.BESS_JPL", side_effect=fake_bess_jpl):
        out = process_BESS_table(df, GEOS5FP_connection=conn, offline_mode=False)

    assert len(captured_geom) == 1
    assert isinstance(captured_geom[0], list)
    assert isinstance(captured_geom[0][0], Point)
    conn.Ta_C.assert_called_once()
    conn.RH.assert_called_once()
    assert len(out) == 1
