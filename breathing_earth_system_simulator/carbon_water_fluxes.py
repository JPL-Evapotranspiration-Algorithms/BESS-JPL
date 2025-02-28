from os.path import join, abspath, dirname
from typing import Union

import numpy as np
import rasters as rt
from rasters import Raster, RasterGeometry
from .C3_photosynthesis import calculate_C3_photosynthesis
from .C4_photosynthesis import calculate_C4_photosynthesis
from .canopy_longwave_radiation import canopy_longwave_radiation
from .canopy_energy_balance import canopy_energy_balance
from .soil_energy_balance import soil_energy_balance

PASSES = 1


def load_ball_berry_intercept_C3(self, geometry: RasterGeometry) -> Raster:
    filename = join(abspath(dirname(__file__)), "ball_berry_intercept_C3.tif")
    image = rt.Raster.open(filename, geometry=geometry, resampling=self.resampling)

    return image

def load_ball_berry_slope_C3(self, geometry: RasterGeometry) -> Raster:
    filename = join(abspath(dirname(__file__)), "ball_berry_slope_C3.tif")
    image = rt.Raster.open(filename, geometry=geometry, resampling=self.resampling)

    return image

def load_ball_berry_slope_C4(self, geometry: RasterGeometry) -> Raster:
    filename = join(abspath(dirname(__file__)), "ball_berry_slope_C4.tif")
    image = rt.Raster.open(filename, geometry=geometry, resampling=self.resampling)

    return image

def carbon_water_fluxes(
        canopy_temperature_K: np.ndarray,  # canopy temperature in Kelvin
        soil_temperature_K: np.ndarray,  # soil temperature in Kelvin
        LAI: np.ndarray,  # leaf area index
        Ta_K: np.ndarray,  # air temperature in Kelvin
        APAR_sunlit: np.ndarray,  # sunlit leaf absorptance to photosynthetically active radiation [umol m-2 s-1]
        APAR_shaded: np.ndarray,  # shaded leaf absorptance to photosynthetically active radiation [umol m-2 s-1]
        ASW_sunlit: np.ndarray,  # sunlit absorbed shortwave radiation [W m-2]
        ASW_shaded: np.ndarray,  # shaded absorbed shortwave radiation [W m-2]
        ASW_soil: np.ndarray,  # absorbed shortwave radiation in soil
        Vcmax25_sunlit: np.ndarray,  # sunlit maximum carboxylation rate at 25C
        Vcmax25_shaded: np.ndarray,  # shaded maximum carboxylation rate at 25C
        ball_berry_slope: np.ndarray,  # Ball-Berry slope
        ball_berry_intercept: Union[np.ndarray, float],  # Ball-Berry intercept
        sunlit_fraction: np.ndarray,  # sunlit fraction of canopy
        G: np.ndarray,  # soil heat flux in W m-2
        SZA: np.ndarray,  # solar zenith angle in degrees
        Ca: np.ndarray,  # atmospheric CO2 concentration in umol mol-1
        Ps_Pa: np.ndarray,  # surface pressure in Pascal
        gamma: np.ndarray,  # psychrometric constant in Pa K-1
        Cp: np.ndarray,  # specific heat capacity of air in J kg-1 K-1
        rhoa: np.ndarray,  # air density in kg m-3
        VPD_Pa: np.ndarray,  # vapor pressure deficit in Pascal
        RH: np.ndarray,  # relative humidity as a fraction
        desTa: np.ndarray,
        ddesTa: np.ndarray,
        epsa: np.ndarray,
        Rc: np.ndarray,
        Rs: np.ndarray,
        carbon_uptake_efficiency: np.ndarray,  # intrinsic quantum efficiency of carbon uptake
        fStress: np.ndarray,
        C4_photosynthesis: bool,  # C3 or C4 photosynthesis
        passes: int = PASSES):  # number of iterations
    # carbon = 4 if C4_photosynthesis else 3
    GPP_max = 50 if C4_photosynthesis else 40

    # this model originally initialized soil and canopy temperature to air temperature
    Tf_K_sunlit = canopy_temperature_K
    Tf_K_shaded = canopy_temperature_K
    Tf_K = canopy_temperature_K
    Ts_K = soil_temperature_K

    # initialize intercellular CO2 concentration to atmospheric CO2 concentration depending on C3 or C4 photosynthesis
    chi = 0.4 if C4_photosynthesis else 0.7
    Ci_sunlit = Ca * chi
    Ci_shaded = Ca * chi

    ball_berry_intercept = ball_berry_intercept * fStress

    epsf = 0.98
    epss = 0.96

    # initialize sunlit partition (overwritten when iterations process)

    # initialize sunlit net assimilation rate to zero
    An_sunlit = Tf_K_sunlit * 0

    # initialize sunlit net radiation to zero
    Rn_sunlit = Tf_K_sunlit * 0

    # initialize sunlit latent heat flux to zero
    LE_sunlit = Tf_K_sunlit * 0

    # initialize sunlit sensible heat flux to zero
    H_sunlit = Tf_K_sunlit * 0

    # initialize shaded partition (overwritten when iterations process)

    # initialize shaded net assimilation rate to zero
    An_shaded = Tf_K_shaded * 0

    # initialize shaded net radiation to zero
    Rn_shaded = Tf_K_shaded * 0

    # initialize shaded latent heat flux to zero
    LE_shaded = Tf_K_shaded * 0

    # initialize shaded sensible heat flux to zero
    H_shaded = Tf_K_shaded * 0

    # initialize soil partition (overwritten when iterations process)

    # initialize soil net radiation to zero
    Rn_soil = Ts_K * 0

    # initialize soil latent heat flux to zero
    LE_soil = Ts_K * 0

    # Iteration
    for iter in range(1, passes + 1):

        # Longwave radiation
        # CLR:[ALW_Sun, ALW_shaded, ALW_Soil, Ls, La]
        ALW_sunlit, ALW_shaded, ALW_soil, Ls, La, Lf = canopy_longwave_radiation(
            LAI=LAI,  # leaf area index (LAI) [-]
            SZA=SZA,  # solar zenith angle (degrees)
            Ts_K=Ts_K,  # soil temperature (Ts) [K]
            Tf_K=Tf_K,  # foliage temperature (Tf) [K]
            Ta_K=Ta_K,  # air temperature (Ta) [K]
            epsa=epsa,  # clear-sky emissivity (epsa) [-]
            epsf=epsf,  # foliage emissivity (epsf) [-]
            epss=epss  # soil emissivity (epss) [-],
        )

        # calculate sunlit photosynthesis
        if C4_photosynthesis:
            # calculate sunlit photosynthesis for C4 plants
            An_sunlit = calculate_C4_photosynthesis(
                Tf_K=Tf_K_sunlit,  # sunlit leaf temperature (Tf) [K]
                Ci=Ci_sunlit,  # sunlit intercellular CO2 concentration (Ci) [umol mol-1]
                APAR=APAR_sunlit,  # sunlit leaf absorptance to photosynthetically active radiation [umol m-2 s-1]
                Vcmax25=Vcmax25_sunlit  # sunlit maximum carboxylation rate at 25C (Vcmax25) [umol m-2 s-1]
            )
        else:
            # calculate sunlit photosynthesis for C3 plants
            An_sunlit = calculate_C3_photosynthesis(
                Tf_K=Tf_K_sunlit,  # sunlit leaf temperature (Tf) [K]
                Ci=Ci_sunlit,  # sunlit intercellular CO2 concentration (Ci) [umol mol-1]
                APAR=APAR_sunlit,  # sunlit leaf absorptance to photosynthetically active radiation [umol m-2 s-1]
                Vcmax25=Vcmax25_sunlit,  # sunlit maximum carboxylation rate at 25C (Vcmax25) [umol m-2 s-1]
                Ps_Pa=Ps_Pa,  # surface pressure (Ps) [Pa]
                carbon_uptake_efficiency=carbon_uptake_efficiency  # intrinsic quantum efficiency for carbon uptake
            )

        # calculate sunlit energy balance
        Rn_sunlit_new, LE_sunlit_new, H_sunlit_new, Tf_K_sunlit_new, gs2_sunlit_new, Ci_sunlit_new = canopy_energy_balance(
            An=An_sunlit,  # net assimulation (An) [umol m-2 s-1]
            ASW=ASW_sunlit,  # total absorbed shortwave radiation by sunlit canopy (ASW) [umol m-2 s-1]
            ALW=ALW_sunlit,  # total absorbed longwave radiation by sunlit canopy (ALW) [umol m-2 s-1]
            Tf_K=Tf_K_sunlit,  # sunlit leaf temperature (Tf) [K]
            Ps_Pa=Ps_Pa,  # surface pressure (Ps) [Pa]
            Ca=Ca,  # ambient CO2 concentration (Ca) [umol mol-1]
            Ta_K=Ta_K,  # air temperature (Ta) [K]
            RH=RH,  # relative humidity (RH) [-]
            VPD_Pa=VPD_Pa,  # water vapour deficit (VPD) [Pa]
            desTa=desTa,  # 1st derivative of saturated vapour pressure (desTa)
            ddesTa=ddesTa,  # 2nd derivative of saturated vapour pressure (ddesTa)
            gamma=gamma,  # psychrometric constant (gamma) [pa K-1]
            Cp=Cp,  # specific heat of air at constant pressure (Cp) [J kg-1 K-1]
            rhoa=rhoa,  # air density (rhoa) [kg m-3]
            Rc=Rc,  # TODO is this Ra or Rc in Ball-Berry?
            ball_berry_slope=ball_berry_slope,  # Ball-Berry slope (m) [-]
            ball_berry_intercept=ball_berry_intercept,  # Ball-Berry intercept (b0) [-]
            C4_photosynthesis=C4_photosynthesis  # process for C4 plants instead of C3
        )

        # filter in sunlit energy balance estimates
        Rn_sunlit = np.where(np.isnan(Rn_sunlit_new), Rn_sunlit, Rn_sunlit_new)
        LE_sunlit = np.where(np.isnan(LE_sunlit_new), LE_sunlit, LE_sunlit_new)
        H_sunlit = np.where(np.isnan(H_sunlit_new), H_sunlit, H_sunlit_new)
        Tf_K_sunlit = np.where(np.isnan(Tf_K_sunlit_new), Tf_K_sunlit, Tf_K_sunlit_new)
        Ci_sunlit = np.where(np.isnan(Ci_sunlit_new), Ci_sunlit, Ci_sunlit_new)

        # Photosynthesis (shade)
        if C4_photosynthesis:
            An_shaded = calculate_C4_photosynthesis(
                Tf_K=Tf_K_shaded,  # shaded leaf temperature (Tf) [K]
                Ci=Ci_shaded,  # shaded intercellular CO2 concentration (Ci) [umol mol-1]
                APAR=APAR_shaded,  # shaded absorbed photosynthetically active radiation (APAR) [umol m-2 s-1]
                Vcmax25=Vcmax25_shaded  # shaded maximum carboxylation rate at 25C (Vcmax25) [umol m-2 s-1]
            )
        else:
            An_shaded = calculate_C3_photosynthesis(
                Tf_K=Tf_K_shaded,  # shaded leaf temperature (Tf) [K]
                Ci=Ci_shaded,  # shaed intercellular CO2 concentration (Ci) [umol mol-1]
                APAR=APAR_shaded,  # shaded absorbed photosynthetically active radiation (APAR) [umol m-2 s-1]
                Vcmax25=Vcmax25_shaded,  # shaded maximum carboxylation rate at 25C (Vcmax25) [umol m-2 s-1]
                Ps_Pa=Ps_Pa,  # surface pressure (Ps) [Pa]
                carbon_uptake_efficiency=carbon_uptake_efficiency  # intrinsic quantum efficiency for carbon uptake
            )

        # calculated shaded energy balance
        Rn_shaded_new, LE_shaded_new, H_shaded_new, Tf_K_shaded_new, gs2_shaded_new, Ci_shaded_new = canopy_energy_balance(
            An=An_shaded,  # net assimulation (An) [umol m-2 s-1]
            ASW=ASW_shaded,  # total absorbed shortwave radiation by shaded canopy (ASW) [umol m-2 s-1]
            ALW=ALW_shaded,  # total absorbed longwave radiation by shaded canopy (ALW) [umol m-2 s-1]
            Tf_K=Tf_K_shaded,  # shaded leaf temperature (Tf) [K]
            Ps_Pa=Ps_Pa,  # surface pressure (Ps) [Pa]
            Ca=Ca,  # ambient CO2 concentration (Ca) [umol mol-1]
            Ta_K=Ta_K,  # air temperature (Ta) [K]
            RH=RH,  # relative humidity as a fraction
            VPD_Pa=VPD_Pa,  # water vapour deficit (VPD) [Pa]
            desTa=desTa,  # 1st derivative of saturated vapour pressure (desTa)
            ddesTa=ddesTa,  # 2nd derivative of saturated vapour pressure (ddesTa)
            gamma=gamma,  # psychrometric constant (gamma) [pa K-1]
            Cp=Cp,  # specific heat of air (Cp) [J kg-1 K-1]
            rhoa=rhoa,  # air density (rhoa) [kg m-3]
            Rc=Rc,
            ball_berry_slope=ball_berry_slope,  # Ball-Berry slope (m) [-]
            ball_berry_intercept=ball_berry_intercept,  # Ball-Berry intercept (b0) [-]
            C4_photosynthesis=C4_photosynthesis  # process for C4 plants instead of C3
        )

        # filter in shaded energy balance estimates
        Rn_shaded = np.where(np.isnan(Rn_shaded_new), Rn_shaded, Rn_shaded_new)
        LE_shaded = np.where(np.isnan(LE_shaded_new), LE_shaded, LE_shaded_new)
        H_shaded = np.where(np.isnan(H_shaded_new), H_shaded, H_shaded_new)
        Tf_K_shaded = np.where(np.isnan(Tf_K_shaded_new), Tf_K_shaded, Tf_K_shaded_new)
        Ci_shaded = np.where(np.isnan(Ci_shaded_new), Ci_shaded, Ci_shaded_new)

        # calculate soil energy balance
        Rn_soil_new, LE_soil_new, Ts_K_soil_new = soil_energy_balance(
            Ts_K=Ts_K,  # soil temperature in Kelvin
            Ta_K=Ta_K,  # air temperature in Kelvin
            G=G,  # soil heat flux (G) [W m-2]
            VPD=VPD_Pa,  # water vapour deficit in Pascal
            RH=RH,  # relative humidity as a fraction
            gamma=gamma,  # psychrometric constant (gamma) [pa K-1]
            Cp=Cp,  # specific heat of air (Cp) [J kg-1 K-1]
            rhoa=rhoa,  # air density (rhoa) [kg m-3]
            desTa=desTa,
            Rs=Rs,
            ASW_soil=ASW_soil,  # total absorbed shortwave radiation by soil (ASW) [umol m-2 s-1]
            ALW_soil=ALW_soil,  # total absorbed longwave radiation by soil (ALW) [umol m-2 s-1]
            Ls=Ls,
            epsa=epsa
        )

        # filter in soil energy balance estimates
        # where new estimates are missing, retain the prior estimates
        Rn_soil = np.where(np.isnan(Rn_soil_new), Rn_soil, Rn_soil_new)
        LE_soil = np.where(np.isnan(LE_soil_new), LE_soil, LE_soil_new)
        Ts_K = np.where(np.isnan(Ts_K_soil_new), Ts_K, Ts_K_soil_new)

        # combine sunlit and shaded foliage temperatures
        Tf_K_new = (((Tf_K_sunlit ** 4) * sunlit_fraction + (Tf_K_shaded ** 4) * (1 - sunlit_fraction)) ** 0.25)
        Tf_K = np.where(np.isnan(Tf_K_new), Tf_K, Tf_K_new)

    # calculate canopy latent heat flux
    LE_canopy = np.clip(LE_sunlit + LE_shaded, 0, 1000)

    # calculate latent heat flux
    LE = np.clip(LE_sunlit + LE_shaded + LE_soil, 0, 1000)  # [W m-2]

    # calculate gross primary productivity
    GPP = np.clip(An_sunlit + An_shaded, 0, GPP_max)  # [umol m-2 s-1]

    # calculate canopy net radiation
    Rn_canopy = np.clip(Rn_sunlit + Rn_shaded, 0, None)

    # calculate net radiation
    Rn = np.clip(Rn_sunlit + Rn_shaded + Rn_soil, 0, 1000)  # [W m-2]

    return GPP, LE, LE_soil, LE_canopy, Rn, Rn_soil, Rn_canopy
