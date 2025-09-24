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

def carbon_water_fluxes(
        canopy_temperature_K: np.ndarray,  # canopy temperature in Kelvin
        soil_temperature_K: np.ndarray,  # soil temperature in Kelvin
        LAI: np.ndarray,  # leaf area index
        Ta_K: np.ndarray,  # air temperature in Kelvin
        APAR_sunlit_μmolm2s1: np.ndarray,  # sunlit leaf absorptance to photosynthetically active radiation [μmol m⁻² s⁻¹]
        APAR_shaded_μmolm2s1: np.ndarray,  # shaded leaf absorptance to photosynthetically active radiation [μmol m⁻² s⁻¹]
        ASW_sunlit_Wm2: np.ndarray,  # sunlit absorbed shortwave radiation [W m⁻²]
        ASW_shaded_Wm2: np.ndarray,  # shaded absorbed shortwave radiation [W m⁻²]
        ASW_soil_Wm2: np.ndarray,  # absorbed shortwave radiation in soil [W m⁻²]
        Vcmax25_sunlit: np.ndarray,  # sunlit maximum carboxylation rate at 25°C [μmol m⁻² s⁻¹]
        Vcmax25_shaded: np.ndarray,  # shaded maximum carboxylation rate at 25°C [μmol m⁻² s⁻¹]
        ball_berry_slope: np.ndarray,  # Ball-Berry slope
        ball_berry_intercept: Union[np.ndarray, float],  # Ball-Berry intercept
        sunlit_fraction: np.ndarray,  # sunlit fraction of canopy
        G_Wm2: np.ndarray,  # soil heat flux [W m⁻²]
        SZA_deg: np.ndarray,  # solar zenith angle in degrees
        Ca: np.ndarray,  # atmospheric CO2 concentration [μmol mol⁻¹]
        Ps_Pa: np.ndarray,  # surface pressure [Pa]
        gamma: np.ndarray,  # psychrometric constant [Pa K⁻¹]
        Cp: np.ndarray,  # specific heat capacity of air [J kg⁻¹ K⁻¹]
        rhoa: np.ndarray,  # air density [kg m⁻³]
        VPD_Pa: np.ndarray,  # vapor pressure deficit [Pa]
        RH: np.ndarray,  # relative humidity as a fraction
        desTa: np.ndarray,  # 1st derivative of saturated vapor pressure [Pa K⁻¹]
        ddesTa: np.ndarray,  # 2nd derivative of saturated vapor pressure [Pa K⁻²]
        epsa: np.ndarray,  # clear-sky emissivity [-]
        Rc: np.ndarray,  # canopy resistance [s m⁻¹]
        Rs: np.ndarray,  # soil resistance [s m⁻¹]
        carbon_uptake_efficiency: np.ndarray,  # intrinsic quantum efficiency of carbon uptake [-]
        fStress: np.ndarray,  # stress factor [-]
        C4_photosynthesis: bool,  # C3 or C4 photosynthesis
        passes: int = PASSES):  # number of iterations
    """
    Calculate carbon and water fluxes for a canopy-soil system.

    Parameters:
        canopy_temperature_K: Canopy temperature [K].
        soil_temperature_K: Soil temperature [K].
        LAI: Leaf area index [-].
        Ta_K: Air temperature [K].
        APAR_sunlit: Sunlit leaf absorptance to photosynthetically active radiation [μmol m⁻² s⁻¹].
        APAR_shaded: Shaded leaf absorptance to photosynthetically active radiation [μmol m⁻² s⁻¹].
        ASW_sunlit_Wm2: Sunlit absorbed shortwave radiation [W m⁻²].
        ASW_shaded: Shaded absorbed shortwave radiation [W m⁻²].
        ASW_soil_Wm2: Absorbed shortwave radiation in soil [W m⁻²].
        Vcmax25_sunlit: Sunlit maximum carboxylation rate at 25°C [μmol m⁻² s⁻¹].
        Vcmax25_shaded: Shaded maximum carboxylation rate at 25°C [μmol m⁻² s⁻¹].
        ball_berry_slope: Ball-Berry slope [-].
        ball_berry_intercept: Ball-Berry intercept [-].
        sunlit_fraction: Sunlit fraction of canopy [-].
        G: Soil heat flux [W m⁻²].
        SZA: Solar zenith angle [degrees].
        Ca: Atmospheric CO2 concentration [μmol mol⁻¹].
        Ps_Pa: Surface pressure [Pa].
        gamma: Psychrometric constant [Pa K⁻¹].
        Cp: Specific heat capacity of air [J kg⁻¹ K⁻¹].
        rhoa: Air density [kg m⁻³].
        VPD_Pa: Vapor pressure deficit [Pa].
        RH: Relative humidity as a fraction [-].
        desTa: 1st derivative of saturated vapor pressure [Pa K⁻¹].
        ddesTa: 2nd derivative of saturated vapor pressure [Pa K⁻²].
        epsa: Clear-sky emissivity [-].
        Rc: Canopy resistance [s m⁻¹].
        Rs: Soil resistance [s m⁻¹].
        carbon_uptake_efficiency: Intrinsic quantum efficiency of carbon uptake [-].
        fStress: Stress factor [-].
        C4_photosynthesis: Whether to use C4 photosynthesis (True) or C3 (False).
        passes: Number of iterations.

    Returns:
        GPP: Gross primary productivity [μmol m⁻² s⁻¹].
        LE: Latent heat flux [W m⁻²].
        LE_soil: Soil latent heat flux [W m⁻²].
        LE_canopy: Canopy latent heat flux [W m⁻²].
        Rn: Net radiation [W m⁻²].
        Rn_soil: Soil net radiation [W m⁻²].
        Rn_canopy: Canopy net radiation [W m⁻²].
    """

    # carbon = 4 if C4_photosynthesis else 3
    GPP_max = 50 if C4_photosynthesis else 40

    # this model originally initialized soil and canopy temperature to air temperature
    Tf_sunlit_K = canopy_temperature_K
    Tf_shaded_K = canopy_temperature_K
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
    net_assimilation_sunlit_μmolm2s1 = Tf_sunlit_K * 0

    # initialize sunlit net radiation to zero
    Rn_sunlit_Wm2 = Tf_sunlit_K * 0

    # initialize sunlit latent heat flux to zero
    LE_sunlit_Wm2 = Tf_sunlit_K * 0

    # initialize sunlit sensible heat flux to zero
    H_sunlit_Wm2 = Tf_sunlit_K * 0

    # initialize shaded partition (overwritten when iterations process)

    # initialize shaded net assimilation rate to zero
    net_assimilation_shade_μmolm2s1 = Tf_shaded_K * 0

    # initialize shaded net radiation to zero
    Rn_shaded_Wm2 = Tf_shaded_K * 0

    # initialize shaded latent heat flux to zero
    LE_shaded_Wm2 = Tf_shaded_K * 0

    # initialize shaded sensible heat flux to zero
    H_shaded_Wm2 = Tf_shaded_K * 0

    # initialize soil partition (overwritten when iterations process)

    # initialize soil net radiation to zero
    Rn_soil_Wm2 = Ts_K * 0

    # initialize soil latent heat flux to zero
    LE_soil_Wm2 = Ts_K * 0

    # Iteration
    for iter in range(1, passes + 1):

        # Longwave radiation
        # CLR:[ALW_Sun, ALW_shaded, ALW_Soil, Ls, La]
        ALW_sunlit_Wm2, ALW_shaded_Wm2, ALW_soil_Wm2, Ls, La, Lf = canopy_longwave_radiation(
            LAI=LAI,  # leaf area index (LAI) [-]
            SZA=SZA_deg,  # solar zenith angle (degrees)
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
            photosynthesis_sunlit_C4_results = calculate_C4_photosynthesis(
                Tf_K=Tf_sunlit_K,  # sunlit leaf temperature (Tf) [K]
                Ci_μmol_per_mol=Ci_sunlit,  # sunlit intercellular CO2 concentration (Ci) [umol mol-1]
                APAR_μmolm2s1=APAR_sunlit_μmolm2s1,  # sunlit leaf absorptance to photosynthetically active radiation [umol m-2 s-1]
                Vcmax25_μmolm2s1=Vcmax25_sunlit  # sunlit maximum carboxylation rate at 25C (Vcmax25) [umol m-2 s-1]
            )

            net_assimilation_sunlit_μmolm2s1 = photosynthesis_sunlit_C4_results['net_assimilation_C4_μmolm2s1']
        else:
            # calculate sunlit photosynthesis for C3 plants
            photosynthesis_sunlit_C3_results = calculate_C3_photosynthesis(
                Tf_K=Tf_sunlit_K,  # sunlit leaf temperature (Tf) [K]
                Ci=Ci_sunlit,  # sunlit intercellular CO2 concentration (Ci) [umol mol-1]
                APAR_μmolm2s1=APAR_sunlit_μmolm2s1,  # sunlit leaf absorptance to photosynthetically active radiation [umol m-2 s-1]
                Vcmax25=Vcmax25_sunlit,  # sunlit maximum carboxylation rate at 25C (Vcmax25) [umol m-2 s-1]
                Ps_Pa=Ps_Pa,  # surface pressure (Ps) [Pa]
                carbon_uptake_efficiency=carbon_uptake_efficiency  # intrinsic quantum efficiency for carbon uptake
            )

            net_assimilation_sunlit_μmolm2s1 = photosynthesis_sunlit_C3_results['net_assimilation_C3_μmolm2s1']

        # calculate sunlit energy balance
        Rn_sunlit_new_Wm2, LE_sunlit_new_Wm2, H_sunlit_new_Wm2, Tf_sunlit_new_K, gs2_sunlit_new, Ci_sunlit_new = canopy_energy_balance(
            An=net_assimilation_sunlit_μmolm2s1,  # net assimulation (An) [umol m-2 s-1]
            ASW_Wm2=ASW_sunlit_Wm2,  # total absorbed shortwave radiation by sunlit canopy (ASW) [W/m^2]
            ALW_Wm2=ALW_sunlit_Wm2,  # total absorbed longwave radiation by sunlit canopy (ALW) [W/m^2]
            Tf_K=Tf_sunlit_K,  # sunlit leaf temperature (Tf) [K]
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
        Rn_sunlit_Wm2 = np.where(np.isnan(Rn_sunlit_new_Wm2), Rn_sunlit_Wm2, Rn_sunlit_new_Wm2)
        LE_sunlit_Wm2 = np.where(np.isnan(LE_sunlit_new_Wm2), LE_sunlit_Wm2, LE_sunlit_new_Wm2)
        H_sunlit_Wm2 = np.where(np.isnan(H_sunlit_new_Wm2), H_sunlit_Wm2, H_sunlit_new_Wm2)
        Tf_sunlit_K = np.where(np.isnan(Tf_sunlit_new_K), Tf_sunlit_K, Tf_sunlit_new_K)
        Ci_sunlit = np.where(np.isnan(Ci_sunlit_new), Ci_sunlit, Ci_sunlit_new)

        # Photosynthesis (shade)
        if C4_photosynthesis:
            photosynthesis_shade_C4_results = calculate_C4_photosynthesis(
                Tf_K=Tf_shaded_K,  # shaded leaf temperature (Tf) [K]
                Ci_μmol_per_mol=Ci_shaded,  # shaded intercellular CO2 concentration (Ci) [umol mol-1]
                APAR_μmolm2s1=APAR_shaded_μmolm2s1,  # shaded absorbed photosynthetically active radiation (APAR) [umol m-2 s-1]
                Vcmax25_μmolm2s1=Vcmax25_shaded  # shaded maximum carboxylation rate at 25C (Vcmax25) [umol m-2 s-1]
            )

            net_assimilation_shade_μmolm2s1 = photosynthesis_shade_C4_results['net_assimilation_C4_μmolm2s1']
        else:
            photosynthesis_shade_C3_results = calculate_C3_photosynthesis(
                Tf_K=Tf_shaded_K,  # shaded leaf temperature (Tf) [K]
                Ci=Ci_shaded,  # shaed intercellular CO2 concentration (Ci) [umol mol-1]
                APAR_μmolm2s1=APAR_shaded_μmolm2s1,  # shaded absorbed photosynthetically active radiation (APAR) [umol m-2 s-1]
                Vcmax25=Vcmax25_shaded,  # shaded maximum carboxylation rate at 25C (Vcmax25) [umol m-2 s-1]
                Ps_Pa=Ps_Pa,  # surface pressure (Ps) [Pa]
                carbon_uptake_efficiency=carbon_uptake_efficiency  # intrinsic quantum efficiency for carbon uptake
            )

            net_assimilation_shade_μmolm2s1 = photosynthesis_shade_C3_results['net_assimilation_C3_μmolm2s1']

        # calculated shaded energy balance
        Rn_shaded_new, LE_shaded_new, H_shaded_new, Tf_K_shaded_new, gs2_shaded_new, Ci_shaded_new = canopy_energy_balance(
            An=net_assimilation_shade_μmolm2s1,  # net assimulation (An) [umol m-2 s-1]
            ASW_Wm2=ASW_shaded_Wm2,  # total absorbed shortwave radiation by shaded canopy (ASW) [umol m-2 s-1]
            ALW_Wm2=ALW_shaded_Wm2,  # total absorbed longwave radiation by shaded canopy (ALW) [umol m-2 s-1]
            Tf_K=Tf_shaded_K,  # shaded leaf temperature (Tf) [K]
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
        Rn_shaded_Wm2 = np.where(np.isnan(Rn_shaded_new), Rn_shaded_Wm2, Rn_shaded_new)
        LE_shaded_Wm2 = np.where(np.isnan(LE_shaded_new), LE_shaded_Wm2, LE_shaded_new)
        H_shaded_Wm2 = np.where(np.isnan(H_shaded_new), H_shaded_Wm2, H_shaded_new)
        Tf_shaded_K = np.where(np.isnan(Tf_K_shaded_new), Tf_shaded_K, Tf_K_shaded_new)
        Ci_shaded = np.where(np.isnan(Ci_shaded_new), Ci_shaded, Ci_shaded_new)

        # calculate soil energy balance
        Rn_soil_new, LE_soil_new, Ts_K_soil_new = soil_energy_balance(
            Ts_K=Ts_K,  # soil temperature in Kelvin
            Ta_K=Ta_K,  # air temperature in Kelvin
            G_Wm2=G_Wm2,  # soil heat flux (G) [W m-2]
            VPD_Pa=VPD_Pa,  # water vapour deficit in Pascal
            RH=RH,  # relative humidity as a fraction
            gamma=gamma,  # psychrometric constant (gamma) [pa K-1]
            Cp=Cp,  # specific heat of air (Cp) [J kg-1 K-1]
            rhoa=rhoa,  # air density (rhoa) [kg m-3]
            desTa=desTa,
            Rs=Rs,
            ASW_soil_Wm2=ASW_soil_Wm2,  # total absorbed shortwave radiation by soil (ASW) [umol m-2 s-1]
            ALW_soil_Wm2=ALW_soil_Wm2,  # total absorbed longwave radiation by soil (ALW) [umol m-2 s-1]
            Ls=Ls,
            epsa=epsa
        )

        # filter in soil energy balance estimates
        # where new estimates are missing, retain the prior estimates
        Rn_soil_Wm2 = np.where(np.isnan(Rn_soil_new), Rn_soil_Wm2, Rn_soil_new)
        LE_soil_Wm2 = np.where(np.isnan(LE_soil_new), LE_soil_Wm2, LE_soil_new)
        Ts_K = np.where(np.isnan(Ts_K_soil_new), Ts_K, Ts_K_soil_new)

        # combine sunlit and shaded foliage temperatures
        Tf_K_new = (((Tf_sunlit_K ** 4) * sunlit_fraction + (Tf_shaded_K ** 4) * (1 - sunlit_fraction)) ** 0.25)
        Tf_K = np.where(np.isnan(Tf_K_new), Tf_K, Tf_K_new)

    # calculate canopy latent heat flux
    LE_canopy_Wm2 = np.clip(LE_sunlit_Wm2 + LE_shaded_Wm2, 0, 1000)

    # calculate latent heat flux
    LE_Wm2 = np.clip(LE_sunlit_Wm2 + LE_shaded_Wm2 + LE_soil_Wm2, 0, 1000)  # [W m-2]

    # calculate gross primary productivity
    GPP = np.clip(net_assimilation_sunlit_μmolm2s1 + net_assimilation_shade_μmolm2s1, 0, GPP_max)  # [umol m-2 s-1]

    # calculate canopy net radiation
    Rn_canopy_Wm2 = np.clip(Rn_sunlit_Wm2 + Rn_shaded_Wm2, 0, None)

    # calculate net radiation
    Rn_Wm2 = np.clip(Rn_sunlit_Wm2 + Rn_shaded_Wm2 + Rn_soil_Wm2, 0, 1000)  # [W m-2]

    return GPP, LE_Wm2, LE_soil_Wm2, LE_canopy_Wm2, Rn_Wm2, Rn_soil_Wm2, Rn_canopy_Wm2
