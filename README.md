# Breathing Earth System Simulator (BESS) Model Python Implementation

[![CI](https://github.com/JPL-Evapotranspiration-Algorithms/breathing-earth-system-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/JPL-Evapotranspiration-Algorithms/breathing-earth-system-simulator/actions/workflows/ci.yml)

This software package is a Python implementation of the Breathing Earth System Simulator (BESS) model. It was re-implemented in Python by Gregory Halverson at Jet Propulsion Laboratory based on [MATLAB code](https://www.environment.snu.ac.kr/bess-flux) produced by Youngryel Ryu at Seoul University. The BESS model was designed to quantify global gross primary productivity (GPP) and evapotranspiration (ET) using MODIS with a spatial resolution of 1–5 km and a temporal resolution of 8 days. It couples atmospheric and canopy radiative transfer processes with photosynthesis, stomatal conductance, and transpiration models on sunlit and shaded portions of vegetation and soil. An artificial neural network emulator of Hideki Kobayashi's Forest Light Environmental Simulator (FLiES) radiative transfer model is used to estimate incoming solar radiation. This implementation of BESS was designed to process GPP at fine spatial resolution using surface temperature from the ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) mission and normalized difference vegetation index (NDVI) and albedo from the [Spatial Timeseries for Automated high-Resolution multi-Sensor (STARS) data fusion system](https://github.com/STARS-Data-Fusion). The software was developed as part of a research grant by the NASA Research Opportunities in Space and Earth Sciences (ROSES) program. It was designed for use by the ECOSTRESS mission as a precursor for the Surface Biology and Geology (SBG) mission. However, it may also be useful for general remote sensing and GIS projects in Python. This package can be utilized for remote sensing research in Jupyter notebooks and deployed for operations in data processing pipelines. This software is being released according to the SPD-41 open-science requirements of NASA-funded ROSES projects.

Gregory H. Halverson (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
Lead developer<br>
NASA Jet Propulsion Laboratory 329G

Youngryel Ryu (he/him)<br>
[yryu@snu.ac.kr](mailto:yryu@snu.ac.kr)<br>
BESS algorithm inventor<br>
Seoul National University

Hideki Kobayashi (he/him)<br>
[hkoba@jamstec.go.jp](mailto:hkoba@jamstec.go.jp)<br>
FLiES algorithm inventor<br>
Japan Agency for Marine-Earth Science and Technology

Robert Freepartner (he/him)<br>
[robert.freepartner@jpl.nasa.gov](robert.freepartner@jpl.nasa.gov)<br>
MATLAB to python translation<br>
Raytheon

Joshua Fisher (he/him)<br>
[jbfisher@chapman.edu](mailto:jbfisher@chapman.edu)<br>
Concept development and project management<br>
Chapman University

Kerry Cawse-Nicholson (she/her)<br>
[kerry-anne.cawse-nicholson@jpl.nasa.gov](mailto:kerry-anne.cawse-nicholson@jpl.nasa.gov)<br>
Project management<br>
NASA Jet Propulsion Laboratory 329G

Zoe Pierrat (she/her)<br>
[zoe.a.pierrat@jpl.nasa.gov](mailto:zoe.a.pierrat@jpl.nasa.gov)<br>
Algorithm maintenance<br>
NASA Jet Propulsion Laboratory 329G

Claire Villanueva-Weeks (she/her)<br>
[claire.s.villanueva-weeks@jpl.nasa.gov](mailto:claire.s.villanueva-weeks@jpl.nasa.gov)<br>
Code maintenance<br>
NASA Jet Propulsion Laboratory 329G

## Installation

Use the pip package manager to install this package

```
pip install breathing-earth-system-simulator
```

## References

The following scientific references provide detailed information about the BESS model and its underlying algorithms:

1. Ryu, Y., Baldocchi, D. D., Kobayashi, H., van Ingen, C., Li, J., Black, T. A., ... & Ueyama, M. (2011). Integration of MODIS land and atmosphere products with a coupled-process model to estimate gross primary productivity and evapotranspiration from 1 km to global scales. *Remote Sensing of Environment, 115*(8), 1865-1874. https://doi.org/10.1016/j.rse.2011.03.009

2. Kobayashi, H., Ryu, Y., Baldocchi, D. D., Welles, J. M., & Norman, J. M. (2012). On the correct estimation of gap fraction: How to remove scattered radiation in gap fraction measurements? *Agricultural and Forest Meteorology, 160*, 14-25. https://doi.org/10.1016/j.agrformet.2012.03.001

3. Fisher, J. B., Lee, B., Purdy, A. J., Halverson, G. H., Dohlen, M. B., & Tu, K. P. (2020). ECOSTRESS: NASA's next generation mission to measure evapotranspiration from the International Space Station. *Water Resources Research, 56*(4), e2019WR026058. https://doi.org/10.1029/2019WR026058

4. Ryu, Y., Jiang, C., Kobayashi, H., & Detto, M. (2018). Modis-derived global land products of shortwave radiation and diffuse and total photosynthetically active radiation at 5 km resolution from 2000. *Remote Sensing of Environment, 204*, 812-825. https://doi.org/10.1016/j.rse.2017.09.021

5. Kobayashi, H., & Iwabuchi, H. (2008). A coupled 1-D atmosphere and canopy radiative transfer model for an atmosphere with a nonlambertian surface. *Journal of Quantitative Spectroscopy and Radiative Transfer, 109*(17-18), 2955-2970. https://doi.org/10.1016/j.jqsrt.2008.07.008
