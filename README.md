# CMIP6-Atlas

This code represents a data processing project to convert the NASA Earth Exchange (NEX) Global Daily Downscaled Projections (GDDP) for Coupled Model Intercomparison Project 6 (CMIP6) into derived climate projections at the local administrative level. The [NEX-GDDP-CMIP6](https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6) is a freely available and unique dataset that provides 0.25°/pixel (~25 km/pixel) gridded rasters of several key climate variables from 1950-2100. These rasters were downscaled (increased in resolution) from their source model outputs using statistical techniques. This code is intended to produce several derived metrics from the CMIP6 dataset for use in interactive visualizations, in particular it will derive:

- Annual Mean Temperature, as well as Mean Summer/Winter Temperature
- Annual Mean Precipitation, as well as Mean Summer/Winter Precipitation
- A histogram binning the average number of days where the daily high is greater than a given value (e.g., average number of days with tmax > 35°C)
- A histogram binning the average number of days where the nightly low is less than a given value (e.g., average number of days with tmin < 0°C)
- Growing Degree Days (GDD)
- Average number of days with a Wet Bulb Temperature exceeding human survival (Wet Bulb Days)
- Hot, Dry, Windy Index (HDWI), used as a predictor for fire danger

## Motivation

This project was motivated by a desire to reproduce national-level climate projections such as the [5th National Climate Assessment](https://atlas.globalchange.gov/) for the United States or the [climatedata.ca](https://climatedata.ca/) data dashboard for Canada. In general, people from around the world are interested in accessible, legible climate projection visualizations for their local communities, but existing web projects have often been inadequate at addressing this need in many nations, especially in the Global South. While the World Bank provides a [Climate Change Knowledge Portal](https://climateknowledgeportal.worldbank.org/country-profiles), its reports use outdated CMIP5 projections for many nations, and is generally provided in a format that is not easy to navigate, nor does it take advantage of downscaled model data. Likewise, while NASA provides a [dashboard](https://cmip6.surge.sh/explore/ssp585?map=0%2C0%2C1.04&layers=crossingyear&lState=crossingyear%7C1%7C0) for monthly gridded NEX-GDDP-CMIP6 data, it is represented in terms of CMIP6 variables themselves, which may be challenging for laypeople to understand.

It is my goal, therefore, to process NEX-GDDP-CMIP6 into actionable statistics averaged to a "local administrative level"; that is, counties for the United States, census divisions for Canada, and so on. The outputs from this processing will then be served to a vector mapping client frontend similar to the US-only prototype here (desktop only): [https://jackiepi.xyz/nca_atlas/](https://jackiepi.xyz/nca_atlas/)