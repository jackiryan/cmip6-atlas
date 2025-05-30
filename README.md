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

## Installation and Usage (dev-only for now)

Presently, running this code requires cloning this repo, pip installing the module, and running the entrypoint scripts but when the entire workflow is finalized, the process of running the code will be simplified. For now, begin by cloning this repo then installing (from the cmip6-atlas directory that is created):

```bash
pip install -e ".[dev]"
```

The current most complete entrypoint script is `calculate_metrics_time_series`, which downloads CMIP6 granules, computes a climate metric for each year within a specified time range, performs spatial aggregation for a given country (or all countries if not specified), and computes a bayesian model average using weights from Massoud et al., 2023. The tool has the following usage:

```
usage: calculate_metrics_time_series [-h] -m {annual_temp,annual_precip,winter_mean_temp,growing_season_length,ndays_gt_35c} -s {historical,ssp126,ssp245,ssp370,ssp585}
                                     --start-year START_YEAR --end-year END_YEAR [-i INPUT_DIR] [-o OUTPUT_DIR] [--models MODELS [MODELS ...]] [-c COUNTRY_CODE]
                                     [--bounds-dir BOUNDS_DIR]

Calculate climate metrics for individual models over a time range

options:
  -h, --help            show this help message and exit
  -m, --metric {annual_temp,annual_precip,winter_mean_temp,growing_season_length,ndays_gt_35c}
                        Climate metric to calculate (default: None)
  -s, --scenario {historical,ssp126,ssp245,ssp370,ssp585}
                        Climate scenario (historical is only valid for 1950-2014) (default: None)
  --start-year START_YEAR
                        Start year (between 1950 and 2100) (default: None)
  --end-year END_YEAR   End year (between 1950 and 2100) (default: None)
  -i, --input-dir INPUT_DIR
                        Directory containing raw climate data (default: ./nex-gddp-cmip6)
  -o, --output-dir OUTPUT_DIR
                        Directory to save processed metric files (default: ./climate-metrics)
  --models MODELS [MODELS ...]
                        Specific models to process (default: predefined list of 16 models) (default: None)
  -c, --country-code COUNTRY_CODE
                        ISO 3-letter country code for region masking (e.g., USA, PRT, CAN). Admin level will be looked up automatically. (default: None)
  --bounds-dir BOUNDS_DIR
                        Directory to store/find boundary files (default: ./country-bounds)
```

Below is an example command for testing that will output a geojson file of annual average temperature for the United States at a county level from 2021-2025 for the SSP5-8.5 emissions scenario. The input and output directories (`-i`, `-o` and `--bounds-dir`) will be created below the directory you run the script if they do not exist. You can provide different directories if you would like to store data in a specific place, as the process will download several GB of data worth of granules even for a short time window.

```bash
calculate_metrics_time_series -m annual_temp -s ssp585 --start-year 2021 --end-year 2025 -i nex-gddp-data -o climate-metrics -c USA --bounds-dir country-bounds
```
