#!/usr/bin/env python3

from cmip6atlas.metrics.pipeline import calculate_climate_metric

if __name__ == "__main__":
    metric_name = "annual_precip"
    start_year = 2021
    end_year = 2025
    scenario = "ssp585"
    model = "GFDL-CM4"
    calculate_climate_metric(
        metric_name, start_year, end_year, scenario, model, output_format="netcdf"
    )
