#!/usr/bin/env python3

from cmip6atlas.metrics.pipeline import calculate_multi_model_metric

if __name__ == "__main__":
    metric_name = "ndays_gt_35c"
    start_year = 2021
    end_year = 2025
    scenario = "ssp585"
    base_dir = "./nex-gddp-data"
    output_dir = "./climate-metrics"
    calculate_multi_model_metric(
        metric_name,
        start_year,
        end_year,
        scenario,
        base_dir=base_dir,
        output_dir=output_dir,
    )
