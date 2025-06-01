"""
Copyright (c) 2025 Jacqueline Ryan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import os
from typing import cast

import pandas as pd

from cmip6atlas.metrics.config import CLIMATE_METRICS
from cmip6atlas.metrics.pipeline import calculate_multi_model_metric
from cmip6atlas.region_mask import calculate_global_regions_means
from cmip6atlas.schema import GDDP_CMIP6_SCHEMA


# Default models from Massoud et al., 2023
DEFAULT_MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CanESM5",
    "EC-Earth3",
    "FGOALS-g3",
    "GFDL-ESM4",
    "INM-CM4-8",
    "INM-CM5-0",
    "IPSL-CM6A-LR",
    "MIROC6",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NorESM2-LM",
    "NorESM2-MM",
]

# Model weights from Massoud et al., 2023
MODEL_WEIGHTS = {
    "ACCESS-CM2": 0.0412,
    "ACCESS-ESM1-5": 0.0581,
    "BCC-CSM2-MR": 0.0723,
    "CanESM5": 0.029,
    "EC-Earth3": 0.0498,
    "FGOALS-g3": 0.0716,
    "GFDL-ESM4": 0.0589,
    "INM-CM4-8": 0.0646,
    "INM-CM5-0": 0.0649,
    "IPSL-CM6A-LR": 0.0449,
    "MIROC6": 0.0767,
    "MPI-ESM1-2-HR": 0.0731,
    "MPI-ESM1-2-LR": 0.0755,
    "MRI-ESM2-0": 0.073,
    "NorESM2-LM": 0.0736,
    "NorESM2-MM": 0.0727,
}


def process_metrics_time_series(
    metric: str,
    scenario: str,
    start_year: int,
    end_year: int,
    input_dir: str,
    output_dir: str,
    models: list[str] | None = None,
    country_code: str | None = None,
    bounds_dir: str = "./country-bounds",
) -> None:
    """
    Calculate a climate metric for the specified time period using multiple models,
    then apply region masking using the global_regions.geojson file.

    Args:
        metric: Name of the metric to calculate
        scenario: Climate scenario
        start_year: Start year of the period
        end_year: End year of the period
        input_dir: Directory containing raw climate data
        output_dir: Directory to save processed metric files
        models: List of models to process (defaults to predefined list)
        country_code: ISO 3-letter country code to filter regions (optional)
        bounds_dir: Directory containing the global_regions.geojson file
    """
    if metric not in CLIMATE_METRICS:
        raise ValueError(
            f"Unknown metric: {metric}. Available metrics: {list(CLIMATE_METRICS.keys())}"
        )

    if models is None:
        models = DEFAULT_MODELS

    print(f"Processing metric '{metric}' for scenario '{scenario}'")
    print(f"Years: {start_year}-{end_year}")
    print(f"Models: {models}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    try:
        # Step 1: Calculate multi-model metric (or check if it already exists)
        # First, determine what the output path would be
        expected_netcdf_path = os.path.join(
            output_dir, f"{metric}_multi_model_{scenario}_{start_year}-{end_year}.nc"
        )

        if os.path.exists(expected_netcdf_path):
            print(f"\n✓ Multi-model dataset already exists: {expected_netcdf_path}")
            print("Skipping metric calculation...")
            netcdf_output_path = expected_netcdf_path
        else:
            print("Calculating multi-model metric...")
            netcdf_output_path = calculate_multi_model_metric(
                metric_name=metric,
                start_year=start_year,
                end_year=end_year,
                scenario=scenario,
                models=models,
                base_dir=input_dir,
                output_dir=output_dir,
            )

            if not netcdf_output_path or not os.path.exists(netcdf_output_path):
                print(f"\n✗ No NetCDF output generated")
                return

            print(f"\n✓ Successfully created multi-model dataset: {netcdf_output_path}")

        # Step 2: Apply region masking using global_regions.geojson
        global_regions_path = os.path.join(bounds_dir, "global_regions.geojson")

        # Check if global_regions.geojson exists
        if not os.path.exists(global_regions_path):
            print(f"✗ Global regions file not found at: {global_regions_path}")
            print("Please ensure global_regions.geojson exists in the bounds directory")
            return

        print(f"\nApplying region masking using: {global_regions_path}")
        if country_code:
            print(f"Filtering for country: {country_code}")

        # For multi-model outputs, the variable name is typically the metric name
        variable_name = "mean_" + metric

        # Create output JSON path
        if country_code:
            output_json_path = os.path.join(
                output_dir,
                f"{metric}_{country_code}_{scenario}_{start_year}-{end_year}.json",
            )
        else:
            output_json_path = os.path.join(
                output_dir,
                f"{metric}_global_{scenario}_{start_year}-{end_year}.json",
            )

        print(f"Calculating regional means...")
        success = calculate_global_regions_means(
            global_regions_path=global_regions_path,
            netcdf_path=netcdf_output_path,
            variable_name=variable_name,
            output_json_path=output_json_path,
            model_weights=MODEL_WEIGHTS,
            country=country_code,
        )

        if success:
            print(f"✓ Created regional dataset: {output_json_path}")
        else:
            print(f"✗ Failed to create regional dataset")

    except Exception as e:
        print(f"\n✗ Error processing: {e}")
        raise


def cli() -> None:
    """
    Command Line Interface for computing metrics time series for individual models.
    """
    parser = argparse.ArgumentParser(
        description="Calculate climate metrics for individual models over a time range",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Get available metrics and scenarios from schema
    available_metrics = list(CLIMATE_METRICS.keys())
    schema = GDDP_CMIP6_SCHEMA
    scenarios = cast(list[str], schema["scenarios"])

    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        required=True,
        choices=available_metrics,
        help="Climate metric to calculate",
    )

    parser.add_argument(
        "-s",
        "--scenario",
        type=str,
        required=True,
        choices=scenarios,
        help=f"Climate scenario (historical is only valid for {schema['min_year']}-{schema['historical_end_year']})",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        required=True,
        help=f"Start year (between {schema['min_year']} and {schema['max_year']})",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        required=True,
        help=f"End year (between {schema['min_year']} and {schema['max_year']})",
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="./nex-gddp-cmip6",
        help="Directory containing raw climate data",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./climate-metrics",
        help="Directory to save processed metric files",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help=f"Specific models to process (default: predefined list of {len(DEFAULT_MODELS)} models)",
    )

    parser.add_argument(
        "-c",
        "--country-code",
        type=str,
        help="ISO 3-letter country code for region masking (e.g., USA, PRT, CAN). Admin level will be looked up automatically.",
    )

    parser.add_argument(
        "--bounds-dir",
        type=str,
        default="./country-bounds",
        help="Directory to store/find boundary files",
    )

    args = parser.parse_args()

    process_metrics_time_series(
        args.metric,
        args.scenario,
        args.start_year,
        args.end_year,
        args.input_dir,
        args.output_dir,
        args.models,
        args.country_code,
        args.bounds_dir,
    )


if __name__ == "__main__":
    cli()
