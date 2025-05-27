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

import geopandas as gpd  # type: ignore [import-untyped]

from cmip6atlas.metrics.config import CLIMATE_METRICS
from cmip6atlas.metrics.pipeline import calculate_multi_model_metric
from cmip6atlas.region_mask import calculate_geojson_feature_means
from cmip6atlas.download_gadm import (
    get_default_csv_path,
    download_single_file,
    URL_TEMPLATE,
)
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


def lookup_admin_level(country_code: str) -> int:
    """
    Look up the administrative level for a country from the GADM countries CSV.

    Args:
        country_code: ISO 3-letter country code

    Returns:
        Administrative level from the CSV, or 1 as default
    """
    try:
        csv_path = get_default_csv_path()
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Look for matching country code (case insensitive)
            match = df[df["country_code"].str.upper() == country_code.upper()]
            if not match.empty:
                return int(match.iloc[0]["admin_level"])
    except Exception as e:
        print(f"Warning: Could not look up admin level for {country_code}: {e}")

    # Default to admin level 1
    print(f"Using default admin level 1 for {country_code}")
    return 1


def get_all_countries_from_csv() -> list[tuple[str, int]]:
    """
    Get all countries and their admin levels from the GADM countries CSV.

    Returns:
        List of tuples (country_code, admin_level)
    """
    try:
        csv_path = get_default_csv_path()
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Return list of (country_code, admin_level) tuples
            return [
                (row["country_code"], int(row["admin_level"]))
                for _, row in df.iterrows()
            ]
    except Exception as e:
        print(f"Warning: Could not read countries from CSV: {e}")

    return []


def combine_geojson_files(geojson_paths: list[str], output_path: str) -> bool:
    """
    Combine multiple GeoJSON files into a single file.

    Args:
        geojson_paths: List of paths to individual GeoJSON files
        output_path: Path for the combined output file

    Returns:
        True if successful, False otherwise
    """
    try:
        combined_gdfs = []

        for geojson_path in geojson_paths:
            if os.path.exists(geojson_path):
                gdf = gpd.read_file(geojson_path)
                # Add country identifier from filename
                country_code = os.path.basename(geojson_path).split("_")[1]
                gdf["country_code"] = country_code
                combined_gdfs.append(gdf)
            else:
                print(f"Warning: {geojson_path} not found")

        if not combined_gdfs:
            print("No valid GeoJSON files to combine")
            return False

        # Combine all GeoDataFrames
        combined_gdf = gpd.pd.concat(combined_gdfs, ignore_index=True)

        # Save combined file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_gdf.to_file(output_path, driver="GeoJSON")

        print(
            f"Combined {len(combined_gdfs)} datasets with {len(combined_gdf)} total features"
        )
        return True

    except Exception as e:
        print(f"Error combining GeoJSON files: {e}")
        return False


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
    then optionally apply region masking to administrative boundaries.

    Args:
        metric: Name of the metric to calculate
        scenario: Climate scenario
        start_year: Start year of the period
        end_year: End year of the period
        input_dir: Directory containing raw climate data
        output_dir: Directory to save processed metric files
        models: List of models to process (defaults to predefined list)
        country_code: ISO 3-letter country code for region masking (optional)
        bounds_dir: Directory to store/find boundary files
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

        # Step 2: Apply region masking
        if country_code:
            # Process single country
            print(f"\nApplying region masking for country: {country_code}")
            countries_to_process = [(country_code, lookup_admin_level(country_code))]
        else:
            # Process all countries from CSV
            print(f"\nApplying region masking for all countries...")
            countries_to_process = get_all_countries_from_csv()

        if not countries_to_process:
            print("No countries to process for region masking")
            return

        # Download boundaries and process each country
        os.makedirs(bounds_dir, exist_ok=True)
        successful_geojsons = []

        for country_code_iter, admin_level in countries_to_process:
            print(f"\nProcessing {country_code_iter} (admin level {admin_level})...")

            geojson_path = os.path.join(
                bounds_dir, f"gadm41_{country_code_iter}_{admin_level}.json"
            )

            # Download boundaries if they don't exist
            if not os.path.exists(geojson_path):
                print(f"  Downloading boundaries...")
                url = URL_TEMPLATE.format(country_code_iter.upper(), admin_level)
                success, _ = download_single_file(
                    url, geojson_path, skip_existing_flag=False
                )
                if not success:
                    print(f"  ✗ Failed to download boundaries for {country_code_iter}")
                    continue

            if os.path.exists(geojson_path):
                # For multi-model outputs, the variable name is typically the metric name
                variable_name = "mean_" + metric

                # Create output geojson path for this country
                country_geojson_path = os.path.join(
                    output_dir,
                    f"{metric}_{country_code_iter}_{admin_level}_{scenario}_{start_year}-{end_year}.geojson",
                )

                print(f"  Calculating regional means...")
                success = calculate_geojson_feature_means(
                    geojson_path=geojson_path,
                    netcdf_path=netcdf_output_path,
                    variable_name=variable_name,
                    output_geojson_path=country_geojson_path,
                    model_weights=MODEL_WEIGHTS,
                )

                if success:
                    print(f"  ✓ Created: {country_geojson_path}")
                    successful_geojsons.append(country_geojson_path)
                else:
                    print(
                        f"  ✗ Failed to create regional dataset for {country_code_iter}"
                    )
            else:
                print(f"  ✗ Could not find boundaries for {country_code_iter}")

        # Step 3: Combine all individual GeoJSON files into one
        if len(successful_geojsons) > 1:
            print(f"\nCombining {len(successful_geojsons)} country datasets...")
            combined_geojson_path = os.path.join(
                output_dir,
                f"{metric}_global_{scenario}_{start_year}-{end_year}.geojson",
            )

            success = combine_geojson_files(successful_geojsons, combined_geojson_path)
            if success:
                print(f"✓ Combined dataset saved to: {combined_geojson_path}")
            else:
                print(f"✗ Failed to combine datasets")
        elif len(successful_geojsons) == 1:
            print(f"✓ Single country dataset: {successful_geojsons[0]}")
        else:
            print("✗ No successful regional datasets created")

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
