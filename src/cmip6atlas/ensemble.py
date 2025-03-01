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

"""
CMIP6 Ensemble PDF Generator

Processes NEX-GDDP-CMIP6 NetCDF files to create a gridded raster where each cell
represents a Probability Distribution Function (PDF) of annual mean temperature
values across all models.

Input: A directory containing NetCDF files for tas variable from multiple models
       for SSP5-8.5 scenario in year 2025
Output: NetCDF file with PDF parameters for each grid cell
"""

import argparse
import glob
import numpy as np
import os
from scipy import stats
import warnings
import xarray as xr

warnings.filterwarnings("ignore")


def create_cmip6_ensemble_nc(
    input_dir: str, output_file: str, variable: str, scenario: str, year: int
) -> xr.Dataset:
    """
    Create a gridded raster where each cell contains PDF parameters of annual mean
    temperature values across all models.

    Arguments:
        input_dir (str): Directory containing the NetCDF files
        output_file (str): Path to save the output NetCDF file
        variable (str): Variable name
        scenario (str): Scenario name (default: 'ssp585')
        year (int): Year to process

    Returns:
        xr.Dataset: a netCDF dataset with the combined probability distribution of
            the model means.
    """
    # Pattern to match NetCDF files for the specified variable, scenario, and year
    pattern = f"{variable}_day_*_{scenario}_*_{year}*.nc"
    file_pattern = os.path.join(input_dir, pattern)

    # Find all matching files
    nc_files = glob.glob(file_pattern)

    if not nc_files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    print(f"Found {len(nc_files)} model files to process")

    # Storage for annual means from each model
    model_means = []
    model_names = []

    for nc_file in nc_files:
        # Extract model name from filename
        filename = os.path.basename(nc_file)
        model_name = filename.split("_")[2]
        print(f"Processing model: {model_name}")

        # Open the NetCDF file
        ds = xr.open_dataset(nc_file)

        # Convert from K to C if needed (NEX-GDDP-CMIP6 tas is in K)
        if ds[variable].attrs.get("units") == "K":
            ds[variable] = ds[variable] - 273.15
            ds[variable].attrs["units"] = "C"

        # Calculate annual mean
        annual_mean = ds[variable].mean(dim="time")

        # Store model annual mean
        model_means.append(annual_mean)
        model_names.append(model_name)

        ds.close()

    # Combine all model means into a single dataset
    # This creates a new dimension 'model' for the ensemble
    lat = model_means[0].lat
    lon = model_means[0].lon

    # Create a new dataset to store models data
    distribution_ds = xr.Dataset(
        data_vars={
            f"{variable}_annual_mean": (["model", "lat", "lon"], np.stack(model_means)),
        },
        coords={
            "model": (["model"], model_names),
            "lat": (["lat"], lat.values),
            "lon": (["lon"], lon.values),
        },
    )
    # For debugging, can save output distribution to file
    # Remember to comment out the other file saving operation
    # distribution_ds.to_netcdf(output_file)

    # Create output dataset for PDF parameters
    ensemble_ds = xr.Dataset(
        data_vars={
            "mean": (["lat", "lon"], np.zeros_like(lat * lon)),
            "std": (["lat", "lon"], np.zeros_like(lat * lon)),
        },
        coords={
            "lat": (["lat"], lat.values),
            "lon": (["lon"], lon.values),
        },
    )

    # Add units, currently only supporting temperature
    ensemble_ds["mean"].attrs["units"] = "C"
    ensemble_ds["std"].attrs["units"] = "C"
    ensemble_ds["mean"].attrs[
        "long_name"
    ] = f"Mean of {variable} annual averages across models"
    ensemble_ds["std"].attrs[
        "long_name"
    ] = f"Standard deviation of {variable} annual averages across models"

    # For each grid cell, compute PDF parameters
    for i in range(len(lat)):
        for j in range(len(lon)):
            # Extract values for this grid cell across all models
            cell_values = distribution_ds[f"{variable}_annual_mean"][:, i, j].values

            # Filter out NaN values (e.g., ocean points in some models)
            valid_values = cell_values[~np.isnan(cell_values)]

            if len(valid_values) > 1:  # Need at least 2 values for stats
                # Calculate PDF parameters
                mean = np.mean(valid_values)
                std = np.std(valid_values)

                # Store in output dataset
                ensemble_ds["mean"][i, j] = mean
                ensemble_ds["std"][i, j] = std

    # Add metadata
    ensemble_ds.attrs["description"] = (
        f"Ensemble parameters for {variable} under {scenario} in {year}"
    )
    ensemble_ds.attrs["variable"] = variable
    ensemble_ds.attrs["scenario"] = scenario
    ensemble_ds.attrs["year"] = year
    ensemble_ds.attrs["models_used"] = ", ".join(model_names)

    # Save output
    ensemble_ds.to_netcdf(output_file)
    print(f"Saved ensemble parameters to {output_file}")

    return ensemble_ds


def parse_arguments() -> argparse.Namespace:
    """Define and parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate ensemble PDFs from CMIP6 NetCDF files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        default="./nex-gddp-cmip6/",
        help="Directory containing NetCDF files",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        default="./ensemble-means",
        help="Directory for output files",
    )

    parser.add_argument(
        "-v",
        "--variable",
        help="Climate variable to process (e.g., tas, pr, tasmax)",
    )

    parser.add_argument(
        "-s",
        "--scenario",
        help="Climate scenario (e.g., ssp585, ssp245, historical)",
    )

    parser.add_argument("-y", "--year", type=int, help="Year to process")

    parser.add_argument(
        "--output-file",
        help="Specify custom output filename (overrides default naming pattern)",
    )

    return parser.parse_args()


def main() -> None:
    """Main function to process command line arguments and run the processing"""
    # Parse command line arguments
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine output file path
    if args.output_file:
        output_file = args.output_file
        if not os.path.isabs(output_file):
            output_file = os.path.join(args.output_dir, output_file)
    else:
        # Default naming pattern
        output_file = os.path.join(
            args.output_dir, f"{args.variable}_ensemble_{args.scenario}_{args.year}.nc"
        )

    # Process files
    try:
        create_cmip6_ensemble_nc(
            input_dir=args.input_dir,
            output_file=output_file,
            variable=args.variable,
            scenario=args.scenario,
            year=args.year,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
