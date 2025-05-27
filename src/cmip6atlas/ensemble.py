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
CMIP6 Ensemble Mean Generation

Processes NEX-GDDP-CMIP6 NetCDF files to create a gridded raster where each cell
represents a mean of the time-averaged model means. The output files can then be
combined in 5-year windowed averages, or used to create 30 year normals.

Input: A directory containing NetCDF files for a cmip6 variable from multiple models
       for a given SSP emissions scenario and year
Output: NetCDF file with mean of model outputs and cross-model standard deviation
        for each grid cell
"""

import calendar
from datetime import datetime
import glob
import numpy as np
import os
import warnings
import xarray as xr

from cmip6atlas.time_utils import time_subset
from cmip6atlas.cli import get_parser

warnings.filterwarnings("ignore")

VAR_NAME_MAP = {
    "hurs": "relative humidity",
    "huss": "specific humidity",
    "pr": "precipitation flux",
    "sfcWind": "daily-mean surface wind speed",
    "tas": "surface air temperature",
    "tasmax": "daily max surface air temperature",
    "tasmin": "daily min surface air temperature",
}


def get_month_boundaries(year: int, month: int) -> tuple[datetime, datetime]:
    """
    Returns a tuple of datetime objects representing the first and last day
    of the specified month in the specified year.
    """
    first_day = datetime(year, month, 1)
    _, num_days = calendar.monthrange(year, month)
    last_day = datetime(year, month, num_days)

    return (first_day, last_day)


def create_cmip6_ensemble_mean(
    input_granules: list[str],
    variable: str,
    scenario: str,
    year: int,
    subset: tuple[datetime, datetime] | None = None,
) -> xr.Dataset:
    """
    Create a gridded raster where each cell contains the mean and standard deviation of
    each model mean for the given variable.

    Args:
        input_granules (list[str]): List of input netCDF files (granules) representing
            model outputs for a single year
        variable (str): CMIP6 variable name
        scenario (str): SSP scenario name
        year (int): Year to process
        subset (tuple[datetime, datetime] | None): temporal subset of the input data as
            a tuple representing a start and end date

    Returns:
        xr.Dataset: a netCDF dataset with the combined mean/std. dev. of the model means
    """
    # Storage for annual/subsetted means from each model
    model_means: list[xr.DataArray] = []
    model_names: list[str] = []

    # Determine period name for metadata
    if subset is not None:
        start_month, end_month = subset[0].month, subset[1].month
        if start_month == end_month:
            period_name = subset[0].strftime("%b").lower()
            period_description = subset[0].strftime("%B")
        else:
            period_name = f"{start_month:02d}_{end_month:02d}"
            period_description = (
                f"subset period {subset[0].strftime('%m-%d')} to "
                f"{subset[1].strftime('%m-%d')}"
            )
    else:
        period_name = "ann"
        period_description = "annual average"

    for nc_file in input_granules:
        # Extract model name from filename
        filename = os.path.basename(nc_file)
        model_name = filename.split("_")[2]
        print(f"Processing model: {model_name}")

        ds = xr.open_dataset(nc_file)

        # Convert from K to C (NEX-GDDP-CMIP6 tas is in K)
        if ds[variable].attrs.get("units") == "K":
            canonical_unit = "C"
            ds[variable] = ds[variable] - 273.15
            ds[variable].attrs["units"] = canonical_unit

        # Apply temporal subset if provided
        if subset is not None:
            # Use the time_utils function to subset the ds, since not all
            # models use the same calendar
            ds_subset = time_subset(ds, subset[0], subset[1])

            # Calculate model mean for the subset period
            model_mean = ds_subset[variable].mean(dim="time")
        else:
            # Calculate annual model mean
            model_mean = ds[variable].mean(dim="time")

        # Store model annual/subset mean
        model_means.append(model_mean)
        model_names.append(model_name)

        ds.close()

    # Combine all model means into a single dataset
    # This creates a new dimension 'model' for the ensemble
    lat = model_means[0].lat
    lon = model_means[0].lon

    # Create a new dataset to store the output distribution; in other words,
    # the values of all the model means concatenated
    distribution_ds = xr.Dataset(
        data_vars={
            f"{variable}_{period_name}_mean": (
                ["model", "lat", "lon"],
                np.stack(model_means),
            ),
        },
        coords={
            "model": (["model"], model_names),
            "lat": (["lat"], lat.values),
            "lon": (["lon"], lon.values),
        },
    )

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
    ensemble_ds["mean"].attrs["units"] = canonical_unit
    ensemble_ds["std"].attrs["units"] = canonical_unit
    ensemble_ds["mean"].attrs[
        "long_name"
    ] = f"CMIP6 mean of {period_description} {VAR_NAME_MAP.get(variable)}"
    ensemble_ds["std"].attrs[
        "long_name"
    ] = f"Cross-model standard deviation of {period_description} {VAR_NAME_MAP.get(variable)}"

    # For each grid cell, compute mean parameters
    print("Generating ensemble dataset...")
    for i in range(len(lat)):
        for j in range(len(lon)):
            # Extract values for this grid cell across all models
            cell_values = distribution_ds[f"{variable}_{period_name}_mean"][
                :, i, j
            ].values

            # Filter out NaN values (e.g., ocean points in some models)
            valid_values = cell_values[~np.isnan(cell_values)]

            if len(valid_values) > 1:
                # Calculate mean parameters
                mean = np.mean(valid_values)
                std = np.std(valid_values)
            else:
                mean = None
                std = None
            # Store in output dataset
            ensemble_ds["mean"][i, j] = mean
            ensemble_ds["std"][i, j] = std

    # Add metadata
    if subset is not None:
        ensemble_ds.attrs["description"] = (
            f"Ensemble parameters for {variable} under {scenario} in {year}, "
            f"{period_description}"
        )
        ensemble_ds.attrs["subset_start"] = subset[0].strftime("%m-%d")
        ensemble_ds.attrs["subset_end"] = subset[1].strftime("%m-%d")
    else:
        ensemble_ds.attrs["description"] = (
            f"Ensemble parameters for {variable} under {scenario} in {year}"
        )

    ensemble_ds.attrs["variable"] = variable
    ensemble_ds.attrs["scenario"] = scenario
    ensemble_ds.attrs["year"] = year
    ensemble_ds.attrs["period"] = period_name
    ensemble_ds.attrs["models_used"] = ", ".join(model_names)

    return ensemble_ds


def ensemble_process(
    input_dir: str,
    output_dir: str,
    variable: str,
    scenario: str,
    year: int,
    month: int | None = None,
    output_file: str | None = None,
) -> None:
    """
    Intermediate function for automating file names before creating
    cmip6 ensemble mean.

    Args:
        input_dir (str): Directory containing pre-downloaded model outputs
        output_dir (str): Directory to store ensemble mean files
        variable (str): CMIP6 variable name
        scenario (str): SSP scenario name
        year (int): Year to process
        month (int): Month to subset model data
        output_file (str | None): Optionally override default file name convention

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Handle subsetting to one month, do this early as it affects the output filename
    if month is not None:
        subset = get_month_boundaries(year, month)
        print(
            f"Subsetting data from {subset[0].strftime('%Y-%m-%d')} to {subset[1].strftime('%Y-%m-%d')}"
        )
        month_str = f"_{subset[0].strftime('%b').lower()}"
    else:
        subset = None
        month_str = ""

    # Determine output file path
    if output_file:
        ensemble_file = output_file
        if not os.path.isabs(ensemble_file):
            output_file = os.path.join(output_dir, output_file)
    else:
        # Default naming pattern
        ensemble_file = os.path.join(
            output_dir, f"{variable}_ensemble_{scenario}_{year}{month_str}.nc"
        )

    # Pattern to match NetCDF files for the specified variable, scenario, and year
    pattern = f"{variable}_day_*_{scenario}_*_{year}*.nc"
    file_pattern = os.path.join(input_dir, pattern)

    # Find all matching granules in the input dir
    granules = glob.glob(file_pattern)

    if not granules:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    print(f"Found {len(granules)} model outputs to process")

    ensemble_ds = create_cmip6_ensemble_mean(granules, variable, scenario, year, subset)

    # Save output
    ensemble_ds.to_netcdf(ensemble_file)
    print(f"Saved ensemble parameters to {ensemble_file}")


def cli() -> None:
    """Command Line Interface for generating an ensemble mean file for a single year."""
    parser = get_parser(
        "Generate ensemble means from locally saved CMIP6 NetCDF model outputs",
        task="ensemble",
    )

    args = parser.parse_args()
    ensemble_process(
        args.input_dir,
        args.output_dir,
        args.variable,
        args.scenario,
        args.year,
        args.month,
        args.output_file,
    )
