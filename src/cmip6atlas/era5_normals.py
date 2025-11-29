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
import cdsapi
import os
import xarray as xr

from cmip6atlas.metrics.config import CLIMATE_METRICS


ERA5_START = 1950
ERA5_DFT_START = 1991
ERA5_END = 2024
METRIC_ERA5 = {
    "annual_temp": "2m_temperature",
    "annual_precip": "total_precipitation"
}


def request_era5_data(
    variable: str,
    start_year: int,
    end_year: int,
    output_fname: str,
    use_land: bool = False,
) -> None:
    """
    Makes a request to the ECMWF CDS API and retrieves monthly gridded
    reanalysis data over the requested time period in netCDF format.
    
    Args:
        variable (str): ERA5 variable name
        start_year (int): Start year of the period
        end_year (int): End year of the period
        output_fname (str): Output filename of the ERA5 dataset
        use_land (bool): Flag to request the land-masked 0.1-degree product. Default is False
    """
    # Initialize the CDS API client
    c = cdsapi.Client()

    # Download monthly mean {variable} data over the time range
    dataset = "reanalysis-era5-single-levels-monthly-means"
    if use_land:
        dataset = "reanalysis-era5-land-monthly-means"
    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": variable,
        "year": [str(year) for year in range(start_year, end_year + 1)],
        "month": [str(month).zfill(2) for month in range(1, 13)],
        "time": "00:00",
        "format": "netcdf"
    }
    c.retrieve(dataset, request, output_fname)


def process_era5_normal(
    metric: str,
    start_year: int,
    end_year: int,
    output_dir: str,
    use_land: bool = False,
) -> None:
    """
    Calculate a climate metric for the specified time period based on the
    ERA5 modeled observational record.

    Args:
        metric (str): Name of the metric to calculate
        start_year (int): Start year of the period
        end_year (int): End year of the period
        output_dir (str): Directory to save processed metric files
        use_land (bool): Flag to use the land-masked 0.1-degree product. Default is False
    """
    if metric not in CLIMATE_METRICS:
        raise ValueError(
            f"Unknown metric: {metric}. Available metrics: {list(CLIMATE_METRICS.keys())}"
        )
    
    if start_year < ERA5_START or end_year > ERA5_END or start_year >= end_year:
        raise ValueError(
            f"Invalid time range {start_year}-{end_year}, must be between {ERA5_START}-{ERA5_END}."
        )
    
    print(f"Processing metric '{metric}' from ERA5 record")
    print(f"Years: {start_year}-{end_year}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    monthly_era5_dataset = os.path.join(
        output_dir, f"{metric}_era5_monthly_{start_year}-{end_year}.nc"
    )

    if os.path.exists(monthly_era5_dataset):
        print(f"\nERA5 monthly dataset already exists: {monthly_era5_dataset}")
        print("Skipping cdsapi request...")
    else:
        print("Making request to CDS API...")
        request_era5_data(
            METRIC_ERA5[metric],
            start_year,
            end_year,
            monthly_era5_dataset,
            use_land,
        )

    # Load the ERA5 monthly data with chunking for memory efficiency
    print(f"\nLoading ERA5 monthly data from {monthly_era5_dataset}")
    ds = xr.open_dataset(monthly_era5_dataset, chunks={"valid_time": 12})

    # Load the land mask
    land_mask_path = os.path.join(
        os.path.dirname(__file__), "nex_gddp_cmip6_land_mask.nc"
    )
    print(f"Loading land mask from {land_mask_path}")
    land_mask_ds = xr.open_dataset(land_mask_path)

    # Determine the data variable name based on the metric
    if metric == "annual_temp":
        data_var = "t2m"
    elif metric == "annual_precip":
        data_var = "tp"
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Align ERA5 data to NEX CMIP6 grid and apply mask
    print(f"Aligning {data_var} data to grid used by models...")
    # Rename ERA5 coordinates to match CMIP6 datasets (latitude/longitude -> lat/lon)
    era5_data = ds[data_var].rename({"latitude": "lat", "longitude": "lon"})

    # Handle periodic boundary at prime meridian (0°/360° longitude)
    # Add a wrap-around point at 360° with data from 0° to enable interpolation at boundaries
    print("Handling periodic longitude boundary...")
    era5_data_wrapped = xr.concat(
        [era5_data, era5_data.isel(lon=0).assign_coords(lon=360.0)],
        dim="lon"
    )

    # Interpolate ERA5 data to the NEX CMIP6 grid
    # CMIP6 grid: centered at 0.125, 0.375, etc.
    # ERA5 grid: aligned at 0, 0.25, etc.
    print("Interpolating ERA5 data to model grid resolution...")
    era5_aligned = era5_data_wrapped.interp(
        lat=land_mask_ds["lat"],
        lon=land_mask_ds["lon"],
        method="linear"
    )

    # Apply land mask: set ocean pixels (where land_mask == False) to NaN
    print(f"Applying land mask to {data_var} data...")
    # The land_mask is True (1) for land and False (0) for ocean
    masked_data = era5_aligned.where(land_mask_ds["land_mask"])

    # Convert monthly data to annual metric
    print(f"Converting monthly data to {metric}...")
    if metric == "annual_temp":
        # For temperature: calculate annual mean and convert from Kelvin to Celsius
        # Group by year and calculate mean across months
        annual_data = masked_data.groupby("valid_time.year").mean(dim="valid_time")
        # Convert from Kelvin to Celsius
        annual_data = annual_data - 273.15
        annual_data.attrs["units"] = "°C"
        annual_data.attrs["long_name"] = "Annual Mean Temperature"
    elif metric == "annual_precip":
        # For precipitation: calculate total annual precipitation
        # ERA5 monthly means 'tp' is mean daily precipitation rate in meters per day
        # We need to multiply by days in each month to get monthly totals, then sum

        # Get days in each month
        days_in_month = masked_data.valid_time.dt.days_in_month

        # Convert mean daily rate (m/day) to monthly total (m/month)
        monthly_totals = masked_data * days_in_month

        # Group by year and sum across months to get annual total
        # Use skipna=False to preserve NaN values in masked ocean cells
        annual_data = monthly_totals.groupby("valid_time.year").sum(dim="valid_time", skipna=False)

        # Convert from meters to millimeters
        annual_data = annual_data * 1000.0
        annual_data.attrs["units"] = "mm/yr"
        annual_data.attrs["long_name"] = "Total Annual Precipitation"

    # Create output dataset
    output_ds = xr.Dataset(
        {metric: annual_data},
        attrs={
            "title": f"ERA5 {metric} ({start_year}-{end_year})",
            "source": "ERA5 Reanalysis",
            "description": f"Annual {metric} derived from ERA5 monthly data",
        }
    )

    output_fname = os.path.join(
        output_dir, f"{metric}_era5_{start_year}-{end_year}.nc"
    )
    print(f"Saving processed data to {output_fname}")
    output_ds.to_netcdf(output_fname, compute=True)

    ds.close()
    land_mask_ds.close()
    output_ds.close()

    print(f"\nProcessing complete! Output saved to: {output_fname}")


def cli() -> None:
    """
    Command Line Interface for computing annual metrics from ERA5 monthly reanalysis data.
    """

    parser = argparse.ArgumentParser(
        description="Compile annual metrics from observational record of ERA5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    available_metrics = list(CLIMATE_METRICS.keys())

    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        required=True,
        choices=available_metrics,
        help="Climate metric to calculate",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=ERA5_DFT_START,
        help=f"Start year (between {ERA5_START} and {ERA5_END}), default is {ERA5_DFT_START}",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=ERA5_END,
        help=f"End year (between {ERA5_START} and {ERA5_END}), default is {ERA5_END}",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./climate-normals",
        help="Directory to save processed metric files",
    )
    parser.add_argument(
        "--land",
        action="store_true",
        help="Specifies to use the land-masked 0.1-degree reanalysis product from ERA5 if set",
    )

    args = parser.parse_args()

    process_era5_normal(
        args.metric,
        args.start_year,
        args.end_year,
        args.output_dir,
        args.land,
    )


if __name__ == "__main__":
    cli()