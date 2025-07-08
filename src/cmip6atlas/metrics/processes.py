import xarray as xr
import numpy as np
from typing import cast
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

from cmip6atlas.metrics.definitions import SeasonDefinition
from cmip6atlas.metrics.units import process_variable


def _process_single_year_temp(
    fname: str, variable: str, season: SeasonDefinition
) -> tuple[int, xr.DataArray]:
    """Process a single year's temperature data for mean temperature calculation."""
    with xr.open_dataset(fname) as ds:
        year: int = ds.time.dt.year.values[0].item()
        season_ds = extract_seasonal_data(ds, variable, season, year)
        season_ds_c = process_variable(season_ds, variable)
        season_mean = season_ds_c[variable].mean(dim=["time"])
        return year, season_mean


def _process_single_year_precip(
    fname: str, variable: str, season: SeasonDefinition, max_flux: float
) -> tuple[int, xr.DataArray]:
    """Process a single year's precipitation data for total precipitation calculation."""
    with xr.open_dataset(fname) as ds:
        year: int = ds.time.dt.year.values[0].item()

        # Apply precipitation cap
        capped_ds = ds.copy()
        capped_ds[variable] = ds[variable].clip(max=max_flux)

        season_ds = extract_seasonal_data(capped_ds, variable, season, year)
        season_ds_mm = process_variable(season_ds, variable)

        season_sums = season_ds_mm[variable].sum(dim="time", skipna=True)
        ocean_mask = season_ds_mm[variable].isnull().all(dim="time")
        season_total = season_sums.where(~ocean_mask, np.nan)

        return year, season_total


def _process_single_year_threshold(
    fname: str, variable: str, threshold: float, operation: str
) -> tuple[int, xr.DataArray]:
    """Process a single year's data for threshold days calculation."""
    with xr.open_dataset(fname) as ds:
        year: int = ds.time.dt.year.values[0].item()
        ds_c = process_variable(ds, variable)

        # Apply threshold operation
        if operation == "gt":
            threshold_met = ds_c[variable] > threshold
        elif operation == "lt":
            threshold_met = ds_c[variable] < threshold
        elif operation == "ge":
            threshold_met = ds_c[variable] >= threshold
        elif operation == "le":
            threshold_met = ds_c[variable] <= threshold
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        days_count = threshold_met.sum(dim="time", skipna=True)
        ocean_mask = ds_c[variable].isnull().all(dim="time")
        days_total = days_count.where(~ocean_mask, np.nan)

        return year, days_total


def _process_single_year_growing_season(
    fname: str, variable: str, base_temp: float, min_length: int
) -> tuple[int, xr.DataArray]:
    """Process a single year's data for growing season length calculation."""
    with xr.open_dataset(fname) as ds:
        year: int = ds.time.dt.year.values[0].item()
        ds_c = process_variable(ds, variable)

        above_base = ds_c[variable] > base_temp
        growing_season_length = xr.zeros_like(ds_c[variable].isel(time=0))

        # Process each lat/lon grid cell
        for lat_idx in range(len(ds_c.lat)):
            for lon_idx in range(len(ds_c.lon)):
                cell_series = above_base.isel(lat=lat_idx, lon=lon_idx).values

                # Find runs of consecutive days
                runs = np.diff(np.hstack(([0], cell_series, [0])))
                run_starts = np.where(runs == 1)[0]
                run_ends = np.where(runs == -1)[0]
                run_lengths = run_ends - run_starts

                valid_runs = run_lengths[run_lengths >= min_length]

                if len(valid_runs) > 0:
                    growing_season_length[lat_idx, lon_idx] = sum(valid_runs)

        ocean_mask = ds_c[variable].isnull().all(dim="time")
        season_total = growing_season_length.where(~ocean_mask, np.nan)

        return year, season_total


def extract_seasonal_data(
    ds: xr.Dataset,
    variable: str,
    season_def: SeasonDefinition,
    year: int,
) -> xr.Dataset:
    """
    Extract seasonal data from a dataset, handling hemisphere differences.

    Args:
        ds: Input dataset
        variable: Climate variable name
        season_def: Season definition
        year: Year to extract

    Returns:
        Dataset with only the seasonal data
    """
    # Get month information
    ds = ds.assign_coords(month=ds.time.dt.month)

    # Create hemisphere masks
    lat = ds.lat.values
    hemisphere_info = season_def.get_months_for_hemisphere(lat)
    nh_mask = hemisphere_info["nh_mask"]
    sh_mask = hemisphere_info["sh_mask"]
    nh_months: list[int] = cast(list[int], hemisphere_info["nh_months"])
    sh_months: list[int] = cast(list[int], hemisphere_info["sh_months"])

    # Handle year boundaries
    if season_def.spans_year_boundary:
        # For seasons that span year boundary (like DJF), we need to handle them specially

        # Northern Hemisphere
        if any(m > 9 for m in nh_months):  # Contains months from previous year
            nh_prev_year_months = [m for m in nh_months if m > 9]
            nh_curr_year_months = [m for m in nh_months if m <= 9]

            # Get data from previous year's months
            ds_prev = ds.sel(
                time=(
                    (ds.time.dt.year == year - 1)
                    & ds.time.dt.month.isin(nh_prev_year_months)
                )
            )
            # Get data from current year's months
            ds_curr = ds.sel(
                time=(
                    (ds.time.dt.year == year)
                    & ds.time.dt.month.isin(nh_curr_year_months)
                )
            )

            # Combine the datasets
            ds_nh = xr.concat([ds_prev, ds_curr], dim="time")
        else:
            ds_nh = ds.sel(
                time=((ds.time.dt.year == year) & ds.time.dt.month.isin(nh_months))
            )

        # Southern Hemisphere
        if any(m > 9 for m in sh_months):  # Contains months from previous year
            sh_prev_year_months = [m for m in sh_months if m > 9]
            sh_curr_year_months = [m for m in sh_months if m <= 9]

            # Get data from previous year's months
            ds_prev = ds.sel(
                time=(
                    (ds.time.dt.year == year - 1)
                    & ds.time.dt.month.isin(sh_prev_year_months)
                )
            )
            # Get data from current year's months
            ds_curr = ds.sel(
                time=(
                    (ds.time.dt.year == year)
                    & ds.time.dt.month.isin(sh_curr_year_months)
                )
            )

            # Combine the datasets
            ds_sh = xr.concat([ds_prev, ds_curr], dim="time")
        else:
            ds_sh = ds.sel(
                time=((ds.time.dt.year == year) & ds.time.dt.month.isin(sh_months))
            )
    else:
        # For seasons that don't cross year boundary
        ds_nh = ds.sel(
            time=((ds.time.dt.year == year) & ds.time.dt.month.isin(nh_months))
        )
        ds_sh = ds.sel(
            time=((ds.time.dt.year == year) & ds.time.dt.month.isin(sh_months))
        )

    # Apply hemisphere masks to variable data
    # We need to create a new variable with the correct seasonal data by hemisphere
    all_data = ds[variable].copy()

    # Create a mask to select the right data for each latitude
    seasonal_mask = xr.zeros_like(all_data, dtype=bool)

    # Apply Northern Hemisphere mask and months
    times_nh = ds_nh.time
    for t in times_nh:
        # Find the index of this time in the original dataset
        idx = np.where(ds.time.values == t.values)[0]
        if len(idx) > 0:
            # For points in NH, set mask to True for this time
            seasonal_mask[idx[0], nh_mask, :] = True

    # Apply Southern Hemisphere mask and months
    times_sh = ds_sh.time
    for t in times_sh:
        # Find the index of this time in the original dataset
        idx = np.where(ds.time.values == t.values)[0]
        if len(idx) > 0:
            # For points in SH, set mask to True for this time
            seasonal_mask[idx[0], sh_mask, :] = True

    # Create a new dataset with only the seasonal data
    seasonal_data = all_data.where(seasonal_mask, drop=True)

    # Create a new dataset with the seasonal data
    result = xr.Dataset(
        data_vars={variable: seasonal_data},
        coords={"time": seasonal_data.time, "lat": ds.lat, "lon": ds.lon},
    )

    # Add metadata
    result.attrs["season"] = season_def.name
    result.attrs["year"] = year

    return result


def calculate_mean_temp(
    datasets: dict[str, list[str]], season: SeasonDefinition, **kwargs
) -> xr.Dataset:
    """
    Calculate seasonal temperature normal, considering hemisphere differences.
    Uses parallel processing to safely handle NetCDF-4 files.

    Args:
        datasets: Dictionary mapping variables to lists of annual dataset filenames
        season: Season definition
        **kwargs: Additional parameters

    Returns:
        Dataset with seasonal temperature normals
    """
    variable = "tas"  # Temperature variable
    metric_name = f"mean_{season.name}_temp"
    temp_datasets = datasets[variable]

    # Use parallel processing to handle each year
    max_workers = 6 # min(len(temp_datasets), mp.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_fname = {
            executor.submit(_process_single_year_temp, fname, variable, season): fname
            for fname in temp_datasets
        }

        # Collect results
        seasonal_data: list[xr.DataArray] = []
        years: list[int] = []

        for future in as_completed(future_to_fname):
            try:
                year, season_mean = future.result()
                years.append(year)
                seasonal_data.append(season_mean)
            except Exception as exc:
                fname = future_to_fname[future]
                print(f"File {fname} generated an exception: {exc}")
                raise

    # Sort by year to maintain consistent ordering
    sorted_pairs = sorted(zip(years, seasonal_data), key=lambda x: x[0])
    sorted_years, sorted_data = zip(*sorted_pairs)
    years = list(sorted_years)
    seasonal_data = list(sorted_data)

    # Create result dataset
    result = xr.Dataset()
    result = result.assign_coords(
        {
            "lat": (["lat"], seasonal_data[0].lat.values),
            "lon": (["lon"], seasonal_data[0].lon.values),
            "year": (["year"], np.array(years, dtype=np.int32)),
        }
    )

    # Add the data variable with year dimension
    result[metric_name] = (
        ["year", "lat", "lon"],
        np.stack([data.values for data in seasonal_data]),
    )

    # Add metadata
    result[metric_name].attrs["units"] = "°C"
    result[metric_name].attrs[
        "long_name"
    ] = f"Average {season.name.capitalize()} Temperature"
    result.attrs["season"] = season.name

    return result


def calculate_total_precip(
    datasets: dict[str, list[str]], season: SeasonDefinition, **kwargs
) -> xr.Dataset:
    """
    Calculate seasonal precipitation total, considering hemisphere differences.
    Uses parallel processing to safely handle NetCDF-4 files.

    Args:
        datasets: Dictionary mapping variables to lists of annual dataset filenames
        season: Season definition
        **kwargs: Additional parameters

    Returns:
        Dataset with seasonal precipitation totals
    """
    variable = "pr"  # Precipitation variable
    metric_name = f"total_{season.name}_precip"
    # Sometimes the model has a bad pixel that should be excluded
    # Tropical cyclones are not well modeled in CMIP anyways, so
    # clip any flux over 0.007 kg/m2/s to 0.007
    max_physical_precip_flux = 0.007  # kg m^-2 s^-1
    precip_datasets = datasets[variable]

    # Use parallel processing to handle each year
    max_workers = 6 # min(len(precip_datasets), mp.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_fname = {
            executor.submit(
                _process_single_year_precip,
                fname,
                variable,
                season,
                max_physical_precip_flux,
            ): fname
            for fname in precip_datasets
        }

        # Collect results
        seasonal_data: list[xr.DataArray] = []
        years: list[int] = []

        for future in as_completed(future_to_fname):
            try:
                year, season_total = future.result()
                years.append(year)
                seasonal_data.append(season_total)
            except Exception as exc:
                fname = future_to_fname[future]
                print(f"File {fname} generated an exception: {exc}")
                raise

    # Sort by year to maintain consistent ordering
    sorted_pairs = sorted(zip(years, seasonal_data), key=lambda x: x[0])
    sorted_years, sorted_data = zip(*sorted_pairs)
    years = list(sorted_years)
    seasonal_data = list(sorted_data)

    # Create result dataset
    result = xr.Dataset()
    result = result.assign_coords(
        {
            "lat": (["lat"], seasonal_data[0].lat.values),
            "lon": (["lon"], seasonal_data[0].lon.values),
            "year": (["year"], np.array(years, dtype=np.int32)),
        }
    )

    # Add the data variable with year dimension
    result[metric_name] = (
        ["year", "lat", "lon"],
        np.stack([data.values for data in seasonal_data]),
    )

    # Add metadata
    result[metric_name].attrs["units"] = "mm/yr"
    result[metric_name].attrs[
        "long_name"
    ] = f"{season.name.capitalize()} Precipitation Total"
    result.attrs["season"] = season.name

    return result


def calculate_threshold_days(
    datasets: dict[str, list[str]],
    threshold: float,
    operation: str = "gt",  # 'gt' for greater than, 'lt' for less than, etc.
    **kwargs,
) -> xr.Dataset:
    """
    Calculate the average number of days per year that meet a threshold condition.
    Uses parallel processing to safely handle NetCDF-4 files.

    Args:
        datasets: Dictionary mapping variables to lists of annual dataset filenames
        threshold: Threshold value to compare against
        operation: Comparison operation ('gt', 'lt', 'ge', 'le')
        **kwargs: Additional parameters

    Returns:
        Dataset with average days per year meeting the threshold

    Raises:
        ValueError: if operation is not supported.
    """
    # Get the primary variable (first one in the dataset)
    variable = list(datasets.keys())[0]
    variable_ezname = "Max Temp." if variable == "tasmax" else "Min Temp."
    variable_datasets = datasets[variable]
    metric_name = f"ndays_{operation}_{int(threshold)}c"

    # Use parallel processing to handle each year
    max_workers = min(len(variable_datasets), mp.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_fname = {
            executor.submit(
                _process_single_year_threshold, fname, variable, threshold, operation
            ): fname
            for fname in variable_datasets
        }

        # Collect results
        annual_threshold_days: list[xr.DataArray] = []
        years: list[int] = []

        for future in as_completed(future_to_fname):
            try:
                year, days_total = future.result()
                years.append(year)
                annual_threshold_days.append(days_total)
            except Exception as exc:
                fname = future_to_fname[future]
                print(f"File {fname} generated an exception: {exc}")
                raise

    # Sort by year to maintain consistent ordering
    sorted_pairs = sorted(zip(years, annual_threshold_days), key=lambda x: x[0])
    sorted_years, sorted_data = zip(*sorted_pairs)
    years = list(sorted_years)
    annual_threshold_days = list(sorted_data)

    # Create result dataset
    result = xr.Dataset()
    result = result.assign_coords(
        {
            "lat": (["lat"], annual_threshold_days[0].lat.values),
            "lon": (["lon"], annual_threshold_days[0].lon.values),
            "year": (["year"], np.array(years, dtype=np.int32)),
        }
    )
    result[metric_name] = (
        ["year", "lat", "lon"],
        np.stack([data.values for data in annual_threshold_days]),
    )

    # Add metadata
    op_map = {"gt": ">", "lt": "<", "ge": "≥", "le": "≤"}
    op_symbol = op_map.get(operation, operation)
    result[metric_name].attrs["units"] = "days/year"
    result[metric_name].attrs[
        "long_name"
    ] = f"Days with {variable_ezname} {op_symbol} {threshold}°C"
    result[metric_name].attrs["threshold"] = threshold
    result[metric_name].attrs["operation"] = operation

    return result


def calculate_growing_season_length(
    datasets: dict[str, list[str]],
    base_temp: float = 5.0,
    min_length: int = 5,  # Minimum consecutive days to count as growing season
    **kwargs,
) -> xr.Dataset:
    """
    Calculate average growing season length (consecutive days above base temperature).
    Uses parallel processing to safely handle NetCDF-4 files.

    Args:
        datasets: Dictionary mapping variables to lists of annual dataset filenames
        base_temp: Base temperature defining growing season (typically 5°C or 10°C)
        min_length: Minimum consecutive days to count as growing season
        **kwargs: Additional parameters

    Returns:
        Dataset with average growing season length
    """
    variable = "tas"  # Temperature variable
    metric_name = "growing_season_length"
    temp_datasets = datasets[variable]

    # Use parallel processing to handle each year
    max_workers = min(len(temp_datasets), mp.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_fname = {
            executor.submit(
                _process_single_year_growing_season,
                fname,
                variable,
                base_temp,
                min_length,
            ): fname
            for fname in temp_datasets
        }

        # Collect results
        annual_growing_seasons: list[xr.DataArray] = []
        years: list[int] = []

        for future in as_completed(future_to_fname):
            try:
                year, season_total = future.result()
                years.append(year)
                annual_growing_seasons.append(season_total)
            except Exception as exc:
                fname = future_to_fname[future]
                print(f"File {fname} generated an exception: {exc}")
                raise

    # Sort by year to maintain consistent ordering
    sorted_pairs = sorted(zip(years, annual_growing_seasons), key=lambda x: x[0])
    sorted_years, sorted_data = zip(*sorted_pairs)
    years = list(sorted_years)
    annual_growing_seasons = list(sorted_data)

    # Create result dataset
    result = xr.Dataset()
    result = result.assign_coords(
        {
            "lat": (["lat"], annual_growing_seasons[0].lat.values),
            "lon": (["lon"], annual_growing_seasons[0].lon.values),
            "year": (["year"], np.array(years, dtype=np.int32)),
        }
    )
    result[metric_name] = (
        ["year", "lat", "lon"],
        np.stack([data.values for data in annual_growing_seasons]),
    )

    # Add metadata
    result[metric_name].attrs["units"] = "days"
    result[metric_name].attrs["long_name"] = f"Growing Season Length (>{base_temp}°C)"
    result[metric_name].attrs["base_temp"] = base_temp
    result[metric_name].attrs["min_consecutive_days"] = min_length

    return result
