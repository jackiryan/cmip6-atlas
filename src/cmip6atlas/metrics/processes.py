import xarray as xr
import numpy as np
from typing import Any, cast

from cmip6atlas.metrics.definitions import SeasonDefinition
from cmip6atlas.metrics.units import process_variable


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
    datasets: dict[str, list[xr.Dataset]], season: SeasonDefinition, **kwargs
) -> xr.Dataset:
    """
    Calculate seasonal temperature normal, considering hemisphere differences.

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

    # Extract the seasonal data for each year
    seasonal_data: list[xr.DataArray] = []
    years: list[int] = []
    for fname in temp_datasets:
        ds = xr.open_dataset(fname)

        # Get the year from dataset
        year: int = ds.time.dt.year.values[0].item()
        years.append(year)

        # Extract seasonal data -- this extracts all data if the season is annual
        season_ds = extract_seasonal_data(ds, variable, season, year)

        # Convert K to degC
        season_ds_c = process_variable(season_ds, variable)

        # Calculate the temporal mean for each grid cell
        season_mean = season_ds_c[variable].mean(dim=["time"])

        seasonal_data.append(season_mean)
        ds.close()

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

    # Extract the seasonal data for each year
    seasonal_data: list[xr.DataArray] = []
    years: list[int] = []
    for fname in precip_datasets:
        ds = xr.open_dataset(fname)

        # Get the year from dataset
        year: int = ds.time.dt.year.values[0].item()
        years.append(year)

        # Apply a heuristic cap to the raw precipitation data
        # This caps any values exceeding the threshold
        capped_ds = ds.copy()
        capped_ds[variable] = ds[variable].clip(max=max_physical_precip_flux)

        # Extract seasonal data -- this extracts all data if the season is annual
        season_ds = extract_seasonal_data(capped_ds, variable, season, year)

        # Convert from kg/m²/s to mm/day
        season_ds_mm = process_variable(season_ds, variable)

        # Summing over time creates 0 values where there were nodata values, so
        # we need to re-mask them
        season_sums = season_ds_mm[variable].sum(dim="time", skipna=True)
        ocean_mask = season_ds_mm[variable].isnull().all(dim="time")
        season_total = season_sums.where(~ocean_mask, np.nan)

        seasonal_data.append(season_total)
        ds.close()

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

    # Calculate threshold days for each year
    annual_threshold_days: list[xr.DataArray] = []
    years: list[int] = []
    for fname in variable_datasets:
        ds = xr.open_dataset(fname)

        # Get the year from dataset
        year: int = ds.time.dt.year.values[0].item()
        years.append(year)

        # Convert K to degC
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

        # Sum days where threshold is met for each grid cell
        days_count = threshold_met.sum(dim="time", skipna=True)
        ocean_mask = ds_c[variable].isnull().all(dim="time")
        days_total = days_count.where(~ocean_mask, np.nan)
        annual_threshold_days.append(days_total)
        ds.close()

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
    datasets: dict[str, list[xr.Dataset]],
    base_temp: float = 5.0,
    min_length: int = 5,  # Minimum consecutive days to count as growing season
    **kwargs,
) -> xr.Dataset:
    """
    Calculate average growing season length (consecutive days above base temperature).

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

    # Calculate growing season length for each year
    annual_growing_seasons: list[xr.DataArray] = []
    years: list[int] = []
    for fname in temp_datasets:
        ds = xr.open_dataset(fname)

        # Get the year from dataset
        year: int = ds.time.dt.year.values[0].item()
        years.append(year)

        # Convert K to degC
        ds_c = process_variable(ds, variable)

        # Get daily temperatures above base temperature
        above_base = ds_c[variable] > base_temp

        # Use a rolling window to find consecutive days
        # This is a simplified approach - for each grid cell separately
        growing_season_length = xr.zeros_like(ds_c[variable].isel(time=0))

        # Process each lat/lon grid cell
        for lat_idx in range(len(ds_c.lat)):
            for lon_idx in range(len(ds_c.lon)):
                # Get the time series for this grid cell
                cell_series = above_base.isel(lat=lat_idx, lon=lon_idx).values

                # Find runs of consecutive days
                # 1. Identify where values change
                runs = np.diff(np.hstack(([0], cell_series, [0])))
                # 2. Start indices of runs
                run_starts = np.where(runs == 1)[0]
                # 3. End indices of runs
                run_ends = np.where(runs == -1)[0]
                # 4. Lengths of runs
                run_lengths = run_ends - run_starts

                # Filter for runs that meet minimum length
                valid_runs = run_lengths[run_lengths >= min_length]

                # For growing season length, sum the lengths of all valid runs
                if len(valid_runs) > 0:
                    # If hemisphere-specific logic is needed, check latitude
                    # In NH: just sum all growing periods
                    # In SH: handle potential split across year boundary
                    lat_value = float(ds.lat.isel(lat=lat_idx).values)

                    if lat_value >= 0:  # Northern Hemisphere
                        growing_season_length[lat_idx, lon_idx] = sum(valid_runs)
                    else:  # Southern Hemisphere
                        # For SH, we might need special handling if the growing season
                        # spans the year boundary
                        growing_season_length[lat_idx, lon_idx] = sum(valid_runs)

        ocean_mask = ds_c[variable].isnull().all(dim="time")
        season_total = growing_season_length.where(~ocean_mask, np.nan)
        annual_growing_seasons.append(season_total)
        ds.close()

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
