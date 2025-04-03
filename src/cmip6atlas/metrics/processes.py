import xarray as xr
import numpy as np
from typing import Any

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
    nh_months = hemisphere_info["nh_months"]
    sh_months = hemisphere_info["sh_months"]
    
    # Handle year boundaries
    if season_def.spans_year_boundary:
        # For seasons that span year boundary (like DJF), we need to handle them specially
        
        # Northern Hemisphere
        if any(m > 9 for m in nh_months):  # Contains months from previous year
            nh_prev_year_months = [m for m in nh_months if m > 9]
            nh_curr_year_months = [m for m in nh_months if m <= 9]
            
            # Get data from previous year's months
            ds_prev = ds.sel(time=((ds.time.dt.year == year-1) & ds.time.dt.month.isin(nh_prev_year_months)))
            # Get data from current year's months
            ds_curr = ds.sel(time=((ds.time.dt.year == year) & ds.time.dt.month.isin(nh_curr_year_months)))
            
            # Combine the datasets
            ds_nh = xr.concat([ds_prev, ds_curr], dim="time")
        else:
            ds_nh = ds.sel(time=((ds.time.dt.year == year) & ds.time.dt.month.isin(nh_months)))
        
        # Southern Hemisphere
        if any(m > 9 for m in sh_months):  # Contains months from previous year
            sh_prev_year_months = [m for m in sh_months if m > 9]
            sh_curr_year_months = [m for m in sh_months if m <= 9]
            
            # Get data from previous year's months
            ds_prev = ds.sel(time=((ds.time.dt.year == year-1) & ds.time.dt.month.isin(sh_prev_year_months)))
            # Get data from current year's months
            ds_curr = ds.sel(time=((ds.time.dt.year == year) & ds.time.dt.month.isin(sh_curr_year_months)))
            
            # Combine the datasets
            ds_sh = xr.concat([ds_prev, ds_curr], dim="time")
        else:
            ds_sh = ds.sel(time=((ds.time.dt.year == year) & ds.time.dt.month.isin(sh_months)))
    else:
        # For seasons that don't cross year boundary
        ds_nh = ds.sel(time=((ds.time.dt.year == year) & ds.time.dt.month.isin(nh_months)))
        ds_sh = ds.sel(time=((ds.time.dt.year == year) & ds.time.dt.month.isin(sh_months)))
    
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
        coords={
            "time": seasonal_data.time,
            "lat": ds.lat,
            "lon": ds.lon
        }
    )
    
    # Add metadata
    result.attrs["season"] = season_def.name
    result.attrs["year"] = year
    
    return result


def calculate_seasonal_temp_normal(
    datasets: dict[str, list[xr.Dataset]],
    season: SeasonDefinition,
    **kwargs
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
    temp_datasets = datasets[variable]
    
    # Extract the seasonal data for each year
    seasonal_data = []
    for fname in temp_datasets:
        ds = xr.open_dataset(fname)
        # Get the year from dataset
        year = ds.time.dt.year.values[0].item()
        
        # Extract seasonal data
        season_ds = extract_seasonal_data(ds, variable, season, year)
        seasonal_data.append(season_ds)
    
    # Combine all years of seasonal data
    combined = xr.concat(seasonal_data, dim="year")
    # Convert K to degC
    combined_c = process_variable(combined, variable)
    
    # Calculate the temporal mean for each grid cell
    season_mean = combined_c[variable].mean(dim=["year", "time"])
    
    # Create result dataset
    result = xr.Dataset(
        data_vars={f"{season.name}_temp_normal": (["lat", "lon"], season_mean.values)},
        coords={
            "lat": (["lat"], season_mean.lat.values),
            "lon": (["lon"], season_mean.lon.values),
        }
    )
    
    # Add metadata
    result[f"{season.name}_temp_normal"].attrs["units"] = "°C"
    result[f"{season.name}_temp_normal"].attrs["long_name"] = f"{season.name.capitalize()} Temperature Normal"
    result.attrs["season"] = season.name
    
    return result


def calculate_seasonal_precip_total(
    datasets: dict[str, list[str]],
    season: SeasonDefinition,
    **kwargs
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
    # Sometimes the model has a bad pixel that should be excluded
    # The rainiest place on Earth is about 12000mm/yr, so use this
    # heuristic to filter unphysical results
    max_physical_precip = 15000.0
    precip_datasets = datasets[variable]
    
    # Extract the seasonal data for each year
    seasonal_data = []
    for fname in precip_datasets:
        ds = xr.open_dataset(fname)

        # Get the year from dataset
        year: int = ds.time.dt.year.values[0].item()
        
        # Extract seasonal data
        season_ds = extract_seasonal_data(ds, variable, season, year)
        # Convert from kg/m²/s to mm/day
        season_ds_mm = process_variable(season_ds, variable)

        year_sums = season_ds_mm[variable].sum(dim="time", skipna=True)
        ocean_mask = season_ds_mm[variable].isnull().all(dim="time")
        year_da = year_sums.where(~ocean_mask, np.nan)

        seasonal_data.append(year_da)
        ds.close()
    
    # Combine all years of seasonal data
    combined = xr.concat(seasonal_data, dim="year")
    combined = combined.where(combined <= max_physical_precip)
    
    # Calculate the average seasonal total across years
    avg_season_total = combined.mean(dim="year")
    
    # Create result dataset
    result = avg_season_total.to_dataset(name=f"{season.name}_precip_total")

    # Add metadata
    result[f"{season.name}_precip_total"].attrs["units"] = "mm"
    result[f"{season.name}_precip_total"].attrs["long_name"] = f"{season.name.capitalize()} Precipitation Total"
    result.attrs["season"] = season.name
    
    return result


def calculate_threshold_days(
    datasets: dict[str, list[str]],
    threshold: float,
    operation: str = "gt",  # 'gt' for greater than, 'lt' for less than, etc.
    **kwargs
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
    """
    # Get the primary variable (first one in the dataset)
    variable = list(datasets.keys())[0]
    variable_datasets = datasets[variable]
    
    # Calculate threshold days for each year
    annual_threshold_days = []
    
    for fname in variable_datasets:
        ds = xr.open_dataset(fname)

        # Apply threshold operation
        if operation == "gt":
            threshold_met = ds[variable] > threshold
        elif operation == "lt":
            threshold_met = ds[variable] < threshold
        elif operation == "ge":
            threshold_met = ds[variable] >= threshold
        elif operation == "le":
            threshold_met = ds[variable] <= threshold
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        # Sum days where threshold is met for each grid cell
        days_count = threshold_met.sum(dim="time")
        annual_threshold_days.append(days_count)
    
    # Stack all years
    stacked = xr.concat(annual_threshold_days, dim="year")
    
    # Calculate average annual days meeting threshold
    avg_days = stacked.mean(dim="year")
    
    # Create result dataset
    op_map = {"gt": ">", "lt": "<", "ge": "≥", "le": "≤"}
    op_symbol = op_map.get(operation, operation)
    
    var_name = f"{variable}_{operation}_{threshold}_days"
    
    result = xr.Dataset(
        data_vars={var_name: (["lat", "lon"], avg_days.values)},
        coords={
            "lat": (["lat"], avg_days.lat.values),
            "lon": (["lon"], avg_days.lon.values),
        }
    )
    
    # Add metadata
    result[var_name].attrs["units"] = "days/year"
    result[var_name].attrs["long_name"] = f"Days with {variable} {op_symbol} {threshold}"
    result[var_name].attrs["threshold"] = threshold
    result[var_name].attrs["operation"] = operation
    
    return result

def calculate_growing_season_length(
    datasets: dict[str, list[xr.Dataset]],
    base_temp: float = 5.0,
    min_length: int = 5,  # Minimum consecutive days to count as growing season
    **kwargs
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
    temp_datasets = datasets[variable]
    
    # Calculate growing season length for each year
    annual_growing_seasons = []
    
    for fname in temp_datasets:
        ds = xr.open_dataset(fname)

        # Get daily temperatures above base temperature
        above_base = ds[variable] > base_temp
        
        # Use a rolling window to find consecutive days
        # This is a simplified approach - for each grid cell separately
        growing_season_length = xr.zeros_like(ds[variable].isel(time=0))
        
        # Process each lat/lon grid cell
        for lat_idx in range(len(ds.lat)):
            for lon_idx in range(len(ds.lon)):
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
        
        annual_growing_seasons.append(growing_season_length)
    
    # Stack all years and calculate average
    stacked = xr.concat(annual_growing_seasons, dim="year")
    avg_length = stacked.mean(dim="year")
    
    # Create result dataset
    result = xr.Dataset(
        data_vars={"growing_season_length": (["lat", "lon"], avg_length.values)},
        coords={
            "lat": (["lat"], avg_length.lat.values),
            "lon": (["lon"], avg_length.lon.values),
        }
    )
    
    # Add metadata
    result["growing_season_length"].attrs["units"] = "days"
    result["growing_season_length"].attrs["long_name"] = f"Growing Season Length (>{base_temp}°C)"
    result["growing_season_length"].attrs["base_temp"] = base_temp
    result["growing_season_length"].attrs["min_consecutive_days"] = min_length
    
    return result