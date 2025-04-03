import os
from cmip6atlas.download import download_granules
from cmip6atlas.metrics.config import CLIMATE_METRICS
from cmip6atlas.metrics.convert import xr_to_geotiff

def calculate_climate_metric(
    metric_name: str,
    start_year: int,
    end_year: int,
    scenario: str,
    model: str,
    base_dir: str = "./nex-gddp-data",
    output_dir: str = "./climate-metrics",
    output_format: str = "geotiff"
) -> str:
    """
    Calculate a climate metric for a specified time period for a specific model.
    
    Args:
        metric_name (str): Name of the metric to calculate
        start_year (int): Start year of the period
        end_year (int): End year of the period
        scenario (str): Climate scenario
        model (str): Model to compute this climate metric
        base_dir (str): Directory for raw data
        output_dir (str): Directory for output metrics
        output_format (str): Format for output - 'netcdf', 'geotiff', or 'both'
        
    Returns:
        str: Path to the output metric file (NetCDF or GeoTIFF based on output_format)
    """
    # Get metric definition
    if metric_name not in CLIMATE_METRICS:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    metric_def = CLIMATE_METRICS[metric_name]
    temporal_window = metric_def.temporal_window
    if end_year - start_year + 1 != temporal_window.years:
        raise ValueError(
            f"Time period ({start_year}-{end_year}) does not match required "
            f"length for metric {metric_name} ({temporal_window.years} years)"
        )
    
    # For metrics with seasons that span year boundaries, we need data from the previous year
    if (temporal_window.is_seasonal() and 
        temporal_window.season.spans_year_boundary and
        any(m > 9 for m in temporal_window.season.nh_months + temporal_window.season.sh_months)):
        # Adjust start year to get December from previous year (e.g., northern winter)
        effective_start_year = start_year - 1
    else:
        effective_start_year = start_year
    
    model_datasets: dict[str, list[str]] = {var: [] for var in metric_def.variables}
    for variable in metric_def.variables:
        var_dir = os.path.join(base_dir, variable)
        model_datasets[variable] = download_granules(
            variable=variable,
            scenario=scenario,
            start_year=effective_start_year,  # Adjusted start year
            end_year=end_year,
            output_dir=var_dir,
            models=[model],
            skip_prompt=True
        )

    # Calculate the metric using all required variables
    metric_result = metric_def.calculation_func(
        model_datasets, 
        # seasonal and threshold parameters default to None
        season=metric_def.temporal_window.season if metric_def.temporal_window.is_seasonal() else None,
        threshold=metric_def.threshold if metric_def.threshold_based else None
    )
    
    # Add global metadata
    metric_result.attrs["metric_name"] = metric_name
    metric_result.attrs["description"] = metric_def.description
    metric_result.attrs["scenario"] = scenario
    metric_result.attrs["start_year"] = start_year
    metric_result.attrs["end_year"] = end_year
    metric_result.attrs["variables_used"] = ", ".join(metric_def.variables)
    
    # Save the result
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = None
    
    # Handle different output formats
    if output_format.lower() in ["netcdf", "both"]:
        netcdf_file = os.path.join(
            output_dir, 
            f"{metric_name}_{model}_{scenario}_{start_year}-{end_year}.nc"
        )
        metric_result.to_netcdf(netcdf_file)
        output_path = netcdf_file
    
    if output_format.lower() in ["geotiff", "both"]:
        # Get the main variable name from the dataset
        if len(metric_result.data_vars) > 0:
            main_var = list(metric_result.data_vars)[0]
            geotiff_file = os.path.join(
                output_dir, 
                f"{metric_name}_{model}_{scenario}_{start_year}-{end_year}.tif"
            )
            xr_to_geotiff(metric_result, geotiff_file, variable_name=main_var)
            output_path = geotiff_file
            metric_result.close()
    
    return output_path
    
