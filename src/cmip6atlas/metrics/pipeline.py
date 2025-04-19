import os
import xarray as xr
from cmip6atlas.download import (
    create_s3_client,
    download_granules,
    get_available_models,
)
from cmip6atlas.schema import GDDP_CMIP6_SCHEMA
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
    output_format: str = "netcdf",
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

    Raises:
        ValueError: If metric is not defined or time period is invalid (end year
            before start year)
    """
    # Get metric definition
    if metric_name not in CLIMATE_METRICS:
        raise ValueError(f"Unknown metric: {metric_name}")

    metric_def = CLIMATE_METRICS[metric_name]
    temporal_window = metric_def.temporal_window
    if end_year < start_year:
        raise ValueError(f"Time period ({start_year}-{end_year}) is invalid.")

    # For metrics with seasons that span year boundaries, we need data from the previous year
    if (
        temporal_window.is_seasonal()
        and temporal_window.season.spans_year_boundary
        and any(
            m > 9
            for m in temporal_window.season.nh_months + temporal_window.season.sh_months
        )
    ):
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
            skip_prompt=True,
        )

    # Not every variable is available as an output from every model
    missing_data = [
        model_datasets[var] == None or model_datasets[var] == []
        for var in metric_def.variables
    ]
    if any(missing_data):
        print(
            f"One or more required variables not available for the {model} model."
            " Skipping..."
        )
        return ""

    # Calculate the metric using all required variables
    metric_result = metric_def.calculation_func(
        model_datasets,
        # seasonal and threshold parameters default to None
        season=metric_def.temporal_window.season,
        threshold=metric_def.threshold if metric_def.threshold_based else None,
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

    output_path = ""

    # Handle different output formats
    if output_format.lower() in ["netcdf", "both"]:
        netcdf_file = os.path.join(
            output_dir, f"{metric_name}_{model}_{scenario}_{start_year}-{end_year}.nc"
        )
        metric_result.to_netcdf(netcdf_file)
        output_path = netcdf_file

    if output_format.lower() in ["geotiff", "both"]:
        # Get the main variable name from the dataset
        if len(metric_result.data_vars) > 0:
            main_var = list(metric_result.data_vars)[0]
            geotiff_file = os.path.join(
                output_dir,
                f"{metric_name}_{model}_{scenario}_{start_year}-{end_year}.tif",
            )
            xr_to_geotiff(metric_result, geotiff_file, variable_name=main_var)
            output_path = geotiff_file

    # Close the dataset regardless of output format
    metric_result.close()

    return output_path


def calculate_multi_model_metric(
    metric_name: str,
    start_year: int,
    end_year: int,
    scenario: str,
    exclude_models: list[str] | None = None,
    base_dir: str = "./nex-gddp-data",
    output_dir: str = "./climate-metrics",
) -> str:
    """
    Calculate a climate metric for multiple models and combine the results.

    Args:
        metric_name (str): Name of the metric to calculate
        start_year (int): Start year of the period
        end_year (int): End year of the period
        scenario (str): Climate scenario
        exclude_models (list[str] | None): Models to exclude from processing
        base_dir (str): Directory for raw data
        output_dir (str): Directory for output metrics

    Returns:
        str: Path to the combined output NetCDF file, or empty string if no data
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of available models from NEX-GDDP-CMIP6
    s3_client = create_s3_client()

    # Determine the available models for this dataset schema
    bucket = str(GDDP_CMIP6_SCHEMA["bucket"])
    prefix_str = str(GDDP_CMIP6_SCHEMA["prefix"])
    if not prefix_str.endswith("/"):
        prefix_str += "/"
    available_models = get_available_models(s3_client, bucket, prefix_str)

    # Filter out excluded models
    if exclude_models:
        models_to_process = [m for m in available_models if m not in exclude_models]
    else:
        models_to_process = available_models

    print(f"Processing {len(models_to_process)} models: {', '.join(models_to_process)}")

    # Create list to store successful model results
    model_datasets = []
    model_names = []

    # Process each model
    for model in models_to_process:
        print(f"Processing model: {model}")

        # Calculate metric for this model
        output_path = calculate_climate_metric(
            metric_name=metric_name,
            start_year=start_year,
            end_year=end_year,
            scenario=scenario,
            model=model,
            base_dir=base_dir,
            output_dir=output_dir,
            output_format="netcdf",  # Always use netcdf for combining
        )

        # Check if calculation was successful
        if output_path and os.path.exists(output_path):
            # Load the dataset and add it to our list
            try:
                ds = xr.open_dataset(output_path)
                model_datasets.append(ds)
                model_names.append(model)
                print(f"Successfully processed model: {model}")
            except Exception as e:
                print(f"Error loading results for model {model}: {e}")
        else:
            print(f"No results generated for model {model}")

    # If no models were successfully processed, exit
    if not model_datasets:
        print(f"No models could be processed for metric {metric_name}")
        return ""

    # Get the main variable name from the first dataset
    main_vars = list(model_datasets[0].data_vars)
    if not main_vars:
        print("No data variables found in model outputs")
        return ""

    main_var = main_vars[0]
    print(f"Combining model outputs for variable: {main_var}")

    # Combine all datasets into a single dataset with a new 'model' dimension
    # First, extract the data arrays for the main variable and stack them
    model_data_arrays = []

    for i, ds in enumerate(model_datasets):
        # Extract the data array and add model information
        da = ds[main_var]

        # Check if 'year' is already a dimension
        if "year" in da.dims:
            # If year dimension exists, preserve it
            model_data_arrays.append(da)
        else:
            # If no year dimension, we need to add it
            # This handles the case where calculate_climate_metric produced a single year result
            # We create a dummy year dimension with a single value
            da = da.expand_dims(dim={"year": [start_year]})
            model_data_arrays.append(da)

    # Create combined dataset with both model and year dimensions
    combined_data = xr.concat(model_data_arrays, dim="model")

    # Set the model coordinate to use model names
    combined_data = combined_data.assign_coords(model=model_names)

    # Create a new dataset with the combined data
    combined_ds = xr.Dataset({main_var: combined_data})

    # Add metadata
    combined_ds.attrs = model_datasets[0].attrs.copy()
    combined_ds.attrs["models"] = ", ".join(model_names)
    combined_ds.attrs["metric_name"] = metric_name
    combined_ds.attrs["scenario"] = scenario
    combined_ds.attrs["start_year"] = start_year
    combined_ds.attrs["end_year"] = end_year

    # Save the combined dataset
    output_file = os.path.join(
        output_dir, f"{metric_name}_multi_model_{scenario}_{start_year}-{end_year}.nc"
    )
    combined_ds.to_netcdf(output_file)

    # Close all datasets
    for ds in model_datasets:
        ds.close()
    combined_ds.close()

    print(f"Combined multi-model dataset saved to: {output_file}")
    return output_file
