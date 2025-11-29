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

import concurrent.futures
import geopandas as gpd  # type: ignore [import-untyped]
import json
import os
import numpy as np
import pandas as pd
import traceback
from tqdm import tqdm # type: ignore [import-untyped]
import warnings
import xarray as xr


def apply_model_weights(
    data_var: xr.DataArray, model_weights: dict[str, float]
) -> xr.DataArray:
    """
    Apply weights to average across climate models.

    Args:
        data_var: xarray DataArray with a 'model' dimension
        model_weights: Dictionary mapping model names to weights

    Returns:
        xarray DataArray with model dimension reduced via weighted average
    """
    if not model_weights:
        # If no weights provided, return original data
        return data_var

    # Verify models in weights exist in the data
    available_models = data_var.model.values
    weights_dict = {
        model: weight
        for model, weight in model_weights.items()
        if model in available_models
    }

    if not weights_dict:
        warnings.warn(
            "None of the provided model weights match available models. "
            f"Available models: {list(available_models)}"
        )
        return data_var

    # Check for models in data that aren't in weights and warn
    missing_models = set(available_models) - set(weights_dict.keys())
    if missing_models:
        warnings.warn(
            f"The following models don't have weights and will be excluded: {missing_models}"
        )

    # Create weights array matching the model dimension
    weights_arr = xr.DataArray(
        [weights_dict.get(model, 0) for model in available_models],
        coords={"model": available_models},
        dims=["model"],
    )

    # Handle case where all weights are zero (potential user input)
    if weights_arr.sum() == 0:
        warnings.warn("All model weights are zero. Using unweighted average.")
        return data_var.mean(dim="model")

    # Normalize weights to sum to 1 (only for models with weights)
    masked_data = data_var.where(weights_arr > 0)

    # Calculate weighted sum without converting NaNs to 0
    numerator = (masked_data * weights_arr).sum(dim="model", skipna=True)
    denominator = weights_arr.where(masked_data.notnull()).sum(dim="model")

    # Perform weighted average along model dimension, preserve NaN
    weighted_avg = numerator / denominator

    return weighted_avg


def calculate_geojson_feature_means(
    geojson_path: str,
    netcdf_path: str,
    variable_name: str,
    output_geojson_path: str,
    lat_name: str = "lat",
    lon_name: str = "lon",
    use_all_touched: bool = True,
    model_weights: dict[str, float] = {},
    max_workers: int = 4,
) -> bool:
    """
    Legacy wrapper function for backward compatibility.

    This function maintains the original interface but now uses the global_regions.geojson
    file internally. The geojson_path parameter is ignored.
    """
    warnings.warn(
        "calculate_geojson_feature_means is deprecated. Use calculate_global_regions_means instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Convert geojson output path to json output path
    json_output_path = output_geojson_path.replace(".geojson", ".json")

    return calculate_global_regions_means(
        global_regions_path="global_regions.geojson",
        netcdf_path=netcdf_path,
        variable_name=variable_name,
        output_json_path=json_output_path,
        lat_name=lat_name,
        lon_name=lon_name,
        use_all_touched=use_all_touched,
        model_weights=model_weights,
        #max_workers=max_workers,
        country=None,
    )


def calculate_global_regions_means(
    global_regions_path: str,
    netcdf_path: str,
    variable_name: str,
    output_json_path: str,
    lat_name: str = "lat",
    lon_name: str = "lon",
    use_all_touched: bool = True,
    model_weights: dict[str, float] = {},
    max_workers: int = 8,
    country: str | None = None,
) -> bool:
    """
    Calculates the spatial mean of a NetCDF variable for each region
    in the global_regions.geojson file, preserving other dimensions (like time, model).

    This implementation allows grid cells to contribute to multiple
    overlapping regions, ensuring small regions receive data.

    Args:
        global_regions_path: Path to the global_regions.geojson file containing vector features.
        netcdf_path: Path to the input NetCDF file with gridded data.
        variable_name: The name of the variable to average within the NetCDF file.
        output_json_path: Path where the output JSON file will be saved.
        lat_name: Name of the latitude coordinate variable in the NetCDF file.
        lon_name: Name of the longitude coordinate variable in the NetCDF file.
        use_all_touched: Whether grid cells touching a feature are included in its mask.
        model_weights: Optionally provide weights to pre-compute average value across models.
        max_workers: Maximum number of parallel processes to use.
        country: Optional country code to filter regions (e.g., 'USA', 'AFG'). If None, all regions are processed.

    Returns:
        True if the process completed successfully, False otherwise.
    """
    print(
        f"Processing {variable_name} from {netcdf_path} with regions from {global_regions_path}"
    )

    gdf = None
    ds = None

    try:
        # 1. Read and prepare global regions
        gdf = gpd.read_file(global_regions_path)

        # Filter by country if specified
        if country is not None:
            gdf = gdf[gdf["source_country_code"] == country].copy()
            if len(gdf) == 0:
                print(f"No regions found for country code: {country}")
                return False
            print(f"Processing {len(gdf)} regions for country {country}")
        else:
            print(f"Processing {len(gdf)} regions from global regions file")

        # Ensure geographic CRS (EPSG:4326)
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif not gdf.crs.is_geographic:
            gdf = gdf.to_crs("EPSG:4326")

        # 2. Read NetCDF data
        ds = xr.open_dataset(netcdf_path, chunks="auto", decode_coords="all")

        # Standardize Dimensions
        if lat_name != 'y':
            ds = ds.rename({lat_name: 'y'})
        if lon_name != 'x':
            ds = ds.rename({lon_name: 'x'})

        # 3. Coordinate Alignment
        ds_min_x = ds.x.min().item()
        ds_max_x = ds.x.max().item()

        if ds_min_x >= 0 and ds_max_x > 180:
            vect_min_x = gdf.bounds.minx.min()
            if vect_min_x < 0:
                print("Aligning coordinates: Rolling Raster from 0/360 to -180/180")
                ds.coords['x'] = (ds.coords['x'] + 180) % 360 - 180
                ds = ds.sortby('x')

        ds.rio.write_crs("EPSG:4326", inplace=True)

        # Validate inputs
        if variable_name not in ds.data_vars:
            raise ValueError(f"Variable '{variable_name}' not found in NetCDF file")

        data_var = ds[variable_name]
        print(f"Processing '{variable_name}' with dimensions: {data_var.dims}")

        # 4. Apply model weights if provided
        if "model" in data_var.dims and model_weights:
            print(f"Applying weights to {len(model_weights)} models")
            data_var = apply_model_weights(data_var, model_weights)
            print(f"After applying weights, dimensions: {data_var.dims}")
            # Ensure CRS persists after calculation
            data_var.rio.write_crs("EPSG:4326", inplace=True)

        # 5. Region Loop
        data_var = data_var.load()
        def process_single_region(args):
            region_id, geom = args
            try:
                # A. Optimization: Bounding Box Clip
                minx, miny, maxx, maxy = geom.bounds
                
                # Buffer by 0.25 degrees to ensure we catch edge pixels
                subset = data_var.rio.clip_box(
                    minx=minx - 0.25,
                    miny=miny - 0.25,
                    maxx=maxx + 0.25,
                    maxy=maxy + 0.25,
                )
                
                if subset.size == 0:
                    return region_id, None

                # B. Precise Geometry Mask
                masked = subset.rio.clip(
                    [geom],
                    ds.rio.crs,
                    all_touched=use_all_touched,
                    drop=False
                )

                # C. Calculate Mean
                region_mean = masked.mean(dim=['x', 'y'], skipna=True)
                
                # D. Extract Value
                if region_mean.size == 1:
                    val = region_mean.item()
                    # Check for pure-NaN result (e.g. region was all ocean)
                    if np.isnan(val):
                        return region_id, None
                    return region_id, {variable_name: val}
                else:
                    # Handle time-series/multi-dim
                    df_reg = region_mean.to_dataframe(name=variable_name)
                    reg_dict = {}
                    for idx_val, row in df_reg.iterrows():
                        if isinstance(idx_val, tuple):
                            suffix = "_".join(map(str, idx_val))
                        else:
                            suffix = str(idx_val)
                        # Check for NaNs in time series
                        if not pd.isna(row[variable_name]):
                            reg_dict[f"{variable_name}_{suffix}"] = row[variable_name]
                    
                    if not reg_dict:
                        return region_id, None
                    return region_id, reg_dict

            except Exception:
                return region_id, None
            
        # 5. Parallel Execution
        final_data = {}
        print(f"Processing {len(gdf)} regions with {max_workers} workers...")

        # Create argument list
        work_args = [(idx, row.geometry) for idx, row in gdf.iterrows()]
        
        # Use ThreadPoolExecutor (Works great for in-memory Numpy operations)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(process_single_region, work_args), 
                total=len(work_args),
                unit="region"
            ))

        # Collect results
        for region_id, data in results:
            if data:
                final_data[region_id] = data

        # 6. Combine results and create output with region metadata
        print(f"Combining results for {len(final_data)} regions with data")
        #result_df = pd.DataFrame.from_dict(final_data, orient="index")
        #result_df.index.name = "region_id"

        all_ids = set(gdf.index) # or gdf['region_id']
        processed_ids = set(final_data.keys())
        missing_ids = all_ids - processed_ids

        print(f"Missing {len(missing_ids)} regions.")
        missing_rows = gdf.loc[list(missing_ids)]
        print(missing_rows[['region_identifier', 'source_country_name', 'source_admin_level']])

        # Create output data structure with region metadata but no geometry
        output_data = []
        for region_id, row in gdf.iterrows():
            region_info = {
                "region_id": region_id,
                "region_identifier": row["region_identifier"],
                "source_country_code": row["source_country_code"],
                "source_country_name": row["source_country_name"],
                "source_admin_level": row["source_admin_level"],
                "GID_0": row["GID_0"],
                "COUNTRY": row["COUNTRY"],
                "NAME_1": row["NAME_1"],
            }

            # Add climate data if available for this region
            if region_id in final_data:
                region_info.update(final_data[region_id])

            output_data.append(region_info)

        # 7. Write output JSON with 3 decimal precision
        def round_floats(obj):
            """Recursively round all float values to 3 decimal places."""
            if isinstance(obj, float):
                return round(obj, 3)
            elif isinstance(obj, dict):
                return {k: round_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [round_floats(item) for item in obj]
            return obj

        # Round all float values before writing
        output_data_rounded = round_floats(output_data)

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(output_data_rounded, f, indent=2, default=str)
        print(f"Results saved to {output_json_path}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        if 'ds' in locals() and ds is not None:
            ds.close()


if __name__ == "__main__":
    country_code = None # "BHS"  # Optional: set to None to process all regions
    var_name = "annual_temp"
    input_global_regions = "/Users/jryan/general/cmip6-atlas-backend/data/sources/global_regions.geojson"
    input_netcdf = f"climate-metrics/{var_name}_multi_model_ssp585_2021-2025.nc"
    if country_code:
        output_json = f"climate-metrics/{var_name}_{country_code}_ssp585_2021-2025.json"
    else:
        output_json = f"climate-metrics/{var_name}_global_ssp585_2021-2025.json"

    # Derived from Massoud et al, 2023 (https://doi.org/10.1038/s43247-023-01009-8)
    # This is not necessarily valid for this downscaled dataset, but is okay for a
    # first pass at an ensemble mean
    weights = {
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

    # Create dummy output dir
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    print("Running example calculation...")
    # Check if files exist before running
    if not os.path.exists(input_global_regions):
        print(f"ERROR: Global regions file not found at '{input_global_regions}'")
    elif not os.path.exists(input_netcdf):
        print(f"ERROR: Example input NetCDF not found at '{input_netcdf}'")
    else:
        success = calculate_global_regions_means(
            global_regions_path=input_global_regions,
            netcdf_path=input_netcdf,
            variable_name=var_name,
            output_json_path=output_json,
            use_all_touched=True,
            model_weights=weights,
            country=country_code,
        )

        if success:
            print("\nExample calculation finished successfully.")
        else:
            print("\nExample calculation encountered errors.")
