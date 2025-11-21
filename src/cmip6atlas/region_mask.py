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

    # Normalize weights to sum to 1 (only for models with weights)
    weights_sum = weights_arr.sum()
    if weights_sum > 0:
        weights_arr = weights_arr / weights_sum
    else:
        warnings.warn("All model weights are zero. Using unweighted average.")
        return data_var.mean(dim="model")

    # Filter out models with zero weight
    masked_data = data_var.where(weights_arr > 0)

    # Perform weighted average along model dimension
    weighted_avg = (masked_data * weights_arr).sum(dim="model", skipna=True)

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
        if lat_name != 'y': ds = ds.rename({lat_name: 'y'})
        if lon_name != 'x': ds = ds.rename({lon_name: 'x'})

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
        #if lat_name not in ds.coords or lon_name not in ds.coords:
        #    raise ValueError(f"Required coordinates not found: {lat_name}/{lon_name}")

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
        final_data = {}

        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Clipping Regions"):
            try:
                region_id = idx # or row.name
                geom = row.geometry
                
                # First clip the array to the bounding box of the geometry
                minx, miny, maxx, maxy = geom.bounds
                
                # Buffer slightly to ensure we catch 'all_touched' pixels on the edge
                # Assuming roughly 0.25 degree res, buffer by 0.25
                subset = data_var.rio.clip_box(
                    minx=minx - 0.25,
                    miny=miny - 0.25,
                    maxx=maxx + 0.25,
                    maxy=maxy + 0.25,
                )
                
                # Check if subset is empty (region outside data coverage)
                if subset.size == 0:
                    continue

                # Perform geometry mask on subset of data
                # from_disk=True ensures we don't load the whole array into memory
                masked = subset.rio.clip(
                    [geom],
                    ds.rio.crs,
                    all_touched=use_all_touched,
                    drop=False,
                    from_disk=True
                )

                masked = masked.where(masked != 0)

                """
                valid_pixels = masked.values.flatten()
                valid_pixels = valid_pixels[~np.isnan(valid_pixels)] # Remove NaNs

                if region_id == 1250:
                    print(f"--- DEBUG: {region_id} ---")
                    print(f"Pixel values: {valid_pixels}")
                    print(f"Min: {valid_pixels.min()}, Max: {valid_pixels.max()}")
                    print(f"Count of Zeros: {np.sum(valid_pixels == 0)}")
                """

                # Calculate Mean
                region_mean = masked.mean(dim=['x', 'y'], skipna=True)
                region_mean = region_mean.compute()
                
                # Processing Output
                if region_mean.size == 1:
                    # Single scalar case (e.g., time-averaged or single time slice)
                    val = region_mean.item()
                    final_data[region_id] = {variable_name: val}
                else:
                    # Multi-dimensional case (e.g., time series)
                    df_reg = region_mean.to_dataframe(name=variable_name)
                    
                    # Handle the index to create flat column names like 'annual_temp_2021'
                    reg_dict = {}
                    for idx_val, row in df_reg.iterrows():
                        # If index is a tuple (multi-index), join parts; otherwise just use the value
                        if isinstance(idx_val, tuple):
                            suffix = "_".join(map(str, idx_val))
                        else:
                            suffix = str(idx_val)
                            
                        col_name = f"{variable_name}_{suffix}"
                        reg_dict[col_name] = row[variable_name]
                        
                    final_data[region_id] = reg_dict

            except Exception as e:
                # Often happens if region is too small or outside bounds
                print(f"Skip {idx}: {e}")
                pass

        # 6. Combine results and create output with region metadata
        print(f"Combining results for {len(final_data)} regions with data")
        result_df = pd.DataFrame.from_dict(final_data, orient="index")
        result_df.index.name = "region_id"

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

        # 7. Write output JSON
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
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
            variable_name="mean_"+var_name,
            output_json_path=output_json,
            use_all_touched=True,
            model_weights=weights,
            country=country_code,
        )

        if success:
            print("\nExample calculation finished successfully.")
        else:
            print("\nExample calculation encountered errors.")
