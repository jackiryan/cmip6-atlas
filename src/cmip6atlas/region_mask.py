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
import os
import numpy as np
import pandas as pd
from rasterio.features import rasterize  # type: ignore [import-untyped]
from rasterio.transform import Affine  # type: ignore [import-untyped]
import shapely.ops
import warnings
import xarray as xr
import concurrent.futures
from typing import Any
from tqdm import tqdm


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
        max_workers=max_workers,
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
    max_workers: int = 6,
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
        ds = xr.open_dataset(netcdf_path, chunks={}, decode_coords="all")

        # Validate inputs
        if variable_name not in ds.data_vars:
            raise ValueError(f"Variable '{variable_name}' not found in NetCDF file")
        if lat_name not in ds.coords or lon_name not in ds.coords:
            raise ValueError(f"Required coordinates not found: {lat_name}/{lon_name}")

        data_var = ds[variable_name]
        print(f"Processing '{variable_name}' with dimensions: {data_var.dims}")

        # Apply model weights if provided
        if "model" in data_var.dims and model_weights:
            print(f"Applying weights to {len(model_weights)} models")
            data_var = apply_model_weights(data_var, model_weights)
            print(data_var)
            print(f"After applying weights, dimensions: {data_var.dims}")

        # 3. Setup for multi-region grid cell contribution
        # Get coordinate info
        lat_vals = ds[lat_name].values
        lon_vals = ds[lon_name].values
        lat_res = abs(lat_vals[1] - lat_vals[0]) if len(lat_vals) > 1 else 0
        lon_res = abs(lon_vals[1] - lon_vals[0]) if len(lon_vals) > 1 else 0
        is_lat_ascending = lat_vals[1] > lat_vals[0] if len(lat_vals) > 1 else True
        out_shape = (len(lat_vals), len(lon_vals))

        # Create a copy for coordinates conversion (0-360°)
        gdf_raster = gdf.copy()

        # Convert geometries from -180/180° to 0/360° longitude
        def convert_to_0_360(geom):
            def shift_lon(x, y, z=None):
                new_x = (x + 360) % 360
                return (new_x, y) if z is None else (new_x, y, z)

            return shapely.ops.transform(shift_lon, geom)

        # Apply conversion
        gdf_raster.geometry = gdf_raster.geometry.apply(convert_to_0_360)

        # Fix any invalid geometries
        invalid_count = sum(1 for geom in gdf_raster.geometry if not geom.is_valid)
        if invalid_count > 0:
            gdf_raster.geometry = gdf_raster.geometry.buffer(0)

        # Calculate transform for pixel centers
        transform = Affine.translation(
            lon_vals[0] - lon_res / 2, lat_vals[0] - lat_res / 2
        ) * Affine.scale(lon_res, lat_res)

        if not is_lat_ascending:
            # Flip y-scaling for descending latitudes
            transform = Affine.translation(
                lon_vals[0] - lon_res / 2, lat_vals[0] + lat_res / 2
            ) * Affine.scale(lon_res, -lat_res)

        # 4. Define function to process a single region (for parallelization)
        def process_region(region_id: int, geometry) -> tuple[int, dict[str, Any]]:
            """Process a single region and return its statistics."""
            try:
                # Create a mask for just this one region
                region_mask_array = rasterize(
                    shapes=[(geometry, 1)],  # One region = one shape
                    out_shape=out_shape,
                    transform=transform,
                    fill=np.nan,
                    all_touched=use_all_touched,
                    dtype=np.float64,
                )

                # Skip regions with no grid cells
                if np.isnan(region_mask_array).all():
                    print(
                        f"Region {region_id} has no corresponding grid cells - skipping"
                    )
                    return region_id, {}

                # Create xarray mask with correct coordinates
                region_mask = xr.DataArray(
                    region_mask_array,
                    coords={lat_name: ds[lat_name], lon_name: ds[lon_name]},
                    dims=[lat_name, lon_name],
                )

                # Calculate means specifically for this region
                # TO DO: consider using a weighted mean to account for pixels whose
                # centers are outside the region. SDF maybe?
                masked_data = data_var.where(region_mask == 1)
                region_mean = masked_data.mean(dim=[lat_name, lon_name], skipna=True)

                # Convert xarray result to pandas DataFrame with proper column names
                df = region_mean.to_dataframe(name=variable_name)

                # Process the DataFrame into a dictionary
                region_data = {}

                # Handle multi-level index case (common with multiple dimensions)
                if isinstance(df.index, pd.MultiIndex):
                    print("multi-index")
                    for idx, row in df.iterrows():
                        # Create clean column names from multi-index
                        column_parts = [variable_name]
                        for dim_name, dim_val in zip(
                            df.index.names, idx if isinstance(idx, tuple) else (idx,)
                        ):
                            column_parts.append(f"{dim_name}_{dim_val}")
                        column_name = "_".join(column_parts)

                        # Use iloc to access by position (fix for the warning)
                        region_data[column_name] = row.iloc[0]
                else:
                    for idx, row in df.iterrows():
                        # Usually just '{variable_name}_{year}'
                        column_name = f"{df.columns[0]}_{idx}"
                        region_data[column_name] = row.iloc[0]

                return region_id, region_data

            except Exception as e:
                print(f"Error processing region {region_id}: {e}")
                return region_id, {}

        # 5. Process regions in parallel
        print(
            f"Processing {len(gdf_raster)} regions with up to {max_workers} parallel workers"
        )
        final_data = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all regions for processing
            futures = {
                executor.submit(process_region, idx, row.geometry): idx
                for idx, row in gdf_raster.iterrows()
            }

            # Collect results as they complete with progress bar
            with tqdm(total=len(futures), desc="Processing regions", unit="region") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    region_id, region_data = future.result()
                    if region_data:  # Only store if we have data
                        final_data[region_id] = region_data
                    pbar.update(1)
                    pbar.set_postfix(completed=len(final_data))

        if not final_data:
            warnings.warn("No regions contained any grid cells")
            return False

        # 6. Convert to DataFrame and create output with region metadata
        print(f"Combining results for {len(final_data)} regions with data")
        result_df = pd.DataFrame.from_dict(final_data, orient="index")
        result_df.index.name = "region_id"

        # Create output data structure with region metadata but no geometry
        output_data = []
        for region_id, row in gdf.iterrows():
            region_info = {
                "region_id": row["region_id"],
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
        import json

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Results saved to {output_json_path}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up
        if ds is not None:
            ds.close()


if __name__ == "__main__":
    country_code = "PRT"  # Optional: set to None to process all regions
    var_name = "ndays_gt_35c"
    input_global_regions = "global_regions.geojson"
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
