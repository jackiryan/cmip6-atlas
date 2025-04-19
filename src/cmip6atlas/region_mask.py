import geopandas as gpd
import xarray as xr
import regionmask
import pandas as pd
import numpy as np
import os
import warnings
import traceback


def calculate_geojson_feature_means(
    geojson_path: str,
    netcdf_path: str,
    variable_name: str,
    output_geojson_path: str,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> bool:
    """
    Calculates the spatial mean of a NetCDF variable for each feature
    in a GeoJSON file, preserving other dimensions (like time, model).

    Writes a new GeoJSON file containing the original features and
    geometry, but with added properties representing the mean values
    for each combination of the non-spatial dimensions.

    Args:
        geojson_path: Path to the input GeoJSON file containing vector features.
        netcdf_path: Path to the input NetCDF file with gridded data.
        variable_name: The name of the variable to average within the NetCDF file
                       (e.g., "ndays_gt_35c").
        output_geojson_path: Path where the output GeoJSON file will be saved.
                              The directory will be created if it doesn't exist.
        lat_name: Name of the latitude coordinate variable in the NetCDF file.
        lon_name: Name of the longitude coordinate variable in the NetCDF file.

    Returns:
        True if the process completed successfully and the file was written,
        False otherwise.
    """
    print("--- Starting spatial averaging ---")
    print(f"Input GeoJSON: {geojson_path}")
    print(f"Input NetCDF: {netcdf_path}")
    print(f"Variable to average: {variable_name}")

    gdf = None
    ds = None
    temp_id_col = "_regionmask_id_"  # Define temporary column name

    try:
        # 1. Read GeoJSON and prepare features
        print("Reading GeoJSON...")
        gdf = gpd.read_file(geojson_path)
        print(f"Read {len(gdf)} features from GeoJSON.")

        # Ensure CRS is geographic (EPSG:4326) for compatibility with lat/lon grids
        if gdf.crs is None:
            warnings.warn(
                "GeoJSON CRS not set. Assuming geographic coordinates (EPSG:4326)."
            )
            gdf.set_crs("EPSG:4326", inplace=True)
        elif not gdf.crs.is_geographic:
            warnings.warn(
                f"GeoJSON CRS ({gdf.crs}) is not geographic. Reprojecting to EPSG:4326."
            )
            try:
                gdf = gdf.to_crs("EPSG:4326")
            except Exception as e:
                print(
                    f"Error: Could not reproject GeoJSON to geographic coordinates: {e}"
                )
                return False

        # Create a temporary column holding the index values (region numbers)
        # This works around an issue where regionmask might mishandle index objects directly.
        if temp_id_col in gdf.columns:
            warnings.warn(
                f"Temporary column name '{temp_id_col}' already exists. Overwriting."
            )
        gdf[temp_id_col] = gdf.index

        # 2. Read NetCDF Data
        print("Reading NetCDF...")
        ds = xr.open_dataset(netcdf_path, chunks={}, decode_coords="all")

        # Validate variable and coordinate names
        if variable_name not in ds.data_vars:
            raise ValueError(
                f"Variable '{variable_name}' not found. Available: {list(ds.data_vars)}"
            )
        if lat_name not in ds.coords:
            raise ValueError(
                f"Latitude coordinate '{lat_name}' not found. Available: {list(ds.coords)}"
            )
        if lon_name not in ds.coords:
            raise ValueError(
                f"Longitude coordinate '{lon_name}' not found. Available: {list(ds.coords)}"
            )

        data_var = ds[variable_name]
        print(f"Found variable '{variable_name}' with dimensions: {data_var.dims}")

        # 3. Create region mask from GeoDataFrame
        print("Creating region mask object...")
        try:
            # Use the temporary column name containing index values for robust region numbering
            regions = regionmask.from_geopandas(
                gdf, numbers=temp_id_col, name="GADM Features"
            )
            print(f"Created mask object with {len(regions)} regions.")
        except Exception as e:
            # Catch errors specifically during regionmask object creation
            raise RuntimeError(f"Error creating regionmask Regions object: {e}") from e

        # 4. Create mask array and calculate spatial mean per region
        print("Preparing data and creating mask array...")
        data_var_for_masking = data_var.copy()

        # Temporarily rename coordinates if needed for regionmask's default inference ('lon', 'lat')
        rename_dict = {}
        if lon_name != "lon" and "lon" not in data_var_for_masking.coords:
            if lon_name in data_var_for_masking.coords:
                rename_dict[lon_name] = "lon"
            else:
                raise ValueError(f"Longitude coordinate '{lon_name}' missing.")
        if lat_name != "lat" and "lat" not in data_var_for_masking.coords:
            if lat_name in data_var_for_masking.coords:
                rename_dict[lat_name] = "lat"
            else:
                raise ValueError(f"Latitude coordinate '{lat_name}' missing.")

        if rename_dict:
            print(
                f"Temporarily renaming coordinates {list(rename_dict.keys())} to {list(rename_dict.values())} for masking."
            )
            data_var_for_masking = data_var_for_masking.rename(rename_dict)

        # Create the mask array (grid cells contain region number or NaN)
        mask = regions.mask(data_var_for_masking)

        print("Calculating spatial means using groupby...")
        # Group original data by the mask and average over spatial dimensions
        feature_means = data_var.groupby(mask).mean(skipna=True)
        print("Averaging complete.")

        # 5. Reshape results and join back to GeoDataFrame
        print("Reshaping results and joining to GeoDataFrame...")
        output_gdf = gdf.copy()  # Start with the original gdf (contains temp_id_col)

        coord_name_for_regions = "mask"  # Coordinate name created by groupby(mask)

        if coord_name_for_regions not in feature_means.coords:
            warnings.warn(
                f"Coordinate '{coord_name_for_regions}' not found in averaged data. No results will be added."
            )
        else:
            # Convert results to DataFrame for joining
            df_means = feature_means.to_dataframe(name=f"avg_{variable_name}")

            # Unstack non-region dimensions (e.g., 'model', 'year') into columns
            other_levels = [
                d for d in df_means.index.names if d != coord_name_for_regions
            ]
            if other_levels:
                df_means = df_means.unstack(level=other_levels)
            else:
                # Handle case with only region dimension if necessary
                print("Note: Only the region dimension was found in averaged data.")

            # Flatten the multi-level column names
            if isinstance(df_means.columns, pd.MultiIndex):
                new_column_names = []
                level_names_sorted = sorted(
                    other_levels
                )  # Use the previously identified levels
                for col_tuple in df_means.columns:
                    prop_name_parts = [col_tuple[0]]  # Base variable name
                    level_values = {
                        name: val
                        for name, val in zip(df_means.columns.names[1:], col_tuple[1:])
                    }
                    for level_name in level_names_sorted:
                        prop_name_parts.append(
                            f"{level_name}_{str(level_values[level_name])}"
                        )
                    new_column_names.append("_".join(prop_name_parts))
                df_means.columns = new_column_names
            else:
                # Handle case where columns might not be MultiIndex (e.g., only one result column)
                df_means.columns = [str(col) for col in df_means.columns]

            # Ensure index types match for joining (target index is gdf integer index)
            if not pd.api.types.is_integer_dtype(df_means.index.dtype):
                print(
                    f"Converting means DataFrame index from {df_means.index.dtype} to integer."
                )
                # NaN indices shouldn't occur if groupby/mean worked, but handle potential float type
                df_means.index = df_means.index.astype(
                    pd.Int64Dtype()
                )  # Use nullable Int64

            # Join the calculated means; use left join to keep all original polygons
            output_gdf = output_gdf.join(df_means, how="left")
            print(f"Joined {len(df_means.columns)} columns of averaged data.")

        # Clean up the temporary ID column before writing
        if temp_id_col in output_gdf.columns:
            output_gdf = output_gdf.drop(columns=[temp_id_col])
            print(f"Removed temporary column '{temp_id_col}'.")

        # 6. Write output GeoJSON
        print(f"Writing output GeoJSON to: {output_geojson_path}")
        output_dir = os.path.dirname(output_geojson_path)
        if output_dir:  # Create directory only if path includes one
            os.makedirs(output_dir, exist_ok=True)

        # Write file, GeoPandas handles NaN -> null conversion
        output_gdf.to_file(output_geojson_path, driver="GeoJSON")
        print("Output file written successfully.")
        return True

    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}")
        return False
    except (ValueError, RuntimeError, KeyError) as e:  # Catch specific expected errors
        print(f"Error during processing: {e}")
        # traceback.print_exc()
        return False
    except Exception as e:  # Catch unexpected errors
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        return False
    finally:
        # Ensure the NetCDF dataset is always closed if it was opened
        if ds is not None:
            ds.close()
            print("NetCDF dataset closed.")
        # Clean up temporary column from original gdf if error occurred before output_gdf creation
        if gdf is not None and temp_id_col in gdf.columns:
            try:
                gdf.drop(columns=[temp_id_col], inplace=True)
            except Exception:
                pass  # Ignore cleanup errors


# --- Example Usage (if run directly or called from another script) ---
if __name__ == "__main__":
    # dummy test data
    input_geojson = "country-bounds/gadm41_CAN_2.json"
    input_netcdf = "climate-metrics/ndays_gt_35c_multi_model_ssp585_2021-2025.nc"
    output_geojson = "climate-metrics/ndays_gt_35c_CAN_2_ssp585_2021-2025.geojson"
    var_name = "ndays_gt_35c"

    # Create dummy output dir
    os.makedirs(os.path.dirname(output_geojson), exist_ok=True)

    print("Running example calculation...")
    # Check if files exist before running
    if not os.path.exists(input_geojson):
        print(f"ERROR: Example input GeoJSON not found at '{input_geojson}'")
    elif not os.path.exists(input_netcdf):
        print(f"ERROR: Example input NetCDF not found at '{input_netcdf}'")
    else:
        success = calculate_geojson_feature_means(
            geojson_path=input_geojson,
            netcdf_path=input_netcdf,
            variable_name=var_name,
            output_geojson_path=output_geojson,
        )

        if success:
            print("\nExample calculation finished successfully.")
        else:
            print("\nExample calculation encountered errors.")
