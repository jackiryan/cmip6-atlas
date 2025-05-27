import xarray as xr
import rasterio  # type: ignore [import-untyped]
from rasterio.transform import from_bounds  # type: ignore [import-untyped]
import numpy as np


def netcdf_to_geotiff(
    netcdf_file: str, output_geotiff: str, variable_name: str, crs: str = "EPSG:4326"
) -> None:
    """
    Convert a NetCDF file to GeoTIFF.

    Args:
        netcdf_file: Path to the NetCDF file
        output_geotiff: Path for the output GeoTIFF file
        variable_name: Name of the variable to extract
        crs: Coordinate reference system (default: WGS84)
    """
    # Open the NetCDF file
    ds = xr.open_dataset(netcdf_file)

    xr_to_geotiff(ds, output_geotiff, variable_name, crs)

    # Close the NetCDF dataset
    ds.close()

    print(f"Converted {netcdf_file} to {output_geotiff}")


def xr_to_geotiff(
    dataset: xr.Dataset,
    output_file: str,
    variable_name: str | None = None,
    crs: str = "EPSG:4326",
) -> str:
    """
    Convert an xarray Dataset directly to GeoTIFF.

    Args:
        dataset: The xarray Dataset to convert
        output_file: Path for the output GeoTIFF file
        variable_name: Name of the variable to extract (if None, uses the first data variable)
        crs: Coordinate reference system (default: WGS84)

    Returns:
        str: Path to the saved GeoTIFF file
    """
    # If no variable name is specified, use the first data variable
    if variable_name is None:
        variable_name = list(dataset.data_vars)[0]

    # Get the data array
    data = dataset[variable_name].values.copy()

    # Handle missing values (if _FillValue is set)
    if hasattr(dataset[variable_name], "_FillValue"):
        fill_value = dataset[variable_name]._FillValue
        data = np.where(data == fill_value, np.nan, data)

    # Get spatial dimensions
    lats = dataset.lat.values
    lons = dataset.lon.values.copy()

    # Check if we need to transform longitude range (from 0-360 to -180-180)
    if np.max(lons) > 180:
        # Find the index where longitude crosses 180 degrees
        split_idx = np.searchsorted(lons, 180)

        # Rearrange the data: move the 180-360 portion to the beginning
        data = np.hstack((data[:, split_idx:], data[:, :split_idx]))

        # Convert longitudes from 0-360 to -180-180
        lons_adjusted = lons.copy()
        lons_adjusted[lons > 180] -= 360
        lons = np.sort(lons_adjusted)  # Sort longitudes to ensure proper order

    # Calculate pixel sizes
    if len(lons) > 1 and len(lats) > 1:
        pixel_width = abs(lons[1] - lons[0])
        pixel_height = abs(lats[1] - lats[0])
    else:
        raise ValueError("Dataset must have multiple latitude and longitude values")

    # Check if latitudes are in ascending order (south to north)
    lat_ascending = lats[0] < lats[-1]

    # If latitudes are in ascending order, flip the data
    if lat_ascending:
        data = np.flipud(data)
        # Also flip the latitudes for bound calculations
        lats = np.flip(lats)

    # Set standard -180 to 180 bounds
    left = -180 - pixel_width / 2
    right = 180 + pixel_width / 2
    bottom = lats.min() - pixel_height / 2
    top = lats.max() + pixel_height / 2

    # Calculate the affine transform
    transform = from_bounds(left, bottom, right, top, data.shape[1], data.shape[0])

    # Define metadata
    meta = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
    }

    # Write the GeoTIFF
    with rasterio.open(output_file, "w", **meta) as dst:
        dst.write(data, 1)

        # Add variable metadata as tags
        tags = {}
        for attr_name, attr_value in dataset[variable_name].attrs.items():
            # Convert non-string attributes to strings
            tags[attr_name] = str(attr_value)

        dst.update_tags(**tags)

    return output_file
