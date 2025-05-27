# units.py
import xarray as xr
from typing import Callable


def process_temperature(ds: xr.Dataset, variable: str, **kwargs) -> xr.Dataset:
    """Process temperature variables (tas, tasmax, tasmin)."""
    # Convert K to C if needed
    if (
        ds[variable].attrs.get("units") is None
        or ds[variable].attrs.get("units") == "K"
    ):
        ds[variable] = ds[variable] - 273.15
        ds[variable].attrs["units"] = "°C"
    return ds


def process_precipitation(ds: xr.Dataset, variable: str, **kwargs) -> xr.Dataset:
    """Process precipitation variable (pr)."""
    # CMIP6 precipitation is in kg/m²/s, which is equivalent to mm/s
    ds[variable] = ds[variable] * 86400  # seconds in a day
    ds[variable].attrs["units"] = "mm/day"
    return ds


def process_humidity(ds: xr.Dataset, variable: str, **kwargs) -> xr.Dataset:
    """Process humidity variables (hurs, huss)."""
    # hurs is already in %, huss is in kg/kg
    # No conversion needed generally
    return ds


def process_wind(ds: xr.Dataset, variable: str, **kwargs) -> xr.Dataset:
    """Process wind variables (sfcWind)."""
    # Wind speed is in m/s
    # No conversion needed generally
    return ds


# Registry of variable processors
VARIABLE_PROCESSORS: dict[str, Callable] = {
    # Temperature variables
    "tas": process_temperature,
    "tasmax": process_temperature,
    "tasmin": process_temperature,
    # Precipitation
    "pr": process_precipitation,
    # Humidity
    "hurs": process_humidity,
    "huss": process_humidity,
    # Wind
    "sfcWind": process_wind,
}


def process_variable(ds: xr.Dataset, variable: str, **kwargs) -> xr.Dataset:
    """
    Process a variable-specific dataset with the appropriate processor.

    Args:
        ds: Input dataset
        variable: CMIP6 variable name
        **kwargs: Additional arguments for processor

    Returns:
        Processed dataset
    """
    if variable in VARIABLE_PROCESSORS:
        return VARIABLE_PROCESSORS[variable](ds, variable, **kwargs)
    else:
        # Default - return as is
        return ds
