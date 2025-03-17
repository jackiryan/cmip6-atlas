import xarray as xr
from datetime import datetime
import numpy as np
from cftime import Datetime360Day, DatetimeNoLeap

def time_subset(
    ds: xr.Dataset, 
    start_date: datetime, 
    end_date: datetime
) -> xr.Dataset:
    """
    Standardize time selection across different calendar types.
    
    Args:
        ds: Input dataset
        start_date: Start date for selection
        end_date: End date for selection
        
    Returns:
        Subset of dataset with standardized time selection
    """
    # Check the time format used in the dataset
    time_sample = ds.time.values[0]
    
    if isinstance(time_sample, np.datetime64):
        # Standard numpy datetime
        return ds.sel(time=slice(
            np.datetime64(start_date), 
            np.datetime64(end_date)
        ))
    
    elif isinstance(time_sample, DatetimeNoLeap):
        # No leap year calendar
        # Adjust date if it's Feb 29 (not in this calendar)
        if start_date.month == 2 and start_date.day == 29:
            start_date = datetime(start_date.year, 2, 28)
        if end_date.month == 2 and end_date.day == 29:
            end_date = datetime(end_date.year, 2, 28)
            
        return ds.sel(time=slice(
            DatetimeNoLeap(start_date.year, start_date.month, start_date.day),
            DatetimeNoLeap(end_date.year, end_date.month, end_date.day)
        ))
    
    elif isinstance(time_sample, Datetime360Day):
        # 360-day calendar (all months have 30 days)
        # Adjust any day > 30 to 30
        start_day = min(start_date.day, 30)
        end_day = min(end_date.day, 30)
        
        return ds.sel(time=slice(
            Datetime360Day(start_date.year, start_date.month, start_day),
            Datetime360Day(end_date.year, end_date.month, end_day)
        ))
    
    else:
        # Other calendar types - try generic approach
        try:
            return ds.sel(time=slice(start_date, end_date))
        except Exception as e:
            raise ValueError(f"Unsupported time format: {type(time_sample)}") from e