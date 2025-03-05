import cdsapi
import numpy as np
import xarray as xr

'''
# Initialize the CDS API client
c = cdsapi.Client()

# Step 1: Download monthly mean 2m temperature data for 1991-2020
response = c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': '2m_temperature',
        'year': [str(year) for year in range(1991, 2021)],
        'month': [str(month).zfill(2) for month in range(1, 13)],
        'time': '00:00',
        'format': 'netcdf',
    },
    'era5_monthly_2m_temp_1991_2020.nc'
)
'''

# Step 2: Calculate the monthly normals
ds = xr.open_dataset('era5_monthly_2m_temp_1991_2020.nc')

# Extract month information from valid_time
if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
    # Add a month coordinate based on valid_time
    ds = ds.assign_coords(month=ds.valid_time.dt.month)
    
    # Group by the new month coordinate and calculate the mean
    monthly_normals = ds.groupby('month').mean()
else:
    # Fallback to original approach if structure is different
    print("Warning: 'valid_time' not found, attempting alternative grouping")
    if 'time' in ds.dims or 'time' in ds.coords:
        monthly_normals = ds.groupby('time.month').mean()
    else:
        raise ValueError("Neither 'valid_time' nor 'time' found in dataset")

# Identify the temperature variable (could be 't2m' or other names)
temp_vars = [var for var in ds.data_vars if '2m' in var and 'temp' in var.lower()]
if not temp_vars and 't2m' in ds:
    temp_vars = ['t2m']  # Common ERA5 variable name
if not temp_vars:
    # Try to find any temperature-like variable
    temp_vars = [var for var in ds.data_vars if any(term in var.lower() for term in ['temp', 't2m', '2m'])]

if not temp_vars:
    raise ValueError("Could not identify temperature variable in dataset")

temp_var = temp_vars[0]
print(f"Using temperature variable: {temp_var}")

# Convert from Kelvin to Celsius
monthly_normals[temp_var] = monthly_normals[temp_var] - 273.15
monthly_normals[temp_var].attrs['units'] = 'degrees_C'

# Save the normals to a new NetCDF file
monthly_normals.to_netcdf('era5_monthly_2m_temp_normals_1991_2020.nc')

print("Monthly temperature normals calculation complete!")