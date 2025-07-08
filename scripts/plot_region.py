import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_climate_data(filename):
    """Load climate data from JSON file."""
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{filename}'.")
        return None

def merge_historical_and_projection_data(historical_data, projection_data):
    """Merge historical (1991-2020) and projection (2021-2100) datasets."""
    merged_data = []
    
    # Create a lookup dictionary for projection data
    projection_lookup = {region['region_id']: region for region in projection_data}
    
    for hist_region in historical_data:
        region_id = hist_region['region_id']
        
        # Find matching region in projection data
        if region_id in projection_lookup:
            proj_region = projection_lookup[region_id]
            
            # Create merged region by starting with historical data
            merged_region = hist_region.copy()
            
            # Add all projection data fields to the merged region
            for key, value in proj_region.items():
                if key.startswith(('total_annual_precip_', 'mean_annual_temp_')):
                    # Add projection year data
                    merged_region[key] = value
                elif key not in merged_region:
                    # Add any other fields not in historical data
                    merged_region[key] = value
            
            merged_data.append(merged_region)
        else:
            print(f"Warning: Region ID {region_id} found in historical data but not in projection data")
    
    return merged_data

def find_region_by_id(data, region_id):
    """Find a region by its ID."""
    for region in data:
        if region.get('region_id') == region_id:
            return region
    return None

def extract_climate_data(region_data, data_type='precipitation'):
    """Extract years and climate values from region data."""
    years = []
    values = []
    
    if data_type == 'precipitation':
        prefix = 'total_annual_precip_'
    elif data_type == 'temperature':
        prefix = 'mean_annual_temp_'
    else:
        raise ValueError("data_type must be 'precipitation' or 'temperature'")
    
    for key, value in region_data.items():
        if key.startswith(prefix):
            year = int(key.split('_')[-1])
            years.append(year)
            values.append(value)
    
    # Sort by year to ensure proper ordering
    sorted_data = sorted(zip(years, values))
    years, values = zip(*sorted_data)
    
    return list(years), list(values)

def calculate_moving_average(years, values, window=5):
    """Calculate moving average for the climate data."""
    df = pd.DataFrame({'year': years, 'values': values})
    df['moving_avg'] = df['values'].rolling(window=window, center=True).mean()
    
    # Remove NaN values for plotting
    valid_data = df.dropna()
    return valid_data['year'].tolist(), valid_data['moving_avg'].tolist()

def plot_climate_data(region_data, years, values, data_type='precipitation'):
    """Create and display the climate data plot."""
    plt.figure(figsize=(14, 7))
    
    # Set up units and labels based on data type
    if data_type == 'precipitation':
        unit = 'mm'
        ylabel = 'Total Annual Precipitation (mm)'
        title_suffix = 'Precipitation Projections'
        color_annual = 'lightblue'
        color_ma = 'blue'
    elif data_type == 'temperature':
        unit = '°C'
        ylabel = 'Mean Annual Temperature (°C)'
        title_suffix = 'Temperature Projections'
        color_annual = 'lightcoral'
        color_ma = 'red'
    
    # Calculate 5-year moving average
    ma_years, ma_values = calculate_moving_average(years, values, window=5)
    
    # Create the plot with original data
    plt.plot(years, values, linewidth=1, marker='o', markersize=2, alpha=0.5, 
             color=color_annual, label='Annual Data')
    
    # Add 5-year moving average
    plt.plot(ma_years, ma_values, linewidth=3, color=color_ma, 
             label='5-Year Moving Average')
    
    # Add trend line (using original data)
    z = np.polyfit(years, values, 1)
    p = np.poly1d(z)
    plt.plot(years, p(years), "--", color='darkgreen', alpha=0.8, 
             label=f'Overall Trend: {z[0]:.3f} {unit}/year')
    
    # Add vertical line to separate historical from projection data
    plt.axvline(x=2020.5, color='gray', linestyle=':', alpha=0.7, 
                label='Historical | Projection')
    
    # Formatting
    plt.title(f"Historical + Projected {title_suffix} (1991-2100)\n{region_data['source_country_name']} - {region_data['NAME_1']}", 
              fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some statistics as text
    avg_value = np.mean(values)
    min_value = min(values)
    max_value = max(values)
    ma_avg = np.mean(ma_values)
    
    # Calculate separate stats for historical (1991-2020) and projection (2021-2100) periods
    historical_mask = np.array(years) <= 2020
    projection_mask = np.array(years) >= 2021
    
    if np.any(historical_mask):
        hist_avg = np.mean(np.array(values)[historical_mask])
        hist_text = f'Historical Avg (1991-2020): {hist_avg:.1f} {unit}\n'
    else:
        hist_text = ''
    
    if np.any(projection_mask):
        proj_avg = np.mean(np.array(values)[projection_mask])
        proj_text = f'Projection Avg (2021-2100): {proj_avg:.1f} {unit}\n'
    else:
        proj_text = ''
    
    stats_text = f'{hist_text}{proj_text}5-Year MA Average: {ma_avg:.1f} {unit}\nMin: {min_value:.1f} {unit}\nMax: {max_value:.1f} {unit}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the climate data plotter."""
    print("Climate Data Plotter (Historical + Projections)")
    print("=" * 50)
    
    # Choose data type
    while True:
        data_choice = input("What would you like to plot?\n1. Precipitation\n2. Temperature\nEnter your choice (1 or 2): ").strip()
        if data_choice == '1':
            data_type = 'precipitation'
            historical_filename = "annual_precip_global_ssp585_1991-2020.json"
            projection_filename = "annual_precip_global_ssp585_2021-2100.json"
            break
        elif data_choice == '2':
            data_type = 'temperature'
            historical_filename = "annual_temp_global_ssp585_1991-2020.json"
            projection_filename = "annual_temp_global_ssp585_2021-2100.json"
            break
        else:
            print("Please enter 1 for precipitation or 2 for temperature.")
    
    # Load historical data
    print(f"Loading historical {data_type} data (1991-2020)...")
    historical_data = load_climate_data(historical_filename)
    if historical_data is None:
        return
    
    # Load projection data
    print(f"Loading projection {data_type} data (2021-2100)...")
    projection_data = load_climate_data(projection_filename)
    if projection_data is None:
        return
    
    # Merge the datasets
    print("Merging historical and projection data...")
    merged_data = merge_historical_and_projection_data(historical_data, projection_data)
    
    print(f"Successfully merged data for {len(merged_data)} regions.")
    print(f"Data range: 1991-2100 ({len(merged_data)} regions)")
    
    # Get region ID from user
    while True:
        try:
            region_id = int(input("Enter the region ID you want to plot: "))
            break
        except ValueError:
            print("Please enter a valid integer for region ID.")
    
    # Find the region
    region_data = find_region_by_id(merged_data, region_id)
    if region_data is None:
        print(f"Error: Region ID {region_id} not found in the merged data.")
        
        # Show available regions
        print("\nAvailable regions:")
        for region in merged_data[:10]:  # Show first 10 regions as examples
            print(f"  ID: {region['region_id']} - {region['source_country_name']}, {region['NAME_1']}")
        if len(merged_data) > 10:
            print(f"  ... and {len(merged_data) - 10} more regions")
        return
    
    # Extract climate data
    years, values = extract_climate_data(region_data, data_type)
    
    print(f"Plotting {data_type} data for: {region_data['source_country_name']} - {region_data['NAME_1']}")
    print(f"Complete data range: {min(years)} - {max(years)} ({len(years)} years)")
    
    # Create the plot
    plot_climate_data(region_data, years, values, data_type)

if __name__ == "__main__":
    main()